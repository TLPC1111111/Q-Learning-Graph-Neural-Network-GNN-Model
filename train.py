import os
import os.path as osp
import torch
import torch.optim as optim
import time
import tqdm
import random
import numpy as np
from models import Q_learning
from torch_geometric.datasets import Planetoid
from torch.utils.tensorboard import SummaryWriter

writer_loss = SummaryWriter(log_dir='./Q-Learning/Loss_Curve')

loss = 0

# 创建检查点文件夹
os.makedirs('checkpoints', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
dataset = Planetoid(path, name='Cora')

num_nodes = dataset.data.train_mask.size(0)
edge_index = dataset.data.edge_index
w = torch.ones(num_nodes, dtype=torch.float32)
batch = torch.zeros(num_nodes, dtype=torch.long)

# 参数设置
select_rate = 0.003  #随机选择概率
gamma = 0.8  #折扣因子
batch_size = 64  # 设置经验回放批次大小 现最佳是64
experience_buffer = []  # 经验回放缓冲区
experience_buffer_size = 3000  # 经验缓冲区大小

model = Q_learning().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)  # 使用Adam优化器
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer , patience=10 , factor=0.1 , verbose=True)

mis_best = 0
l_mis = 0
iter_times = 1800   #待修改，目前800---1051     900--1385    1200 --1348   1800--1407   1500--1207



for epoch in tqdm.tqdm(range(50), desc="Processing"):
    s = time.time()
    x_v = torch.zeros(num_nodes, dtype=torch.float32)
    x_v_origin = torch.zeros(num_nodes, dtype=torch.float32)
    if epoch == 0:
        Q, Q_mask, adj = model(w.to(device), edge_index.to(device), batch.to(device), x_v.to(device))

    selected = [set() for _ in range(adj.size(0))]
    rejected = [set() for _ in range(adj.size(0))]
    selected_list = [[] for _ in range(adj.size(0))]

    sort_Q = torch.sort(Q, descending=True, dim=1)
    reword_flag = 0
    i = 0
    flag = 0
    for b in range(adj.size(0)):
        optimizer.zero_grad()
        while flag < len(sort_Q.values[b]):

            print(f"当前计数器为{flag}\n")

            node_index = sort_Q.indices[b][i].item()   #注意这里的i，当sort一次后i=0，选择最大的Q值
            neighbors = torch.where(adj[b][node_index] == 1)[0]

            if random.random() < select_rate:
                random_node = random.choice(range(num_nodes))
                if random_node not in rejected[b] and random_node not in selected[b]:
                    selected[b].add(random_node)
                    selected_list[b].append(random_node)
                    reword_flag += 1
                    x_v[random_node] = 1
                    i += 1
                    flag += 1
                    random_node_neighbors = torch.where(adj[b][random_node] == 1)[0]
                    for n2 in random_node_neighbors.tolist():
                        rejected[b].add(n2)

            elif len(neighbors) == 0:
                selected[b].add(node_index)
                selected_list[b].append(node_index)
                reword_flag += 1
                x_v[node_index] = 1
                i += 1
                flag += 1

            elif node_index not in rejected[b] and node_index not in selected[b]:
                selected[b].add(node_index)
                selected_list[b].append(node_index)
                reword_flag += 1
                x_v[node_index] = 1
                i += 1
                flag += 1
                for n in neighbors.tolist():
                    rejected[b].add(n)

            else:
                selected_list[b].append(-1)
                i += 1
                flag += 1


            if flag >= iter_times and flag < num_nodes:      #当遍历了iter_times次节点后(这些节点根据Q值从大到小依次遍历)
                i = flag
                loss_list = []
                # 将经验存入缓冲区  (缓冲区中的transition是一个六元组，形如(S_t , a_t , r_t , S_{t+n} , selected_list , rejected)
                if i - iter_times > 0 and selected_list[b][i - iter_times] != -1:  #如果不是第一次且S_t做了动作就记录S_t状态向量
                    x_v_origin[selected_list[b][i - iter_times]] = 1  #记录S_t的状态向量，当然初始S_0当然是zeros向量
                if selected_list[b][i - iter_times] != -1: #如果在状态S_t处有动作，则放入经验池。
                    experience_buffer.append((x_v_origin.clone(), selected_list[b][i - iter_times], reword_flag, x_v.clone() , selected[b] , rejected[b]))
                    reword_flag -= 1

                # 如果经验缓冲区溢出，则删掉最初加入的
                if len(experience_buffer) > experience_buffer_size:
                    experience_buffer.pop(0)

                # 如果经验缓冲区中的六元组少于batch_size，则继续往池中塞入六元组
                if len(experience_buffer) < batch_size:
                    print("experience_buffer not satisfact stock!")

                # 从经验缓冲区中采样
                if len(experience_buffer) >= batch_size:
                    #从经验回放池中随机挑选batch_size个transition。
                    batch_sample = random.sample(experience_buffer, batch_size)
                    batch_states, batch_actions, batch_rewards, batch_next_states , batch_selected , batch_rejected = zip(*batch_sample)

                    batch_states = torch.stack(batch_states).to(device)
                    batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(device)
                    batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
                    batch_next_states = torch.stack(batch_next_states).to(device)

                    for buffer in range(batch_size):
                        Q_cur_set, _ , _ = model(w.to(device) , edge_index.to(device), batch.to(device) , batch_states[buffer].to(device))
                        Q_cur = Q_cur_set[b][batch_actions[buffer].item()]  #计算Q(h(S_t) , v_t ; \theta)

                        #记下来计算 max_{v'} Q(h(S_{t+1} , v' ; \theta)
                        Q_next_set , _ , _ = model(w.to(device), edge_index.to(device) , batch.to(device), batch_next_states[buffer].to(device))
                        sort_Q_next_set = torch.sort(Q_next_set , dim = 1 , descending = True)

                        node_index_2 = -100
                        for index in range(len(sort_Q_next_set.values[b])):
                            node_index_2 = sort_Q_next_set.indices[b][index] #node_index_2实际上就是v_{t+n}
                            if node_index_2 not in batch_rejected[buffer] and node_index_2 not in batch_selected[buffer]:
                                break
                        if node_index_2 != -100:
                            y = gamma * Q_next_set[b][node_index_2] + batch_rewards[buffer]
                        else: #如果S_{t+n}找不到合理的最大Q值节点，则直接下一轮。
                            Q, Q_mask, adj = model(w.to(device), edge_index.to(device), batch.to(device), x_v.to(device))
                            sort_Q = torch.sort(Q, descending=True, dim=1)
                            i = 0
                            continue

                        loss = (y - Q_cur)**2
                        loss_list.append(loss)
                        # 更新模型
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    Q, Q_mask, adj = model(w.to(device), edge_index.to(device), batch.to(device), x_v.to(device))
                    sort_Q = torch.sort(Q, descending=True, dim=1)
                    i = 0
                    loss_mean = torch.mean(torch.tensor(loss_list))
                    print(loss_mean.item())
                    writer_loss.add_scalar('LOSS/Train' , loss_mean.item() , flag)



        mwis = np.array(list(selected[b]))
        masked_mwis = mwis[mwis < len(Q_mask[b][Q_mask[b] == True])]
        selected[b] = list(masked_mwis)
        l_mis = len(selected[0])

        if l_mis > mis_best:
            mis_best = l_mis
            torch.save(model.state_dict(), 'checkpoints/zzy_very_handsome.pkl')

        print('[*] Epoch: {}, Loss: {:.3f}, |MIS| = {}, Best: {} Time: {:.1f}'
                .format(epoch, loss.item(), l_mis, mis_best, time.time() - s))
