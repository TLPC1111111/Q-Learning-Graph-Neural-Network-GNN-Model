import torch
import time
import pprint
import tqdm
import sys
import os.path as osp
from models import Q_learning
from torch_geometric.datasets import Planetoid
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##测试一种新的搜寻Q-值的线上训练，效果差，待改进##

def split_nodes(nodes_list , split_num , device):
    random_nodes = torch.randperm(len(nodes_list) , device = device)
    chunks = torch.chunk(random_nodes , split_num)
    return chunks

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
dataset = Planetoid(path, name='Cora')

data = dataset[0]
num_nodes = data.num_nodes
edge_index = data.edge_index
w = torch.ones(num_nodes, dtype=torch.float32)
batch = torch.zeros(num_nodes, dtype=torch.long)

model = Q_learning().to(device)
model.load_state_dict(torch.load('best_checkpoints/zzy_very_handsome.pkl'))
model.eval()

print(num_nodes)
split_num = 20

with torch.no_grad():
    x_v = torch.zeros(data.num_nodes, dtype=torch.float32).to(device)
    Q, Q_mask, adj = model(w.to(device), edge_index.to(device), batch.to(device), x_v.to(device))
    selected = [set() for _ in range(adj.size(0))]
    rejected = [set() for _ in range(adj.size(0))]
    sort_Q_list = []
    split_list = []
    sort_Q = torch.sort(Q, descending=True, dim=1)
    print(sort_Q.indices[0])
    nodes_chunks = split_nodes(range(num_nodes), split_num=split_num , device = device)
    for k in tqdm.tqdm(range(split_num) , desc = "Processing"):
        for i in range(num_nodes):
            if sort_Q.indices[0][i] in nodes_chunks[k]:
                sort_Q_list.append(sort_Q.indices[0][i].item())
        split_list.append(sort_Q_list)
        sort_Q_list = []
    print(split_list)
    for i in range(len(split_list)):
        print(len(split_list[i]))
    min_length = sys.maxsize
    for i in range(split_num):
        if len(split_list[i]) < min_length:
            min_length = len(split_list[i])
    for i in tqdm.tqdm(range(min_length) , desc = "Processing_NN"):
        for j in range(split_num):
            neighbors = torch.where(adj[0][split_list[j][i]] == 1)[0]
            if len(neighbors) == 0:
                selected[0].add(split_list[j][i])
                x_v[split_list[j][i]] = 1
            elif split_list[j][i] not in selected[0] and split_list[j][i] not in rejected[0]:
                selected[0].add(split_list[j][i])
                x_v[split_list[j][i]] = 1
                for n in neighbors.tolist():
                    rejected[0].add(n)

print(len(selected[0]))






