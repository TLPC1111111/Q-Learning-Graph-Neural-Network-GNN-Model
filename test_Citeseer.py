import os.path as osp
import torch
import pprint
from models import Q_learning
from torch_geometric.datasets import Planetoid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)) , 'data')
dataset = Planetoid(path , name = 'Citeseer')

data = dataset[0]
num_nodes = data.num_nodes
edge_index = data.edge_index
w = torch.ones(num_nodes, dtype=torch.float32)
batch = torch.zeros(num_nodes, dtype=torch.long)

model = Q_learning().to(device)
model.load_state_dict(torch.load('best_checkpoints/zzy_very_handsome.pkl'))
model.eval()
with torch.no_grad():
    x_v = torch.zeros(data.num_nodes, dtype=torch.float32).to(device)
    Q, Q_mask, adj = model(w.to(device), edge_index.to(device), batch.to(device), x_v.to(device))
    selected = [set() for _ in range(adj.size(0))]
    rejected = [set() for _ in range(adj.size(0))]
    sort_Q = torch.sort(Q, descending=True, dim=1)
    print(sort_Q.indices[0])
    for b in range(adj.size(0)):
        for i in range(len(sort_Q.values[b])):  #遍历节点
            node_index = sort_Q.indices[b][i].item()
            neighbors = torch.where(adj[b][node_index] == 1)[0]
            if len(neighbors) == 0:
                selected[b].add(node_index)
                x_v[node_index] = 1
            if node_index not in rejected[b] and node_index not in selected[b]:
                selected[b].add(node_index)
                x_v[node_index] = 1
                for n in neighbors.tolist():
                    rejected[b].add(n)
        print(f"利用Q-learning找出数据集Citeseer上最大独立集的元素个数为：{len(selected[b])}")
        pprint.pprint(f"利用Q-learning找出数据集Citeseer上最大独立集的元素为：{selected[b]}")

