import os.path as osp
import torch
from models import Q_learning
from torch_geometric.datasets import Planetoid


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
dataset = Planetoid(path, name='Cora')

# 选择一张图（Cora数据集中只有一张图，所以选择第0张）
data = dataset[0]
num_nodes = data.num_nodes
edge_index = data.edge_index
w = torch.ones(num_nodes, dtype=torch.float32)
batch = torch.zeros(num_nodes, dtype=torch.long)

# 初始化模型并加载预训练的权重
model = Q_learning().to(device)
model.load_state_dict(torch.load('best_checkpoints/zzy_very_handsome.pkl'))

model.eval()
with torch.no_grad():
    x_v = torch.zeros(data.num_nodes, dtype=torch.float32).to(device)
    Q, Q_mask, adj = model(w.to(device), edge_index.to(device), batch.to(device), x_v.to(device))
    selected = [set() for _ in range(adj.size(0))]
    rejected = [set() for _ in range(adj.size(0))]
    sort_Q = torch.sort(Q, descending=True, dim=1)
    for b in range(adj.size(0)):
        for i in range(len(sort_Q.values[b])):
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
        print(len(selected[b]))