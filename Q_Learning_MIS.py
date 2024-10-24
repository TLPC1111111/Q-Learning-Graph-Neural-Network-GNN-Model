import torch
import pprint
from models import *
from torch_geometric.data import Data
from models import Q_learning
from Read_AFIT import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
afit = AFIT("reqlf13")

num_nodes = afit.total_length()
print(f"--------afscn转化为图问题的总节点个数为{num_nodes}")

w = torch.ones(num_nodes, dtype=torch.float32)

edge_index = [[],[]]
with open("SRS_edge.txt" , 'r') as rf:
    for line in rf:
        start, end = map(int, line.strip().split())
        edge_index[0].append(start)
        edge_index[1].append(end)
edge_index = torch.tensor(edge_index , dtype = int)
print(f"--------邻接矩阵(2*ndim)为{edge_index}")

batch = torch.zeros(num_nodes, dtype=torch.long)

data = Data(x = w , edge_index = edge_index)

zzy = Q_learning().to(device)
zzy.load_state_dict(torch.load("./checkpoints/zzy_very_handsome.pkl"))

zzy.eval()
with torch.no_grad():
    x_v = torch.zeros(data.num_nodes, dtype=torch.float32).to(device)
    Q, Q_mask, adj = zzy(w.to(device), edge_index.to(device), batch.to(device), x_v.to(device))
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
        #print(len(selected[b]))
        print(f"-------Q_Learning在afscn数据集上所求出的最大数据集节点个数为{len(selected[b])}")
        pprint.pprint(f"-------Q_Learning所求得的afscn数据集上的可行解为{selected[b]}")

support_id_list = afit.read_support_id_list()
sup_list = []
for id in selected[0]:
    sup_list.append(support_id_list[id])
print(sup_list)
missing_support = list(set(range(1,138)) - set(sup_list))
print(missing_support)