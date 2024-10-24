import torch
import torch.nn as nn
from torch_geometric.nn import GINConv , global_add_pool , global_mean_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj

class MLP(nn.Module):
    def __init__(self , size_list , batch_norm = False , dropout = 0 , activation = nn.ReLU()):
        super().__init__()
        self.mlp = nn.ModuleList()
        for i in range(len(size_list) - 1):
            self.mlp.append(nn.Linear(in_features = int(size_list[i]) , out_features = int(size_list[i+1])))
            if i != len(size_list) - 2:
                self.mlp.append(activation)

            if batch_norm is True:
                self.mlp.append(nn.BatchNorm1d(num_features = size_list[i+1]))

        self.mlp.append(nn.Dropout(p = dropout))

    def forward(self , x):
        for layer in self.mlp:
            if 'Batch' in layer.__class__.__name__:
                if len(x.size()) == 2:
                    x = layer(x)
                else:
                    x = layer(x.view(-1 , x.size(-1))).view(x.size())
            else:
                x = layer(x)
        return x

class Q_learning(nn.Module):
    '''
    \mu_{v}^{(t+1)} <---- Relu(\sigma_1 * x_v + \sigma_2 * \sum_{u \in N(v)} \mu_{u}^{t})    这的 N(v)表示节点v的邻接节点。
    '''
    def __init__(self):
        super().__init__()
        self.Q_full_connect1 = nn.Linear(in_features = 2 * 64 , out_features = 1)
        self.Q_full_connect2 = nn.Linear(in_features = 64 , out_features = 64)
        self.Q_full_connect3 = nn.Linear(in_features = 64 , out_features = 64)

        self.full_connect1 = nn.Linear(in_features=1, out_features=64)
        self.full_connect2 = nn.Linear(in_features=64 , out_features=64)
        self.full_connect3 = nn.Linear(in_features=64 , out_features=64)
        self.full_connect4 = nn.Linear(in_features=64 , out_features=64)

        self.conv1 = GINConv(nn = MLP([1 , 64]) , train_eps = False , eps = -1)
        self.conv2 = GINConv(nn = MLP([64 , 64]) , train_eps = False , eps = -1)
        self.conv3 = GINConv(nn = MLP([64 , 64]) , train_eps = False , eps = -1)
        self.conv4 = GINConv(nn = MLP([64 , 64]) , train_eps = False , eps = -1)

        self.b1 = nn.BatchNorm1d(num_features = 64)
        self.b2 = nn.BatchNorm1d(num_features = 64)
        self.b3 = nn.BatchNorm1d(num_features = 64)

    def forward(self , w , edge_index , batch , x_v):
        x_v2 = self.full_connect1(x_v.unsqueeze(1))
        prob = torch.relu(self.b1(self.conv1(w.unsqueeze(1) , edge_index) + x_v2))
        x_v3 = self.full_connect2(x_v2)
        prob = torch.relu(self.b2(self.conv2(prob , edge_index) + x_v3))
        x_v4 = self.full_connect3(x_v3)
        prob = torch.relu(self.b3(self.conv3(prob , edge_index) + x_v4))
        x_v5 = self.full_connect4(x_v4)
        prob = torch.relu(self.conv4(prob , edge_index) + x_v5)
        #得到最终的prob维数为(nodes_num , 64)

        graph_features = global_mean_pool(prob , batch = batch)
        #graph_features = global_sum_pool(prob , batch = batch)
        #得到的graph_features维数为(graphs_num , 64)
        weighted_graph_features = self.Q_full_connect2(graph_features)

        weighted_prob = self.Q_full_connect3(prob)

        graph_features_expanded = weighted_graph_features.T.expand(64 , len(prob)).T

        prob_and_graph_features_cat = torch.cat([graph_features_expanded , weighted_prob] , dim = 1)

        Q_final = self.Q_full_connect1(torch.relu(prob_and_graph_features_cat))
        #最终得到的Q_final的维数为(nodes_num , 1)

        Q_final_dense , Q_final_mask = to_dense_batch(Q_final , batch)

        adj = to_dense_adj(edge_index, batch)


        return Q_final_dense , Q_final_mask , adj


