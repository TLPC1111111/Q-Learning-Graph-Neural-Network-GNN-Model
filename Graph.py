from Read_AFIT import *
import torch

read_afit = AFIT("./reqlf13")

num_nodes = read_afit.total_length()
print(f"节点个数(可见弧段总数)为{num_nodes}")

support_id_list = read_afit.ead_support_id_list()
antenna_id_list = read_afit.read_antenna_id_list()
visbable_window = read_afit.read_visbable_window()
task_duration = read_afit.read_task_duration()
antenna_turnaround = read_afit.read_antenna_turnaround()

edge_index = []


for i in range(num_nodes):
    for j in range(num_nodes):
        if i ==j :
            continue
        if support_id_list[i] == support_id_list[j]:
            edge_index.append([i,j])
        elif antenna_id_list[i] == antenna_id_list[j] and \
            visbable_window[i][1] + antenna_turnaround[i] > visbable_window[j][0] and \
            visbable_window[j][1] + antenna_turnaround[j] > visbable_window[i][0] :
            edge_index.append([i,j])
edge_index = torch.tensor(edge_index)
edge_index_t = edge_index.t()
print(edge_index)
with open('SRS_edge.txt' , 'w') as wf:
     for i in range(len(edge_index)):
        wf.write(f"{edge_index[i][0]} {edge_index[i][1]}\n")
