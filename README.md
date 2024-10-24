# Q-Learning Graph Neural Network (GNN) Model

## Overview
This repository contains a Graph Neural Network (GNN) model built using PyTorch and PyTorch Geometric. The model is designed for Q-learning tasks, particularly on graph-structured data. It leverages Graph Isomorphism Networks (GINConv) layers for graph convolution, along with Multi-Layer Perceptrons (MLPs) for feature learning and transformation.

## Model Architecture

### 1. MLP (Multi-Layer Perceptron)
The `MLP` class is a fully connected neural network with optional batch normalization and dropout. It takes a list of layer sizes and applies linear transformations, ReLU activations, and optional batch normalization and dropout.

### 2. Q_Learning Model
The `Q_Learning` model is the main architecture implementing a GNN using GINConv layers. It operates on graph-structured data and is designed for Q-learning scenarios. 

- **Graph Convolution Layers**: Multiple GINConv layers are used to process node and edge information, learning node embeddings through graph aggregation.
- **Batch Normalization**: Applied after each graph convolution layer to stabilize training.
- **Global Pooling**: The learned node embeddings are pooled using mean pooling (`global_mean_pool`) to create graph-level representations.
- **Fully Connected Layers**: After pooling, fully connected layers process the combined node and graph-level information to predict Q-values for Q-learning.

### Q-Learning Formula
The model is based on the following Q-learning update formula:

\[
\mu_{v}^{(t+1)} \leftarrow \text{ReLU}(\sigma_1 \cdot x_v + \sigma_2 \cdot \sum_{u \in N(v)} \mu_{u}^{t})
\]

Where \(N(v)\) represents the neighboring nodes of node \(v\).

## Training Data
The model is designed for tasks where each graph represents an environment, and the nodes represent different states or agents. It is particularly suited for reinforcement learning tasks on graphs, where Q-values are predicted to guide decision-making.

## Dependencies
To run the model, you will need the following dependencies:

- **Python**: 3.6+
- **PyTorch**: 1.8+
- **PyTorch Geometric**: Latest version
- **torch-scatter**, **torch-sparse**, **torch-cluster**, **torch-spline-conv**: Required by PyTorch Geometric.


## Usage

To use the model, you can follow this example:

```python
import torch
from model import Q_learning

# Example inputs
w = torch.rand(10)  # node features
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])  # edge index in COO format
batch = torch.tensor([0, 0, 1, 1])  # batch assignments for graphs
x_v = torch.rand(4)  # specific node features

# Initialize and forward pass
model = Q_learning()
Q_final_dense, Q_final_mask, adj = model(w, edge_index, batch, x_v)

print(Q_final_dense, Q_final_mask, adj)
```


# Q-Learning Graph Neural Network (GNN) for Maximum Independent Set (MIS)

## Overview

This repository contains a PyTorch implementation of a Q-Learning Graph Neural Network (GNN) designed to solve the Maximum Independent Set (MIS) problem on graph-structured data. The model uses Graph Isomorphism Networks (GINConv) for learning node embeddings and applies Q-learning to iteratively select nodes, maximizing the independent set.

## Model Architecture

### Main Components

1. **Q-Learning GNN**: The core of the model, utilizing multiple GINConv layers and fully connected layers to predict Q-values for each node in the graph.
2. **Reinforcement Learning**: The model operates in a Q-learning setting, where nodes are selected based on their Q-values. The selected nodes are added to the independent set while their neighbors are rejected, ensuring the independence property.
3. **Experience Replay**: A replay buffer stores previous transitions, allowing the model to sample from past experiences for better stability during training.
4. **TensorBoard Logging**: Loss curves during training are logged to TensorBoard.

## Dependencies

- **Python**: 3.6+
- **PyTorch**: 1.8+
- **PyTorch Geometric**: Latest version
- **tensorboard**: For logging training metrics

You can install the necessary dependencies using:

```bash
pip install torch torch-geometric tensorboard
```

## Dataset
The model uses the Cora dataset, a popular citation network dataset, for training and testing the model. This dataset is included as part of PyTorch Geometric's dataset library.


##  Training Process
The Q-learning process involves the following steps:

1.**Node Selection**: In each epoch, nodes are selected based on their Q-values, starting from the highest value. Nodes are added to the independent set if they are not adjacent to previously selected nodes.

2.**Experience Buffer**: The state, action, reward, and next state are stored in an experience buffer. A batch of transitions is sampled from this buffer during training to update the Q-values.

3.**Optimization**: The loss is calculated as the squared difference between the predicted Q-value for the current state-action pair and the target value (reward plus discounted future reward). The model is optimized using Adam.

4.**Checkpointing**: The model is periodically saved if it finds a better Maximum Independent Set (MIS) during training.








