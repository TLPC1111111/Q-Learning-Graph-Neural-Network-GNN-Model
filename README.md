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

### 1. Clone the Repository
```bash
git clone https://github.com/TLPC1111111/Q-Learning-Graph-Neural-Network-GNN-Model.git
cd Q-Learning-Graph-Neural-Network-GNN-Model
```

### 2. Install Dependencies
```bash
pip install torch torch-geometric tensorboard
```

### 3. Training the Model
To train the model, simply run:

```bash
python train.py
```

This will train the model on the Cora dataset, log the loss values to TensorBoard, and save the best model checkpoint.

### Example Training Script

Here’s an example of the training loop that utilizes Q-learning, experience replay, and GNN:

```python
import torch
import torch.optim as optim
from models import Q_learning
from torch_geometric.datasets import Planetoid
from torch.utils.tensorboard import SummaryWriter
import os
import random

# Initialize TensorBoard writer
writer_loss = SummaryWriter(log_dir='./Q-Learning/Loss_Curve')

# Load the dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
dataset = Planetoid(path, name='Cora')

# Initialize model and optimizer
model = Q_learning().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Main training loop
for epoch in range(50):
    # Your training logic
    ...
    writer_loss.add_scalar('LOSS/Train', loss_mean.item(), flag)
```

### 4. Monitoring Training with TensorBoard
You can visualize the training process using TensorBoard by running the following command:

```bash
tensorboard --logdir=./Q-Learning/Loss_Curve
```

### 5. Saving and Loading Models
The model checkpoints are saved in the `checkpoints` directory. The best-performing model (based on the size of the MIS) is saved as `zzy_very_handsome.pkl`. You can load this model for inference using:

```python
model = Q_learning()
model.load_state_dict(torch.load('checkpoints/zzy_very_handsome.pkl'))
```

## Parameters
- `select_rate`: Random node selection probability (default: 0.003).
- `gamma`: Discount factor for future rewards in Q-learning (default: 0.8).
- `batch_size`: Number of samples used for each experience replay batch (default: 64).
- `experience_buffer_size`: The size of the experience buffer (default: 3000).
- `iter_times`: Number of iterations for node selection during Q-learning.

## Results
The model outputs the Maximum Independent Set (MIS) after training. The best size of the independent set is recorded, and the model is saved when a better MIS is found during training.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## Acknowledgments
This project uses the following libraries:
- PyTorch
- PyTorch Geometric
- TensorBoard



# Q-Learning with Graph Neural Networks (GNN)

This repository contains a Q-learning algorithm implemented using a Graph Neural Network (GNN) for solving the Maximum Independent Set (MIS) problem. The model utilizes Q-learning with experience replay and optimizes through a custom GNN architecture.

## Requirements

Install the required Python libraries with the following command:

```bash
pip install torch torch-geometric tensorboard
```

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/TLPC1111111/Q-Learning-Graph-Neural-Network-GNN-Model.git
cd Q-Learning-Graph-Neural-Network-GNN-Model
```

### 2. Install Dependencies

Install the required dependencies using the command:

```bash
pip install torch torch-geometric tensorboard
```

### 3. Training the Model

To train the model, simply run:

```bash
python train.py
```

This will train the model on the Cora dataset, log the loss values to TensorBoard, and save the best model checkpoint.

### 4. Monitoring Training with TensorBoard

You can visualize the training process using TensorBoard by running the following command:

```bash
tensorboard --logdir=./Q-Learning/Loss_Curve
```

### 5. Saving and Loading Models

The model checkpoints are saved in the `checkpoints` directory. The best-performing model (based on the size of the MIS) is saved as `zzy_very_handsome.pkl`. You can load this model for inference using:

```python
model = Q_learning()
model.load_state_dict(torch.load('checkpoints/zzy_very_handsome.pkl'))
```

## Parameters

- `select_rate`: Random node selection probability (default: 0.003).
- `gamma`: Discount factor for future rewards in Q-learning (default: 0.8).
- `batch_size`: Number of samples used for each experience replay batch (default: 64).
- `experience_buffer_size`: The size of the experience buffer (default: 3000).
- `iter_times`: Number of iterations for node selection during Q-learning.

## Results

The model outputs the Maximum Independent Set (MIS) after training. The best size of the independent set is recorded, and the model is saved when a better MIS is found during training.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

This project uses the following libraries:

- PyTorch
- PyTorch Geometric
- TensorBoard







