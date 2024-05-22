## gnn_module.py

import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data as Graph
import torch_geometric.nn as gnn
import pdb
import torch.nn.functional as F

def get_simple_gnn_layer(input_dim,output_dim,edge_dim):
    
    # torch.save(deg,f'/root/autodl-tmp/SAT-main/datasets/result/deg.pt')
    # print(edge_dim)
    mlp = mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(True),
            nn.Linear(input_dim, input_dim),
        )
    return gnn.GINEConv(mlp, train_eps=True)
    

class GNNModule(torch.nn.Module):
    def __init__(self, input_dim: int = 32, output_dim: int = 32, num_layers = 5, edge_dim = None):
        super(GNNModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.relu = nn.ReLU()
        self.layers = []
        self.layers.append(get_simple_gnn_layer(input_dim, input_dim, edge_dim)) #.to('cuda'))  # Make sure to place the layer on CUDA
        for _ in range(num_layers-2):
            self.layers.append(get_simple_gnn_layer(input_dim, input_dim, edge_dim)) #.to('cuda'))  # Place each layer on CUDA
        self.layers.append(get_simple_gnn_layer(input_dim, input_dim, edge_dim)) #.to('cuda'))  # Place the last layer on CUDA
        self.gcn = nn.ModuleList(self.layers)
        self.training = True  # Initialize training state

    def train_mode(self, mode: bool):
        self.training = mode

    def forward(self, graph: Graph) -> torch.Tensor:
        for gcn_layer in self.gcn:
            x, edge_index, edge_attr = graph['x'], graph['edge_index'], graph['edge_attr'] #.to('cuda')
            # print(x.shape, edge_index.shape, edge_attr.shape)
            # torch.save(x, "x.pt")
            # torch.save(edge_index, "edge_index.pt")
            # torch.save(edge_attr, "edge_attr.pt")
            x = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
            if self.training:
                x = torch.dropout(x, p=0.5, train=True)
        return x



