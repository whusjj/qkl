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
    def __init__(self, input_dim: int = 32, output_dim: int = 32, embed_dim: int = 32, num_layers = 5, edge_dim = None,pre_train = 0):
        super(GNNModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.relu = nn.ReLU()
        self.layers = []
        self.layers.append(get_simple_gnn_layer(input_dim, input_dim, edge_dim).to('cuda'))  # Make sure to place the layer on CUDA
        for _ in range(num_layers-2):
            self.layers.append(get_simple_gnn_layer(input_dim, input_dim, edge_dim).to('cuda'))  # Place each layer on CUDA
        self.layers.append(get_simple_gnn_layer(input_dim, input_dim, edge_dim).to('cuda'))  # Place the last layer on CUDA
        self.gcn = nn.ModuleList(self.layers)
        if pre_train == 0:
            self.gcn.load_state_dict(torch.load('/root/autodl-tmp/pre_train/generate/gnn_model.pth'))
        self.training = True  # Initialize training state
        self.time = 0

    def train_mode(self, mode: bool):
        self.training = mode

    def forward(self, graph: Graph) -> torch.Tensor:
        self.time=self.time + 1
        for gcn_layer in self.gcn:
            x, edge_index, edge_attr = graph['x'].to('cuda'), graph['edge_index'].to('cuda'), graph['edge_attr'].to('cuda')
            x = self.relu(gcn_layer(x, edge_index.to(torch.long), edge_attr=edge_attr))
            if self.training:
                x = torch.dropout(x, p=0.5, train=True)
        return x



