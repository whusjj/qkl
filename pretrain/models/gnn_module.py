## gnn_module.py

import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data as Graph
import torch_geometric.nn as gnn
import pdb
import torch.nn.functional as F

def get_simple_gnn_layer(input_dim,output_dim,hidden_dim):
    
    # torch.save(deg,f'/root/autodl-tmp/SAT-main/datasets/result/deg.pt')
    # print(edge_dim)
    mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim , output_dim),
            # nn.ReLU(True)
            # nn.Dropout(p=0.2)
        )
    return gnn.GINEConv(mlp, train_eps=True)
    

class GNNModule(torch.nn.Module):
    def __init__(self, input_dim: int = 32, output_dim: int = 32, num_layers = 3, edge_dim = None,sequence = 128):
        super(GNNModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = int(2 * input_dim)  // sequence
        self.num_layers = num_layers
        self.sequence = sequence
        self.relu = nn.ReLU()
        self.layers = []
        self.mlp_layers = []
        self.layers_output = []
        self.number = 0
        # layer1 = get_simple_gnn_layer(input_dim, input_dim , input_dim // 2)
        # layer2 = get_simple_gnn_layer(input_dim, input_dim, input_dim // 2)
        layer3 = get_simple_gnn_layer(input_dim, int(2 * input_dim), input_dim // 2)
        # self.layers.append(layer1) #.to('cuda'))  # three layer of GNN
        # self.layers.append(layer2) #.to('cuda'))  
        self.layers.append(layer3) #.to('cuda'))  
        self.gcn = nn.ModuleList(self.layers)
        self.mlp = nn.ModuleList(self.mlp_layers)
        # self.mlp = [1,2,3]
        #self.training = True  # Initialize training state

    def train_mode(self, mode: bool):
        self.training = mode

    def forward(self, graph: Graph) -> torch.Tensor:
        self.number += 1
        x, edge_index, edge_attr = graph['x'], graph['edge_index'], graph['edge_attr'] #.to('cuda')
        for i, gcn_layer in enumerate(self.gcn):
            # torch.save(edge_attr, "edge_attr.pt")
            x = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
            # edge_attr = mlp_layer(edge_attr)
            if self.training:
                x = torch.dropout(x, p=0.5, train=True)

        return x




