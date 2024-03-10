## gnn_module.py

import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data as Graph
import torch_geometric.nn as gnn
import pdb
def get_simple_gnn_layer(input_dim,output_dim,edge_dim, deg):
    
    aggregators = ['mean', 'sum', 'max']
    scalers = ['identity']
    torch.save(deg,f'/root/autodl-tmp/SAT-main/datasets/result/deg.pt')
    print(edge_dim)
    layer = gnn.PNAConv(input_dim, output_dim,
                           aggregators=aggregators, scalers=scalers,
                           deg=deg, towers=4, pre_layers=1, post_layers=1,
                           divide_input=True, edge_dim=edge_dim)
    return layer

class GNNModule(torch.nn.Module):
    def __init__(self, input_dim: int = 32, output_dim: int = 32, embed_dim: int = 32, num_layers = 5,edge_dim = None,deg_path = None):
        """
        Initializes the GNNModule with input and output dimensions.
        
        Parameters:
        - input_dim (int): Dimensionality of the input features.
        - output_dim (int): Dimensionality of the output embedding.
        """
        super(GNNModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.deg = torch.load(deg_path)
        self.relu = nn.ReLU()
        self.layers = []
        self.layers.append(get_simple_gnn_layer(input_dim, input_dim,edge_dim,self.deg))
        for _ in range(num_layers-2):
            self.layers.append(get_simple_gnn_layer(input_dim, input_dim,edge_dim,self.deg))
        self.layers.append(get_simple_gnn_layer(input_dim, input_dim,edge_dim,self.deg))
        self.gcn = nn.ModuleList(self.layers)
        self.training = True  # Initialize training state

    def train_mode(self, mode: bool):
        """
        Sets the module's training mode.

        Parameters:
        - mode (bool): If True, sets the module to training mode. Otherwise, sets it to evaluation mode.
        """
        self.training = mode

    def forward(self, graph: Graph) -> torch.Tensor:
        """
        Processes the input graph through two GCN layers to produce embeddings.

        Parameters:
        - graph (Graph): A PyTorch Geometric Data object representing the input graph.

        Returns:
        - torch.Tensor: The output embedding tensor from the GNN.
        """
        for gcn_layer in self.gcn:
            x, edge_index,edge_attr = graph['node feature'], graph['Edge indices'][0],graph['Edge attributes']
            # pdb.set_trace()
            x = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr[0].T))
            
            if self.training:
                x = torch.dropout(x, p=0.1, train=True)
        

        return x
