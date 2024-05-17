## transformer_module.py
import torch
from torch import nn
from transformers import GPT2Model, GPT2Config
import torch_geometric.nn as gnn
import pdb
class TransformerModule(nn.Module):
    def __init__(self, input_dim: int = 768, output_dim: int = 768,pre_train = 0):
        """
        Initializes the Transformer module using GPT-2 as the base model.
        
        Args:
            input_dim (int): The dimensionality of the input embeddings.
            output_dim (int): The dimensionality of the output text summary embeddings.
        """
        super(TransformerModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling = gnn.global_mean_pool
        # Initialize GPT-2 configuration with the specified output dimension
        self.config = GPT2Config(n_head = 8,n_embd=output_dim,hidden_size = input_dim)
        state_dict = torch.load('/root/autodl-tmp/pre_train/generate/transformer_model.pth')
        self.transformer = GPT2Model(self.config)
        # print(self.transformer.config)
        if pre_train == 0 :
            self.transformer.load_state_dict(state_dict, strict=False)
        # print(self.transformer.config)
        self.out_proj = nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size, self.transformer.config.hidden_size),
                nn.ReLU(True),
                nn.Linear(self.transformer.config.hidden_size, 3)
            )
        

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for generating text summaries from embeddings.
        
        Args:
            embeddings (torch.Tensor): The input embeddings from the GNN module.
        
        Returns:
            torch.Tensor: The output text summary embeddings.
        """
        # Ensure the input embeddings match the expected input dimension
        if embeddings.size(-1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, but got {embeddings.size(-1)}")
        # Pass the embeddings through the transformer model
        transformer_outputs = self.transformer(inputs_embeds=embeddings)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Extract the last hidden state as the output
        output = transformer_outputs.last_hidden_state
        compressed_tensor = output
        # output_embeddings = output
        output_embeddings = self.out_proj(compressed_tensor)
        return embeddings,output_embeddings
