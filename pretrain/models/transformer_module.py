## transformer_module.py
import torch
from torch import nn
from transformers import GPT2Model, GPT2Config,BertConfig,BertModel
#from transformers import RobertaConfig, RobertaModel
import torch_geometric.nn as gnn
import pdb
class TransformerModule(nn.Module):
    def __init__(self, gpt_hidden_dim, n_head=12):
        """
        Initializes the Transformer module using GPT-2 as the base model.
        
        Args:
            output_dim (int): The dimensionality of the output embeddings.
        """
        super(TransformerModule, self).__init__()
        # self.vocab_size = vocab_size
        self.pooling = gnn.global_mean_pool
        # Initialize GPT-2 configuration with the specified output dimension
        # print("output_dim:",output_dim)
        #if using_gnn_hidden_dim:
        self.config = BertConfig(hidden_size = gpt_hidden_dim)
        self.transformer = BertModel(self.config)
        self.gpt_hidden_dim = gpt_hidden_dim
        #self.config = GPT2Config(n_embd = gpt_hidden_dim, n_head=n_head) # 使用默认设置，舍弃原来的初始化方式 n_head = 8,n_embd=output_dim,hidden_size = input_dim
        #self.transformer = GPT2Model(self.config)

 
    

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for generating text summaries from embeddings.
        
        Args:
            embeddings (torch.Tensor): The input embeddings from the GNN module.
        
        Returns:
            torch.Tensor: The output text summary embeddings.
        """
        # Ensure the input embeddings match the expected input dimension
        if embeddings.size(-1) != self.gpt_hidden_dim:
            raise ValueError(f"Expected input dimension {self.gpt_hidden_dim}, but got {embeddings.size(-1)}")
        # Pass the embeddings through the transformer model
        transformer_outputs = self.transformer(inputs_embeds=embeddings)
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Extract the last hidden state as the output
        output = transformer_outputs.last_hidden_state
        # compressed_tensor = output.mean(dim=0)
        output_embeddings = output
        # output_embeddings = self.out_proj(compressed_tensor)
        return embeddings,output_embeddings
