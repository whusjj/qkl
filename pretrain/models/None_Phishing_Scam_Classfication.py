# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
from pretrain.models.TrxGNNGPT import TrxGNNGPT
import torch.nn.functional as F
import pdb
import numpy as np
from transformers import get_linear_schedule_with_warmup
# https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Clone-detection-BigCloneBench/code/model.py    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config,num_class):
        super().__init__()
        self.dense = nn.Linear(config.gnn_hidden_dim, config.gnn_hidden_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.gnn_hidden_dim, num_class)

    def forward(self, features, **kwargs):
        # take <s> token (equiv. to [CLS])
        # take mean/sum of all tokens
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class None_Phishing_Scam_Classfication(nn.Module):   
    def __init__(self, encoder: TrxGNNGPT, model_config,num_class):
        super(None_Phishing_Scam_Classfication, self).__init__()
        self.encoder = encoder
        self.num_class = num_class
        self.classifier=ClassificationHead(model_config,num_class)    
    def forward(self, graph_data,labels=None): 
        
        _, _, embeddings_proj, _=self.encoder.get_embedding(graph_data)
        # Apply dropout
        outputs = self.classifier(embeddings_proj)
        logits=torch.sum(outputs,axis=0)
        prob=F.softmax(logits)
        return torch.tensor(graph_data.y),prob
    

      
        
 