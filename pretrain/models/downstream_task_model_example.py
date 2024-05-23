# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
from TrxGNNGPT import TrxGNNGPT
import torch.nn.functional as F
# https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Clone-detection-BigCloneBench/code/model.py    
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        # take <s> token (equiv. to [CLS])
        # take mean/sum of all tokens
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class DownStreamModelExample(nn.Module):   
    def __init__(self, encoder: TrxGNNGPT, config,tokenizer,args):
        super(DownStreamModelExample, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.classifier=ClassificationHead(config)
        
    
        
    def forward(self, graph_data,labels=None): 
        _, _, embeddings_proj, _=self.encoder.get_embedding(graph_data)

        # Apply dropout
        outputs = self.classifier(self.dropout(embeddings_proj))

        logits=outputs
        prob=F.softmax(logits)
        return prob
      
        
 