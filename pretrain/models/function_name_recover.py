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
import random
import re

class function_name_recover(nn.Module):   
    def __init__(self, encoder: TrxGNNGPT, model_config,function_name_vocab_length):
        super(function_name_recover, self).__init__()
        self.encoder = encoder
        self.model_config = model_config
        self.lm_head=nn.Linear(model_config.gpt_hidden_dim,function_name_vocab_length)
        self.flag = 0    
    def forward(self, graph_data, preprocessor, generate = 0): 
        self.flag += 1
        graph_data,mask_contract,tokenized_edge,labels = self.deal_with_edge(graph_data, preprocessor)
        _, logits, _, _=self.encoder.get_masked_embedding(graph_data,mask_contract,self.model_config.masked_edge,self.lm_head,preprocessor,generate)
        logits = logits[mask_contract]# num(edge)*seq*64 要改
        logits = self.lm_head(logits)
        # logits = F.softmax(logits,dim=-1)
        return labels,logits,tokenized_edge
    
    def get_embedding(self, graph_data,preprocessor):
        edge_offset = torch.tensor([0,graph_data['edge_attr'].shape[0]])
        mask_contract = self.mask_choose_random(edge_offset,[graph_data['edge_attr_label']])
        #pdb.set_trace()
        labels = [item for sublist in graph_data['edge_attr_label'] for item in sublist]
        labels = [labels[i]['calltrace'] for i in mask_contract]
        labels = self.edge_to_function_name_token(preprocessor.function_dataset.tokenizer,labels)
        labels = torch.stack(preprocessor.function_dataset.tokenizer_node(labels))
        _, logits, _, _=self.encoder.get_masked_embedding(graph_data,mask_contract,self.model_config.masked_edge,self.lm_head,preprocessor,generate = 1)
        # logits = logits[mask_contract]# num(edge)*seq*64 要改
        logits = self.lm_head(logits)
        logits = F.softmax(logits,dim=-1)
        return labels,logits

    def deal_with_edge(self,graph_data, preprocessor):
        edge_offset = [torch.where(graph_data['edge_index'][0] == i)[0][0] for i in graph_data['y'][:-1]]
        edge_offset.append(torch.tensor(graph_data['edge_attr'].shape[0]))
        edge_offset = torch.tensor(edge_offset)
        mask_contract = self.mask_choose_random(edge_offset,graph_data['edge_attr_label'])
        group_edge_attr_label = [item for sublist in graph_data['edge_attr_label'] for item in sublist].copy()
        # labels = [group_edge_attr_label[i][0]['calltrace'] for i in mask_contract]
        # labels = self.edge_to_function_name_token(preprocessor.function_dataset.tokenizer,labels)
        # labels = torch.stack(preprocessor.function_dataset.tokenizer_node(labels))
        unmasked_edge = [group_edge_attr_label[i] for i in mask_contract]
        masked_edge,labels = self.mask_first_function(unmasked_edge)
        labels = torch.stack(preprocessor.function_dataset.tokenizer_node(labels))
        tokenized_edge = preprocessor.esperanto_dataset.tokenizer_node(masked_edge)
        tokenized_edge = torch.stack(tokenized_edge)
        for i in range(len(mask_contract)):
            graph_data['edge_attr'][mask_contract[i]]  = tokenized_edge[i]
        seq = 128#参数化
        labels_pos = torch.zeros_like(tokenized_edge)
        for i,edge in enumerate(tokenized_edge):
            labels_length = torch.where(labels[i] == 2)[0]
            last_token = torch.where(edge == 2)[0]
            if last_token < seq - 10:#参数化
                edge[last_token : last_token + 10] = labels[i,1:11]
                labels_pos[i][last_token : last_token + labels_length] = 1
            else:
                edge[-10 : ] = labels[i,1:11]
                labels_pos[i][-10 : -10 + labels_length] = 1
        tokenized_edge = torch.where((tokenized_edge*labels_pos) == 0, -100, (tokenized_edge*labels_pos))
        return graph_data,mask_contract,tokenized_edge,labels


    def get_mean(self, edge_offset, logits):
        embedding = []
        for i in range(edge_offset.shape[0] - 1):
            prob = logits[edge_offset[i]:edge_offset[i+1]].mean(dim = 0)
            embedding.append(prob)
        return torch.stack(embedding)


    def mask_choose_random(self, edge_offset, edge_attr_label):
        mask_contract = []
        edge_label =[]
        for i in range(edge_offset.shape[0] - 1):
            for j, label in enumerate(edge_attr_label[i]):
                if 'calltrace' in label[0]:
                    edge_label.append(j)   
            rand = edge_offset[i] + random.choice(edge_label)
            mask_contract.append(rand)
            edge_label = []
        return torch.stack(mask_contract)
    
    def edge_to_function_name_token(self,tokenizer,labels):
        token = []
        for i in labels:
            call_function = re.findall(r"call: ([^-]+)", i) 
            call_function = [word  for word in call_function if 'calltrace' not in word]
            call_function = [re.sub(r'\(.*?\)', '', word) for word in call_function ]
            actual_token = ''
            for word in call_function:
                actual_token += word
            token.append(actual_token)
        return token
    
    def mask_first_function(self,edges):
        label = []
        for edge in edges :
            text = edge[0]['calltrace']
            pos = text.find("call: ")
            if pos != -1:
        # 计算 "call:" 后第一个单词的起始位置
                start_pos = pos + len("call: ")  # 跳过 "call:" 的长度
        # 寻找第一个空格，即单词的结束位置
                end_pos = text.find("-from:", start_pos)
                if end_pos == -1:
                    end_pos = len(text)
        # 替换第一个单词为 "<mask>"
                if start_pos != end_pos:
                    label.append(text[start_pos:end_pos])
                else:
                    label.append(' ')            
                text = text[:start_pos] + "<mask> " + text[end_pos:]
            else:
                text = text
            edge[0]['calltrace'] = text
        label = [re.sub(r'\(.*?\)', '', word) for word in label]
        return edges,label

# 示例

      