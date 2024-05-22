
from models.gnn_module import GNNModule
from models.transformer_module import TransformerModule
from models.LSTMEmbedding import LSTMEmbedding
import random
import json
import torch
import torch.nn as nn
from utils.argument import TokenizerArguments
from loguru import logger

class TrxGNNGPT(nn.Module):
    def __init__(self, gnn_module: GNNModule, transformer_module: TransformerModule, tokenizaer_args:TokenizerArguments, gnn_hidden_dim = 64,mlm_probability = 0.15, device = 'cpu', is_tighted_lm_head = True, masked_node=False, masked_edge=True):
        super(TrxGNNGPT, self).__init__()
        self.gnn_module = gnn_module
        self.transformer_module = transformer_module
        # self.esperanto_dataset = EsperantoDataset()
        self.vocab_size = tokenizaer_args.vocab_size
        # self.text_embedding_model = LSTMEmbedding(self.vocab_size)
        self.pad_id = tokenizaer_args.PAD_TOKEN_ID
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, gnn_hidden_dim, padding_idx=self.pad_id)
        self.mlm_probability = mlm_probability
        self.device = device
        self.masked_node = masked_node
        self.masked_edge = masked_edge
        self.special_token_id = [tokenizaer_args.S_TOKEN_ID, tokenizaer_args.PAD_TOKEN_ID, tokenizaer_args.E_TOKEN_ID, tokenizaer_args.UNK_TOKEN_ID] #[0,1,2,3]# 特殊令牌id
        self.mask_id = tokenizaer_args.MASK_TOKEN_ID
        self.gnn_hidden_dim = gnn_hidden_dim
        self.liner_connection_1 = None
        self.liner_connection_2 = None
        if self.transformer_module.gpt_hidden_dim != self.gnn_hidden_dim:
            self.liner_connection_1 = torch.nn.Linear(gnn_hidden_dim, self.transformer_module.gpt_hidden_dim)
            self.liner_connection_2 = torch.nn.Linear(self.transformer_module.gpt_hidden_dim, gnn_hidden_dim)
        
        if is_tighted_lm_head:
            # Tie weights
            #assert self.transformer_module.gpt_hidden_dim == self.gnn_hidden_dim
            self.lm_head = torch.nn.Linear(self.gnn_hidden_dim, self.vocab_size, bias=False)
            self._tie_or_clone_weights(self.lm_head, self.embedding_layer)
        else:
            self.lm_head = torch.nn.Linear(self.gnn_hidden_dim, self.vocab_size)
   
    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending on whether TorchScript is used"""
        # if self.torchscript:
        #     output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        # else:
        #     output_embeddings.weight = input_embeddings.weight
        output_embeddings.weight = input_embeddings.weight

        if hasattr(output_embeddings, "bias") and output_embeddings.bias is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
                "constant",
                0
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings
            
    def forward(self, graph_data):
        # tmp = graph_data
        #logger.info(f'Total memory: {torch.cuda.get_device_properties(self.device).total_memory} Available memory: {torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(self.device)}')
        tmp = graph_data.clone()
        # rand = random.randint(0,1)
        # rand = 0
        device = self.device
        # 使用 EsperantoDataset 实例调用 tokenizer_node 方法
        # token_data = self.esperanto_dataset.tokenizer_node(graph_data["x"])
        labels = None
        masked_node_data = None
        masked_edge_data = None
        if self.masked_node: #rand为1就掩码点否则掩码边
            masked_node_data,labels= self.mask_label(tmp["x"], self.masked_node, False,graph_data['y'])
        else:
            masked_node_data = tmp["x"]
        token_data = masked_node_data.to(device)
        # print(token_data.shape)
        embeddings = []
        for batch in token_data:
            batch = torch.unsqueeze(batch, dim=0)
            batch = batch.to(device)
            output = self.embedding_layer(batch)
            embeddings.append(output)
        embeddings = torch.cat(embeddings, dim=0)
        tmp["x"] = embeddings.reshape(embeddings.shape[0],-1).to(device)
        if self.masked_edge:
            masked_edge_data,labels= self.mask_label(tmp["edge_attr"],False, self.masked_edge, graph_data['y'],index = graph_data['edge_index'])
        else :
            masked_edge_data = tmp["edge_attr"]
        # token_data = self.esperanto_dataset.tokenizer_node(graph_data["edge_attr"])
        token_data = masked_edge_data.to(device)
        embeddings = []
        for batch in token_data:
            batch = torch.unsqueeze(batch, dim=0)
            batch = batch.to(device)
            output = self.embedding_layer(batch)
            embeddings.append(output)
        embeddings = torch.cat(embeddings, dim=0)
        tmp["edge_attr"] = embeddings.reshape(embeddings.shape[0],-1).to(device)
        tmp['edge_index'] = tmp['edge_index'].to(device)
        embeddings = self.gnn_module.forward(tmp)
        if self.masked_edge:
            edge_embedding = torch.empty(labels.shape[0],embeddings.shape[1])
            for i in range(0,labels.shape[0]):
                edge_embedding[i] = embeddings[tmp["edge_index"][0][i]] + embeddings[tmp["edge_index"][1][i]]
            embeddings = edge_embedding
        embeddings = embeddings.reshape(labels.shape[0],-1, self.gnn_hidden_dim).to(device)
        # graph_data = tmp
        if self.liner_connection_1:
            embeddings = self.liner_connection_1(embeddings)
        text_summary = self.transformer_module(embeddings)
        if self.liner_connection_2:
            embeddings_proj = self.liner_connection_2(text_summary[1])
        logits = self.lm_head(embeddings_proj)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        #logger.info(f'Total memory: {torch.cuda.get_device_properties(self.device).total_memory} Available memory: {torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(self.device)}')
        return labels, probs

    def get_special_tokens_mask(self,value):
        condition = torch.isin(value, torch.tensor(self.special_token_id,dtype=torch.int))
        output_tensor = torch.where(condition, torch.zeros_like(value), torch.ones_like(value))
        return output_tensor
    
    def mask_edge(self,labels,y,index):
        prob_matrix = []
        for i in range(y.shape[0] - 1):
            edge_number = ((index[0] >= y[i]) & (index[0] < y[i+1])).sum()
            #求出每一个图的边数
            rand = random.randint(0,edge_number-1)
            indices_replaced = torch.bernoulli(torch.full((edge_number, 1),self.mlm_probability)).bool().to(self.device)
            # probability_matrix[torch.nonzero(indices_replaced == False)[0][0],:]  = 1
            probability_matrix = torch.zeros((edge_number,labels.shape[1])).to(self.device)
            if torch.any(indices_replaced == True):
                probability_matrix[torch.nonzero(indices_replaced == True)[0][0],:]  = 1
            else:
                probability_matrix[rand,:] = 1
            prob_matrix.append(probability_matrix)
        prob_matrix = torch.cat(prob_matrix, dim = 0)
        return prob_matrix

    def mask_node(self,labels,y):
        prob_matrix = []
        for i in range(y.shape[0] - 1):
            rand = random.randint(0,y[i+1]-y[i]-1)
            indices_replaced = torch.bernoulli(torch.full((y[i + 1] - y[i], 1),self.mlm_probability)).bool().to(self.device)
            # probability_matrix[torch.nonzero(indices_replaced == False)[0][0],:]  = 1
            probability_matrix = torch.zeros((y[i + 1] - y[i],labels.shape[1])).to(self.device)
            if torch.any(indices_replaced == True):
                probability_matrix[torch.nonzero(indices_replaced == True)[0][0],:]  = 1
            else:
                probability_matrix[rand,:] = 1
            prob_matrix.append(probability_matrix)
        prob_matrix = torch.cat(prob_matrix, dim = 0)
        return prob_matrix

    def mask_label(self,data, masked_node, masked_edge,y,index = None):
        labels = data.clone()
        masked_data = data.clone()
        # print("=============")
        # print(data.shape)
        # torch.save(data,"data.pt")
        # torch.save(y,"y.pt")
        # rand = 1
        if masked_node:
            probability_matrix = self.mask_node(labels,y)
        if masked_edge:
            probability_matrix = self.mask_edge(labels,y,index)
        # pdb.set_trace()
        special_tokens_mask = [self.get_special_tokens_mask(val).tolist() for val in labels]
        tensor_list = torch.tensor(special_tokens_mask, dtype=torch.int).to(self.device)
        masked_indices = probability_matrix*tensor_list
        masked_indices = masked_indices.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        masked_data[masked_indices] = self.mask_id
        # Assert to check if all elements in labels are -100
       # assert torch.all(labels == -100), f"Not all elements in labels are -100, {data}, \n {labels}, \n masked_node {masked_node}, masked_edge {masked_edge}"
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(device) & masked_indices
        # data[indices_replaced] = self.mask_ids
        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().to(device) & masked_indices & ~indices_replaced
        # random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long).to(device)
        # data[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return masked_data, labels
  