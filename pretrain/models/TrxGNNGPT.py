
from models.gnn_module import GNNModule
from models.transformer_module import TransformerModule
from LSTMEmbedding import LSTMEmbedding
import random
import json
import torch



class TrxGNNGPT(torch.nn.Module):
    def __init__(self, gnn_module: GNNModule, transformer_module: TransformerModule,hidden_dim = 64,mlm_probability = 0.15,device = 'cpu',pre_train=0):
        super(TrxGNNGPT, self).__init__()
        self.gnn_module = gnn_module
        self.transformer_module = transformer_module
        # self.esperanto_dataset = EsperantoDataset()
        self.vocab_size = get_vocab_size()
        self.text_embedding_model = LSTMEmbedding(self.vocab_size)
        self.pad_id = 1
        if pre_train == 0:
            embedding_model = torch.load("/root/autodl-tmp/ETHGPT-main/large-scale-regression/tokengt/pretrain/nn_embedding_model.pth")
            self.embedding_model = torch.nn.Embedding(self.vocab_size,hidden_dim,padding_idx=self.pad_id).from_pretrained(embedding_model['weight'])
        else:
            self.embedding_model = torch.nn.Embedding(self.vocab_size,hidden_dim,padding_idx=self.pad_id)
        self.mlm_probability = mlm_probability
        self.device = device
        self.len = 128
        
        self.special_token_id = [0,1,2,3]# 特殊令牌id
        self.mask_ids = 4
        self.hidden_dim = hidden_dim
        self.lm_head = torch.nn.Linear(hidden_dim,self.vocab_size)
        # self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        # self.text_embedding_model = AutoModel.from_pretrained(text_model_name)

    def forward(self, graph_data):
        # tmp = graph_data
        tmp = graph_data.clone()
        rand = random.randint(0,1)
        rand = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 使用 EsperantoDataset 实例调用 tokenizer_node 方法
        # token_data = self.esperanto_dataset.tokenizer_node(graph_data["x"])
        labels = torch.empty(1,1)
        masked_node_data = torch.empty(1,1)
        masked_edge_data = torch.empty(1,1)
        if rand:#rand为1就掩码点否则掩码边
            masked_node_data,labels= self.mask_label(tmp["x"],rand,graph_data['y'])
        else:
            masked_node_data = tmp["x"]
        token_data = masked_node_data.to(device)
        # print(token_data.shape)
        embeddings = []
        for batch in token_data:
            batch = torch.unsqueeze(batch, dim=0)
            batch = batch.to(device)
            # print(type(batch))
            # print(batch)
            output = self.embedding_model(batch)
            embeddings.append(output)
        embeddings = torch.cat(embeddings, dim=0)
        tmp["x"] = embeddings.reshape(embeddings.shape[0],-1).to(device)
        if not rand:
            masked_edge_data,labels= self.mask_label(tmp["edge_attr"],rand,graph_data['y'],index = graph_data['edge_index'])
        else :
            masked_edge_data = tmp["edge_attr"]
        # token_data = self.esperanto_dataset.tokenizer_node(graph_data["edge_attr"])
        token_data = masked_edge_data.to(device)
        embeddings = []
        for batch in token_data:
            batch = torch.unsqueeze(batch, dim=0)
            batch = batch.to(device)
            # print(batch)
            # print(type(batch))
            output = self.embedding_model(batch)
            embeddings.append(output)
        embeddings = torch.cat(embeddings, dim=0)
        tmp["edge_attr"] = embeddings.reshape(embeddings.shape[0],-1).to(device)
        embeddings = self.gnn_module.forward(tmp)
        if not rand:
            edge_embedding = torch.empty(labels.shape[0],embeddings.shape[1])
            for i in range(0,labels.shape[0]):
                edge_embedding[i] = embeddings[tmp["edge_index"][0][i]] + embeddings[tmp["edge_index"][1][i]]
            embeddings = edge_embedding
        embeddings = embeddings.reshape(labels.shape[0],-1,self.hidden_dim).to(device)
        # graph_data = tmp
        text_summary = self.transformer_module(embeddings)
        logits = self.lm_head(text_summary[1])
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return labels,probs

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

    def mask_label(self,data,rand,y,index = None):
        labels = data.clone()
        rand = 1
        if rand == 0:
            probability_matrix = self.mask_node(labels,y)
        else:
            probability_matrix = self.mask_edge(labels,y,index)
        # pdb.set_trace()
        special_tokens_mask = [self.get_special_tokens_mask(val).tolist() for val in labels]
        tensor_list = torch.tensor(special_tokens_mask, dtype=torch.int).to(self.device)
        masked_indices = probability_matrix*tensor_list
        masked_indices = masked_indices.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(device) & masked_indices
        # data[indices_replaced] = self.mask_ids
        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().to(device) & masked_indices & ~indices_replaced
        # random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long).to(device)
        # data[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return labels, data
  