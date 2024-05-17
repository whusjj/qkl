## training_pipeline.py
import torch
from torch_geometric.data import DataLoader,Data
from torch.utils.tensorboard import SummaryWriter
from data_preprocessor import DataPreprocessor
from gnn_module import GNNModule
from transformer_module import TransformerModule
import tqdm
from accelerate import Accelerator
import wandb
import logging
import os
import apex
from transformers import get_linear_schedule_with_warmup
import pdb
from MyTokenizer import EsperantoDataset
from MyEmbedding import LSTMEmbedding
import json
import random
import numpy

def get_vocab_size():
    # 读取 esperberto-vocab.json 文件
    with open('esperberto-vocab.json', 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    # 获取词汇表的长度
    vocab_size = len(vocab_data)
    return vocab_size

def setup_training_log(accelerator,project_name,lr,batch_size):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt = '%m/%d/%Y %H:%M:%S',level=logging.INFO,handlers=[
        logging.FileHandler(f'log/training_log_{accelerator.process_index}.txt'),
        logging.StreamHandler()
        ]
    ) 
    if accelerator.is_main_process:
        wandb.init(project=project_name,name = 'pretraining', config={"learning_rate": lr, "batch_size": batch_size},  # 参数配置，可记录训练参数
                  resume="allow")
        run_name = wandb.run.name
        tb_writer = SummaryWriter()
        tb_writer.add_hparams({"lr": lr, "bsize": batch_size},{'0':0})
        logger.setLevel(logging.INFO)
    else:
        tb_writer = None
        run_name = ''
        logger.setLevel(logging.ERROR)
    return logger,tb_writer,run_name





class CustomModel(torch.nn.Module):
    def __init__(self, gnn_module: GNNModule, transformer_module: TransformerModule,hidden_dim = 64,mlm_probability = 0.15,device = 'cpu',pre_train=0):
        super(CustomModel, self).__init__()
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
        return data, labels
    
    


class TrainingPipeline:
    def __init__(self, epochs: int = 100, batch_size: int = 32,hidden_dim = 64,mlm_probability = 0.15,device = "cpu", vocab = None,pre_train = 0 ): # 调整batch_size,原来是32,可以改成16，等8的倍数
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = 0.001
        self.output_dir = './model'
        self.fp16 = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.start_step = 100
        self.gradient_accumulation_steps = 10
        self.max_grad_norm = 5
        self.logging_steps = 1000
        self.save_steps = 1000
        self.hidden_dim = hidden_dim
        self.mlm_probability = mlm_probability
        self.device = device
        self.vocab_size = vocab

    def collate_fn(self,batch_size, data):
        if data == None:
            print("no data!")
            return None
        concatenated_tensors = []
        for i in range(0, len(data)//batch_size):
            edge_attrs_batch = []
            xs_batch = []
            edge_index_batch = []
            mark = []
            offset = 0
            mark.append(offset)
            for sample in data[i:i+batch_size]:
                edge_index_batch.append(sample['edge_index'] + offset)
                offset = offset + sample['x'].shape[0]
                edge_attrs_batch.append(sample['edge_attr'])
                xs_batch.append(sample['x'])
                mark.append(offset)
            group_edge_attr = torch.cat(edge_attrs_batch, dim=0)
            group_x = torch.cat(xs_batch, dim=0)
            group_edge_index = torch.cat(edge_index_batch, dim=1)
            group_y = torch.tensor(mark, dtype=int)
            group = Data(x = group_x, y = group_y, edge_attr=group_edge_attr,edge_index=group_edge_index)
            concatenated_tensors.append(group)
        return concatenated_tensors
        

    def train(self, dataset,gnn_module, transformer_module) -> CustomModel:
        # data = dataset.dataset
        # random.shuffle(data)
        # accelerator = Accelerator()
        data =  self.collate_fn(self.batch_size,dataset)
        model = self.initialize_model(gnn_module,transformer_module,self.hidden_dim,self.mlm_probability,self.device)
        model.to(self.device)
        # data = DataLoader(dataset,shuffle=True,batch_size=8,collate_fn = )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000,
                                                num_training_steps=100)
        checkpoint_last = os.path.join(self.output_dir, 'checkpoint-last')
        scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
        optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
        if os.path.exists(scheduler_last):
            scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
        if os.path.exists(optimizer_last):
            optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu")) 
        if self.fp16:
            try:
                from apex.apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        tr_loss, logging_loss,avg_loss,tr_nb = 0.0, 0.0,0.0,0
        # model,optimizer,data  = accelerator.prepare(model,optimizer,data)
        criterion = torch.nn.CrossEntropyLoss(reduce=False)
        model.train()
        step = 0 
        global_step = self.start_step
        for epoch in range(self.epochs):
            print("epoch:",epoch)
            total_loss = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for i,batch in tqdm.tqdm(enumerate(data)):
                step += 1
                labels,predictions = model(batch)
                loss_1 = criterion(predictions.view(-1, self.vocab_size).to(self.device), labels.view(-1).to(self.device))
                a_loss = loss_1.mean()
                if self.gradient_accumulation_steps > 1:
                    loss = a_loss / self.gradient_accumulation_steps
                if self.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                tr_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()  
                    global_step += 1
                    output_flag=True
                    avg_loss=round((tr_loss - logging_loss) /(global_step- tr_nb),6)
                    if global_step % 100 == 0:
                        print(" steps: %s loss: %s", global_step, round(avg_loss,6))
                    if  self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        logging_loss = tr_loss
                        tr_nb = global_step
                    #if self.save_steps > 0 and global_step % self.save_steps == 0:
                total_loss += a_loss                
            print(f"Epoch {epoch+1}/{self.epochs}, total Loss: {total_loss}")
        return model

    def initialize_model(self,gnn_module,transformer_module,hidden_dim,mlm_probability,device) -> CustomModel:
        model = CustomModel(gnn_module, transformer_module,hidden_dim,mlm_probability,device)
        return model

# Note: It's assumed that the dataset class provides a method `get_labels(batch)` to access the target labels.
# This should be implemented in the dataset class to ensure compatibility with this training pipeline.
