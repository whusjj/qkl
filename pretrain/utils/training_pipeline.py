## training_pipeline.py
import torch
from torch_geometric.data import DataLoader,Data
from torch.utils.tensorboard import SummaryWriter
import tqdm
from accelerate import Accelerator
import wandb
import logging
import os
import apex
from utils.argument import TrainingArguments, ModelArguments, TokenizerArguments
from transformers import get_linear_schedule_with_warmup
from models.TrxGNNGPT import TrxGNNGPT
from loguru import logger
import random




class TrainingPipeline:
    def __init__(self,train_args:TrainingArguments,model_args: ModelArguments, tokenizer_args: TokenizerArguments): # 调整batch_size,原来是32,可以改成16，等8的倍数
        self.tokenizer_args = tokenizer_args 
        self.train_args = train_args
        self.model_args = model_args
        self.epochs = train_args.epochs
        self.batch_size = train_args.batch_size
        self.lr = train_args.learning_rate
        self.output_dir = model_args.output_dir
        self.fp16 = train_args.fp16
        self.device = train_args.device
        self.start_step = train_args.start_step
        self.gradient_accumulation_steps = train_args.gradient_accumulation_steps
        self.max_grad_norm = train_args.max_grad_norm
        self.logging_steps = train_args.logging_steps
        self.save_steps = train_args.save_steps
        self.gnn_hidden_dim = train_args.gnn_hidden_dim
        self.mlm_probability = train_args.mlm_probability
        self.vocab_size = tokenizer_args.vocab_size
        self.is_tighted_lm_head = train_args.is_tighted_lm_head
        self.masked_node = train_args.masked_node
        self.masked_edge = train_args.masked_edge
        assert self.masked_node != self.masked_edge

    def collate_fn(self,batch_size, data):
        if data == None:
            logger.warning("no data!")
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
            # for d in edge_attrs_batch:
            #     print(d.shape)
            group_edge_attr = torch.cat(edge_attrs_batch, dim=0)
            group_x = torch.cat(xs_batch, dim=0)
            group_edge_index = torch.cat(edge_index_batch, dim=1)
            group_y = torch.tensor(mark, dtype=int)
            group = Data(x = group_x, y = group_y, edge_attr=group_edge_attr,edge_index=group_edge_index)
            concatenated_tensors.append(group)
        
        
        return concatenated_tensors
        

    def train(self, dataset, gnn_module, transformer_module) -> TrxGNNGPT:
        # data = dataset.dataset
        # random.shuffle(data)
        # accelerator = Accelerator()
        data =  self.collate_fn(self.batch_size, dataset)
        model = self.initialize_model(gnn_module, transformer_module, self.gnn_hidden_dim, \
            self.mlm_probability, self.device, self.is_tighted_lm_head, self.masked_node, self.masked_edge)
        model.to(self.device)
        logger.info(f"model is on {model.device}")
        logger.info(f"gnn_module is on {next(gnn_module.parameters()).device}")
        logger.info(f"transformer_module is on {next(transformer_module.parameters()).device}")
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
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                
        tr_loss, logging_loss,avg_loss,tr_nb = 0.0, 0.0,0.0,0
        # model,optimizer,data  = accelerator.prepare(model,optimizer,data)
        criterion = torch.nn.CrossEntropyLoss(reduce=False)
        model.train()
        step = 0 
        global_step = self.start_step
        for epoch in range(self.epochs):
            logger.info(f"epoch: {epoch}")
            total_loss = 0
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for i,batch in enumerate(tqdm.tqdm(data, desc=f"Processing items Available memory: {torch.cuda.get_device_properties(self.device).total_memory/ (1024**3) - torch.cuda.memory_allocated(self.device)/ (1024**3):.4f} G")):
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
                        logger.info(" steps: %s loss: %s", global_step, round(avg_loss,6))
                    if  self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        logging_loss = tr_loss
                        tr_nb = global_step
                    #if self.save_steps > 0 and global_step % self.save_steps == 0:
                total_loss += a_loss                
            logger.info(f"Epoch {epoch+1}/{self.epochs}, total Loss: {total_loss}")
        return model

    def initialize_model(self, gnn_module, transformer_module, gnn_hidden_dim, mlm_probability, device,\
        is_tighted_lm_head, masked_node, masked_edge) -> TrxGNNGPT:
        model = TrxGNNGPT(gnn_module, transformer_module, self.tokenizer_args, gnn_hidden_dim, \
            mlm_probability, device, is_tighted_lm_head,masked_node, masked_edge)
        return model
    
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

# Note: It's assumed that the dataset class provides a method `get_labels(batch)` to access the target labels.
# This should be implemented in the dataset class to ensure compatibility with this training pipeline.
