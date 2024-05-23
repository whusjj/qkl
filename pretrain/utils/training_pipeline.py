## training_pipeline.py
import torch
from torch_geometric.data import DataLoader,Data
from torch.utils.tensorboard import SummaryWriter
import tqdm
from accelerate import Accelerator
import wandb
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
        #assert self.masked_node != self.masked_edge

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
            # Initialize tqdm progress bar with an initial description
            pbar = tqdm.tqdm(data, desc="Initial")
            for i,batch in enumerate(pbar): #(tqdm.tqdm(data, desc=f"Processing items Available memory: {torch.cuda.get_device_properties(self.device).total_memory/ (1024**3) - torch.cuda.memory_allocated(self.device)/ (1024**3):.4f} G")):
                pbar.set_description(f"Processing items Available memory: {torch.cuda.get_device_properties(self.device).total_memory/ (1024**3) - torch.cuda.memory_allocated(self.device)/ (1024**3):.4f} G")
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
                        logger.info( f" steps: {global_step} loss: {round(avg_loss,6)}" )
                    if  self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        logging_loss = tr_loss
                        tr_nb = global_step
                    #if self.save_steps > 0 and global_step % self.save_steps == 0:
                total_loss += a_loss                
            logger.info(f"Epoch {epoch+1}/{self.epochs}, total Loss: {total_loss}")
            logger.info(f"Saving model at the end of epoch {epoch}, "+"{}/{}".format(self.model_args.output_dir, epoch))
            self.save_end_epoch(model, epoch, self.model_args)
        return model
    
    def save_end_epoch(self, model:TrxGNNGPT, epoch:int, model_args: ModelArguments):
        os.makedirs("{}/epoch_{}".format(model_args.output_dir, epoch), exist_ok=True)
        gnn_model_path = "{}/epoch_{}/gnn_model.pth".format(model_args.output_dir, epoch)
        transformer_model_path = "{}/epoch_{}/transformer_model.pth".format(model_args.output_dir, epoch)
        emb_model_path = "{}/epoch_{}/nn_embedding_model.pth".format(model_args.output_dir, epoch)
        
        torch.save(model.embedding_layer.state_dict(), emb_model_path)
        torch.save(model.transformer_module.transformer.state_dict(), transformer_model_path )
        torch.save(model.gnn_module.gcn.state_dict(), gnn_model_path)
        torch.save(model.state_dict(), "{}/epoch_{}/model.pth".format(model_args.output_dir, epoch))
    
    def initialize_model(self, gnn_module, transformer_module, gnn_hidden_dim, mlm_probability, device,\
        is_tighted_lm_head, masked_node, masked_edge) -> TrxGNNGPT:
        model = TrxGNNGPT(gnn_module, transformer_module, self.tokenizer_args, gnn_hidden_dim, \
            mlm_probability, device, is_tighted_lm_head,masked_node, masked_edge)
        return model
    
   