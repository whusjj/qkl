## training_pipeline.py
import torch
from torch_geometric.data import DataLoader,Data
from torch.utils.tensorboard import SummaryWriter
import tqdm
from accelerate import Accelerator
import wandb
import os
import apex
from .argument import TrainingArguments, ModelArguments, TokenizerArguments
from transformers import get_linear_schedule_with_warmup
from ..models.TrxGNNGPT import TrxGNNGPT
from loguru import logger
import random
import pdb
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import re
import copy
# import deepspeed

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
            group_edge_attr_label = []
            offset = 0
            mark.append(offset)
            for sample in data[i:i+batch_size]:
                edge_index_batch.append(sample['edge_index'] + offset)
                offset = offset + sample['x'].shape[0]
                edge_attrs_batch.append(sample['edge_attr'])
                xs_batch.append(sample['x'])
                mark.append(offset)
                if hasattr(sample,'edge_attr_label'):
                    group_edge_attr_label.append(sample['edge_attr_label'])
            # for d in edge_attrs_batch:
            #     print(d.shape)
            group_edge_attr = torch.cat(edge_attrs_batch, dim=0)
            group_x = torch.cat(xs_batch, dim=0)
            group_edge_index = torch.cat(edge_index_batch, dim=1)
            group_y = torch.tensor(mark, dtype=int)
            group = Data(x = group_x, y = group_y, edge_attr=group_edge_attr,edge_index=group_edge_index,edge_attr_label = group_edge_attr_label)
            concatenated_tensors.append(group)
        
        
        return concatenated_tensors
        

    def train(self, dataset, gnn_module, transformer_module) -> TrxGNNGPT:#预训练的训练函数
        # data = dataset.dataset
        # random.shuffle(data)
        # accelerator = Accelerator()
        data =  self.collate_fn(self.batch_size, dataset)
        model = self.initialize_model(gnn_module, transformer_module, self.gnn_hidden_dim, \
            self.mlm_probability, self.device, self.is_tighted_lm_head, self.masked_node, self.masked_edge)
        logger.info(f"TrxGNNGPT has {count_parameters(model)} parameters")
        logger.info(f"gnn_module has {count_parameters(gnn_module)} parameters")
        logger.info(f"transformer_module has {count_parameters(transformer_module)} parameters")
        
        model.to(self.device)
        logger.info(f"model is on {model.device}")
        logger.info(f"gnn_module is on {next(gnn_module.parameters()).device}")
        logger.info(f"transformer_module is on {next(transformer_module.parameters()).device}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        logger.info("Use AdamW optimizer")
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
                model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
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
    
    def train_model_classification(self, train_dataset,test_dataset,valid_dataset, model,num_class) -> TrxGNNGPT:#下游任务分类的训练函数
        train_data = train_dataset
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        logger.info("Use AdamW optimizer")
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
        step = 0 
        global_step = self.start_step
        max_eval_acc = 0
        max_test_acc = 0
        for epoch in range(self.epochs):
            model.train()
            logger.info(f"epoch: {epoch}")
            total_loss = 0
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Initialize tqdm progress bar with an initial description
            pbar = tqdm.tqdm(train_data, desc="Initial")
            for i,batch in enumerate(pbar): #(tqdm.tqdm(data, desc=f"Processing items Available memory: {torch.cuda.get_device_properties(self.device).total_memory/ (1024**3) - torch.cuda.memory_allocated(self.device)/ (1024**3):.4f} G")):
                pbar.set_description(f"Processing items Available memory: {torch.cuda.get_device_properties(self.device).total_memory/ (1024**3) - torch.cuda.memory_allocated(self.device)/ (1024**3):.4f} G")
                step += 1
                labels,predictions = model(batch)
                loss_1 = criterion(predictions.to(torch.float32).to(self.device), labels.to(torch.float32).to(self.device))
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
            eval_acc,eval_macro_F1 = self.eval_model_classification(valid_dataset, model, epoch,num_class)
            if eval_acc > max_eval_acc:
                max_eval_acc = eval_acc
                logger.info(f"max eval acc is {max_eval_acc} and macro F1 is {eval_macro_F1} in epoch {epoch}")
                print(f"\033[32mmax eval acc is {max_eval_acc} and macro F1 is {eval_macro_F1} in epoch {epoch}\033[0m")
                test_acc,test_macro_F1  = self.eval_model_classification(test_dataset, model, epoch,num_class)
                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    logger.info(f"max test acc is {max_test_acc} and macro F1 is {test_macro_F1} in epoch {epoch}")
                    print(f"\033[31mmax test acc is {max_test_acc} and macro F1 is {test_macro_F1} in epoch {epoch}\033[0m")
        return model
    
    def eval_model_classification(self,valid_data, model, epoch,num_class):
        model.eval()
        acc = 0
        macro_F1 = 0
        with torch.no_grad():
            labels = torch.empty((len(valid_data),num_class),dtype=torch.int).to(self.device)
            predictions_threhold = torch.empty((len(valid_data),num_class),dtype=torch.float32).to(self.device)
            predictions = torch.empty((len(valid_data),num_class),dtype=torch.int).to(self.device)
            pbar = tqdm.tqdm(valid_data, desc="Initial")
            for i,batch in enumerate(pbar): #(tqdm.tqdm(data, desc=f"Processing items Available memory: {torch.cuda.get_device_properties(self.device).total_memory/ (1024**3) - torch.cuda.memory_allocated(self.device)/ (1024**3):.4f} G")):
                pbar.set_description(f"Processing items Available memory: {torch.cuda.get_device_properties(self.device).total_memory/ (1024**3) - torch.cuda.memory_allocated(self.device)/ (1024**3):.4f} G")
                label,prediction = model(batch)
                labels[i] = label.to(torch.int).to(self.device)
                predictions_threhold[i] = prediction.to(torch.float)
            predictions = torch.zeros_like(predictions_threhold).scatter_(1, torch.argmax(predictions_threhold,dim=1).unsqueeze(1), 1)
            confusion_matrix = (labels.to(torch.float).T @ predictions).T
            acc = torch.trace(confusion_matrix) / len(valid_data)
            macro_precision = sum([confusion_matrix[i][i]/confusion_matrix[i].sum() for i in range(num_class)]) / num_class
            macro_recall = sum([confusion_matrix[i][i]/confusion_matrix[i].T.sum() for i in range(num_class)]) / num_class
            macro_F1 = macro_precision * macro_recall * 2 / (macro_recall + macro_precision)
            logger.info(f"acc is {acc}, macro-F1 is {macro_F1}")
            print(f"\033[33macc is {acc}, macro-F1 is {macro_F1}\033[0m")    
        logger.info(f"Saving model at the end of epoch {epoch}, "+"{}/{}".format(self.model_args.output_dir, epoch))  
        return acc,macro_F1

    def train_model_recover(self, train_dataset,test_dataset,valid_dataset, model,preprocessor) -> TrxGNNGPT:#下游任务分类的训练函数
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        train_dataset =  self.collate_fn(self.batch_size, train_dataset)
        del train_dataset[850:890]
        test_dataset = self.collate_fn(self.batch_size, test_dataset)
        valid_dataset = self.collate_fn(self.batch_size, valid_dataset)
        logger.info("Use AdamW optimizer")
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
        step = 0 
        global_step = self.start_step
        max_eval_bleu_score = 0
        max_test_bleu_score = 0
        chars_to_remove = "[]\"\',;{\}:"
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.epochs):
            model.train()
            logger.info(f"epoch: {epoch}")
            total_loss = 0
            # eval_bleu_score = self.eval_model_recover(valid_dataset, model, epoch, preprocessor)
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Initialize tqdm progress bar with an initial description
            pbar = tqdm.tqdm(range(len(train_dataset)), desc="Initial")
            for i in pbar: #(tqdm.tqdm(data, desc=f"Processing items Available memory: {torch.cuda.get_device_properties(self.device).total_memory/ (1024**3) - torch.cuda.memory_allocated(self.device)/ (1024**3):.4f} G")):
                pbar.set_description(f"Processing items Available memory: {torch.cuda.get_device_properties(self.device).total_memory/ (1024**3) - torch.cuda.memory_allocated(self.device)/ (1024**3):.4f} G")
                step += 1
                batch = copy.deepcopy(train_dataset[i])
                _,predictions,tokenized_edge= model(batch, preprocessor, generate = 0)
                try:
                    loss_1 = criterion(predictions.reshape(-1, predictions.shape[-1]).to(self.device), tokenized_edge.view(-1).to(self.device))
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
                except Exception as e:
                    # 捕获其他潜在的异常，确保程序不会意外终止
                    print(f"{e}")
            logger.info(f"Epoch {epoch+1}/{self.epochs}, total Loss: {total_loss}")
            eval_bleu_score = self.eval_model_recover(valid_dataset, model, epoch, preprocessor)
            if eval_bleu_score > max_eval_bleu_score:
                max_eval_bleu_score = eval_bleu_score
                print(f"\033[32mmax eval bleu_score is {max_eval_bleu_score} in epoch {epoch}\033[0m")
                logger.info(f"max eval bleu_score is {max_eval_bleu_score} in epoch {epoch}")
                test_bleu_score = self.eval_model_recover(test_dataset, model, epoch, preprocessor)
                if test_bleu_score > max_test_bleu_score:
                    max_test_bleu_score = test_bleu_score
                    print(f"\033[31mmax test bleu_score is {max_test_bleu_score} in epoch {epoch}\033[0m")
                    logger.info(f"max test bleu_score is {max_test_bleu_score} in epoch {epoch}")
        return model

    def eval_model_recover(self,valid_data, model, epoch,preprocessor):
        model.eval()
        avg_bleu_score = 0
        with torch.no_grad():
            pbar = tqdm.tqdm(valid_data, desc="Initial")
            for i,batch in enumerate(pbar): #(tqdm.tqdm(data, desc=f"Processing items Available memory: {torch.cuda.get_device_properties(self.device).total_memory/ (1024**3) - torch.cuda.memory_allocated(self.device)/ (1024**3):.4f} G")):
                pbar.set_description(f"Processing items Available memory: {torch.cuda.get_device_properties(self.device).total_memory/ (1024**3) - torch.cuda.memory_allocated(self.device)/ (1024**3):.4f} G")
                labels,predictions,_ = model(batch, preprocessor, generate = 1)
                # 用tokenizer2还原预测
                predicted_token_ids = torch.max(predictions,dim=2)[1]
                predicted_token = []
                for token in predicted_token_ids:
                    labels_length = predicted_token_ids.shape[1]
                    if torch.where(token==2)[0].numel() != 0:
                        labels_length = torch.where(token==2)[0]
                    predicted_token.append(preprocessor.function_dataset.tokenizer.decode(token[:labels_length].tolist()))
                predicted_token = [re.sub('<[^>]*>', '', i) for i in predicted_token]
                predicted_token = [i.split()  for i in predicted_token]
                actual_token = [preprocessor.function_dataset.tokenizer.decode(i.tolist()) for i in labels]
                actual_token = [re.sub('<[^>]*>', '', i) for i in actual_token]
                try:
                    bleu_score = sum([sentence_bleu([actual_token[i]], predicted_token[i], smoothing_function=SmoothingFunction().method1) for i in range(len(predicted_token))]) / len(predicted_token)
                except Exception as e:
                    print(f"发生未知错误: {e}")
                avg_bleu_score += bleu_score
            print(f"\033[33mbleu_score is {avg_bleu_score / len(valid_data)}\033[0m") 
            logger.info(f"bleu_score is {avg_bleu_score / len(valid_data)}")    
        logger.info(f"Saving model at the end of epoch {epoch}, "+"{}/{}".format(self.model_args.output_dir, epoch))  
        return avg_bleu_score / len(valid_data)

    def load(self, gnn_module, transformer_module) -> TrxGNNGPT:
        # data = dataset.dataset
        # random.shuffle(data)
        # accelerator = Accelerator()
        model = self.initialize_model(gnn_module, transformer_module, self.gnn_hidden_dim, \
            self.mlm_probability, self.device, self.is_tighted_lm_head, self.masked_node, self.masked_edge)
        logger.info(f"TrxGNNGPT has {count_parameters(model)} parameters")
        logger.info(f"gnn_module has {count_parameters(gnn_module)} parameters")
        logger.info(f"transformer_module has {count_parameters(transformer_module)} parameters")
        model.to(self.device)
        logger.info(f"model is on {model.device}")
        logger.info(f"gnn_module is on {next(gnn_module.parameters()).device}")
        logger.info(f"transformer_module is on {next(transformer_module.parameters()).device}")
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
    
    
   