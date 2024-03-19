## training_pipeline.py
import torch
from torch_geometric.data import DataLoader
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
    def __init__(self, gnn_module: GNNModule, transformer_module: TransformerModule):
        super(CustomModel, self).__init__()
        self.gnn_module = gnn_module
        self.transformer_module = transformer_module
        # self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        # self.text_embedding_model = AutoModel.from_pretrained(text_model_name)

    def forward(self, graph_data):
        # inputs = self.tokenizer(graph_data['node_text'], return_tensors='pt', padding=True, truncation=True)
        # text_embeddings = self.text_embedding_model(**inputs).last_hidden_state[:, 0, :]
        embeddings = self.gnn_module.forward(graph_data)
        text_summary = self.transformer_module(embeddings)
        return text_summary

class TrainingPipeline:
    def __init__(self, epochs: int = 100, batch_size: int = 32):
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

    def train(self, dataset,gnn_module, transformer_module) -> CustomModel:
        
        data_loader = dataset
        # accelerator = Accelerator()
        model = self.initialize_model(gnn_module,transformer_module)
        model.to(self.device)
        data = DataLoader(dataset,shuffle=True,batch_size=1)
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
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer)
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
            for i,batch in tqdm.tqdm(enumerate(dataset)):
                step += 1
                for key, tensor in batch.items():
                    if torch.is_tensor(tensor):
                        batch[key] = tensor.to(device)
                # pdb.set_trace()
                number,predictions = model(batch)
                shift_labels = number[..., 1:].contiguous()
                shift_logits = predictions[..., :-1,].contiguous()
                # Assuming the dataset provides a method to get the target labels for each batch
                # target_labels = dataset.dataset[i]['Node labels'].to(device).squeeze(1)
                loss_1 = criterion(shift_logits,shift_labels)
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
                        tr_nb=global_step
                    #if self.save_steps > 0 and global_step % self.save_steps == 0:
                total_loss += a_loss                
            print(f"Epoch {epoch+1}/{self.epochs}, total Loss: {total_loss}")
        return model

    def initialize_model(self,gnn_module,transformer_module) -> CustomModel:
        model = CustomModel(gnn_module, transformer_module)
        return model

# Note: It's assumed that the dataset class provides a method `get_labels(batch)` to access the target labels.
# This should be implemented in the dataset class to ensure compatibility with this training pipeline.
