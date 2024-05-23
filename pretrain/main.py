## main.py
import os
import torch
from utils.data_preprocessor import DataPreprocessor
from models.gnn_module import GNNModule
from models.transformer_module import TransformerModule
from utils.training_pipeline import TrainingPipeline
import pickle
import json
from transformers import HfArgumentParser
from utils.argument import TrainingArguments, DataArguments, TokenizerArguments,ModelArguments
from loguru import logger

import torch
import numpy as np
import random
# import deepspeed



logger = logger.bind(name="main")
logger = logger.opt(colors=True)
def get_vocab_size(token_vocab_path):
        # 读取 esperberto-vocab.json 文件
        with open(token_vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        # 获取词汇表的长度
        vocab_size = len(vocab_data)
        return vocab_size
    
class Main:
    def __init__(self,training_args: TrainingArguments, data_args: DataArguments, tokenizaer_args: TokenizerArguments, model_args: ModelArguments):
        self.training_args = training_args
        self.data_args = data_args
        self.tokenizer_args = tokenizaer_args
        self.model_args = model_args
        
        self.device = training_args.device
        self.sequence = self.training_args.sequence
        self.gnn_hidden_dim = self.training_args.gnn_hidden_dim
        self.mlm_probability = self.training_args.mlm_probability
        self.batch_size = self.training_args.batch_size
        # self.epoch = self.training_args.epoch
        self.debug = self.training_args.debug
        
        self.pickle_path = self.data_args.pickle_path #"/root/autodl-tmp/pre_train/generate/data.pickle"
        self.raw_data_folder = self.data_args.raw_data_folder
        
        self.data_preprocessor = DataPreprocessor(  self.raw_data_folder, \
                                                    sequence = self.sequence,\
                                                    batch = self.batch_size,\
                                                    tokenizaer_args=tokenizaer_args, debug=self.debug )
        
        self.gnn_module = GNNModule(input_dim=self.sequence*self.gnn_hidden_dim, \
                                    output_dim=self.sequence*self.gnn_hidden_dim, \
                                    edge_dim = self.sequence*self.gnn_hidden_dim)
        
        self.transformer_module = TransformerModule(gpt_hidden_dim=training_args.gpt_hidden_dim, n_head=training_args.gpt_num_head)  # Example dimensions
        
        self.training_pipeline = TrainingPipeline(  self.training_args, self.model_args, self.tokenizer_args )  # 


    def train_model(self):
        if not os.path.exists(self.raw_data_folder):
            logger.warning("File path does not exist.")
            return None
            # Load and preprocess data
        if self.debug:
            self.pickle_path = f"{self.data_args.pickle_path}_debug"
        if not os.path.exists(self.pickle_path):
            graph_data = self.data_preprocessor.process_data()
            with open(self.pickle_path, 'wb') as handle:
                pickle.dump(graph_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.pickle_path, 'rb') as handle:
                graph_data = pickle.load(handle)
            # print(1)

            # Validate processed data
            # Train the model
        graph_data.shuffle()
        model = self.training_pipeline.train(graph_data, self.gnn_module, self.transformer_module)
        return model

import sys

# def setup_logger():
#     logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
#     logger.add("file.log", format="{time} {level} {message}", level="ERROR")

#     def catch(*args):
#         logger.exception("An uncaught exception occurred:")

#     sys.excepthook = catch

if __name__ == "__main__":
    # filepath = "/root/autodl-tmp/pre_train/generate/result"  # Placeholder path
    # split_dict_filepath = 'graph/split_dict.pt'
    # pickle_path = "/root/autodl-tmp/pre_train/generate/data.pickle"
    # token_vocab = "esperberto-vocab.json"
    # token_merge = "esperberto-merges.txt"
    parser = HfArgumentParser((TrainingArguments, DataArguments, TokenizerArguments,ModelArguments))
    #parser = deepspeed.add_config_arguments(parser)
    training_args, data_args, tokenizaer_args, model_args = parser.parse_args_into_dataclasses()
    #os.makedirs("logs", exist_ok=True)
    tokenizaer_args.token_vocab = "{}/{}".format(tokenizaer_args.tokenizer_dir, tokenizaer_args.token_vocab)
    tokenizaer_args.token_merge = "{}/{}".format(tokenizaer_args.tokenizer_dir, tokenizaer_args.token_merge)
    
    model_args.output_dir = f"{model_args.output_dir}_vocab_size_{get_vocab_size(tokenizaer_args.token_vocab)}_{training_args.gpt_hidden_dim}_{training_args.gpt_num_head}_{training_args.gnn_hidden_dim}_{training_args.mlm_probability}_{training_args.batch_size}_{training_args.epochs}_{training_args.learning_rate}_mask_node_{training_args.masked_node}_mask_edge_{training_args.masked_edge}_{training_args.is_tighted_lm_head}_{training_args.debug}"
    data_args.pickle_path = f"{data_args.pickle_path}_vocab_size_{get_vocab_size(tokenizaer_args.token_vocab)}"
    os.makedirs(model_args.output_dir, exist_ok=True)
    logger.add(f"{model_args.output_dir}/pretrain.log")
    #setup_logger()
    logger.info("Training arguments: {}", training_args)
    logger.info("Data arguments: {}", data_args)
    logger.info("Tokenizer arguments: {}", tokenizaer_args)
    logger.info("Model arguments: {}", model_args)
    # Set seeds to ensure reproducibility
    logger.info(f"Setting seeds for reproducibility Seed {training_args.seed}")
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)

    # # CUDA settings for reproducibility, be aware this can degrade performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    training_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}", training_args.device)
    tokenizaer_args.vocab_size = get_vocab_size(tokenizaer_args.token_vocab)
    
    # model_args.gnn_model_path = "{}/gnn_model.pth".format(model_args.output_dir)
    # model_args.transformer_model_path = "{}/transformer_model.pth".format(model_args.output_dir)
    # model_args.emb_model_path = "{}/nn_embedding_model.pth".format(model_args.output_dir)
    
    if os.path.exists(model_args.output_dir):
        os.makedirs(model_args.output_dir, exist_ok=True)
        
    main = Main(training_args, data_args, tokenizaer_args, model_args)
    model = main.train_model()
    # # 这边还要model save
    # torch.save(model.embedding_layer.state_dict(), model_args.emb_model_path)
    # torch.save(model.transformer_module.transformer.state_dict(), model_args.transformer_model_path )
    # torch.save(model.gnn_module.gcn.state_dict(), model_args.gnn_model_path)
    if model is not None:
        print("Training completed.")
    else:
        print("Training failed due to errors.")
