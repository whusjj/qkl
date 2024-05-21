## main.py
import os
import torch
from utils.data_preprocessor import DataPreprocessor
from models.gnn_module import GNNModule
from models.transformer_module import TransformerModule
from utils.training_pipeline import TrainingPipeline
import pickle
import json

def get_vocab_size(token_vocab_path):
        # 读取 esperberto-vocab.json 文件
        with open(token_vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        # 获取词汇表的长度
        vocab_size = len(vocab_data)
        return vocab_size
    
class Main:
    def __init__(self,training_args: TrainingArguments, data_args: DataArguments, tokenizaer_args: TokenizerArguments, model_args: ModelArguments):
        self.train_args = training_args
        self.data_args = data_args
        self.tokenizer_args = tokenizaer_args
        self.model_args = model_args
        
        self.device = training_args.device
        self.filepath = self.data_args.filepath
        self.split_file_path = self.data_args.split_dict_filepath
        self.sequence = self.train_args.sequence
        self.hidden_dim = self.train_args.hidden_dim
        self.mlm_probability = self.train_args.mlm_probability
        self.batch_size = self.train_args.batch_size
        self.epoch = self.train_args.epoch
        self.pre_train = self.train_args.pre_train
        self.pickle_path = self.train_args.pickle_path #"/root/autodl-tmp/pre_train/generate/data.pickle"
        
        self.data_preprocessor = DataPreprocessor(  self.filepath, \
                                                    sequence = self.sequence,\
                                                    batch = self.batch_size,\
                                                    tokenizaer_args=tokenizaer_args )
        
        self.gnn_module = GNNModule(input_dim=self.sequence*self.hidden_dim, \
                                    output_dim=self.sequence*self.hidden_dim, \
                                    edge_dim = self.sequence*self.hidden_dim)
        
        self.transformer_module = TransformerModule(output_dim = self.tokenizer_args.vocab_size )  # Example dimensions
        
        self.training_pipeline = TrainingPipeline(  self.train_args, \
                                                    vocab_size = self.tokenizer_args.vocab_size )  # 


    def train_model(self):
        if not os.path.exists(self.filepath):
            print("File path does not exist.")
            return None
            # Load and preprocess data
        if not os.path.exists(self.pickle_path):
            graph_data = self.data_preprocessor.process_data()
            with open(self.pickle_path, 'wb') as handle:
                pickle.dump(graph_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.pickle_path, 'rb') as handle:
                graph_data = pickle.load(handle)
            print(1)

            # Validate processed data
            # Train the model
        model = self.training_pipeline.train(graph_data, self.gnn_module, self.transformer_module)
        return model

from transformers import HfArgumentParser
from utils.argument import TrainingArguments, DataArguments, TokenizerArguments,ModelArguments

if __name__ == "__main__":
    # filepath = "/root/autodl-tmp/pre_train/generate/result"  # Placeholder path
    # split_dict_filepath = 'graph/split_dict.pt'
    # pickle_path = "/root/autodl-tmp/pre_train/generate/data.pickle"
    # token_vocab = "esperberto-vocab.json"
    # token_merge = "esperberto-merges.txt"
    parser = HfArgumentParser((TrainingArguments, DataArguments, TokenizerArguments,ModelArguments))
    training_args, data_args, tokenizaer_args, model_args = parser.parse_args_into_dataclasses()
    training_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tokenizaer_args.token_vocab = "{}/esperberto-vocab.json".format(tokenizaer_args.tokenizer_dir)
    tokenizaer_args.token_merge = "{}/esperberto-merges.txt".format(tokenizaer_args.tokenizer_dir)
    tokenizaer_args.vocab_size = get_vocab_size(tokenizaer_args.token_vocab)
    
    model_args.gnn_model_path = "{}/gnn_model.pth".format(model_args.output_dir)
    model_args.transformer_model_path = "{}/transformer_model.pth".format(model_args.output_dir)
    model_args.emb_model_path = "{}/nn_embedding_model.pth".format(model_args.output_dir)
    
    main = Main(training_args, data_args, tokenizaer_args, model_args)
    model = main.train_model()
    # 这边还要model save
    torch.save(model.embedding_layer.state_dict(), model_args.emb_model_path)
    torch.save(model.transformer_module.transformer.state_dict(), model_args.transformer_model_path )
    torch.save(model.gnn_module.gcn.state_dict(), model_args.gnn_model_path)
    if model is not None:
        print("Training completed.")
    else:
        print("Training failed due to errors.")
