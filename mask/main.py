## main.py
import os
import torch
from data_preprocessor import DataPreprocessor, GraphData, ProcessedGraphData,GraphCollection
from gnn_module import GNNModule
from transformer_module import TransformerModule
from training_pipeline import TrainingPipeline
from training_pipeline import get_vocab_size
import web3
import pdb

class Main:
    def __init__(self,filepath,deg_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.filepath = filepath
        self.split_file_path = split_dict_filepath
        self.deg_path = deg_path
        self.sequence = 128
        self.hidden_dim = 64
        self.mlm_probability = 0.15
        self.data_preprocessor = DataPreprocessor(filepath,split_dict_filepath,sequence = self.sequence)
        self.gnn_module = GNNModule(input_dim=self.sequence*self.hidden_dim,deg_path = self.deg_path,
                                     output_dim=self.sequence*self.hidden_dim,edge_dim = self.sequence*self.hidden_dim)
        # self.gnn_module = GNNModule(input_dim=self.data_preprocessor.dataset[0]['node feature'].shape[1],deg_path = self.deg_path,
        #                              output_dim=64,edge_dim = self.data_preprocessor.dataset[0]['node feature'].shape[1])  # Example dimensions
        self.transformer_module = TransformerModule(input_dim=self.hidden_dim, output_dim=get_vocab_size())  # Example dimensions
        self.training_pipeline = TrainingPipeline(epochs=10, batch_size=1,hidden_dim = self.hidden_dim,mlm_probability = self.mlm_probability,device = device,vocab=get_vocab_size())  # Default values
        self.gnn_modle = self.gnn_module.to(device)
        self.transforumer_module = self.transformer_module.to(device)


    def train_model(self):
        if not os.path.exists(self.filepath):
            print("File path does not exist.")
            return None
            # Load and preprocess data
        graph_data = self.data_preprocessor.process_data()
            # Validate processed data
            # Train the model


        model = self.training_pipeline.train(graph_data, self.gnn_module, self.transformer_module)
        
        
        return model

if __name__ == "__main__":
    # filepath = "graph/raw/graph_list.npy"  # Placeholder path
    filepath = '/root/autodl-tmp/pre_train/generate/result'
    split_dict_filepath = 'graph/split_dict.pt'
    deg_path = 'graph/deg.pt'
    main = Main(filepath,deg_path)
    model = main.train_model()
    
    # 这边还要model save
    torch.save(model.embedding_model.state_dict(), 'nn_embedding_model.pth')
    torch.save(model.text_embedding_model.state_dict(), 'lstm_embedding_model.pth')
    if model is not None:
        print("Training completed.")
    else:
        print("Training failed due to errors.")
