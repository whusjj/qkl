## main.py
import os
import torch
from data_preprocessor import DataPreprocessor, GraphData, ProcessedGraphData,GraphCollection
from gnn_module import GNNModule
from transformer_module import TransformerModule
from training_pipeline import TrainingPipeline
from training_pipeline import get_vocab_size
import pdb
import pickle

class Main:
    def __init__(self,filepath):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.filepath = filepath
        self.sequence = 128
        self.hidden_dim = 64
        self.batch_size = 1
        self.epoch = 500
        self.pre_train = 0
        self.mlm_probability = 0.15
        self.pickle_path = "/root/autodl-tmp/generate/data.pickle"
        self.data_preprocessor = DataPreprocessor(filepath,sequence = self.sequence,batch = self.batch_size,pre_train = self.pre_train)
        self.gnn_module = GNNModule(input_dim=self.sequence*self.hidden_dim,pre_train = self.pre_train,num_layers=5,
                                     output_dim=self.sequence*self.hidden_dim,edge_dim = self.sequence*self.hidden_dim)
        # self.gnn_module = GNNModule(input_dim=self.data_preprocessor.dataset[0]['x'].shape[1],deg_path = self.deg_path,
        #                              output_dim=64,edge_dim = self.data_preprocessor.dataset[0]['x'].shape[1])  # Example dimensions
        self.transformer_module = TransformerModule(pre_train = self.pre_train,input_dim=self.hidden_dim, output_dim=get_vocab_size())  # Example dimensions
        self.training_pipeline = TrainingPipeline(pre_train = self.pre_train,epochs=self.epoch, batch_size=self.batch_size,hidden_dim = self.hidden_dim,mlm_probability = self.mlm_probability,device = device,vocab=get_vocab_size())  # Default values
        # self.gnn_modle = self.gnn_module.to(device)
        # self.transformer_module = self.transformer_module.to(device)


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

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    filepath = "/root/autodl-tmp/generate/downstream"  # Placeholder path
    # filepath = '/root/autodl-tmp/pre_train/generate/result'
    main = Main(filepath)
    model = main.train_model()
    # 这边还要model save
    if model is not None:
        print("Training completed.")
    else:
        print("Training failed due to errors.")
