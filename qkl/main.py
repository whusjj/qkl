## main.py
import os
import torch
from data_preprocessor import DataPreprocessor, GraphData, ProcessedGraphData
from gnn_module import GNNModule
from transformer_module import TransformerModule
from training_pipeline import TrainingPipeline

class Main:
    def __init__(self,filepath,deg_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.filepath = filepath
        self.split_file_path = split_dict_filepath
        self.deg_path = deg_path
        self.data_preprocessor = DataPreprocessor(filepath,split_dict_filepath)
        self.gnn_module = GNNModule(input_dim=self.data_preprocessor.dataset[0]['node feature'].shape[1],deg_path = self.deg_path,
                                     output_dim=64,edge_dim = self.data_preprocessor.dataset[0]['node feature'].shape[1])  # Example dimensions
        self.transformer_module = TransformerModule(input_dim=32, output_dim=1)  # Example dimensions
        self.training_pipeline = TrainingPipeline(epochs=10, batch_size=1)  # Default values
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
    filepath = "graph/raw/graph_list.npy"  # Placeholder path
    split_dict_filepath = 'graph/split_dict.pt'
    deg_path = 'graph/deg.pt'
    main = Main(filepath,deg_path)
    model = main.train_model()
    if model is not None:
        print("Training completed.")
    else:
        print("Training failed due to errors.")
