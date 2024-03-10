## training_pipeline.py
import torch
from torch_geometric.data import DataLoader
from data_preprocessor import DataPreprocessor
from gnn_module import GNNModule
from transformer_module import TransformerModule
import tqdm
class CustomModel(torch.nn.Module):
    def __init__(self, gnn_module: GNNModule, transformer_module: TransformerModule):
        super(CustomModel, self).__init__()
        self.gnn_module = gnn_module
        self.transformer_module = transformer_module

    def forward(self, graph_data):
        embeddings = self.gnn_module.forward(graph_data)
        text_summary = self.transformer_module(embeddings)
        return text_summary

class TrainingPipeline:
    def __init__(self, epochs: int = 100, batch_size: int = 32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, dataset,gnn_module, transformer_module) -> CustomModel:
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        model = self.initialize_model(gnn_module,transformer_module)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for epoch in range(self.epochs):
            print(epoch)
            total_loss = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for i,batch in tqdm.tqdm(enumerate(data_loader)):
                for key, tensor in batch.items():
                    if torch.is_tensor(tensor):
                        batch[key] = tensor.to(device)
                optimizer.zero_grad()
                predictions = model(batch)
                # Assuming the dataset provides a method to get the target labels for each batch
                target_labels = dataset.dataset[i]['Node labels'].to(device)
                loss = abs(predictions-target_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss}")
        return model

    def initialize_model(self,gnn_module,transformer_module) -> CustomModel:
        model = CustomModel(gnn_module, transformer_module)
        return model

# Note: It's assumed that the dataset class provides a method `get_labels(batch)` to access the target labels.
# This should be implemented in the dataset class to ensure compatibility with this training pipeline.
