## data_preprocessor.py
try:
    import torch
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"ImportError: {e}")
    raise e

from typing import Optional

class GraphData:
    """
    A simple data structure to hold graph data.
    """
    def __init__(self,dataset,split_dict):
        self.dataset = dataset
        self.split_dict = split_dict
    
    def __len__(self):
        return len(self.split_dict['train'])
    
    def __getitem__(self,index):
        return self.dataset[index]
    
class ProcessedGraphData:
    """
    A data structure for holding processed graph data.
    """
    def __init__(self, edge_index: torch.Tensor, node_features: torch.Tensor):
        self.edge_index = edge_index
        self.node_features = node_features

class DataPreprocessor:
    """
    The DataPreprocessor class is responsible for loading and preprocessing graph data.
    """
    def __init__(self, filepath: str, split_dict_filepath: str):
        try:
            dataset = np.load(filepath, allow_pickle=True)
            split_dict = torch.load(split_dict_filepath)
            self.dataset = dataset
            self.split_dict = split_dict
        except FileNotFoundError:
            print(f"Error: The file {filepath} was not found.")
            return None

    def process_data(self) -> Optional[GraphData]:
        """
        Loads graph data from a file.

        Parameters:
        - filepath (str): The path to the file containing the graph data.

        Returns:
        - GraphData: An instance of the GraphData class containing the loaded data, or None if an error occurs.
        """
        for i,data in enumerate(self.dataset):
            self.dataset[i]['Edge indices'] = torch.tensor(data['Edge indices'], dtype=torch.long)
            self.dataset[i]['Edge attributes'] = torch.tensor(data['Edge attributes'], dtype=torch.float)
            self.dataset[i]['Node labels'] = torch.tensor(data['Node labels'], dtype=torch.float)
            self.dataset[i]['node feature'] = torch.tensor(data['node feature'], dtype=torch.float)
            data_list = [0]*(self.dataset[i]['node feature']).shape[0]
            self.dataset[i]['Edge attributes'] = self.dataset[i]['Edge attributes'].repeat(self.dataset[i]['node feature'].shape[1], 1)
            for i in range(len(data['Edge indices'][0])):
                data_list[data['Edge indices'][0][i]] += 1
                data_list[data['Edge indices'][1][i]] += 1
            self.dataset[i]['deg'] = torch.tensor(data_list) / 2

        return GraphData(self.dataset,self.split_dict)

    def preprocess(self, graph_data: GraphData) -> Optional[ProcessedGraphData]:
        """
        Preprocesses the graph data.

        Parameters:
        - graph_data (GraphData): The graph data to preprocess.

        Returns:
        - ProcessedGraphData: An instance of the ProcessedGraphData class containing the processed data, or None if an error occurs.
        """
        if graph_data is None:
            print("Error: graph_data is None, cannot preprocess.")
            return None

        node_features_mean = graph_data.node_features.mean(dim=0, keepdim=True)
        node_features_std = graph_data.node_features.std(dim=0, keepdim=True) + 1e-6  # Adding a small value to avoid division by zero
        processed_node_features = (graph_data.node_features - node_features_mean) / node_features_std

        return ProcessedGraphData(edge_index=graph_data.edge_index, node_features=processed_node_features)
