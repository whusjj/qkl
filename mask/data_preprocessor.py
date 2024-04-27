## data_preprocessor.py
try:
    import torch
    import pandas as pd
    import numpy as np
    import pdb
    import os
    import pickle
    # from torch_geometric.data import Data
    from MyTokenizer import EsperantoDataset
except ImportError as e:
    print(f"ImportError: {e}")
    raise e

from typing import Optional

class GraphCollection:
    def __init__(self):
        self.graphs = []

    def add_graph(self, graph):
        self.graphs.append(graph)

    def __getitem__(self, idx):
        return self.graphs[idx]

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
    def __init__(self, filepath: str, split_dict_filepath: str,sequence = 128):
        # try:
        # dataset = np.load(filepath, allow_pickle=True)
        directory_path = '/root/autodl-tmp/pre_train/generate/result'
        dataset = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "rb") as f:
                graph = pickle.load(f)
            from torch_geometric.data import Data
            data = Data(x = graph['x'],edge_attr=graph['edge_attr'],y = graph['y'],edge_index = torch.tensor(graph['edge_index'],dtype=torch.long))
            dataset.append(data)
        #with open(filepath, "rb") as f:
        #    pdb.set_trace()
        #    dataset = pickle.load(f)
        split_dict = torch.load(split_dict_filepath)
        self.dataset = dataset
        self.split_dict = split_dict
        self.esperanto_dataset = EsperantoDataset(max_length = sequence)
        # except FileNotFoundError:
        #     print(f"Error: The file {filepath} was not found.")
        #     return None

    def process_data(self) -> Optional[GraphData]:
        """
        Loads graph data from a file.

        Parameters:
        - filepath (str): The path to the file containing the graph data.

        Returns:
        - GraphData: An instance of the GraphData class containing the loaded data, or None if an error occurs.
        """
        for i,data in enumerate(self.dataset):
            tensor_list = self.esperanto_dataset.tokenizer_node(data["x"])
            stacked_tensor = torch.stack(tensor_list)
            self.dataset[i]["node feature"] = stacked_tensor
            self.dataset[i]['Edge indices'] = torch.tensor(data['edge_index'], dtype=torch.long).t()#可以保留
            tensor_list = self.esperanto_dataset.tokenizer_node(data["edge_attr"])
            stacked_tensor = torch.stack(tensor_list)
            self.dataset[i]["Edge attributes"] = stacked_tensor
            if data['y']: 
                self.dataset[i]['Node labels'] = torch.tensor(data['y'], dtype=torch.float).repeat(10,1)
            # for i, data in enumerate(self.dataset):
            #     for item in self.dataset[0].keys():
            #         if isinstance(self.dataset[i][item], torch.Tensor):
            #             print(item)
            #             print(self.dataset[i][item].device)

            # for item in self.dataset[0].keys():
            #     print(self.dataset[i][item].device)
            # data_list = [0]*(self.dataset[i]['x']).shape[0]
            # for i in range(len(data['edge_index'][0])):
            #     data_list[data['edge_index'][0][i]] += 1
            #     data_list[data['edge_index'][1][i]] += 1
            # self.dataset[i]['deg'] = torch.tensor(data_list) / 2

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

