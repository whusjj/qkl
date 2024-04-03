import torch
import numpy as np
from torch_geometric.data import Data


class GraphPyG:
    def __init__(self, node_dict):
        self.node_feature = node_dict.get("node feature", None)
        self.edge_indices = np.array(node_dict.get("Edge indices", None))
        self.edge_attributes = node_dict.get("Edge attributes", None)
        self.node_labels = np.array(node_dict.get("Node labels", None))
        self.tag = node_dict.get("tag")
        self.node_feature_embedding = []    
        self.edge_attributes_embedding = []
        self.global_context = None
        self.data = self._create_pyg_data()

    def _create_pyg_data(self):
        x = self.node_feature
        edge_index = self.edge_indices
        edge_attr = self.edge_attributes
        node_labels = self.node_labels
        y = self.tag
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_labels = node_labels, y=y)

    def add_node(self, feature, label):
        if self.node_feature is None:
            self.node_feature = [feature]
        else:
            self.node_feature.append(feature)
        self.node_labels = np.append(self.node_labels, label)
        self.data = self._create_pyg_data()

    def add_edge(self, indices, attributes):
        try:
            if self.edge_indices == None:
                self.edge_indices = np.array(indices)
            else:
                # print("*************************************")
                # print(np.array(indices))
                self.edge_indices = np.vstack([self.edge_indices, np.array(indices)])
        except ValueError:
            if self.edge_indices is None:
                self.edge_indices = np.array(indices)
            else:
                # print("*************************************")
                # print(np.array(indices))
                self.edge_indices = np.vstack([self.edge_indices, np.array(indices)])
            # 如果出现 ValueError 错误，且 self.edge_indices 为 None，则执行相应操

        if self.edge_attributes is None:
            self.edge_attributes = [attributes]
        else:
            self.edge_attributes.append(attributes)
        self.data = self._create_pyg_data()
    
    def get_pyg(self):
        return self.data
        
    def print_pyg(self):
        print("Node features:\n", self.data.x)
        print("Edge indices:\n", self.data.edge_index)
        print("Edge attributes:\n", self.data.edge_attr)
        print("Node labels:\n", self.data.node_labels)
        print("tag:\n", self.data.y)
        print("Node length:\n", len(self.data.node_labels))
        print("global context is\n:", self.global_context)
        if np.any(self.edge_indices):
            print("Edge length:\n", len(self.edge_indices))

def init_graph(label):
    node_dict = {
        "node feature": [],
        # "Edge indices": np.array([[]]),
        # "Edge attributes": [],
        "Node labels": np.array([]),
        "tag": label
    }

    graph = GraphPyG(node_dict)
    graph.print_pyg()
    return graph

# graph = init_graph(0)
# graph.global_context = "woshidashabi"
# graph.print_pyg()
# indice = [[0, 1]]
# attributes = ["sfjklds"]
# graph.add_edge(indice,attributes)
# graph.print_pyg()
# indice = [[0, 34354]]
# attributes = ["dfsdfsfd"]
# graph.add_edge(indice,attributes)
# graph.print_pyg()
# indice = [[0, 34354]]
# attributes = ["dfsdfsfd"]
# graph.add_edge(indice,attributes)
# graph.print_pyg()
# file = "../data/ethereum9月主交易-3.csv"
# flag = 0
# if "疑似" in file:
#     flag = 1
#
# print(flag)
#
# if "主" in file:
#     flag = 1
# print(flag)