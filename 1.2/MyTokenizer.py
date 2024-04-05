from torch.utils.data import Dataset
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torch

class EsperantoDataset(Dataset):
    def __init__(self, evaluate: bool = False, batch_size=32):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = ByteLevelBPETokenizer(
            "esperberto-vocab.json",
            "esperberto-merges.txt",
        )
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", self.tokenizer.token_to_id("</s>")),
            ("<s>", self.tokenizer.token_to_id("<s>")),
        )
        self.tokenizer.enable_truncation(max_length=512)


    def tokenizer_node(self, graph_data):
        # 存储编码后数据的列表
        encoded_data = []

        # 对 graph_data 中的每个项进行处理
        for item in graph_data:
            encoded_batches = self.tokenizer.encode_batch([str(item)])
            for encoded_batch in encoded_batches:
                encoded_ids_tensor = torch.tensor(encoded_batch.ids)
            encoded_data.append(encoded_ids_tensor.to(self.device))

        return encoded_data