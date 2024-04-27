from torch.utils.data import Dataset
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torch
import pdb
class EsperantoDataset(Dataset):
    def __init__(self, evaluate: bool = False, batch_size=32,max_length = 128):
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
        self.max_length = max_length
        self.tokenizer.enable_truncation(max_length=max_length)


    def tokenizer_node(self, graph_data):
        # 存储编码后数据的列表
        encoded_sequence = [torch.tensor(self.tokenizer.encode(str(seq), add_special_tokens=True).ids).to(self.device) for seq in graph_data]
        max_length = self.max_length
        padded_tensor_list = [
            torch.nn.functional.pad(
            tensor[:max_length], 
            pad=(0, max_length - len(tensor)),
            mode='constant',
            value=1
        ).to(self.device)
        for tensor in encoded_sequence
        ]
        padded_tensor_list = [torch.cat((tensor.to(self.device), torch.ones(max_length - len(tensor), dtype=torch.long).to(self.device))) for tensor in encoded_sequence]
        # 对 graph_data 中的每个项进行处理
        return padded_tensor_list