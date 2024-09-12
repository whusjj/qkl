from torch.utils.data import Dataset
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torch
import json
import pdb
# def get_vocab_size():
#     # 读取 esperberto-vocab.json 文件
#     with open('esperberto-vocab.json', 'r', encoding='utf-8') as f:
#         vocab_data = json.load(f)

#     # 获取词汇表的长度
#     vocab_size = len(vocab_data)
#     return vocab_size

class EsperantoDataset(Dataset):
    def __init__(self,token_vocab_path="esperberto-vocab.json", token_merge_path="esperberto-merges.txt", max_length = 128):
        #self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = ByteLevelBPETokenizer(
            token_vocab_path,
            token_merge_path,
        )
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", self.tokenizer.token_to_id("</s>")),
            ("<s>", self.tokenizer.token_to_id("<s>")),
        )
        # pdb.set_trace()
        self.max_length = max_length
        self.tokenizer.enable_truncation(max_length=max_length)
        self.token_vocab_path = token_vocab_path
        self.token_merge_path = token_merge_path


    def tokenizer_node(self, graph_data):
        # 存储编码后数据的列表
        encoded_sequence = [torch.tensor(self.tokenizer.encode(str(seq), add_special_tokens=True).ids) for seq in graph_data]
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
        padded_tensor_list = [torch.cat((tensor, torch.ones(max_length - len(tensor), dtype=torch.long))) for tensor in encoded_sequence]
        # 对 graph_data 中的每个项进行处理
        return padded_tensor_list