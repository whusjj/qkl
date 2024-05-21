from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingArguments:
    learning_rate: float = 3e-5
    train_file: str = None
    eval_file: str = None
    do_train: bool = False
    do_eval: bool = False
    seed: int = 42
    epochs: int  = 100
    batch_size: int   =  32
    fp16: int   = True
    device: int   = 'cpu' #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_step: int   = 100
    gradient_accumulation_steps: int   =  10
    max_grad_norm: int   = 5
    logging_steps: int   = 1000
    save_steps: int   = 1000
    hidden_dim: int   = 64
    mlm_probability: int   = 0.15
    input_dim: int = 768
    sequence = 128
    epoch = 5
    pre_train = 1
    is_tighted_lm_head = True
    
@dataclass
class ModelArguments:
    output_dir: int   = './saved_model'
    gnn_model_path = '/root/autodl-tmp/pre_train/generate/gnn_model.pth'
    transformer_model_path = '/root/autodl-tmp/pre_train/generate/transformer_model.pth' 
    emb_model_path = '/root/autodl-tmp/ETHGPT-main/large-scale-regression/tokengt/pretrain/nn_embedding_model.pth'
    

@dataclass
class DataArguments:
    filepath = "/root/autodl-tmp/pre_train/generate/result"  # Placeholder path
    split_dict_filepath = 'graph/split_dict.pt'
    pickle_path = "/root/autodl-tmp/pre_train/generate/data.pickle"
    
    

@dataclass
class TokenizerArguments:    
    tokenizer_dir: str = "tokenizer"
    token_vocab = "esperberto-vocab.json"
    token_merge = "esperberto-merges.txt"
    vocab_size = None
    #"<s>":0,"<pad>":1,"</s>":2,"<unk>":3,"<mask>":4
    S_TOKEN_ID = 0
    PAD_TOKEN_ID = 1
    E_TOKEN_ID = 2
    UNK_TOKEN_ID = 3
    MASK_TOKEN_ID = 4
    

    
    