from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingArguments:
    learning_rate: float = 3e-5
    do_train: bool = False
    do_eval: bool = False
    seed: int = 42
    epochs: int  = 100
    batch_size: int   =  32
    fp16: bool   = True
    device: str   = 'cpu' #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_step: int   = 100
    gradient_accumulation_steps: int   =  10
    max_grad_norm: int   = 5
    logging_steps: int   = 1000
    save_steps: int   = 1000
    gnn_hidden_dim: int   = 64
    gpt_hidden_dim: int = 64
    gpt_num_head: int = 12
    mlm_probability: float   = 0.15
    sequence: int = 128
    epoch: int = 5
    is_tighted_lm_head: bool  = True
    masked_node: bool  = False
    masked_edge: bool  = True
    debug: bool = False
    
@dataclass
class ModelArguments:
    output_dir: str   = './saved_model'
    gnn_model_path: str   = 'gnn_model.pth'
    transformer_model_path: str   = 'transformer_model.pth' 
    emb_model_path: str   = 'nn_embedding_model.pth'
    

@dataclass
class DataArguments:
    raw_data_folder: str   = "./data/raw_data"  # Placeholder path
    pickle_path: str   = "./data/preprocessed/data.pickle"
    train_file: str = './data/split/train.pickle'
    eval_file: str = './data/split/eval.pickle'
    test_file: str = './data/split/test.pickle'
    
    

@dataclass
class TokenizerArguments:    
    tokenizer_dir: str = "tokenizer"
    token_vocab: str   = "esperberto-vocab.json"
    token_merge: str   = "esperberto-merges.txt"
    vocab_size: str   = None
    #"<s>":0,"<pad>":1,"</s>":2,"<unk>":3,"<mask>":4
    S_TOKEN_ID: int = 0
    PAD_TOKEN_ID: int = 1
    E_TOKEN_ID: int = 2
    UNK_TOKEN_ID: int = 3
    MASK_TOKEN_ID: int = 4
    

    
    