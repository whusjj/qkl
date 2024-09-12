from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingArguments:
    learning_rate: float = 3e-5
    do_train: bool = False
    do_eval: bool = False
    seed: int = 42
    epochs: int  = 5
    batch_size: int   =  1
    fp16: bool   = False
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
    is_tighted_lm_head: bool  = False
    masked_node: bool  = False
    masked_edge: bool  = False
    debug: bool = False
    hidden_dropout_prob :float = 0.5
    num_class : int = 3
    
@dataclass
class ModelArguments:
    output_dir: str   = './saved_model'
    gnn_model_path: str   = 'gnn_model.pth'
    transformer_model_path: str   = 'transformer_model.pth' 
    emb_model_path: str   = 'nn_embedding_model.pth'
    model_path: str  = '../../../saved_model_gpt_hidden_dim_64_8_masked_node_False_masked_edge_True_vocab_size_5000/epoch_0/model.pth'
    

@dataclass
class DataArguments:
    raw_data_folder: str   = "./data/raw_data"  # Placeholder path
    pickle_path: str   = "./data/preprocessed/data.pickle"
    train_file: str = './data/split/train.pickle'
    eval_file: str = './data/split/eval.pickle'
    test_file: str = './data/split/test.pickle'
    
    

@dataclass
class TokenizerArguments:    
    function_tokenizer_dir: str = "tokenizer_function"
    function_token_vocab: str   = "esperberto-vocab.json"
    function_token_merge: str   = "esperberto-merges.txt"
    function_vocab_size = None
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
    

    
    