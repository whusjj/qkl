#!/bin/bash
tokenizer_dir="tokenizer"
token_vocab="esperberto-vocab.json"
token_merge="esperberto-merges.txt"

bash run_gpt_flexible_dim.sh   $tokenizer_dir $token_vocab $token_merge 192 './data/raw_data' './data/preprocessed/data.pickle'
bash run_gpt_flexible_dim.sh   $tokenizer_dir $token_vocab $token_merge 384 './data/raw_data' './data/preprocessed/data.pickle'

bash run_gpt_flexible_dim.sh   $tokenizer_dir $token_vocab $token_merge 192 './data/raw_data_6w' './data/preprocessed/data_6w.pickle'
bash run_gpt_flexible_dim.sh   $tokenizer_dir $token_vocab $token_merge 384 './data/raw_data_6w' './data/preprocessed/data_6w.pickle'



# bash run_large_tokenizer.sh
# bash run_gpt_dim_144_larger.sh