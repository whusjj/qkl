#!/bin/bash
tokenizer_dir="tokenizer"
token_vocab="esperberto-vocab.json"
token_merge="esperberto-merges.txt"

bash run.sh   $tokenizer_dir $token_vocab $token_merge
bash run_gpt_dim_144.sh   $tokenizer_dir $token_vocab $token_merge
bash run_6w.sh   $tokenizer_dir $token_vocab $token_merge
bash run_gpt_dim_144_6w.sh   $tokenizer_dir $token_vocab $token_merge
# bash run_large_tokenizer.sh
# bash run_gpt_dim_144_larger.sh