#!/bin/bash
tokenizer_dir="tokenizer"
token_vocab="vocab.json"
token_merge="merges.txt"

bash run_masked_node_edge.sh $tokenizer_dir $token_vocab $token_merge
bash run_gpt_dim_144_masked_node_edge.sh  $tokenizer_dir $token_vocab $token_merge
bash run_6w_masked_node_edge.sh  $tokenizer_dir $token_vocab $token_merge
bash run_gpt_dim_144_6w_masked_node_edge.sh  $tokenizer_dir $token_vocab $token_merge