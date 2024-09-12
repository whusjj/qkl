#!/bin/bash

[ -d processed/train ] || mkdir -p processed/train
[ -d processed/test ] || mkdir -p processed/test
[ -d processed/valid ] || mkdir -p processed/valid

gpt_hidden_dim=384
gpt_num_head=12
tokenizer_dir="tokenizer"
token_vocab="esperberto-vocab.json"
token_merge="esperberto-merges.txt"
raw_data_folder='./data'
pickle_path='processed'  #./data/preprocessed/data_larger_tokenizaer_6w.pickle
saved_model="saved_model_node_edge_144_64"
script_name=$(basename "$0")
BASENAME_NO_EXT="${script_name%.*}"
# [ -d $saved_model ] || mkdir $saved_model


python main.py \
    --learning_rate 3e-5 \
    --do_train \
    --do_eval \
    --seed 42 \
    --epochs 500 \
    --batch_size 1 \
    --masked_node \
    --num_class 2 \
    --device 'cuda:0' \
    --start_step 100 \
    --gradient_accumulation_steps 10 \
    --max_grad_norm 5 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --gnn_hidden_dim 64 \
    --gpt_hidden_dim $gpt_hidden_dim \
    --gpt_num_head $gpt_num_head \
    --mlm_probability 0.15 \
    --sequence 128 \
    --is_tighted_lm_head \
    --output_dir $saved_model \
    --gnn_model_path 'gnn_model.pth' \
    --transformer_model_path 'transformer_model.pth' \
    --emb_model_path 'nn_embedding_model.pth' \
    --model_path '../../../saved_model_6w_gnn_8_dim_128_vocab_size_5000_128_384_12_64_0.15_32_1_3e-05_mask_node_True_mask_edge_True_True_False/epoch_0/model.pth' \
    --raw_data_folder $raw_data_folder \
    --pickle_path $pickle_path \
    --tokenizer_dir $tokenizer_dir \
    --token_vocab $token_vocab \
     --hidden_dropout_prob 0.5 \
    --token_merge $token_merge 2>&1 | tee  log_${BASENAME_NO_EXT}_${tokenizer_dir}.txt

    # --train_file './data/split/train.pickle' \
    # --eval_file './data/split/eval.pickle' \
    # --test_file './data/split/test.pickle' \

