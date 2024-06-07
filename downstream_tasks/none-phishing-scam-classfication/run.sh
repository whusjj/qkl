#!/bin/bash


[ -d processed/train ] || mkdir -p processed/train
[ -d processed/test ] || mkdir -p processed/test
[ -d processed/valid ] || mkdir -p processed/test

gpt_hidden_dim=64
gpt_num_head=8
tokenizer_dir="tokenizer"
token_vocab="esperberto-vocab.json"
token_merge="esperberto-merges.txt"
pickle_path='processed' #在这个文件夹下分为train，test，valid，0.64:0.2:0.16,每个类别都有自己的pickle
saved_model="saved_model"
raw_data_folder='./data'
script_name=$(basename "$0")
BASENAME_NO_EXT="${script_name%.*}"
# export TORCH_USE_CUDA_DSA=1
# CUDA_LAUNCH_BLOCKING=1 去掉掩码把那三个模块路径换掉加上预训练路径 
python main.py \
    --learning_rate 3e-5 \
    --do_train \
    --do_eval \
    --seed 42 \
    --epochs 500 \
    --batch_size 1 \
    --masked_node \
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
    --model_path '../../../saved_model_gpt_hidden_dim_64_8_masked_node_False_masked_edge_True_vocab_size_5000/epoch_0/model.pth' \
    --epoch 4 \
    --raw_data_folder $raw_data_folder \
    --pickle_path $pickle_path \
    --tokenizer_dir $tokenizer_dir \
    --token_vocab $token_vocab \
    --hidden_dropout_prob 0.5 \
    --token_merge $token_merge 2>&1 | tee  log_${BASENAME_NO_EXT}_${tokenizer_dir}.txt

    # --train_file './data/split/train.pickle' \
    # --eval_file './data/split/eval.pickle' \
    # --test_file './data/split/test.pickle' \
