#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

# -------------------GAIN_BERT_base Training Shell Script--------------------

if true; then
  batch_size=16
  margin=1
  model_name=BERT_DoubleMLP_Body_best

  nohup python3 -u inference.py \
    --tokenize_model_name clue/roberta_chinese_base \
    --encode_model_name clue/roberta_chinese_base \
    --checkpoint_dir ./checkpoint/ \
    --loader_save_file ./data/prepro_data/ \
    --batch_size ${batch_size} \
    --margin ${margin} \
    --model_name ${model_name} \
    --use_bert_finetune  \
    --model_output_dim 128 \
    --dropout 0.2 \
    --lr_cosine_warm \
    --pretrain_model checkpoint/BERT_DoubleMLP_Body_best.pt \
    >logs/test_${model_name}.log 2>&1 &
fi


# -------------------additional options--------------------

# option below is used to resume training, it should be add into the shell scripts above
    # --pretrain_model checkpoint/Bert_MLP_best.pt \