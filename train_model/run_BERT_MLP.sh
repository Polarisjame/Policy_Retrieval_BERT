#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

# -------------------Policy_BERT Training Shell Script--------------------

if true; then
  lr=2e-3
  batch_size=16
  epochs=50
  reload=1
  log_step=20
  save_round=100
  margin=1
  weight_decay=1e-4
  index_type=Body
  model_name=BERT_DoubleMLP_${index_type}

  nohup python3 -u train.py \
    --tokenize_model_name clue/roberta_chinese_base \
    --encode_model_name clue/roberta_chinese_base \
    --index_type  ${index_type}\
    --lr ${lr} \
    --log_step ${log_step} \
    --weight_decay ${weight_decay} \
    --checkpoint_dir ./checkpoint/ \
    --loader_save_file ./data/prepro_data/ \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --margin ${margin} \
    --model_name ${model_name} \
    --save_round ${save_round} \
    --use_bert_finetune  \
    --model_output_dim 128 \
    --dropout 0.4 \
    --lr_cosine_warm \
    >logs/train_${model_name}.log 2>&1 &
fi


# -------------------additional options--------------------

# option below is used to resume training, it should be add into the shell scripts above
    # --pretrain_model checkpoint/BERT_DoubleMLP_epoch9.pt \