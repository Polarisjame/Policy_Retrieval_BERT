#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

# -------------------GAIN_BERT_base Training Shell Script--------------------

if true; then
  batch_size=16
  reload=1
  log_step=20
  index_type=Body
  model_name=BERT_DoubleMLP_${index_type}_best

  nohup python3 -u model_prepare_title.py \
    --tokenize_model_name clue/roberta_chinese_base \
    --encode_model_name clue/roberta_chinese_base \
    --index_type  ${index_type}\
    --log_step ${log_step} \
    --checkpoint_dir ./checkpoint/ \
    --loader_save_file ./data/prepro_data/ \
    --full_dataloader_save_file ./Conver2vec/origin_data/prepro_data/ \
    --full_data_save_file ./Conver2vec/results/ \
    --batch_size ${batch_size} \
    --model_name ${model_name} \
    --model_output_dim 128 \
    --dropout 0.2 \
    --pretrain_model checkpoint/${model_name}.pt \
    >logs/word2vec_${model_name}.log 2>&1 &
fi


# -------------------additional options--------------------

# option below is used to resume training, it should be add into the shell scripts above
    # --pretrain_model checkpoint/Bert_MLP_best.pt \