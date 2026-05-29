MODEL_PATH_LIST=('YOUR_PATH/unsup-bert-base-uncased' 'YOUR_PATH/unsup-ModernBERT-base') # 'YOUR_PATH/unsup-Bio_ClinicalBERT' 'google-bert/bert-base-uncased' 'emilyalsentzer/Bio_ClinicalBERT' 'answerdotai/ModernBERT-base') # 
FREEZE_LM=false
VAL_MAX_SIZE=1000
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=64
NUM_EPOCHS=10
EVAL_EVERY_PERCENT=0.2
CACHE_DIR='/YOUR_PATH/.cache'

task_type="crf_task" # "chronicity_task" #  "tlocvsdyspnea_task"
dataset_path="YOUR_PATH/crf-second-batch-item-by-item-balanced" #"YOUR_PATH/chronicity-classification-task" # "YOUR_PATH/tloc-classification-task"

for MODEL_PATH in "${MODEL_PATH_LIST[@]}"; do 
  echo "Starting training for model: $MODEL_PATH"
  
  if [[ "$FREEZE_LM" = true ]]; then
    nohup python l_baselines_bert.py --model_path $MODEL_PATH \
        --freeze_lm \
        --val_max_size $VAL_MAX_SIZE \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --eval_every_percent $EVAL_EVERY_PERCENT \
        --cache_dir $CACHE_DIR \
        --task_type $task_type \
        --dataset_path $dataset_path \
        &> nohup_BERT_BASELINE_${MODEL_PATH##*/}_${task_type} \
        2>&1 & wait
  else
    nohup python l_baselines_bert.py --model_path $MODEL_PATH \
        --val_max_size $VAL_MAX_SIZE \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --eval_every_percent $EVAL_EVERY_PERCENT \
        --cache_dir $CACHE_DIR \
        --task_type $task_type \
        --dataset_path $dataset_path \
        &> nohup_BERT_BASELINE_${MODEL_PATH##*/}_${task_type} \
        2>&1 & wait
  fi
  echo "DONE for model: $MODEL_PATH"
done
