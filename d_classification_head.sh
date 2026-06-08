export PYTHONPATH="$PYTHONPATH:$PWD"


MODEL_PATH_LIST=(
  'ferrazzipietro/DecSelfMask-Llama-3.2-1B-Instruct'
  )

FREEZE_LM=true
VAL_MAX_SIZE=0.5
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=64
NUM_EPOCHS=1
EVAL_EVERY_PERCENT=0.5
CACHE_DIR="/data01/pferrazzi/.cache" 

TYPE_OF_PROMPT='mask'

datastet_path="ferrazzipietro/crf-second-batch-item-by-item-balanced" 
target_col_name='crf_item'
label_col_name='label'
val_max_size=32
test_max_size=32
train_max_size=32

USE_NON_LINEARITY=false

items_file="data/targets_for_self_masking.txt"

items_list=()
while IFS= read -r line; do
    items_list+=("$line")
done < "$items_file"
echo "Items to train on: ${items_list[*]}"

for MODEL_PATH in "${MODEL_PATH_LIST[@]}"; do 
  mkdir -p nohup_h_CLASSIFIER_OVER_DecSelfMask_${MODEL_PATH##*/}/${TYPE_OF_PROMPT}_nonlin${USE_NON_LINEARITY}
done

echo "----- Starting training -------"
# for MODEL_PATH in "${MODEL_PATH_LIST[@]}"; do 
#   nohup_dir_path=nohup_h_CLASSIFIER_OVER_DecSelfMask_${MODEL_PATH##*/}

#   for item in "${items_list[@]}"; do
#     echo "Starting training for model: $MODEL_PATH, Item: $item"
#     item_no_spaces=${item// /}
#     item_no_spaces=${item_no_spaces//\//____}
#     item=${item//\//____}
    
#     echo "path: ${nohup_dir_path}/${TYPE_OF_PROMPT}_nonlin${USE_NON_LINEARITY}/${item_no_spaces}"
        
#     nohup python scripts/train_classification_head.py --model_path $MODEL_PATH \
#       --freeze_lm $FREEZE_LM \
#       --train_max_size $train_max_size \
#       --val_max_size $val_max_size \
#       --test_max_size $test_max_size \
#       --dataset_path ${datastet_path} \
#       --target_col_name $target_col_name \
#       --label_col_name $label_col_name \
#       --train_batch_size $TRAIN_BATCH_SIZE \
#       --eval_batch_size $EVAL_BATCH_SIZE \
#       --num_epochs $NUM_EPOCHS \
#       --eval_every_percent $EVAL_EVERY_PERCENT \
#       --cache_dir $CACHE_DIR \
#       --type_of_prompt $TYPE_OF_PROMPT \
#       --use_non_linearity $USE_NON_LINEARITY \
#       --item "$item" \
#       &> ${nohup_dir_path}/${TYPE_OF_PROMPT}_nonlin${USE_NON_LINEARITY}/${item_no_spaces}  2>&1 & wait
#   done
#   echo "DONE for model: $MODEL_PATH"
# done


datastet_path="ferrazzipietro/crf-second-batch-item-by-item-balanced"
USE_SAME_SUBSET_OF_SFT=true
MAX_VALIDATION_EXAMPLES=-1
BATCH_SIZE=256
cache_dir="/data01/pferrazzi/.cache" 
split_name="test" 
item_col_name="crf_item"
label_col_name="label"

echo "----- Starting evaluation -------"
for index in "${!MODEL_PATH_LIST[@]}"; do
  MODEL_NAME=$(basename "${MODEL_PATH_LIST[index]}")
  model_root="data/d_classification_head/classifier_over_DecSelfMask/one_head_per_item/${MODEL_NAME}"
  # get the model name without the path
  echo "Processing model root: $model_root"
  echo "Looking for models in: ${model_root}/item_*/freeze_lm_true/epochs_${NUM_EPOCHS}/"
  # for MODEL_PATH in "${model_root}"/item_*/freeze_lm_true/epochs_${NUM_EPOCHS}/; do
  #   if [[ ! -d "$MODEL_PATH" ]]; then
  #     echo "Directory $MODEL_PATH does not exist, skipping."
  #     continue
  #   fi
  #   echo "Evaluating model: $MODEL_PATH"
  #   # the item name is the name of the folder that is 4 levels up from the model path, removing the "item_" prefix and replacing "____" with "/"
  #   item_name=$(basename "$(dirname "$(dirname "$MODEL_PATH")")")
  #   item_name=${item_name#item_}
  #   item_name=${item_name//____/\/}
  #   nohup python scripts/train_classification_head_eval.py \
  #     --dataset_path "$datastet_path" \
  #     --item_col_name "$item_col_name" \
  #     --label_col_name "$label_col_name" \
  #     --model_path "$MODEL_PATH" \
  #     --USE_SAME_SUBSET_OF_SFT $USE_SAME_SUBSET_OF_SFT \
  #     --batch_size $BATCH_SIZE \
  #     --cache_dir $cache_dir \
  #     --split_name $split_name \
  #     --MAX_VALIDATION_EXAMPLES $MAX_VALIDATION_EXAMPLES  &> nohup_d_eval_"${MODEL_NAME}_${item_name}_${split_name}" 2>&1 & wait
  # done
  echo "------ DONE for model: $MODEL_NAME ------"
  nohup python scripts/train_classification_head_aggregate_results.py \
    --model_name "${MODEL_NAME}" \
    --split_name $split_name 2>&1 & wait 
done

