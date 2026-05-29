
MODEL_PATH_LIST=(
  # 'YOUR_PATH/unsup-Qwen3-1.7B-datav3-only_mask'  
  # 'YOUR_PATH/unsup-Qwen3-1.7B-datav3-only_mask_w_item' 
  # 'YOUR_PATH/unsup-Qwen3-8B-datav3-only_mask' 
  # 'YOUR_PATH/unsup-Qwen3-8B-datav3-only_mask_w_item' 
  # 'YOUR_PATH/unsup-Llama-3.2-1B-Instruct-only_mask' 
  # 'YOUR_PATH/unsup-Llama-3.2-1B-Instruct-only_mask_w_item' 
  # 'YOUR_PATH/unsup-Llama-3.1-8B-Instruct-datav2-only_mask' 
  # 'YOUR_PATH/unsup-Llama-3.1-8B-Instruct-datav2-only_mask_w_item' 
  # 'YOUR_PATH/unsup-gemma-3-1b-it-datav3-only_mask' 
  # 'YOUR_PATH/unsup-gemma-3-1b-it-datav3-only_mask_w_item' 
  # 'YOUR_PATH/unsup-gemma-3-4b-it-datav3-only_mask' 
  # 'YOUR_PATH/unsup-gemma-3-4b-it-datav3-only_mask_w_item' 
  # 'YOUR_PATH/unsup-Qwen3-1.7B-datav3-only_mask_w_item_mesh'
  # 'Qwen/Qwen3-1.7B'
  # 'YOUR_PATH/unsup-Llama-3.2-1B-Instruct-only_mask_w_item_mesh'
  # 'meta-llama/Llama-3.2-1B-Instruct'
  # 'YOUR_PATH/unsup-Llama-3.1-8B-Instruct-datav2-only_mask_w_item_mesh'
  # 'Qwen/Qwen3-1.7B'
  # 'meta-llama/Llama-3.1-8B-Instruct'
  # 'Qwen/Qwen3-8B'
  # 'meta-llama/Llama-3.2-1B-Instruct'
  # 'microsoft/Phi-3.5-mini-instruct'
  # 'YOUR_PATH/unsup-Phi-3.5-mini-instruct-only_mask_w_item'
  # 'YOUR_PATH/unsup-MediPhi-only_mask_w_item'
  # 'microsoft/MediPhi'



  # 'google/gemma-3-4b-it'
  # 'YOUR_PATH/unsup-gemma-3-4b-it-datav3-only_mask_w_item'
  # 'YOUR_PATH/unsup-Llama-3.1-8B-Instruct-datav2'
  # 'YOUR_PATH/unsup-Llama-3.1-8B-Instruct-datav2-only_mask_w_item'
  # 'YOUR_PATH/unsup-Qwen3-8B-datav3-only_mask_w_item' 
  'YOUR_PATH/unsup-Qwen3-8B-datav3-cpt'
  # 'google/gemma-3-1b-it'
  )

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path_list)
      MODEL_PATH_LIST=()
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        MODEL_PATH_LIST+=("$1")
        shift
      done
      ;;
    --help)
      echo "Usage: $0 [options]"
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

FREEZE_LM=true
VAL_MAX_SIZE=0.5
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=64
NUM_EPOCHS=20
EVAL_EVERY_PERCENT=0.5
CACHE_DIR='/workspace/.cache' # '/YOUR_PATH/.cache' # 
TYPE_OF_PROMPT='mask' # instruction' # 

datastet_path="YOUR_PATH/crf-second-batch-item-by-item-balanced" # "YOUR_PATH/mesh_class_20perc" # #
task="crf_task" # "mesh_task" # "instruct_chronic_task" # "instruct_tlocvsdyspnea_task" # "tlocvsdyspnea_task" # "qa_task" #
val_max_size=0.5
test_max_size=100000
train_max_size=1000000

USE_NON_LINEARITY=false # true # false

# read the crf_items_list from data/crf_items.txt

if [[ "$task" == "crf_task" ]]; then
    items_file="data/crf_items_all.txt"
elif [[ "$task" == "mesh_task" ]]; then
    items_file="data/mesh_for_unsup_train.txt"
else
    echo "Unsupported task: $task"
    exit 1
fi

crf_items_list=()
while IFS= read -r line; do
    crf_items_list+=("$line")
done < "$items_file"
echo "CRF items to train on: ${crf_items_list[*]}"

for MODEL_PATH in "${MODEL_PATH_LIST[@]}"; do 
  mkdir -p nohup_h_CLASSIFIER_OVER_UNSUP_${MODEL_PATH##*/}/${TYPE_OF_PROMPT}_nonlin${USE_NON_LINEARITY}
done

for MODEL_PATH in "${MODEL_PATH_LIST[@]}"; do 
  nohup_dir_path=nohup_h_CLASSIFIER_OVER_UNSUP_${MODEL_PATH##*/}

  for crf_item in "${crf_items_list[@]}"; do
    echo "Starting training for model: $MODEL_PATH, CRF item: $crf_item"
    crf_item_no_spaces=${crf_item// /}
    crf_item_no_spaces=${crf_item_no_spaces//\//____}
    crf_item=${crf_item//\//____}
    
    if [[ "$FREEZE_LM" = true ]]; then
      echo "path: ${nohup_dir_path}/${TYPE_OF_PROMPT}_nonlin${USE_NON_LINEARITY}/${crf_item_no_spaces}"
          
      nohup python h_train_class_over_unsup.py --model_path $MODEL_PATH \
          --freeze_lm \
          --train_max_size $train_max_size \
          --val_max_size $val_max_size \
          --test_max_size $test_max_size \
          --dataset_path ${datastet_path} \
          --train_batch_size $TRAIN_BATCH_SIZE \
          --eval_batch_size $EVAL_BATCH_SIZE \
          --num_epochs $NUM_EPOCHS \
          --eval_every_percent $EVAL_EVERY_PERCENT \
          --cache_dir $CACHE_DIR \
          --type_of_prompt $TYPE_OF_PROMPT \
          --use_non_linearity $USE_NON_LINEARITY \
          --use_non_linearity $USE_NON_LINEARITY \
          --task_type $task \
          --crf_item "$crf_item" \
          &> ${nohup_dir_path}/${TYPE_OF_PROMPT}_nonlin${USE_NON_LINEARITY}/${crf_item_no_spaces}  2>&1 & wait
    else
      nohup python h_train_class_over_unsup.py --model_path $MODEL_PATH \
          --val_max_size $val_max_size \
          --train_max_size $train_max_size \
          --test_max_size $test_max_size \
          --dataset_path ${datastet_path} \
          --train_batch_size $TRAIN_BATCH_SIZE \
          --eval_batch_size $EVAL_BATCH_SIZE \
          --num_epochs $NUM_EPOCHS \
          --eval_every_percent $EVAL_EVERY_PERCENT \
          --cache_dir $CACHE_DIR \
          --type_of_prompt $TYPE_OF_PROMPT \
          --use_non_linearity $USE_NON_LINEARITY \
          --use_non_linearity $USE_NON_LINEARITY \
          --task_type $task \
          --crf_item "$crf_item" \
          &> ${nohup_dir_path}/${TYPE_OF_PROMPT}_nonlin${USE_NON_LINEARITY}/${crf_item_no_spaces}  2>&1 & wait
    fi
  done
  echo "DONE for model: $MODEL_PATH"
done
