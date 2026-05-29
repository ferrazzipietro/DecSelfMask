
export PYTHONPATH="$PYTHONPATH:$PWD"
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp/$USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"
eval "$(conda shell.bash hook)"


NUM_GPUS=1 # $(nvidia-smi --list-gpus | wc -l)
echo "Number of available GPUs: $NUM_GPUS"

TRAIN_DATA_PATH="YOUR_PATH/crf-second-batch-item-by-item-balanced" #"YOUR_PATH/discharge-admission" # "YOUR_PATH/mesh_class_20perc" # "YOUR_PATH/chronicity-classification-task" # "YOUR_PATH/tloc-classification-task" # "YOUR_PATH/emrqa-msquad" # 
TRAIN_DATA_SPLIT="train"
VAL_DATA_PATH="YOUR_PATH/crf-second-batch-item-by-item-balanced" # "YOUR_PATH/discharge-admission" # "YOUR_PATH/mesh_class_20perc" #" "YOUR_PATH/chronicity-classification-task" # "YOUR_PATH/tloc-classification-task" # "YOUR_PATH/crf-second-batch-item-by-item-balanced" # "YOUR_PATH/emrqa-msquad" #
VAL_DATA_SPLIT="validation"
TEST_DATA_SPLIT="test"
MAX_N_EXAMPLES_TRAIN=None # 60000 # 25000 # 100 # 60000
MAX_N_EXAMPLES_VAL=1000 # None # 1000 # 10000
MAX_N_EXAMPLES_TEST=None # 10000
CACHE_DIR="/workspace/.cache" # "/YOUR_PATH/.cache" # 
# special_tokens_assistant_start="</options>\\nAssistant:"

calculate_loss_on_prompt="True"

task_type="crf_task" # "admission_task" # "instruct_crf_task" #  "mesh_task" # "instruct_chronic_task" # "instruct_tlocvsdyspnea_task" #"tlocvsdyspnea_task" # "qa_task" #
tag_token_start="<crf_item>" # "<admission_item>" #"<mesh_item>"

if [[ "$task_type" == *crf* ]]; then
  task_type_dir_config="crf_task"
else
  task_type_dir_config="mesh_task"
fi
if [[ "$task_type" == *admission* ]]; then
  task_type_dir_config="admission_task"
fi

CONFIG=("unsup_qwen8B_crf_lora") # "llama_crf_lora_datav2_v1"  "medphi_35" "phi_35") #  "unsup_phi_35"  "unsup_medphi") #  "unsup_qwen_crf_lora_datav3" "unsup_llama_crf_lora_datav2_v1") #  "unsup_qwen8B_crf_lora".   # "llama8B_crf_lora" "llama_crf_lora_datav2_v1" "qwen_crf_lora_datav3"  "gemma_crf_lora" "gemma4B_crf_lora" )


for config in "${CONFIG[@]}"; do 
  custom_config_file="train_configs/${task_type_dir_config}/${config}.yaml"
  accelerate_config_file="train_configs/deepspeed_${NUM_GPUS}.json"

  echo "Starting training for model: $config, using config file: $custom_config_file, accelerate config: $accelerate_config_file"
  
  nohup bash f_train_task.sh --accelerate_config_file $accelerate_config_file\
    --main_process_port 29503 \
    --custom_config_file $custom_config_file \
    --train_data_path $TRAIN_DATA_PATH \
    --train_data_split $TRAIN_DATA_SPLIT \
    --val_data_path $VAL_DATA_PATH \
    --val_data_split $VAL_DATA_SPLIT \
    --max_n_examples_train $MAX_N_EXAMPLES_TRAIN \
    --max_n_examples_val $MAX_N_EXAMPLES_VAL \
    --max_n_examples_test $MAX_N_EXAMPLES_TEST \
    --cache_dir $CACHE_DIR \
    --task_type $task_type \
    --tag_token_start $tag_token_start \
    --calculate_loss_on_prompt $calculate_loss_on_prompt \
    &> nohup_TRAIN_${config}_${task_type}_${MAX_N_EXAMPLES_TRAIN} \
    2>&1 & wait

#     --special_tokens_assistant_start $special_tokens_assistant_start \
  echo "DONE for model: $config"
done

# bash pretrain/train_one.sh --accelerate_config_file train_configs/deepspeed_2.json --custom_config_file pretrain/qwen1B_LoRa_base.yaml