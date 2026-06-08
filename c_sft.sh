
export PYTHONPATH="$PYTHONPATH:$PWD"
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp/$USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"
eval "$(conda shell.bash hook)"


NUM_GPUS=1 
echo "Number of available GPUs: $NUM_GPUS"

TRAIN_DATA_PATH="ferrazzipietro/crf-second-batch-item-by-item-balanced"
TRAIN_DATA_SPLIT="train"
VAL_DATA_SPLIT="validation"
TEST_DATA_SPLIT="test"
MAX_N_EXAMPLES_TRAIN=32 
MAX_N_EXAMPLES_VAL=32  
MAX_N_EXAMPLES_TEST=32  
CACHE_DIR="/data02/shared/pferrazzi/.cache"

calculate_loss_on_prompt="True"

task_type="sft_task"
tag_token_start="<sft_item>"
item_column_name="crf_item"
options_column_name="options"


CONFIG=("llama_1b")


for config in "${CONFIG[@]}"; do 
  custom_config_file="train_configs/sft/${config}.yaml"
  accelerate_config_file="train_configs/deepspeed_${NUM_GPUS}.json"

  echo "Starting training for model: $config, using config file: $custom_config_file, accelerate config: $accelerate_config_file"
  
  nohup accelerate launch \
    --config_file ${accelerate_config_file}\
      scripts/sft_train_task.py \
    --custom_config_file ${custom_config_file} \
    --train_data_path ${TRAIN_DATA_PATH} \
    --train_data_split ${TRAIN_DATA_SPLIT} \
    --val_data_split ${VAL_DATA_SPLIT} \
    --test_data_split ${TEST_DATA_SPLIT} \
    --max_n_examples_val ${MAX_N_EXAMPLES_VAL} \
    --max_n_examples_train ${MAX_N_EXAMPLES_TRAIN}  \
    --max_n_examples_test ${MAX_N_EXAMPLES_TEST} \
    --task_type ${task_type} \
    --item_column_name ${item_column_name} \
    --options_column_name ${options_column_name} \
    --calculate_loss_on_prompt ${calculate_loss_on_prompt} \
    --tag_token_start ${tag_token_start} \
    --cache_dir ${CACHE_DIR} &> nohup_TRAIN_${config}_${task_type}_${MAX_N_EXAMPLES_TRAIN} \
    2>&1 & wait
  echo "DONE for model: $config"
done

