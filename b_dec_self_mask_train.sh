
export PYTHONPATH="$PYTHONPATH:$PWD"
export NCCL_P2P_DISABLE=1
eval "$(conda shell.bash hook)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_DATA_PATH="ferrazzipietro/DecSelfMask-gaussian_Llama-3.1-8B-Instruct"
TRAIN_DATA_SPLIT="train"
VAL_DATA_SPLIT="validation" 
MAX_N_EXAMPLES_TRAIN=None 
MAX_N_EXAMPLES_VAL=None

CACHE_DIR="/data02/shared/pferrazzi/.cache"

calculate_loss_on_prompt=True
task_type="dec_self_mask"

CONFIG=("llama_1b")

NUM_GPUS=1


export HF_HOME=${CACHE_DIR}

echo "Number of available GPUs: $NUM_GPUS"

for config in "${CONFIG[@]}"; do 
  custom_config_file="train_configs/DecSelfMask/${config}.yaml"
  accelerate_config_file="train_configs/deepspeed_${NUM_GPUS}.json"

  echo "Starting training for model: $config, using config file: $custom_config_file, accelerate config: $accelerate_config_file"

  RUN_NAME=$(basename ${custom_config_file} .yaml)

  echo "output at nohup_TRAIN_${RUN_NAME}"
  accelerate launch \
    --config_file ${accelerate_config_file}\
    --main_process_port 29500\
      scripts/dec_self_mask_train.py \
    --custom_config_file ${custom_config_file} \
    --cache_dir ${CACHE_DIR} \
    --train_data_path ${TRAIN_DATA_PATH} \
    --train_data_split ${TRAIN_DATA_SPLIT} \
    --val_data_path ${TRAIN_DATA_PATH} \
    --val_data_split ${VAL_DATA_SPLIT} \
    --calculate_loss_on_prompt ${calculate_loss_on_prompt} \
    --task_type ${task_type} \
    --max_n_examples_train ${MAX_N_EXAMPLES_TRAIN} \
    --max_n_examples_val ${MAX_N_EXAMPLES_VAL} \
    2>&1 & wait

  echo "DONE for model: $config"
done

# bash pretrain/train_one.sh --accelerate_config_file train_configs/deepspeed_2.json --custom_config_file pretrain/qwen1B_LoRa_base.yaml