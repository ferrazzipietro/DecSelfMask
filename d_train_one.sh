#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$PWD"

eval "$(conda shell.bash hook)"

CACHE_DIR="/YOUR_PATH/.cache" #"/workspace/.cache" # 
ACCELERATE_CONFIG_FILE="train_configs/deepspeed_4.json"
CUSTOM_CONFIG_FILE="pretrain/qwen1B_LoRa.yaml"
TRAIN_DATA_PATH="/YOUR_PATH/unannotated_crf/data/cc_train_sequences/all_group/<|reserved_special_token_246|>/gaussian_Llama-3.1-8B-Instruct.json"
TRAIN_DATA_SPLIT="train"
VAL_DATA_PATH="data/cc_train_sequences/single_token/<|reserved_special_token_246|>/gaussian_Llama-3.1-8B-Instruct_VAL.json"
VAL_DATA_SPLIT="validation"
MAX_N_EXAMPLES_TRAIN=None
MAX_N_EXAMPLES_VAL=None
calculate_loss_on_prompt=True
task_type="crf_task"

train_adding_item_to_seq=False

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --accelerate_config_file)
      ACCELERATE_CONFIG_FILE="$2"
      shift 2
      ;;
    --custom_config_file)
      CUSTOM_CONFIG_FILE="$2"
      shift 2
      ;;
    --train_data_path)
      TRAIN_DATA_PATH="$2"
      shift 2
      ;;
    --train_data_split)
      TRAIN_DATA_SPLIT="$2"
      shift 2
      ;;
    --val_data_path)
      VAL_DATA_PATH="$2"
      shift 2
      ;;
    --val_data_split)
      VAL_DATA_SPLIT="$2"
      shift 2
      ;;
    --max_n_examples_train)
      MAX_N_EXAMPLES_TRAIN="$2"
      shift 2
      ;;
    --max_n_examples_val)
      MAX_N_EXAMPLES_VAL="$2"
      shift 2
      ;;
    --cache_dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --calculate_loss_on_prompt)
      calculate_loss_on_prompt="$2"
      shift 2
      ;;
    --train_adding_item_to_seq)
      train_adding_item_to_seq="$2"
      shift 2
      ;;
    --task_type)
      task_type="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --accelerate_config_file CONFIG    Path to configuration file for accelerate (required)"
      echo "  --custom_config_file CONFIG        Path to configuration file for the custom training run (required)"
      echo "  --devices ID                       GPU devices ID )"
      echo "  --cache_dir DIR                    Cache directory (default: ${CACHE_DIR})"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done
export HF_HOME=${CACHE_DIR}
RUN_NAME=$(basename ${CUSTOM_CONFIG_FILE} .yaml)

# WARNING: main_process_port to 0 breaks everything
# CUDA_VISIBLE_DEVICES=3,4,5,6  nohup accelerate launch \
echo "output at nohup_TRAIN_${RUN_NAME}"
accelerate launch \
 --config_file ${ACCELERATE_CONFIG_FILE}\
 --main_process_port 29506\
  d_train.py \
 --custom_config_file ${CUSTOM_CONFIG_FILE} \
 --cache_dir ${CACHE_DIR} \
  --train_data_path ${TRAIN_DATA_PATH} \
  --train_data_split ${TRAIN_DATA_SPLIT} \
  --val_data_path ${VAL_DATA_PATH} \
  --calculate_loss_on_prompt ${calculate_loss_on_prompt} \
  --task_type ${task_type} \
  --val_data_split ${VAL_DATA_SPLIT} \
  --max_n_examples_train ${MAX_N_EXAMPLES_TRAIN} \
  --max_n_examples_val ${MAX_N_EXAMPLES_VAL} \
  --train_adding_item_to_seq ${train_adding_item_to_seq}


# se non va, runna direttamente da coomand line
# CUDA_VISIBLE_DEVICES=6 nohup accelerate launch --main_process_port 29502 --config_file train_configs/deepspeed_1.json src/train.py --custom_config_file pretrain/qwen1B_LoRa.yaml &> pretrain/nohup_TRAIN_qwen1B_LoRa
