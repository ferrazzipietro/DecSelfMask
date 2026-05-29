#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$PWD"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp/$USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"

eval "$(conda shell.bash hook)"

CACHE_DIR="/workspace/.cache" # "/YOUR_PATH/.cache" #
ACCELERATE_CONFIG_FILE="train_configs/deepspeed_4.json"
CUSTOM_CONFIG_FILE="pretrain/qwen1B_LoRa.yaml"
TRAIN_DATA_PATH="/YOUR_PATH/unannotated_crf/data/cc_train_sequences/all_group/<|reserved_special_token_246|>/gaussian_Llama-3.1-8B-Instruct.json"
TRAIN_DATA_SPLIT="train"
VAL_DATA_PATH="data/cc_train_sequences/single_token/<|reserved_special_token_246|>/gaussian_Llama-3.1-8B-Instruct_VAL.json"
VAL_DATA_SPLIT="validation"
MAX_N_EXAMPLES_TRAIN=None
MAX_N_EXAMPLES_VAL=None
MAX_N_EXAMPLES_TEST=None
task_type="medqa_task"
main_process_port=29500
calculate_loss_on_prompt=False
tag_token_start="<mesh_item>"

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
    --tag_token_start)
      tag_token_start="$2"
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
    --max_n_examples_test)
      MAX_N_EXAMPLES_TEST="$2"
      shift 2
      ;;
    --cache_dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --task_type)
      task_type="$2"
      shift 2
      ;;
    --main_process_port)
      main_process_port="$2"
      shift 2
      ;;
    --calculate_loss_on_prompt)
      calculate_loss_on_prompt="$2"
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

echo "which python: $(which python)"

# WARNING: main_process_port to 0 breaks everything
# CUDA_VISIBLE_DEVICES=3,4,5,6  nohup accelerate launch \
accelerate launch \
 --config_file ${ACCELERATE_CONFIG_FILE}\
 --main_process_port ${main_process_port} \
  f_train_task.py \
 --custom_config_file ${CUSTOM_CONFIG_FILE} \
  --train_data_path ${TRAIN_DATA_PATH} \
  --train_data_split ${TRAIN_DATA_SPLIT} \
  --val_data_path ${VAL_DATA_PATH} \
  --val_data_split ${VAL_DATA_SPLIT} \
  --max_n_examples_val ${MAX_N_EXAMPLES_VAL} \
  --max_n_examples_train ${MAX_N_EXAMPLES_TRAIN}  \
  --task_type ${task_type} \
  --calculate_loss_on_prompt ${calculate_loss_on_prompt} \
  --max_n_examples_test ${MAX_N_EXAMPLES_TEST} \
  --tag_token_start ${tag_token_start} \
  --cache_dir ${CACHE_DIR} 

# se non va, runna direttamente da coomand line
# CUDA_VISIBLE_DEVICES=6 nohup accelerate launch --main_process_port 29502 --config_file train_configs/deepspeed_1.json src/train.py --custom_config_file pretrain/qwen1B_LoRa.yaml &> pretrain/nohup_TRAIN_qwen1B_LoRa
