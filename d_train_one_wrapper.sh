
export PYTHONPATH="$PYTHONPATH:$PWD"
export NCCL_P2P_DISABLE=1
eval "$(conda shell.bash hook)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/triton"

# TRAIN_DATA_PATH="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268" # "YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_gemma-3-1b-it" #"YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_Qwen3-1.7B" #"YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_gemma-3-1b-it" #   "YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268"
TRAIN_DATA_SPLIT="train"
# VAL_DATA_PATH="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268" # "YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_gemma-3-1b-it" #"YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_Qwen3-1.7B" #  "YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_Qwen3-1.7B" # "YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_gemma-3-1b-it" #  "YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268"
VAL_DATA_SPLIT="validation"
MAX_N_EXAMPLES_TRAIN=None # 10 # 
MAX_N_EXAMPLES_VAL=1000
CACHE_DIR="/workspace/.cache" # "/YOUR_PATH/.cache"

calculate_loss_on_prompt=True
train_adding_item_to_seq=False
task_type="continual_pretraining_crf" # "crf_task" # "mesh_task"

# CONFIG=("gemma_CLUSTER_NAME_datav3_only_mask"  "qwen_CLUSTER_NAME_datav3_only_mask"  "llama_CLUSTER_NAME_only_mask" "llama8B_CLUSTER_NAME_v2_only_mask" "qwen8B_CLUSTER_NAME_v3_only_mask" "gemma4B_CLUSTER_NAME_v3_only_mask") # qwen32B_CLUSTER_NAME_v3") # "qwen_CLUSTER_NAME_datav3_05ep") #gemma_CLUSTER_NAME_datav3_3ep") # "qwen8B_CLUSTER_NAME_v3") # "llama8B_CLUSTER_NAME_v2") # llama_CLUSTER_NAME") # "qwen8B_CLUSTER_NAME") # "llama8B_CLUSTER_NAME") # "llama_CLUSTER_NAME") #
CONFIG=( "qwen8B_CLUSTER_NAME_v3_only_mask") # "qwen8B_CLUSTER_NAME_v3_only_mask_w_item_mesh" 

NUM_GPUS=2 # $(nvidia-smi --list-gpus | wc -l)



echo "Number of available GPUs: $NUM_GPUS"

for config in "${CONFIG[@]}"; do 
  custom_config_file="train_configs/unsupervised/${config}.yaml"
  accelerate_config_file="train_configs/deepspeed_${NUM_GPUS}.json"

  echo "Starting training for model: $config, using config file: $custom_config_file, accelerate config: $accelerate_config_file"

  # if "llama" in $config; then use the TRAIN_DATA_PATH="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268" 
  if [[ "$config" == *"llama"* ]]; then 
    if [[ "$task_type" == "mesh_task" ]]; then 
      TRAIN_DATA_PATH="YOUR_PATH/mesh_gaussian_Llama-3.1-8B-Instruct_1203965" 
      VAL_DATA_PATH="YOUR_PATH/mesh_gaussian_Llama-3.1-8B-Instruct_1203965"
    else
      TRAIN_DATA_PATH="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268" 
      VAL_DATA_PATH="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268"
    fi
  elif [[ "$config" == *"qwen"* ]]; then 
      if [[ "$task_type" == "mesh_task" ]]; then 
        TRAIN_DATA_PATH="YOUR_PATH/mesh_gaussian_Llama-3.1-8B-Instruct_1203965_for_Qwen3-1.7B" 
        VAL_DATA_PATH="YOUR_PATH/mesh_gaussian_Llama-3.1-8B-Instruct_1203965_for_Qwen3-1.7B"
      else
        TRAIN_DATA_PATH="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_Qwen3-1.7B" 
        VAL_DATA_PATH="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_Qwen3-1.7B"
      fi
    TRAIN_DATA_PATH="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_Qwen3-1.7B"
    VAL_DATA_PATH="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_Qwen3-1.7B"
  elif [[ "$config" == *"gemma"* ]]; then 
      if [[ "$task_type" == "mesh_task" ]]; then 
        TRAIN_DATA_PATH="YOUR_PATH/mesh_gaussian_Llama-3.1-8B-Instruct_1203965_for_gemma-3-1b-it" 
        VAL_DATA_PATH="YOUR_PATH/mesh_gaussian_Llama-3.1-8B-Instruct_1203965_for_gemma-3-1b-it"
      else
        TRAIN_DATA_PATH="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_gemma-3-1b-it" 
        VAL_DATA_PATH="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_gemma-3-1b-it"
      fi
  elif [[ "$config" == *"bert"* ]]; then 
    TRAIN_DATA_PATH="YOUR_PATH/mesh_gaussian_Llama-3.1-8B-Instruct_1203965" 
    VAL_DATA_PATH="YOUR_PATH/mesh_gaussian_Llama-3.1-8B-Instruct_1203965"
  elif [[ "$config" == *"phi"* ]]; then 
    if [[ "$task_type" == "mesh_task" ]]; then 
      TRAIN_DATA_PATH="YOUR_PATH/mesh_gaussian_Llama-3.1-8B-Instruct_1203965_for_Phi-3.5-mini-instruct" 
      VAL_DATA_PATH="YOUR_PATH/mesh_gaussian_Llama-3.1-8B-Instruct_1203965_for_Phi-3.5-mini-instruct"
    else
      TRAIN_DATA_PATH="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_Phi-3.5-mini-instruct" 
      VAL_DATA_PATH="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct_2004268_for_Phi-3.5-mini-instruct"
    fi
  else
    echo "Model name does not contain 'llama', 'qwen', or 'gemma', 'bert'. Using default data paths."
  fi
  
  bash d_train_one_CLUSTER_NAME.sh --accelerate_config_file $accelerate_config_file\
    --custom_config_file $custom_config_file \
    --train_data_path $TRAIN_DATA_PATH \
    --train_data_split $TRAIN_DATA_SPLIT \
    --val_data_path $VAL_DATA_PATH \
    --val_data_split $VAL_DATA_SPLIT \
    --max_n_examples_train $MAX_N_EXAMPLES_TRAIN \
    --max_n_examples_val $MAX_N_EXAMPLES_VAL \
    --cache_dir $CACHE_DIR \
    --task_type $task_type \
    --calculate_loss_on_prompt $calculate_loss_on_prompt \
    --train_adding_item_to_seq $train_adding_item_to_seq \
    &> nohup_TRAIN_${config}_UNSUP_${MAX_N_EXAMPLES_TRAIN} \
    2>&1 & wait

  echo "DONE for model: $config"
done

# bash pretrain/train_one.sh --accelerate_config_file train_configs/deepspeed_2.json --custom_config_file pretrain/qwen1B_LoRa_base.yaml