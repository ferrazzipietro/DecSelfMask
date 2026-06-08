import logging
import os
import re
import sys
import torch

import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoConfig,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from huggingface_hub import login 
from dotenv import dotenv_values
from peft import PeftModel

from src.training.config import ModelArguments
from src.training.dataset import DataCollatorForMedLlm, MedLlmDataset
from src.training.load_model import load_model
from src.training.trainer import MedLlmTrainer




HF_TOKEN=dotenv_values('.env')['HF_TOKEN']
login(HF_TOKEN)


def train(training_args: Seq2SeqTrainingArguments, model_args: ModelArguments, force_instructed_tokenizer):
    """
    Train the model

    Args:
        training_args (Seq2SeqTrainingArguments): Training arguments
        model_args (ModelArguments): Model arguments
    """
    os.makedirs(training_args.output_dir, exist_ok=True)

    training_args.label_names = ["labels", "loss_weight_mask"]

    print("READY TO LOAD MODEL")
    model, tokenizer = load_model(
        inference=False,
        model_weights_name_or_path=model_args.model_name_or_path,
        quantization=model_args.quantization,
        use_lora=model_args.use_lora,
        lora_r=model_args.lora_r,
        lora_target_modules=model_args.lora_target_modules,
        torch_dtype=model_args.torch_dtype,
        force_auto_device_map=model_args.force_auto_device_map,
        use_gradient_checkpointing=training_args.gradient_checkpointing,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention=model_args.use_flash_attention,
        fsdp_training=len(training_args.fsdp) > 1
        or training_args.fsdp_config is not None,
        max_memory_MB=model_args.max_memory_MB,
        rope_scaling_factor=model_args.rope_scaling_factor,
        force_different_tokenizer=force_instructed_tokenizer,
        different_tokenizer_name_or_path=instructed_tokenizers.get(model_args.model_name_or_path, None),
        max_seq_length=model_args.max_seq_length,
        cache_dir=args.cache_dir,
    )
    print("MODEL LOADED ON DEVICE", model.device)
    print(f"Tokenizer: {tokenizer}")
    print(f"Model: {model}")

    print(f"Model_max_length: {tokenizer.model_max_length}")

    if training_args.local_rank == 0:
        train_dataset = MedLlmDataset(
            tokenizer=tokenizer, 
            max_n_examples=max_n_examples_train, 
            train_data_path=train_data_path, 
            train_data_split=train_data_split,
            training_type=training_type,
            calculate_loss_on_prompt=calculate_loss_on_prompt
            )
        validation_dataset = MedLlmDataset(
            tokenizer=tokenizer, 
            max_n_examples=max_n_examples_val, 
            train_data_path=val_data_path, 
            train_data_split=val_data_split,
            training_type=training_type,
            calculate_loss_on_prompt=calculate_loss_on_prompt)
    else:
        train_dataset = None
        validation_dataset = None


    # Broadcast the datasets to all processes
    if dist.is_initialized():
        object_list = [train_dataset, validation_dataset]
        dist.broadcast_object_list(object_list, src=0)
        train_dataset, validation_dataset = object_list

    print('train_dataset: ', train_dataset)
    print('train_dataset[0]: ', train_dataset[0]    )


    trainer = MedLlmTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForMedLlm(tokenizer=tokenizer),
    )

    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            logging.warning(f"Failed to destroy process group cleanly: {e}")
    if is_main_process:
        import glob
        checkpoints = sorted(
            glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")),
            key=os.path.getmtime
        )
        upload_path = checkpoints[-1] if checkpoints else training_args.output_dir
        try:
            tokenizer.save_pretrained(upload_path)
        except Exception as e:
            print(f"Tokenizer save failed: {e}")
        logging.info(f"Pushing model from {upload_path} to {training_args.hub_model_id}")
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=upload_path,
            repo_id=training_args.hub_model_id,
            repo_type="model",
        )

if __name__ == "__main__":
    from argparse import ArgumentParser
    cli_parser = ArgumentParser()
    cli_parser.add_argument('--train_data_path', type=str, required=False, default="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct")
    cli_parser.add_argument('--train_data_split', type=str, required=False, default="train")  # as_many_unknown_valid_as
    cli_parser.add_argument('--val_data_path', type=str, required=False, default="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct")
    cli_parser.add_argument('--val_data_split', type=str, required=False, default="validation")
    cli_parser.add_argument('--max_n_examples_train', type=str, required=False, default=None)
    cli_parser.add_argument('--max_n_examples_val', type=str, required=False, default=None)
    cli_parser.add_argument('--calculate_loss_on_prompt', type=str, choices=['True', 'False'], required=False, default='False', help="Whether to calculate loss on prompt tokens (default: False)")
    cli_parser.add_argument('--custom_config_file', type=str, required=True, help='Path to Hugging Face YAML or JSON config file')
    cli_parser.add_argument('--cache_dir', type=str, required=False, default="/workspace/.cache") 
    cli_parser.add_argument('--task_type', type=str, required=False, default="dec_self_mask", help="A string identifier for the task type, used for logging and experiment tracking (default: dec_self_mask)")
    args, unknown = cli_parser.parse_known_args()

    os.environ["HF_HOME"] = args.cache_dir
    print(f"Set HF_HOME to {args.cache_dir}")

    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    train_data_split = args.train_data_split
    val_data_split = args.val_data_split
    calculate_loss_on_prompt = args.calculate_loss_on_prompt == 'True'
    training_type = 'instruct_task'

    max_n_examples_train = None if args.max_n_examples_train == 'None' else int(args.max_n_examples_train)
    max_n_examples_val = None if args.max_n_examples_val == 'None' else int(args.max_n_examples_val)
    logging.basicConfig(level=logging.INFO)

    force_instructed_tokenizer = False
    
    hf_parser = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments))
    logging.info(f"Sys args {sys.argv}")
    config_path = os.path.abspath(args.custom_config_file)
    logging.info(f"Loading HF config {config_path}")

    if config_path.endswith(".json"):
        model_args, training_args = hf_parser.parse_json_file(json_file=config_path)
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        model_args, training_args = hf_parser.parse_yaml_file(yaml_file=config_path)
    else:
        raise ValueError("unsupported config file format for --custom_config_file. Use .yaml, .yml, or .json.")

    if "wandb" in training_args.report_to:
        import wandb
        wandb.login(key=dotenv_values('.env')['WANDB_KEY'])
        wandb.init(
            project="DecSelfMask_train",
            name=training_args.run_name,
            config=training_args.to_dict()
        )
    login(dotenv_values('.env')['HF_TOKEN'])
    try:
        print("VISIBLE gpus", os.environ["CUDA_VISIBLE_DEVICES"])
    except:
        pass
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=args.cache_dir,
    )

    train(training_args, model_args, force_instructed_tokenizer)

    if model_args.use_lora:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                logging.warning(f"Failed to destroy process group cleanly: {e}")
