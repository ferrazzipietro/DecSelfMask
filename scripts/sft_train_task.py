import logging
import os
import sys
import torch

import torch.distributed as dist
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainingArguments
from huggingface_hub import login 
from dotenv import dotenv_values
from peft import PeftModel

from src.training.config import ModelArguments
from src.training.dataset import DataCollatorForMedLlm, MedLlmDataset
from src.training.load_model import load_model, merge_lora_model
from src.training.trainer import MedLlmTrainer
from src.training.evaluation_functions import compute_metrics_sft, preprocess_logits_for_metrics


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
        different_tokenizer_name_or_path=None,
        max_seq_length=model_args.max_seq_length,
    )

    SPECIAL_TOKENS_ASSISTANT_START = tokenizer(special_tokens_assistant_start, add_special_tokens=False)['input_ids']
    SPECIAL_TOKENS= [v for v in tokenizer.added_tokens_decoder.keys()]
    if 'DecSelfMask-gemma' in model_args.model_name_or_path:
        SPECIAL_TOKENS += [tokenizer.convert_tokens_to_ids('model')]
    if 'Qwen' in model_args.model_name_or_path:
        SPECIAL_TOKENS.append(198)
    
    print("SPECIAL_TOKENS_ASSISTANT_START: ", SPECIAL_TOKENS_ASSISTANT_START)
    print("SPECIAL_TOKENS: ", SPECIAL_TOKENS)
    print("MODEL LOADED ON DEVICE", model.device)
    print(f"Tokenizer: {tokenizer}")
    print(f"Model: {model}")


    print(f"Model_max_length: {tokenizer.model_max_length}")

    # Load dataset only on the main process
    if training_args.local_rank == 0:
        train_dataset = MedLlmDataset(
            tokenizer=tokenizer, 
            max_n_examples=max_n_examples_train,
            train_data_path=train_data_path, 
            train_data_split=train_data_split,
            training_type=args.task_type,
            calculate_loss_on_prompt=calculate_loss_on_prompt,
            tag_token_start=args.tag_token_start,
            item_column_name=args.item_column_name,
            options_column_name=args.options_column_name,
        )
        
        validation_dataset = MedLlmDataset(
            tokenizer=tokenizer, 
            max_n_examples=max_n_examples_val,
            train_data_path=val_data_path, 
            train_data_split=val_data_split,
            training_type=args.task_type,
            calculate_loss_on_prompt=calculate_loss_on_prompt,
            tag_token_start=args.tag_token_start,
            item_column_name=args.item_column_name,
            options_column_name=args.options_column_name,
        )
        max_n_examples_test_local = (
            1_000_000
            if max_n_examples_test is None
            else max_n_examples_test + (max_n_examples_val or 0)
        )
        test_set = MedLlmDataset(
            tokenizer=tokenizer, 
            max_n_examples=max_n_examples_test_local,
            train_data_path=val_data_path, 
            train_data_split=test_data_split,
            training_type=args.task_type,
            calculate_loss_on_prompt=calculate_loss_on_prompt,
            tag_token_start=args.tag_token_start,
            item_column_name=args.item_column_name,
            options_column_name=args.options_column_name,
        )
    else:
        train_dataset = None
        validation_dataset = None
    if dist.is_initialized():
        object_list = [train_dataset, validation_dataset]
        dist.broadcast_object_list(object_list, src=0)
        train_dataset, validation_dataset = object_list

    print('train_dataset: ', train_dataset)
    print('train_dataset[0]: ', train_dataset[0]    )

    compute_metrics = compute_metrics_sft

    tag_token_start = tokenizer(args.tag_token_start, add_special_tokens=False)["input_ids"]
    tag_token_start = tag_token_start[:-1]
    special_ids = tokenizer.added_tokens_decoder.keys()
    token_for_masking = tokenizer(tokenizer.added_tokens_decoder[list(special_ids)[-2]].content, add_special_tokens=False)['input_ids'][0]

    def find_tokens_of_item(input_ids, tag_token_start, token_for_masking):
        input_ids_as_str = " ".join(map(str, input_ids))
        tag_token_start_str = " ".join(map(str, tag_token_start))
        found = input_ids_as_str.split(tag_token_start_str)[-1]
        found = found.split(str(token_for_masking))[0]
        text = tokenizer.decode([int(i) for i in found.split()], skip_special_tokens=False).replace('<','').replace('>','').replace('?', '').strip()
        return text
    print('tag_token_start: ', tag_token_start)
    items_list_in_dataset_validation = list(validation_dataset.dataset.map(lambda x: {'sft_item': find_tokens_of_item(x['input_ids'], tag_token_start, token_for_masking)}, num_proc=16)['sft_item'])
    items_list_in_dataset_test = list(test_set.dataset.map(lambda x: {'sft_item': find_tokens_of_item(x['input_ids'], tag_token_start, token_for_masking)}, num_proc=16)['sft_item'])
    print(f"Items in validation dataset [:10] over{len(set(items_list_in_dataset_validation))}: {set(items_list_in_dataset_validation[:10])}")
    print(f"Items in test dataset [:10] over{len(set(items_list_in_dataset_test))}: {set(items_list_in_dataset_test[:10])}")
    
    calc_per_item = True
    trainer = MedLlmTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=lambda x: compute_metrics(x, SPECIAL_TOKENS_ASSISTANT_START, SPECIAL_TOKENS, items_list_in_dataset_validation, items_list_in_dataset_test, calc_per_item=calc_per_item),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=DataCollatorForMedLlm(tokenizer=tokenizer),
    )

    if training_args.local_rank == 0:
        trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if training_args.local_rank == 0:
        logging.info("Running evaluation on test set...")
        test_results = trainer.evaluate(eval_dataset=test_set)
        logging.info(f"Test set results: {test_results}")
        if "wandb" in training_args.report_to:
            import wandb
            wandb.log({"test/test_end_of_training_" + k: v for k, v in test_results.items()})


        logging.info("Running final evaluation on best checkpoint using the existing trainer...")
        import glob

        best_model_path = trainer.state.best_model_checkpoint
        checkpoint_candidates = sorted(
            glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")),
            key=os.path.getmtime,
        )

        if best_model_path is not None and os.path.isdir(best_model_path):
            eval_model_path = best_model_path
            logging.info(f"Using trainer.state.best_model_checkpoint: {eval_model_path}")
        elif checkpoint_candidates:
            eval_model_path = checkpoint_candidates[-1]
            logging.warning(
                "No valid best checkpoint found in trainer state. "
                f"Falling back to latest checkpoint directory: {eval_model_path}"
            )
        else:
            raise FileNotFoundError(
                "Cannot run final best-checkpoint test evaluation: no valid best checkpoint "
                f"and no checkpoint-* directories found under {training_args.output_dir}."
            )

        if (
            training_args.load_best_model_at_end
            and trainer.state.best_model_checkpoint is not None
            and os.path.abspath(eval_model_path) == os.path.abspath(trainer.state.best_model_checkpoint)
        ):
            logging.info("Best checkpoint already loaded in current trainer.")
        else:
            logging.info(f"Loading checkpoint in-place for evaluation: {eval_model_path}")
            trainer._load_from_checkpoint(eval_model_path)

        best_test_results = trainer.evaluate(eval_dataset=test_set)
        logging.info(f"Test set results of best model: {best_test_results}")
        if "wandb" in training_args.report_to:
            import wandb
            wandb.log({"test/test_best_model_" + k: v for k, v in best_test_results.items()})
            
    logging.info(f"Pushing model to {training_args.hub_model_id}")
    try:
        trainer.push_to_hub()
    except Exception as e:
        logging.warning(f"Failed to push model to hub: {e}")
        logging.info("Saving model locally instead.")
        trainer.save_model()


def merge_lora(training_args: Seq2SeqTrainingArguments, model_args: ModelArguments):
    merge_lora_model(
        weights_path=model_args.model_name_or_path,
        lora_weights_name_or_path=training_args.output_dir,
        output_path=training_args.output_dir,
        torch_dtype=model_args.torch_dtype,
    )

def manual_merge_lora(base_model_path, lora_path, output_path):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(model, lora_path)
    try:
        merged_model = model.merge_and_unload(safe_merge=True)
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Successfully merged and saved to {output_path}")
    except Exception as e:
        print(f"Merge failed: {e}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Saved unmerged LoRA model to {output_path}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    cli_parser = ArgumentParser()
    cli_parser.add_argument('--train_data_path', type=str, required=False, default="/YOUR_PATH/unannotated_crf/data/cc_train_sequences/all_group/<|reserved_special_token_246|>/gaussian_Llama-3.1-8B-Instruct.json")
    cli_parser.add_argument('--train_data_split', type=str, required=False, default="train")  # as_many_unknown_valid_as
    cli_parser.add_argument('--val_data_split', type=str, required=False, default="validation")
    cli_parser.add_argument('--test_data_split', type=str, required=False, default="validation")
    cli_parser.add_argument('--max_n_examples_train', type=str, required=False, default=None)
    cli_parser.add_argument('--max_n_examples_val', type=str, required=False, default=None)
    cli_parser.add_argument('--max_n_examples_test', type=str, required=False, default=None)
    cli_parser.add_argument('--special_tokens_assistant_start', type=str, required=False, default="<|start_header_id|>assistant<|end_header_id|>\n\n", help='List of token IDs representing the start of the assistant special token sequence') # 
    cli_parser.add_argument('--calculate_loss_on_prompt', type=str, choices=['True', 'False'], required=False, default='False', help="Whether to calculate loss on prompt tokens (default: False)")
    cli_parser.add_argument('--tag_token_start', type=str, required=False, default="<sft_item>", help='Token indicating the start of item to classify on')
    cli_parser.add_argument('--custom_config_file', type=str, required=True, help='Path to Hugging Face YAML or JSON config file')
    cli_parser.add_argument('--task_type', type=str, default="sft_task", help='Type of task to train on, used by dataset to determine how to format inputs and calculate metrics')
    cli_parser.add_argument('--item_column_name', type=str, default="sft_item", help='Name of the column in the dataset that contains the item to classify on, used for per-item F1 calculation')
    cli_parser.add_argument('--options_column_name', type=str, default="options", help='Name of the column in the dataset that contains the options for classification, used for per-item F1 calculation')
    args, unknown = cli_parser.parse_known_args()

    train_data_path = args.train_data_path
    val_data_path = args.train_data_path
    train_data_split = args.train_data_split
    val_data_split = args.val_data_split
    test_data_split = args.test_data_split
    calculate_loss_on_prompt = args.calculate_loss_on_prompt == 'True'
    max_n_examples_train = None if args.max_n_examples_train == 'None' else int(args.max_n_examples_train)
    max_n_examples_val = None if args.max_n_examples_val == 'None' else int(args.max_n_examples_val)
    max_n_examples_test = 100_000 if args.max_n_examples_test == 'None' else int(args.max_n_examples_test)
    max_n_examples = max(max_n_examples_train or 0, max_n_examples_val or 0) or None
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
        raise ValueError("DecSelfMaskported config file format for --custom_config_file. Use .yaml, .yml, or .json.")
    

    if "qwen" in model_args.model_name_or_path.lower():
        special_tokens_assistant_start = "<|im_end|>\n<|im_start|>assistant\n"
    elif "gemma" in model_args.model_name_or_path.lower():
        special_tokens_assistant_start = "<start_of_turn>model\n"
    elif "llama" in model_args.model_name_or_path.lower():
        special_tokens_assistant_start = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        raise ValueError(f"Cannot determine special_tokens_assistant_start for model {model_args.model_name_or_path}.  or set it manually.")

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    if "wandb" in training_args.report_to:
        import wandb
        wandb.login(key=dotenv_values('.env')['WANDB_KEY'])
        wandb.init(
            project=f"DecSelfMask_{args.task_type}",
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
    train(training_args, model_args, force_instructed_tokenizer)