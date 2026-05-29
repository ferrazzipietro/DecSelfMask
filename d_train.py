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
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Trainer,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from huggingface_hub import login 
from dotenv import dotenv_values
from peft import PeftModel

from src.training.config import ModelArguments
from src.training.dataset import DataCollatorForMedLlm, MedLlmDataset
from src.training.load_model import load_model, merge_lora_model
from src.training.trainer import MedLlmTrainer




HF_TOKEN=dotenv_values('.env')['HF_TOKEN']
login(HF_TOKEN)


def _load_dataset_split(dataset_path: str, dataset_split: str):
    """Load a dataset split from HF hub or local json file."""
    try:
        return load_dataset(dataset_path, split=dataset_split)
    except Exception:
        try:
            dataset = load_dataset(dataset_path, dataset_split)
            if isinstance(dataset, dict):
                if dataset_split in dataset:
                    return dataset[dataset_split]
                if "train" in dataset:
                    return dataset["train"]
                return dataset[list(dataset.keys())[0]]
            return dataset
        except Exception:
            # Fallback to local json/jsonl path
            dataset = load_dataset("json", data_files=dataset_path)
            if dataset_split in dataset:
                return dataset[dataset_split]
            return dataset["train"]


def _build_encoder_mlm_dataset(
    raw_dataset,
    tokenizer,
    text_column: str,
    label_column: str,
    max_seq_length: int,
    max_n_examples: int | None = None,
):
    """Create MLM examples where labels are supervised only at mask-token positions."""
    if tokenizer.mask_token_id is None:
        raise ValueError(
            "The selected tokenizer has no mask token. Use a tokenizer/model with a mask token (e.g. BERT)."
        )

    required_columns = {text_column, label_column}
    missing = [c for c in required_columns if c not in raw_dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns {missing}. "
            f"Available columns: {raw_dataset.column_names}"
        )

    if max_n_examples is not None:
        raw_dataset = raw_dataset.select(range(min(max_n_examples, len(raw_dataset))))

    mask_token_id = tokenizer.mask_token_id
    span_mask_sentinel = "<|reserved_special_token_246|>"

    def encode_batch(batch):
        # We intentionally return more examples than input rows:
        # for each target token, we create one sequence with a single [MASK].
        single_mask_texts = []
        target_texts = []

        texts = batch[text_column]
        targets = batch[label_column]
        for text_value, target_value in zip(texts, targets):
            text = str(text_value)
            target_text = str(target_value).strip()

            normalization_candidates = [
                tokenizer.mask_token,
                "<|reserved_special_token_246|>",
            ]
            for placeholder in normalization_candidates:
                if placeholder:
                    text = text.replace(placeholder, span_mask_sentinel)

            # Support reserved special token variants like <|reserved_special_token_123|>.
            text = re.sub(r"<\|reserved_special_token_\d+\|>", span_mask_sentinel, text)

            n_placeholders = text.count(span_mask_sentinel)
            if n_placeholders == 0:
                continue

            # With no separator in labels we can only align reliably to one masked span.
            if n_placeholders != 1:
                continue

            single_mask_text = text.replace(span_mask_sentinel, tokenizer.mask_token, 1)
            single_mask_text = single_mask_text.replace(span_mask_sentinel, tokenizer.mask_token)

            single_mask_texts.append(single_mask_text)
            target_texts.append(target_text)

        if len(single_mask_texts) == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        tokenized_targets = tokenizer(target_texts, add_special_tokens=False)
        target_ids_batch = tokenized_targets["input_ids"]

        tokenized_inputs = tokenizer(
            single_mask_texts,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
        )

        out = {k: [] for k in tokenized_inputs.keys()}
        out["labels"] = []

        for sample_idx, input_ids in enumerate(tokenized_inputs["input_ids"]):
            target_ids = target_ids_batch[sample_idx]
            if len(target_ids) == 0:
                continue

            mask_positions = [pos for pos, tok_id in enumerate(input_ids) if tok_id == mask_token_id]
            if len(mask_positions) == 0:
                continue

            mask_pos = mask_positions[0]
            for target_id in target_ids:
                labels = [-100] * len(input_ids)
                labels[mask_pos] = target_id
                for k in tokenized_inputs.keys():
                    out[k].append(tokenized_inputs[k][sample_idx])
                out["labels"].append(labels)

        if len(out["labels"]) == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        return out

    # Tokenization is CPU-bound. Use a bounded number of workers to avoid
    # oversubscription while still improving throughput.
    map_num_proc = max(1, min(8, (os.cpu_count() or 1)))
    map_batch_size = 512
    print(f"Tokenizing with num_proc={map_num_proc}, batch_size={map_batch_size}")
    
    tokenized = raw_dataset.map(
        encode_batch,
        batched=True,
        batch_size=map_batch_size,
        num_proc=map_num_proc,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing dataset for encoder MLM",
    )

    def has_valid_mlm_labels(batch):
        return [any(lbl != -100 for lbl in labels) for labels in batch["labels"]]

    tokenized = tokenized.filter(
        has_valid_mlm_labels,
        batched=True,
        batch_size=map_batch_size,
        num_proc=map_num_proc,
        desc="Dropping examples without valid MLM supervision",
    )


    if len(tokenized) == 0:
        raise ValueError(
            "No valid MLM examples found after tokenization. "
            "Check mask token usage in the input sentence and target label tokenization."
        )

    return tokenized


def _infer_encoder_mlm_columns(raw_dataset):
    """Infer text/label columns for encoder MLM training."""
    columns = list(raw_dataset.column_names)

    preferred_text = ["sentence", "text", "masked_sentence", "input", "prompt"]
    preferred_label = ["label", "target", "masked_label", "output", "answer"]

    text_column = next((c for c in preferred_text if c in columns), None)
    label_column = next((c for c in preferred_label if c in columns), None)

    # Fallbacks for unusual datasets.
    if text_column is None and columns:
        text_column = columns[0]
    if label_column is None:
        for c in columns:
            if c != text_column:
                label_column = c
                break

    if text_column is None or label_column is None:
        raise ValueError(
            f"Unable to infer MLM columns from dataset columns: {columns}. "
            "Provide a dataset with at least two columns (text and label)."
        )

    return text_column, label_column


def _is_decoder_only_model(model_config):
    """Best-effort detection of decoder-only architectures."""
    if getattr(model_config, "is_encoder_decoder", False):
        return False

    # Main path used by transformers for causal LMs.
    if model_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return True

    # Fallback heuristic for configs not in the static causal mapping.
    architectures = [a.lower() for a in getattr(model_config, "architectures", []) if isinstance(a, str)]
    if any("causallm" in a or "forcausallm" in a for a in architectures):
        return True

    return False


def train_encoder_mlm(
    training_args: Seq2SeqTrainingArguments,
    model_args: ModelArguments,
):
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.label_names = ["labels"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token

    max_seq_length = model_args.max_seq_length or tokenizer.model_max_length
    if max_seq_length is None or max_seq_length <= 0:
        max_seq_length = 512

    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=(
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        ),
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=args.cache_dir,
    )

    train_raw = _load_dataset_split(train_data_path, train_data_split)
    val_raw = _load_dataset_split(val_data_path, val_data_split)

    # train_raw = train_raw.select(range(100)) 
    # val_raw = val_raw.select(range(100))

    text_column, label_column = _infer_encoder_mlm_columns(train_raw)
    logging.info(
        "Encoder MLM mode inferred columns: text_column=%s, label_column=%s",
        text_column,
        label_column,
    )

    train_dataset = _build_encoder_mlm_dataset(
        raw_dataset=train_raw,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_seq_length=max_seq_length,
        max_n_examples=max_n_examples_train,
    )
    validation_dataset = _build_encoder_mlm_dataset(
        raw_dataset=val_raw,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_seq_length=max_seq_length,
        max_n_examples=max_n_examples_val,
    )

    if max_n_examples_train is not None:
        train_dataset = train_dataset.select(range(min(max_n_examples_train, len(train_dataset))))
    if max_n_examples_val is not None:
        validation_dataset = validation_dataset.select(range(min(max_n_examples_val, len(validation_dataset))))

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForMedLlm(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    if is_main_process and training_args.hub_model_id:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=training_args.output_dir,
            repo_id=training_args.hub_model_id,
            repo_type="model",
        )

def train(training_args: Seq2SeqTrainingArguments, model_args: ModelArguments, force_instructed_tokenizer):
    """
    Train the model

    Args:
        training_args (Seq2SeqTrainingArguments): Training arguments
        model_args (ModelArguments): Model arguments
    """
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Keep the extra supervision column so the data collator can pass it through to the loss
    training_args.label_names = ["labels", "loss_weight_mask"]

    # Load model only on the main process
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
        #use_liger_kernel=training_args.use_liger_kernel,
    )
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
        import numpy as np
        # validation_dataset.dataset = validation_dataset.dataset.map(lambda x: {'len': len(x['input_ids'])}, num_proc=16)
        # third_quartile = np.percentile(validation_dataset.dataset['len'], 90)
        # validation_dataset.dataset = validation_dataset.dataset.filter(lambda x: x['len'] <= third_quartile, num_proc=16)
    
        train_dataset.dataset = train_dataset.dataset.map(lambda x: {'len': len(x['input_ids'])}, num_proc=16)
        third_quartile_train = np.percentile(train_dataset.dataset['len'], 99)
        train_dataset.dataset = train_dataset.dataset.filter(lambda x: x['len'] <= third_quartile_train, num_proc=16)
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

    # if training_args.local_rank == 0:
    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if False:
        trainer.save_model()
        try:
            tokenizer.save_pretrained(training_args.output_dir)
            # Ensure the updated vocab_size is in config.json
            if hasattr(model, "config") and getattr(model.config, "vocab_size", None) != len(tokenizer):
                model.resize_token_embeddings(len(tokenizer))
                model.config.save_pretrained(training_args.output_dir)
            print(f"Saved tokenizer and config to {training_args.output_dir}")
        except Exception as e:
            print(f"Tokenizer save failed: {e}")
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    # Step 1: Best-effort distributed teardown.
    # Avoid a hard barrier here: if one rank is slower (e.g. end-of-run I/O),
    # other ranks can block for a long NCCL timeout.
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            logging.warning(f"Failed to destroy process group cleanly: {e}")
    # Step 2: Only rank 0 uploads to Hub — pure HTTP, no collective ops needed
    if is_main_process:
        import glob
        # Prefer the checkpoint saved by save_strategy during training;
        # fall back to output_dir directly if no checkpoint subdirs exist.
        checkpoints = sorted(
            glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")),
            key=os.path.getmtime
        )
        upload_path = checkpoints[-1] if checkpoints else training_args.output_dir
        # Ensure tokenizer files are present in the upload directory
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


def merge_lora(training_args: Seq2SeqTrainingArguments, model_args: ModelArguments):
    merge_lora_model(
        weights_path=model_args.model_name_or_path,
        lora_weights_name_or_path=training_args.output_dir,
        output_path=training_args.output_dir,
        torch_dtype=model_args.torch_dtype,
    )

def manual_merge_lora(base_model_path, lora_path, output_path):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load LoRA model
    model = PeftModel.from_pretrained(model, lora_path)
    
    # Merge with safe_merge=True to handle dimension mismatches
    try:
        merged_model = model.merge_and_unload(safe_merge=True)
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Successfully merged and saved to {output_path}")
    except Exception as e:
        print(f"Merge failed: {e}")
        # Save unmerged model instead
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Saved unmerged LoRA model to {output_path}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    # CLI parser for custom script arguments
    cli_parser = ArgumentParser()
    cli_parser.add_argument('--train_data_path', type=str, required=False, default="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct")
    cli_parser.add_argument('--train_data_split', type=str, required=False, default="train")  # as_many_unknown_valid_as
    cli_parser.add_argument('--val_data_path', type=str, required=False, default="YOUR_PATH/gaussian_Llama-3.1-8B-Instruct")
    cli_parser.add_argument('--val_data_split', type=str, required=False, default="validation")
    cli_parser.add_argument('--max_n_examples_train', type=str, required=False, default=None)
    cli_parser.add_argument('--max_n_examples_val', type=str, required=False, default=None)
    cli_parser.add_argument('--calculate_loss_on_prompt', type=str, choices=['True', 'False'], required=False, default='False', help="Whether to calculate loss on prompt tokens (default: False)")
    cli_parser.add_argument('--train_adding_item_to_seq', type=str, choices=['True', 'False'], required=False, default='False', help="Whether to train the model to add the next item to the sequence (default: False).")
    # Path to HF YAML/JSON config (provided by wrapper as --custom_config_file)
    cli_parser.add_argument('--custom_config_file', type=str, required=True, help='Path to Hugging Face YAML or JSON config file')
    cli_parser.add_argument('--cache_dir', type=str, required=False, default="/workspace/.cache") # "/YOUR_PATH/.cache")
    cli_parser.add_argument('--task_type', type=str, required=False, default="crf_task", help="A string identifier for the task type, used for logging and experiment tracking (default: unsupervised_crf)")
    args, unknown = cli_parser.parse_known_args()

    os.environ["HF_HOME"] = args.cache_dir
    print(f"Set HF_HOME to {args.cache_dir}")

    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    train_data_split = args.train_data_split
    val_data_split = args.val_data_split
    calculate_loss_on_prompt = args.calculate_loss_on_prompt == 'True'
    if args.task_type == 'crf_task':
        if args.train_adding_item_to_seq == 'True':
            training_type = 'instruct_plus_item_crf' 
        else:
            training_type = 'instruct_crf'
    elif args.task_type == 'mesh_task':
        if args.train_adding_item_to_seq == 'True':
            training_type = 'instruct_plus_item_mesh'
        else:
            training_type = 'instruct_mesh'
    elif args.task_type == 'continual_pretraining_crf':
        training_type = 'continual_pretraining_crf'

    max_n_examples_train = None if args.max_n_examples_train == 'None' else int(args.max_n_examples_train)
    max_n_examples_val = None if args.max_n_examples_val == 'None' else int(args.max_n_examples_val)
    logging.basicConfig(level=logging.INFO)
    # extra args: force_instructed_tokenizer, different_tokenizer_name_or_path

    force_instructed_tokenizer = False
    instructed_tokenizers = {
        'meta-llama/Llama-3.2-1B': 'meta-llama/Llama-3.2-1B-Instruct',
        'Pretrain- -YOUR_PATH_ORG/Llama-3.2-1B_AllDataSourcesClinical_0.0002_cosine_1024_paper': 'meta-llama/Llama-3.2-1B-Instruct',
        'google/gemma-3-1b-pt': 'google/gemma-3-1b-it',
        'Pretrain- -YOUR_PATH_ORG/gemma-3-1b-pt_AllDataSourcesClinical_0.0002_cosine_1024_paper': 'google/gemma-3-1b-it'
    }
    

    # HF parser reads only from the provided config file
    hf_parser = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments))
    logging.info(f"Sys args {sys.argv}")
    config_path = os.path.abspath(args.custom_config_file)
    logging.info(f"Loading HF config {config_path}")

    if config_path.endswith(".json"):
        model_args, training_args = hf_parser.parse_json_file(json_file=config_path)
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        model_args, training_args = hf_parser.parse_yaml_file(yaml_file=config_path)
    else:
        raise ValueError("Unsupported config file format for --custom_config_file. Use .yaml, .yml, or .json.")

    # if len(sys.argv) > 0 and sys.argv[-1].endswith(".json"):
    #     # If we pass only one argument to the script, and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     logging.info(f"Loading json config {sys.argv[-1]}")
    #     model_args, training_args = parser.parse_json_file(
    #         json_file=os.path.abspath(sys.argv[-1])
    #     )

    # elif len(sys.argv) > 0 and sys.argv[-1].endswith(".yaml"):
    #     # If we pass only one argument to the script, and it's the path to a yaml file,
    #     # let's parse it to get our arguments.
    #     logging.info(f"Loading yaml config {sys.argv[-1]}")
    #     model_args, training_args = parser.parse_yaml_file(
    #         yaml_file=os.path.abspath(sys.argv[-1])
    #     )
    # else:
    #     logging.info("No config file passed, using command line arguments.")
    #     model_args, training_args = parser.parse_args_into_dataclasses()
    if "wandb" in training_args.report_to:
        import wandb
        wandb.login(key=dotenv_values('.env')['WANDB_KEY'])
        wandb.init(
            project="unsupervised_crf_unsup_train",
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
    run_decoder_mode = _is_decoder_only_model(model_config)

    if not run_decoder_mode:
        logging.info('Running encoder-only MLM training path')
        train_encoder_mlm(
            training_args=training_args,
            model_args=model_args,
        )
    else:
        logging.info('Running decoder-only training path')
        train(training_args, model_args, force_instructed_tokenizer)

    if model_args.use_lora:
        print("SKIP LORA MERGE")
        # Check if this is the main process (rank 0)
        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     try:
        #         merge_lora(training_args, model_args)
        #     except Exception as e:
        #         logging.error(f"Error during LoRA merging: {e}")
        #         # Attempt manual merge as a fallback
        #         print("Attempting manual LoRA merge...")
        #         manual_merge_lora(
        #             base_model_path=model_args.model_name_or_path,
        #             lora_path=training_args.output_dir,
        #             output_path=training_args.output_dir + '_merged'
        #         )

        # If using distributed training, teardown without a blocking barrier.
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                logging.warning(f"Failed to destroy process group cleanly: {e}")
