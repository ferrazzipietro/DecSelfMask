from __future__ import annotations

from dataclasses import dataclass
import glob
import logging
import os
from typing import Any

import torch
import torch.distributed as dist
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

from ._hf import login_if_available


@dataclass(slots=True)
class SFTTrainerConfig:
    custom_config_file: str
    train_data_path: str = "YOUR_PATH/gaussian_Llama-3.1-8B-Instruct"
    train_data_split: str = "train"
    val_data_split: str = "validation"
    test_data_split: str = "validation"
    max_n_examples_train: int | None = None
    max_n_examples_val: int | None = None
    max_n_examples_test: int | None = None
    calculate_loss_on_prompt: bool = False
    cache_dir: str | None = None
    task_type: str = "sft_task"
    tag_token_start: str = "<sft_item>"
    item_column_name: str = "sft_item"
    options_column_name: str = "options"


class SFTTrainer:
    def __init__(self, config: SFTTrainerConfig):
        self.config = config
        self._model_args = None
        self._training_args = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def _parse_config(self):
        if self._model_args is not None and self._training_args is not None:
            return self._model_args, self._training_args
        from ._training.config import ModelArguments

        parser = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments))
        config_path = os.path.abspath(self.config.custom_config_file)
        if config_path.endswith(".json"):
            self._model_args, self._training_args = parser.parse_json_file(json_file=config_path)
        elif config_path.endswith((".yaml", ".yml")):
            self._model_args, self._training_args = parser.parse_yaml_file(yaml_file=config_path)
        else:
            raise ValueError("Unsupported config file format for custom_config_file.")
        return self._model_args, self._training_args

    def _special_tokens_assistant_start(self, model_name: str) -> str:
        lowered = model_name.lower()
        if "qwen" in lowered:
            return "<|im_end|>\n<|im_start|>assistant\n"
        if "gemma" in lowered:
            return "<start_of_turn>model\n"
        if "llama" in lowered:
            return "<|start_header_id|>assistant<|end_header_id|>\n\n"
        raise ValueError(f"Cannot determine special_tokens_assistant_start for model {model_name}.")

    def prepare(self):
        from ._training.dataset import DataCollatorForMedLlm, MedLlmDataset
        from ._training.evaluation_functions import compute_metrics_sft, preprocess_logits_for_metrics
        from ._training.load_model import load_model
        from ._training.trainer import MedLlmTrainer

        model_args, training_args = self._parse_config()
        os.environ["HF_HOME"] = self.config.cache_dir
        login_if_available()

        self.model, self.tokenizer = load_model(
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
            fsdp_training=len(training_args.fsdp) > 1 or training_args.fsdp_config is not None,
            max_memory_MB=model_args.max_memory_MB,
            rope_scaling_factor=model_args.rope_scaling_factor,
            force_different_tokenizer=False,
            different_tokenizer_name_or_path=None,
            max_seq_length=model_args.max_seq_length,
        )

        special_tokens_assistant_start = self._special_tokens_assistant_start(model_args.model_name_or_path)
        special_tokens_assistant_start_ids = self.tokenizer(special_tokens_assistant_start, add_special_tokens=False)["input_ids"]
        special_tokens = [v for v in self.tokenizer.added_tokens_decoder.keys()]
        if "DecSelfMask-gemma" in model_args.model_name_or_path:
            special_tokens += [self.tokenizer.convert_tokens_to_ids("model")]
        if "Qwen" in model_args.model_name_or_path:
            special_tokens.append(198)

        if training_args.local_rank == 0:
            self.train_dataset = MedLlmDataset(
                tokenizer=self.tokenizer,
                max_n_examples=self.config.max_n_examples_train,
                train_data_path=self.config.train_data_path,
                train_data_split=self.config.train_data_split,
                training_type="sft_task",
                calculate_loss_on_prompt=self.config.calculate_loss_on_prompt,
                tag_token_start=self.config.tag_token_start,
                item_column_name=self.config.item_column_name,
                options_column_name=self.config.options_column_name,
            )
            self.validation_dataset = MedLlmDataset(
                tokenizer=self.tokenizer,
                max_n_examples=self.config.max_n_examples_val,
                train_data_path=self.config.train_data_path,
                train_data_split=self.config.val_data_split,
                training_type="sft_task",
                calculate_loss_on_prompt=self.config.calculate_loss_on_prompt,
                tag_token_start=self.config.tag_token_start,
                item_column_name=self.config.item_column_name,
                options_column_name=self.config.options_column_name,
            )
            max_n_examples_test_local = 1_000_000 if self.config.max_n_examples_test is None else self.config.max_n_examples_test + (self.config.max_n_examples_val or 0)
            self.test_dataset = MedLlmDataset(
                tokenizer=self.tokenizer,
                max_n_examples=max_n_examples_test_local,
                train_data_path=self.config.train_data_path,
                train_data_split=self.config.test_data_split,
                training_type="sft_task",
                calculate_loss_on_prompt=self.config.calculate_loss_on_prompt,
                tag_token_start=self.config.tag_token_start,
                item_column_name=self.config.item_column_name,
                options_column_name=self.config.options_column_name,
            )
        else:
            self.train_dataset = None
            self.validation_dataset = None
            self.test_dataset = None

        if dist.is_initialized():
            object_list = [self.train_dataset, self.validation_dataset]
            dist.broadcast_object_list(object_list, src=0)
            self.train_dataset, self.validation_dataset = object_list

        tag_token_start = self.tokenizer(self.config.tag_token_start, add_special_tokens=False)["input_ids"][:-1]
        special_ids = self.tokenizer.added_tokens_decoder.keys()
        token_for_masking = self.tokenizer(self.tokenizer.added_tokens_decoder[list(special_ids)[-2]].content, add_special_tokens=False)["input_ids"][0]

        def find_tokens_of_item(input_ids):
            input_ids_as_str = " ".join(map(str, input_ids))
            tag_token_start_str = " ".join(map(str, tag_token_start))
            found = input_ids_as_str.split(tag_token_start_str)[-1]
            found = found.split(str(token_for_masking))[0]
            text = self.tokenizer.decode([int(i) for i in found.split()], skip_special_tokens=False).replace("<", "").replace(">", "").replace("?", "").strip()
            return text

        items_list_in_dataset_validation = list(self.validation_dataset.dataset.map(lambda x: {"sft_item": find_tokens_of_item(x["input_ids"])}, num_proc=16)["sft_item"])
        items_list_in_dataset_test = list(self.test_dataset.dataset.map(lambda x: {"sft_item": find_tokens_of_item(x["input_ids"])}, num_proc=16)["sft_item"])

        self.trainer = MedLlmTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            compute_metrics=lambda x: compute_metrics_sft(x, special_tokens_assistant_start_ids, special_tokens, items_list_in_dataset_validation, items_list_in_dataset_test, calc_per_item=True),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            data_collator=DataCollatorForMedLlm(tokenizer=self.tokenizer),
        )
        return self

    def train(self):
        if self.trainer is None:
            self.prepare()
        assert self.trainer is not None
        self.trainer.train()
        return self.trainer

    def evaluate(self):
        if self.trainer is None:
            self.prepare()
        assert self.trainer is not None and self.test_dataset is not None
        return {
            "validation": self.trainer.evaluate(eval_dataset=self.validation_dataset),
            "test": self.trainer.evaluate(eval_dataset=self.test_dataset),
        }
