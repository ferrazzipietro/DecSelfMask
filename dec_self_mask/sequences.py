from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict
from scipy.ndimage import gaussian_filter1d
from transformers import AutoTokenizer

from ._hf import login_if_available


@dataclass(slots=True)
class SequenceMakerConfig:
    input_path: str = "data/attention_relevancy_unannotated/ClinicalWhole/Llama-3.1-8B-Instruct/combined_mid.json"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    smoothing_method: str = "gaussian"
    threshold_upper: float = 0.4
    threshold_lower: float = 0.2
    sep_token: str = "Ġ"
    type_of_masking: str = "all_group"
    cache_dir: str | None = None
    exclude_single_group_sequences: bool = False
    output_dir: str = "dec_self_mask_train_sequences"
    hf_account_name: str | None = None
    hf_token: str | None = None


class DecSelfMaskSequencesMaker:
    def __init__(self, config: SequenceMakerConfig):
        self.config = config
        self.tokenizer = None
        self.relevance_data: dict[str, Any] | None = None

    @staticmethod
    def gaussian_smoothing(importance, kernel_size=2, sigma=1.0):
        return gaussian_filter1d(importance, sigma=sigma, truncate=((kernel_size - 1) / 2) / sigma)

    @staticmethod
    def find_consecutive_high_importance(importance_scores, threshold_upper, threshold_lower):
        high_importance_groups = []
        current_group = []
        has_above_upper = False
        for token_i, score in enumerate(importance_scores):
            if score > threshold_lower:
                current_group.append(token_i)
                if score > threshold_upper:
                    has_above_upper = True
            else:
                if current_group and has_above_upper:
                    high_importance_groups.append(current_group)
                current_group = []
                has_above_upper = False
        if current_group and has_above_upper:
            high_importance_groups.append(current_group)
        return high_importance_groups

    @staticmethod
    def assign_word_groups(tokens, groups_labels, start_token="Ġ"):
        word_groups_labels = []
        current_word_tokens = []
        current_word_labels = []
        for token, group_label in zip(tokens, groups_labels):
            if token.startswith(start_token) and current_word_tokens:
                non_zero = [l for l in current_word_labels if l != 0]
                label = non_zero[0] if non_zero else 0
                word_groups_labels.extend([label] * len(current_word_tokens))
                current_word_tokens = []
                current_word_labels = []
            current_word_tokens.append(token)
            current_word_labels.append(group_label)
        if current_word_tokens:
            non_zero = [l for l in current_word_labels if l != 0]
            label = non_zero[0] if non_zero else 0
            word_groups_labels.extend([label] * len(current_word_tokens))
        unique_labels = sorted(set(word_groups_labels) - {0})
        if unique_labels:
            remap = {old: new for new, old in enumerate(unique_labels, start=1)}
            remap[0] = 0
            word_groups_labels = [remap[l] for l in word_groups_labels]
        return word_groups_labels

    def _load_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer
        login_if_available(self.config.hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, cache_dir=self.config.cache_dir)
        return self.tokenizer

    def _load_relevance_data(self):
        if self.relevance_data is not None:
            return self.relevance_data
        with open(self.config.input_path, "r") as handle:
            self.relevance_data = json.load(handle)
        return self.relevance_data

    def _token_for_masking(self, tokenizer):
        special_ids = list(tokenizer.added_tokens_decoder.keys())
        return tokenizer.added_tokens_decoder[special_ids[-2]].content

    def _build_example_dataset(self) -> Dataset:
        part_out = self._load_relevance_data()

        def examples():
            for key, value in part_out.items():
                rpit = value["relevancy_attn_lrp"]["relevancy_prompt_input_text"]
                yield {
                    "id": key,
                    "prompt": value["prompt"],
                    "tokens": [t[0] for t in rpit],
                    "relevancy_prompt_input_text": [[t, str(s)] for t, s in rpit],
                    "token_position": value["token_position"],
                }

        return Dataset.from_generator(examples, cache_dir=self.config.cache_dir, num_proc=8)

    def _add_groups(self, example, tokenizer):
        tokens = [t[0] for t in example["relevancy_prompt_input_text"]]
        smoothed = self.gaussian_smoothing([float(t[1]) for t in example["relevancy_prompt_input_text"]], kernel_size=3, sigma=1.0)
        groups = self.find_consecutive_high_importance(smoothed, threshold_upper=self.config.threshold_upper, threshold_lower=self.config.threshold_lower)
        groups_dict = {t_id: i for i, group in enumerate(groups, start=1) for t_id in group}
        groups_labels = [groups_dict.get(i, 0) for i in range(len(example["relevancy_prompt_input_text"]))]
        example["high_importance_groups"] = self.assign_word_groups(tokens, groups_labels, start_token=self.config.sep_token)
        return example

    def _create_masked_seq(self, example, tokenizer, token_for_masking):
        valid_masking_types = ["single_token", "all_group"]
        if self.config.type_of_masking not in valid_masking_types:
            raise ValueError(f"Invalid masking_type: {self.config.type_of_masking}. Choose {valid_masking_types}.")

        masked_sequences = []
        labels = []
        sentences = []
        has_left_list = []
        if self.config.type_of_masking == "single_token":
            positions_to_mask = [i for i, el in enumerate(example["high_importance_groups"]) if el != 0]
            has_group_left = [el > 1 for i, el in enumerate(example["high_importance_groups"]) if i in positions_to_mask]
            for mask_pos, has_left in zip(positions_to_mask, has_group_left):
                sequence = example["tokens"].copy()
                target_token = sequence[mask_pos]
                sequence[mask_pos] = token_for_masking
                masked_sequences.append(sequence)
                sequence[mask_pos] = self.config.sep_token + token_for_masking if target_token.startswith(self.config.sep_token) else token_for_masking
                sentences.append(tokenizer.convert_tokens_to_string(sequence))
                labels.append(example["tokens"][mask_pos])
                has_left_list.append(has_left)
        else:
            how_many_groups = max(set(example["high_importance_groups"]))
            for i in range(1, how_many_groups + 1):
                sequence = example["tokens"].copy()
                idxs_to_mask = [idx for idx, g_label in enumerate(example["high_importance_groups"]) if g_label == i]
                sequence[idxs_to_mask[0]] = token_for_masking + self.config.sep_token if example["tokens"][idxs_to_mask[0]].startswith(self.config.sep_token) else token_for_masking
                sequence = [s for j, s in enumerate(sequence) if j not in idxs_to_mask[1:]]
                sentences.append(tokenizer.convert_tokens_to_string(sequence))
                labels.append(tokenizer.convert_tokens_to_string([example["tokens"][idx] for idx in idxs_to_mask]))
                has_left_list.append(False if i == 1 else True)
                masked_sequences.append(sequence)
        return {
            "id": example["id"],
            "masked_sequences": masked_sequences,
            "labels": labels,
            "sentences": sentences,
            "has_left_list": has_left_list,
        }

    def build_datasets(self) -> DatasetDict:
        tokenizer = self._load_tokenizer()
        dataset = self._build_example_dataset()
        dataset = dataset.map(lambda x: self._add_groups(x, tokenizer), num_proc=8)
        if self.config.exclude_single_group_sequences:
            dataset = dataset.filter(lambda x: len(x["high_importance_groups"]) > 1, num_proc=8)
        token_for_masking = self._token_for_masking(tokenizer)
        dataset = dataset.map(lambda x: self._create_masked_seq(x, tokenizer, token_for_masking), num_proc=8)
        flattened = {
            "id": [i for ex in dataset for i in [ex["id"]] * len(ex["masked_sequences"])],
            "label": [lbl for ex in dataset for lbl in ex["labels"]],
            "sentence": [sent for ex in dataset for sent in ex["sentences"]],
            "has_group_left": [hgl for ex in dataset for hgl in ex["has_left_list"]],
        }
        dataset = Dataset.from_dict(flattened)

        def extract_note_id(example):
            example["note_id"] = example["id"].split("_span_")[0]
            return example

        dataset = dataset.map(extract_note_id, num_proc=16)
        note_ids = list(set(dataset["note_id"]))
        random.Random(42).shuffle(note_ids)
        train_note_size = int(0.9 * len(note_ids))
        train_dataset = dataset.filter(lambda x: x["note_id"] in note_ids[:train_note_size], num_proc=16)
        validation_dataset = dataset.filter(lambda x: x["note_id"] in note_ids[train_note_size:], num_proc=16)
        return DatasetDict({"train": train_dataset, "validation": validation_dataset})

    def save_local(self, datasets: DatasetDict) -> tuple[str, str]:
        tokenizer = self._load_tokenizer()
        token_for_masking = self._token_for_masking(tokenizer)
        model_name = self.config.model_name.split("/")[-1]
        out_dir_path = Path(self.config.output_dir) / self.config.type_of_masking / token_for_masking
        out_dir_path.mkdir(parents=True, exist_ok=True)
        train_path = out_dir_path / f"{self.config.smoothing_method}_{model_name}_train.json"
        validation_path = out_dir_path / f"{self.config.smoothing_method}_{model_name}_validation.json"
        datasets["train"].to_json(str(train_path))
        datasets["validation"].to_json(str(validation_path))
        return str(train_path), str(validation_path)

    def push_to_hub(self, datasets: DatasetDict):
        if not self.config.hf_account_name:
            raise ValueError("hf_account_name is required to push datasets to the Hugging Face Hub")
        login_if_available(self.config.hf_token)
        tokenizer = self._load_tokenizer()
        token_for_masking = self._token_for_masking(tokenizer)
        model_name = self.config.model_name.split("/")[-1]
        repo_id = f"{self.config.hf_account_name}/DecSelfMask-{self.config.smoothing_method}_{model_name}"
        datasets["train"].push_to_hub(repo_id, split="train", token=self.config.hf_token)
        datasets["validation"].push_to_hub(repo_id, split="validation", token=self.config.hf_token)
        return repo_id
