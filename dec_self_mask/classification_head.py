from __future__ import annotations

from dataclasses import dataclass
import functools
import logging
import os
import time
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ._hf import login_if_available


@dataclass(slots=True)
class ClassificationHeadTrainerConfig:
    model_path: str
    dataset_path: str = "ferrazzipietro/crf-second-batch-item-by-item-balanced"
    target_col_name: str = "crf_item"
    label_col_name: str = "label"
    freeze_lm: bool = True
    train_max_size: int = 100_000
    val_max_size: float = 0.5
    test_max_size: int = 100_000
    train_batch_size: int = 128
    eval_batch_size: int = 64
    num_epochs: int = 20
    eval_every_percent: float = 0.10
    cache_dir: str = "/workspace/.cache"
    type_of_prompt: str = "mask"
    use_non_linearity: bool = True
    item: str = "all"
    use_same_subset_of_sft: bool = True
    max_validation_examples: int = -1
    split_name: str = "test"


class ClassificationHeadTrainer:
    def __init__(self, config: ClassificationHeadTrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.label2id = None
        self.id2label = None
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.save_dir = None
        self.best_checkpoint_path = None
        self.latest_metrics = None

    @staticmethod
    def load_dataset_with_compat_retry(dataset_path: str, cache_dir: str):
        try:
            return load_dataset(dataset_path, cache_dir=cache_dir)
        except ValueError as exc:
            if "Feature type 'List' not found" not in str(exc):
                raise
            fallback_cache_dir = os.path.join(cache_dir, "datasets_schema_compat")
            os.makedirs(fallback_cache_dir, exist_ok=True)
            return load_dataset(dataset_path, cache_dir=fallback_cache_dir, download_mode="force_redownload")

    @staticmethod
    def _collate_with_padding(batch, pad_token_id):
        from ._classifier_training.data import collate_fn

        return collate_fn(batch, pad_token_id=pad_token_id)

    def _build_model(self, num_classes: int):
        from ._classifier_training.model import LlamaLastTokenClassifier

        self.model = LlamaLastTokenClassifier(
            model_path=self.config.model_path,
            num_classes=num_classes,
            cache_dir=self.config.cache_dir,
            use_non_linearity=self.config.use_non_linearity,
            freeze_lm=self.config.freeze_lm,
        ).to(self.device)
        return self.model

    def _make_datasets(self, dataset):
        from ._classifier_training.data import ClassificationHeadDataset, ClassificationHeadDatasetInstruction

        dataset_cls = ClassificationHeadDataset if self.config.type_of_prompt == "mask" else ClassificationHeadDatasetInstruction
        if self.config.item != "all":
            for split in dataset:
                dataset[split] = dataset[split].filter(lambda x: x[self.config.target_col_name] == self.config.item.replace("____", "/"))

        all_labels = set()
        for split in dataset:
            all_labels.update(dataset[split][self.config.label_col_name])
        self.label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, cache_dir=self.config.cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        special_ids = list(self.tokenizer.added_tokens_decoder.keys())
        mask_token = self.tokenizer.added_tokens_decoder[special_ids[-2]].content
        pad_token_id = self.tokenizer.pad_token_id

        train_split = dataset["train"].select(range(min(self.config.train_max_size, len(dataset["train"]))))
        if self.config.val_max_size <= 1.0:
            val_max_size = int(len(dataset["validation"]) * self.config.val_max_size)
        else:
            val_max_size = int(self.config.val_max_size)
        validation_split = dataset["validation"].select(range(val_max_size)) if val_max_size > 1 and len(dataset["validation"]) > val_max_size else dataset["validation"]
        test_split = dataset[self.config.split_name].select(range(min(self.config.test_max_size, len(dataset[self.config.split_name])))) if self.config.split_name in dataset else validation_split

        self.train_dataset = dataset_cls(train_split, self.tokenizer, self.label2id, mask_token=mask_token, label_col_name=self.config.label_col_name)
        self.validation_dataset = dataset_cls(validation_split, self.tokenizer, self.label2id, mask_token=mask_token, label_col_name=self.config.label_col_name)
        self.test_dataset = dataset_cls(test_split, self.tokenizer, self.label2id, mask_token=mask_token, label_col_name=self.config.label_col_name)

        collate = functools.partial(self._collate_with_padding, pad_token_id=pad_token_id)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.train_batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=self.config.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate)
        return dataset_cls

    def _evaluation(self, dataloader, f1, model, calc_per_item: bool = True, verbose: bool = False):
        model.eval()
        all_val_preds = []
        all_val_labels = []
        all_val_items = []
        with torch.no_grad():
            for val_batch in dataloader:
                val_input_ids = val_batch["input_ids"].to(self.device)
                val_attention_mask = val_batch["attention_mask"].to(self.device)
                val_labels = val_batch["labels"].to(self.device)
                if self.device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda"):
                        val_logits = model(val_input_ids, val_attention_mask)
                else:
                    val_logits = model(val_input_ids, val_attention_mask)
                val_preds = val_logits.argmax(dim=-1)
                all_val_preds.extend(val_preds.cpu().tolist())
                all_val_labels.extend(val_labels.cpu().tolist())
                if calc_per_item and self.config.target_col_name in val_batch:
                    all_val_items.extend(val_batch[self.config.target_col_name])

        f1_scores_per_item = None
        if calc_per_item and all_val_items:
            f1_scores_per_item = {}
            for item in set(all_val_items):
                item_indices = [i for i, x in enumerate(all_val_items) if x == item]
                item_preds = [all_val_preds[i] for i in item_indices]
                item_labels = [all_val_labels[i] for i in item_indices]
                f1_scores_per_item[item] = f1.compute(predictions=item_preds, references=item_labels, average="macro")["f1"]

        model.train()
        f1_macro = f1.compute(predictions=all_val_preds, references=all_val_labels, average="macro")
        f1_micro = f1.compute(predictions=all_val_preds, references=all_val_labels, average="micro")
        f1_weighted = f1.compute(predictions=all_val_preds, references=all_val_labels, average="weighted")
        try:
            f1_per_class = f1.compute(predictions=all_val_preds, references=all_val_labels, average=None)
            f1_per_class = {label: score for label, score in zip(sorted(self.label2id.keys()), f1_per_class["f1"])}
        except TypeError:
            f1_per_class = None
        return f1_macro, f1_micro, f1_weighted, f1_per_class, f1_scores_per_item

    def train(self):
        login_if_available()
        dataset = self.load_dataset_with_compat_retry(self.config.dataset_path, self.config.cache_dir)
        self._make_datasets(dataset)
        self._build_model(num_classes=len(self.label2id))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.classifier.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler()
        f1 = evaluate.load("f1")
        eval_every_n_steps = max(1, int(len(self.train_loader) * self.config.eval_every_percent))
        self.save_dir = Path("data") / "d_classification_head" / "classifier_over_DecSelfMask" / ("all" if self.config.item == "all" else "one_head_per_item") / self.config.model_path.split("/")[-1] / f"item_{self.config.item}" / f"freeze_lm_{self.config.freeze_lm}" / f"epochs_{self.config.num_epochs}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        best_eval_f1_macro_so_far = 0.0

        for epoch in range(self.config.num_epochs):
            for i, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                optimizer.zero_grad()
                if self.device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda"):
                        logits = self.model(input_ids, attention_mask)
                        loss = criterion(logits, labels)
                else:
                    logits = self.model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if (i + 1) % eval_every_n_steps == 0:
                    f1_macro, f1_micro, f1_weighted, f1_per_class, f1_scores_per_item = self._evaluation(self.validation_loader, f1, self.model)
                    if f1_macro["f1"] > best_eval_f1_macro_so_far:
                        best_eval_f1_macro_so_far = f1_macro["f1"]
                    torch.save({
                        "classifier_state_dict": self.model.classifier.state_dict(),
                        "label2id": self.label2id,
                        "id2label": self.id2label,
                        "num_classes": len(self.label2id),
                        "model_path": self.config.model_path,
                        "freeze_lm": self.config.freeze_lm,
                    }, self.save_dir / "classifier_head_BEST_MODEL.pt")

        self.latest_metrics = self.evaluate()
        return self.latest_metrics

    def evaluate(self):
        if self.model is None:
            raise RuntimeError("Call train() first so the classifier model and datasets are initialized.")
        f1 = evaluate.load("f1")
        end_metrics = self._evaluation(self.test_loader, f1, self.model, verbose=True)
        ckpt_path = self.save_dir / "classifier_head_BEST_MODEL.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        from ._classifier_training.model import LlamaLastTokenClassifier

        best_model = LlamaLastTokenClassifier(
            model_path=ckpt["model_path"],
            num_classes=ckpt["num_classes"],
            cache_dir=self.config.cache_dir,
            freeze_lm=self.config.freeze_lm,
            use_non_linearity=self.config.use_non_linearity,
        ).to(self.device)
        best_model.classifier.load_state_dict(ckpt["classifier_state_dict"])
        best_metrics = self._evaluation(self.test_loader, f1, best_model, verbose=True)
        return {"end_of_training": end_metrics, "best_model": best_metrics}

    def aggregate_results(self):
        data_path = Path("data") / "d_classification_head" / "eval" / self.config.model_path.split("/")[-1] / self.config.split_name
        data_path.mkdir(parents=True, exist_ok=True)
        results_df_end = pd.read_excel(data_path / "results_table_sft_class_per_item.xlsx")
        results_df_best = pd.read_excel(data_path / "results_table_sft_class_per_item_best.xlsx")

        def compute_f1(tp, fp, fn):
            return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else -1

        def get_total_tp_fp_fn(sum_dict, categories):
            return (
                sum(sum_dict.get(f"{cat}_tp", 0) for cat in categories),
                sum(sum_dict.get(f"{cat}_fp", 0) for cat in categories),
                sum(sum_dict.get(f"{cat}_fn", 0) for cat in categories),
            )

        def get_f1_scores_by_category(sum_dict, categories):
            return {cat: compute_f1(sum_dict.get(f"{cat}_tp", 0), sum_dict.get(f"{cat}_fp", 0), sum_dict.get(f"{cat}_fn", 0)) for cat in categories}

        def get_proportion_by_category(sum_dict, categories):
            cardinality = {cat: sum_dict.get(f"{cat}_tp", 0) + sum_dict.get(f"{cat}_fp", 0) + sum_dict.get(f"{cat}_fn", 0) for cat in categories}
            total = sum(cardinality.values())
            return {cat: cardinality[cat] / total if total > 0 else 0 for cat in categories}

        outputs = {}
        for name, results_df in {"end_of_training": results_df_end, "best_macro_f1": results_df_best}.items():
            fp_columns = [col for col in results_df.columns if col.endswith("_fp")]
            fn_columns = [col for col in results_df.columns if col.endswith("_fn")]
            tp_columns = [col for col in results_df.columns if col.endswith("_tp")]
            sum_dict = {**results_df[fp_columns].sum().to_dict(), **results_df[fn_columns].sum().to_dict(), **results_df[tp_columns].sum().to_dict()}
            categories = set(c.split("_")[0] for c in fp_columns + fn_columns + tp_columns)
            total_tp, total_fp, total_fn = get_total_tp_fp_fn(sum_dict, categories)
            f1_scores = get_f1_scores_by_category(sum_dict, categories)
            proportion = get_proportion_by_category(sum_dict, categories)
            nonempty = [cat for cat in categories if (sum_dict.get(f"{cat}_tp", 0) + sum_dict.get(f"{cat}_fp", 0) + sum_dict.get(f"{cat}_fn", 0)) > 0]
            outputs[name] = {
                "model_name": self.config.model_path,
                "macro": sum(f1_scores[cat] for cat in nonempty) / len(nonempty) if nonempty else -1,
                "micro": compute_f1(total_tp, total_fp, total_fn),
                "weighted": sum(proportion[cat] * f1_scores[cat] for cat in categories),
            }
        return outputs
