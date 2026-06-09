from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hf import login_if_available


@dataclass(slots=True)
class RelevanceCalculatorConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    data_path: str = "Pretrain- -YOUR_PATH_ORG/ClinicalWhole"
    data_config: str = "default"
    data_split: str = "train"
    id_column_name: str = "id"
    text_column_name: str = "text"
    max_text_length: int = -1
    file_with_targets: str = "data/targets_for_self_masking.txt"
    targets: list[str] | None = None
    start_from_note: int = 0
    end_at_note: int = -1
    cache_dir: str = "/YOUR_PATH/.cache/"
    use_which_token: str = "mid"
    keep_n_sequences_per_note: int = -1
    path_save: str = "/YOUR_PATH/unannotated_crf/data/"
    batch_size_lrp: int = 32
    hf_token: str | None = None


class RelevanceCalculator:
    def __init__(self, config: RelevanceCalculatorConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.scorer = None

    def _load_tokenizer_and_model(self):
        if self.tokenizer is not None and self.model is not None:
            return self.tokenizer, self.model

        login_if_available(self.config.hf_token)
        cache_dir = self.config.cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=cache_dir,
            token=self.config.hf_token or os.getenv("HF_TOKEN"),
        )
        model_output_attentions = False
        if torch.cuda.device_count() > 1:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                output_attentions=model_output_attentions,
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir,
                token=self.config.hf_token or os.getenv("HF_TOKEN"),
                tp_plan="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                output_attentions=model_output_attentions,
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir,
                device_map="auto",
                token=self.config.hf_token or os.getenv("HF_TOKEN"),
            )

        from .score_relevancy import AttnLRPScorer
        from transformers.models.llama import modeling_llama

        self.scorer = AttnLRPScorer(modeling_type=modeling_llama, model=self.model)
        return self.tokenizer, self.model

    def _load_dataset(self) -> Dataset:
        cache_dir = self.config.cache_dir
        if os.path.isfile(self.config.data_path):
            data = Dataset.from_json(self.config.data_path)
        else:
            data = load_dataset(
                self.config.data_path,
                self.config.data_config,
                split=self.config.data_split,
                cache_dir=cache_dir,
            )
        end_idx = self.config.end_at_note if self.config.end_at_note != -1 else len(data)
        return data.select(range(self.config.start_from_note, end_idx))

    def _load_targets(self) -> list[str]:
        if self.targets is not None:
            return self.targets
        with open(self.config.file_with_targets, "r") as handle:
            self.targets = handle.read().splitlines()
        return self.targets

    @staticmethod
    def _find_tokens_position(prompt_text: str, label_annotation: str, tokenizer) -> tuple[int, int]:
        prompt_tokens = tokenizer.tokenize(prompt_text)
        if " - " in label_annotation:
            label_annotation = label_annotation.split(" - ")[-1]
        label_tokens = tokenizer.tokenize(label_annotation)
        out = (-1, -1)
        for i in range(len(prompt_tokens) - len(label_tokens) + 1):
            if prompt_tokens[i : i + len(label_tokens)] == label_tokens:
                out = (i, i + len(label_tokens) - 1)
        if out == (-1, -1):
            label_annotation = " " + label_annotation
            label_tokens = tokenizer.tokenize(label_annotation)
            for i in range(len(prompt_tokens) - len(label_tokens) + 1):
                if prompt_tokens[i : i + len(label_tokens)] == label_tokens:
                    out = (i, i + len(label_tokens) - 1)
        if out == (-1, -1):
            logging.warning(
                "Could not find label tokens in prompt for annotation %s", label_annotation
            )
        return out

    @staticmethod
    def _is_cuda_oom_error(error: RuntimeError) -> bool:
        message = str(error).lower()
        return (
            "out of memory" in message
            or "cuda error: out of memory" in message
            or "cublas_status_alloc_failed" in message
            or "cuda out of memory" in message
            or "cudacachingallocator.cpp" in message
            or "nvml_success == r" in message
            or ("cuda error" in message and "alloc" in message)
        )

    @staticmethod
    def _release_cuda_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    def _flush_pending_jobs(self, jobs, tokenizer, out_dict):
        if not jobs:
            return

        cursor = 0
        while cursor < len(jobs):
            remaining = len(jobs) - cursor
            current_try_batch_size = min(self.config.batch_size_lrp, remaining)

            while current_try_batch_size >= 1:
                try:
                    batch_jobs = jobs[cursor : cursor + current_try_batch_size]
                    batch_prompts = [j["prompt"] for j in batch_jobs]
                    batch_positions = [j["token_position"] for j in batch_jobs]
                    batch_relevancy = self.scorer.get_relevancy_batch(
                        prompts=batch_prompts,
                        tokenizer=tokenizer,
                        token_positions=batch_positions,
                    )
                    for job, relevancy_lrp in zip(batch_jobs, batch_relevancy):
                        out_dict[job["key"]] = {
                            "prompt": job["prompt"].prompt,
                            "relevancy_attn_lrp": relevancy_lrp,
                            "token_position": job["token_position"],
                        }

                    cursor += current_try_batch_size
                    break
                except RuntimeError as error:
                    if not self._is_cuda_oom_error(error):
                        raise
                    self._release_cuda_memory()
                    if current_try_batch_size == 1:
                        failed_job = jobs[cursor]
                        logging.error(
                            "CUDA memory failure persisted at batch_size=1. Skipping job %s. Error: %s",
                            failed_job["key"],
                            error,
                        )
                        cursor += 1
                        break
                    current_try_batch_size = max(1, current_try_batch_size // 2)

    def calculate(self) -> dict[str, Any]:
        from .prompt import PromptNoGT

        tokenizer, _ = self._load_tokenizer_and_model()
        data = self._load_dataset()
        target_items = self._load_targets()

        out: dict[str, Any] = {}
        pending_jobs = []

        for note_pos, note in enumerate(data):
            note_targets = target_items
            if self.config.keep_n_sequences_per_note > 0 and len(note_targets) > self.config.keep_n_sequences_per_note:
                note_targets = random.sample(note_targets, self.config.keep_n_sequences_per_note)

            note_text = note[self.config.text_column_name][: self.config.max_text_length]
            for target in note_targets:
                prompt = PromptNoGT(note_text=note_text, target_item=target, tokenizer=tokenizer)
                start_answer_token_pos, end_answer_token_pos = self._find_tokens_position(
                    prompt.prompt,
                    target,
                    tokenizer,
                )
                if self.config.use_which_token == "first":
                    token_position = start_answer_token_pos
                elif self.config.use_which_token == "last":
                    token_position = end_answer_token_pos
                elif self.config.use_which_token == "mid":
                    token_position = int((end_answer_token_pos + start_answer_token_pos) / 2)
                elif self.config.use_which_token == "random":
                    token_position = torch.randint(0, end_answer_token_pos + 1, (1,)).item()
                else:
                    raise ValueError("use_which_token must be one between 'first', 'last', 'mid', or 'random'")

                pending_jobs.append(
                    {
                        "key": f"note_{note_pos}_span_{target}",
                        "prompt": prompt,
                        "token_position": token_position,
                    }
                )
                if len(pending_jobs) >= self.config.batch_size_lrp:
                    self._flush_pending_jobs(pending_jobs[: self.config.batch_size_lrp], tokenizer, out)
                    pending_jobs = pending_jobs[self.config.batch_size_lrp :]

        if pending_jobs:
            self._flush_pending_jobs(pending_jobs, tokenizer, out)

        save_path = Path(self.config.path_save) / "a_attention_relevancy_unannotated" / Path(self.config.data_path).name / Path(self.config.model_name).name
        save_path.mkdir(parents=True, exist_ok=True)
        txt_path = save_path / f"attention_relevancy_results_{self.config.start_from_note}_{self.config.end_at_note}_{self.config.use_which_token}.txt"
        json_path = save_path / f"attention_relevancy_results_{self.config.start_from_note}_{self.config.end_at_note}_{self.config.use_which_token}.json"
        combined_path = save_path / f"combined_{self.config.use_which_token}.json"

        with open(txt_path, "w") as handle:
            handle.write(str(out))
        with open(json_path, "w") as handle:
            json.dump(out, handle)

        combined_out: dict[str, Any] = {}
        for json_file in save_path.iterdir():
            if json_file.name.startswith("attention_relevancy_results_") and json_file.suffix == ".json":
                with open(json_file, "r") as handle:
                    part_out = json.load(handle)
                    name = json_file.stem.replace("attention_relevancy_results_", "")
                    combined_out.update({f"{k}_{name}": v for k, v in part_out.items()})
        with open(combined_path, "w") as handle:
            json.dump(combined_out, handle)
        return combined_out
