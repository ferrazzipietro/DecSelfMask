# DecSelfMaskervised CRF (Submission Trim)

This repository is trimmed for paper submission. It keeps only the scripts and configs needed to reproduce the main pipeline with Qwen8 on CRF, plus the baseline and item-by-item classifier runs.


create a .env file with HF_TOKEN and WANDB_KEY

## Kept Pipeline Scripts

- **B - Unannotated scoring**
  - `b_calculate_scores_unannotated_sequencial.sh`
  - `b_calculate_scores_unannotated_speedup.py`

- **CC - Build training sequences**
  - `cc_create_train_seq.py`

- **D - DecSelfMaskervised training**
  - `d_train.py`
  - `d_train_one_CLUSTER_NAME.sh`
  - `d_train_one_wrapper_CLUSTER_NAME.sh`

- **F - SFT on CRF task**
  - `f_train_task.py`
  - `f_train_task.sh`
  - `f_train_crf_task_one_wrapper_CLUSTER_NAME.sh`

- **H - Item-by-item classifier**
  - `h_train_class_over_DecSelfMask.py`
  - `h_train_one_class_per_item.sh`

- **I - Baselines**
  - `l_baselines_bert.py`
  - `l_baselines_bert.sh`

## Kept Configs

- `train_configs/DecSelfMaskervised/qwen8B_CLUSTER_NAME_v3_only_mask.yaml`
- `train_configs/crf_task/DecSelfMask_qwen8B_crf_lora.yaml`
- `train_configs/chronicity_task/DecSelfMask_qwen8B_crf_lora.yaml`
- `train_configs/admission_task/DecSelfMask_qwen8B_crf_lora_only_mask_w_item.yaml`
- `train_configs/deepspeed_*.json` and `train_configs/accelerate.json`

## Requirements

Python version: `3.14`
```bash
pip install -r requirements.txt
pip install torch==2.12 torchvision=0.27
```

Scripts expect:
- `.env` with `HF_TOKEN=...`
- `wandb` login if you want experiment tracking enabled

## Notes

- Source code remains under `src/` for training, datasets, and model loading.
- All other analysis notebooks, legacy scripts, and extra configs were removed for submission.
	- Builds QA-style dataset and publishes to HF.
- `fff_create_results_table_sft.py`
	- Post-training evaluation script (vLLM-based generation + F1 reporting).
- `fff_create_results_table_sft.sh`
	- Batch evaluation wrapper for multiple SFT models.

### Typical SFT Run Path

1. Choose task (`crf_task`, `qa_task`, `medqa_task`).
2. Select/create YAML config in `train_configs/<task_type>/`.
3. Select DeepSpeed runtime config in `train_configs/deepspeed_*.json`.
4. Launch training via `f_train_task.sh` or `f_train_task_one_wrapper.sh`.
5. Evaluate resulting model with `fff_create_results_table_sft.py`.

Minimal launch example:

```bash
bash f_train_task.sh \
	--accelerate_config_file train_configs/deepspeed_1.json \
	--custom_config_file train_configs/crf_task/DecSelfMask_qwen_crf_lora_datav3.yaml \
	--train_data_path YOUR_PATH/crf-second-batch-item-by-item-balanced \
	--train_data_split train \
	--val_data_path YOUR_PATH/crf-second-batch-item-by-item-balanced \
	--val_data_split validation \
	--task_type crf_task \
	--cache_dir /workspace/.cache
```

### Data Contracts You Should Preserve

SFT loaders expect specific columns depending on task formatting in `src/training/dataset.py`:
- CRF task expects fields like `sentence`, `crf_item`, `options`, `label`.
- QA task expects fields like `sentence`, `question`, `label`.
- MEDQA task formatting uses option fields (for example `translated_answer_opa` ... `translated_answer_opd`) and answer index.

When changing formatting logic, keep train/eval prompt conventions aligned between:
- `src/training/dataset.py`
- `fff_create_results_table_sft.py`

### High-Impact Extension Points

If you want to modify SFT behavior, these are the safest places:

- Change task prompt formatting:
	- Edit task-specific formatter functions in `src/training/dataset.py`.
- Change metrics definition:
	- Edit `src/training/evaluation_functions.py`.
- Change LoRA/quantization/model loading:
	- Edit `src/training/load_model.py` and corresponding YAML config values.
- Change optimizer/scheduler/checkpoint/eval cadence:
	- Edit the task YAML in `train_configs/<task_type>/` and HF `Seq2SeqTrainingArguments` values there.
- Change wrapper-level experiment orchestration:
	- Edit `f_train_task_one_wrapper.sh` or `f_train_crf_task_one_wrapper_CLUSTER_NAME.sh`.

### SFT Outputs and Evaluation

Training outputs are controlled by the YAML config `output_dir` and optional HF Hub target (`hub_model_id`).

For evaluation:
- Run `fff_create_results_table_sft.py` with `--model_path`.
- Results are appended to `data/fff/results_table_sft.xlsx`.

### Contribution Checklist for SFT Changes

When opening a PR that touches SFT code, include:

1. Task type affected (`crf_task`, `qa_task`, or `medqa_task`).
2. YAML config file path and any changed config fields.
3. Exact training launch command.
4. Exact evaluation command.
5. Output model/checkpoint path or Hub id.
6. Metrics summary (for example macro/micro/weighted F1 for CRF, or accuracy for QA/MEDQA).
7. Any tokenizer special-token assumptions added/changed.

If you have any questions, open an issue and reference the exact script and command you are using.