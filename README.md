# DecSelfMask 

This repository contains the code and minimal artifacts required to run the DecSelfMask method

If you want to use it as a python package, go to the `import-as-python-package` branch! _Be aware that it is a beta version_


**Environment & quick links**
- **Code:** [scripts/](scripts/)
- **Configs:** [train_configs/](train_configs/)
- **Top-level run wrappers:** [a_create_dec_self_mask_sequences.sh](a_create_dec_self_mask_sequences.sh), [b_dec_self_mask_train.sh](b_dec_self_mask_train.sh), [c_sft.sh](c_sft.sh), [d_classification_head.sh](d_classification_head.sh)

**Required environment variables**
- **HF_TOKEN:** HuggingFace token with read/write access (if pushing to HF Hub)
- **WANDB_KEY:** (optional) Weights & Biases API key for experiment tracking

Create a `.env` (or export variables) before running scripts, e.g.:

```bash
export HF_TOKEN=your_hf_token
export WANDB_KEY=your_wandb_key
export PYTHONPATH="$PYTHONPATH:$PWD"
```

**Dependencies**
- Install from the provided lockfile:

```bash
pip install -r requirements.txt
```

- Notable packages (from `requirements.txt`): accelerate, transformers, datasets, bitsandbytes, wandb, torch-compatible runtime. See `requirements.txt` for pinned versions.

**Directory overview**
- **scripts/**: entry-point scripts used in experiments (training, evaluation, scoring)
- **src/**: data loaders, training utilities, model loading and evaluation functions
- **train_configs/**: YAML configs for model + runtime (DeepSpeed / accelerate)
- **data/**: example data artifacts and helper files

## Reproducing experiments (high level)

1) Prepare data and training sequences for the DecSelfMask training
- Run: [a_create_dec_self_mask_sequences.sh](a_create_dec_self_mask_sequences.sh)
- This script builds pretraining / fine-tuning sequences from your dataset and writes outputs into `data/`.

2) Train DecSelfMask pretraining runs
- Run: [b_dec_self_mask_train.sh](b_dec_self_mask_train.sh)
- Use the YAMLs in [train_configs/DecSelfMask/](train_configs/DecSelfMask/) and a runtime config from [train_configs/](train_configs/).

3) SFT finetuning on target task
- Run: [c_sft.sh](c_sft.sh)
- Choose the task-specific YAML in [train_configs/](train_configs/) (e.g. `train_configs/DecSelfMask/llama_1b.yaml`) and point `--train_data_path` to the prepared dataset.

4) Item-by-item classification head experiments
- Run: [d_classification_head.sh](d_classification_head.sh)
- This wrapper runs `scripts/train_classification_head.py` over items listed in `data/targets_for_self_masking.txt` and stores results under `data/d_classification_head/...`.

5) Evaluation & aggregation
- After training, use the evaluation scripts in `scripts/`, e.g. `scripts/train_classification_head_eval.py` and `scripts/train_classification_head_aggregate_results.py` (wrapped by [d_classification_head.sh](d_classification_head.sh)).


## Various
- See `src/training/dataset.py` for the exact loader logic and formatting conventions.

### Configs and tuning
- Runtime (DeepSpeed / accelerate) configs live in [train_configs/](train_configs/).
- Model/training hyperparameters are set in the YAML files under [train_configs/DecSelfMask/](train_configs/DecSelfMask/).

### Wandb 
By default, all runs are logged into `wandb`.



**If you use this work, please cite the paper:**

```
TO BE ADDED
```
