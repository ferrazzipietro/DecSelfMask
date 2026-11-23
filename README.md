# DecSelfMask 

This repository contains the code and minimal artifacts required to run the DecSelfMask method.

**IMPORTANT NOTE**: to obtain the data to replicate the experiments in the paper, please email `pferrazzi@fbk.eu`.

If you want to use it as a python package, go to the `use-as-python-package` branch!


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
@inproceedings{ferrazzi-etal-2026-decselfmask,
    title = "{D}ec{S}elf{M}ask: Leveraging Unlabeled Text via Self-Relevance-Guided Masking for Decoder-Only Classification",
    author = "Ferrazzi, Pietro  and
      Merler, Matteo  and
      Bonetta, Giovanni  and
      Lavelli, Alberto  and
      Magnini, Bernardo",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Junyi Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.1372/",
    doi = "10.18653/2026.emnlp-main.1372",
    pages = "31311--31327",
    ISBN = "979-8-89176-459-0",
    abstract = "Classification tasks require annotated data, which can often be expensive, time-consuming, or even unfeasible to collect.This is the case of the medical domain, where large datasets often have few annotated examples. To address this, we propose DecSelfMask (Decoder Self-learning by Masking), an approach to enhance decoder-only performance on classification tasks. We build on common self-learning approaches by leveraging a model to create training examples from unlabeled data, and propose a novel relevance-guided masking strategy.We use relevance attribution methods to determine what portions of unannotated texts are relevant for a task. We then create self-supervised training examples by masking out those portions, training the model to reconstruct them via next-token-prediction. We hypothesize that those examples convey knowledge about the structure and semantics of unannotated data that can be useful for downstream performance. We test our approach on 136 tasks from a collection of 1.9M clinical notes from an Italian hospital. We quantify DecSelfMask{'}s impact on downstream tasks on 5 models of different scales and families, including a probing analysis. Experiments show consistent gains, outperforming the base models (+9.1 points in Macro F1), continual pretraining (+6.3), synthetic label generation (+12.5), as well as common baselines. The results show that relevance attribution can serve as a powerful source of supervision for decoder learning from unlabeled data, beyond existing methods."
}
```
