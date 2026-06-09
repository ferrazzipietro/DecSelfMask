# DecSelfMask — Self-Contained Package

This repository contains the class-first, self-contained Python package for the DecSelfMask method. The old script wrappers and the old `src/` training stack have been folded into the package so the main workflow can now be used directly from Python.

**Quick links**
- **Package:** [dec_self_mask/](dec_self_mask/)
- **Example notebook:** [dec_self_mask_example.ipynb](dec_self_mask_example.ipynb)
- **Configs:** [train_configs/](train_configs/)
- **Data artifacts:** [data/](data/)

**Required environment variables**
- **HF_TOKEN:** Hugging Face token with read/write access if you push to the Hub.
- **WANDB_KEY:** optional Weights & Biases API key for experiment tracking.

Create a `.env` file or export the variables before running any step:

```bash
export HF_TOKEN=your_hf_token
export WANDB_KEY=your_wandb_key
```

**Dependencies**
- Install the package in editable mode:

```bash
pip install -e .
```

- The project uses standard Hugging Face and PyTorch tooling. See [requirements.txt](requirements.txt) for the pinned environment used by the repository.

## Package Overview

The package exports the class-first API from [dec_self_mask/__init__.py](dec_self_mask/__init__.py).

- `RelevanceCalculator` computes token relevancy from a Hugging Face dataset or local JSON file.
- `DecSelfMaskSequencesMaker` turns relevancy results into DecSelfMask training sequences.
- `DecSelfMaskTrainer` runs DecSelfMask training and exposes `train()` and `evaluate()`.
- `SFTTrainer` runs the supervised fine-tuning stage and exposes `train()` and `evaluate()`.
- `ClassificationHeadTrainer` trains, evaluates, and aggregates the classification-head experiments.

The package also includes the reusable internal modules required by those classes, so it no longer depends on the deleted `src/` training stack.

## End-to-End Workflow

The full pipeline is:

1. Calculate relevance scores for the source dataset.
2. Build masked DecSelfMask sequences from the relevancy output.
3. Train the DecSelfMask model.
4. Fine-tune the model with SFT.
5. Train and evaluate a classification head on top of the trained model.

The example notebook [dec_self_mask_example.ipynb](dec_self_mask_example.ipynb) shows each of these steps in order.

## Example Usage

The notebook is the best starting point, but the same flow can be used directly in Python:

```python
from dec_self_mask import (
    RelevanceCalculator,
    RelevanceCalculatorConfig,
    DecSelfMaskSequencesMaker,
    SequenceMakerConfig,
    DecSelfMaskTrainingArguments,
    DecSelfMaskTrainer,
    SFTTrainer,
    SFTTrainerConfig,
    ClassificationHeadTrainer,
    ClassificationHeadTrainerConfig,
)

relevance = RelevanceCalculator(RelevanceCalculatorConfig(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    data_path="wikimedia/wikipedia",
    data_config="20231101.es",
    id_column_name="id",
    text_column_name="text",
))

relevance_output = relevance.calculate()

sequences = DecSelfMaskSequencesMaker(SequenceMakerConfig(
    input_path="data/a_attention_relevancy_unannotated/wikipedia/Llama-3.2-1B-Instruct/combined_mid.json",
    hf_account_name="ferrazzipietro",
))
datasets = sequences.build_datasets()

trainer = DecSelfMaskTrainer(DecSelfMaskTrainingArguments(
    custom_config_file="train_configs/DecSelfMask/llama_1b.yaml",
))
trainer.train()
trainer.evaluate()

sft = SFTTrainer(SFTTrainerConfig(
    custom_config_file="train_configs/sft/llama_1b.yaml",
))
sft.train()
sft.evaluate()

classifier = ClassificationHeadTrainer(ClassificationHeadTrainerConfig(
    model_path="ferrazzipietro/DecSelfMask-Llama-3.2-1B-Instruct",
    item="all",
))
classifier.train()
classifier.evaluate()
classifier.aggregate_results()
```

## Configuration

- Runtime and model configs live in [train_configs/](train_configs/).
- The DecSelfMask and SFT model YAMLs live under [train_configs/DecSelfMask/](train_configs/DecSelfMask/) and [train_configs/sft/](train_configs/sft/).
- Default dataset and formatting logic are embedded in the package classes, so you do not need the old shell scripts to run the workflow.

## Notes

- The notebook and package code are designed to be run from the repository root with `pip install -e .`.
- By default, experiment tracking uses `wandb` when configured in the training arguments.
- If you push to the Hugging Face Hub, set `HF_TOKEN` in the environment.

## Contact & Licensing

- **Author / contact:** Pietro Ferrazzipietro (see repository metadata)
- **License:** check the repository LICENSE or contact the authors for reuse permissions

**If you use this work, please cite the paper:**

- **Paper:** Ferrazzipietro et al., DecSelfMask: ... (full citation here)
- **BibTeX:** to be added