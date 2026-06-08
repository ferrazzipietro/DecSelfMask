import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import login
from dotenv import dotenv_values
import functools
import os
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from logging import basicConfig, getLogger
basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level='INFO')
logger = getLogger(__name__)
from src.classifier_training.model import LlamaLastTokenClassifier
from src.classifier_training.data import ClassificationHeadDataset, collate_fn

login(token=dotenv_values('.env')['HF_TOKEN'])




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument('--USE_SAME_SUBSET_OF_SFT', type=str, default='false', help='Whether to use the same subset of SFT for all models (the one with the shortest sentences)')
    parser.add_argument('--MAX_VALIDATION_EXAMPLES', type=int, default=-1, help='Maximum number of validation examples to use. If -1, use all examples.')
    parser.add_argument('--dataset_path', type=str, default="ferrazzipietro/crf-second-batch-item-by-item-balanced", help='Path to the dataset to use for evaluation')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for evaluation')
    parser.add_argument('--label_col_name', type=str, default="label", help='Name of the column containing the labels in the dataset')
    parser.add_argument('--item_col_name', type=str, required=True, help='Column name to use for filtering the dataset (e.g., crf_item)')
    parser.add_argument('--cache_dir', type=str, default="/data01/pferrazzi/.cache", help='Cache directory for loading the dataset and the model')
    parser.add_argument('--split_name', type=str, default="validation", help='Split name to use for evaluation')
    args = parser.parse_args()

    heads_in_dir = sorted(os.listdir(args.model_path), reverse=True)
    args.USE_SAME_SUBSET_OF_SFT = args.USE_SAME_SUBSET_OF_SFT == 'true'

    label_col_name = args.label_col_name

    ClassificationDataset = ClassificationHeadDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = args.cache_dir

    model = None
    tokenizer = None
    token_for_masking = None
    current_model_path = None
    data = load_dataset(args.dataset_path, cache_dir=cache_dir)
    logger.info(f"Loaded dataset from {args.dataset_path}\n{data}")

    for head in heads_in_dir:
        ckpt_path = os.path.join(args.model_path, head)
        print('\n\n\n\n==============================')
        print(f"Processing head: {ckpt_path}")
        print('==============================\n\n\n\n')
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_path = ckpt["model_path"]
        num_classes = ckpt["num_classes"]
        label2id = ckpt["label2id"]
        id2label = ckpt["id2label"]

        if model is None or model_path != current_model_path:
            logger.info(f"Loading base model {model_path}")
            model = LlamaLastTokenClassifier(
                model_path=model_path,
                num_classes=num_classes,
                cache_dir=cache_dir,
                freeze_lm=True,
            )
            model = model.to(device)
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            special_ids = list(tokenizer.added_tokens_decoder.keys())
            token_for_masking = tokenizer.added_tokens_decoder[special_ids[-2]].content
            current_model_path = model_path

        model.classifier.load_state_dict(ckpt["classifier_state_dict"])

        logger.info(f"Loaded checkpoint from {ckpt_path}")
        logger.info(f"num_classes={num_classes}, label2id={label2id}")

        item = ckpt_path.split('/')[-4].replace(f'item_', '').replace('____', ('/'))
        d = data.filter(lambda x: x[args.item_col_name] == item)
        logger.info(f"Filtered dataset for item {item}, resulting in {d}")
        test = d[args.split_name]
        logger.info(f"Using split '{args.split_name}' with {len(test)} examples for evaluation")
        # if args.USE_SAME_SUBSET_OF_SFT and args.MAX_VALIDATION_EXAMPLES == -1:
        #     raise ValueError("If you want to use the same subset of SFT, you need to set MAX_VALIDATION_EXAMPLES to -1")
        if args.USE_SAME_SUBSET_OF_SFT:
            test = test.map(lambda x: {"len": len(x["sentence"])})
            third_quartile = np.percentile(test['len'], 60)
            test = test.filter(lambda x: x['len'] <= third_quartile)
        if args.MAX_VALIDATION_EXAMPLES != -1:
            max_valid = min(args.MAX_VALIDATION_EXAMPLES, len(test))
            test = test.select(range(max_valid))

        val_dataset = ClassificationDataset(
            test,
            tokenizer,
            label2id,
            token_for_masking,
            label_col_name=label_col_name,
        )
        pad_token_id = tokenizer.pad_token_id
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=functools.partial(collate_fn, pad_token_id=pad_token_id),
        )

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                if device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda"):
                        logits = model(input_ids, attention_mask)
                else:
                    logits = model(input_ids, attention_mask)

                preds = logits.argmax(dim=-1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        possible_labels = set(label2id.values())
        keys = [[f"{id2label[label]}_tp", f"{id2label[label]}_fp", f"{id2label[label]}_fn"] for label in possible_labels]
        keys = [item for sublist in keys for item in sublist]
        res_dict = {l: 0 for l in keys}
        for label in possible_labels:
            tp = sum((p == label and l == label) for p, l in zip(all_preds, all_labels))
            fp = sum((p == label and l != label) for p, l in zip(all_preds, all_labels))
            fn = sum((p != label and l == label) for p, l in zip(all_preds, all_labels))
            res_dict[f"{id2label[label]}_tp"] = tp
            res_dict[f"{id2label[label]}_fp"] = fp
            res_dict[f"{id2label[label]}_fn"] = fn

        model_name = ckpt_path.split('/')[-5].lower().replace(f'task-', '').replace('-merged', '').replace('all', '')
        partial = {
            "model_path": ckpt_path,
            "model_name": model_name,
            f"item": item,
            "unsup": 'unsup' in ckpt_path.split('/')[-5].lower(),
            "MAX_VALIDATION_EXAMPLES": args.MAX_VALIDATION_EXAMPLES,
            "USE_SAME_SUBSET_OF_SFT": args.USE_SAME_SUBSET_OF_SFT,
            "date_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        out = {**partial, **res_dict}
        logger.info(f'RESULTS: {out}')
        model_name_uppercase = ckpt_path.split('/')[-5].replace(f'Task-', '').replace('-merged', '').replace('all', '')
        is_best = "best_model" in head.lower() or "best" in ckpt_path.lower()
        results_filename = "results_table_sft_class_per_item_best.xlsx" if is_best else "results_table_sft_class_per_item.xlsx"
        data_path = f"data/d_classification_head/eval/{model_name_uppercase}/{args.split_name}/{results_filename}"
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        results_df = pd.DataFrame([out])
        if os.path.exists(data_path):
            existing_data_df = pd.read_excel(data_path)
            if item in existing_data_df[args.item_col_name].values:
                existing_data_df = existing_data_df[existing_data_df[args.item_col_name] != item]
            results_df = pd.concat([existing_data_df, results_df], ignore_index=True)
            results_df.to_excel(data_path, index=False)
        else:
            results_df.to_excel(data_path, index=False)