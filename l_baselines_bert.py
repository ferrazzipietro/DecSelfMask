import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from src.classifier_training.data import collate_fn, CRFClassificationDataset, CRFClassificationDatasetInstruction, TlocVsDyspneaClassificationDataset, TlocVsDyspneaClassificationDatasetInstruction

from datasets import load_dataset
from huggingface_hub import login
from dotenv import dotenv_values
import os
import logging 
import time
import wandb
from argparse import ArgumentParser
import evaluate
import torch.optim as optim
login(token=dotenv_values('.env')['HF_TOKEN'])

# set logging
logger = logging.getLogger()
logger.setLevel("INFO")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def evaluation(dataloader_val, f1, model):
    model.eval()
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for val_batch in dataloader_val:
            val_input_ids = val_batch["input_ids"].to(device)
            val_attention_mask = val_batch["attention_mask"].to(device)
            val_labels = val_batch["labels"].to(device)
            with torch.amp.autocast(device_type="cuda"):
                val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask)
            val_preds = val_outputs.logits.argmax(dim=-1)
            all_val_preds.extend(val_preds.cpu().tolist())
            all_val_labels.extend(val_labels.cpu().tolist())
    model.train()
    f_m = f1.compute(predictions=all_val_preds, references=all_val_labels, average='macro')['f1']
    f_mic = f1.compute(predictions=all_val_preds, references=all_val_labels, average='micro')['f1']
    f_w = f1.compute(predictions=all_val_preds, references=all_val_labels, average='weighted')['f1']
    f1_per_class = f1.compute(predictions=all_val_preds, references=all_val_labels, average=None)
    f1_per_class = {
                            label: score
                            for label, score in zip(sorted(label2id.keys()), f1_per_class['f1'])
                        }
    logger.info(f"Epoch {epoch+1} Step {i+1} | Val F1 — macro: {f_m:.4f}, micro: {f_mic:.4f}, weighted: {f_w:.4f}")
    return f_m, f_mic, f_w, f1_per_class

class SimpleClassificationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, label2id, label_col_name, task_type, max_length=512):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.label_col_name = label_col_name
        self.task_type = task_type

    def __len__(self):
        return len(self.data)

    def _make_text(self, example):
        if self.task_type == 'crf_task':
            return example["sentence"] + " [SEP] " + example["crf_item"]
        else:
            return example["text"]
        return ""

    def __getitem__(self, idx):
        example = self.data[idx]
        text = self._make_text(example)
        label = self.label2id[example[self.label_col_name]]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default='google-bert/bert-base-uncased', help="Path to the pretrained model")
    parser.add_argument("--dataset_path", type=str, default="YOUR_PATH/crf-second-batch-item-by-item-balanced", help="Path to the dataset")
    parser.add_argument("--freeze_lm", action="store_true", help="Whether to freeze the language model weights (bert encoder) during training")
    parser.add_argument("--val_max_size", type=int, default=1000, help="Maximum number of examples to use from validation set")
    parser.add_argument("--test_max_size", type=int, default=100_000, help="Maximum number of examples to use from test set")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--eval_every_percent", type=float, default=0.10, help="Evaluate every X% of an epoch")
    parser.add_argument("--cache_dir", type=str, default='/workspace/.cache', help="Directory to cache datasets and models")
    parser.add_argument("--task_type",  choices=["crf_task", "tlocvsdyspnea_task", "chronicity_task"], type=str, required=True, help='Type of task to train on')
    parser.add_argument("--crf_item", type=str, default='all', help="The CRF item to classify.")
    args = parser.parse_args()

    labels_mapper = {
        'crf_task': 'label',
        'tlocvsdyspnea_task': 'condition',
        'chronicity_task': 'condition',
    }
    label_col_name = labels_mapper[args.task_type]

    d = load_dataset(args.dataset_path, cache_dir=args.cache_dir)
    if args.task_type=='crf_task' and args.crf_item != 'all':
        for split in d:
            d[split] = d[split].filter(lambda x: x['crf_item'] == args.crf_item.replace("____", "/"))

    os.environ['HF_HOME'] = args.cache_dir   

    config_wandb ={
        "model_path": args.model_path,
        "freeze_lm": args.freeze_lm,
        "val_max_size": args.val_max_size,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_epochs": args.num_epochs,
        "eval_every_percent": args.eval_every_percent,
    }

    crf_item_appendix_name = '' if args.crf_item == 'all' else f"_one_per_item"
    wandb.init(
        project=f"classifier-bert-baseline-{args.task_type}-{crf_item_appendix_name}",
        name=f"{args.model_path.split('/')[-1]}_freezeLM{args.freeze_lm}_epochs_{args.num_epochs}_batchsize{args.train_batch_size}_{args.crf_item}",
        config=config_wandb,
    )

    all_labels = set()
    for split in d:
        all_labels.update(d[split][label_col_name])
    num_classes = len(all_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = d['train']
    validation = d['validation'].select(range(min(args.val_max_size, len(d['validation'])))) if args.val_max_size > 0 else d['validation']
    test = d['validation'].select(range(len(validation), len(validation) + min(args.test_max_size, len(d['validation'])-args.val_max_size)))

    unique_labels = sorted(set(train[label_col_name]))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        cache_dir=args.cache_dir
    )
    
    if args.freeze_lm:
        for param in model.base_model.parameters():
            param.requires_grad = False
    
    model = model.to(device)

    train_dataset = SimpleClassificationDataset(train, tokenizer, label2id, label_col_name, args.task_type)
    dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id))
    
    validation_dataset = SimpleClassificationDataset(validation, tokenizer, label2id, label_col_name, args.task_type)
    test_dataset = SimpleClassificationDataset(test, tokenizer, label2id, label_col_name, args.task_type)
    dataloader_val = DataLoader(validation_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                num_workers=4, pin_memory=True, collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id))
    dataloader_test = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True, collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id))

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    f1 = evaluate.load("f1")
    eval_every_n_steps = max(1, int(len(dataloader) * args.eval_every_percent))

    best_eval_f1_macro_so_far = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()
            total = labels.size(0)

            global_step = epoch * len(dataloader) + (i + 1)
            wandb.log({"train/loss": loss.item(), "train/accuracy": correct / total}, step=global_step)

            if (i + 1) % eval_every_n_steps == 0:
                f_m, f_mic, f_w, f1_per_class = evaluation(dataloader_val, f1, model)
                wandb.log({"val/f1_macro": f_m, "val/f1_micro": f_mic, "val/f1_weighted": f_w, "val/f1_per_class": f1_per_class, "epoch": epoch + 1}, step=global_step)
                if f_m > best_eval_f1_macro_so_far:
                    best_eval_f1_macro_so_far = f_m
                    model.save_pretrained(os.path.join("best_models", f"best_macro_f1"))
    # at the end of training, evaluate on the test set
    f_m, f_mic, f_w, f1_per_class = evaluation(dataloader_test, f1, model)
    wandb.log({"test/f1_macro_end_of_training": f_m, "test/f1_micro_end_of_training": f_mic, "test/f1_weighted_end_of_training": f_w, "test/f1_per_class_end_of_training": f1_per_class}, step=global_step)
    # do the same on the best model according to macro F1
    best_model = AutoModelForSequenceClassification.from_pretrained(os.path.join("best_models", f"best_macro_f1")).to(device)
    f_m, f_mic, f_w, f1_per_class = evaluation(dataloader_test, f1, best_model)
    wandb.log({"test/f1_macro_best_model": f_m, "test/f1_micro_best_model": f_mic, "test/f1_weighted_best_model": f_w, "test/f1_per_class_best_model": f1_per_class}, step=global_step)

    wandb.finish()
    
    save_dir = os.path.join("trainer_output", "bert_baseline", args.task_type, args.model_path.split("/")[-1], f"freeze_lm_{args.freeze_lm}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Model saved to {save_dir}")
