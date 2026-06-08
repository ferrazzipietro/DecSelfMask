import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from src.classifier_training.model import LlamaLastTokenClassifier
from src.classifier_training.data import ClassificationHeadDataset, collate_fn, ClassificationHeadDatasetInstruction

from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import login
from dotenv import dotenv_values
import os
import torch
import logging 
import time
import wandb
from argparse import ArgumentParser
import time
import evaluate
import torch.optim as optim
import functools
login(token=dotenv_values('.env')['HF_TOKEN'])


# set logging to print time and date
logger = logging.getLogger()
logger.setLevel("INFO")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Starting classifier training over DecSelfMaskervised model...")


def load_dataset_with_compat_retry(dataset_path: str, cache_dir: str):
    try:
        return load_dataset(dataset_path, cache_dir=cache_dir)
    except ValueError as exc:
        error_message = str(exc)
        if "Feature type 'List' not found" not in error_message:
            raise

        fallback_cache_dir = os.path.join(cache_dir, "datasets_schema_compat")
        os.makedirs(fallback_cache_dir, exist_ok=True)
        logger.warning(
            "Detected incompatible cached dataset schema ('List'). Retrying with force_redownload in %s",
            fallback_cache_dir,
        )
        return load_dataset(
            dataset_path,
            cache_dir=fallback_cache_dir,
            download_mode="force_redownload",
        )


def collate_with_padding(batch, pad_token_id):
    return collate_fn(batch, pad_token_id=pad_token_id)

def evaluation(dataloader_val, f1, model, calc_per_item:bool = True, verbose=False):
                model.eval()
                all_val_preds = []
                all_val_labels = []
                all_val_items = []
                with torch.no_grad():
                    for val_batch in dataloader_val:
                        if verbose: print('val_batch: ', val_batch)
                        val_input_ids = val_batch["input_ids"].to(device)
                        val_attention_mask = val_batch["attention_mask"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        with torch.amp.autocast(device_type="cuda"):
                            val_logits = model(val_input_ids, val_attention_mask)
                        val_preds = val_logits.argmax(dim=-1)
                        all_val_preds.extend(val_preds.cpu().tolist())
                        all_val_labels.extend(val_labels.cpu().tolist())
                        if calc_per_item:
                            all_val_items.extend(val_batch[args.target_col_name])
                    if verbose: print(f"all_val_preds: {all_val_preds}")
                    if verbose: print(f"all_val_labels: {all_val_labels}")
                if calc_per_item:
                    f1_scores_per_item = {}
                    unique_items = set(all_val_items)
                    for item in unique_items:
                        if verbose: print(f"Calculating F1 for item: {item}")
                        item_indices = [i for i, x in enumerate(all_val_items) if x == item]
                        item_preds = [all_val_preds[i] for i in item_indices]
                        item_labels = [all_val_labels[i] for i in item_indices]
                        f1_item = f1.compute(predictions=item_preds, references=item_labels, average='macro')
                        f1_scores_per_item[item] = f1_item['f1']
                model.train()
                f1_macro = f1.compute(predictions=all_val_preds, references=all_val_labels, average='macro')
                f1_micro = f1.compute(predictions=all_val_preds, references=all_val_labels, average='micro')
                f1_weighted = f1.compute(predictions=all_val_preds, references=all_val_labels, average='weighted')
                try:
                    f1_per_class = f1.compute(predictions=all_val_preds, references=all_val_labels, average=None)
                    if verbose: print(f"f1_per_class (before mapping to labels): {f1_per_class}")
                    f1_per_class = {
                                        label: score
                                        for label, score in zip(sorted(label2id.keys()), f1_per_class['f1'])
                                    }
                except TypeError:
                    f1_per_class = None 
                
                return f1_macro, f1_micro, f1_weighted, f1_per_class, f1_scores_per_item if calc_per_item else None

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default='meta-llama/Llama-3.2-1B-Instruct', help="Path to the pretrained model")
    parser.add_argument("--dataset_path", type=str, default="YOUR_PATH/crf-second-batch-item-by-item-balanced", help="Path to the dataset (Hugging Face format)")
    parser.add_argument("--target_col_name", type=str, required=True, help="Name of the column in the dataset that contains the labels")
    parser.add_argument("--label_col_name", type=str, default='label', help="Name of the column in the dataset that contains the labels")
    parser.add_argument("--freeze_lm", type=str, default='false', choices=['true', 'false'], help="Whether to freeze the language model weights during training")
    parser.add_argument("--train_max_size", type=int, default=100_000, help="Maximum number of examples to use from the training set")
    parser.add_argument("--val_max_size", type=float, default=0.5, help="Maximum number of examples to use from the validation set")
    parser.add_argument("--test_max_size", type=int, default=100_000, help="Maximum number of examples to use from test set")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--eval_every_percent", type=float, default=0.10, help="Evaluate every percentage of an epoch (e.g., 0.10)")
    parser.add_argument("--cache_dir", type=str, default='/workspace/.cache', help="Directory to cache datasets and models")
    parser.add_argument("--type_of_prompt", type=str, default='mask', choices=['mask', 'instruction'], help="Whether to use 'mask' or 'instruction' prompts for the classifier input")
    parser.add_argument("--use_non_linearity", type=str, default='true', choices=['true', 'false'], help="Whether to use a non-linearity (ReLU) in the classifier head. Set to 'false' for very small datasets to reduce overfitting.")
    parser.add_argument("--item", type=str, default='all', help="The item to classify. If 'all', the model will be trained to classify all items together (multiclass classification). If a specific item is provided, the model will be trained to classify that item only.")
    args = parser.parse_args()

    print('args: ', args)

    label_col_name = args.label_col_name

    if args.type_of_prompt == 'mask':
        DatasetClassificationClass = ClassificationHeadDataset
    elif args.type_of_prompt == 'instruction':
        DatasetClassificationClass = ClassificationHeadDatasetInstruction
   
        raise ValueError(f"DecSelfMaskported combination of type_of_prompt {args.type_of_prompt}")

    d = load_dataset_with_compat_retry(args.dataset_path, args.cache_dir)
    logger.info(f"Loaded dataset from {args.dataset_path}\n{d}")
    if args.item != 'all':
        logger.info(f"Filtering dataset for item: {args.item}")
        for split in d:
            d[split] = d[split].filter(lambda x: x[args.target_col_name] == args.item.replace("____", "/"))
            logger.info(f"After filtering, split '{split}' has {len(d[split])} examples.")
            if len(d[split]) == 0:
                logger.warning(f"No examples found for item '{args.item.replace("____", "/")}' in split '{split}'. Please check the dataset and item name.")
                exit(1)
    
    os.environ['HF_HOME'] = args.cache_dir   

    instruct_mode = 'instruction' if args.type_of_prompt == 'instruction' else '' 
    non_linearity_mode = '_linear' if args.use_non_linearity != 'true' else ''
    config_wandb ={
            "model_path": args.model_path,
            "freeze_lm": args.freeze_lm,
            "val_max_size": args.val_max_size,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "num_epochs": args.num_epochs,
            "eval_every_percent": args.eval_every_percent,
            "use_non_linearity": args.use_non_linearity,
        }
    print(config_wandb)

    item_appendix_name = '' if args.item == 'all' else f"_one_per_item"
    wandb.init(
        project=f"classification-head-over-DecSelfMask-{item_appendix_name}",
        name=f"{args.model_path.split('/')[-1]}_freezeLM{args.freeze_lm}_epochs_{args.num_epochs}_batchsize{args.train_batch_size}{instruct_mode}{non_linearity_mode}_{args.item}",
        config=config_wandb,
    )

    all_labels = set()
    for split in d:
        all_labels.update(d[split][label_col_name])
    num_classes = len(all_labels)
    logger.info(f"num_classes (from all splits): {num_classes}")
    logger.info(f"train_batch_size: {args.train_batch_size}, eval_batch_size: {args.eval_batch_size}, num_epochs: {args.num_epochs}, eval_every_percent: {args.eval_every_percent}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("USING NON LINEARITY: ", args.use_non_linearity == 'true')
    model = LlamaLastTokenClassifier(
        model_path=args.model_path,
        num_classes=num_classes,
        cache_dir=args.cache_dir,
        use_non_linearity=args.use_non_linearity == 'true',
        freeze_lm=args.freeze_lm == 'true' 
    )
    print(model)

    model = model.to(device)

    train = d['train'].select(range(min(args.train_max_size, len(d['train']))))
    if args.val_max_size <= 1.0:  # interpret as percentage
        val_max_size = int(len(d['validation']) * args.val_max_size)
    else:
        val_max_size = int(args.val_max_size)

    if val_max_size > 1 and len(d['validation']) > args.val_max_size:
        validation = d['validation'].select(range(val_max_size))
    else:
        validation = d['validation']
    if 'test' not in d:
        test = d['validation'].select(range(len(validation), len(validation) + min(args.test_max_size, len(d['validation'])-val_max_size)))
    else:
        test = d['test'].select(range(min(args.test_max_size, len(d['test']))))
    logger.info(f"Dataset splits: train={len(train)}, validation={len(validation)}, test={len(test)}")
    logger.info(f"Loaded dataset splits — ONE EXAMPLE: {train[0]}")

    # Build label mapping
    unique_labels = sorted(set(train[label_col_name]))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    logger.info(f"Label mapping: {label2id}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token



    pad_token_id = tokenizer.pad_token_id
    special_ids = tokenizer.added_tokens_decoder.keys()
    token_for_masking = tokenizer.added_tokens_decoder[list(special_ids)[-2]].content
    mask_token = token_for_masking

    train_dataset = DatasetClassificationClass(train, tokenizer, label2id, mask_token=mask_token, label_col_name=label_col_name)
    logger.info(f"Created train dataset with {len(train_dataset)} examples.")
    train_collate = functools.partial(collate_with_padding, pad_token_id=pad_token_id)
    dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                            num_workers=2, pin_memory=True,
                            collate_fn=train_collate)
    logger.info(f"validation: {validation}, test: {test}")
    validation_dataset = DatasetClassificationClass(validation, tokenizer, label2id, mask_token=mask_token, label_col_name=label_col_name)
    logger.info(f"Created validation dataset with {len(validation_dataset)} examples.")
    logger.info(f"First example in validation dataset: {validation_dataset[0]}")
    test_dataset = DatasetClassificationClass(test, tokenizer, label2id, mask_token=mask_token, label_col_name=label_col_name)

    eval_collate = functools.partial(collate_with_padding, pad_token_id=pad_token_id)
    dataloader_val = DataLoader(validation_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                num_workers=4, pin_memory=True,
                                collate_fn=eval_collate)
    dataloader_test = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                num_workers=4, pin_memory=True,
                                collate_fn=eval_collate)

    logger.info(f"Dataset size: {len(train_dataset)}, Batches: {len(dataloader)}")
    # Quick sanity check
    sample = train_dataset[0]
    logger.info(f"input_ids shape: {sample['input_ids'].shape}, label: {id2label[sample['labels'].item()]}")


    if args.item == 'all':
        save_dir = os.path.join("data", "d_classification_head", "classifier_over_DecSelfMask", 'all', args.model_path.split("/")[-1], f"item_{args.item}", f"freeze_lm_{args.freeze_lm}", f"epochs_{args.num_epochs}")
    else:
        save_dir = os.path.join("data", "d_classification_head", "classifier_over_DecSelfMask", 'one_head_per_item', args.model_path.split("/")[-1], f"item_{args.item}", f"freeze_lm_{args.freeze_lm}", f"epochs_{args.num_epochs}")
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler()
    model.train()
    f1 = evaluate.load("f1")
    eval_every_n_steps = max(1, int(len(dataloader) * args.eval_every_percent))  # e.g., every 48% of an epoch
    logger.info(f"Will evaluate every {eval_every_n_steps} steps (every {args.eval_every_percent*100}% of an epoch)")
    best_eval_f1_macro_so_far = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        total_loss = 0.0
        correct = 0
        total = 0

        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if i == 0 and epoch == 0:  # print the first batch of the first epoch for sanity check
                print("input_ids: ", input_ids)
                print("input_ids[0]: ", input_ids[0])
                print("input_ids[0][0]: ", input_ids[0][0])
                print("attention_mask: ", attention_mask)
                print("labels: ", labels)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            global_step = epoch * len(dataloader) + (i + 1)
            wandb.log({"train/loss": loss.item(), "train/accuracy": correct / total}, step=global_step)

            if (i + 1) % eval_every_n_steps == 0:
                start_eval_time = time.time()
                print(f"  Epoch {epoch+1}/{args.num_epochs} | Step {i+1}/{len(dataloader)} | Loss: {loss.item():.4f}"   )
                f1_macro, f1_micro, f1_weighted, f1_per_class, f1_scores_per_item = evaluation(dataloader_val, f1, model)
                end_eval_time = time.time()
                logger.info(f"  Evaluation took {end_eval_time - start_eval_time:.2f} seconds")
                logger.info(f"  Val F1 — macro: {f1_macro['f1']:.4f}, micro: {f1_micro['f1']:.4f}, weighted: {f1_weighted['f1']:.4f}\n")
                logger.info(f"  f1_scores_per_item {f1_scores_per_item}\n")
                wandb.log({
                    "val/f1_macro": f1_macro['f1'],
                    "val/f1_micro": f1_micro['f1'],
                    "val/f1_weighted": f1_weighted['f1'],
                    "val_per_class/f1_per_class": f1_per_class,
                    "val_per_item/f1_scores_per_item": f1_scores_per_item,
                    "epoch": epoch + 1,
                }, step=global_step)
                if f1_macro['f1'] > best_eval_f1_macro_so_far:
                    best_eval_f1_macro_so_far = f1_macro['f1']    
                timestamp = torch.tensor(int(time.time()))  # for unique naming
                torch.save({
                    "classifier_state_dict": model.classifier.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "num_classes": num_classes,
                    "model_path": args.model_path,
                    "freeze_lm": args.freeze_lm,
                }, os.path.join(save_dir, f"classifier_head_BEST_MODEL.pt"))
    # at the end of training, evaluate on the test set
    f1_macro, f1_micro, f1_weighted, f1_per_class, f1_scores_per_item = evaluation(dataloader_test, f1, model, verbose=True)
    wandb.log({"test/f1_macro_end_of_training": f1_macro['f1'], "test/f1_micro_end_of_training": f1_micro['f1'], "test/f1_weighted_end_of_training": f1_weighted['f1'], "test/f1_per_class_end_of_training": f1_per_class, "test/f1_scores_per_item_end_of_training": f1_scores_per_item}, step=global_step)
    # do the same on the best model according to macro F1
    ckpt_path = os.path.join(save_dir, f"classifier_head_BEST_MODEL.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_path = ckpt["model_path"]
    num_classes = ckpt["num_classes"]
    label2id = ckpt["label2id"]
    id2label = ckpt["id2label"]

    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"num_classes={num_classes}, label2id={label2id}")

    # --- Build model & load classifier weights ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model = LlamaLastTokenClassifier(
        model_path=model_path,
        num_classes=num_classes,
        cache_dir=args.cache_dir,
        freeze_lm=args.freeze_lm == 'true'
    )
    best_model.classifier.load_state_dict(ckpt["classifier_state_dict"])
    best_model = best_model.to(device)
    f1_macro, f1_micro, f1_weighted, f1_per_class, f1_scores_per_item = evaluation(dataloader_test, f1, best_model, verbose=True)
    wandb.log({"test/f1_macro_end_of_training": f1_macro['f1'], "test/f1_micro_end_of_training": f1_micro['f1'], "test/f1_weighted_end_of_training": f1_weighted['f1'], "test/f1_per_class_end_of_training": f1_per_class, "test/f1_scores_per_item_end_of_training": f1_scores_per_item}, step=global_step)
    

    wandb.finish()


    # Save classifier head weights + config

    timestamp = torch.tensor(int(time.time()))  # for unique naming
    torch.save({
        "classifier_state_dict": model.classifier.state_dict(),
        "label2id": label2id,
        "id2label": id2label,
        "num_classes": num_classes,
        "model_path": args.model_path,
        "freeze_lm": args.freeze_lm,
    }, os.path.join(save_dir, f"classifier_head_{timestamp.item()}.pt"))

    # Save full model (LM + classifier) if LM was also trained
    if not args.freeze_lm:
        model.lm.save_pretrained(os.path.join(save_dir, "lm"))
        tokenizer.save_pretrained(os.path.join(save_dir, "lm"))

    print(f"Model saved to {save_dir}/classifier_head_{timestamp.item()}.pt")
        