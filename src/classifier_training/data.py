from torch.utils.data import Dataset, DataLoader
import torch 




class ClassificationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, label2id, mask_token, label_col_name, max_length=512):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.mask_token = mask_token
        self.label_col_name = label_col_name

    def __len__(self):
        return len(self.data)

    def _make_text(self, example):
        pass

    def __getitem__(self, idx):
        example = self.data[idx]
        text = self._make_text(example)
        label = self.label2id[example[self.label_col_name]]

        # Tokenize WITHOUT padding — padding is done per-batch in the collate fn
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        # Keep CRF item metadata (when available) for per-item metrics at eval time.
        if "crf_item" in example:
            item["crf_item"] = example["crf_item"]
        if "mesh_target" in example:
            item["mesh_target"] = example["mesh_target"]
        return item


class CRFClassificationDataset(ClassificationDataset):
    def _make_text(self, example):
        return example["sentence"] + f"{example['crf_item']}? {self.mask_token}"

class MeshClassificationDataset(ClassificationDataset):
    def _make_text(self, example):
        return example["abstractText"] + f"{example['mesh_target']}? {self.mask_token}"

class CRFClassificationDatasetInstruction(ClassificationDataset):
    def _make_text(self, example):
        messages = [
            {"role": "system", "content": "You are an expert medical doctor. Your task is to fill the Case Report Form item with the correct answer based on the provided sentence."},
            {"role": "user", "content": f"Sentence: {example['sentence']}\n<crf_item>{example['crf_item']}</crf_item>"}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    

class TlocVsDyspneaClassificationDataset(ClassificationDataset):
     def _make_text(self, example):
        return example["text"]

class TlocVsDyspneaClassificationDatasetInstruction(ClassificationDataset):
    def _make_text(self, example):
        messages = [
            {"role": "system", "content": "You are an expert medical doctor. Your task is to determine whether the patient's symptoms are more indicative of TLOC (transient loss of consciousness) or Dyspnea (difficulty breathing) based on the provided sentence."},
            {"role": "user", "content": f"Sentence: {example['text']}"}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def collate_fn(batch, pad_token_id=0):
    """Dynamic padding: pad each batch to the longest sequence in that batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        # Right-pad
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        attention_masks.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(item["labels"])

    collated = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
    }
    if "crf_item" in batch[0]:
        collated["crf_item"] = [item["crf_item"] for item in batch]
    if "mesh_target" in batch[0]:
        collated["mesh_target"] = [item["mesh_target"] for item in batch]
    return collated
    