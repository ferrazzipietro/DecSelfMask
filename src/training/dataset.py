from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, List, Optional, Union
import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from datasets import Dataset as HFDataset
from datasets import load_dataset
from dotenv import dotenv_values
from huggingface_hub import login
login(dotenv_values('.env')['HF_TOKEN'])


def _format_instruct_conversation(example: Dict[str, str], system_prompt: str, return_without_label: bool=False) -> List[Dict[str, str]]:
    if return_without_label:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["sentence"]},
        ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["sentence"]},
        {"role": "assistant", "content": example["label"]},
    ]

def _format_instruct_conversation_sft_task(
        example: Dict[str, str], 
        system_prompt: str, 
        mask_token :str, 
        return_without_label: bool=False, 
        tag_token_start: str="",
        item_column_name: str = "sft_item",
        options_column_name: str = "options"
        ) -> List[Dict[str, str]]:
    if return_without_label:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["sentence"] + f"\n{tag_token_start}{example[item_column_name]}? {mask_token} </{tag_token_start}> <options>{example[options_column_name]}</options>"},
        ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["sentence"] + f"\n{tag_token_start}{example[item_column_name]}? {mask_token} </{tag_token_start}> <options>{example[options_column_name]}</options>"},
        {"role": "assistant", "content": example["label"]},
    ]


def _format_and_tokenize_instruct_batch(
    tokenizer: PreTrainedTokenizerBase,
    batch: Dict[str, List[str]],
    format_convesation_fn,
    system_prompt: str = "Fill in the masked word in the following sentence.",
    has_printed: bool = False,
    calculate_loss_on_prompt: bool = False,
) -> Dict[str, List[List[int]]]:
    # Build formatted strings first, then tokenize in a single batched call
    first_col_name = list(batch.keys())[0]
    batch = [{k: v[i] for k, v in batch.items()} for i in range(len(batch[first_col_name]))]
    if not has_printed:
        print('batch:', batch[0])
    conversations = [
        format_convesation_fn(example)
        for example in batch
    ]
    conversations_no_labels = [
        format_convesation_fn(example, return_without_label=True)
        for example in batch
    ]
    formatted_inputs = [
        tokenizer.apply_chat_template(conversation=conv, tokenize=False) for conv in conversations
    ]
    formatted_inputs_no_labels = [
        tokenizer.apply_chat_template(conversation=conv, tokenize=False, add_generation_prompt=True) for conv in conversations_no_labels
    ]
    if not has_printed:
        print("Formatted input example:\n", formatted_inputs[0], '\n________________________')
    # print(formatted_inputs_no_labels[0])

    # Ensure EOS at the end of each assistant reply
    eos = tokenizer.eos_token
    if eos is not None:
        formatted_inputs = [
            fi if fi.endswith(eos) else (fi + eos) for fi in formatted_inputs
        ]

    tokenized = tokenizer(
        text=formatted_inputs,
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    tokenized_no_labels = tokenizer(
        text=formatted_inputs_no_labels,
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )

    # print('tokenized_no_labels: ',tokenized_no_labels)
    # print('tokenized_no_labels[0]: ',tokenized_no_labels[0])


    prompt_length_list = [len(formatted_no_label) for formatted_no_label in tokenized_no_labels["input_ids"]]
    print()
    input_length_list = [len(formatted ) for formatted in tokenized["input_ids"]]

    loss_weight_mask_list = []
    for input_length, prompt_length in zip(input_length_list, prompt_length_list):
        loss_weight_mask = np.ones(input_length, dtype=np.float32)
        if not calculate_loss_on_prompt: 
            loss_weight_mask[:prompt_length] = 0.0
        loss_weight_mask_list.append(loss_weight_mask)
    
    # Ensure EOS token position is NOT masked - we want to train on it
    # This is crucial for teaching the model when to stop generating
    if tokenizer.eos_token_id is not None:
        for i, tok in enumerate(tokenized["input_ids"]):
            if input_length_list[i] > 0:
                if tok[-1] == tokenizer.eos_token_id:
                    # Make sure the EOS token is included in training (not masked)
                    loss_weight_mask_list[i][-1] = 1.0
    # print('detokenized tokenized["input_ids"][0]', tokenizer.decode(tokenized["input_ids"][0]))
    input_tokens_list = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": [tok if tok else [[1] * len(ids) for ids in tokenized["input_ids"]] for tok in tokenized["attention_mask"]],
        "labels": tokenized["input_ids"],
        "loss_weight_mask": loss_weight_mask_list,
    }

    return input_tokens_list
    
def get_default_chat_template() -> str:
    default_chat_template = """{% for message in messages %}
{% if message['role'] == 'user' %}
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}{% if not loop.last or add_generation_prompt == false %}{{ eos_token }}{% endif %}
{% elif message['role'] == 'system' %}
<|begin_of_text|> System: {{ message['content'] }}
{% endif %}
{% endfor %}{% if add_generation_prompt %}
Assistant:{% endif %}"""   
    return default_chat_template

class MedLlmDataset(Dataset):
    """
    Dataset for the MedLlm training.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 512,
        max_n_examples: Optional[int] = None,
        train_data_path: str = '',
        train_data_split: str = 'train',
        training_type: str = 'next_token_pred',
        calculate_loss_on_prompt: bool = False,
        tag_token_start: str = "",
        item_column_name: str = "sft_item",
        options_column_name: str = "options",
    ):
        print(f"Loading dataset {train_data_path}...")

        # Load entire JSON into Arrow-backed dataset (fast and memory-efficient compared to Python lists)
        try:
            data = load_dataset(train_data_path, split=train_data_split)
            print("Dataset loaded via load_dataset from path:", train_data_path)
        except:
            try:
                data = load_dataset(train_data_path, train_data_split)
                if 'train' in data:
                    data = data['train']
                print("Dataset loaded via load_dataset from path:", train_data_path)
            except:
                data = HFDataset.from_json(train_data_path)
                print("Dataset loaded via HFDataset.from_json from path:", train_data_path)
        if tokenizer.chat_template is None:
            tokenizer.chat_template = get_default_chat_template()

        if max_n_examples is not None:
            max_n_examples = min(max_n_examples, len(data))
            data = data.select(range(max_n_examples))

        special_ids = tokenizer.added_tokens_decoder.keys()
        token_for_masking = tokenizer.added_tokens_decoder[list(special_ids)[-2]].content
        mask_token = token_for_masking
        if training_type == 'dec_self_mask':
            system_prompt = "Fill in the masked word in the following sentence."
            format_convesation_fn = lambda example, return_without_label=False: _format_instruct_conversation(example, system_prompt=system_prompt, return_without_label=return_without_label)
        elif training_type == 'sft_task':
            system_prompt = "Fill in the masked word in the following sentence."
            format_convesation_fn = lambda example, return_without_label=False: _format_instruct_conversation_sft_task(example, system_prompt=system_prompt, mask_token=mask_token, return_without_label=return_without_label, tag_token_start=tag_token_start, item_column_name=item_column_name, options_column_name=options_column_name)
        else:
            raise NotImplementedError("next_token_pred training type is not implemented yet.")
        has_pretokenized = all(
            col in data.column_names for col in ["input_ids", "labels"]
        )

        if has_pretokenized:
            def ensure_masks(batch: Dict[str, List[Any]]):
                input_ids = batch["input_ids"]
                attn = batch.get("attention_mask")
                mask = batch.get("loss_weight_mask")
                out_attention = []
                out_loss_mask = []
                for iids, maybe_attn in zip(input_ids, attn if attn is not None else [None] * len(input_ids)):
                    length = len(iids)
                    if maybe_attn is None or len(maybe_attn) != length:
                        out_attention.append([1] * length)
                    else:
                        out_attention.append(maybe_attn)
                    out_loss_mask.append(mask[i] if mask is not None else [1.0] * length)
                return {
                    "attention_mask": out_attention,
                    "loss_weight_mask": out_loss_mask,
                }
            data = data.map(
                ensure_masks,
                batched=True,
                batch_size=batch_size,
                desc=f"Ensuring masks for pre-tokenized dataset {train_data_path}",
            )
        else:
            # Batched tokenization via map; avoids O(n^2) islice and huge Python lists
            data = data.map(
                lambda batch, idx: _format_and_tokenize_instruct_batch(
                    tokenizer, 
                    batch, 
                    format_convesation_fn=format_convesation_fn, 
                    system_prompt=system_prompt,
                    has_printed=(idx[0] > 0),
                    calculate_loss_on_prompt=calculate_loss_on_prompt,
                ),
                batched=True,
                with_indices=True,
                batch_size=batch_size,
                num_proc=16,  # set >1 if your tokenizer is pickle-safe in multiprocessing
                desc=f"Tokenizing {train_data_path}",
            )
        # Batched tokenization via map; avoids O(n^2) islice and huge Python lists
        data = data.map(
            lambda batch, idx: _format_and_tokenize_instruct_batch(
                tokenizer, 
                batch, 
                format_convesation_fn=format_convesation_fn, 
                system_prompt=system_prompt,
                has_printed=(idx[0] > 0),
                calculate_loss_on_prompt=calculate_loss_on_prompt,
            ),
            batched=True,
            with_indices=True,
            batch_size=batch_size,
            num_proc=16,  # set >1 if your tokenizer is pickle-safe in multiprocessing
            desc=f"Tokenizing {train_data_path}",
        )
        # print("Data columns after tokenization: ", data)
        # print("Example 0 after tokenization: ", data[0])
        # Keep only the fields needed for training
        keep_cols = {"input_ids", "attention_mask", "labels", "loss_weight_mask"}
        drop_cols = [c for c in data.column_names if c not in keep_cols]
        if drop_cols:
            data = data.remove_columns(drop_cols)


        self.dataset = data
        print("data is: ", data)
        print(f"Dataset for split {train_data_path} has {len(self.dataset)} examples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    

@dataclass
class DataCollatorForMedLlm:
    """
    Adapted from transformers.DataCollatorForSeq2Seq to handle CoLLIE data.

    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        loss_weight_mask = (
            [feature["loss_weight_mask"] for feature in features]
            if "loss_weight_mask" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        if loss_weight_mask is not None:
            max_loss_weight_mask_length = max(len(l) for l in loss_weight_mask)
            if self.pad_to_multiple_of is not None:
                max_loss_weight_mask_length = (
                    (max_loss_weight_mask_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [0.0 if self.label_pad_token_id == -100 else 1.0] * (
                    max_loss_weight_mask_length - len(feature["loss_weight_mask"])
                )
                if isinstance(feature["loss_weight_mask"], list):
                    feature["loss_weight_mask"] = (
                        feature["loss_weight_mask"] + remainder
                        if padding_side == "right"
                        else remainder + feature["loss_weight_mask"]
                    )
                elif padding_side == "right":
                    feature["loss_weight_mask"] = np.concatenate(
                        [feature["loss_weight_mask"], remainder]
                    ).astype(np.float32)
                else:
                    feature["loss_weight_mask"] = np.concatenate(
                        [remainder, feature["loss_weight_mask"]]
                    ).astype(np.float32)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        # print('features: ', features)
        # print('features["labels"]: ', features["labels"])
        # print('features["labels"][0]: ', features["labels"][0])
        return features
