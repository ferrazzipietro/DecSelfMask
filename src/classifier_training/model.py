import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM
from torch.utils.data import Dataset, DataLoader

class LlamaLastTokenClassifier(nn.Module):
    def __init__(self, model_path, num_classes, cache_dir=None, freeze_lm=True, torch_dtype=torch.float32, use_non_linearity: bool=True):
        super().__init__()
        if 'gemma' in model_path:
            attn_implementation='eager'
            ModelClass = Gemma3ForCausalLM
        else:
            attn_implementation=None
            ModelClass = AutoModelForCausalLM
        
        self.lm = ModelClass.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            output_hidden_states=True,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        
        # Detect padding side from the tokenizer
        tok = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.padding_side = tok.padding_side  # "left" or "right"
        
        hidden_size = self.lm.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256, dtype=torch_dtype),
            nn.ReLU() if use_non_linearity else nn.Identity(),
            # nn.ReLU() if use_non_linearity else nn.Identity(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes, dtype=torch_dtype)
        )
        
        self.freeze_lm = freeze_lm
        if self.freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False
            self.lm.eval()  # disable dropout in frozen LM

    def forward(self, input_ids, attention_mask):
        # Skip gradient computation for frozen LM to save memory
        if self.freeze_lm:
            with torch.no_grad():
                outputs = self.lm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
        else:
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        last_hidden = outputs.hidden_states[-1]  # (B, T, H)
        if self.freeze_lm:
            last_hidden = last_hidden.detach()
        batch_size = input_ids.size(0)

        if self.padding_side == "left":
            # Left-padding: last real token is always at the rightmost position
            last_token_indices = torch.tensor(
                [input_ids.size(1) - 1] * batch_size, device=last_hidden.device
            )
        else:
            # Right-padding: last real token is at (number of non-pad tokens - 1)
            last_token_indices = attention_mask.sum(dim=1) - 1  # (B,)

        last_token_embeddings = last_hidden[
            torch.arange(batch_size, device=last_hidden.device), last_token_indices
        ]  # (B, H)

        logits = self.classifier(last_token_embeddings)

        return logits
