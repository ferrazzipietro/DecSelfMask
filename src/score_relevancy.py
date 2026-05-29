from pydantic import BaseModel, Field
import torch 
from lxt.efficient import monkey_patch
from typing import List, Optional, Dict, Any
from .prompt import Prompt
import torch.nn.functional as F


class BaseRelevancyScorer(BaseModel):

    temperature_softmax: float = Field(0.8, description="Temperature for softmax normalization of relevancy scores.")
    
    def get_relevancy(self, attentions, token_position):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        minv, maxv = min(scores), max(scores)
        rng = (maxv - minv) if (maxv - minv) > 1e-12 else 1.0
        normalized = [(s - minv) / rng for s in scores]
        return normalized
    
    def make_scores_sum_to_one(self, scores: List[float]) -> List[float]:
        total = sum(scores)
        if total == 0:
            return scores
        return [s / total for s in scores]    

    def softmax(self, scores: List[float], temperature: float = 1.0) -> List[float]:
        if not scores:
            return []
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        x = torch.tensor(scores, dtype=torch.float32)
        p = F.softmax(x / temperature, dim=0)
        return p.tolist()
    
    def softmax_on_abs(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        if self.temperature_softmax <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature_softmax}")
        x = torch.tensor([abs(s) for s in scores], dtype=torch.float32)
        p = F.softmax(x / self.temperature_softmax, dim=0)
        return p.tolist()

    def in_0_1(self, scores: List[float]) -> List[float]:
        # print('scores before in_0_1: ', scores)
        scores = [abs(s) for s in scores]
        maxv = max(scores)
        normalized = [s /maxv if maxv != 0 else 0.0 for s in scores]
        return normalized
    
    def _process_relevancy(self, relevancy: List[float], prompt: Prompt, tokenizer, return_empty_all_tokens:bool=True) -> List[float]:
        """
        Process relevancy scores by normalizing them to [-1, 1].
        """
        # print('relevancy before processing: ', relevancy)
        relevancy = self.in_0_1(relevancy)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer(prompt.prompt, return_tensors="pt", add_special_tokens=False).input_ids[0])
        # print('tokens: ', tokens)
        original_tokens_relevancy  = [(t, r) for t, r in zip(tokens, relevancy)]
        # print('original_tokens_relevancy: ', original_tokens_relevancy)
        # print('prompt.tokens_before : -prompt.tokens_after: ', prompt.tokens_before, -prompt.tokens_after)
        relevancy_scores = [r for t, r in original_tokens_relevancy[prompt.tokens_before : -prompt.tokens_after]]
        # print('relevancy_scores before in_0_1: ', relevancy_scores)
        relevancy_scores = self.in_0_1(relevancy_scores)
        # print('relevancy_scores after in_0_1: ', relevancy_scores)
        relevancy = [(t,r) for t, r in zip(tokens[prompt.tokens_before : -prompt.tokens_after], relevancy_scores)]
        # print('relevancy after processing: ', relevancy)
        if return_empty_all_tokens:
            original_tokens_relevancy = []
        return {'relevancy_prompt_input_text': relevancy, 'relevancy_all_tokens': original_tokens_relevancy}


class PureAttentionRelevancyScorer(BaseRelevancyScorer):
    """
    Relevancy based solely on attention scores.
    """

    def get_tokens_attentions_outputs_inputs(self, prompt, model, tokenizer, char_to_remove_tokens=''):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        tokens = [t.replace(char_to_remove_tokens, '') for t in tokenizer.convert_ids_to_tokens(inputs.input_ids[0])]
        with torch.no_grad():
            outputs = model(**inputs)
        attentions = outputs.attentions
        return tokens, attentions, outputs, inputs

    def _avg_among_layers(self, attentions, token_position, which_layers:list=None):
        """
        which_layers: list of layer indices to average over. If None, average over all layers.
        """
        num_layers = len(attentions)
        att_sum = torch.zeros(
            (attentions[0].size(2),),
            device=attentions[0].device,
            dtype=attentions[0].dtype,
        )
        if not which_layers:
            which_layers = list(range(num_layers))
        for layer in which_layers:
            att_sum += torch.mean(attentions[layer][0], dim=0)[token_position, :]
        att_avg = att_sum / len(which_layers)
        # Ensure dtype is convertible to NumPy (bf16 -> float32)
        to_plot_avg_over_layers = att_avg.to(torch.float32).cpu().numpy().tolist()
        return to_plot_avg_over_layers
    
    def get_relevancy(self, prompt: Prompt, which_layers,  char_to_remove_tokens:list=[], attentions=None, model=None, tokenizer=None, token_position=-1):
        """
        Get relevancy scores for a specific token position.
        If attentions are not provided, compute them using the model and tokenizer on the given prompt.
        """
        if not attentions and (model is None or tokenizer is None):
            raise ValueError("If attentions are not provided, model, tokenizer, and prompt must be provided.")
        if not attentions:
            _, attentions, _, _ = self.get_tokens_attentions_outputs_inputs(prompt=prompt.prompt, model=model, tokenizer=tokenizer, char_to_remove_tokens=char_to_remove_tokens)
        
        relevancy = self._avg_among_layers(attentions, token_position, which_layers)
        # except RuntimeError:
        #     attentions = [att.to('cuda') for att in attentions]
        #     relevancy = self._avg_among_layers(attentions, token_position, which_layers)
        #     print("Recovered from CUDA RuntimeError during attentions processing.")
        relevancy_dict = self._process_relevancy(relevancy, prompt, tokenizer)
        return relevancy_dict
    

class AttnLRPScorer(BaseRelevancyScorer):
    """
    Relevancy based on Layer-wise Relevance Propagation (LRP) applied to attention scores.
    """
    modeling_type: Any
    model: Any = Field(repr=False)

    def model_post_init(self, __context: Any) -> None:
        monkey_patch(self.modeling_type, verbose=True)
        self.model.train()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        for p in self.model.parameters():
            p.requires_grad = False

    def get_relevancy(self, prompt: Prompt, tokenizer, token_position=-1):
        """
        Get relevancy scores using LRP.
        """
        input_ids = tokenizer(prompt.prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
        input_embeds = self.model.get_input_embeddings()(input_ids)
        # print('input_embeds: ', input_embeds)
        output_logits = self.model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
        max_logits, _ = torch.max(output_logits[0, token_position, :], dim=-1)
        max_logits.backward()
        relevance = (input_embeds * input_embeds.grad).float().sum(-1).detach().cpu()[0] # cast to float32 before summation for higher precision
        # print('relevance: ', relevance)
        relevancy = relevance.cpu().numpy().tolist()        
        relevancy_dict = self._process_relevancy(relevancy, prompt, tokenizer)
        return relevancy_dict

    def get_relevancy_batch(self, prompts: List[Prompt], tokenizer, token_positions: List[int]):
        """
        Batched LRP relevancy.

        Runs a single forward/backward pass for a list of prompts and returns
        one relevancy dictionary per prompt.
        """
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if not prompts:
            return []
        if len(prompts) != len(token_positions):
            raise ValueError("prompts and token_positions must have the same length")

        prompt_texts = [p.prompt for p in prompts]
        encoded = tokenizer(
            prompt_texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        input_ids = encoded.input_ids.to(self.model.device)
        attention_mask = encoded.attention_mask.to(self.model.device)

        input_embeds = self.model.get_input_embeddings()(input_ids)
        output_logits = self.model(
            inputs_embeds=input_embeds.requires_grad_(),
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

        seq_lens = attention_mask.sum(dim=1)
        fixed_positions = []
        for pos, seq_len_t in zip(token_positions, seq_lens):
            seq_len = int(seq_len_t.item())
            fixed_pos = seq_len + pos if pos < 0 else pos
            if fixed_pos < 0 or fixed_pos >= seq_len:
                raise ValueError(
                    f"token_position {pos} is out of range for prompt length {seq_len}"
                )
            fixed_positions.append(fixed_pos)

        batch_idx = torch.arange(len(prompts), device=output_logits.device)
        pos_idx = torch.tensor(fixed_positions, device=output_logits.device)
        selected_logits = output_logits[batch_idx, pos_idx, :]
        max_logits = torch.max(selected_logits, dim=-1).values
        max_logits.sum().backward()

        relevance = (input_embeds * input_embeds.grad).float().sum(-1).detach().cpu()

        out = []
        for i, p in enumerate(prompts):
            seq_len = int(seq_lens[i].item())
            relevancy = relevance[i, :seq_len].numpy().tolist()
            out.append(self._process_relevancy(relevancy, p, tokenizer))
        return out