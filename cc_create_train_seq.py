import json
import os
from scipy.ndimage import gaussian_filter1d
from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import Dataset
from dotenv import dotenv_values
from logging import basicConfig, getLogger


basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level='INFO')
logger = getLogger(__name__)

HF_TOKEN=dotenv_values('.env')['HF_TOKEN']



def gaussian_smoothing(importance, kernel_size=2, sigma=1.0):
    smoothed_importance = gaussian_filter1d(importance, sigma=sigma, truncate=((kernel_size - 1) / 2) / sigma)
    return smoothed_importance

def find_consecutive_high_importance(importance_scores, threshold_upper, threshold_lower):
    high_importance_groups = []
    current_group = []
    has_above_upper = False
    
    for token_i, score in enumerate(importance_scores):
        if score > threshold_lower:
            current_group.append(token_i)
            if score > threshold_upper:
                has_above_upper = True
        else:
            if current_group and has_above_upper:
                high_importance_groups.append(current_group)
            current_group = []
            has_above_upper = False
    
    if current_group and has_above_upper:
        high_importance_groups.append(current_group)
    
    return high_importance_groups

def create_training_sequence_by_masking(tokens, groups_labels, tokenizer, sep_token, masked = '<MASK>'):
    masked_sequences = []
    labels = []
    sentences = []
    has_left_list = []
    positions_to_mask = [i for i, el in enumerate(groups_labels) if el != 0]
    has_group_left = [el > 1 for i, el in enumerate(groups_labels) if i in positions_to_mask]
    for mask_pos, has_left in zip(positions_to_mask, has_group_left):
        sequence = tokens.copy()
        target_token = sequence[mask_pos]
        sequence[mask_pos] = masked
        masked_sequences.append(sequence)
        sequence[mask_pos] = sep_token + masked if target_token.startswith(sep_token) else masked
        detokenize_seq = tokenizer.convert_tokens_to_string(sequence)
        sentences.append(detokenize_seq)
        labels.append(tokens[mask_pos])
        has_left_list.append(has_left)
    return masked_sequences, labels, sentences, has_left_list

def create_training_sequence_by_masking_all_group(tokens, groups_labels, tokenizer, sep_token, masked = '<MASK>'):
    masked_sequences = []
    labels = []
    sentences = []
    has_left_list = []
    how_many_groups = max(set(groups_labels))
    for i in range(1, how_many_groups + 1):
        sequence = tokens.copy()
        idxs_to_mask = [idx for idx, g_label in enumerate(groups_labels) if g_label == i]
        sequence[idxs_to_mask[0]] = masked + sep_token if tokens[idxs_to_mask[0]].startswith(sep_token) else masked
        sequence = [s for i, s in enumerate(sequence) if i not in idxs_to_mask[1:]]
        detokenize_seq = tokenizer.convert_tokens_to_string(sequence)
        sentences.append(detokenize_seq)
        label = tokenizer.convert_tokens_to_string([tokens[idx] for idx in idxs_to_mask])
        labels.append(label)
        has_left = False if i == 1 else True
        has_left_list.append(has_left)
        masked_sequences.append(sequence)
    return masked_sequences, labels, sentences, has_left_list

def add_groups(example, smoothing_fn, grouping_fn, smoothing_fn_params, grouping_fn_params):
    tokens = [t[0] for t in example['relevancy_prompt_input_text']]
    smoothed = smoothing_fn([float(t[1]) for t in example['relevancy_prompt_input_text']], **smoothing_fn_params)
    groups = grouping_fn(smoothed, **grouping_fn_params)
    groups_dict = {t_id:i for i, group in enumerate(groups, start=1) for t_id in group}
    groups_labels = [groups_dict.get(i, 0) for i in range(len(example['relevancy_prompt_input_text']))]
    groups_labels = assign_word_groups(tokens, groups_labels)
    example['high_importance_groups'] = groups_labels
    return example


def assign_word_groups(tokens, groups_labels, start_token='Ġ'):
    """For all sub-word tokens that belong to the same word, if ANY of them
    belongs to a group (non-zero label), assign that group label to ALL tokens
    in the word. A new word starts whenever a token begins with `start_token`.
    After propagation, group numbers are renumbered to be contiguous (1, 2, 3, ...)."""
    word_groups_labels = []
    current_word_tokens = []
    current_word_labels = []

    for token, group_label in zip(tokens, groups_labels):
        # A token starting with the separator marks the beginning of a new word
        if token.startswith(start_token) and current_word_tokens:
            # Flush the previous word: pick the non-zero label if any
            non_zero = [l for l in current_word_labels if l != 0]
            label = non_zero[0] if non_zero else 0
            word_groups_labels.extend([label] * len(current_word_tokens))
            current_word_tokens = []
            current_word_labels = []

        current_word_tokens.append(token)
        current_word_labels.append(group_label)

    # Flush the last word
    if current_word_tokens:
        non_zero = [l for l in current_word_labels if l != 0]
        label = non_zero[0] if non_zero else 0
        word_groups_labels.extend([label] * len(current_word_tokens))

    # Renumber groups to be contiguous (1, 2, 3, ...) so that
    # create_training_sequence_by_masking_all_group doesn't hit gaps
    unique_labels = sorted(set(word_groups_labels) - {0})
    if unique_labels:
        remap = {old: new for new, old in enumerate(unique_labels, start=1)}
        remap[0] = 0
        word_groups_labels = [remap[l] for l in word_groups_labels]

    return word_groups_labels

def create_masked_seq(example, masking_type, tokenizer, token_for_masking, sep_token):
    if masking_type not in VALID_MASKING_TYPES:
        raise ValueError(f"Invalid masking_type: {masking_type}. Choose {VALID_MASKING_TYPES}.")
    if masking_type == 'single_token':
        masked_sequences, labels, sentences, has_left_list = create_training_sequence_by_masking(tokens=example['tokens'], 
                                                                                               groups_labels=example['high_importance_groups'], 
                                                                                               tokenizer=tokenizer, 
                                                                                               sep_token=sep_token,
                                                                                               masked=token_for_masking)
    elif masking_type == 'all_group':
        masked_sequences, labels, sentences, has_left_list = create_training_sequence_by_masking_all_group(tokens=example['tokens'], 
                                                                                                            groups_labels=example['high_importance_groups'], 
                                                                                                            tokenizer=tokenizer, 
                                                                                                            sep_token=sep_token,
                                                                                                            masked=token_for_masking)
    return {
        'id': example['id'],
        'masked_sequences': masked_sequences,
        'labels': labels,
        'sentences': sentences,
        'has_left_list': has_left_list
    }


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--input_path', type=str, required=False, default='data/attention_relevancy_unannotated/ClinicalWhole/Llama-3.1-8B-Instruct/combined_mid.json', help='Path to the input JSON file containing relevancy scores.')
    parser.add_argument('--smoothing_method', type=str, required=False, default='gaussian', help='Smoothing method to apply on importance scores.')
    parser.add_argument('--threshold_upper', type=float, required=False, default=0.4, help='Upper threshold for high importance. One token in each group must exceed this value.')
    parser.add_argument('--threshold_lower', type=float, required=False, default=0.2, help='Lower threshold for importance. Tokens below this value end a group.')
    parser.add_argument('--sep_token', type=str, required=False, default='Ġ', help='Separator token used by the tokenizer.')
    parser.add_argument('--type_of_masking', type=str, required=False, default='all_group', help='Type of masking to create training sequences: single_token or all_group.')
    parser.add_argument('--cache_dir', type=str, required=False, default='/workspace/.cache', help='Cache directory for the tokenizer.')
    parser.add_argument('--exclude_single_group_sequences', action='store_true', help='Whether to exclude sequences with only a single high-importance group.')
    parser.add_argument('--output_dir', type=str, required=False, default='data/cc_train_sequences_mesh_batch_2', help='Directory to save the output training sequences.')
    args = parser.parse_args()


    with open(args.input_path, 'r') as f:
        part_out = json.load(f)

    model_id = 'meta-llama/Llama-3.1-8B-Instruct'
    model_mapper = {'Llama-3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct'}
    VALID_MASKING_TYPES = ['single_token', 'all_group']

    model = model_id # args.input_path.split('/')[-2]
    tokenizer = AutoTokenizer.from_pretrained(model_mapper.get(model, model), cache_dir=args.cache_dir)
    logger.info(f"Loaded tokenizer for {model}.")

    # d_for_dataset = {
    #     'id':[k for k in part_out.keys()],
    #     'prompt':[part_out[k]['prompt'] for k in part_out.keys()],
    #     'tokens':[[t[0]  for t in part_out[k]['relevancy_attn_lrp']['relevancy_prompt_input_text']] for k in part_out.keys() ],   
    #     'relevancy_prompt_input_text':[[[t, str(s)] for t, s in part_out[k]['relevancy_attn_lrp']['relevancy_prompt_input_text']] for k in part_out.keys() ],
    #     'token_position':[part_out[k]args['token_position'] for k in part_out.keys()],
    #     }
    # d = Dataset.from_dict(d_for_dataset)
    def examples():
        for k, v in part_out.items():
            rpit = v['relevancy_attn_lrp']['relevancy_prompt_input_text']
            yield {
                'id': k,
                'prompt': v['prompt'],
                'tokens': [t[0] for t in rpit],
                'relevancy_prompt_input_text': [[t, str(s)] for t, s in rpit],
                'token_position': v['token_position'],
            }

    d = Dataset.from_generator(examples, cache_dir=args.cache_dir, num_proc=8)
    logger.info("Prepared dataset.")
    
    smoothing_fn = gaussian_smoothing if args.smoothing_method == 'gaussian' else None
    grouping_fn = find_consecutive_high_importance

    # d = d.select(range(5))

    d = d.map(lambda example: add_groups(
        example, 
        smoothing_fn=smoothing_fn, 
        grouping_fn=grouping_fn, 
        smoothing_fn_params={'kernel_size':3, 'sigma':1.0}, 
        grouping_fn_params={'threshold_upper':args.threshold_upper, 'threshold_lower':args.threshold_lower}),
        num_proc=8
        )
    if args.exclude_single_group_sequences:
        d = d.filter(lambda x: len(x['high_importance_groups']) > 1,
                     num_proc=8) 
    special_ids = tokenizer.added_tokens_decoder.keys()
    token_for_masking = tokenizer.added_tokens_decoder[list(special_ids)[-2]].content
    d = d.map(lambda x: create_masked_seq(
            x, 
            masking_type=args.type_of_masking, 
            tokenizer=tokenizer, 
            token_for_masking=token_for_masking,
            sep_token=args.sep_token),
            num_proc=8
        )
    d_out = {
        'id': [i for i_list in [[ex['id']]*len(ex['masked_sequences']) for ex in d] for i in i_list],
        # 'masked_sequence': [ms for ex in d for ms in ex['masked_sequences']],
        'label': [lbl for ex in d for lbl in ex['labels']],
        'sentence': [sent for ex in d for sent in ex['sentences']],
        'has_group_left': [hgl for ex in d for hgl in ex['has_left_list']],
    }
    logger.info("Created masked sequences and labels.")
    d = Dataset.from_dict(d_out)

    def extract_note_id(example):
        example['note_id'] = example['id'].split('_span_')[0]
        return example

    d = d.map(extract_note_id, num_proc=16)

    # try:
    #     single_note_ids = set(d['train']['note_id'])
    # except KeyError:
    #     single_note_ids = set(d['note_id'])
    single_note_ids = set(d['note_id'])

    import random
    single_note_ids = list(single_note_ids)
    random.Random(42).shuffle(single_note_ids)
    train_ratio = 0.9
    train_note_size = int(train_ratio * len(single_note_ids))
    train_dataset = d.filter(lambda x: x['note_id'] in single_note_ids[:train_note_size], num_proc=16)
    validation_dataset = d.filter(lambda x: x['note_id'] in single_note_ids[train_note_size:], num_proc=16)

    out_dir_path = os.path.join(args.output_dir, args.type_of_masking, token_for_masking)
    os.makedirs(out_dir_path, exist_ok=True)
    model = model.split('/')[-1]
    try:
        train_dataset.push_to_hub(f'YOUR_PATH/{args.smoothing_method}_{model}_{len(d)}', split='train', token=HF_TOKEN, commit_message=args.input_path)
        validation_dataset.push_to_hub(f'YOUR_PATH/{args.smoothing_method}_{model}_{len(d)}', split='validation', token=HF_TOKEN, commit_message=args.input_path)
        logger.info(f"Pushed to hub at YOUR_PATH/{args.smoothing_method}_{model}_{len(d)}.")
    except Exception as e:
        logger.warning(f"Could not push to hub: {e}")
    try:
        train_dataset.to_json(f'{out_dir_path}/{args.smoothing_method}_{model}_train.json')
        logger.info(f"Saved locally at {out_dir_path}/{args.smoothing_method}_{model}_train.json.")
        validation_dataset.to_json(f'{out_dir_path}/{args.smoothing_method}_{model}_validation.json')
        logger.info(f"Saved locally at {out_dir_path}/{args.smoothing_method}_{model}_validation.json.")
    except Exception as e:
        logger.warning(f"Could not save locally: {e}")