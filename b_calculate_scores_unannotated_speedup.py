from src.score_relevancy import AttnLRPScorer, PureAttentionRelevancyScorer
from src.prompt import Prompt, PromptNER, PromptNERNoAssistant, PromptNoGT, PromptNoGTAdmission, PromptNoGTDyspneaClassification, PromptNoGTMesh
from src.mapper_tokens_to_span import MapperTokensToSpans
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from argparse import ArgumentParser
import re
from dotenv import dotenv_values
from huggingface_hub import login
import torch
import json
import random
import logging
import gc

os.environ['NCCL_P2P_DISABLE'] = '1'

HF_TOKEN = dotenv_values(".env").get("HF_TOKEN", "") 
login(token=HF_TOKEN)
def get_tokens_attentions_outputs_inputs(prompt, model, tokenizer, char_to_remove_tokens=''):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    tokens = [t.replace(char_to_remove_tokens, '') for t in tokenizer.convert_ids_to_tokens(inputs.input_ids[0])]
    with torch.no_grad():
        outputs = model(**inputs)
    attentions = outputs.attentions
    return tokens, attentions, outputs, inputs



if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Model name or path')
    argparser.add_argument('--data_path', type=str, default="Pretrain- -YOUR_PATH_ORG/ClinicalWhole", help='Dataset name or path') # annotated_crf/processed/all_data_200.json "YOUR_PATH_ORG- /e3c-sentences-EN-native"
    # argparser.add_argument('--cuda_visible_devices', type=str, default='0', help='GPU id')
    argparser.add_argument('--start_from_note', type=int, default=0, help='Start from note index')
    argparser.add_argument('--end_at_note', type=int, default=-1, help='End at note index')
    # argparser.add_argument('--token_position', type=int, default=-2, help='Token position to compute relevancy for')
    argparser.add_argument('--cache_dir', type=str, default='/YOUR_PATH/.cache/', help='Cache directory for models and tokenizers')
    argparser.add_argument('--use_which_token', type=str, default='mid', help='Which token position of the generated text to use for relevancy metric scoring. Must be one between "first", "last" or "mid"')
    argparser.add_argument('--data_type', type=str, default='raw', help='Type of data to use: crf_annotation or ner_annotation or crf_annotation_second, raw, dyspnea_classification')    
    argparser.add_argument('--keep_n_sequences_per_note', type=int, default=-1, help='Whether to keep only one sequence per note. If -1, keep all sequences. If a positive integer, keep at most n sequences per note.')
    argparser.add_argument('--path_save', type=str, default='/YOUR_PATH/unannotated_crf/data/', help='Path to save the results')
    argparser.add_argument('--batch_size_lrp', type=int, default=32, help='Batch size for LRP relevancy computation')
    args = argparser.parse_args()
    keep_n_sequences_per_note = int(args.keep_n_sequences_per_note)
    batch_size_lrp = max(1, int(args.batch_size_lrp))

    CALCULATE_LRP = True
    remove_noteID = True
    model_output_attentions = not CALCULATE_LRP

    if args.data_type == 'ner_annotation':
        type_of_entity = 'CLINENTITY'
        Prompt = PromptNERNoAssistant

    number_of_available_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {number_of_available_gpus}")

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    cache_dir = args.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir,  token=HF_TOKEN) # device_map='cuda',
    print(f'Loading {args.model_name}...')
    if number_of_available_gpus > 1:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            output_attentions=model_output_attentions,
            attn_implementation='eager',
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            token=HF_TOKEN,
            tp_plan="auto",
        ) # ,   ='cuda',
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            output_attentions=model_output_attentions,
            attn_implementation='eager',
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            device_map='auto',
            token=HF_TOKEN,
        ) # ,  device_map='cuda',

    # if data_path is a local json file, load it as Dataset
    if os.path.isfile(args.data_path):
        print(f"Loading dataset from local file: {args.data_path}")
        data = Dataset.from_json(args.data_path)
    else:
        print(f"Loading dataset from HuggingFace Hub: {args.data_path}")
        data = load_dataset(args.data_path, cache_dir=cache_dir)
        if 'train' in data:
            data = data['train']
        if 'train_unsup' in data:
            data = data['train_unsup']
    data = data.select(range(args.start_from_note, args.end_at_note if args.end_at_note != -1 else len(data)))
    # print(data)

    out = {}
    from transformers.models.llama import modeling_llama

    if CALCULATE_LRP: scorer_lrp = AttnLRPScorer(modeling_type=modeling_llama, model=model)
    # scorer = PureAttentionRelevancyScorer()

    def find_tokens_position(prompt_text, label_annotation, tokenizer, verbose=False):
        prompt_tokens = tokenizer.tokenize(prompt_text)
        if " - " in label_annotation:
            label_annotation = label_annotation.split(" - ")[-1]
        label_tokens = tokenizer.tokenize(label_annotation)
        if verbose: print('Prompt tokens:', prompt_tokens)
        if verbose: print('Looking for label tokens:', label_tokens)
        out = -1, -1
        for i in range(len(prompt_tokens) - len(label_tokens) + 1):
            if prompt_tokens[i:i+len(label_tokens)] == label_tokens:
                if verbose: print(f"Found label tokens at positions {i} to {i + len(label_tokens) - 1}")
                out =  i, i + len(label_tokens) - 1
        if verbose: print("\n\n")
        if out == (-1, -1):
            if verbose: print("SECOND LAP")
            label_annotation = ' '+label_annotation
            label_tokens = tokenizer.tokenize(label_annotation)
            for i in range(len(prompt_tokens) - len(label_tokens) + 1):
                if prompt_tokens[i:i+len(label_tokens)] == label_tokens:
                    if verbose: print(f"Found label tokens at positions {i} to {i + len(label_tokens) - 1} (after adding leading space)")
                    out =  i, i + len(label_tokens) - 1
            if verbose: print("\n\n")
        if out == (-1, -1):
            print("WARNING: Could not find label tokens in prompt.\n",
                  f"Label annotation: {label_annotation}\n",
                  f"Label tokens: {label_tokens}\n",
                  f"Prompt tokens: {prompt_tokens}",
                 )
        if verbose: print(f"\n\n\nToken positions for label '{label_annotation}': {out}\n FROM PROMPT: {prompt_tokens[out[0]-5:out[1]+1]}\n\n\n")
        return out


    def remove_noteID_from_begin(text:str):
        text_remove = text
        len_to_remove = 0
        targets = ['ANAMNESIS', 'CLINICAL_DIARY', 'DISCHARGE', 'MEDICAL_VISIT', 'NURSING_CARE_NOTES', 'SPECIALIST_CONSULTANCY', 'TRIAGE']
        for t in targets:
            patterns = f"\n---{t}---\n"
            text_remove = re.sub(patterns, ' ', text_remove)
            removed_pattern_bool = re.search(patterns, text) is not None
            if removed_pattern_bool:
                len_to_remove = len(patterns)
        # remove teh NoteID: int pattern
        patterns = r"NoteID: \d+"
        text_remove = re.sub(patterns, '', text_remove)
        removed_pattern_bool = re.search(patterns, text) is not None
        if removed_pattern_bool:
            len_to_remove = len_to_remove + len(re.search(patterns, text).group(0))
        return text_remove, len_to_remove

 
    if args.data_type == 'ner_annotation':
        data_orig = data
        data = []   
        type_of_entity = 'CLINENTITY'
        for i, note in enumerate(data_orig):
            new_d = {'id': str(i)+'_'+note['original_id'], 'text': note['sentence'], 'spans': [{'end':e['offsets'][1], 'labels':e['text'], 'type': e['type'], 'start':e['offsets'][0], 'text':e['text'], 'to_name':'text'} for e in note['entities'] if e['type']==type_of_entity]}
            data.append(new_d)
    elif args.data_type == 'crf_annotation_second':
        data_orig = data
        data = []   
        for i, note in enumerate(data_orig):
            if remove_noteID:
                text_remove, len_to_remove = remove_noteID_from_begin(note['text'])
            else:
                len_to_remove = 0
            new_d = {'id': str(i)+'_'+note['note_id'], 'text': text_remove, 'spans': [{'end':e['end']-len_to_remove+1, 'labels':[e['crf_item'] + ' - ' + e['label_assigned']], 'start':e['start']-len_to_remove+1, 'text':e['text'], 'to_name':'text'} for e in note['spans']]}
            data.append(new_d)
    elif args.data_type == 'raw':
        data_orig = data
        data = []   
        for i, note in enumerate(data_orig):
            new_d = {'id': note['id'], 'text': note['chunk'], 'spans': []}
            data.append(new_d)
    elif args.data_type == 'dyspnea_classification':
        data_orig = data
        data = []   
        for i, note in enumerate(data_orig):
            new_d = {'id': note['id'], 'text': note['chunk'], 'to_name':'text'}
            data.append(new_d)
    elif args.data_type == 'admission':
        data_orig = data
        data = []   
        for i, note in enumerate(data_orig):
            new_d = {'id': note['hadm_id'], 'text': note['text'], 'spans': []}
            data.append(new_d)

    # print(data)



with open('annotated_crf/crf.json', 'r') as f:
    crf_items = json.load(f)
target_items = [e for ll in crf_items.values() for e in ll]

if args.data_type == 'mesh':
    with open('data/mesh_for_unsup_train.txt', 'r') as f:
        target_items = f.read().splitlines()

if args.data_type == 'admission':
    with open('data/admission_levels.txt', 'r') as f:
        target_items = f.read().splitlines()

pending_jobs = []


def _is_cuda_oom_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return (
        'out of memory' in message
        or 'cuda error: out of memory' in message
        or 'cublas_status_alloc_failed' in message
        or 'cuda out of memory' in message
        or 'cudacachingallocator.cpp' in message
        or 'nvml_success == r' in message
        or 'cuda error' in message and 'alloc' in message
    )


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            # Some CUDA runtime combinations do not support IPC collection.
            pass


def flush_pending_jobs(jobs, out_dict):
    if not jobs:
        return

    cursor = 0
    while cursor < len(jobs):
        remaining = len(jobs) - cursor
        # Always start retries from the configured maximum for each chunk.
        current_try_batch_size = min(batch_size_lrp, remaining)

        while current_try_batch_size >= 1:
            try:
                batch_jobs = jobs[cursor: cursor + current_try_batch_size]
                batch_prompts = [j['prompt'] for j in batch_jobs]
                batch_positions = [j['token_position'] for j in batch_jobs]
                batch_relevancy = scorer_lrp.get_relevancy_batch(
                    prompts=batch_prompts,
                    tokenizer=tokenizer,
                    token_positions=batch_positions,
                )
                for job, relevancy_lrp in zip(batch_jobs, batch_relevancy):
                    out_dict[job['key']] = {
                        'prompt': job['prompt'].prompt,
                        'relevancy_attn_lrp': relevancy_lrp,
                        'token_position': job['token_position'],
                    }

                cursor += current_try_batch_size
                break
            except RuntimeError as error:
                if not _is_cuda_oom_error(error):
                    raise

                _release_cuda_memory()
                if current_try_batch_size == 1:
                    failed_job = jobs[cursor]
                    logging.error(
                        "CUDA memory failure persisted at batch_size=1. "
                        f"Skipping job {failed_job['key']}. Error: {error}"
                    )
                    cursor += 1
                    break

                new_batch_size = max(1, current_try_batch_size // 2)
                logging.warning(
                    f"CUDA OOM at batch_size={current_try_batch_size}. Retrying with batch_size={new_batch_size}."
                )
                current_try_batch_size = new_batch_size


for note_pos in tqdm(range(len(data))):
    if args.data_type == 'ner_annotation':
        target_items_this_note = [s['labels'].split(' -')[0] for s in data[note_pos]['spans']]
    elif args.data_type == 'raw' or args.data_type == 'mesh' or args.data_type == 'admission':
        target_items_this_note = target_items
    elif args.data_type == 'dyspnea_classification':
        target_items_this_note = ['dyspnea']
    else:
        target_items_this_note = [s['labels'][0].split(' -')[0] for s in data[note_pos]['spans']]

    if keep_n_sequences_per_note > 0 and len(target_items_this_note) > keep_n_sequences_per_note:
        # randomly select n target items for this note
        target_items_this_note = random.sample(target_items_this_note, keep_n_sequences_per_note)

    for target in target_items_this_note:
        # prompt = Prompt(example=data[note_pos], answer=target, tokenizer=tokenizer)
        if args.data_type == 'ner_annotation':
            prompt = PromptNER(example=data[note_pos], span_pos=0, tokenizer=tokenizer, entity_type=type_of_entity)
        elif args.data_type == 'raw':
            prompt = PromptNoGT(note_text=data[note_pos]['text'], target_item=target, tokenizer=tokenizer)
        elif args.data_type == 'mesh':
            prompt = PromptNoGTMesh(note_text=data[note_pos]['abstractText'], target_item=target, tokenizer=tokenizer)
        elif args.data_type == 'dyspnea_classification':
            prompt = PromptNoGTDyspneaClassification(note_text=data[note_pos]['text'], tokenizer=tokenizer)
        elif args.data_type == 'admission':
            prompt = PromptNoGTAdmission(note_text=data[note_pos]['text'], target_item=target, tokenizer=tokenizer)
        if note_pos==0:
            # print('target:', target)
            print('\n\n\nEXAMPLE PROMPT' + prompt.prompt)
        
        if args.data_type == 'dyspnea_classification':
            start_answer_token_pos, end_answer_token_pos = -1, -1
        else:
            start_answer_token_pos, end_answer_token_pos = find_tokens_position(prompt.prompt, target, tokenizer)

        if args.use_which_token == 'first':
            token_position = start_answer_token_pos
        elif args.use_which_token == 'last':
            token_position = end_answer_token_pos
        elif args.use_which_token == 'mid':
            token_position = int((end_answer_token_pos + start_answer_token_pos) / 2)
        elif args.use_which_token == 'random':
            token_position = torch.randint(0, end_answer_token_pos + 1, (1,)).item()
        else:
            raise ValueError("use_which_token must be one between 'first', 'last' or 'mid'")
        pending_jobs.append(
            {
                'key': f'note_{note_pos}_span_{target}',
                'prompt': prompt,
                'token_position': token_position,
            }
        )

        if len(pending_jobs) >= batch_size_lrp:
            flush_pending_jobs(pending_jobs[:batch_size_lrp], out)
            pending_jobs = pending_jobs[batch_size_lrp:]

logging.info(f"Finished processing all notes. Flushing remaining {len(pending_jobs)} pending jobs...")
if pending_jobs:
    flush_pending_jobs(pending_jobs, out)

path = f'{args.path_save}/attention_relevancy_unannotated/{args.data_path.split("/")[-1]}/{args.model_name.split("/")[-1]}'
os.makedirs(path, exist_ok=True)
# before writing the json file, just write a txt with the raw data
with open(f'{path}/attention_relevancy_results_{args.start_from_note}_{args.end_at_note}_{args.use_which_token}_{args.data_type}.txt', 'w') as f:
    f.write(str(out))
import json
with open(f'{path}/attention_relevancy_results_{args.start_from_note}_{args.end_at_note}_{args.use_which_token}_{args.data_type}.json', 'w') as f:
    json.dump(out, f)
all_json_files = os.listdir(path)
combined_out = {}
for json_file in all_json_files:
    if json_file.startswith('attention_relevancy_results_') and json_file.endswith(f'{args.data_type}.json'):
        print(json_file)
        with open(os.path.join(path, json_file), 'r') as f:
            part_out = json.load(f)
            name = json_file.replace('.json','').replace('attention_relevancy_results_','')
            new_this_data = { f"{k}_{name}": v for k, v in part_out.items() }
            combined_out.update(new_this_data)
            print(f'Updated combined_out to have {len(combined_out)} entries')
with open(f'{path}/combined_{args.use_which_token}_{args.data_type}.json', 'w') as f:
    json.dump(combined_out, f)

print('saved results to', path)