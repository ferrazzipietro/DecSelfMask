
import evaluate
f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")


def split_sequence_at_special_tokens(sequence, special_token_sequence, SPECIAL_TOKENS):
    """
    Splits the sequence at the first occurrence of the special token sequence.

    Args:
        sequence (list): The input sequence to be split.
        special_token_sequence (list): The sequence of special tokens to split at.

    Returns:
        tuple: A tuple containing two lists - the part before the special token sequence
               and the part after it (including the special token sequence).
    """
    seq_len = len(sequence)
    token_seq_len = len(special_token_sequence)

    for i in range(seq_len - token_seq_len + 1):
        if sequence[i:i + token_seq_len] == special_token_sequence:
            ret = sequence[(i + token_seq_len):]
            ret = [tok for tok in ret if tok not in SPECIAL_TOKENS]
            return ret  

    return [tok for tok in sequence if tok not in SPECIAL_TOKENS]  # Return the whole sequence and an empty list if not found

def preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to argmax predictions before accumulation to avoid OOM.
    Without this, the trainer accumulates full (batch, seq_len, vocab_size) tensors
    across all eval batches, which can easily exceed available RAM."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics_sft(p, SPECIAL_TOKENS_ASSISTANT_START, SPECIAL_TOKENS,  items_list_in_dataset_validation:list=[], items_list_in_dataset_test:list=[], calc_per_item=True, verbose=True):
    predictions, labels = p
    labels = labels[0]
    if verbose:
        print('One label sequence:\n', labels[0])
        print('One prediction sequence:\n', predictions[0])
    true_predictions = [
        [int(p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [int(l) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_predictions = [tp[-(len(tl)): ] for tp, tl in zip(true_predictions, true_labels)]
    true_predictions = [split_sequence_at_special_tokens(tp, SPECIAL_TOKENS_ASSISTANT_START, SPECIAL_TOKENS) for tp in true_predictions]
    true_labels = [split_sequence_at_special_tokens(tl, SPECIAL_TOKENS_ASSISTANT_START, SPECIAL_TOKENS) for tl in true_labels]
    
    seqeval = False
    if seqeval:
        true_predictions = [[' '.join(map(str, tok))] for tok in true_predictions]
        true_labels = [int(tok) if tok else -1 for tok in true_labels]
    else:
        true_predictions = [''.join(map(str, l)) for l in true_predictions]
        true_predictions = [str(tok) if tok else -1 for tok in true_predictions]
        true_labels = [''.join(map(str, l)) for l in true_labels]
        true_labels = [str(tok) if tok else -2 for tok in true_labels]
        if verbose:
            print('One prediction: ', true_predictions[0])
            print('One label: ', true_labels[0])
        classes = set(list(true_labels) + list(true_predictions))
        map_for_eval_fn = {str(real_val): i for i, real_val in enumerate(classes)}
        map_for_eval_fn['-1'] = -1
        map_for_eval_fn[-1] = -1
        true_labels = [map_for_eval_fn[real_val] for real_val in true_labels]
        true_predictions = [map_for_eval_fn[real_val] for real_val in true_predictions]

    if len(items_list_in_dataset_validation) == len(true_labels):
        items_list_in_dataset = items_list_in_dataset_validation
    elif len(items_list_in_dataset_test) == len(true_labels):
        items_list_in_dataset = items_list_in_dataset_test
    else:
        print("Warning: Length of items list in dataset does not match number of predictions. Skipping per-item F1 calculation.")
        calc_per_item = False
    if calc_per_item:
        f1_scores_per_item = {}
        unique_items = set(items_list_in_dataset)
        for item in unique_items:
            item_indices = [i for i, x in enumerate(items_list_in_dataset) if x == item]
            item_preds = [true_predictions[i] for i in item_indices]
            item_labels = [true_labels[i] for i in item_indices]
            f1_item = f1.compute(predictions=item_preds, references=item_labels, average='macro')
            f1_scores_per_item[item] = f1_item['f1']

    results_micro = f1.compute(predictions=true_predictions, references=true_labels, average='micro')
    results_macro = f1.compute(predictions=true_predictions, references=true_labels, average='macro')
    results_weighted = f1.compute(predictions=true_predictions, references=true_labels, average='weighted')
    results_per_class = f1.compute(predictions=true_predictions, references=true_labels, average=None)
    label_set = set(true_labels)
    results_per_class = {
                            label: score
                            for label, score in zip(classes, results_per_class['f1'])
                            if label in label_set
                        }
    results = {
        "f1_micro": results_micro["f1"],
        "f1_macro": results_macro["f1"],
        "f1_weighted": results_weighted["f1"],
        "class/f1_results_per_class": results_per_class,
        "items/f1_scores_per_item": f1_scores_per_item if calc_per_item else None
    }
    return results