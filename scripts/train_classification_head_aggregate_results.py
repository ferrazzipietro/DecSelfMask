from argparse import ArgumentParser
import os
import pandas as pd

def compute_f1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else -1

def get_f1_scores_by_cathegory(sum_dict, cathegories):
    sum_dict_by_cathegory = {cat: {'fp': sum_dict.get(f"{cat}_fp", 0), 'fn': sum_dict.get(f"{cat}_fn", 0), 'tp': sum_dict.get(f"{cat}_tp", 0)} for cat in cathegories}
    f1_scores_by_cathegory = {cat: compute_f1(sum_dict_by_cathegory[cat]['tp'], sum_dict_by_cathegory[cat]['fp'], sum_dict_by_cathegory[cat]['fn']) for cat in cathegories}
    return f1_scores_by_cathegory

def get_total_tp_fp_fn(sum_dict, cathegories):
    total_tp = sum(sum_dict.get(f"{cat}_tp", 0) for cat in cathegories)
    total_fp = sum(sum_dict.get(f"{cat}_fp", 0) for cat in cathegories)
    total_fn = sum(sum_dict.get(f"{cat}_fn", 0) for cat in cathegories)
    return total_tp, total_fp, total_fn


def get_proportion_by_cathegory(sum_dict, cathegories):
    cardinality_by_cathegory = {cat: sum_dict.get(f"{cat}_tp", 0) + sum_dict.get(f"{cat}_fp", 0) + sum_dict.get(f"{cat}_fn", 0) for cat in cathegories}
    total = sum(cardinality_by_cathegory.values())
    return {cat: cardinality_by_cathegory[cat] / total if total > 0 else 0 for cat in cathegories}



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--split_name", type=str, required=True)
    args = parser.parse_args()
    data_path = f"data/d_classification_head/eval/{args.model_name}/{args.split_name}" 
    os.makedirs(data_path, exist_ok=True)

    results_df_end_of_training = pd.read_excel(data_path+f"/results_table_sft_class_per_item.xlsx")
    results_df_best_macro_f1 = pd.read_excel(data_path+f"/results_table_sft_class_per_item_best.xlsx")
    df_dict = {
        "end_of_training": results_df_end_of_training,
        "best_macro_f1": results_df_best_macro_f1,
    }
    for name, results_df in df_dict.items():
        # find all coulmns ending with _fp, _fn, or _tp
        fp_columns = [col for col in results_df.columns if col.endswith("_fp")]
        fn_columns = [col for col in results_df.columns if col.endswith("_fn")]
        tp_columns = [col for col in results_df.columns if col.endswith("_tp")]
        sum_dict = {
            **results_df[fp_columns].sum().to_dict(),
            **results_df[fn_columns].sum().to_dict(),
            **results_df[tp_columns].sum().to_dict(),
        }
        cathegories = set(c.split("_")[0] for c in fp_columns + fn_columns + tp_columns)
        
        total_tp, total_fp, total_fn = get_total_tp_fp_fn(sum_dict, cathegories)
        f1_scores_by_cathegory = get_f1_scores_by_cathegory(sum_dict, cathegories)
        proportion_by_cathegory = get_proportion_by_cathegory(sum_dict, cathegories)
        # Macro F1 over categories with any support.
        nonempty_cats = [cat for cat in cathegories if (sum_dict.get(f"{cat}_tp", 0) + sum_dict.get(f"{cat}_fp", 0) + sum_dict.get(f"{cat}_fn", 0)) > 0]
        macro_f1 = sum(f1_scores_by_cathegory[cat] for cat in nonempty_cats) / len(nonempty_cats) if nonempty_cats else -1
        micro_f1 = compute_f1(tp=total_tp, fp=total_fp, fn=total_fn)
        weighted_f1 = sum([proportion_by_cathegory[cat] * f1_scores_by_cathegory[cat] for cat in cathegories])
        base_data = results_df.iloc[0].to_dict()
        our = {
            'model_name': args.model_name,
            'unsup': 'unsup' in args.model_name,
            'MAX_VALIDATION_EXAMPLES': base_data.get('MAX_VALIDATION_EXAMPLES', None),
            'USE_SAME_SUBSET_OF_SFT': base_data.get('USE_SAME_SUBSET_OF_SFT', None),
            'date_time': base_data.get('date_time', None),
            'macro': macro_f1,
            'micro': micro_f1,
            'weighted': weighted_f1,
        }
        print('FINAL RESULTS: ', our)
        name = "summary_best_macro_f1.xlsx" if "best" in name else "summary_end_of_training.xlsx"
        pd.DataFrame([our]).to_excel(f"{data_path}/{name}", index=False)