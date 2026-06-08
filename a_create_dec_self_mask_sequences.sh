export PYTHONPATH="$PYTHONPATH:$PWD"
export  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="meta-llama/Llama-3.2-1B-Instruct" 
data_path="wikimedia/wikipedia" # "NLP-FBK/adapt-sllm-italian-medical-tasks-CP-data" 
data_config="20231101.es"  # "clinical"
data_split="train"
id_column_name="id"
text_column_name="text" # "chunk"
start_pos=0
end_pos=32
max_text_length=100 # set to -1 to use the whole text (recommended, but may slow down the process a lot)

hf_account_name="ferrazzipietro"

keep_n_sequences_per_note=1
path_save="data/" 
cache_dir="/data01/pferrazzi/.cache" 


echo "Processing notes from $start_pos to $end_pos"
# nohup python scripts/calculate_relevance_scores.py \
#     --model_name $MODEL \
#     --start_from_note $start_pos \
#     --max_text_length $max_text_length \
#     --data_config $data_config \
#     --id_column_name $id_column_name \
#     --text_column_name $text_column_name \
#     --end_at_note $end_pos \
#     --data_path $data_path \
#     --use_which_token mid \
#     --keep_n_sequences_per_note $keep_n_sequences_per_note \
#     --path_save $path_save \
#     --batch_size_lrp 4 \
#     --cache_dir $cache_dir &> nohup_b_${start_pos}_${end_pos}.log \
#     2>&1 & wait

echo "Creating training sequences for notes from $start_pos to $end_pos"
input_path="data/a_attention_relevancy_unannotated/$(basename $data_path)/$(basename $MODEL)/combined_mid.json" 

python scripts/create_train_seq.py \
    --input_path $input_path \
    --hf_account_name $hf_account_name \
    --cache_dir $cache_dir