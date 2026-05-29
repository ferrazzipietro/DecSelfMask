export  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
start_list=(120000 150000 180000 210000 240000  ) # 0 30000 60000 90000  (1050000 1100000 1150000 1300000 1350000 1400000 1450000 1500000 1550000 1600000) # 
end_list=(150000 180000 210000 240000 256794) # 30000 60000 90000 120000 
MODEL="meta-llama/Llama-3.1-8B-Instruct"  # "google/medgemma-27b-text-it" # "meta-llama/Llama-3.1-8B-Instruct" # "google/gemma-3-27b-it" 
data_path="YOUR_PATH/discharge-admission-unsup" # "YOUR_PATH/mesh_unsup" # "Pretrain- -YOUR_PATH_ORG/ClinicalWhole" # "YOUR_PATH_ORG- /crf-second-batch" # "YOUR_PATH_ORG- /e3c-sentences-IT-native" #
data_type="admission" # "raw" # "dyspnea_classification" # "crf_annotation_second" # "ner_annotation" #
keep_n_sequences_per_note=9 # 2
path_save="data/" # "/workspace/tmp_data"

cache_dir="/workspace/.cache" # "/YOUR_PATH/.cache" # 

for i in ${!start_list[@]}; do
    start=${start_list[$i]}
    end=${end_list[$i]}
    echo "Processing notes from $start to $end"
    nohup python b_calculate_scores_unannotated_speedup.py \
    --model_name $MODEL \
    --start_from_note $start \
    --end_at_note $end \
    --data_type $data_type \
    --data_path $data_path \
    --use_which_token mid \
    --keep_n_sequences_per_note $keep_n_sequences_per_note \
    --path_save $path_save \
    --batch_size_lrp 4 \
    --cache_dir $cache_dir &> nohup_b_${start}_${end}.log \
    2>&1 & wait
done

# CUDA_VISIBLE_DEVICES=2 nohup python b_calculate_scores.py  --data_type crf_annotation_second --use_which_token mid --data_path YOUR_PATH_ORG- /crf-second-batch &> nohup_mid