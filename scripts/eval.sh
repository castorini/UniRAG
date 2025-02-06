for run_name in "CLIP_SF_k1_fewshot_random" "CLIP_SF_k0_zeroshot" "CLIP_SF_k5_fewshot_rag" "CLIP_SF_k1_fewshot_rag" "CLIP_SF_k5_fewshot_random"; do
        CUDA_VISIBLE_DEVICES=5 python eval.py \
        --candidate_path "" \
        --result_dir ./llava_outputs/${run_name} \
        --retrieval_jsonl_path "" \
        --calculate_retriever_metrics 
    done

for run_name in "BLIP_FF_k1_fewshot_rag" "BLIP_FF_k5_fewshot_random" "BLIP_FF_k1_fewshot_random" "BLIP_FF_k5_fewshot_rag"; do
        CUDA_VISIBLE_DEVICES=5 python eval.py \
        --candidate_path "" \
        --result_dir ./llava_outputs/${run_name} \
        --retrieval_jsonl_path "" \
        --calculate_retriever_metrics 
    done