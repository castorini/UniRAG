#!/bin/bash

base_retrieval_path=/mnt/users/s8sharif/UniIR/data/UniIR/retrieval_results # --> TODO: change the base retrieval path.
base_mbeir_path=/mnt/users/s8sharif/M-BEIR # --> TODO: change the base mbeir path
base_output_dir=/mnt/users/s8sharif/UniIR # --> TODO: change the base output dir

# Table 3: caption generation for MSCOCO with all three models
for model_name in "llava" "gpt" "gemini"; do
    for retriever_name in "BLIP_FF" "CLIP_SF"; do
        for prompt_mode in "k0_zeroshot" "k1_fewshot_rag" "k5_fewshot_random" "k1_fewshot_random" "k5_fewshot_rag"; do
            result_dir=${base_output_dir}/mscoco_llm_outputs/${model_name}_outputs/${retriever_name}_${prompt_mode}
            if [[ ! -d "$result_dir" ]]; then
                echo "$result_dir doesn't exist!"
                continue
            fi
            echo "Evaluating results at $result_dir..."
            python ./src/unirag/eval_caption_generation.py \
            --candidate_path ${base_mbeir_path}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
            --result_dir $result_dir \
            --retrieval_jsonl_path ${base_retrieval_path}/${retriever_name}/Large/Instruct/UniRAG/retrieved_candidates/mbeir_mscoco_task3_union_pool_test_k10_retrieved.jsonl \
            --calculate_retriever_metrics 
        done
    done
done

# Table 4: caption generation for fashion200k with llava
for retriever_name in "BLIP_FF" "CLIP_SF"; do
    for prompt_mode in "k0_zeroshot" "k1_fewshot_rag" "k5_fewshot_random" "k1_fewshot_random" "k5_fewshot_rag"; do
        result_dir=${base_output_dir}/fashion200k_llm_outputs/llava_outputs/${retriever_name}_${prompt_mode}
        if [[ ! -d "$result_dir" ]]; then
                echo "$result_dir doesn't exist!"
                continue
            fi
            echo "Evaluating results at $result_dir..."
        python ./src/unirag/eval_caption_generation.py \
        --candidate_path ${base_mbeir_path}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --result_dir $result_dir \
        --retrieval_jsonl_path ${base_retrieval_path}/${retriever_name}/Large/Instruct/UniRAG/retrieved_candidates/mbeir_fashion200k_task3_union_pool_test_k50_retrieved.jsonl \
        --calculate_retriever_metrics 
    done
done
