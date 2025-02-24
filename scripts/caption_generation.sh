#!/bin/bash

bs=1000
base_retrieval_path=/mnt/users/s8sharif/UniIR/data/UniIR/retrieval_results # --> TODO: change the base retrieval path.
base_mbeir_path=/mnt/users/s8sharif/M-BEIR # --> TODO: change the base mbeir path
base_output_dir=/mnt/users/s8sharif/UniIR # --> TODO: change the base output dir


# All the retative paths assume running the script from the root UniRAG folder `bash scripts/caption_generation.sh`

# Table 3: caption generation for MSCOCO with all three models
for model_name in "llava"; do # "gemini" "gpt"; do
    for retriever_name in "BLIP_FF" "CLIP_SF"; do
        query_path=$base_retrieval_path"/"$retriever_name"/Large/Instruct/UniRAG/retrieved_candidates/mbeir_mscoco_task3_union_pool_test_k10_retrieved.jsonl"
            image_count=$(wc -l < $query_path)
            echo $image_count
        for params in "1 fewshot_rag" "5 fewshot_rag" "1 fewshot_random" "5 fewshot_random"; do
            set -- $params
            k=$1
            echo $k
            prompt_mode=$2
            echo $prompt_mode
            for ((i = 0; i < $image_count; i += $bs)); do
                next_index=$((i + bs))
                echo "batch "$i"_"$next_index":"
                python ./src/unirag/caption_generation_inference.py \
                --prompt_mode $prompt_mode \
                --max_output_tokens 400 \
                --base_mbeir_path $base_mbeir_path \
                --candidates_file_path $base_mbeir_path/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
                --prompt_file ./src/unirag/prompts/$model_name-caption-prompt-with-examples.txt \
                --k $k \
                --model_name $model_name \
                --index $i"_"$next_index \
                --output_dir $base_output_dir/mscoco_llm_outputs \
                --retrieved_results_path $query_path \
                --retriever_name $retriever_name
            done
        done
        # zeroshot
        prompt_mode=zeroshot
        echo $prompt_mode
        for ((i = 0; i < $image_count; i += $bs)); do
            next_index=$((i + bs))
            echo "batch "$i"_"$next_index":"
            python ./src/unirag/caption_generation_inference.py \
            --prompt_mode $prompt_mode \
            --max_output_tokens 400 \
            --base_mbeir_path $base_mbeir_path \
            --candidates_file_path $base_mbeir_path/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
            --prompt_file ./src/unirag/prompts/caption-prompt-without-examples.txt \
            --k 0 \
            --model_name $model_name \
            --index $i"_"$next_index \
            --output_dir $base_output_dir/mscoco_llm_outputs \
            --retrieved_results_path $query_path \
            --retriever_name $retriever_name
        done
    done
done

# Table 4: caption generation for fashion200k with llava
for retriever_name in "BLIP_FF" "CLIP_SF"; do
    query_path=$base_retrieval_path"/"$retriever_name"/Large/Instruct/UniRAG/retrieved_candidates/mbeir_fashion200k_task3_union_pool_test_k50_retrieved.jsonl"
        image_count=$(wc -l < $query_path)
        echo $image_count
    for params in "1 fewshot_rag" "5 fewshot_rag" "1 fewshot_random" "5 fewshot_random"; do
        set -- $params
        k=$1
        echo $k
        prompt_mode=$2
        echo $prompt_mode
        for ((i = 0; i < $image_count; i += $bs)); do
            next_index=$((i + bs))
            echo "batch "$i"_"$next_index":"
            python ./src/unirag/caption_generation_inference.py \
            --prompt_mode $prompt_mode \
            --max_output_tokens 400 \
            --base_mbeir_path $base_mbeir_path \
            --candidates_file_path $base_mbeir_path/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
            --prompt_file ./src/unirag/prompts/llava-caption-prompt-with-examples.txt \
            --k $k \
            --model_name llava \
            --index $i"_"$next_index \
            --output_dir $base_output_dir/fashion200k_llm_outputs \
            --retrieved_results_path $query_path \
            --retriever_name $retriever_name
        done
    done
    # zeroshot
    prompt_mode=zeroshot
    echo $prompt_mode
    for ((i = 0; i < $image_count; i += $bs)); do
        next_index=$((i + bs))
        echo "batch "$i"_"$next_index":"
        python ./src/unirag/caption_generation_inference.py \
        --prompt_mode $prompt_mode \
        --max_output_tokens 400 \
        --base_mbeir_path $base_mbeir_path \
        --candidates_file_path $base_mbeir_path/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --prompt_file ./src/unirag/prompts/caption-prompt-without-examples.txt \
        --k 0 \
        --model_name llava \
        --index $i"_"$next_index \
        --output_dir $base_output_dir/fashion200k_llm_outputs \
        --retrieved_results_path $query_path \
        --retriever_name $retriever_name
    done
done

