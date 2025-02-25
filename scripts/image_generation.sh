#!/bin/bash

bs=1000
base_query_path=/mnt/users/s8sharif/UniIR/ # --> TODO: change the base query path to where prepare_dataset.sh will store the query folders.
base_mbeir_path=/mnt/users/s8sharif/M-BEIR # --> TODO: change the base mbeir path
base_output_dir=/mnt/users/s8sharif/UniIR # --> TODO: change the base output dir


# All the retative paths assume running the script from the root UniRAG folder `bash scripts/image_generation.sh`

# Table 5: image generation for MSCOCO with lavit and emu2
for model_name in "lavit" "emu2"; do
    model_path="/mnt/users/s8sharif/Emu2-Gen" # --> TODO: change to the local file where you have downloaded BAAI/Emu2-Gen hf repository.
    for retriever_name in "BLIP_FF" "CLIP_SF"; do
        query_path=$base_query_path"/mscoco_queries/"$retriever_name"_queries/retrieved.jsonl"
            caption_count=$(wc -l < $query_path)
            echo $caption_count
        for params in "0 zeroshot" "1 fewshot_rag" "5 fewshot_rag" "1 fewshot_random" "5 fewshot_random"; do
            set -- $params
            k=$1
            echo $k
            prompt_mode=$2
            echo $prompt_mode
            for ((i = 0; i < $caption_count; i += $bs)); do
                next_index=$((i + bs))
                echo "batch "$i"_"$next_index":"
                python ./src/unirag/image_generation_inference.py \
                    --prompt_mode $prompt_mode \
                    --base_mbeir_path $base_mbeir_path \
                    --candidates_file_path $base_mbeir_path/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
                    --k $k \
                    --model_name $model_name \
                    --index $i"_"$next_index \
                    --output_dir  $base_output_dir/mscoco_llm_outputs \
                    --retrieved_results_path $query_path \
                    --retriever_name $retriever_name \
                    --model_path $model_path # ignored for lavit
                    break
            done
            break
        done
        break
    done
    break
done

# Table 6: image generation for fashion200k with lavit
for retriever_name in "BLIP_FF" "CLIP_SF"; do
    query_path=$base_query_path"/fashion200k_queries/"$retriever_name"_queries/retrieved.jsonl"
        caption_count=$(wc -l < $query_path)
        echo $caption_count
    for params in "0 zeroshot" "1 fewshot_rag" "5 fewshot_rag" "1 fewshot_random" "5 fewshot_random"; do
        set -- $params
        k=$1
        echo $k
        prompt_mode=$2
        echo $prompt_mode
        for ((i = 0; i < $caption_count; i += $bs)); do
            next_index=$((i + bs))
            echo "batch "$i"_"$next_index":"
            python ./src/unirag/image_generation_inference.py \
                --prompt_mode $prompt_mode \
                --base_mbeir_path $base_mbeir_path \
                --candidates_file_path $base_mbeir_path/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
                --k $k \
                --model_name lavit \
                --index $i"_"$next_index \
                --output_dir  $base_output_dir/fashion200k_llm_outputs \
                --retrieved_results_path $query_path \
                --retriever_name $retriever_name
        done
    done
done
