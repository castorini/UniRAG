#!bin/bash

base_retrieval_path=/mnt/users/s8sharif/UniIR/data/UniIR/retrieval_results # --> TODO: change the base retrieval path.
base_mbeir_path=/mnt/users/s8sharif/M-BEIR # --> TODO: change the base mbeir path
base_uniir_dir=/mnt/users/s8sharif/UniIR # --> TODO: change the base uniir dir

# Table 5: image generation for MSCOCO with lavit and emu2
for model_name in "lavit" "emu2"; do
    for retriever_name in "BLIP_FF" "CLIP_SF"; do
        for prompt_mode in "k0_zeroshot" "k1_fewshot_rag" "k5_fewshot_random" "k1_fewshot_random" "k5_fewshot_rag"; do
            run_name=${retriever_name}_${prompt_mode}
            result_dir=${base_uniir_dir}/mscoco_llm_outputs/${model_name}_outputs/${run_name}/generated_images
            if [[ ! -d "$result_dir" ]]; then
                echo "$result_dir doesn't exist!"
                continue
            fi
            echo "Evaluating results at $result_dir..."
            python ./src/unirag/eval_image_generation.py \
                --retrieved-results ${base_uniir_dir}/mscoco_queries/${retriever_name}_queries/retrieved.jsonl \
                --generated-img-dir $result_dir \
                --ground-truth-img-dir ${base_uniir_dir}/mscoco_queries/${retriever_name}_queries/images \
                --output-file ${base_uniir_dir}/mscoco_llm_outputs/${model_name}_outputs/metrics/${run_name}.json \
                --base-image-dir $base_mbeir_path \
                --calculate-metrics-for-retriever # \
                # --compare-generated-vs-retrieved-images  # uncomment to use retrieved images as GT 
        done
    done
done

# Table 6: image generation for fashion200k with lavit
for retriever_name in "BLIP_FF" "CLIP_SF"; do
    for prompt_mode in "k0_zeroshot" "k1_fewshot_rag" "k5_fewshot_random" "k1_fewshot_random" "k5_fewshot_rag"; do
        run_name=${retriever_name}_${prompt_mode}
        result_dir=${base_uniir_dir}/fashion200k_llm_outputs/${model_name}_outputs/${run_name}/generated_images
        if [[ ! -d "$result_dir" ]]; then
            echo "$result_dir doesn't exist!"
            continue
        fi
        echo "Evaluating results at $result_dir..."
        python ./src/unirag/eval_image_generation.py \
            --retrieved-results ${base_uniir_dir}/fashion200k_queries/${retriever_name}_queries/retrieved.jsonl \
            --generated-img-dir $result_dir \
            --ground-truth-img-dir ${base_uniir_dir}/fashion200k_queries/${retriever_name}_queries/images \
            --output-file ${base_uniir_dir}/fashion200k_llm_outputs/${model_name}_outputs/metrics/${run_name}.json \
            --base-image-dir $base_mbeir_path \
            --calculate-metrics-for-retriever # \
            # --compare-generated-vs-retrieved-images  # uncomment to use retrieved images as GT 
    done
done