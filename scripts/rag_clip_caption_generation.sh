#!/bin/bash
export CACHE_DIR=""
export TOKEN=""

bs=1000
uniir_query_path=""
image_count=$(wc -l < $query_path)
echo $image_count

for ((i = 0; i < $image_count; i += $bs)); do
    next_index=$((i + bs))
    echo "batch "$i"_"$next_index":"
    python caption_generation_inference.py \
    --prompt_mode fewshot_random \
    --max_output_tokens 400 \
    --base_mbeir_path "" \
    --candidates_file_path "" \
    --prompt_file prompts/llava-caption-prompt-with-examples.txt \
    --k 5 \
    --model_name llava \
    --index $i"_"$next_index \
    --output_dir ./ \
    --retrieved_results_path $query_path \
    --retriever_name "BLIP_FF"
done