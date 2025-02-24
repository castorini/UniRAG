base_retrieval_path=/mnt/users/s8sharif/UniIR/data/UniIR/retrieval_results # --> TODO: change the base retrieval path.
base_mbeir_path=/mnt/users/s8sharif/M-BEIR # --> TODO: change the base mbeir path
base_output_dir=/mnt/users/s8sharif/UniIR # --> TODO: change the base output dir

# Do not sample for the fashion200k dataset
for retriever_name in "BLIP_FF" "CLIP_SF"; do
    python ./src/unirag/prepare_dataset.py \
        --output-dir $base_output_dir/fashion200k_queries/$retriever_name"_queries" \
        --dataset-queries  $base_retrieval_path/$retriever_name/Large/Instruct/UniRAG/retrieved_candidates/mbeir_fashion200k_task0_union_pool_test_k50_retrieved.jsonl \
        --candidates-file  $base_mbeir_path/cand_pool/local/mbeir_fashion200k_task0_cand_pool.jsonl \
        --candidates-base-dir $base_mbeir_path
 done


# For MSCOCO sample the caption queries for BLIP_FF, use the same qids for CLIP_FF rather than resampling.
python ./src/unirag/prepare_dataset.py \
    --output-dir $base_output_dir/mscoco_queries/BLIP_FF_queries \
    --dataset-queries  $base_retrieval_path/BLIP_FF/Large/Instruct/UniRAG/retrieved_candidates/mbeir_mscoco_task0_union_pool_test_k10_retrieved.jsonl \
    --candidates-file  $base_mbeir_path/cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl \
    --candidates-base-dir $base_mbeir_path \
    --do-sample

python ./src/unirag/prepare_dataset.py \
    --output-dir $base_output_dir/mscoco_queries/CLIP_SF_queries \
    --dataset-queries  $base_retrieval_path/CLIP_SF/Large/Instruct/UniRAG/retrieved_candidates/mbeir_mscoco_task0_union_pool_test_k10_retrieved.jsonl \
    --candidates-file  $base_mbeir_path/cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl \
    --candidates-base-dir $base_mbeir_path \
    --sampled-qids-file $base_output_dir/mscoco_queries/BLIP_FF_queries/sampled_qids.json
