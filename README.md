# UniRAG
This repository is the codebase for paper [UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models](https://arxiv.org/abs/2405.10311), accepted in NAACL2025 Findings.
<p align="center">
<img src="docs/images/unirag_overview.png" alt="UniRAG Overview" style="width:95%;">
</p>

## Prerequirements
This repository works with [UniIR](https://github.com/TIGER-AI-Lab/UniIR)'s BLIP_FF and CLIP_SF retreivers and its M-BEIR dataset. Before proceeding to the next step, you should follow its instructions for [unirag](https://github.com/TIGER-AI-Lab/UniIR/tree/main?tab=readme-ov-file#unirag-evaluation) eval to get the retrieved candidates in a jsonl file. As shown below, each line entry in this file will be a json object containing the query as well as its retrieved candidates and complement candidates that together make the image, text pairs used as in context examples in the generation step.
```json
{
    "query": {
        "qid": "9:6", "query_txt": null, "query_img_path": "mbeir_images/mscoco_images/val2014/COCO_val2014_000000391895.jpg", "query_modality": "image", "query_src_content": null, "pos_cand_list": ["9:2", "9:3", "9:4", "9:5", "9:6"], "neg_cand_list": [], "task_id": 3
    },
    "candidates": [
        {"txt": null, "img_path": "mbeir_images/mscoco_images/val2014/COCO_val2014_000000391895.jpg", "modality": "image", "did": "9:1", "src_content": null},
        ...,
        {"txt": "A man riding on the back of a motorcycle down a road.", "img_path": null, "modality": "text", "did": "9:14174", "src_content": null}
    ],
    "complement_candidates": [
        {"txt": "Man riding a motor bike on a dirt road on the countryside.", "img_path": null, "modality": "text", "did": "9:3", "src_content": null},
        ...,
        {"txt": null, "img_path": "mbeir_images/mscoco_images/val2014/COCO_val2014_000000214369.jpg", "modality": "image", "did": "9:14173", "src_content": null}
    ]
}
```

At end of this step you should have the following base dirs configured:
- `base_mbeir_path` is where you have downloaded and extracted the M-BEIR dataset.
- `base_retrieval_path` is where the UniIR retrieval results will be stored. If you follow the UniIR instructions it would be a path ending in `/retrieval_results`

## Installation
We recommend separate conda environments for retrieval(UniIR) vs generation (UniRAG).


## Environment variables
API Keys and other env variables must be in `.env.local`
```
# Open AI 
OPEN_AI_API_KEY=<your api key>
GPT_MODEL_NAME=<gpt model name e.g., gpt-4o-mini>

# Vertex AI
GCLOUD_PROJECT=<GCP project id> 
GCLOUD_REGION=<region e.g, us-central1>
GEMINI_MODEL_NAME=<gemini model e.g., gemini-pro-vision>

# Other variables
CACHE_DIR=<cache dir>
HF_TOKEN=<HF access token>
```


## Caption Generation
`caption_generation_inference.py` generates captions for a the jsonl file created in the previous step.
The prompt mode, retriever model, generator model, number of fewshot examples are all configurable.
To regenerate all our caption generation results (tables 3 and 4) run the following script from the root repository dir.
```bash
bash scripts/caption_generation.sh
```
### Evaluation
`eval_caption_generation.py` measures BLEU1-4, CIDER, ROUGE and SPICE for the generated captions.
To evaluate all results generated in the previous step run the following script from the root repository dir.
```bash
bash scripts/eval_caption_generation.sh
```

## Image Generation

### Query Sampling

### Evaluation
