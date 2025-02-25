import argparse
from datetime import datetime
from enum import Enum
import json
import os
from PIL import Image
from tqdm import tqdm
from typing import Dict, Tuple

from diffusers import DiffusionPipeline
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import build_model
from incontext_example_provider import (
    get_rag_fewshot_examples,
    get_random_fewshot_examples,
)


def build_lavit_model():
    # Todo: make all values configurable
    model_path = "models/lavit"

    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda")

    # For Multi-modal Image Generation, must set `load_tokenizer=True` to load the tokenizer to tokenize input image.
    # If you have already install xformers, set `use_xformers=True` to save the GPU memory (Xformers is not supported on V100 GPU)
    # If you have already download the checkpoint, set `local_files_only=True`` to avoid auto-downloading from remote
    model_dtype = "bf16"
    model = build_model(
        model_path=model_path,
        model_dtype=model_dtype,
        check_safety=False,
        use_xformers=True,
        understanding=False,
        load_tokenizer=True,
    )
    model = model.to(device)
    print("Building Model Finsished")
    return model


def infer_lavit(
    captions,
    retrieval_dict: Dict[str, Tuple[str, str]],
    model,
    output_image_dir,
):
    outputs = []
    for caption in tqdm(captions, desc="Generating images"):
        try:
            qid, candidates = retrieval_dict[caption]
            prompts = []
            for txt, img_path in candidates:
                prompts.append((txt, "text"))
                prompts.append((img_path, "image"))
            prompts.append((caption, "text"))
            # Todo: make params configurable
            height, width = 1024, 1024
            torch_dtype = torch.bfloat16
            with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
                images = model.multimodal_synthesis(
                    prompts,
                    width=width,
                    height=height,
                    guidance_scale_for_llm=4.0,
                    num_return_images=1,
                    num_inference_steps=25,
                    top_k=50,
                )
        except Exception as e:
            print(f"Exception: processing {qid} with {prompts} caused {e}")
            continue
        output_img_path = os.path.join(
            output_image_dir, f"{qid}_{datetime.isoformat(datetime.now())}.jpg"
        )
        images[0].save(output_img_path)
        print(f"Processed caption: {caption}")
        print(f"saved generated image at {output_img_path}")
        outputs.append(
            {
                "qid": qid,
                "caption": caption,
                "prompt": prompts,
                "response": output_img_path,
            }
        )
        print("-" * 79)

    return outputs


def infer_emu2(
    captions,
    retrieval_dict: Dict[str, Tuple[str, str]],
    model_path,
    output_image_dir,
):
    multimodal_encoder = AutoModelForCausalLM.from_pretrained(
        f"{model_path}/multimodal_encoder",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="bf16",
    )
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")

    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        custom_pipeline="pipeline_emu2_gen",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="bf16",
        multimodal_encoder=multimodal_encoder,
        tokenizer=tokenizer,
    )
    pipe.to("cuda")
    outputs = []
    for caption in tqdm(captions, desc="Generating images"):
        try:
            qid, candidates = retrieval_dict[caption]
            prompts = []
            for txt, img_path in candidates:
                prompts.append(txt)
                image = Image.open(img_path)
                keep = image.copy()
                image.close()
                prompts.append(keep)
            prompts.append(caption)

            ret = pipe(prompts)
        except Exception as e:
            print(f"Exception: processing {qid} with {prompts} caused {e}")
            continue
        output_img_path = os.path.join(
            output_image_dir, f"{qid}_{datetime.isoformat(datetime.now())}.jpg"
        )
        ret.image.save(output_img_path)
        print(f"Processed caption: {caption}")
        print(f"saved generated image at {output_img_path}")
        outputs.append(
            {
                "qid": qid,
                "caption": caption,
                "prompt": prompts,
                "response": output_img_path,
            }
        )
        print("-" * 79)

    return outputs


def main(args):
    if args.prompt_mode == PromptMode.FEWSHOT_RANDOM:
        captions, retrieval_dict = get_random_fewshot_examples(
            args.retrieved_results_path,
            args.base_mbeir_path,
            args.candidates_file_path,
            args.index,
            args.k,
        )
    else:
        captions, retrieval_dict = get_rag_fewshot_examples(
            args.retrieved_results_path, args.base_mbeir_path, args.index, args.k
        )
    infer_mapping = {"lavit": infer_lavit, "emu2": infer_emu2}

    model = args.model_name
    if model == "lavit":
        model = build_lavit_model()
    else:
        model = args.model_path
    result_dir = os.path.join(
        args.output_dir,
        f"{args.model_name}_outputs",
        f"{args.retriever_name}_k{args.k}_{args.prompt_mode}",
    )
    os.makedirs(result_dir, exist_ok=True)
    output_image_dir = os.path.join(result_dir, "generated_images")
    os.makedirs(output_image_dir, exist_ok=True)
    result = infer_mapping[args.model_name](
        captions, retrieval_dict, model, output_image_dir
    )
    output_path = os.path.join(
        result_dir, f"{args.index}_{datetime.isoformat(datetime.now())}.json"
    )
    with open(output_path, "w") as outfile:
        json.dump(result, outfile)
    print(f"Output file at: {output_path}")


class PromptMode(Enum):
    ZEROSHOT = "zeroshot"
    FEWSHOT_RANDOM = "fewshot_random"
    FEWSHOT_RAG = "fewshot_rag"

    def __str__(self):
        return self.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_mode", type=PromptMode, choices=list(PromptMode))
    parser.add_argument("--base_mbeir_path", type=str, help="The base path to MBEIR")
    parser.add_argument(
        "--candidates_file_path", required=True, help="candidates file path"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Number of retrieved examples included in the prompt",
    )
    parser.add_argument("--model_name", default="lavit")
    parser.add_argument(
        "--index", default="full", help="Add start end indices in x_y format"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Base directory to store llm outputs, the output dir would be: output_dir/'model_name'_outputs/'retriever_name'_k",
    )
    parser.add_argument(
        "--retrieved_results_path",
        required=True,
        help="path to the jsonl file containing query + candidates pairs",
    )
    parser.add_argument(
        "--retriever_name",
        required=True,
        help="Name of the retriever that has retrieved input candidates",
    )
    parser.add_argument("--model_path", default="", help="Model path, used for emu2")
    args = parser.parse_args()
    main(args)
