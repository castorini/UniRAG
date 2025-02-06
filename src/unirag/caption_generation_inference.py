import argparse
from datetime import datetime
from enum import Enum
import json
from mimetypes import guess_type
import os
import pathlib
import time
from tqdm import tqdm
from typing import Dict, List, Tuple

from dotenv import load_dotenv
import openai
from openai import AzureOpenAI, OpenAI
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)
import vertexai
from vertexai import generative_models
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig

import generator_prompt
from incontext_example_provider import (
    get_rag_fewshot_examples,
    get_random_fewshot_examples,
)

load_dotenv()

SYSTEM_MESSAGE = "You are an intelligent helpful assistant AI that is an expert in generating captions for provided images."


def infer_gemini(
    images: List[str],
    p_class: generator_prompt.Prompt,
    retrieval_dict: Dict[str, Tuple[str, List[str]]],
    max_output_tokens: int,
    samples_dir: str,
):
    vertexai.init(
        project=os.environ["GCLOUD_PROJECT"], location=os.environ["GCLOUD_REGION"]
    )
    model_name = os.environ["GEMINI_MODEL_NAME"]

    outputs = []
    for idx, image_path in enumerate(images):
        model = GenerativeModel(model_name)
        temp = generative_models.Image.load_from_file(image_path)
        image_data = generative_models.Part.from_image(temp)

        qid, retrieval_results = retrieval_dict.get(image_path)

        fewshot_images = p_class.get_fewshot_image_data(retrieval_results)
        fewshot_captions = p_class.get_fewshot_captions(retrieval_results)
        assert len(fewshot_images) == len(fewshot_captions)

        message = p_class.prepare_gemini_message(len(fewshot_images))
        if fewshot_images:
            json_data = [message]
            content = [message]
            for fewshot_image, fewshot_caption, result in zip(
                fewshot_images, fewshot_captions, retrieval_results
            ):
                json_data.append(result[1])
                json_data.append(fewshot_caption)
                content.append(fewshot_image)
                content.append(fewshot_caption)
            json_data.append(image_path)
            content.append(image_data)
        else:
            json_data = [message, image_path]
            content = [message, image_data]
        if idx < 10:
            with open(f"{samples_dir}/prompt_{idx}.txt", "w") as f:
                json.dump(json_data, f)
                f.write("\n")
        try:
            response = model.generate_content(
                content,
                generation_config=GenerationConfig(max_output_tokens=max_output_tokens),
            )
            print(f"Processed image: {image_path}")
            print(response.text)
            output = response.text
        except:
            output = ""
        outputs.append(
            {"qid": qid, "image": image_path, "prompt": message, "response": output}
        )
        print("-" * 79)
    return outputs


def infer_gpt(
    images: List[str],
    p_class: generator_prompt.Prompt,
    retrieval_dict: Dict[str, Tuple[str, List[str]]],
    max_output_tokens: int,
    samples_dir: str,
):
    azure_openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
    azure_openai_api_base = os.environ["AZURE_OPENAI_API_BASE"]
    open_ai_api_key = os.environ["OPEN_AI_API_KEY"]
    model_name = os.environ["MODEL_NAME"]

    if all([open_ai_api_key, azure_openai_api_base, azure_openai_api_version]):
        client = AzureOpenAI(
            api_key=open_ai_api_key,
            api_version=azure_openai_api_version,
            azure_endpoint=azure_openai_api_base,
        )
    else:
        client = OpenAI(api_key=open_ai_api_key)

    outputs = []
    for idx, image_path in enumerate(images):
        qid, retrieval_results = retrieval_dict.get(image_path)
        encoded_query_image_url = p_class.encode_image_as_url(image_path)
        fewshot_image_urls = p_class.get_fewshot_image_urls(retrieval_results)
        fewshot_captions = p_class.get_fewshot_captions(retrieval_results)
        assert len(fewshot_image_urls) == len(fewshot_captions)
        message = p_class.prepare_gpt_message(len(fewshot_image_urls))
        if len(fewshot_image_urls):
            image_content = [
                {"type": "image_url", "image_url": {"url": image_url}}
                for image_url in fewshot_image_urls
            ]
            caption_content = [
                {"type": "text", "text": f"Caption [{index + 1}]: {caption}"}
                for index, caption in enumerate(fewshot_captions)
            ]
            fewshot_content = []
            for image, caption in zip(image_content, caption_content):
                fewshot_content.append(image)
                fewshot_content.append(caption)
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": [
                        *fewshot_content,
                        {
                            "type": "image_url",
                            "image_url": {"url": encoded_query_image_url},
                        },
                        {"type": "text", "text": message},
                    ],
                },
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": encoded_query_image_url},
                        },
                        {"type": "text", "text": message},
                    ],
                },
            ]
        if idx < 10:
            with open(f"{samples_dir}/prompt_{idx}.txt", "w") as f:
                json.dump(messages, f)
                f.write("\n")
        while True:
            # Try calling the inference in a while loop to avoid errors related to rate limits, API unreliablity, connection issues, etc.
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_output_tokens,
                )
                output = (
                    response.choices[0].message.content.lower()
                    if response.choices[0].message.content
                    else ""
                )
                break
            except Exception as e:
                # Repeating the request won't help if OpenAI refuses to create the caption due to policy violation.
                if "ResponsibleAIPolicyViolation" in str(e):
                    output = ""
                    break
                print(f"Encountered {e}")
                # Wait for a second before retrying the request
                time.sleep(1)

        print(f"Processed image: {image_path}")
        print(output)
        outputs.append(
            {"qid": qid, "image": image_path, "prompt": message, "response": output}
        )
        print("-" * 79)
    return outputs


def infer_llava(
    images: List[str],
    p_class: generator_prompt.Prompt,
    retrieval_dict: Dict[str, Tuple[str, List[str]]],
    max_output_tokens: int,
    samples_dir: str,
    model_name: str = "llava-hf/llava-v1.6-vicuna-7b-hf",
    bs: int = 1,
):
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=os.environ["CACHE_DIR"],
    )
    processor = LlavaNextProcessor.from_pretrained(
        model_name, use_fast=True, cache_dir=os.environ["CACHE_DIR"]
    )
    # recommended for batch mode generation
    processor.tokenizer.padding_side = "left"

    prompts = []
    input_images = []
    qids = []

    for idx, image_path in enumerate(images):
        qid, retrieval_results = retrieval_dict.get(image_path)
        print(f"retrieval_results: {retrieval_results}")
        image = p_class.merge_images(retrieval_results, image_path)
        input_images.append(image)
        message = p_class.prepare_llava_message(retrieval_results)
        if idx < 10:
            image.save(f"{samples_dir}/concat_img_{idx}.jpg")
            with open(f"{samples_dir}/prompt_{idx}.txt", "w") as f:
                f.write(f"USER: <image>\n{SYSTEM_MESSAGE}\n{message}\nASSISTANT:")
        prompts.append(f"USER: <image>\n{SYSTEM_MESSAGE}\n{message}\nASSISTANT:")
        qids.append(qid)

    outputs = []
    for i in tqdm(range(0, len(prompts), bs), desc="Batching inputs"):
        inputs = processor(
            prompts[i : i + bs],
            images=input_images[i : i + bs],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        output = model.generate(**inputs, max_new_tokens=max_output_tokens)
        generated_text = processor.batch_decode(output, skip_special_tokens=True)
        for text, image_path, prompt, qid in tqdm(
            zip(
                generated_text,
                images[i : i + bs],
                prompts[i : i + bs],
                qids[i : i + bs],
            )
        ):
            print(f"Processed image: {image_path}")
            print(f"prompt: {prompt}")
            print(text.split("ASSISTANT:")[-1])
            outputs.append(
                {
                    "qid": qid,
                    "image": image_path,
                    "prompt": prompt,
                    "response": text.split("ASSISTANT:")[-1],
                }
            )
            print("-" * 79)

    return outputs


def infer_blip(
    images: List[str],
    p_class: generator_prompt.Prompt,
    retrieval_dict: Dict[str, Tuple[str, List[str]]],
    max_output_tokens: int,
    samples_dir: str,
    model_name: str = "Salesforce/blip2-flan-t5-xl",
    bs: int = 4,
):
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=os.environ["CACHE_DIR"],
    )
    processor = Blip2Processor.from_pretrained(
        model_name, use_fast=True, cache_dir=os.environ["CACHE_DIR"]
    )

    prompts = []
    input_images = []
    qids = []

    for image_path in images:
        image = Image.open(image_path)
        keep = image.copy()
        input_images.append(keep)
        image.close()

        qid, retrieval_results = retrieval_dict.get(os.path.basename(image))
        message = p_class.prepare_message(retrieval_results)
        prompts.append(message)
        qids.append(qid)

    inputs = processor(
        prompts, images=input_images, padding=True, return_tensors="pt"
    ).to("cuda")
    outputs = []
    for i in tqdm(range(0, len(prompts), bs), desc="Batching inputs"):
        inputs = processor(
            prompts[i : i + bs],
            images=input_images[i : i + bs],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        output = model.generate(**inputs, max_new_tokens=max_output_tokens)
        generated_text = processor.batch_decode(output, skip_special_tokens=True)
        for text, image_path, prompt, qid in tqdm(
            zip(
                generated_text,
                images[i : i + bs],
                prompts[i : i + bs],
                qids[i : i + bs],
            )
        ):
            print(f"Processed image: {image_path}")
            print(text)
            outputs.append(
                {"qid": qid, "image": image_path, "prompt": prompt, "response": text}
            )
            print("-" * 79)
    return outputs


class PromptMode(Enum):
    ZEROSHOT = "zeroshot"
    FEWSHOT_RANDOM = "fewshot_random"
    FEWSHOT_RAG = "fewshot_rag"

    def __str__(self):
        return self.value


def main(args):
    if args.k == 0 and "-with-" in args.prompt_file:
        raise ValueError("Invalid template file for zero-shot inference.")
    elif args.k > 0 and "-without-" in args.prompt_file:
        raise ValueError("Invalid template file for few-shot inference.")

    max_output_tokens = args.max_output_tokens
    infer_mapping = {
        "gpt": infer_gpt,
        "gemini": infer_gemini,
        "llava": infer_llava,
        "blip": infer_blip,
    }
    if args.prompt_mode == PromptMode.FEWSHOT_RANDOM:
        images, retrieval_dict = get_random_fewshot_examples(
            args.retrieved_results_path,
            args.base_mbeir_path,
            args.candidates_file_path,
            args.index,
        )
    else:
        images, retrieval_dict = get_rag_fewshot_examples(
            args.retrieved_results_path, args.base_mbeir_path, args.index
        )

    p_class = generator_prompt.Prompt(args.prompt_file, args.k)
    result_dir = os.path.join(
        args.output_dir,
        f"{args.model_name}_outputs",
        f"{args.retriever_name}_k{args.k}_{args.prompt_mode}",
    )
    os.makedirs(result_dir, exist_ok=True)
    samples_dir = os.path.join(result_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    result = infer_mapping[args.model_name](
        images, p_class, retrieval_dict, max_output_tokens, samples_dir
    )
    output_path = os.path.join(
        result_dir, f"{args.index}_{datetime.isoformat(datetime.now())}.json"
    )
    with open(output_path, "w") as outfile:
        json.dump(result, outfile)
    print(f"Output file at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_mode", type=PromptMode, choices=list(PromptMode))
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=500,
        help="Maximum number of generated output tokens",
    )
    parser.add_argument("--base_mbeir_path", type=str, help="The base path to MBEIR")
    parser.add_argument(
        "--candidates_file_path", required=True, help="candidates file path"
    )
    parser.add_argument("--prompt_file", required=True, help="Prompt file")
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Number of retrieved examples included in the prompt",
    )
    parser.add_argument("--model_name", default="gpt")
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
    args = parser.parse_args()
    main(args)
