import argparse
import cv2
import numpy as np
import os
from typing import List
import json

from tqdm import tqdm
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieved-results",
        type=str,
        required=True,
        help="The jsonl file containing the query and candidates from the retreiver, each query will have a prompt",
    )
    parser.add_argument(
        "--generated-img-dir",
        type=str,
        required=True,
        help="root dir containing subdirs, each subdir contains a subset of generated images",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="the clip model name to use for calculating clip metric",
    )
    parser.add_argument(
        "--ground-truth-img-dir",
        type=str,
        required=True,
        help="The directory where the ground truth images are stored",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="The batch size for calculating metrics in batches",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Where to store the json metrics file",
    )
    parser.add_argument(
        "--calculate-metrics-for-retriever",
        action="store_true",
        help="Whether or not to calculate the metrics for the retrieved candidates",
    )
    parser.add_argument(
        "--compare-generated-vs-retrieved-images",
        action="store_true",
        help="Whether or not to report the FID of comparing generated images vs top 1 retrieved images",
    )
    parser.add_argument(
        "--base-image-dir",
        type=str,
        default="",
        help="The base dir where retrieved candidate images are stored, only used when --calculate-metrics-for-retriever or --compare-generated-vs-retrieved-images is enabled",
    )
    return parser.parse_args()


def load_prompts(retrieved_results: str):
    prompts = []
    qids = []
    with open(retrieved_results, "r") as f:
        for l in f:
            obj = json.loads(l)
            prompt = obj["query"]["query_txt"]
            qid = obj["query"]["qid"]
            prompts.append(prompt)
            qids.append(qid)
    return prompts, qids


def load_images(dir: str, qids: List[str]):
    images = []
    # Sort generated image files by the qid, add noise for missing images
    imgs_dict = {}
    imgs = os.listdir(dir)
    for img in imgs:
        qid = img[:-4].split("_")[0]
        qid = qid.replace("-", ":")
        imgs_dict[qid] = img
    print(len(imgs_dict))

    for qid in tqdm(qids):
        if qid in imgs_dict:
            img = cv2.imread(os.path.join(dir, imgs_dict[qid]))
            resize_img = cv2.resize(img, (224, 224))
            images.append(resize_img)
            del img
        else:
            print(f"didn't find generated image for qid: {qid}")
            images.append(np.random.randint(256, size=(224, 224, 3)).astype("uint8"))
    return images


def load_retrieved_images(retrieved_results: str, base_image_dir: str):
    images = []
    with open(retrieved_results, "r") as f:
        for l in f:
            obj = json.loads(l)
            top_cand = obj["candidates"][0]
            if top_cand["img_path"]:
                img_file = os.path.join(base_image_dir, top_cand["img_path"])
                assert os.path.exists(
                    img_file
                ), f"image file doesn't exist at {img_file}"
                img = cv2.imread(img_file)
                resize_img = cv2.resize(img, (224, 224))
                images.append(resize_img)
                del img
            elif obj["complement_candidates"][0]["img_path"]:
                img_file = os.path.join(
                    base_image_dir, obj["complement_candidates"][0]["img_path"]
                )
                assert os.path.exists(
                    img_file
                ), f"image file doesn't exist at {img_file}"
                img = cv2.imread(img_file)
                resize_img = cv2.resize(img, (224, 224))
                images.append(resize_img)
                del img
            else:
                # Add noise for queries where k=1 retrieved candidate is not an image
                print("added random noise as image")
                images.append(
                    np.random.randint(256, size=(224, 224, 3)).astype("uint8")
                )
    return images


def calculate_clip_score(images, prompts, batch_size, model_name, device):
    assert len(images) == len(prompts), "The number of images and prompts must be equal"
    metric = CLIPScore(model_name_or_path=model_name)
    for i in tqdm(range(0, len(images), batch_size)):
        image_chunk = torch.from_numpy(
            np.asarray(images[i : i + batch_size]).astype("uint8")
        )
        prompt_chunk = prompts[i : i + batch_size]
        metric.update(image_chunk.permute(0, 3, 1, 2).to(device), prompt_chunk)
    return metric.compute()


def calculate_fid(real_images, gen_images, batch_size, device):
    assert len(real_images) == len(
        gen_images
    ), "The number of ground truth and generated images must be equal"
    metric = FrechetInceptionDistance()
    for i in tqdm(range(0, len(real_images), batch_size)):
        real = np.asarray(real_images[i : i + batch_size]).astype("uint8")
        real = torch.from_numpy(real)
        real = real.permute(0, 3, 1, 2).to(device)
        metric.update(real, real=True)
        gen = np.asarray(gen_images[i : i + batch_size]).astype("uint8")
        gen = torch.from_numpy(gen)
        gen = gen.permute(0, 3, 1, 2).to(device)
        metric.update(gen, real=False)
    return metric.compute()


def calculate_inception_score(images, batch_size, device):
    images = [cv2.resize(img, (299, 299)) for img in images]
    metric = InceptionScore()
    for i in tqdm(range(0, len(images), batch_size)):
        image_chunk = torch.from_numpy(
            np.asarray(images[i : i + batch_size]).astype("uint8")
        )
        metric.update(image_chunk.permute(0, 3, 1, 2).to(device))
    return metric.compute()


def main(args):
    # assert torch.cuda.is_available(), "torch cuda is required!"
    device = "cpu"
    torch.set_default_device(device)
    print("Loading prompts ...")
    prompts, qids = load_prompts(args.retrieved_results)
    print("Loading ground truth images ...")
    real_images = load_images(args.ground_truth_img_dir, qids)
    print("Loading generated images ...")
    gen_images = load_images(args.generated_img_dir, qids)
    retrieved_images = None
    if (
        args.calculate_metrics_for_retriever
        or args.compare_generated_vs_retrieved_images
    ):
        print("Loading retrieved images ...")
        retrieved_images = load_retrieved_images(
            args.retrieved_results, args.base_image_dir
        )
    assert len(real_images) == len(gen_images) == len(prompts)

    print("Calculating FID:")
    fid = calculate_fid(real_images, gen_images, args.batch_size, device)
    print("Calculating CLIPScore:")
    clip_score = calculate_clip_score(
        gen_images, prompts, args.batch_size, args.clip_model, device
    )
    # calculate inception score last since it resizes the generated images to 299x299
    print("Calculating IS:")
    IS = calculate_inception_score(gen_images, args.batch_size, device)
    metrics = {
        "FID_VS_GT": round(fid.detach().cpu().numpy().tolist(), 2),
        "CLIPScore": round(clip_score.detach().cpu().numpy().tolist(), 2),
        "IS": round(IS[0].detach().cpu().numpy().tolist(), 2),
        "IS_Standard_Deviation": round(IS[1].detach().cpu().numpy().tolist(), 2),
    }
    if args.compare_generated_vs_retrieved_images:
        print("Calculating FID vs retrieved images:")
        retr_fid = calculate_fid(retrieved_images, gen_images, args.batch_size, device)
        metrics["FID_VS_RETRIEVED"] = round(retr_fid.detach().cpu().numpy().tolist(), 2)
    print(metrics.__repr__())
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(metrics, f)

    if args.calculate_metrics_for_retriever:
        print("Calculating metrics for retrieved candidates")
        print("Calculating FID:")
        fid = calculate_fid(real_images, retrieved_images, args.batch_size, device)
        print("Calculating CLIPScore:")
        clip_score = calculate_clip_score(
            retrieved_images, prompts, args.batch_size, args.clip_model, device
        )
        # calculate inception score last since it resizes the generated images to 299x299
        print("Calculating IS:")
        IS = calculate_inception_score(retrieved_images, args.batch_size, device)
        metrics = {
            "FID": round(fid.detach().cpu().numpy().tolist(), 2),
            "CLIPScore": round(clip_score.detach().cpu().numpy().tolist(), 2),
            "IS": round(IS[0].detach().cpu().numpy().tolist(), 2),
            "IS_Standard_Deviation": round(IS[1].detach().cpu().numpy().tolist(), 2),
        }
        print(metrics.__repr__())
        with open(f"{args.output_file}_retrieved_k1", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
