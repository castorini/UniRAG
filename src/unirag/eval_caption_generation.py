import argparse
import json
import os
import re

import candidate_retrieval as cr
import jsonlines
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from tqdm import tqdm


def file_in_dir(dirname, extension):
    if os.path.isdir(dirname):
        files = os.listdir(dirname)
        result = [
            os.path.join(dirname, file) for file in files if file.endswith(extension)
        ]
    else:
        result = []
    return result


def convert_to_tokenizer_input_format(dictionary):
    new_dictionary = {}
    for id, captions in dictionary.items():
        captions_list = []
        for c in captions:
            captions_list.append({"image_id": id, "caption": c})
        new_dictionary[id] = captions_list
    return new_dictionary


def get_ground_truth(candidate_path, retrieval_jsonl_path, res):
    clu = cr.CandidateLookUp(candidate_path)
    gts = {}
    with jsonlines.open(retrieval_jsonl_path) as reader:
        for obj in tqdm(reader, desc="Reading docs"):
            img = os.path.basename(obj["query"]["query_img_path"])
            if img in res:
                pos_cand = obj["query"]["pos_cand_list"]
                candidates = []
                for cand in pos_cand:
                    candidates.append(clu.retrieve_candidate_txt_from_did(cand))
                gts[img] = candidates
            else:
                assert False, "retrieved queries and llm queies must match"

            if len(gts) == len(res):
                break
    print(f"ground truth count: {len(gts)}")
    return gts


def remove_caption_prefix(input_string):
    input_string = input_string.strip()
    pattern = r"^Caption \[\d+\]: "
    match = re.match(pattern, input_string)
    if match:
        return input_string[match.end() :]
    return input_string


def get_results(result_dir):
    res_files = file_in_dir(result_dir, ".json")
    print(f"Count of json file: {len(res_files)}")

    res = {}
    for file in res_files:
        with open(file, "r") as file:
            data = json.load(file)
        for ind in data:
            res[os.path.basename(ind["image"])] = [
                remove_caption_prefix(ind["response"])
            ]
    return res


def calculate_metrics(output_path, res, gts):
    # Tokenize before eval
    gts = convert_to_tokenizer_input_format(gts)
    res = convert_to_tokenizer_input_format(res)
    tokenizer = PTBTokenizer()
    _gts = tokenizer.tokenize(gts)
    _res = tokenizer.tokenize(res)

    scorers = {
        "Bleu": Bleu(),
        "Cider": Cider(),
        "Rouge": Rouge(),
        "Spice": Spice(),
    }
    result = {}
    for scn in scorers:
        sc = scorers[scn]
        score, _ = sc.compute_score(_gts, _res)
        print("-" * 79)
        print(f"{scn}: {score}")
        print("-" * 79)
        result[scn] = score

    with open(output_path, "w") as outfile:
        json.dump(result, outfile)
    print(f"Output file at: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate_path",
        default=False,
        required=True,
        help="Path to jsonl file containing the candidates",
    )
    parser.add_argument("--result_dir", required=True, help="Result directory")
    parser.add_argument(
        "--retrieval_jsonl_path",
        required=True,
        help="Path to the retrieved jsonl queries that also contain positive candidates list for ground truth",
    )
    parser.add_argument(
        "--calculate_retriever_metrics",
        default=False,
        action="store_true",
        help="When true, the metrics for the retrieved results are also calcualted.",
    )
    args = parser.parse_args()

    res = get_results(args.result_dir)
    gts = get_ground_truth(args.candidate_path, args.retrieval_jsonl_path, res)

    output_dir = os.path.join(args.result_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"metrics.json")
    calculate_metrics(output_path, res, gts)

    # calculate retriever metrics as the baseline if specified
    if args.calculate_retriever_metrics:
        res = {}
        with jsonlines.open(args.retrieval_jsonl_path) as reader:
            for obj in tqdm(reader, desc="Reading docs"):
                img = os.path.basename(obj["query"]["query_img_path"])
                # TODO:the evaluator expects one predicted caption only,
                # for now it takes the first retrieved caption
                # consider evalaution of each retrieved index indivudually and averaging them out.
                if img in gts:
                    caption = obj["candidates"][0]["txt"]
                    if caption:
                        candidates = [caption]
                    elif (
                        obj["complement_candidates"][0]
                        and obj["complement_candidates"][0]["txt"]
                    ):
                        candidates = [obj["complement_candidates"][0]["txt"]]
                    else:
                        # None captions are not acceptable, replace them with blank
                        print(
                            f'Added empty retriever canption for {obj["query"]["qid"]}'
                        )
                        candidates = [""]

                    res[img] = candidates
        retriever_output_file = (
            "clip_retriever_metrics_k1.json"
            if "CLIP" in args.retrieval_jsonl_path
            else "blip_retriever_metrics_k1.json"
        )
        output_path = os.path.join(output_dir, retriever_output_file)
        calculate_metrics(output_path, res, gts)


if __name__ == "__main__":
    main()
