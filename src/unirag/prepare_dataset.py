import argparse
import json
import os
import shutil
import numpy
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampled-qids-file",
        type=str,
        default="",
        help="A json file containing a list of sampled qids used for filtering the queries, if not specified a new sampling will happen",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=False,
        help="Samples the caption queries for image generation, true for MSCOCO runs and false for fashion200k",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The directory path used for storing the retrieved jsonl file for selected queries, a list of sampled qids when sampling is true, and the images dir containing the gt images",
    )
    parser.add_argument(
        "--dataset-queries",
        type=str,
        required=True,
        help="The jsonl file containing the dataset queries, each entry has a prompt and a ground truth image path",
    )
    parser.add_argument(
        "--candidates-file",
        type=str,
        required=True,
        help="The jsonl file containing the candidates, each image candidate has a did and a path relative to the candidates-base-dir",
    )
    parser.add_argument(
        "--candidates-base-dir",
        type=str,
        required=True,
        help="The base path to the directory containing ground truth images",
    )
    return parser.parse_args()


def load_candidates(candidates_file):
    print("----loading candidates-------")
    candidates = {}
    with open(candidates_file, "r") as file:
        for line in tqdm(file):
            cand = json.loads(line)
            if cand.get("img_path", None):
                assert cand["did"] not in candidates, "candidate dids must be unique"
                candidates[cand["did"]] = cand["img_path"]
    return candidates


def copy_gt_candidates(qid, cand_list, image_dir, candidates_dict, candidates_base_dir):
    print("-----copying gt candidates-----")
    for cand_did in tqdm(cand_list):
        src = os.path.join(candidates_base_dir, candidates_dict[cand_did])
        dst = os.path.join(image_dir, f"{qid}_{os.path.basename(src)}")
        shutil.copy2(src, dst)


def sample_queries(dataset_queries):
    print("-----sampling queries----")
    cand_did_to_queries = {}
    with open(dataset_queries, "r") as queries:
        for line in tqdm(queries):
            obj = json.loads(line)
            query = obj["query"]
            qid = query["qid"]
            cand_did_list = query["pos_cand_list"]
            for cand_did in cand_did_list:
                if cand_did in cand_did_to_queries:
                    cand_did_to_queries[cand_did].append(qid)
                else:
                    cand_did_to_queries[cand_did] = [qid]
    print(len(cand_did_to_queries))
    sampled_qids = set()
    # Sample from queries for each cand_did:
    for _, qid_list in cand_did_to_queries.items():
        idx = numpy.random.randint(low=0, high=len(qid_list))
        while qid_list[idx] in sampled_qids:
            idx = numpy.random.randint(low=0, high=len(qid_list))
        sampled_qids.add(qid_list[idx])
    return sampled_qids


def main(args):
    candidates_dict = load_candidates(args.candidates_file)
    no_sample = False
    if args.sampled_qids_file:
        with open(args.sampled_qids_file, "r") as f:
            sampled_qids = set(json.load(f))
    elif args.do_sample:
        sampled_qids = sample_queries(args.dataset_queries)
    else:
        no_sample = True
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    image_dir = os.path.join(args.output_dir, "images")
    retrieved = []
    with open(args.dataset_queries, "r") as queries:
        for line in tqdm(queries):
            obj = json.loads(line)
            query = obj["query"]
            if query.get("query_txt", None) and (
                no_sample or query["qid"] in sampled_qids
            ):
                retrieved.append(obj)
                copy_gt_candidates(
                    query["qid"],
                    query["pos_cand_list"],
                    image_dir,
                    candidates_dict,
                    args.candidates_base_dir,
                )

    # write selected queries in a jsonl file
    with open(os.path.join(args.output_dir, f"retrieved.jsonl"), "w") as file:
        for r in retrieved:
            json.dump(r, file)
            file.write("\n")
    if not no_sample:
        with open(os.path.join(args.output_dir, f"sampled_qids.json"), "w") as file:
            json.dump(list(sampled_qids), file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
