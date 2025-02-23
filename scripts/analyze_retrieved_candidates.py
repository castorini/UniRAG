import json


def main(args):
    retrieved_cands_dict = {}
    with open(args.retrieved_cand_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            retrieved_cands_dict[obj["query"]["qid"]] = obj
    problem_qids = set()
    no_pair_qids = set()
    for qid, obj in retrieved_cands_dict.items():
        for i, complement_cand in enumerate(obj["complement_candidates"]):
            if not complement_cand and obj["candidates"][i]["modality"] != "image,text":
                no_pair_qids.add(qid)
                break
            if complement_cand["img_path"] == obj["query"]["query_img_path"]:
                problem_qids.add(qid)
                break
    print(
        f"The following {len(no_pair_qids)} qids have retrieved candidates that don't make image,text pairs."
    )
    for qid in no_pair_qids:
        print("\n" + retrieved_cands_dict[qid].__repr__() + "\n")

    print(
        f"The following {len(problem_qids)} qids have a complement candidate with the same image as the query image"
    )
    for qid in problem_qids:
        print("\n" + retrieved_cands_dict[qid].__repr__() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieved_cand_path",
        type=str,
        required=True,
        help="The path to a jsonl file containing retrieve results.",
    )
    args = parser.parse_args()
    main(args)
