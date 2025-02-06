import argparse
import json


class CandidateLookUp:
    def __init__(self, candidate_file: str):
        self._candidates = {}
        with open(candidate_file, "r") as f:
            for line in f:
                candidate = json.loads(line)
                self._candidates[candidate["did"]] = candidate

    def retrieve_candidate_txt_from_did(self, did: str):
        return self._candidates[did]["txt"]

    def retrieve_candidate_img_path_from_did(self, did: str):
        return self._candidates[did]["img_path"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate_path",
        default=False,
        help="Path to jsonl file containing the candidates",
    )
    args = parser.parse_args()
    clu = CandidateLookUp(args.candidate_path)
    print(clu.retrieve_candidate_txt_from_did("1:263527"))
    print(clu.retrieve_candidate_img_path_from_did("1:263527"))


if __name__ == "__main__":
    main()
