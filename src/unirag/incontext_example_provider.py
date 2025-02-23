import os
import random

import jsonlines
from candidate_retrieval import CandidateLookUp
from tqdm import tqdm


def get_candidates(obj, base_mbeir_path, image_query, txt_query):
    candidates = []
    for cand, complement_cand in zip(
        obj.get("candidates"), obj.get("complement_candidates")
    ):
        if cand["modality"] == "text":
            txt = cand.get("txt")
            image_path = (
                os.path.join(base_mbeir_path, complement_cand.get("img_path"))
                if complement_cand and complement_cand.get("img_path")
                else None
            )
        elif cand["modality"] == "image":
            txt = complement_cand.get("txt") if complement_cand else None
            image_path = (
                os.path.join(base_mbeir_path, cand.get("img_path"))
                if cand.get("img_path")
                else None
            )
        else:
            txt = cand.get("txt")
            image_path = (
                os.path.join(base_mbeir_path, cand.get("img_path"))
                if cand.get("img_path")
                else None
            )
        if not txt or not image_path:
            print(f"Failed to find image text pairs for {obj}")
            continue
        # The image/txt of fewshot examples should not be the same as image/text of the query.
        if txt == txt_query or (
            image_query and image_path == os.path.join(base_mbeir_path, image_query)
        ):
            continue
        candidates.append((txt, image_path))
    return candidates


def get_positive_cands(base_mbeir_path, queries_dict, example_ids, candidate_lookup):
    candidates = []
    for id in example_ids:
        query = queries_dict[id]
        positive_cand_id = query["pos_cand_list"][0]
        if query["query_modality"] == "text":
            txt = query["query_txt"]
            image_path = os.path.join(
                base_mbeir_path,
                candidate_lookup.retrieve_candidate_img_path_from_did(positive_cand_id),
            )
        elif query["query_modality"] == "image":
            txt = candidate_lookup.retrieve_candidate_txt_from_did(positive_cand_id)
            image_path = os.path.join(base_mbeir_path, query["query_img_path"])
        else:
            txt = query["query_txt"]
            image_path = os.path.join(base_mbeir_path, query["query_img_path"])
        candidates.append((txt, image_path))
    return candidates


def filter_qids(queries_dict, index):
    qids = sorted(list(queries_dict.keys()))
    if index == "full":
        start = 0
        end = len(qids)
    else:
        temp = index.split("_")
        start = int(temp[0])
        end = int(temp[1])
    return qids[start:end]


def get_random_fewshot_examples(
    retrieved_results_path, base_mbeir_path, candidate_file, index
):
    candidate_lookup = CandidateLookUp(candidate_file)
    queries_dict = {}
    retrieval_dict = {}
    image_queries = []
    txt_queries = []
    with jsonlines.open(retrieved_results_path) as reader:
        for obj in tqdm(reader, desc="Reading docs"):
            queries_dict[obj["query"]["qid"]] = obj["query"]
    qids = sorted(list(queries_dict.keys()))
    filtered_qids = filter_qids(queries_dict, index)
    for qid, q in queries_dict.items():
        if qid not in filtered_qids:
            continue
        image_query = q["query_img_path"]
        txt_query = q["query_txt"]
        example_ids = random.sample([id for id in qids if id != qid], 10)
        candidates = get_positive_cands(
            base_mbeir_path, queries_dict, example_ids, candidate_lookup
        )
        if image_query:
            query = os.path.join(base_mbeir_path, image_query)
            image_queries.append(query)
        elif txt_query:
            query = txt_query
            txt_queries.append(query)
        retrieval_dict[query] = (qid, candidates)

    if len(image_queries):
        assert len(retrieval_dict) == len(
            image_queries
        ), "The number of entries in candidates dict should match the number of image queries"
        return image_queries, retrieval_dict
    else:
        assert len(retrieval_dict) == len(
            txt_queries
        ), "The number of entries in candidates dict should match the number of txt queries"
        return txt_queries, retrieval_dict


def get_rag_fewshot_examples(retrieved_results_path, base_mbeir_path, index):
    queries_dict = {}
    objects_dict = {}
    retrieval_dict = {}
    image_queries = []
    txt_queries = []
    with jsonlines.open(retrieved_results_path) as reader:
        for obj in tqdm(reader, desc="Reading docs"):
            queries_dict[obj["query"]["qid"]] = obj["query"]
            objects_dict[obj["query"]["qid"]] = obj
    filtered_qids = filter_qids(queries_dict, index)
    for qid, q in queries_dict.items():
        if qid not in filtered_qids:
            continue
        image_query = q["query_img_path"]
        txt_query = q["query_txt"]
        candidates = get_candidates(
            objects_dict[qid], base_mbeir_path, image_query, txt_query
        )
        if image_query:
            query = os.path.join(base_mbeir_path, image_query)
            image_queries.append(query)
        elif txt_query:
            query = txt_query
            txt_queries.append(query)
        retrieval_dict[query] = (qid, candidates)

    if len(image_queries):
        assert len(retrieval_dict) == len(
            image_queries
        ), "The number of entries in candidates dict should match the number of image queries"
        return image_queries, retrieval_dict
    else:
        assert len(retrieval_dict) == len(
            txt_queries
        ), "The number of entries in candidates dict should match the number of txt queries"
        return txt_queries, retrieval_dict
