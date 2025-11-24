import os
import torch

def _to_cpu_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.as_tensor(x)


def save_trec_run(score_matrix, query_ids, doc_ids, out_path, run_name="run"):
    """
    Generic TREC run writer.

    score_matrix: shape (num_queries, num_docs)
    query_ids: len = num_queries
    doc_ids:   len = num_docs

    Writes lines: qid 0 docid rank score run_name
    """
    score_matrix = _to_cpu_tensor(score_matrix)
    num_q, num_d = score_matrix.shape

    assert num_q == len(query_ids), "Rows must equal number of query_ids"
    assert num_d == len(doc_ids),   "Cols must equal number of doc_ids"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        for q_idx, qid in enumerate(query_ids):
            scores = score_matrix[q_idx]
            sorted_idx = torch.argsort(scores, descending=True)
            for rank, doc_idx in enumerate(sorted_idx.tolist(), start=1):
                docid = str(doc_ids[doc_idx])
                score = float(scores[doc_idx])
                f.write(f"{qid} 0 {docid} {rank} {score:.6f} {run_name}\n")

    print(f"[TREC] Saved run {run_name} to {out_path}")


def save_qrels_t2v(ids, ids_txt, out_path):
    """
    T2V qrels in TREC format:
    Each text query (ids_txt[i]) has exactly one relevant doc: the video with the same id.
    Line format: qid 0 docid rel
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for qid in ids_txt:
            qid_str = str(qid)
            # relevant doc is the same video id
            f.write(f"{qid_str} 0 {qid_str} 1\n")
    print(f"[QRELS] Saved T2V qrels to {out_path}")


def save_qrels_v2t(ids, ids_txt, out_path):
    """
    V2T qrels in TREC format:
    Each video query (ids[i]) has exactly one relevant text/doc: same id.
    Line format: qid 0 docid rel
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for qid in ids:
            qid_str = str(qid)
            f.write(f"{qid_str} 0 {qid_str} 1\n")
    print(f"[QRELS] Saved V2T qrels to {out_path}")
