"""Day 09's cross-encoder reranker, packaged for reuse by later days.

The code is the one taught in notebooks/day_09_reranking_and_context_selection.ipynb (sections 0, 2, 3
and 10): download two files of an ONNX export of cross-encoder/ms-marco-MiniLM-L6-v2 from a pinned
commit (only when a file is missing), check each file's SHA-256 before loading it, score each
(question, chunk) pair in ONE input with truncation at 512 tokens and padding, rerank a hybrid's top N
by those scores, and the near-tie margin of a reranked ranking.

Two changes in shape: the notebook kept the tokenizer and the ONNX session in global variables; here
they live in a Reranker object that is created once. And the notebook's rerank(q, ranked, n) and
rerank_margin(name, q) read global caches and rankings; here they take what they need:

    reranker = Reranker(repo_root)                       # downloads (first run only) and verifies the files
    scores = reranker.scores(question, texts)            # one logit per (question, text) pair
    top = [cid for cid, _ in ranked[:n]]
    reranked = rerank(ranked, dict(zip(top, reranker.scores(question, [text[c] for c in top]))), n)
    margin = rerank_margin(ranked, reranked, q.answer_chunks, n)

Model files go to models/ms-marco-MiniLM-L-6-v2/<revision[:12]>/ (git-ignored). The model is Apache-2.0;
it was trained on MS MARCO, whose terms say non-commercial research (Day 09's LEARN.md, section 4).
A reranker reads whatever it is given: every input ranking must already be permission-filtered.

tests/test_shared_code.py runs the Day 09 notebook's own code and checks that this module returns
identical scores, rankings and margins.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from .embeddings import file_sha256
from .evaluation import first_relevant

RERANKER = {
    "repo": "Xenova/ms-marco-MiniLM-L-6-v2",                   # https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2
    "revision": "a09144355adeed5f58c8ed011d209bf8ee5a1fec",  # a commit, not "main": the files can't change
    "converted_from": "cross-encoder/ms-marco-MiniLM-L6-v2",  # https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
    "licence": "Apache-2.0 (the original model's; the ONNX repository states none of its own)",
    "files": {                                                 # path in the repository: (SHA-256, bytes)
        "onnx/model.onnx": ("c623d0bcb99f4622beb413eaef00cfbe5db20df9f1dd982da4b4f26022881870", 90_992_115),
        "tokenizer.json": ("d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66", 711_396),
    },
}
MAX_TOKENS = 512         # the model's input limit: question and chunk together


def default_model_dir(repo_root: Path) -> Path:
    return repo_root / "models" / "ms-marco-MiniLM-L-6-v2" / RERANKER["revision"][:12]   # models/ is git-ignored


def reranker_file(name: str, model_dir: Path, repo_root: Path | None = None) -> Path:
    # Download one file of the pinned revision if it isn't here yet, then check its SHA-256 every time (as on Day 04).
    expected_sha256, expected_size = RERANKER["files"][name]
    local = model_dir / name
    shown = (local.relative_to(repo_root) if repo_root and local.is_relative_to(repo_root) else local).as_posix()
    if not local.is_file():   # only the first run touches the network
        try:
            hf_hub_download(RERANKER["repo"], name, revision=RERANKER["revision"], local_dir=model_dir)
        except Exception as error:   # no network, a proxy, a blocked host...: say how to fix it
            raise RuntimeError(f"Could not download {name} ({expected_size / 1e6:.1f} MB) from {RERANKER['repo']}: "
                               f"{error}\nBehind a proxy? Set HTTPS_PROXY. Hugging Face blocked? Set HF_ENDPOINT to a "
                               f"mirror, or download the file yourself and save it as {shown}") from error
    actual = file_sha256(local)
    if actual != expected_sha256:
        raise ValueError(f"{shown} has SHA-256 {actual}, expected {expected_sha256}. "
                         "Delete it and run again. Never load a model file you can't verify.")
    print(f"{name:<16} {local.stat().st_size:>11,} bytes   SHA-256 {actual[:12]}... verified")
    return local


class Reranker:
    """The pinned cross-encoder, loaded once: (question, text) pairs in, one relevance logit per pair out."""

    def __init__(self, repo_root: Path, model_dir: Path | None = None):
        self.model = RERANKER
        self.revision = RERANKER["revision"]
        self.model_dir = model_dir if model_dir is not None else default_model_dir(repo_root)
        self.tokenizer = Tokenizer.from_file(str(reranker_file("tokenizer.json", self.model_dir, repo_root)))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)   # question + chunk together, cut at the limit
        self.tokenizer.enable_padding()                            # in a batch, pad shorter pairs with [PAD]
        self.session = ort.InferenceSession(str(reranker_file("onnx/model.onnx", self.model_dir, repo_root)),
                                            providers=["CPUExecutionProvider"])

    def scores(self, question: str, texts: list[str], batch_size: int = 16) -> list[float]:
        # One number per (question, text) pair. The model reads the question and the chunk TOGETHER, in one input, and
        # outputs a single logit: higher means more relevant. It isn't a probability and has no fixed range.
        scores = []
        for start in range(0, len(texts), batch_size):
            encodings = self.tokenizer.encode_batch([(question, text) for text in texts[start:start + batch_size]])
            inputs = {"input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                      "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
                      "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64)}
            scores.extend(float(s) for s in self.session.run(None, inputs)[0][:, 0])
        return scores


def rerank(ranked, scores, n: int) -> list[tuple[str, float]]:
    # The ranking's top n, put in the cross-encoder's order (ties by chunk id), then the rest of the ranking in its own
    # order. `scores` maps chunk id -> reranker score and must hold every one of the top n (the notebook's pair cache).
    top = [cid for cid, _ in ranked[:n]]
    return sorted(((cid, scores[cid]) for cid in top), key=lambda kv: (-kv[1], kv[0])) + ranked[n:]


def rerank_margin(fused, reranked, relevant, n: int) -> float | None:
    # How close the first labelled chunk is to a tie that would move it: its reranker-score gap to a neighbour among
    # the reranked candidates (or its fused-score gap below them), and, if it sits right at the candidate cut-off,
    # the hybrid's gap there (a swap at #n / #n+1 decides whether the reranker sees it at all).
    rank = first_relevant(reranked, relevant)
    if rank is None:
        return None
    if rank <= n:
        gaps = [reranked[i - 1][1] - reranked[i][1] for i in (rank - 1, rank) if 0 < i < n]
    else:
        gaps = [fused[i - 1][1] - fused[i][1] for i in (rank - 1, rank) if n < i < len(fused)]
    if first_relevant(fused, relevant) in (n, n + 1) and len(fused) > n:
        gaps.append(fused[n - 1][1] - fused[n][1])
    return round(min(gaps), 4) if gaps else None
