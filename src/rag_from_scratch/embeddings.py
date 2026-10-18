"""Day 04's local embedding model, packaged for reuse by later days.

The code is the one taught in notebooks/day_04_embeddings_and_similarity.ipynb (sections 2, 3
and 5): download two files of BAAI/bge-small-en-v1.5 from a pinned commit (only when a file is
missing), check each file's SHA-256 before loading it, tokenize with truncation at 512 tokens and
padding, run the ONNX model on the CPU, keep the [CLS] vector and scale it to length 1. Queries get
the model card's instruction in front of them; passages don't.

One change in shape: the notebook kept the tokenizer and the ONNX session in global variables;
here they live in an Embedder object that is created once:

    embedder = Embedder(repo_root)              # downloads (first run only) and verifies the files
    vectors = embedder.embed(texts)             # (len(texts), 384) float32, every row has length 1
    query = embedder.embed_query(question)      # (384,) float32, with the query instruction

Model files go to models/bge-small-en-v1.5/<revision[:12]>/ (git-ignored). hf_hub_download is
called only when a file is missing, so a run with the files present makes no network requests.

tests/test_shared_code.py runs the Day 04 notebook's own code and checks that this module returns
identical vectors, file checks and error messages.
"""

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL = {
    "repo": "BAAI/bge-small-en-v1.5",                       # https://huggingface.co/BAAI/bge-small-en-v1.5
    "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",  # a commit, not "main": the files can't change
    "licence": "MIT",
    "files": {                                              # path in the repository: (SHA-256, bytes)
        "onnx/model.onnx": ("828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35", 133_093_490),
        "tokenizer.json": ("d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66", 711_396),
    },
}
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "   # model card: queries only
MAX_TOKENS = 512    # the model's input limit; longer texts are cut off


def default_model_dir(repo_root: Path) -> Path:
    return repo_root / "models" / "bge-small-en-v1.5" / MODEL["revision"][:12]   # models/ is git-ignored


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def model_file(name: str, model_dir: Path, repo_root: Path | None = None) -> Path:
    # Download one file of the pinned revision if it isn't here yet, then check its SHA-256 every time.
    expected_sha256, expected_size = MODEL["files"][name]
    local = model_dir / name
    shown = (local.relative_to(repo_root) if repo_root and local.is_relative_to(repo_root) else local).as_posix()
    if not local.is_file():   # only the first run touches the network
        try:
            hf_hub_download(MODEL["repo"], name, revision=MODEL["revision"], local_dir=model_dir)
        except Exception as error:   # no network, a proxy, a blocked host...: say how to fix it
            raise RuntimeError(f"Could not download {name} ({expected_size / 1e6:.1f} MB) from {MODEL['repo']}: "
                               f"{error}\nBehind a proxy? Set HTTPS_PROXY. Hugging Face blocked? Set HF_ENDPOINT to a "
                               f"mirror, or download the file yourself and save it as {shown}") from error
    actual = file_sha256(local)
    if actual != expected_sha256:
        raise ValueError(f"{shown} has SHA-256 {actual}, expected {expected_sha256}. "
                         "Delete it and run again. Never load a model file you can't verify.")
    print(f"{name:<16} {local.stat().st_size:>11,} bytes   SHA-256 {actual[:12]}... verified")
    return local


def model_inputs(encodings) -> dict[str, np.ndarray]:
    # Tokenizer output -> the three integer arrays the model expects, one row per text.
    return {
        "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
    }


class Embedder:
    """The pinned model, loaded once: text in, unit-length 384-number vectors out."""

    def __init__(self, repo_root: Path, model_dir: Path | None = None):
        self.model = MODEL
        self.revision = MODEL["revision"]
        self.query_instruction = QUERY_INSTRUCTION
        self.model_dir = model_dir if model_dir is not None else default_model_dir(repo_root)
        tokenizer_path = model_file("tokenizer.json", self.model_dir, repo_root)
        model_path = model_file("onnx/model.onnx", self.model_dir, repo_root)

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding()      # in a batch, pad shorter texts with [PAD] up to the longest one

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.dimensions = int(self.session.get_outputs()[0].shape[-1])

    def run_model(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        # Token IDs in, one 384-number vector per token out: shape (texts, tokens, 384).
        return self.session.run(None, {name: inputs[name] for name in self.input_names})[0]

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        # text -> tokens -> model -> [CLS] vector -> length 1. One row per text, 384 columns.
        rows = []
        for start in range(0, len(texts), batch_size):
            token_vectors = self.run_model(model_inputs(self.tokenizer.encode_batch(texts[start:start + batch_size])))
            cls = token_vectors[:, 0, :]
            rows.append(cls / np.linalg.norm(cls, axis=1, keepdims=True))
        return np.vstack(rows).astype(np.float32)

    def embed_query(self, question: str, instruction: str = QUERY_INSTRUCTION) -> np.ndarray:
        # The model card asks for this instruction in front of short search queries (not in front of passages).
        return self.embed([instruction + question])[0]
