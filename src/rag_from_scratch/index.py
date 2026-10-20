"""Day 05's exact vector index, packaged for reuse by later days.

The code is the one taught in notebooks/day_05_vector_search_and_lifecycle.ipynb (sections 2 to
6): a flat (exact) index that keeps one vector and one record per chunk, keyed by chunk_id, plus
the model that made the vectors. Copied without changes in behaviour:

    index = VectorIndex(model, revision, dimensions)
    index.upsert(chunks, vectors)                       # insert or replace by chunk_id
    index.search(query_vector, k=3, where={...})        # exact cosine top k, filter before ("pre") or
                                                        # after ("post") the ranking
    save_index(index, path); load_index(path, model, revision)   # one .npz file, atomic swap, no pickle,
                                                                 # refuses vectors from another model
    sync(index, current_chunks, embed)                  # add, re-embed, update metadata, delete

Filters (`where`) are small dictionaries checked by matches(): a value (equal to, or contained in a
list field such as allowed_groups), {"any": [...]} or {"gte": value}. groups_filter(groups) builds
the permission filter. A filter is a tool for access control, not access control by itself: its
input must come from verified identity and it must not be skippable (Day 05, section 6).

tests/test_shared_code.py runs the Day 05 notebook's own code and checks that this module returns
identical search results, filters, files, sync plans and error messages.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk

TOP_K = 3   # chunks returned per question, as on Days 01, 03, 04 and 05


@dataclass(frozen=True)
class Result:
    rank: int
    score: float     # cosine similarity (the vectors have length 1)
    id: str          # chunk_id
    chunk: Chunk     # the stored record: text, permissions, provenance


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    # Positions of the k highest scores, best first. argpartition finds them without sorting everything.
    best = np.argpartition(-scores, k)[:k] if k < len(scores) else np.arange(len(scores))
    return best[np.argsort(-scores[best], kind="stable")]


class VectorIndex:
    """A flat (exact) vector index: one vector and one record per chunk, plus the model that made the vectors."""

    def __init__(self, model: str, revision: str, dimensions: int):
        self.model, self.revision, self.dimensions = model, revision, dimensions
        self.vectors = np.zeros((0, dimensions), dtype=np.float32)   # row i is the vector of records[i]
        self.records: list[Chunk] = []
        self.row: dict[str, int] = {}                                # chunk_id -> row: the primary key

    def __len__(self) -> int:
        return len(self.records)

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        # Insert chunks with a new chunk_id; replace the record and the vector of an existing one.
        if len(chunks) != len(vectors) or vectors.shape[1] != self.dimensions:
            raise ValueError(f"{len(chunks)} chunks, vectors of shape {vectors.shape}")
        new_vectors = []
        for chunk, vector in zip(chunks, vectors):
            if chunk.chunk_id in self.row:
                i = self.row[chunk.chunk_id]
                self.records[i], self.vectors[i] = chunk, vector
            else:
                self.row[chunk.chunk_id] = len(self.records)
                self.records.append(chunk)
                new_vectors.append(vector)
        if new_vectors:
            self.vectors = np.vstack([self.vectors, np.asarray(new_vectors, dtype=np.float32)])

    def update_metadata(self, chunk: Chunk) -> None:
        # Same text, new permissions or provenance: replace the record, keep the vector.
        i = self.row[chunk.chunk_id]
        if self.records[i].chunk_hash != chunk.chunk_hash:
            raise ValueError(f"{chunk.chunk_id}: the text changed, so the vector must be recomputed")
        self.records[i] = chunk

    def delete(self, chunk_ids) -> int:
        # Remove rows and close the gaps. (Large stores mark rows as deleted and compact later.)
        gone = {c for c in chunk_ids if c in self.row}
        keep = [i for i, r in enumerate(self.records) if r.chunk_id not in gone]
        self.vectors = self.vectors[keep]
        self.records = [self.records[i] for i in keep]
        self.row = {r.chunk_id: i for i, r in enumerate(self.records)}
        return len(gone)

    def search(self, query_vector: np.ndarray, k: int = TOP_K, where: dict | None = None,
               filter_mode: str = "pre") -> list[Result]:
        # Exact search: score every candidate row, keep the best k. A filter runs before
        # the ranking ("pre") or on the ranked top k ("post").
        if filter_mode == "pre":
            rows = np.array([i for i, r in enumerate(self.records) if where is None or matches(r, where)],
                            dtype=np.int64)
            scores = self.vectors[rows] @ query_vector
            picked = [(rows[j], scores[j]) for j in top_k(scores, k)]
        elif filter_mode == "post":
            scores = self.vectors @ query_vector
            picked = [(i, scores[i]) for i in top_k(scores, k) if where is None or matches(self.records[i], where)]
        else:
            raise ValueError(f"filter_mode must be 'pre' or 'post', not {filter_mode!r}")
        return [Result(rank, float(score), self.records[i].chunk_id, self.records[i])
                for rank, (i, score) in enumerate(picked, start=1)]


def matches(chunk: Chunk, where: dict) -> bool:
    # where = {field: condition}; every condition must hold. A condition is
    #   a value         -> equal to it (for a list field such as allowed_groups: contained in it)
    #   {"any": [...]}  -> shares at least one value with the list
    #   {"gte": value}  -> at least this value; a missing value (None) fails
    for field, condition in where.items():
        value = getattr(chunk, field)              # a misspelled field raises instead of matching nothing
        values = set(value) if isinstance(value, tuple) else {value}
        if isinstance(condition, dict):
            if set(condition) - {"any", "gte"}:
                raise ValueError(f"unknown operator in {condition}")
            if "any" in condition and not values & set(condition["any"]):
                return False
            if "gte" in condition and (value is None or value < condition["gte"]):
                return False
        elif condition not in values:
            return False
    return True


def groups_filter(groups) -> dict:
    return {"allowed_groups": {"any": sorted(groups)}}


# ----------------------------------------------------------------------------- persistence (Day 05, section 4)
INDEX_FORMAT = 1   # change it when the file layout changes, so an old file is rejected instead of misread


def write_index(f, index: VectorIndex) -> None:
    # One .npz file (a zip of NumPy arrays): the vectors, plus the manifest and the records as UTF-8 JSON bytes.
    manifest = {"format": INDEX_FORMAT, "model": index.model, "revision": index.revision,
                "dimensions": index.dimensions, "normalised": True,
                "similarity": "cosine (dot product of unit vectors)", "count": len(index)}
    records = [asdict(r) for r in index.records]
    as_bytes = lambda obj: np.frombuffer(json.dumps(obj, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    np.savez(f, vectors=index.vectors, manifest=as_bytes(manifest), records=as_bytes(records))


def save_index(index: VectorIndex, path: Path) -> None:
    # Write a complete new file next to the old one, then swap it in with one os.replace call:
    # anyone loading the index gets the old version or the new one, never a mix or half a file.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")      # same folder, so the rename stays on one file system
    try:
        with tmp.open("wb") as f:
            write_index(f, index)
            f.flush()
            os.fsync(f.fileno())                   # on the disk before the swap, not just in a buffer
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)                # still there only if something failed


def load_index(path: Path, model: str, revision: str) -> VectorIndex:
    with np.load(path, allow_pickle=False) as data:      # allow_pickle=False: a data file must never run code
        manifest = json.loads(data["manifest"].tobytes().decode("utf-8"))
        records = json.loads(data["records"].tobytes().decode("utf-8"))
        vectors = data["vectors"]
    if manifest["format"] != INDEX_FORMAT:
        raise ValueError(f"{path.name}: index format {manifest['format']}, this code reads format {INDEX_FORMAT}")
    if (manifest["model"], manifest["revision"]) != (model, revision):
        raise ValueError(f"{path.name} was built with {manifest['model']} @ {manifest['revision'][:12]}, but the "
                         f"queries come from {model} @ {revision[:12]}. Vectors from different models can't be "
                         "compared: rebuild the index.")
    if vectors.shape != (manifest["count"], manifest["dimensions"]) or len(records) != manifest["count"]:
        raise ValueError(f"{path.name}: {len(records)} records and vectors {vectors.shape}, but the manifest "
                         f"says {manifest['count']}")
    index = VectorIndex(model, revision, manifest["dimensions"])
    index.upsert([Chunk(**{**r, "allowed_groups": tuple(r["allowed_groups"])}) for r in records], vectors)
    if len(index) != manifest["count"]:
        raise ValueError(f"{path.name}: duplicate chunk IDs")
    return index


# ----------------------------------------------------------------------------- sync (Day 05, section 5)
def plan_sync(index: VectorIndex, current: list[Chunk]) -> dict[str, list[str]]:
    # Compare the chunks the documents produce now with what the index holds, chunk_id by chunk_id.
    by_id = {c.chunk_id: c for c in current}
    if len(by_id) != len(current):
        raise ValueError("duplicate chunk IDs in the new chunks")
    plan = {"add": [], "re-embed": [], "update metadata": [], "delete": [], "unchanged": []}
    for chunk_id, chunk in by_id.items():
        old = index.records[index.row[chunk_id]] if chunk_id in index.row else None
        if old is None:
            plan["add"].append(chunk_id)               # a new section or a new document
        elif old.chunk_hash != chunk.chunk_hash:
            plan["re-embed"].append(chunk_id)          # the text changed, so the stored vector is wrong
        elif old != chunk:
            plan["update metadata"].append(chunk_id)   # same text: new permissions, title, dates or doc version
        else:
            plan["unchanged"].append(chunk_id)
    plan["delete"] = sorted(set(index.row) - set(by_id))   # in the index, but in no document any more
    return plan


def sync(index: VectorIndex, current: list[Chunk], embed, max_delete_fraction: float = 0.5) -> dict[str, list[str]]:
    # Make the index match `current`, the COMPLETE list of chunks the documents produce now.
    plan = plan_sync(index, current)
    if len(index) and len(plan["delete"]) > max_delete_fraction * len(index):
        raise RuntimeError(f"refusing to delete {len(plan['delete'])} of {len(index)} chunks. Is the new chunk list "
                           "incomplete (a failed or partial ingest)? Raise max_delete_fraction if this is intended.")
    by_id = {c.chunk_id: c for c in current}
    to_embed = [by_id[i] for i in plan["add"] + plan["re-embed"]]
    if to_embed:                                       # the only step that runs the model
        index.upsert(to_embed, embed([c.text for c in to_embed]))
    for chunk_id in plan["update metadata"]:
        index.update_metadata(by_id[chunk_id])
    index.delete(plan["delete"])
    return plan
