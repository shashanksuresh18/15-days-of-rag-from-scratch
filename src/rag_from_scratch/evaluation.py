"""Day 07's evaluation harness, packaged for reuse by later days.

The code is the one taught in notebooks/day_07_evaluation_baseline.ipynb (sections 1, 1.1, 3, 5 and 6):
the labelled query and its loader, the whole-word evidence rule, the snapshot the labels belong to
and validate(), the metrics (hit@k, recall@k, reciprocal rank, MRR), the Wilson interval, the sign
test, outcome() (what a user experiences end to end; added on Day 10, which reuses it), the near-tie
margin and compare(), which lists what changed since an accepted baseline.

One change in shape: the notebook's measure(), tie_margin() and top_score() read a global
`rankings` dictionary by retriever name. Here they take the ranking itself, so a later day can
measure any retriever without defining globals:

    queries = load_queries(Path("evals/questions-v1.jsonl"))
    problems = validate(queries, chunks, manifest)          # [] when the labels match the documents
    m = measure(rankings["BM25"], selection)                 # notebook: measure("BM25", selection)
    margin = tie_margin(ranked, q.answer_chunks)             # notebook: tie_margin("BM25", q)
    changes, ties = compare(run, accepted)                   # what moved since the accepted baseline

A ranking is a list of (chunk_id, score) pairs, best first.

tests/test_shared_code.py runs the Day 07 notebook's own code and checks that this module returns
identical queries, problems, metrics, intervals, p-values, outcomes, margins and comparisons.

Standard library only.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .answering import DIRECTORY

TOP_K = 3                # what Day 06's pipeline reads: a query is an ERROR if no relevant chunk is in its top 3
MAX_K = 10               # how deep the metrics look (hit@10, MRR@10)
K_VALUES = (1, 3, 5, 10)
TIE = 0.001              # scores closer than this are a tie in practice (CPUs differ around the 7th decimal)


# ----------------------------------------------------------------------------- the labelled set (Day 07, section 1)
@dataclass(frozen=True)
class Query:
    id: str
    question: str
    user: str                            # who asks: the right answer depends on what this person may read
    kind: str                            # "question", or "lookup" (an exact value typed as the query, Day 04)
    split: str                           # "dev" (seen while building Days 01-06) or "test" (new on Day 07)
    origin: str                          # where it came from: day01-failure, day03-probe, day04-lookup, ...
    tags: tuple[str, ...]
    evidence: tuple[str, ...]            # strings that must ALL appear in a chunk for it to answer the question
    facts: tuple[str, ...]               # strings a correct answer contains (for answer checks, Day 12)
    answer_chunks: tuple[str, ...]       # THE LABEL: chunks this user may read that hold the evidence
    restricted_chunks: tuple[str, ...]   # chunks that hold the evidence but this user may NOT read
    note: str

    @property
    def answerable(self) -> bool:
        return bool(self.answer_chunks)


def load_queries(path: Path) -> list[Query]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [Query(**{k: tuple(v) if isinstance(v, list) else v for k, v in row.items()}) for row in rows]


# ----------------------------------------------------------------------------- labels belong to a snapshot (1.1)
def holds(evidence, text: str) -> bool:
    # Day 04's whole-match rule: every evidence string, never as part of a longer number ("USD 25" isn't in "USD 250")
    return bool(evidence) and all(re.search(rf"(?<!\w){re.escape(e)}(?!\w|[,.]\d)", text) for e in evidence)


def snapshot(chunks) -> dict:
    # What the labels depend on: every chunk's text (as its hash) and who may read it
    return {c.chunk_id: {"hash": c.chunk_hash, "groups": list(c.allowed_groups)} for c in chunks}


def fingerprint(snap: dict) -> str:
    return hashlib.sha256(json.dumps(snap, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate(queries, chunks, manifest, directory=DIRECTORY) -> list[str]:
    problems = []
    if len({q.id for q in queries}) != len(queries) or len({(q.user, q.question) for q in queries}) != len(queries):
        problems.append("duplicate ids, or the same question twice for the same user")
    if {user: sorted(groups) for user, groups in directory.items()} != manifest["directory"]:
        problems.append("the user directory changed since the labels were written")
    # 1. The snapshot: every chunk's text hash and permissions, against the ones recorded with the labels
    now, then = snapshot(chunks), manifest["snapshot"]["chunks"]
    for cid in sorted(set(now) | set(then)):
        if cid not in now:
            problems.append(f"chunk {cid}: gone")
        elif cid not in then:
            problems.append(f"chunk {cid}: new")
        elif now[cid] != then[cid]:
            problems.append(f"chunk {cid}: {'text' if now[cid]['hash'] != then[cid]['hash'] else 'permissions'} changed")
    # 2. Every label, derived again from its evidence on today's chunks
    for q in queries:
        if q.kind not in ("question", "lookup") or q.split not in ("dev", "test") or q.user not in directory:
            problems.append(f"{q.id}: unknown kind, split or user")
            continue
        holding = [c for c in chunks if holds(q.evidence, c.text)]
        if q.evidence and not holding:
            problems.append(f"{q.id}: no chunk holds its evidence any more")
        readable = tuple(sorted(c.chunk_id for c in holding if set(c.allowed_groups) & directory[q.user]))
        hidden = tuple(sorted(c.chunk_id for c in holding if not set(c.allowed_groups) & directory[q.user]))
        if readable != q.answer_chunks:
            problems.append(f"{q.id}: labelled {list(q.answer_chunks)}, the evidence is now in {list(readable)}")
        if hidden != q.restricted_chunks:
            problems.append(f"{q.id}: restricted {list(q.restricted_chunks)}, now {list(hidden)}")
        if bool(q.facts) != q.answerable:
            problems.append(f"{q.id}: facts must be given exactly when the question is answerable")
    return problems


# ----------------------------------------------------------------------------- metrics that need labels (section 3)
def first_relevant(ranked, relevant) -> int | None:
    # The rank of the first labelled chunk, or None if the retriever never returns one
    return next((rank for rank, (cid, _) in enumerate(ranked, start=1) if cid in relevant), None)


def hit(ranked, relevant, k: int) -> bool:          # hit rate@k is the average of this over queries
    rank = first_relevant(ranked, relevant)
    return rank is not None and rank <= k


def recall(ranked, relevant, k: int) -> float:      # recall@k: the share of the labelled chunks found in the top k
    return len(set(relevant) & {cid for cid, _ in ranked[:k]}) / len(relevant)


def reciprocal_rank(ranked, relevant, k: int = MAX_K) -> float:   # MRR@k is the average of this
    rank = first_relevant(ranked, relevant)
    return 1 / rank if rank is not None and rank <= k else 0.0


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    # A 95% interval for a rate measured on n queries (the Wilson score interval): how unsure a small sample is
    p, denominator = successes / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def measure(rankings, selection, k_values=K_VALUES) -> dict:
    # rankings: {query id: ranking} for ONE retriever (the notebook's rankings[name])
    ranked = [(rankings[q.id], q.answer_chunks) for q in selection]
    result = {"n": len(ranked)}
    result.update({f"hit@{k}": sum(hit(r, relevant, k) for r, relevant in ranked) for k in k_values})
    result["recall@3"] = sum(recall(r, relevant, 3) for r, relevant in ranked) / len(ranked)
    result["MRR@10"] = sum(reciprocal_rank(r, relevant) for r, relevant in ranked) / len(ranked)
    return result


def sign_test(only_a: int, only_b: int) -> float:
    # Two-sided exact test on the queries where the two retrievers disagree: if neither were better, each
    # disagreement would be a coin flip. A small p means "unlikely to be luck"; a large p means "can't tell".
    n = only_a + only_b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n)


# ----------------------------------------------------------------------------- end to end (section 5)
def outcome(a, q: Query) -> str:
    # What the user experiences. "refused": no answer shown (abstained, or blocked by the citation check)
    if a.status != "answered":
        return "refused"
    if not q.answerable:
        return "answered"            # an answer to a question the documents don't answer for this user
    return "correct" if all(f.lower() in a.text.lower() for f in q.facts) else "wrong"


# ----------------------------------------------------------------------------- the baseline (section 6)
def top_score(ranked) -> float | None:
    return round(ranked[0][1], 3) if ranked else None


def tie_margin(ranked, relevant) -> float | None:
    # How far the first labelled chunk's score is from its neighbours' in the ranking. A tiny margin is a near
    # tie: another CPU's rounding could swap the two, which is not a change in the retriever.
    rank = first_relevant(ranked, relevant)
    if rank is None:
        return None
    gaps = [ranked[i - 1][1] - ranked[i][1] for i in (rank - 1, rank) if 0 < i < len(ranked)]
    return round(min(gaps), 4) if gaps else None


def compare(new: dict, old: dict, tolerance: float = 0.005) -> tuple[list[str], list[str]]:
    # What changed since the accepted baseline: (changes, near ties). Ranks and counts must match exactly, scores
    # within a tolerance. A rank that moves where the baseline recorded a near tie is listed apart, not as a change.
    if new["dataset"] != old["dataset"]:
        return ["different labels or a different snapshot: these two runs can't be compared"], []
    changes = ["retriever settings changed"] if new["retrievers"] != old["retrievers"] else []
    ties = []
    for qid, per_retriever in new["queries"].items():
        for name, now in per_retriever.items():
            then = old["queries"].get(qid, {}).get(name)
            if then is None:
                changes.append(f"{qid} {name}: not in the baseline")
            elif now["first_relevant"] != then["first_relevant"]:
                moved = f"{qid} {name}: labelled chunk at rank {then['first_relevant']} -> {now['first_relevant']}"
                if then["margin"] is not None and then["margin"] < TIE:
                    ties.append(f"{moved} (a near tie: {then['margin']} from its neighbour)")
                else:
                    changes.append(moved)
            elif (now["top_score"] is None) != (then["top_score"] is None) or \
                    (now["top_score"] is not None and abs(now["top_score"] - then["top_score"]) > tolerance):
                changes.append(f"{qid} {name}: top score {then['top_score']} -> {now['top_score']}")
    moved_metrics = [f"{key}: {old['metrics'].get(key)} -> {m}" for key, m in new["metrics"].items()
                     if m != old["metrics"].get(key)]
    (ties if ties and not changes else changes).extend(moved_metrics)   # metrics follow from the ranks
    if new["thresholds"] != old["thresholds"]:
        changes.append("measured thresholds or their outcomes changed")
    return changes, ties
