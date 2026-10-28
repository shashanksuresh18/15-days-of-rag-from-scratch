"""Day 08's rank fusion, packaged for reuse by later days.

The code is the one taught in notebooks/day_08_hybrid_search_and_fusion.ipynb (sections 3, 5 and 10):
Reciprocal Rank Fusion, min-max normalisation, the weighted score fusion that Day 08 chose as the
hybrid later days build on, and RRF's near-tie margin.

    fused = rrf([bm25_ranking, dense_ranking])                      # k = 60, the original paper's value
    fused = weighted(bm25_ranking, dense_ranking, ALPHA)            # 0.7 x dense + 0.3 x BM25, both rescaled
    margin = rrf_margin([bm25_ranking, dense_ranking], q.answer_chunks)

A ranking is a list of (chunk_id, score) pairs, best first, and every input ranking must already be
permission-filtered: fusion merges lists, it doesn't check who may read them (Day 08, section 8).

One change in shape: the notebook's rrf_margin(q) read a global `rankings` dictionary; here it takes the
two input rankings and the labelled chunks. ALPHA is Day 08's measured weight, chosen by a rule on the dev
split (not pre-registered: Day 08's LEARN.md, section 7).

tests/test_shared_code.py runs the Day 08 notebook's own code and checks that this module returns
identical rankings, scores and margins.

Standard library only.
"""

from .evaluation import first_relevant

RRF_K = 60               # RRF's constant: the original paper's value (Cormack et al., 2009), not tuned
ALPHA = 0.7              # Day 08's weight on dense retrieval in the weighted fusion (rule on dev)


def rrf(lists, k: int = RRF_K) -> list[tuple[str, float]]:
    # Reciprocal Rank Fusion: every list adds 1 / (k + rank) to each chunk it returns. Only positions are used, so the
    # retrievers' scales never meet. Ties (a chunk at ranks (a, b) and another at (b, a)) are broken by chunk id.
    fused = {}
    for ranked in lists:
        for rank, (cid, _) in enumerate(ranked, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def min_max(ranked) -> dict[str, float]:
    # One retriever's scores for one query, rescaled to 0..1: its best chunk 1, its worst 0
    if not ranked:
        return {}
    hi, lo = max(s for _, s in ranked), min(s for _, s in ranked)
    return {cid: (s - lo) / (hi - lo) if hi > lo else 1.0 for cid, s in ranked}


def weighted(lexical, dense, alpha: float) -> list[tuple[str, float]]:
    # alpha * dense + (1 - alpha) * BM25, both rescaled to 0..1 first. A chunk BM25 didn't return gets 0 from BM25.
    lex, den = min_max(lexical), min_max(dense)
    fused = {cid: alpha * den.get(cid, 0.0) + (1 - alpha) * lex.get(cid, 0.0) for cid in {*lex, *den}}
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def rrf_margin(lists, relevant) -> float | None:
    # The smallest score gap in an INPUT ranking whose swap would move the labelled chunk's RRF rank
    # (the notebook's rrf_margin(q), with the input rankings and the labels as arguments)
    if not relevant:
        return None
    base, gaps = first_relevant(rrf(lists), relevant), []
    for i, inputs in enumerate(lists):
        for j in range(len(inputs) - 1):
            swapped = inputs[:j] + [inputs[j + 1], inputs[j]] + inputs[j + 2:]
            if first_relevant(rrf(lists[:i] + [swapped] + lists[i + 1:]), relevant) != base:
                gaps.append(inputs[j][1] - inputs[j + 1][1])
    return round(min(gaps), 4) if gaps else None
