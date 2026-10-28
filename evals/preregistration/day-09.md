# Day 09: rules and settings, committed before the test split is scored

This file was committed on the `day-09` branch **before any Day 09 run scored a test-split query**,
exploratory runs included. Every setting below was either fixed in advance or chosen by a rule that
reads only the **dev** split (54 queries). The exploration scripts loaded `questions-v1.jsonl` and
dropped every `test` row before building anything. The Day 09 notebook recomputes each rule on dev and
refuses to go on (an `assert`) if it doesn't give the values written here.

The commit's timestamp is local: the branch wasn't pushed when this was written, so it shows the order
only as far as you trust the author. To have a server witness the order, push the pre-registration (or
open a pull request with it) before running anything on the held-out set.

## What was already known when this was written

The v1 test split isn't clean. Days 07 and 08 printed test-split results while exploring, before their
rules were written down (Day 07 LEARN.md section 9.2; Day 08 LEARN.md section 7). Before writing this
file, the author had read:

- the 27 test questions, their labels and their notes;
- every published test-split result of BM25, dense retrieval, RRF and the weighted fusion (Day 08's
  baseline), and of Day 06's pipeline under Day 07's thresholds;
- which 5 unanswerable test questions still get a confident, cited, wrong answer (d07-u01, d07-u03,
  d07-u04, d07-u11, d07-u12). They were *found* on the test split on Day 07, so they are a known
  failure set chosen by looking, not a random sample.

What hadn't been computed: **any cross-encoder score for any test-split query.**

So the test split can still check today's reranker settings, because they were fixed without it. It can't
give clean held-out evidence for the whole pipeline: the fusion weight it reranks (α = 0.7) was chosen
after test results had been seen, and the questions themselves are familiar. A decision that needs clean
held-out evidence needs new test questions (`questions-v2.jsonl`, written and committed before any system
runs on them; `evals/README.md`, rules 3 and 4).

## Fixed settings (not tuned)

| Setting | Value |
|---|---|
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` at commit `a09144355adeed5f58c8ed011d209bf8ee5a1fec`: an ONNX export of `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0) |
| Files | `onnx/model.onnx` 90,992,115 bytes, SHA-256 `c623d0bcb99f4622beb413eaef00cfbe5db20df9f1dd982da4b4f26022881870`; `tokenizer.json` 711,396 bytes, SHA-256 `d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66` |
| Input | the question as typed, then the chunk text **with** its `Title > Section` header (the text both retrievers index), as one pair: `[CLS] question [SEP] chunk [SEP]`, cut at 512 tokens |
| Score | the model's raw output (a logit), higher = more relevant; ties broken by chunk id |
| Candidates | Day 08's two hybrids, unchanged: the weighted fusion (α = 0.7, min-max) and RRF (k = 60), over full rankings that were each permission-filtered before fusion |
| Reranked ranking | the hybrid's top N reranked, followed by the rest of the hybrid's ranking in its own order (so hit@5 and hit@10 stay defined) |
| Near ties | a rank change where the labelled chunk was within 0.001 of a neighbour (its reranker score, or its fused score at the candidate cut-off) is reported apart, as CPU rounding |

The header choice wasn't tuned: it is the indexed text. (On dev, the reranker without the header put
12 questions first instead of 15, so it would not have been chosen anyway.)

## Rule 1: how many candidates to rerank (N)

*For each hybrid separately: the N in {3, 5, 10, 20, 44} that puts a labelled chunk in the top 3 for the
most dev queries (the 16 answerable dev questions and the 30 lookups, 46 in all); ties go to the higher dev
MRR@10, then to the smaller N (fewer model calls).*

Computed on dev only:

| Hybrid | N = 3 | N = 5 | N = 10 | N = 20 | N = 44 | Rule's choice |
|---|---|---|---|---|---|---|
| weighted: dev hit@3 / MRR@10 | **46** / 0.9855 | 45 / 0.9837 | 45 / 0.9837 | 45 / 0.9826 | 45 / 0.9826 | **N = 3** |
| RRF: dev hit@3 / MRR@10 | 45 / 0.9674 | 44 / 0.9620 | 44 / 0.9609 | **45** / **0.9826** | 45 / 0.9826 | **N = 20** |

## Rule 2: a reranker gate (the "similar but wrong" test)

*The gate reads the best reranker score among the primary pipeline's candidates (the weighted fusion's
top 3, reranked) and refuses below it. Its value: the highest value, in steps of 0.1, that refuses no
answerable dev question.* This is Day 07's rule for the dense gate, in the reranker's units. It applies to
questions, not lookups.

Computed on dev only: the weakest answerable dev question is d01-01 ("How much time off do new dads get?")
at −3.5646, so **the gate is −3.6**. On dev it keeps 16 of 16 answerable questions and refuses 7 of 8
unanswerable ones (all but "What is Google's remote work policy?", +4.41).

## Context selection

- **Deduplication:** drop a chunk whose text is the same as an earlier one's (Day 06's `chunk_hash` check),
  and also one whose text **without its header** is the same (a copy saved under another title). The real
  corpus has no duplicate texts or bodies, so this changes no measured result; a demo corpus shows what it
  catches.
- **Source diversity:** no per-document cap. *A setting is adopted only if it gains something on dev and
  loses nothing.* On dev a cap of 2 chunks per document changes nothing; a cap of 1 loses d03-13's labelled
  chunk. Both caps are reported on every split as a description, not a choice.
- **Budget:** Day 06's, unchanged: at most 3 chunks and 300 words, best first, whole chunks, skip a chunk
  that doesn't fit. Token counts (with the reranker's tokenizer) and other budgets are reported as a
  description.

## What will be reported on the test split, whatever it shows

1. The weighted fusion reranked (N = 3) against the weighted fusion from `evals/baselines/day-08-retrieval.json`,
   query by query: first place and top 3, with the sign test (15 answerable test questions).
2. RRF reranked (N = 20) against RRF from the same baseline, the same way.
3. The gate (−3.6) on the 27 test questions: answerable kept (of 15), unanswerable refused (of 12), and each
   of the 5 "similar but wrong" questions by name with its score; also together with Day 07's measured
   dense gate (0.62) and Day 06's names rule.
4. Every test-split miss, with a diagnosis.
5. The N sweep and the context-selection grid on test, labelled as descriptions (they decide nothing).

## Decisions, and what each needs

- **D1, reranking in the retriever:** later days build on "weighted fusion, then rerank its top 3" if, on the
  15 answerable test questions, it puts at least as many labelled chunks first as the weighted fusion alone
  (14) and keeps all 15 in the top 3. Otherwise the reranker stays optional and the weighted fusion stays the
  default. Latency is reported; at N = 3 it was about 40 ms per query on dev on this 2-vCPU machine (N = 20:
  about 240 ms), which is accepted for an educational pipeline.
- **D2, the reranker gate:** adopted (recorded for later days, on top of Day 07's measured gate and the names
  rule) if, on the test split, it refuses **none** of the 15 answerable questions and at least one of the
  unanswerable questions that Day 07's measured gate and the names rule let through. Otherwise it is
  recorded as measured and not adopted.
- **"The reranker separates the five"** is claimed only if all 5 score below the gate while all 15
  answerable test questions score at or above it. Anything less is reported as a count.

Day 06's pipeline and `answering.py` don't change today: adopted settings are recorded in the baseline for
a later day to switch on and re-measure.
