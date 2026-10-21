# evals/ — the labelled set and accepted baselines

Started on Day 07 ([lesson](../docs/day-07/LEARN.md), [notebook](../notebooks/day_07_evaluation_baseline.ipynb)). Every retrieval or answering change from Day 08 on is measured on these questions and compared with the accepted baseline.

| File | What it is |
|---|---|
| [`questions-v1.jsonl`](questions-v1.jsonl) | 81 labelled queries, one JSON object per line |
| [`manifest.json`](manifest.json) | version, file hash, counts, the split rules, the user directory, and the **snapshot** the labels belong to (every chunk's text hash and allowed groups) |
| [`baselines/day-07-retrieval.json`](baselines/day-07-retrieval.json) | the accepted Day 07 run: BM25 and dense retrieval, every query's rank, every error with its diagnosis, and Day 06's thresholds as measured |

## One labelled query

```json
{"id": "d06-02", "question": "When do pay rises take effect?", "user": "alice", "kind": "question",
 "split": "dev", "origin": "day06-check", "tags": ["permission"], "evidence": ["take effect from 1 April"],
 "facts": [], "answer_chunks": [], "restricted_chunks": ["manager-pay-review-guide#overview"],
 "note": "Day 06: the answer is in a managers-only guide alice may not read"}
```

(shown on several lines here; one line in the file)

| Field | Meaning |
|---|---|
| `id` | stable id: `d03-..` Day 03 probes, `d01-..` Day 01 failures, `d06-..` Day 06's check list, `d04-v..`/`d04-i..` Day 04's value and identifier lookups, `d07-a..`/`d07-u..` new answerable/unanswerable questions |
| `user` | who asks: `alice` (all-employees) or `maria` (all-employees, managers). The ground truth depends on what this person may read |
| `kind` | `question`, or `lookup` (an exact value typed as the query) |
| `split` | `dev`: may be used to tune. `test`: written before any system ran on it; report only, never tune |
| `evidence` | strings that must **all** appear, as whole words, in a chunk for it to answer the question alone (written by hand) |
| `answer_chunks` | **the label**: the chunks this user may read that contain the evidence (derived from the evidence on the snapshot, then frozen). Empty: the documents don't answer it for this user |
| `restricted_chunks` | chunks that contain the evidence but this user may not read |
| `facts` | strings a correct answer contains (for answer checks; given exactly when the query is answerable) |
| `tags`, `origin`, `note` | what the query tests, where it came from, and any labelling decision worth recording |

## Rules

1. **Labels come from evidence, not from a retriever's output.** Write the evidence from the documents; let `validate()` in the Day 07 notebook tell you which chunks hold it.
2. **Labels belong to a snapshot.** If anything in `data/sample/` or `access-control.json` changes, `tests/test_evals.py` and the notebook fail and list the stale labels. Review each one; don't edit the manifest to make the failure go away.
3. **Any change to the questions is a new version.** Write `questions-v2.jsonl`, keep `v1` (old baselines refer to it), and update the manifest's version, file name, hash, counts and, if the documents changed, the snapshot. The notebook refuses to compare runs across versions.
4. **Test questions are spent once.** A test question that influenced a decision becomes a dev question; write new test questions.
5. **A baseline changes only by a reviewed commit.** Notebooks write their runs to `outputs/` (git-ignored) and compare them with the file here. To accept a new baseline, read the listed changes, copy the run file over the accepted one, and commit it with the reason. Rank changes where the baseline recorded a near tie (a gap under 0.001) are reported separately, as CPU rounding.
6. **No personal data.** These questions are synthetic. A real evaluation set built from user questions needs the same access control and retention as the documents, and doesn't belong in a public repository.

Checked by [`tests/test_evals.py`](../tests/test_evals.py) (no model, a few seconds) and by the Day 07 notebook's sanity checks.
