"""The labelled set in evals/, checked without running any model (a few seconds).

Every check runs for both versions of the questions: questions-v1.jsonl (manifest.json, Day 07) and
questions-v2.jsonl (manifest-v2.json, Day 10: v1's 81 queries unchanged, its used test questions moved to dev,
and 41 new test questions).

1. The question file matches its manifest (hash, counts) and every row has the expected fields.
2. The documents still match the snapshot the labels were written against, and every label can be
   derived again from its evidence. The rule is written out separately from the Day 07 notebook's
   validate(), so the two check each other.
3. Each committed baseline (Day 07's, Day 08's and Day 09's) belongs to this dataset and is consistent
   with itself: its metrics follow from its per-query ranks, its error list is exactly the queries missed
   in the top 3, and no restricted chunk appears in anyone's top 3. A later baseline keeps every entry it
   inherits unchanged: Day 08's BM25 and dense entries are Day 07's (Day 08 changed only how the two are
   combined), and Day 09's four Day 08 retrievers are Day 08's (Day 09 adds reranked versions). Day 09's
   settings are the ones committed in evals/preregistration/day-09.md before its test split was scored. Day 10's v1
   file keeps Day 09's six retrievers unchanged and adds two; its v2 file agrees with it on the 81 shared queries; and
   its map and outcomes are the ones committed in evals/preregistration/day-10.md.

If test 2 fails after you edited data/sample/, that is the point: run the Day 07 notebook, read the
label problems it lists, review those labels, and record a new version (docs/day-07/LEARN.md, section 8).
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from rag_from_scratch import answering
from rag_from_scratch.chunking import section_chunks
from rag_from_scratch.ingest import ingest

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS = REPO_ROOT / "evals"
FIELDS = ["id", "question", "user", "kind", "split", "origin", "tags", "evidence", "facts", "answer_chunks",
          "restricted_chunks", "note"]


MANIFESTS = ["manifest.json", "manifest-v2.json"]
NEW_TEST_ORIGIN = {1: "day07-new", 2: "day10-new"}     # each version's test split is that day's new questions, only


def _manifest(name: str) -> dict:
    return json.loads((EVALS / name).read_text(encoding="utf-8"))


def _rows(manifest: dict) -> list[dict]:
    return [json.loads(line) for line in (EVALS / manifest["questions_file"]).read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module", params=MANIFESTS)
def manifest(request) -> dict:
    return _manifest(request.param)


@pytest.fixture(scope="module")
def rows(manifest: dict) -> list[dict]:
    return _rows(manifest)


@pytest.fixture(scope="module")
def chunks() -> list:
    documents, _ = ingest(REPO_ROOT)
    return [c for d in documents for c in section_chunks(d)]


def holds(evidence, text: str) -> bool:
    # Every evidence string as a whole word or number: "USD 25" must not match inside "USD 250"
    return bool(evidence) and all(re.search(rf"(?<!\w){re.escape(e)}(?!\w|[,.]\d)", text) for e in evidence)


def test_question_file_matches_its_manifest(manifest: dict, rows: list[dict]) -> None:
    data = (EVALS / manifest["questions_file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["questions_sha256"], \
        "the questions changed but the manifest didn't: record the change and a new version"
    assert data.endswith(b"\n") and b"\r" not in data
    assert all(list(r) == FIELDS for r in rows)
    assert len({r["id"] for r in rows}) == len(rows) == len({(r["user"], r["question"]) for r in rows})
    questions = [r for r in rows if r["kind"] == "question"]
    assert manifest["counts"] == {
        "queries": len(rows), "questions": len(questions),
        "answerable_questions": sum(bool(r["answer_chunks"]) for r in questions),
        "unanswerable_questions": sum(not r["answer_chunks"] for r in questions),
        "lookups": sum(r["kind"] == "lookup" for r in rows),
        "dev": sum(r["split"] == "dev" for r in rows), "test": sum(r["split"] == "test" for r in rows)}
    for r in rows:
        assert r["kind"] in ("question", "lookup") and r["split"] in ("dev", "test"), r["id"]
        assert r["user"] in answering.DIRECTORY and r["question"].strip() and r["note"].strip(), r["id"]
        for field in ("tags", "evidence", "facts", "answer_chunks", "restricted_chunks"):
            assert isinstance(r[field], list) and all(isinstance(v, str) and v for v in r[field]), (r["id"], field)
        assert r["answer_chunks"] == sorted(r["answer_chunks"]) and r["restricted_chunks"] == sorted(r["restricted_chunks"])
        assert bool(r["facts"]) == bool(r["answer_chunks"]), f"{r['id']}: facts exactly when answerable"
        assert r["kind"] == "question" or r["answer_chunks"], f"{r['id']}: every lookup has an answer"
    # The test split is that version's new questions, and only those: they were written before any system ran on them
    origin = NEW_TEST_ORIGIN[manifest["version"]]
    assert {r["origin"] for r in rows if r["split"] == "test"} == {origin}
    assert {r["split"] for r in rows if r["origin"] == origin} == {"test"}
    assert manifest["directory"] == {user: sorted(groups) for user, groups in answering.DIRECTORY.items()}
    if manifest["version"] == 2:   # v1's queries, unchanged except the used test questions, which moved to dev (rule 4)
        v1 = _manifest(manifest["previous"]["manifest"])
        assert manifest["previous"]["questions_file"] == v1["questions_file"] and manifest["snapshot"] == v1["snapshot"]
        old = _rows(v1)
        assert [r["id"] for r in rows[:len(old)]] == [r["id"] for r in old]
        for new, before in zip(rows, old):
            moved = before["split"] == "test"
            assert {k: v for k, v in new.items() if k not in ("split", "note")} == \
                   {k: v for k, v in before.items() if k not in ("split", "note")}, new["id"]
            assert new["split"] == "dev" and new["note"] == before["note"] + (
                " (v1 test split; Day 09 used it to decide, so dev from v2 on)" if moved else ""), new["id"]


def test_labels_match_the_documents(manifest: dict, rows: list[dict], chunks: list) -> None:
    snapshot = {c.chunk_id: {"hash": c.chunk_hash, "groups": list(c.allowed_groups)} for c in chunks}
    assert snapshot == manifest["snapshot"]["chunks"], \
        "the documents changed since the labels were written: run the Day 07 notebook to see which labels to review"
    as_text = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(as_text).hexdigest() == manifest["snapshot"]["fingerprint"]
    assert manifest["snapshot"]["chunks_count"] == len(chunks)
    for r in rows:
        groups = answering.DIRECTORY[r["user"]]
        holding = [c for c in chunks if holds(r["evidence"], c.text)]
        assert holding or not r["evidence"], f"{r['id']}: no chunk holds its evidence"
        assert r["answer_chunks"] == sorted(c.chunk_id for c in holding if set(c.allowed_groups) & groups), r["id"]
        assert r["restricted_chunks"] == sorted(c.chunk_id for c in holding if not set(c.allowed_groups) & groups), \
            r["id"]


BASELINES = {"day-07-retrieval.json": "manifest.json", "day-08-retrieval.json": "manifest.json",
             "day-09-retrieval.json": "manifest.json", "day-10-retrieval.json": "manifest.json",
             "day-10-retrieval-v2.json": "manifest-v2.json"}


@pytest.mark.parametrize("name", list(BASELINES))
def test_baseline_belongs_to_this_dataset_and_adds_up(name: str) -> None:
    manifest = _manifest(BASELINES[name])
    rows = _rows(manifest)
    baseline = json.loads((EVALS / "baselines" / name).read_text(encoding="utf-8"))
    assert baseline["dataset"] == {"questions": f"evals/{manifest['questions_file']}", "version": manifest["version"],
                                   "sha256": manifest["questions_sha256"],
                                   "snapshot": manifest["snapshot"]["fingerprint"]}
    by_id, top_k = {r["id"]: r for r in rows}, baseline["top_k"]
    assert set(baseline["queries"]) == set(by_id) and top_k == answering.TOP_K
    # The metrics follow from the per-query ranks
    for kind in ("questions", "lookups"):
        selection = [r for r in rows if r["answer_chunks"] and (r["kind"] == "question") == (kind == "questions")]
        for retriever in baseline["retrievers"]:
            m, entries = baseline["metrics"][f"{kind}, {retriever}"], [baseline["queries"][r["id"]][retriever]
                                                                       for r in selection]
            ranks = [e["first_relevant"] for e in entries]
            assert m["n"] == len(selection)
            for k in (1, 3, 5, 10):
                assert m[f"hit@{k}"] == sum(rank is not None and rank <= k for rank in ranks), (kind, retriever, k)
            mrr = sum(1 / rank for rank in ranks if rank is not None and rank <= 10) / len(ranks)
            recall = sum(len(set(r["answer_chunks"]) & set(e["top3"])) / len(r["answer_chunks"])
                         for r, e in zip(selection, entries)) / len(selection)
            assert m["MRR@10"] == round(mrr, 4) and m["recall@3"] == round(recall, 4), (kind, retriever)
    # Each query's top 3 agrees with its recorded rank, and holds nothing the asker may not read
    for qid, per_retriever in baseline["queries"].items():
        r = by_id[qid]
        assert list(per_retriever) == list(baseline["retrievers"]), qid
        for retriever, e in per_retriever.items():
            in_top = next((i for i, cid in enumerate(e["top3"], start=1) if cid in r["answer_chunks"]), None)
            if not r["answer_chunks"]:
                assert e["first_relevant"] is None and e["margin"] is None, (qid, retriever)
            assert in_top == (e["first_relevant"] if e["first_relevant"] and e["first_relevant"] <= top_k else None)
            assert not set(r["restricted_chunks"]) & set(e["top3"]), f"{qid} {retriever}: a restricted chunk was retrieved"
    # The error list is exactly the answerable queries missed in the top 3, each with a diagnosis
    missed = {(retriever, qid) for qid, per_retriever in baseline["queries"].items()
              for retriever, e in per_retriever.items()
              if by_id[qid]["answer_chunks"] and (e["first_relevant"] is None or e["first_relevant"] > top_k)}
    assert {(e["retriever"], e["id"]) for e in baseline["errors"]} == missed
    assert all(e["category"] and e["why"] for e in baseline["errors"])
    day07 = json.loads((EVALS / "baselines" / "day-07-retrieval.json").read_text(encoding="utf-8"))
    if name == "day-07-retrieval.json":
        assert baseline["thresholds"]["day06"] == {"min_retrieval_score": answering.MIN_RETRIEVAL_SCORE,
                                                   "min_sentence_score": answering.MIN_SENTENCE_SCORE}
    elif name == "day-10-retrieval-v2.json":   # the same retrievers as Day 10's v1 file, the same results on v1's queries
        earlier = json.loads((EVALS / "baselines" / "day-10-retrieval.json").read_text(encoding="utf-8"))
        shared = set(earlier["queries"])
        for retriever in baseline["retrievers"]:
            assert baseline["retrievers"][retriever] == earlier["retrievers"][retriever]
            assert {qid: per[retriever] for qid, per in baseline["queries"].items() if qid in shared} == \
                   {qid: per[retriever] for qid, per in earlier["queries"].items()}
        assert [e for e in baseline["errors"] if e["id"] in shared] == \
               [e for e in earlier["errors"] if e["retriever"] in baseline["retrievers"]]
        assert baseline["thresholds"] == earlier["thresholds"]
    else:   # a later day adds retrievers without changing the ones it inherits: their entries and errors are unchanged
        earlier_name, inherited = {"day-08-retrieval.json": ("day-07-retrieval.json", ("BM25", "dense")),
                                   "day-09-retrieval.json": ("day-08-retrieval.json", ("BM25", "dense", "RRF", "weighted")),
                                   "day-10-retrieval.json": ("day-09-retrieval.json", ("BM25", "dense", "RRF", "weighted",
                                                                                       "weighted+rerank", "RRF+rerank"))}[name]
        earlier = json.loads((EVALS / "baselines" / earlier_name).read_text(encoding="utf-8"))
        for retriever in inherited:
            assert baseline["retrievers"][retriever] == earlier["retrievers"][retriever]
            assert {qid: per[retriever] for qid, per in baseline["queries"].items()} == \
                   {qid: per[retriever] for qid, per in earlier["queries"].items()}
            for kind in ("questions", "lookups"):
                assert baseline["metrics"][f"{kind}, {retriever}"] == earlier["metrics"][f"{kind}, {retriever}"]
        assert [e for e in baseline["errors"] if e["retriever"] in inherited] == earlier["errors"]
    if name == "day-09-retrieval.json":   # the settings the pre-registration committed to, before the test split was scored
        thresholds = baseline["thresholds"]
        prereg = (REPO_ROOT / thresholds["preregistration"]).read_text(encoding="utf-8")
        depths = {r["reranks"]: r["depth"] for r in baseline["retrievers"].values() if "reranks" in r}
        assert depths == {"weighted": 3, "RRF": 20} and "**N = 3**" in prereg and "**N = 20**" in prereg
        assert thresholds["reranker_gate"]["value"] == -3.6 and "**the gate is −3.6**" in prereg
        assert set(thresholds["decisions"].values()) == {"not adopted"}
    if name.startswith("day-10"):   # the map and the outcomes the pre-registration committed to
        thresholds = baseline["thresholds"]
        prereg = (REPO_ROOT / thresholds["preregistration"]).read_text(encoding="utf-8")
        table = {}   # the map's table in the pre-registration: "| the asker says | the documents say | recorded in |"
        for line in prereg.split("| The asker says |", 1)[1].split("\n\n", 1)[0].splitlines()[2:]:
            asker, documents = [cell.strip() for cell in line.strip("|").split("|")][:2]
            table[tuple(asker.split(", "))] = tuple(documents.split(", "))
        recorded = {tuple(phrases): tuple(terms) for phrases, terms in baseline["retrievers"]["BM25+map"]["map"]}
        assert {k: v for k, v in recorded.items() if k not in (("mfa",), ("multi-factor authentication",))} == table
        assert recorded[("mfa",)] == ("multi-factor authentication",) and "Multi-factor authentication (MFA)" in prereg
        assert baseline["retrievers"]["BM25+map"]["expands"].startswith("BM25's query only") and \
               "**the rule chooses lexical.**" in prereg
        outcomes = thresholds["names_rule"]["outcomes"]
        assert [tuple(outcomes[f"{rule}, dev"].values()) for rule in ("Day 06 names rule", "new names rule")] == \
               [(16, 16, 8, 8)] * 2 and "**the new rule gives the same, 16 of 16 and 8 of 8**" in prereg
        assert set(thresholds["decisions"].values()) == {"not adopted"}
