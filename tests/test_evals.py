"""The labelled set in evals/, checked without running any model (a few seconds).

Every check runs for all four versions of the questions: questions-v1.jsonl (manifest.json, Day 07),
questions-v2.jsonl (manifest-v2.json, Day 10: v1's 81 queries unchanged, its used test questions moved to dev,
and 41 new test questions), questions-v3.jsonl (manifest-v3.json, Day 11: v2's 122 queries unchanged, its used
test questions moved to dev, and 68 new test questions) and questions-v4.jsonl (manifest-v4.json, Day 12: v3's 190
queries unchanged, its used test questions moved to dev, and 76 new test questions).

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
   its map and outcomes are the ones committed in evals/preregistration/day-10.md. Day 11's v3 file keeps Day 10's five
   retrievers unchanged on v2's 122 queries, and its section-expansion contexts follow the rules committed in
   evals/preregistration/day-11.md: the hits first, then only their neighbouring sections, within the budget, nothing
   the asker may not read, and the rule's dev values (checked from the chunk texts, without any model).
4. Day 12's answer baseline (day-12-answers-v4.json) is consistent with itself and with the labels: every outcome
   follows from its answer's text and the question's facts, every stage from the outcome, the context and the
   citations; no answer was given anything its asker may not read; the metrics, the paired comparison and the
   decision follow from the per-question entries and the rule committed in evals/preregistration/day-12.md; and
   every answered (question, answer) pair has the author's review, tied to its exact text.

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


MANIFESTS = ["manifest.json", "manifest-v2.json", "manifest-v3.json", "manifest-v4.json"]
NEW_TEST_ORIGIN = {1: "day07-new", 2: "day10-new", 3: "day11-new", 4: "day12-new"}   # each version's test split is
MOVED = {2: " (v1 test split; Day 09 used it to decide, so dev from v2 on)",     # that day's new questions; the note a
         3: " (v2 test split; Day 10 used it to decide, so dev from v3 on)",     # used test question gets when it moves
         4: " (v3 test split; Day 11 used it to decide, so dev from v4 on)"}     # to dev (rule 4)


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
        assert r["user"] in answering.DIRECTORY and r["question"].strip() and isinstance(r["note"], str), r["id"]
        # A note records a labelling decision worth recording. v1 and v2 wrote one for every query; v3's and v4's new
        # questions leave it empty when there is nothing to record
        assert r["note"].strip() or (manifest["version"] >= 3 and r["origin"] == NEW_TEST_ORIGIN[manifest["version"]]), r["id"]
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
    if manifest["version"] > 1:   # the previous version's queries, unchanged except the used test questions (rule 4)
        previous = _manifest(manifest["previous"]["manifest"])
        assert previous["version"] == manifest["previous"]["version"] == manifest["version"] - 1
        assert manifest["previous"]["questions_file"] == previous["questions_file"]
        assert manifest["snapshot"] == previous["snapshot"]
        old = _rows(previous)
        assert [r["id"] for r in rows[:len(old)]] == [r["id"] for r in old]
        for new, before in zip(rows, old):
            moved = before["split"] == "test"
            assert {k: v for k, v in new.items() if k not in ("split", "note")} == \
                   {k: v for k, v in before.items() if k not in ("split", "note")}, new["id"]
            expected = (before["note"] + MOVED[manifest["version"]]).strip() if moved else before["note"]
            assert new["split"] == "dev" and new["note"] == expected, new["id"]   # an empty note gets the suffix alone


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
             "day-10-retrieval-v2.json": "manifest-v2.json", "day-11-retrieval-v3.json": "manifest-v3.json"}
CONTEXTS = {"weighted+neighbours"}   # Day 11: what was sent to the reader, not a ranking (checked separately below)


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
        for retriever in set(baseline["retrievers"]) - CONTEXTS:
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
              for retriever, e in per_retriever.items() if retriever not in CONTEXTS
              if by_id[qid]["answer_chunks"] and (e["first_relevant"] is None or e["first_relevant"] > top_k)}
    assert {(e["retriever"], e["id"]) for e in baseline["errors"] if e["retriever"] in baseline["retrievers"]
            and e["retriever"] not in CONTEXTS} == missed
    assert all(e["category"] and e["why"] for e in baseline["errors"])
    day07 = json.loads((EVALS / "baselines" / "day-07-retrieval.json").read_text(encoding="utf-8"))
    if name == "day-07-retrieval.json":
        assert baseline["thresholds"]["day06"] == {"min_retrieval_score": answering.MIN_RETRIEVAL_SCORE,
                                                   "min_sentence_score": answering.MIN_SENTENCE_SCORE}
    elif name in ("day-10-retrieval-v2.json", "day-11-retrieval-v3.json"):   # the previous file's retrievers, the same
        earlier = json.loads((EVALS / "baselines" / {"day-10-retrieval-v2.json": "day-10-retrieval.json",   # results on
                                                     "day-11-retrieval-v3.json": "day-10-retrieval-v2.json"}[name])  # its
                             .read_text(encoding="utf-8"))                                                  # queries
        shared = set(earlier["queries"])
        for retriever in set(baseline["retrievers"]) - CONTEXTS:
            assert baseline["retrievers"][retriever] == earlier["retrievers"][retriever]
            assert {qid: per[retriever] for qid, per in baseline["queries"].items() if qid in shared} == \
                   {qid: per[retriever] for qid, per in earlier["queries"].items()}
        assert [e for e in baseline["errors"] if e["id"] in shared and e["retriever"] in earlier["retrievers"]] == \
               [e for e in earlier["errors"] if e["retriever"] in baseline["retrievers"]]
        if name == "day-10-retrieval-v2.json":
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


def test_day_11_contexts_follow_the_preregistered_rules(chunks: list) -> None:
    # Every context in Day 11's baseline, rebuilt from the chunk texts and the document structure (no model): the hits are
    # the weighted fusion's top 3, then only sections next to a hit in the same document, nothing the asker may not read,
    # within the budget; its ranks, words, control results, metrics and errors follow; the rule's dev values are the
    # pre-registered ones.
    manifest = _manifest("manifest-v3.json")
    by_id = {r["id"]: r for r in _rows(manifest)}
    baseline = json.loads((EVALS / "baselines" / "day-11-retrieval-v3.json").read_text(encoding="utf-8"))
    settings, name = baseline["retrievers"]["weighted+neighbours"], "weighted+neighbours"
    assert settings["setting"] == "neighbours" and settings["hits"] == answering.TOP_K
    assert settings["budget_words"] == answering.MAX_CONTEXT_WORDS
    by_cid = {c.chunk_id: c for c in chunks}
    sections = {}
    for c in chunks:
        sections.setdefault(c.doc_id, []).append(c.chunk_id)
    neighbours = {cid: set(ids[max(0, i - 1):i] + ids[i + 1:i + 2]) for ids in sections.values() for i, cid in enumerate(ids)}
    words = {cid: len(c.text.split()) for cid, c in by_cid.items()}
    in_context = {}
    for qid, per in baseline["queries"].items():
        r, e, hits = by_id[qid], per[name], per["weighted"]["top3"]
        groups = answering.DIRECTORY[r["user"]]
        assert e["top3"] == hits == e["sent"][:3] and e["top_score"] == per["weighted"]["top_score"], qid
        assert len(set(e["sent"])) == len(e["sent"]) and set(e["sent"][3:]) <= set().union(*(neighbours[h] for h in hits)), qid
        assert e["words"] == sum(words[cid] for cid in e["sent"]) <= settings["budget_words"], qid
        assert all(set(by_cid[cid].allowed_groups) & groups for cid in e["sent"]), f"{qid}: a section the asker may not read"
        assert not set(r["restricted_chunks"]) & set(e["sent"]), qid
        found_at = next((i for i, cid in enumerate(e["sent"], start=1) if cid in r["answer_chunks"]), None)
        assert e["first_relevant"] == (found_at if r["answer_chunks"] else None), qid
        if r["answer_chunks"]:
            rank = per["weighted"]["first_relevant"]           # the control: the weighted fusion's top m, m = sections sent
            assert e["control_found"] == (rank is not None and rank <= len(e["sent"])), qid
            in_context[qid] = (per["weighted"]["first_relevant"] is not None and per["weighted"]["first_relevant"] <= 3,
                               found_at is not None, e["control_found"])
        else:
            assert e["margin"] is None and e["control_found"] is None, qid
    for label, ids in (("questions", [q for q in in_context if by_id[q]["kind"] == "question"]),
                       ("lookups", [q for q in in_context if by_id[q]["kind"] == "lookup"]),
                       ("v3 test questions", [q for q in in_context if by_id[q]["split"] == "test"])):
        m = baseline["metrics"][f"{label}, {name}"]
        assert m["n"] == len(ids) and [m["top 3 in context"], m["in context"], m["control in context"]] == \
               [sum(in_context[q][i] for q in ids) for i in range(3)], label
    errors = {(e["retriever"], e["id"]) for e in baseline["errors"]}
    assert {k for k in errors if k[0] == name} == {(name, q) for q, v in in_context.items() if not v[1]}
    assert {k for k in errors if k[0] == "control (weighted top m)"} == \
           {("control (weighted top m)", q) for q, v in in_context.items() if not v[2]}
    # The rule's dev values and the decision are the pre-registered ones
    thresholds = baseline["thresholds"]
    prereg = (REPO_ROOT / thresholds["preregistration"]).read_text(encoding="utf-8")
    assert "**The rule chooses neighbours.**" in prereg
    for setting, values in thresholds["setting_rule_on_dev"].items():
        assert f"| {setting} | {values['found']} | {values['mean_words']} |" in prereg, setting
    assert thresholds["decisions"] == {"D1 section expansion in the default context": "not adopted"}


def test_day_12_answers_add_up(chunks: list) -> None:
    # Day 12's answer baseline, checked against the labels and itself (no model): every outcome from its text and the
    # facts, every stage from the outcome, the context and the citations, nothing sent that the asker may not read;
    # the metrics, the paired comparison and the decision from the entries and the pre-registered rule; and the
    # author's review of every answered (question, answer) pair, tied to its exact text.
    manifest = _manifest("manifest-v4.json")
    questions = {r["id"]: r for r in _rows(manifest) if r["kind"] == "question"}
    baseline = json.loads((EVALS / "baselines" / "day-12-answers-v4.json").read_text(encoding="utf-8"))
    assert baseline["dataset"] == {"questions": f"evals/{manifest['questions_file']}", "version": manifest["version"],
                                   "sha256": manifest["questions_sha256"], "snapshot": manifest["snapshot"]["fingerprint"]}
    assert set(baseline["questions"]) == set(questions) and list(baseline["pipelines"]) == ["current", "names-v2"]
    prereg = (REPO_ROOT / baseline["decisions"]["preregistration"]).read_text(encoding="utf-8")
    assert "answerable correct **+1**, refused **0**, wrong **−2**" in prereg
    assert baseline["rubric"]["value"] == {"correct": 1, "refused": 0, "wrong": -2, "answered (unanswerable)": -2}
    value = {"correct": 1, "refused": 0, "wrong": -2, "answered": -2}
    by_cid = {c.chunk_id: c for c in chunks}
    for qid, e in baseline["questions"].items():
        r, top3 = questions[qid], e["retrieval"]["top3"]
        assert e["split"] == r["split"] and len(top3) == 3, qid
        assert all(set(by_cid[cid].allowed_groups) & answering.DIRECTORY[r["user"]] for cid in top3), qid
        label_in = bool(set(top3) & set(r["answer_chunks"])) if r["answer_chunks"] else None
        for p in baseline["pipelines"]:
            a = e[p]
            assert set(a["cited"]) <= set(top3), (qid, p)
            if a["status"] != "answered":
                expected = "refused"
            elif not r["answer_chunks"]:
                expected = "answered"
            else:
                expected = "correct" if all(f.lower() in a["text"].lower() for f in r["facts"]) else "wrong"
            assert a["outcome"] == expected and a["label_in_context"] == label_in, (qid, p)
            if a["status"] == "answered":
                assert a["cited"] and a["grounded"] == (a["support"] >= 0.75) and a["refused_by"] is None, (qid, p)
                if r["answer_chunks"]:
                    assert a["cites_label"] == bool(set(a["cited"]) & set(r["answer_chunks"])), (qid, p)
                    assert a["citation_precision"] == round(sum(c in r["answer_chunks"] for c in a["cited"]) /
                                                            len(a["cited"]), 3), (qid, p)
            else:
                assert a["cites_label"] is None and a["grounded"] is None and a["refused_by"], (qid, p)
            if not r["answer_chunks"]:
                stage = "ok" if expected == "refused" else \
                    "answered: restricted to others" if r["restricted_chunks"] else "answered: nothing answers it"
            elif expected == "correct":
                stage = "ok"
            elif expected == "refused":
                stage = a["stage"] if a["refused_by"] == "names rule" and a["stage"].startswith("names rule: ") \
                    else a["refused_by"]
            else:
                stage = "retrieval: answer not in the context" if not label_in else \
                    "reader: right source, wrong part" if a["cites_label"] else "reader: answer in the context, not used"
            assert a["stage"] == stage, (qid, p)
    # The metrics, the paired comparison and the decision follow from the entries
    for split, label in (("dev", "dev"), ("test", "v4 test")):
        ids = [qid for qid, r in questions.items() if r["split"] == split]
        for p in baseline["pipelines"]:
            outs = {qid: baseline["questions"][qid][p]["outcome"] for qid in ids}
            ans = [o for qid, o in outs.items() if questions[qid]["answer_chunks"]]
            un = [o for qid, o in outs.items() if not questions[qid]["answer_chunks"]]
            assert baseline["metrics"][label][p] == {
                "correct": ans.count("correct"), "wrong": ans.count("wrong"), "refused": ans.count("refused"),
                "unanswerable refused": un.count("refused"), "unanswerable answered": un.count("answered"),
                "value at cost 2": sum(value[o] for o in outs.values())}, (label, p)
        changed = [qid for qid in ids if baseline["questions"][qid]["current"]["outcome"] !=
                   baseline["questions"][qid]["names-v2"]["outcome"]]
        near = [qid for qid in changed if min(baseline["questions"][qid][p]["closest_call"][0] for p in
                                              baseline["pipelines"]) < 0.001]
        order = {"correct": 2, "refused": 1, "wrong": 0, "answered": 0}
        kept = [qid for qid in changed if qid not in near]
        move = {qid: order[baseline["questions"][qid]["names-v2"]["outcome"]] -
                order[baseline["questions"][qid]["current"]["outcome"]] for qid in kept}
        recorded = baseline["decisions"][label]
        assert recorded["near"] == near and recorded["better"] == [q for q in kept if move[q] > 0] and \
            recorded["worse"] == [q for q in kept if move[q] < 0], label
        for cost in (1, 2, 3):
            v = {"correct": 1, "refused": 0, "wrong": -cost, "answered": -cost}
            assert recorded["delta"][str(cost)] == sum(v[baseline["questions"][q]["names-v2"]["outcome"]] -
                                                       v[baseline["questions"][q]["current"]["outcome"]] for q in kept)
    adopted = baseline["decisions"]["v4 test"]["delta"]["2"] > 0
    assert baseline["decisions"]["D1 Day 10's names rule in the current pipeline"] == \
        ("adopted" if adopted else "not adopted")
    for p, line in (("current", "current pipeline"), ("names-v2", "candidate (Day 10's names rule)")):
        m = baseline["metrics"]["dev"][p]
        assert f"| {line} | {m['correct']} / {m['wrong']} / {m['refused']} | {m['unanswerable refused']} / " \
               f"{m['unanswerable answered']} | {m['value at cost 2']} |" in prereg, p
    # The author's reading: one review per distinct answered (question, answer) pair, tied to the exact text
    reviews = [json.loads(line) for line in (REPO_ROOT / baseline["reviews"]["file"]).read_text(encoding="utf-8")
               .splitlines() if line.strip()]
    answered = {(qid, hashlib.sha256(e[p]["text"].encode("utf-8")).hexdigest()[:16]): (qid, p)
                for qid, e in baseline["questions"].items() for p in baseline["pipelines"] if e[p]["status"] == "answered"}
    assert [(r["id"], r["answer_sha256"]) for r in reviews] == list(answered), "one review per answered pair, in order"
    for r in reviews:
        assert set(r) == {"id", "answer_sha256", "grade", "note", "split"} and r["split"] == questions[r["id"]]["split"]
        assert r["grade"] in ("correct", "partly", "wrong") and isinstance(r["note"], str), r
        assert questions[r["id"]]["answer_chunks"] or r["grade"] == "wrong", r      # nothing answers it
        assert r["grade"] == "correct" or r["note"], r                               # a reason for anything else
    for split, label in (("dev", "dev"), ("test", "v4 test")):
        mine = [(r, answered[r["id"], r["answer_sha256"]]) for r in reviews if r["split"] == split]
        agree = sum((baseline["questions"][qid][p]["outcome"] == "correct") == (r["grade"] == "correct")
                    for r, (qid, p) in mine)
        assert baseline["reviews"][label]["pairs"] == len(mine) and baseline["reviews"][label]["agree"] == agree, label
