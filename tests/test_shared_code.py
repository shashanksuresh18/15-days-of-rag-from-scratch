"""The shared modules in src/rag_from_scratch/ are copies of code taught in earlier notebooks.

These tests run each notebook's own code (its code cells, in order, from the repo root) and
check that the shared module gives exactly the same results. If someone edits a notebook or a
module and the two drift apart, a test here fails.

Memory: the tests are grouped in one class per day, and each class creates its own notebook
namespace and embedding model (class-scoped fixtures). When a day's tests finish, its namespace is
cleared and garbage-collected, so only one day's notebook and model are in memory at a time
(Day 07 measured this file's peak memory before and after; docs/day-07/LEARN.md, section 14).
Day 08 added the Day 07 class (the evaluation harness in evaluation.py); Day 09 the Day 08 class (rank fusion in
fusion.py); Day 10 the Day 09 class (the cross-encoder reranker in reranking.py) and a test of outcome(), which Day 10
copied from the Day 07 notebook into evaluation.py; Day 11 the Day 10 class (query analysis and the map in queries.py),
which runs only the Day 10 cells that define them, so it doesn't load a model. Day 12 added the Day 11 class (the
documents' structure, section expansion and the end-to-end steps with a chosen context, in contexts.py), a test of
Day 10's revised names rule and NamesReader (queries.py, with a stand-in embedder, so the Day 10 class still loads no
model), and tests of Day 06's prompt and language-model client (chat.py) against local fake servers. The Day 11 class
and the new Day 10 test execute only the definitions in the cells they need (imports, functions, classes, constants),
not the computations around them. Day 13 added the Day 12 class (the answer rubric in rubric.py: verdicts, values,
closest calls and the comparison with an accepted answer baseline), which runs the Day 12 notebook's setup cells and
only the definitions it needs, on answers built by already-tested shared code. Its local fake servers don't print a
traceback when a client gives up first (QuietServer).
"""

import ast
import dataclasses
import gc
import hashlib
import json
import sys
import threading
import time
import types
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from rag_from_scratch import (answering, chat, chunking, contexts, corpus, embeddings, evaluation, fusion, index, queries,
                              reranking, rubric)
from rag_from_scratch.bm25 import BM25Index
from rag_from_scratch.ingest import ingest
from rag_from_scratch.probes import DAY03_PROBES

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "notebooks"


def run_notebook_code(name: str) -> dict:
    """Execute a notebook's code cells as one module, from the repo root, and return its variables."""
    nb = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    source = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    module = types.ModuleType(name.removesuffix(".ipynb"))
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(REPO_ROOT)
        mp.delenv("DAY06_LIVE_MODEL", raising=False)       # never call a real language model from a test
        mp.setitem(sys.modules, module.__name__, module)   # dataclasses look up their module
        exec(compile(source, name, "exec"), module.__dict__)
    return module.__dict__


class QuietServer(ThreadingHTTPServer):
    """A local test server that doesn't print a traceback when a client closes the connection first (a client that
    timed out on purpose); anything else is still printed."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            super().handle_error(request, client_address)


def release(namespace: dict) -> None:
    """Drop a notebook's variables (its model session, arrays, indexes) as soon as its tests are done."""
    namespace.clear()
    gc.collect()        # the namespace's functions refer back to it, so only the cycle collector frees it


@pytest.fixture(scope="class")
def embedder():
    model = embeddings.Embedder(REPO_ROOT)     # one per class that asks for it, released with the class
    yield model
    del model
    gc.collect()


def _as_dicts(chunks) -> list[dict]:
    return [asdict(c) for c in chunks]


def _results(results) -> list[tuple]:
    # Result objects from two modules: compare their fields, with the chunk as a dict
    return [(r.rank, r.score, r.id, asdict(r.chunk)) for r in results]


def _outcome(call):
    # A return value, or the error type and its full message: both versions must agree on either
    try:
        return ("ok", call())
    except Exception as error:
        return (type(error).__name__, str(error))


# ----------------------------------------------------------------------------- Days 01–03: ingest, BM25, chunking
@pytest.fixture(scope="class")
def day01():
    ns = run_notebook_code("day_01_rag_foundations.ipynb")
    yield ns
    release(ns)


@pytest.fixture(scope="class")
def day03():
    ns = run_notebook_code("day_03_chunking_and_structure.ipynb")
    yield ns
    release(ns)


class TestDays01To03:
    def test_ingest_module_matches_day_02_notebook(self) -> None:
        notebook = run_notebook_code("day_02_ingestion_and_metadata.ipynb")
        documents, quarantined = ingest(REPO_ROOT)
        assert [asdict(d) for d in documents] == [asdict(d) for d in notebook["documents"]]
        assert [asdict(q) for q in quarantined] == [asdict(q) for q in notebook["quarantined"]]
        assert len(documents) == 11 and len(quarantined) == 2
        release(notebook)

    @pytest.mark.parametrize("question", [
        "How much is the home office allowance for remote employees?",
        "How many days do I have to submit an expense claim?",
        "How long do we keep application logs?",
        "How much time off do new dads get?",
        "What is Google's remote work policy?",
        "My account is locked, what now?",
        "How long is parental leave?",
    ])
    def test_bm25_module_matches_day_01_notebook(self, question: str, day01: dict) -> None:
        chunks, searchable_text = day01["chunks"], day01["searchable_text"]
        shared = BM25Index({c.chunk_id: searchable_text(c) for c in chunks})
        expected = day01["index"].search(question, k=len(chunks))
        actual = shared.search(question, k=len(chunks))
        assert [(r.chunk.chunk_id, r.score, r.contributions) for r in expected] == \
               [(h.id, h.score, h.contributions) for h in actual]

    def test_chunking_module_matches_day_03_notebook(self, day03: dict) -> None:
        documents = day03["documents"]
        # The notebook's own 44 chunks, rebuilt by the module with its defaults
        shared = [c for d in documents for c in chunking.section_chunks(d)]
        assert _as_dicts(shared) == _as_dicts(day03["structured"])
        assert len(shared) == 44
        # The same function called directly, with every setting the notebook uses
        for d in documents:
            assert _as_dicts(chunking.section_chunks(d)) == _as_dicts(day03["section_chunks"](d))
            assert _as_dicts(chunking.section_chunks(d, context_header=False)) == \
                   _as_dicts(day03["section_chunks"](d, context_header=False))
            for max_words in (30, 50, 120):   # splitting paths: sentences, table rows, repeated header
                assert _as_dicts(chunking.section_chunks(d, max_words=max_words)) == \
                       _as_dicts(day03["section_chunks"](d, max_words=max_words))
            assert chunking.split_sections(d.body) == day03["split_sections"](d.body)
            assert chunking.count_words(d.body) == day03["count_words"](d.body)
        # The edited document from the notebook's update demo
        edited = day03["edited"]
        assert _as_dicts(chunking.section_chunks(edited)) == _as_dicts(day03["section_chunks"](edited))
        assert chunking.MAX_WORDS == day03["MAX_WORDS"]

    def test_probes_module_matches_day_03_notebook(self, day03: dict) -> None:
        assert [asdict(p) for p in DAY03_PROBES] == [asdict(q) for q in day03["QUESTIONS"]]
        assert len(DAY03_PROBES) == 13

    EDGE_LINES = [
        "a|b", "a | b", "Header | Value", " | ", "no pipes here", "## Heading", "C++ & Rust: 2 langs?",
        "Is it fast? Yes! It is. Really... ok", "x" * 5, "",
    ]
    EDGE_PARAGRAPH = ("Can you split this? Yes! Short one. Another sentence follows here, with a comma. "
                      "Is a question still a sentence? It should be. Done!")

    def test_chunking_helpers_match_day_03_notebook_on_edge_cases(self, day03: dict) -> None:
        # The sample corpus does not exercise every rule (no "?" inside an oversize paragraph, no "|"
        # without spaces), so compare the helpers directly on small made-up inputs as well.
        for line in self.EDGE_LINES:
            assert chunking.is_table_row(line) == day03["is_table_row"](line), line
            assert chunking.count_words(line) == day03["count_words"](line), line
            assert chunking.slugify(line) == day03["slugify"](line), line
        paragraph = self.EDGE_PARAGRAPH
        table = ["Name | Limit | Note", "A | 1 | x", "B | 2 | y y y y", "C | 3 | z", "D | 4 | w w"]
        units = [[paragraph], ["- a list item"], table, ["a|b without spaces"]]
        for room in (3, 6, 10, 25, 200):
            assert chunking.split_unit([paragraph], room) == day03["split_unit"]([paragraph], room)
            assert chunking.split_unit(table, room) == day03["split_unit"](table, room)
            assert chunking.pack(units, room) == day03["pack"](units, room)
        lines = ["Intro line.", *table, "After the table.", "x | y", "Last."]
        assert chunking.to_units(lines) == day03["to_units"](lines)
        body = "Preface text.\n## One\n" + "\n".join(lines) + "\n\n## Two?\n" + paragraph
        assert chunking.split_sections(body) == day03["split_sections"](body)


# ----------------------------------------------------------------------------- Day 04: embeddings
LONG_TEXT = " ".join(f"word{i} policy approval" for i in range(700))   # over 512 tokens: truncation path
EDGE_TEXTS = ["", "x", "USD 220", "How much time off do new dads get?", LONG_TEXT, "Ünïcödé — ✓ 15:00"]


@pytest.fixture(scope="class")
def day04(embedder: embeddings.Embedder):
    # Runs the whole Day 04 notebook (downloads the model on first use, then embeds offline). Afterwards the
    # notebook's run_model() uses the module's ONNX session: both load the same SHA-256-checked file, and each
    # session keeps its peak working memory, which for the 512-token batches below is several hundred MB.
    # The notebook's own tokenizer, input arrays, pooling and normalisation are still the ones compared, and the
    # first assertion below compares the module with the 44 vectors the notebook made with its own session.
    ns = run_notebook_code("day_04_embeddings_and_similarity.ipynb")
    assert ns["INPUT_NAMES"] == embedder.input_names
    ns["session"] = embedder.session
    gc.collect()
    yield ns
    release(ns)


class TestDay04Embeddings:
    def test_embedding_settings_match_day_04_notebook(self, day04: dict, embedder: embeddings.Embedder) -> None:
        assert embeddings.MODEL == day04["MODEL"]
        assert embeddings.QUERY_INSTRUCTION == day04["QUERY_INSTRUCTION"]
        assert embeddings.MAX_TOKENS == day04["MAX_TOKENS"]
        assert embeddings.default_model_dir(REPO_ROOT) == day04["MODEL_DIR"] == embedder.model_dir
        assert embedder.revision == day04["MODEL"]["revision"] and embedder.dimensions == 384
        assert embedder.tokenizer.truncation == day04["tokenizer"].truncation
        assert embedder.tokenizer.padding == day04["tokenizer"].padding
        assert embedder.input_names == day04["INPUT_NAMES"]

    def test_embeddings_match_day_04_notebook(self, day04: dict, embedder: embeddings.Embedder) -> None:
        chunk_texts = [c.text for c in day04["chunks"]]
        headerless_texts = [c.text for c in day04["headerless"]]
        # The 44 chunk vectors the notebook computed, recomputed by the module: identical, not just close
        assert np.array_equal(embedder.embed(chunk_texts), day04["chunk_vectors"])
        assert np.array_equal(embedder.embed(headerless_texts), day04["headerless_vectors"])
        # The same function on other inputs and batch sizes (16 is the default; 17 texts cross a batch boundary)
        for texts in (EDGE_TEXTS, chunk_texts[:17], chunk_texts[5:6]):
            for batch_size in (1, 5, 16):
                assert np.array_equal(embedder.embed(texts, batch_size=batch_size),
                                      day04["embed"](texts, batch_size=batch_size))
        # Queries: with the model card's instruction, and without it (section 8's knob)
        questions = [p.question for p in day04["QUESTIONS"]] + [p.question for p in day04["LOOKUPS"]["values"][:5]]
        for question in questions + EDGE_TEXTS:
            assert np.array_equal(embedder.embed_query(question), day04["embed_query"](question))
            assert np.array_equal(embedder.embed_query(question, instruction=""), day04["embed_query"](question, ""))
        batch = embedder.tokenizer.encode_batch(EDGE_TEXTS)
        for name, array in embeddings.model_inputs(batch).items():
            assert np.array_equal(array, day04["model_inputs"](day04["tokenizer"].encode_batch(EDGE_TEXTS))[name])

    def test_model_file_checks_match_day_04_notebook(self, day04: dict, tmp_path: Path, monkeypatch, capsys) -> None:
        # Point both versions at a scratch model folder, so the real files are never touched.
        model_dir = tmp_path / "models" / "bge-small-en-v1.5" / embeddings.MODEL["revision"][:12]
        model_dir.mkdir(parents=True)
        monkeypatch.setitem(day04, "REPO_ROOT", tmp_path)
        monkeypatch.setitem(day04, "MODEL_DIR", model_dir)
        real = (embeddings.default_model_dir(REPO_ROOT) / "tokenizer.json").read_bytes()
        target = model_dir / "tokenizer.json"

        def both(name: str):
            results = []
            for call in (lambda: day04["model_file"](name), lambda: embeddings.model_file(name, model_dir, tmp_path)):
                try:
                    results.append(("ok", call()))
                except Exception as error:   # compare the type and the full message
                    results.append((type(error).__name__, str(error)))
            return results

        # 1. A good file: same path, same printed line
        target.write_bytes(real)
        capsys.readouterr()
        notebook, module = both("tokenizer.json")
        printed = capsys.readouterr().out.splitlines()
        assert notebook == module == ("ok", target) and printed[0] == printed[1] and printed[0].endswith("verified")
        assert embeddings.file_sha256(target) == day04["file_sha256"](target) == \
               embeddings.MODEL["files"]["tokenizer.json"][0]

        # 2. One byte changed: both refuse to load it, with the same message
        target.write_bytes(real[:-1] + bytes([real[-1] ^ 1]))
        notebook, module = both("tokenizer.json")
        assert notebook == module and notebook[0] == "ValueError" and "Never load a model file" in notebook[1]

        # 3. Missing file and a failing download (no network needed): same helpful error
        target.unlink()

        def no_network(*args, **kwargs):
            raise OSError("network unreachable (simulated by the test)")

        monkeypatch.setitem(day04, "hf_hub_download", no_network)
        monkeypatch.setattr(embeddings, "hf_hub_download", no_network)
        notebook, module = both("tokenizer.json")
        assert notebook == module and notebook[0] == "RuntimeError" and "HF_ENDPOINT" in notebook[1]


# ----------------------------------------------------------------------------- Day 05: vector index and sync
SEARCH_QUESTIONS = [p.question for p in DAY03_PROBES] + [
    "How much time off do new dads get?", "When do pay rises take effect?", "How do I connect to the VPN?",
    "What is Google's remote work policy?", "What is the capital of France?", "My account is locked, what now?",
]


def _shared_index(day05: dict) -> index.VectorIndex:
    shared = index.VectorIndex(day05["index"].model, day05["index"].revision, day05["index"].dimensions)
    shared.upsert(day05["chunks"], day05["chunk_vectors"])
    return shared


@pytest.fixture(scope="class")
def day05():
    # Runs the whole Day 05 notebook (it saves outputs/day-05/index.npz, which Day 06 reads).
    ns = run_notebook_code("day_05_vector_search_and_lifecycle.ipynb")
    del ns["vectors"]   # section 7's 100,000 random vectors (154 MB): only used for timing, not by these tests
    gc.collect()
    yield ns
    release(ns)


class TestDay05Index:
    def test_index_settings_match_day_05_notebook(self, day05: dict) -> None:
        assert index.TOP_K == day05["TOP_K"] and index.INDEX_FORMAT == day05["INDEX_FORMAT"]
        for groups in ({"managers", "all-employees"}, ["people-team", "managers", "all-employees"], (), ["x"]):
            assert index.groups_filter(groups) == day05["groups_filter"](groups)   # a list in any order, sorted
        rng = np.random.default_rng(6)
        for n in (0, 1, 5, 44):
            for scores in (rng.standard_normal(n).astype(np.float32), rng.integers(0, 3, n).astype(np.float32)):
                for k in range(0, n + 3):                                             # (the second has ties)
                    assert np.array_equal(index.top_k(scores, k), day05["top_k"](scores, k)), (n, k)

    def test_index_search_matches_day_05_notebook(self, day05: dict, embedder: embeddings.Embedder) -> None:
        notebook, shared = day05["index"], _shared_index(day05)
        assert shared.records == notebook.records and np.array_equal(shared.vectors, notebook.vectors)
        assert shared.row == notebook.row and len(shared) == len(notebook) == 44
        groups = day05["groups_filter"]
        filters = [None, *day05["FILTERS"].values(), groups({"managers"}), groups(set()),
                   {"doc_id": "no-such-document"}, {"allowed_groups": "managers"},
                   {"section": "Overview", "last_reviewed": {"gte": "2025-01-01"}}]
        for question in SEARCH_QUESTIONS:
            q = embedder.embed_query(question)
            for where in filters:
                for mode in ("pre", "post"):
                    for k in (1, 3, 5, 44, 60):
                        assert _results(shared.search(q, k, where, mode)) == \
                               _results(notebook.search(q, k, where, mode)), (question, where, mode, k)
            assert _results(shared.search(q)) == _results(notebook.search(q))          # the defaults
        q = embedder.embed_query("x")
        assert _outcome(lambda: shared.search(q, filter_mode="both")) == \
               _outcome(lambda: notebook.search(q, filter_mode="both"))

    def test_index_filters_match_day_05_notebook(self, day05: dict) -> None:
        conditions = [*day05["FILTERS"].values(), {"allowed_groups": "managers"}, {"allowed_groups": {"any": []}},
                      {"doc_id": "travel-policy", "allowed_groups": {"any": ["all-employees"]}},
                      {"last_reviewed": {"gte": "2026-04-15"}}, {"last_reviewed": "2026-04-15"},
                      {"allowed_groups": {"lt": 3}}, {"owner": "IT"}, {"last_reviewed": {"gte": None}}]
        for chunk in day05["chunks"]:
            for where in conditions:   # includes an unknown operator, a misspelled field and an impossible comparison
                assert _outcome(lambda: index.matches(chunk, where)) == \
                       _outcome(lambda: day05["matches"](chunk, where))

    def test_index_file_matches_day_05_notebook(self, day05: dict, tmp_path: Path) -> None:
        notebook, shared = day05["index"], _shared_index(day05)
        model, revision = notebook.model, notebook.revision
        # Saved by one version, loaded by the other: identical records and vectors
        index.save_index(shared, tmp_path / "shared.npz")
        day05["save_index"](notebook, tmp_path / "notebook.npz")
        for path in (tmp_path / "shared.npz", tmp_path / "notebook.npz"):
            a, b = index.load_index(path, model, revision), day05["load_index"](path, model, revision)
            assert a.records == b.records == notebook.records and np.array_equal(a.vectors, b.vectors)
            with np.load(path, allow_pickle=False) as data:
                assert sorted(data.files) == ["manifest", "records", "vectors"]
        for name in ("manifest", "records", "vectors"):   # the arrays inside both files are identical
            with np.load(tmp_path / "shared.npz") as a, np.load(tmp_path / "notebook.npz") as b:
                assert np.array_equal(a[name], b[name])
        assert not list(tmp_path.glob("*.tmp"))

        # Every refusal: same error type and message from both loaders
        def variant(name: str, **change) -> Path:
            with np.load(tmp_path / "shared.npz") as data:
                arrays = {k: data[k] for k in data.files}
            manifest = json.loads(arrays["manifest"].tobytes().decode("utf-8"))
            records = json.loads(arrays["records"].tobytes().decode("utf-8"))
            manifest.update(change.get("manifest", {}))
            records = change.get("records", records)
            as_bytes = lambda obj: np.frombuffer(json.dumps(obj).encode("utf-8"), dtype=np.uint8)
            path = tmp_path / f"{name}.npz"
            np.savez(path, vectors=change.get("vectors", arrays["vectors"]), manifest=as_bytes(manifest),
                     records=as_bytes(records))
            return path

        records = [asdict(r) for r in shared.records]
        broken = [variant("format", manifest={"format": 2}), variant("count", manifest={"count": 43}),
                  variant("duplicates", records=records[:-1] + [records[0]])]
        for path, refusal in zip(broken, ("ValueError", "ValueError", "IndexError")):
            assert _outcome(lambda: len(index.load_index(path, model, revision))) == \
                   _outcome(lambda: len(day05["load_index"](path, model, revision)))
            # Known quirk, kept identical on purpose: a duplicate ID inside one upsert batch makes upsert write to
            # a row that isn't stacked yet, so load_index raises IndexError before its "duplicate chunk IDs" check.
            # The file is still refused. Fixing it means changing the notebook and this module together.
            assert _outcome(lambda: index.load_index(path, model, revision))[0] == refusal
        other = "0123456789abcdef0123456789abcdef01234567"
        assert _outcome(lambda: index.load_index(tmp_path / "shared.npz", model, other)) == \
               _outcome(lambda: day05["load_index"](tmp_path / "shared.npz", model, other))
        # A save that fails halfway leaves the previous file loadable and no temporary file
        bad = index.VectorIndex(model, revision, shared.dimensions)
        bad.upsert([dataclasses.replace(shared.records[0], allowed_groups={"x"})], shared.vectors[:1])
        for save in (index.save_index, day05["save_index"]):
            assert _outcome(lambda: save(bad, tmp_path / "shared.npz"))[0] == "TypeError"
            assert len(index.load_index(tmp_path / "shared.npz", model, revision)) == 44
            assert not list(tmp_path.glob("*.tmp"))

    def test_index_sync_matches_day_05_notebook(self, day05: dict, embedder: embeddings.Embedder,
                                                tmp_path: Path) -> None:
        saved = tmp_path / "index.npz"
        index.save_index(_shared_index(day05), saved)
        model, revision = day05["index"].model, day05["index"].revision

        def unchanged(sample: Path) -> None:
            pass

        def retire_most(sample: Path) -> None:   # deletes more than half of the chunks: the guard must refuse
            for path in (sample / "acme-handbook").glob("*.md"):
                path.unlink()

        for change in (day05["day02_changes"], day05["restrict_travel_policy"], unchanged, retire_most):
            current = corpus.corpus_after(REPO_ROOT, change)
            assert _as_dicts(current) == _as_dicts(day05["corpus_after"](change))
            a, b = index.load_index(saved, model, revision), index.load_index(saved, model, revision)
            assert _outcome(lambda: index.plan_sync(a, current)) == _outcome(lambda: day05["plan_sync"](b, current))
            assert _outcome(lambda: index.sync(a, current, embedder.embed)) == \
                   _outcome(lambda: day05["sync"](b, current, embedder.embed))
            assert a.records == b.records and a.row == b.row and np.array_equal(a.vectors, b.vectors)
        # The guard can be lifted; an empty list then deletes everything in both versions
        a, b = index.load_index(saved, model, revision), index.load_index(saved, model, revision)
        assert index.sync(a, [], embedder.embed, max_delete_fraction=1.0) == \
               day05["sync"](b, [], embedder.embed, max_delete_fraction=1.0)
        assert len(a) == len(b) == 0 and a.vectors.shape == b.vectors.shape == (0, 384)
        # Duplicate IDs, a changed text passed as a metadata update, vectors of the wrong shape: same errors
        a, b = index.load_index(saved, model, revision), index.load_index(saved, model, revision)
        chunk = a.records[0]
        edited = dataclasses.replace(chunk, chunk_hash="0" * 64)
        assert _outcome(lambda: index.plan_sync(a, [chunk, chunk])) == \
               _outcome(lambda: day05["plan_sync"](b, [chunk, chunk]))
        assert _outcome(lambda: a.update_metadata(edited)) == _outcome(lambda: b.update_metadata(edited))
        assert _outcome(lambda: index.VectorIndex(model, revision, 384).upsert([chunk], np.zeros((1, 3)))) == \
               _outcome(lambda: day05["VectorIndex"](model, revision, 384).upsert([chunk], np.zeros((1, 3))))
        assert a.delete(["no-such-chunk", chunk.chunk_id]) == b.delete(["no-such-chunk", chunk.chunk_id]) == 1


# ----------------------------------------------------------------------------- Day 06: grounded answering
EXTRA_QUESTIONS = [
    "When do pay rises take effect?", "Who approves pay changes?", "How do I connect to the VPN?",
    "What is the hotel limit per night in Tokyo?", "Will Acme pay for wine with a business dinner?",
    "How much annual leave do I get?", "Do I need the VPN to check my email?", "USD 220", "privacy@acme.example",
    "", "What is the IT policy for BYOD and USB-C sticks at Acme?",
]
USERS = ["alice", "maria", "mallory", ""]


class _ForbiddenStore:
    """A store that ignores the permission filter: answer() must refuse to send what it returns."""

    def __init__(self, real):
        self.real = real

    def search(self, query_vector, k=3, where=None, filter_mode="pre"):
        return self.real.search(query_vector, k)        # no `where`: the managers-only guide can come back


@pytest.fixture(scope="class")
def day06():
    # Runs the whole Day 06 notebook with the live model off (it reads or builds Day 05's index). The module's
    # Pipeline and reader use the notebook's own index and embedder, so this class compares the answering code
    # alone, with one model in memory (the embedder is compared with Day 04's notebook above).
    ns = run_notebook_code("day_06_grounded_rag.ipynb")
    ns["embedder"] = CachedEmbedder(ns["embedder"])    # the notebook's functions read this global on every call
    yield ns
    release(ns)


class CachedEmbedder:
    """The notebook's own embedder behind a cache keyed on the exact input, so the module and the notebook
    get the very same vectors for the same texts and each list of sentences is embedded once, not 16 times.
    What these tests compare is the answering code; the model itself is compared with Day 04's notebook."""

    def __init__(self, inner: embeddings.Embedder):
        self.inner, self.cache = inner, {}

    def embed(self, texts, batch_size: int = 16):
        key = ("embed", tuple(texts), batch_size)
        if key not in self.cache:
            self.cache[key] = self.inner.embed(texts, batch_size)
        return self.cache[key]

    def embed_query(self, question: str, instruction: str = embeddings.QUERY_INSTRUCTION):
        key = ("query", question, instruction)
        if key not in self.cache:
            self.cache[key] = self.inner.embed_query(question, instruction)
        return self.cache[key]


class TestDay06Answering:
    @staticmethod
    def questions(day06: dict) -> list[str]:
        return list(dict.fromkeys([q for _, q, _ in day06["CHECK_LIST"]] + EXTRA_QUESTIONS))

    def test_answering_settings_match_day_06_notebook(self, day06: dict) -> None:
        for name in ("TOP_K", "MIN_RETRIEVAL_SCORE", "MAX_CONTEXT_WORDS", "ABSTAIN", "BLOCKED", "UNAVAILABLE",
                     "DIRECTORY", "MIN_SENTENCE_SCORE", "SENTENCE_MARGIN", "MAX_SENTENCES"):
            assert getattr(answering, name) == day06[name], name
        for name in ("SENTENCE", "LIST_ITEM", "NAME", "CITATION", "NUMBER"):
            assert getattr(answering, name).pattern == day06[name].pattern, name
        assert answering.ExtractiveReader.name == day06["ExtractiveReader"].name
        assert answering.FixedReply.name == day06["FixedReply"].name
        assert [f.name for f in dataclasses.fields(answering.RagAnswer)] == \
               [f.name for f in dataclasses.fields(day06["RagAnswer"])]

    def test_retrieve_matches_day_06_notebook(self, day06: dict, monkeypatch) -> None:
        rag = answering.Pipeline(day06["index"], day06["embedder"])
        for question in self.questions(day06) + [p.question for p in DAY03_PROBES]:
            for user in USERS:
                for k in (1, 3, 10, 60):
                    for store in (None, day06["poisoned"]):
                        assert _results(rag.retrieve(user, question, k, store)) == \
                               _results(day06["retrieve"](user, question, k, store)), (user, question, k)
        # Another directory: the module takes it as a setting, the notebook reads its global on every call
        other = {"alice": {"managers"}, "bob": {"people-team"}}
        monkeypatch.setitem(day06, "DIRECTORY", other)
        custom = answering.Pipeline(day06["index"], day06["embedder"], directory=other)
        for question in self.questions(day06)[:8]:
            for user in ("alice", "bob", "maria"):
                assert _results(custom.retrieve(user, question)) == _results(day06["retrieve"](user, question))

    def test_assemble_matches_day_06_notebook(self, day06: dict) -> None:
        for question in self.questions(day06):
            for user in ("alice", "maria"):
                results = day06["retrieve"](user, question, k=10)
                for max_words in (0, 30, 120, 300, 1000):
                    assert asdict(answering.assemble(question, results, max_words)) == \
                           asdict(day06["assemble"](question, results, max_words))
                assert asdict(answering.assemble(question, results)) == asdict(day06["assemble"](question, results))
        # A repeated text under another chunk id is dropped by both (section 9's copy)
        results = day06["retrieve"]("alice", day06["DADS"]) + [day06["copy"]]
        assert asdict(answering.assemble(day06["DADS"], results)) == asdict(day06["assemble"](day06["DADS"], results))
        assert answering.count_words(" a  b\nc ") == day06["count_words"](" a  b\nc ") == 3

    def test_reader_matches_day_06_notebook(self, day06: dict, monkeypatch) -> None:
        embedder = day06["embedder"]
        nb_source, Source = day06["Source"], answering.Source
        for chunk in day06["index"].records:     # every sentence, list item and table row of every chunk
            fields = (chunk.chunk_id, chunk.title, chunk.section, chunk.last_reviewed, chunk.text, 0.5,
                      chunk.source_path)
            assert answering.answer_units(Source(*fields)) == day06["answer_units"](nb_source(*fields))
        contexts = []
        for question in self.questions(day06):
            for user in ("alice", "maria"):
                results = day06["retrieve"](user, question)
                contexts.append((answering.assemble(question, results), day06["assemble"](question, results)))
        contexts.append((answering.assemble("anything", []), day06["assemble"]("anything", [])))   # no sources
        for question in ("What is Google's plan?", "Is Wi-Fi on?", "CEO? Who", "Ask the IT Service Desk",
                         "How does Acme handle BYOD at the Security Team?", "", "lowercase only"):
            module_ctx, nb_ctx = contexts[0]
            assert answering.unknown_names(question, module_ctx.sources) == \
                   day06["unknown_names"](question, nb_ctx.sources)
        # The reader with Day 06's settings and with others: the notebook reads its globals on every call
        for min_score, margin, max_sentences in [(0.60, 0.03, 3), (0.50, 0.0, 1), (0.65, 0.10, 3), (0.70, 0.03, 2)]:
            monkeypatch.setitem(day06, "MIN_SENTENCE_SCORE", min_score)
            monkeypatch.setitem(day06, "SENTENCE_MARGIN", margin)
            monkeypatch.setitem(day06, "MAX_SENTENCES", max_sentences)
            module_reader = answering.ExtractiveReader(embedder, min_score, margin, max_sentences)
            nb_reader = day06["ExtractiveReader"](embedder)
            for module_ctx, nb_ctx in contexts:
                if module_ctx.sources:
                    assert [(s, src.id, u) for s, src, u in module_reader.scored_units(module_ctx)] == \
                           [(s, src.id, u) for s, src, u in nb_reader.scored_units(nb_ctx)]
                assert asdict(module_reader.generate(module_ctx)) == asdict(nb_reader.generate(nb_ctx))
        # On the boundary: a threshold exactly equal to a question's best score (the notebook answers, "<" not "<=")
        for module_ctx, nb_ctx in contexts[:6]:
            best = answering.ExtractiveReader(embedder).scored_units(module_ctx)[0][0]
            monkeypatch.setitem(day06, "MIN_SENTENCE_SCORE", best)
            on_the_line = answering.ExtractiveReader(embedder, min_score=best).generate(module_ctx)
            assert asdict(on_the_line) == asdict(day06["ExtractiveReader"](embedder).generate(nb_ctx))

    def test_citation_check_matches_day_06_notebook(self, day06: dict) -> None:
        nb_context = day06["context"]          # alice's new-dads context from the notebook
        module_context = answering.Context(nb_context.question,
                                           tuple(answering.Source(**asdict(s)) for s in nb_context.sources),
                                           nb_context.dropped)
        entitlement, eligible = "parental-leave-policy#leave-entitlement", "parental-leave-policy#who-is-eligible"
        texts = [*day06["HAND_WRITTEN"].values(), day06["reply"].text, day06["ABSTAIN"],
                 "I don\u2019t know based on the provided documents.", "I DON'T KNOW based on the provided documents",
                 f"Leave is 16 weeks after six months of service [{entitlement}, {eligible}].",
                 f"- Primary caregivers: 16 weeks [{entitlement}]\n- Secondary caregivers: 6 weeks [{entitlement}]",
                 "Leave is 16 weeks [1].", "I don't know based on the provided documents. But maybe 20 weeks.",
                 "", "   ", f"Primary caregivers get 16 weeks of leave. [{entitlement}]",
                 "Leave is 16 weeks. [parental-leave-policy#leave-entitlement] Done [x].",
                 f"It is 16,000.5 or 12:30 [{entitlement}; {eligible}]! Really? [{eligible}]",
                 f"[{entitlement}] [{eligible}]", f"Six months [ {eligible} ].\n\n• 6 weeks [{entitlement}]"]
        for text in texts:
            assert answering.split_claims(text) == day06["split_claims"](text), text
            module_check, nb_check = answering.check_citations(text, module_context), \
                day06["check_citations"](text, nb_context)
            assert asdict(module_check) == asdict(nb_check), text
            assert module_check.passed == nb_check.passed and module_check.problems == nb_check.problems, text

    def test_answer_matches_day_06_notebook(self, day06: dict, monkeypatch) -> None:
        embedder, nb_answer = day06["embedder"], day06["answer"]
        facts = {(u, q): f for u, q, f in day06["CHECK_LIST"]}

        def same(module_answer, nb_answer_) -> None:
            assert asdict(module_answer) == asdict(nb_answer_)
            assert [s.id for s in answering.cited_sources(module_answer)] == \
                   [s.id for s in day06["cited_sources"](nb_answer_)]
            key = (module_answer.user, module_answer.question)
            if key in facts:
                assert answering.grade(module_answer, facts[key]) == day06["grade"](nb_answer_, facts[key])

        rag = answering.Pipeline(day06["index"], embedder)
        module_reader, nb_reader = answering.ExtractiveReader(embedder), day06["ExtractiveReader"](embedder)
        for question in self.questions(day06):
            for user in USERS:
                same(rag.answer(user, question, module_reader), nb_answer(user, question, nb_reader))
        locked = day06["LOCKED"]       # the poisoned index from section 4.1
        same(rag.answer("alice", locked, module_reader, from_index=day06["poisoned"]),
             nb_answer("alice", locked, nb_reader, from_index=day06["poisoned"]))
        # Hand-written replies, including ones the checks block, and a generator that fails
        for text in day06["HAND_WRITTEN"].values():
            same(rag.answer("alice", day06["DADS"], answering.FixedReply(text)),
                 nb_answer("alice", day06["DADS"], day06["FixedReply"](text)))

        def failing(error_class):
            class Failing:
                name = "failing"

                def generate(self, context):
                    raise error_class("could not reach the model (simulated)")
            return Failing()

        same(rag.answer("alice", day06["DADS"], failing(answering.GeneratorError)),
             nb_answer("alice", day06["DADS"], failing(day06["GeneratorError"])))
        # Another gate: the module takes it as a setting, the notebook reads its global on every call
        for gate in (0.0, 0.62, 0.75):
            monkeypatch.setitem(day06, "MIN_RETRIEVAL_SCORE", gate)
            gated = answering.Pipeline(day06["index"], embedder, min_retrieval_score=gate)
            for question in self.questions(day06)[:12]:
                same(gated.answer("alice", question, module_reader), nb_answer("alice", question, nb_reader))
        # On the boundary: a gate exactly equal to the best retrieval score lets the question through in both
        for question in self.questions(day06)[:6]:
            results = day06["retrieve"]("alice", question)
            if results:
                monkeypatch.setitem(day06, "MIN_RETRIEVAL_SCORE", results[0].score)
                on_the_line = answering.Pipeline(day06["index"], embedder, min_retrieval_score=results[0].score)
                same(on_the_line.answer("alice", question, module_reader), nb_answer("alice", question, nb_reader))
        # A store that skips the permission filter: both refuse before anything is sent
        leaky = _ForbiddenStore(day06["index"])
        pay = "When do pay rises take effect?"
        assert _outcome(lambda: rag.answer("alice", pay, module_reader, from_index=leaky)) == \
               _outcome(lambda: nb_answer("alice", pay, nb_reader, from_index=leaky))
        assert _outcome(lambda: rag.answer("alice", pay, module_reader, from_index=leaky))[0] == "PermissionError"

    def test_grade_matches_day_06_notebook(self, day06: dict) -> None:
        # The check list's facts happen to match the answers' case exactly, so test the grade on made-up answers too
        cases = [("answered", "Primary caregivers get SIXTEEN weeks [x].", ("sixteen weeks",)),
                 ("answered", "16 Weeks for primary caregivers.", ("16 weeks", "6 weeks")),
                 ("answered", "Approved by IT security and your Manager.", ("IT Security", "manager")),
                 ("abstained", day06["ABSTAIN"], None), ("blocked", day06["BLOCKED"], None),
                 ("answered", "Anything at all.", None), ("unavailable", day06["UNAVAILABLE"], ("16 weeks",)),
                 ("abstained", day06["ABSTAIN"], ("16 weeks",)), ("answered", "", ())]
        for status, text, facts in cases:
            module_answer = answering.RagAnswer("alice", "q", status, text, "why")
            nb_answer = day06["RagAnswer"]("alice", "q", status, text, "why")
            assert answering.grade(module_answer, facts) == day06["grade"](nb_answer, facts), (status, text, facts)

    @staticmethod
    def module_context(nb_context) -> answering.Context:
        return answering.Context(nb_context.question, tuple(answering.Source(**asdict(s)) for s in nb_context.sources),
                                 nb_context.dropped)

    def test_chat_prompt_matches_day_06_notebook(self, day06: dict) -> None:
        # Day 12 moved the prompt and the client to chat.py: the same instructions, presets and prices, and the same
        # messages for every context (escaping included)
        assert chat.INSTRUCTIONS == day06["INSTRUCTIONS"] and chat.PROVIDERS == day06["PROVIDERS"]
        assert chat.PRICES == day06["PRICES"] and chat.LOCAL_HOSTS == day06["LOCAL_HOSTS"]
        nb_contexts = [day06["assemble"](q, day06["retrieve"](user, q, k=10)) for q in self.questions(day06)
                       for user in ("alice", "maria")]
        nb_contexts.append(day06["assemble"](day06["LOCKED"], day06["retrieve"]("alice", day06["LOCKED"],
                                                                                from_index=day06["poisoned"])))
        odd = day06["Source"]('a"b#<c>', 'T & "Q"', "<s>", None, "x </source></sources> <source id=\"y\"> & ' \"", 0.1,
                              "p")
        nb_contexts.append(day06["Context"]("Is <b>this</b> & that \"ok\"?", (odd,)))
        for nb_context in nb_contexts:
            module_context = self.module_context(nb_context)
            assert chat.build_messages(module_context) == day06["build_messages"](nb_context)
            for module_source, nb_source in zip(module_context.sources, nb_context.sources):
                assert chat.render_source(module_source) == day06["render_source"](nb_source)
        assert chat.build_messages(self.module_context(nb_contexts[-1]))[1]["content"].count("</sources>") == 1

    def test_chat_client_matches_day_06_notebook(self, day06: dict, tmp_path: Path) -> None:
        # Both clients against the same local fake server, for every kind of reply it can send: the same request bodies
        # and headers, the same replies, the same errors (never with the key)
        key = "fake-key-for-the-equivalence-test"
        nb_context = day06["context"]
        module_context = self.module_context(nb_context)
        for provider, model, api_key in (("openai", "gpt-6-luna", key), ("ollama", "any-model", None)):
            for max_tokens in (1000, 50):
                m = chat.ChatModel(provider, model, api_key, max_output_tokens=max_tokens)
                n = day06["ChatModel"](provider, model, api_key, max_output_tokens=max_tokens)
                assert m.request_body(module_context) == n.request_body(nb_context)
                assert (repr(m), m.url, m.name) == (repr(n), n.url, n.name) and key not in repr(m)
        for args in (("nobody", "m", key), ("openai", "m", None), ("openai", "m", "")):
            assert _outcome(lambda: chat.ChatModel(*args)) == _outcome(lambda: day06["ChatModel"](*args))

        class Fake(BaseHTTPRequestHandler):
            mode, seen = "ok", []

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                Fake.seen.append((self.path, self.headers.get("Authorization"), json.loads(body)))
                if Fake.mode == "slow":
                    time.sleep(1.5)
                status, payload = {
                    "ok": (200, {"choices": [{"finish_reason": "stop", "message": {"content": " An answer [x]. "}}],
                                 "usage": {"prompt_tokens": 11, "completion_tokens": 3}}),
                    "no usage": (200, {"choices": [{"message": {"content": "Hi"}}]}),
                    "echo": (500, {"error": {"message": f"boom {self.headers.get('Authorization')}"}}),
                    "denied": (401, {"error": {"message": f"Incorrect API key {self.headers.get('Authorization')}"}}),
                    "limits": (429, {"error": {"message": "slow down"}}),
                    "no choices": (200, {"choices": []}),
                    "empty": (200, {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}),
                    "slow": (200, {"choices": [{"message": {"content": "late"}}]}),
                }.get(Fake.mode, (200, None))
                data = b"<html>not json</html>" if payload is None else json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = QuietServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        try:
            for mode in ("ok", "no usage", "echo", "denied", "limits", "no choices", "empty", "not json", "slow"):
                Fake.mode, Fake.seen = mode, []
                results = []
                for client in (chat.ChatModel("openai", "gpt-6-luna", key, url, timeout=1),
                               day06["ChatModel"]("openai", "gpt-6-luna", key, url, timeout=1)):
                    try:
                        results.append(("ok", asdict(client.generate(module_context if isinstance(client, chat.ChatModel)
                                                                     else nb_context))))
                    except Exception as error:
                        results.append((type(error).__name__, str(error)))
                assert results[0] == results[1], mode
                assert key not in str(results[0]), mode
                assert Fake.seen[0] == Fake.seen[1], mode
        finally:
            server.shutdown()
            server.server_close()
        # Nothing listens on port 9, so both clients are refused. Windows takes about 2 s to report a refused local
        # connection (it retries first), so a 2 s timeout raced the refusal and each client could get either error
        # (CI run 43446003140, windows-latest / Python 3.14). 10 s leaves the refusal to arrive first everywhere.
        unreachable = [_outcome(lambda: c.generate(ctx)) for c, ctx in (
            (chat.ChatModel("ollama", "m", base_url="http://127.0.0.1:9/v1", timeout=10), module_context),
            (day06["ChatModel"]("ollama", "m", base_url="http://127.0.0.1:9/v1", timeout=10), nb_context))]
        assert unreachable[0] == unreachable[1] and unreachable[0][0] == "GeneratorError"
        # .env files: the same values, and the same client (or none) built from them
        for i, text in enumerate(["", "# only a comment\n", "RAG_CHAT_PROVIDER=ollama\nRAG_CHAT_MODEL=llama\n",
                                  f"RAG_CHAT_PROVIDER = openai\nRAG_CHAT_MODEL='gpt-6-luna'\nRAG_CHAT_API_KEY=\"{key}\"\n",
                                  "RAG_CHAT_PROVIDER=openai # the preset\nRAG_CHAT_MODEL=m\nRAG_CHAT_API_KEY=k # x\n"
                                  "RAG_CHAT_BASE_URL=http://localhost:8000/v1/\nnot a line\n=x\n",
                                  "RAG_CHAT_PROVIDER=openai\nRAG_CHAT_MODEL=m\n"]):
            env = tmp_path / f"env-{i}"
            env.write_text(text, encoding="utf-8")
            assert chat.read_env_file(env) == day06["read_env_file"](env), text
            m, n = _outcome(lambda: chat.chat_model_from_env(env)), _outcome(lambda: day06["chat_model_from_env"](env))
            assert (m[0], repr(m[1]) if m[0] == "ok" else m[1]) == (n[0], repr(n[1]) if n[0] == "ok" else n[1]), text
        assert chat.read_env_file(tmp_path / "missing") == {} == day06["read_env_file"](tmp_path / "missing")


# ----------------------------------------------------------------------------- Day 07: the evaluation harness
@pytest.fixture(scope="class")
def day07():
    # Runs the whole Day 07 notebook (labels, both retrievers, thresholds, the baseline comparison). The module's
    # functions are then compared with the notebook's own on the notebook's own rankings and labels.
    ns = run_notebook_code("day_07_evaluation_baseline.ipynb")
    yield ns
    release(ns)


class TestDay07Evaluation:
    EVIDENCE_CASES = [  # (evidence, text): whole words and numbers only, never part of a longer number
        (("USD 25",), "up to USD 250 per night"), (("USD 25",), "up to USD 25 per night"),
        (("USD 25",), "USD 25,000 a year"), (("USD 25",), "USD 25.50"), (("USD 25",), "USD 25."),
        (("14 days",), "within 14 days."), (("1 April",), "from 1 April 2027"), (("a", "b"), "a b"),
        (("a", "b"), "a c"), ((), "anything"), (("x",), ""), (("C++",), "C++ code"), (("Wi-Fi",), "the Wi-Fi."),
        (("14 days",), "within 114 days"), (("USD 25",), "XUSD 25"), (("Wi-Fi",), "noWi-Fi here")]

    def test_evaluation_settings_match_day_07_notebook(self, day07: dict) -> None:
        for name in ("TOP_K", "MAX_K", "K_VALUES", "TIE"):
            assert getattr(evaluation, name) == day07[name], name
        assert [f.name for f in dataclasses.fields(evaluation.Query)] == \
               [f.name for f in dataclasses.fields(day07["Query"])]

    def test_labels_and_validation_match_day_07_notebook(self, day07: dict, tmp_path: Path) -> None:
        queries = evaluation.load_queries(day07["QUESTIONS_FILE"])
        assert [asdict(q) for q in queries] == [asdict(q) for q in day07["queries"]] and len(queries) == 81
        assert [q.answerable for q in queries] == [q.answerable for q in day07["queries"]]
        # A file with a blank line and a trailing newline: both loaders skip it the same way
        rows = day07["QUESTIONS_FILE"].read_text(encoding="utf-8").splitlines()
        odd = tmp_path / "odd.jsonl"
        odd.write_text(rows[0] + "\n\n" + rows[1] + "\n", encoding="utf-8")
        assert [asdict(q) for q in evaluation.load_queries(odd)] == [asdict(q) for q in day07["load_queries"](odd)]
        # The evidence rule, on every query and chunk and on made-up edge cases
        chunks = day07["chunks"]
        for q in day07["queries"]:
            for c in chunks:
                assert evaluation.holds(q.evidence, c.text) == day07["holds"](q.evidence, c.text), (q.id, c.chunk_id)
        for evidence, text in self.EVIDENCE_CASES:
            assert evaluation.holds(evidence, text) == day07["holds"](evidence, text), (evidence, text)
        snap = evaluation.snapshot(chunks)
        assert snap == day07["snapshot"](chunks) and evaluation.fingerprint(snap) == day07["fingerprint"](snap)
        # validate(): the real labels, the two-edit demo, broken queries, another directory
        manifest, nb_queries = day07["manifest"], day07["queries"]
        assert evaluation.validate(queries, chunks, manifest) == day07["validate"](nb_queries, chunks, manifest) == []
        stale = corpus.corpus_after(REPO_ROOT, day07["update_documents"])
        assert evaluation.validate(queries, stale, manifest) == day07["stale"] and len(day07["stale"]) == 17
        first = nb_queries[0]
        broken = [dataclasses.replace(first, user="mallory"), dataclasses.replace(first, kind="essay"),
                  dataclasses.replace(first, answer_chunks=()), dataclasses.replace(first, facts=()),
                  dataclasses.replace(first, evidence=("no such words anywhere",)),
                  dataclasses.replace(nb_queries[1], id=first.id), dataclasses.replace(first, restricted_chunks=("x",))]
        for q in broken:
            module_q = evaluation.Query(**asdict(q))
            assert evaluation.validate([module_q] + queries[1:], chunks, manifest) == \
                   day07["validate"]([q] + nb_queries[1:], chunks, manifest), q
        other = {"alice": {"all-employees", "managers"}, "maria": {"all-employees"}}
        assert evaluation.validate(queries, chunks, manifest, other) == \
               day07["validate"](nb_queries, chunks, manifest, other)
        assert evaluation.validate(queries, chunks[:-1], manifest) == day07["validate"](nb_queries, chunks[:-1], manifest)

    def test_metrics_match_day_07_notebook(self, day07: dict) -> None:
        rankings, queries = day07["rankings"], day07["queries"]
        for name in day07["RETRIEVERS"]:
            for q in queries:
                ranked = rankings[name][q.id]
                for relevant in (q.answer_chunks, q.restricted_chunks, (ranked[0][0],) if ranked else ()):
                    assert evaluation.first_relevant(ranked, relevant) == day07["first_relevant"](ranked, relevant)
                    for k in range(0, 13):
                        assert evaluation.hit(ranked, relevant, k) == day07["hit"](ranked, relevant, k)
                        assert evaluation.reciprocal_rank(ranked, relevant, k) == \
                               day07["reciprocal_rank"](ranked, relevant, k)
                        if relevant:
                            assert evaluation.recall(ranked, relevant, k) == day07["recall"](ranked, relevant, k)
                    assert evaluation.reciprocal_rank(ranked, relevant) == day07["reciprocal_rank"](ranked, relevant)
            # measure(): the module takes the ranking, the notebook reads its global by name
            selections = list(day07["SELECTIONS"].values()) + [
                [q for q in day07["SELECTIONS"]["questions"] if q.split == split] for split in ("dev", "test")]
            for selection in selections:
                assert evaluation.measure(rankings[name], selection) == day07["measure"](name, selection)
        for n in range(1, 40):
            for s in range(0, n + 1):
                assert evaluation.wilson(s, n) == day07["wilson"](s, n), (s, n)
        assert evaluation.wilson(29, 31, z=2.58) == day07["wilson"](29, 31, z=2.58)
        for a in range(0, 16):
            for b in range(0, 16):
                assert evaluation.sign_test(a, b) == day07["sign_test"](a, b), (a, b)

    def test_outcome_matches_day_07_notebook(self, day07: dict) -> None:
        # Every graded answer of the notebook's end-to-end table, under both settings
        for label, answers in day07["answers"].items():
            for q in day07["natural"]:
                assert evaluation.outcome(answers[q.id], q) == day07["outcome"](answers[q.id], q) == \
                       day07["outcomes"][label][q.id], (label, q.id)
        # Made-up answers: every status, an answerable and an unanswerable query, facts in another case or missing
        answerable = next(q for q in day07["natural"] if q.answerable and len(q.facts) == 1    # a fact with letters,
                          and q.facts[0].lower() != q.facts[0].upper())                     # so its case can differ
        unanswerable = next(q for q in day07["natural"] if not q.answerable)
        fact = answerable.facts[0]
        for status in ("answered", "abstained", "blocked", "unavailable"):
            for text in (fact, fact.upper(), f"It is {fact.lower()}.", "something else", ""):
                a = answering.RagAnswer("alice", "q", status, text, "why")
                for q in (answerable, unanswerable, dataclasses.replace(answerable, facts=(fact, "a second fact"))):
                    assert evaluation.outcome(a, q) == day07["outcome"](a, q), (status, text, q.id)

    def test_comparison_matches_day_07_notebook(self, day07: dict) -> None:
        rankings, queries = day07["rankings"], day07["queries"]
        for name in day07["RETRIEVERS"]:
            for q in queries:
                ranked = rankings[name][q.id]
                assert evaluation.top_score(ranked) == day07["top_score"](name, q.id)
                if q.answerable:
                    assert evaluation.tie_margin(ranked, q.answer_chunks) == day07["tie_margin"](name, q)
        assert evaluation.top_score([]) is None and evaluation.tie_margin([], ("x",)) is None
        assert evaluation.tie_margin([("a", 0.5)], ("a",)) is None     # one result: no neighbour
        accepted = json.loads(day07["BASELINE_FILE"].read_text(encoding="utf-8"))
        run = day07["baseline"]

        def edited(change) -> dict:
            copy = json.loads(json.dumps(run))
            change(copy)
            return copy

        def move(qid, name, rank):
            return lambda b: b["queries"][qid][name].update(first_relevant=rank)

        def score(qid, name, delta):
            return lambda b: b["queries"][qid][name].update(top_score=round(b["queries"][qid][name]["top_score"] + delta, 3))

        variants = [run, accepted,
                    edited(move("d03-13", "dense", 5)),                     # a near tie in the baseline: listed apart
                    edited(move("d03-04", "BM25", 1)),                      # a real change
                    edited(lambda b: [move("d03-13", "dense", 5)(b), b["metrics"]["questions, dense"].update(n=0)]),
                    edited(score("d03-01", "dense", 0.004)), edited(score("d03-01", "dense", 0.006)),
                    edited(lambda b: b["queries"]["d03-01"]["BM25"].update(top_score=None)),
                    edited(lambda b: b["queries"]["d03-01"].update(hybrid=b["queries"]["d03-01"]["BM25"])),
                    edited(lambda b: b["retrievers"]["BM25"].update(k1=1.5)),
                    edited(lambda b: b["dataset"].update(version=2)),
                    edited(lambda b: b["thresholds"]["measured"].update(min_retrieval_score=0.6)),
                    edited(lambda b: b["metrics"].update({"questions, BM25": {}}))]
        for new in variants:
            for old in (accepted, run, variants[3]):
                for tolerance in (0.005, 0.0001):
                    assert evaluation.compare(new, old, tolerance) == day07["compare"](new, old, tolerance)
        # On the boundaries: a score change exactly equal to the tolerance, a near tie exactly at TIE, a margin above it
        changed = edited(score("d03-01", "dense", 0.004))
        diff = abs(changed["queries"]["d03-01"]["dense"]["top_score"] - run["queries"]["d03-01"]["dense"]["top_score"])
        for tolerance in (diff, diff * 0.999, diff * 1.001):
            assert evaluation.compare(changed, run, tolerance) == day07["compare"](changed, run, tolerance)
        assert evaluation.compare(changed, run, diff) == ([], [])
        moved = edited(move("d01-01", "dense", 2))                  # its recorded margin is 0.0073: a real change
        for old in (run, edited(lambda b: b["queries"]["d01-01"]["dense"].update(margin=evaluation.TIE)),
                    edited(lambda b: b["queries"]["d01-01"]["dense"].update(margin=evaluation.TIE / 2))):
            assert evaluation.compare(moved, old) == day07["compare"](moved, old)
        assert evaluation.compare(moved, run)[0] and not evaluation.compare(moved, run)[1]
        assert evaluation.compare(run, accepted) == ([], [])
        assert evaluation.compare(variants[2], accepted)[1] and not evaluation.compare(variants[2], accepted)[0]


# ----------------------------------------------------------------------------- Day 08: rank fusion
@pytest.fixture(scope="class")
def day08():
    # Runs the whole Day 08 notebook (both retrievers, both fusions, the baseline comparison). The module's functions
    # are then compared with the notebook's own on the notebook's own rankings, and on made-up edge cases.
    ns = run_notebook_code("day_08_hybrid_search_and_fusion.ipynb")
    yield ns
    release(ns)


class TestDay08Fusion:
    EDGE_LISTS = [  # rankings no real query produces: empty, one item, equal scores, negative scores, a repeated id
        [], [("a", 1.0)], [("a", 2.0), ("b", 2.0), ("c", 2.0)], [("a", -1.0), ("b", -3.5)],
        [("a", 3.0), ("b", 1.0), ("a", 0.5)], [("b", 0.9), ("a", 0.9), ("c", 0.1)]]

    def test_fusion_settings_match_day_08_notebook(self, day08: dict) -> None:
        assert fusion.RRF_K == day08["RRF_K"] == 60 and fusion.ALPHA == day08["ALPHA"] == 0.7

    def test_rrf_matches_day_08_notebook(self, day08: dict) -> None:
        rankings = day08["rankings"]
        for q in day08["queries"]:
            lists = [rankings["BM25"][q.id], rankings["dense"][q.id]]
            assert fusion.rrf(lists) == day08["rrf"](lists) == rankings["RRF"][q.id], q.id
            for k in (1, 5, 20, 100, 1000):
                assert fusion.rrf(lists, k) == day08["rrf"](lists, k), (q.id, k)
            for window in (1, 5, 10):
                cut = [ranked[:window] for ranked in lists]
                assert fusion.rrf(cut) == day08["rrf"](cut), (q.id, window)
            assert fusion.rrf(lists[::-1]) == day08["rrf"](lists[::-1]) and fusion.rrf(lists[:1]) == day08["rrf"](lists[:1])
        for a in self.EDGE_LISTS:
            for b in self.EDGE_LISTS:
                assert fusion.rrf([a, b]) == day08["rrf"]([a, b]), (a, b)
        assert fusion.rrf([]) == day08["rrf"]([]) == []

    def test_min_max_and_weighted_match_day_08_notebook(self, day08: dict) -> None:
        rankings = day08["rankings"]
        for q in day08["queries"]:
            lexical, dense = rankings["BM25"][q.id], rankings["dense"][q.id]
            assert fusion.min_max(lexical) == day08["min_max"](lexical), q.id
            assert fusion.min_max(dense) == day08["min_max"](dense), q.id
            assert fusion.weighted(lexical, dense, fusion.ALPHA) == rankings["weighted"][q.id], q.id
            for alpha in [i / 10 for i in range(11)] + [0.25, 0.75]:
                assert fusion.weighted(lexical, dense, alpha) == day08["weighted"](lexical, dense, alpha), (q.id, alpha)
            assert fusion.weighted(dense, lexical, 0.3) == day08["weighted"](dense, lexical, 0.3), q.id
            assert fusion.weighted([], dense, 0.7) == day08["weighted"]([], dense, 0.7), q.id
        for a in self.EDGE_LISTS:
            assert fusion.min_max(a) == day08["min_max"](a), a
            for b in self.EDGE_LISTS:
                for alpha in (0.0, 0.7, 1.0):
                    assert fusion.weighted(a, b, alpha) == day08["weighted"](a, b, alpha), (a, b, alpha)

    def test_rrf_margin_matches_day_08_notebook(self, day08: dict) -> None:
        rankings = day08["rankings"]
        for q in day08["queries"]:
            lists = [rankings["BM25"][q.id], rankings["dense"][q.id]]
            assert fusion.rrf_margin(lists, q.answer_chunks) == day08["rrf_margin"](q), q.id
        # The notebook's recorded margins, as accepted in its baseline file
        accepted = json.loads((REPO_ROOT / "evals" / "baselines" / "day-08-retrieval.json").read_text(encoding="utf-8"))
        for q in day08["queries"]:
            lists = [rankings["BM25"][q.id], rankings["dense"][q.id]]
            assert fusion.rrf_margin(lists, q.answer_chunks) == accepted["queries"][q.id]["RRF"]["margin"], q.id
        # Labels the notebook never had: another chunk, two chunks, a chunk no retriever returns, and no label at all
        q = day08["by_qid"]["d03-13"]
        lists = [rankings["BM25"][q.id], rankings["dense"][q.id]]
        for relevant in (("travel-policy#safety-while-travelling",), (lists[1][5][0], lists[1][9][0]), ("no-such#chunk",)):
            fake = dataclasses.replace(q, answer_chunks=relevant)
            assert fusion.rrf_margin(lists, relevant) == day08["rrf_margin"](fake), relevant
        assert fusion.rrf_margin(lists, ()) is None


# ----------------------------------------------------------------------------- Day 09: the cross-encoder reranker
@pytest.fixture(scope="class")
def day09():
    # Runs the whole Day 09 notebook (both models, six retrievers, every depth, the gate). Its pair_scores then hold the
    # reranker's score for every query and each hybrid's top 44, so the module's rerank() can be compared without
    # calling the model again.
    ns = run_notebook_code("day_09_reranking_and_context_selection.ipynb")
    yield ns
    release(ns)


@pytest.fixture(scope="class")
def module_reranker():
    model = reranking.Reranker(REPO_ROOT)     # one per class that asks for it, released with the class
    yield model
    del model
    gc.collect()


class TestDay09Reranking:
    def test_reranker_settings_match_day_09_notebook(self, day09: dict) -> None:
        assert reranking.RERANKER == day09["RERANKER"] and reranking.MAX_TOKENS == day09["MAX_TOKENS"] == 512
        assert reranking.default_model_dir(REPO_ROOT) == day09["RERANKER_DIR"]

    def test_scores_match_day_09_notebook(self, day09: dict, module_reranker: reranking.Reranker) -> None:
        # The model card's example, and every query's top 5 from both hybrids, in batches of 16 (one batch), 5 and 1
        scores, by_cid, rankings = day09["rerank_scores"], day09["by_cid"], day09["rankings"]
        assert module_reranker.scores(day09["CARD_QUESTION"], day09["CARD_PASSAGES"]) == \
               scores(day09["CARD_QUESTION"], day09["CARD_PASSAGES"]) == list(day09["card"])
        for q in day09["queries"]:
            for base, batch_sizes in (("weighted", (16, 1)), ("RRF", (5,))):
                texts = [by_cid[cid].text for cid, _ in rankings[base][q.id][:5]]
                for batch_size in batch_sizes:
                    assert module_reranker.scores(q.question, texts, batch_size) == scores(q.question, texts, batch_size), \
                        (q.id, base, batch_size)
        # The default batch size, on more texts than fit in one batch (every 10th query's top 20)
        for q in day09["queries"][::10]:
            texts = [by_cid[cid].text for cid, _ in rankings["weighted"][q.id][:20]]
            assert module_reranker.scores(q.question, texts) == scores(q.question, texts), q.id
        # Edge texts: empty, unicode, over 512 tokens together with the question, and no texts at all
        long_text = " ".join(c.text for c in day09["chunks"][:12])
        for texts in ([""], ["Ünïcödé text — with dashes"], [long_text], [long_text, "short"]):
            assert module_reranker.scores("How long?", texts) == scores("How long?", texts), texts[0][:30]
        assert module_reranker.scores("How long?", []) == scores("How long?", []) == []
        assert module_reranker.tokenizer.truncation["max_length"] == day09["pair_tokenizer"].truncation["max_length"]

    def test_rerank_matches_day_09_notebook(self, day09: dict) -> None:
        rankings, pair_scores = day09["rankings"], day09["pair_scores"]
        for q in day09["queries"]:
            for base in ("weighted", "RRF"):
                ranked = rankings[base][q.id]
                for n in day09["DEPTHS"]:
                    top = {cid: pair_scores[q.id, cid] for cid, _ in ranked[:n]}
                    assert reranking.rerank(ranked, top, n) == day09["rerank"](q, ranked, n), (q.id, base, n)
                n = day09["DEPTH"][base]
                assert reranking.rerank(ranked, {cid: pair_scores[q.id, cid] for cid, _ in ranked[:n]}, n) == \
                       rankings[f"{base}+rerank"][q.id], (q.id, base)
        # Made-up rankings: ties in the reranker's scores (chunk id decides), n = 0, n past the end, an empty ranking
        fake = dataclasses.replace(day09["queries"][0], id="made-up")
        made_up = {"a": 1.0, "b": 1.0, "c": 2.0, "d": -1.0}
        for cid, score in made_up.items():
            pair_scores["made-up", cid] = score
        ranked = [("b", 0.9), ("a", 0.8), ("d", 0.7), ("c", 0.6)]
        for n in (0, 1, 2, 3, 4, 10):
            assert reranking.rerank(ranked, made_up, n) == day09["rerank"](fake, ranked, n), n
        assert reranking.rerank([], {}, 3) == day09["rerank"](fake, [], 3) == []

    def test_rerank_margin_matches_day_09_notebook(self, day09: dict) -> None:
        rankings, depth = day09["rankings"], day09["DEPTH"]
        accepted = json.loads((REPO_ROOT / "evals" / "baselines" / "day-09-retrieval.json").read_text(encoding="utf-8"))
        for name, base in day09["RERANKED"].items():
            for q in day09["queries"]:
                if not q.answerable:
                    continue
                fused, ranked = rankings[base][q.id], rankings[name][q.id]
                margin = reranking.rerank_margin(fused, ranked, q.answer_chunks, depth[base])
                assert margin == day09["rerank_margin"](name, q), (name, q.id)
                # The baseline was recorded on another machine. Margins are rounded to 4 decimals and the reranker's
                # scores can differ slightly between machines (CI on macOS gave 11.7955 for d07-a11 where Linux and
                # Windows give 11.7956), so the baseline may differ by one in the last digit. The copy and the
                # notebook, both computed here, must agree exactly (above).
                then = accepted["queries"][q.id][name]["margin"]
                assert (margin is None) == (then is None), (name, q.id)
                assert margin is None or round(abs(margin - then), 4) <= 0.0001, (name, q.id, margin, then)
                # Labels the notebook never had: another chunk at each of the first 25 ranks, two chunks, no such chunk
                for relevant in [(cid,) for cid, _ in ranked[:25]] + [(ranked[1][0], ranked[30][0]), ("no-such#chunk",)]:
                    fake = dataclasses.replace(q, answer_chunks=relevant)
                    assert reranking.rerank_margin(fused, ranked, relevant, depth[base]) == \
                           day09["rerank_margin"](name, fake), (name, q.id, relevant)
        assert reranking.rerank_margin([], [], ("a",), 3) is None

    def test_reranker_file_checks_match_day_09_notebook(self, day09: dict, tmp_path: Path, monkeypatch, capsys) -> None:
        # Point both versions at a scratch model folder, so the real files are never touched.
        model_dir = tmp_path / "models" / "ms-marco-MiniLM-L-6-v2" / reranking.RERANKER["revision"][:12]
        model_dir.mkdir(parents=True)
        monkeypatch.setitem(day09, "REPO_ROOT", tmp_path)
        monkeypatch.setitem(day09, "RERANKER_DIR", model_dir)
        real = (reranking.default_model_dir(REPO_ROOT) / "tokenizer.json").read_bytes()
        target = model_dir / "tokenizer.json"

        def both(name: str):
            return [_outcome(lambda: day09["reranker_file"](name)),
                    _outcome(lambda: reranking.reranker_file(name, model_dir, tmp_path))]

        # 1. A good file: same path, same printed line
        target.write_bytes(real)
        capsys.readouterr()
        notebook, module = both("tokenizer.json")
        printed = capsys.readouterr().out.splitlines()
        assert notebook == module == ("ok", target) and printed[0] == printed[1] and printed[0].endswith("verified")
        # 2. One byte changed: both refuse to load it, with the same message
        target.write_bytes(real[:-1] + bytes([real[-1] ^ 1]))
        notebook, module = both("tokenizer.json")
        assert notebook == module and notebook[0] == "ValueError" and "Never load a model file" in notebook[1]
        # 3. Missing file and a failing download (no network needed): same helpful error
        target.unlink()

        def no_network(*args, **kwargs):
            raise OSError("network unreachable (simulated by the test)")

        monkeypatch.setitem(day09, "hf_hub_download", no_network)
        monkeypatch.setattr(reranking, "hf_hub_download", no_network)
        notebook, module = both("onnx/model.onnx")
        assert notebook == module and notebook[0] == "RuntimeError" and "HF_ENDPOINT" in notebook[1]


# ----------------------------------------------------------------------------- Day 10: query analysis and the map
DAY10_CELLS = ("cell-02", "cell-03", "cell-12", "cell-14")   # setup, settings, analyse() (section 3), the map (4)


def run_notebook_cells(name: str, cell_ids, inputs: dict, definitions_of=()) -> dict:
    """Execute some of a notebook's code cells, in order, with the variables they read from earlier cells given; then,
    from the cells in `definitions_of`, only what they define (see definitions() below), in the same namespace."""
    nb = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    cells = {c.get("id"): "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
    source = "\n\n".join([cells[i] for i in cell_ids] + [definitions(cells[i]) for i in definitions_of])
    module = types.ModuleType(name.removesuffix(".ipynb"))
    module.__dict__.update(inputs)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(REPO_ROOT)
        mp.setitem(sys.modules, module.__name__, module)
        exec(compile(source, name, "exec"), module.__dict__)
    return module.__dict__


@pytest.fixture(scope="class")
def day10():
    # Only the cells that define what moved to queries.py, not the whole notebook (which loads both models and ranks
    # 122 queries with eight retrievers). Those cells read the chunks and v1's queries from section 1; they are built
    # here by the shared code that the Day 02, 03 and 07 classes check against their own notebooks.
    documents, _ = ingest(REPO_ROOT)
    chunks = [c for d in documents for c in chunking.section_chunks(d)]
    queries = evaluation.load_queries(REPO_ROOT / "evals" / "questions-v1.jsonl")
    ns = run_notebook_cells("day_10_query_understanding_and_transformations.ipynb", DAY10_CELLS,
                            {"chunks": chunks, "queries": queries, "by_qid": {q.id: q for q in queries}})
    yield ns
    release(ns)


DAY10_TEXTS = [   # made-up questions and lookups for the edges of the analysis and the map
    "", "   ", "?", "USD 1,000", "USD 1,000.50 per night", "#it-help", "privacy@acme.example", "https://it.acme.example/reset",
    "Email privacy@acme.example. Then call?", "Is New Delhi covered? And New York?", "Mexico City or Mexico?",
    "What about South Korea, Seoul and Korea?", "Tokyo hotels", "In Tokyo, what is the limit?", "PARIS or Paris?",
    "My dad works from home. Is that OK?", "How did Baghdad do?", "Can my Dad take TIME OFF?", "wfh on Fridays",
    "Will Acme pay for wine, beer and spirits?", "Does multi-factor authentication need MFA?", "Is MFA needed?",
    "I'm locked. Locked out. What now!", "Shut it down? Switch it off? Power off!", "My laptop has a virus, a Virus.",
    "Working from home vs work from home", "pay forward", "time offset", "a 14-character password?",
    "Who is the CEO? Who knows.", "Wi-Fi, VPN and IT-help", "Ünïcödé café in Zürich?", "12:30 or 10:00-15:00",
    "the limit is USD 250", "USD 250 in London", "Reset it at https://it.acme.example/reset.", "Is it https://x.example/a?",
]


class TestDay10Queries:
    def test_query_settings_match_day_10_notebook(self, day10: dict) -> None:
        assert queries.COUNTRIES == day10["COUNTRIES"] and queries.CAPITALS == day10["CAPITALS"]
        assert queries.PLACES == day10["PLACES"] and len(queries.PLACES) == 38
        for pattern in ("PLACE", "VALUE", "DEFINED"):
            assert getattr(queries, pattern).pattern == day10[pattern].pattern, pattern
        assert queries.VOCABULARY == day10["VOCABULARY"] and len(queries.VOCABULARY) == 8
        assert [f.name for f in dataclasses.fields(queries.QueryAnalysis)] == \
               [f.name for f in dataclasses.fields(day10["QueryAnalysis"])]
        assert [f.name for f in dataclasses.fields(queries.Expansion)] == [f.name for f in dataclasses.fields(day10["Expansion"])]

    def test_analysis_matches_day_10_notebook(self, day10: dict) -> None:
        # Every question of all three versions of the labelled set, and made-up edge texts
        texts = [q.question for q in evaluation.load_queries(REPO_ROOT / "evals" / "questions-v3.jsonl")] + DAY10_TEXTS
        for text in texts:
            assert asdict(queries.analyse(text)) == asdict(day10["analyse"](text)), text
            assert queries.question_names(text) == day10["question_names"](text), text
        assert queries.analyse("Is New Delhi covered?").names == (("New Delhi", "city"),)
        assert queries.analyse("USD 220").exact and not queries.analyse("USD 220?").exact

    def test_abbreviations_and_matching_match_day_10_notebook(self, day10: dict) -> None:
        assert queries.defined_abbreviations(day10["chunks"]) == day10["defined_abbreviations"](day10["chunks"])
        assert queries.VOCABULARY + queries.defined_abbreviations(day10["chunks"]) == day10["ENTRIES"]
        made_up = [types.SimpleNamespace(text=t) for t in (
            "Single sign-on (SSO) and Multi-factor authentication (MFA).", "Two words here (TW) twice: Two words here (TW)",
            "Lower case (lc) is not one; Data protection officer (DPO) is.", "A (B)", "Very long name of a team (VLNT)")]
        assert queries.defined_abbreviations(made_up) == day10["defined_abbreviations"](made_up)
        for text in DAY10_TEXTS:
            for phrase in ("dad", "time off", "pay for", "mfa", "multi-factor authentication", "locked", "it", "usd 1,000"):
                assert queries.contains(text, phrase) == day10["contains"](text, phrase), (text, phrase)

    def test_expansion_matches_day_10_notebook(self, day10: dict) -> None:
        # The notebook's map, the same map matched word by word, and the map plus one more entry (Day 10, section 7.1)
        entries = day10["ENTRIES"]
        word_entries = [(tuple(sorted({w for p in phrases for w in p.split()})), terms, why) for phrases, terms, why in entries]
        meals_entries = entries + [(("dinner", "lunch", "breakfast"), ("meals",), "not recorded: added for this demo")]
        texts = [q.question for q in evaluation.load_queries(REPO_ROOT / "evals" / "questions-v3.jsonl")] + DAY10_TEXTS
        overlapping = [(("a b",), ("x", "y"), "made up"), (("c",), ("y", "z"), "made up"), (("b c",), ("x",), "made up")]
        texts += ["a b c?", "A B and b c, then c?"]
        for e in (entries, word_entries, meals_entries, [], queries.VOCABULARY, overlapping):
            for text in texts:
                assert asdict(queries.expand(text, e)) == asdict(day10["expand"](text, e)), text
        assert [q.id for q in day10["queries"] if queries.expand(q.question, entries).matched] == \
               ["d01-01", "d01-02", "d07-a03", "d07-a08", "d07-a11", "d07-u12"]

    def test_revised_names_rule_matches_day_10_notebook(self, day10: dict) -> None:
        # Day 12 moved Day 10's revised names rule and NamesReader (section 8) to queries.py. Only their definitions are
        # executed (the cells also rank every question); a stand-in embedder keeps this class free of any model.
        ns = run_notebook_cells("day_10_query_understanding_and_transformations.ipynb", DAY10_CELLS,
                                {k: day10[k] for k in ("chunks", "queries", "by_qid")}, definitions_of=("cell-36", "cell-38"))
        assert queries.TYPE_WORDS == ns["TYPE_WORDS"]
        chunks = day10["chunks"]
        by_doc = defaultdict(list)
        for c in chunks:
            by_doc[c.doc_id].append(answering.Source(c.chunk_id, c.title, c.section, c.last_reviewed, c.text, 0.5,
                                                     c.source_path))
        made_up = [answering.Source("made-up#cities", "Made up", "Cities", None, "Use the Other cities row for anywhere "
                                    "else.", 0.5, "made-up.md"),     # one type word each: either form must count
                   answering.Source("made-up#offices", "Made up", "Offices", None, "Each country office sets its own "
                                    "hours.", 0.5, "made-up.md")]
        source_sets = [[], *by_doc.values(), [s for ss in by_doc.values() for s in ss][:5], made_up[:1], made_up[1:]]
        texts = [q.question for q in evaluation.load_queries(REPO_ROOT / "evals" / "questions-v4.jsonl")] + DAY10_TEXTS
        texts += ["Is Mexico City a city? And Italy a country?", "What about New Delhi. And Rome?", "Tokyo",
                  "Is Slack allowed? Can I install Slack at home?", "Can I fly to Italy? Is Italy covered?"]
        for sources in source_sets:
            for text in texts:
                assert queries.unknown_names_v2(text, sources) == ns["unknown_names_v2"](text, sources), text
        fake = HashEmbedder()
        for min_score in (-1.0, 0.0, 0.63):
            for rule_name in ("unknown_names", "unknown_names_v2"):
                module_reader = queries.NamesReader(fake, getattr(queries, rule_name) if rule_name.endswith("v2")
                                                    else answering.unknown_names, min_score=min_score)
                nb_reader = ns["NamesReader"](fake, ns[rule_name], min_score=min_score)
                for sources in source_sets:
                    for text in texts[::7]:
                        context = answering.Context(text, tuple(sources))
                        assert asdict(module_reader.generate(context)) == asdict(nb_reader.generate(context)), text
        assert queries.NamesReader.name == ns["NamesReader"].name == answering.ExtractiveReader.name


class HashEmbedder:
    """A stand-in for the embedding model: a fixed pseudo-random unit vector per text, the same for both readers."""
    model = {"repo": "stand-in"}

    @staticmethod
    def vector(text: str) -> np.ndarray:
        rng = np.random.default_rng(int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16))
        v = rng.standard_normal(16)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def embed(self, texts, batch_size: int = 16):
        return np.stack([self.vector(t) for t in texts])

    def embed_query(self, question: str, instruction: str = ""):
        return self.vector("query: " + question)


def definitions(source: str) -> str:
    """Only what a cell defines: its imports, functions, classes and constants (literal values), not its computations."""
    keep = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            keep.append(node)
        elif isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
                keep.append(node)
            except ValueError:
                pass
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


# ----------------------------------------------------------------------------- Day 11: contexts and end-to-end steps
DAY11 = "day_11_advanced_retrieval_section_expansion.ipynb"


@pytest.fixture(scope="class")
def day11(embedder: embeddings.Embedder):
    # The Day 11 notebook's setup and settings cells, then only the definitions in its sections 3 and 8 (the notebook
    # also ranks 190 queries with six retrievers and answers 160 questions three ways). Their inputs, the sections in
    # order, the records and the rankings, are built here by shared code that earlier classes check.
    documents, _ = ingest(REPO_ROOT)
    chunks = [c for d in documents for c in chunking.section_chunks(d)]
    by_cid = {c.chunk_id: c for c in chunks}
    queries3 = evaluation.load_queries(REPO_ROOT / "evals" / "questions-v3.jsonl")
    store = index.VectorIndex(embedder.model["repo"], embedder.revision, embedder.dimensions)
    index.sync(store, chunks, embedder.embed)
    rag = answering.Pipeline(store, embedder)
    bm25 = {user: BM25Index({c.chunk_id: c.text for c in chunks if set(c.allowed_groups) & groups})
            for user, groups in answering.DIRECTORY.items()}
    rankings = {"dense": {}, "weighted": {}}
    for q in queries3:
        lexical = [(h.id, h.score) for h in bm25[q.user].search(q.question, len(chunks))]
        rankings["dense"][q.id] = [(r.id, r.score) for r in rag.retrieve(q.user, q.question, len(chunks))]
        rankings["weighted"][q.id] = fusion.weighted(lexical, rankings["dense"][q.id], fusion.ALPHA)
    sections = defaultdict(list)                     # what the notebook's section 2 builds
    for c in chunks:
        sections[c.doc_id].append(c.chunk_id)
    position = {cid: (doc, i) for doc, ids in sections.items() for i, cid in enumerate(ids)}
    ns = run_notebook_cells(DAY11, ("cell-02", "cell-03"), {"chunks": chunks, "by_cid": by_cid, "rankings": rankings,
                                                             "SECTIONS": sections, "POSITION": position},
                            definitions_of=("cell-10", "cell-12", "cell-26"))
    ns.update({"queries3": queries3, "embedder": embedder, "store": store})
    yield ns
    release(ns)


class TestDay11Contexts:
    @staticmethod
    def restricted(day11: dict) -> dict:
        secret = "security-incident-reporting#after-you-report"     # section 6's in-memory restriction
        return {**day11["by_cid"], secret: dataclasses.replace(day11["by_cid"][secret], allowed_groups=("managers",))}

    def test_settings_and_structure_match_day_11_notebook(self, day11: dict) -> None:
        assert contexts.SETTINGS == day11["SETTINGS"]
        assert [f.name for f in dataclasses.fields(contexts.Step)] == [f.name for f in dataclasses.fields(day11["Step"])]
        assert [f.name for f in dataclasses.fields(contexts.SectionContext)] == \
               [f.name for f in dataclasses.fields(day11["Context"])]
        structure = contexts.DocumentStructure(day11["chunks"])
        assert dict(structure.sections) == dict(day11["SECTIONS"]) and structure.position == day11["POSITION"]
        restricted = self.restricted(day11)
        for c in day11["chunks"]:
            assert structure.where(c.chunk_id) == day11["where"](c.chunk_id)
            for setting in ("off",) + contexts.SETTINGS:
                assert structure.around(c.chunk_id, setting) == day11["around"](c.chunk_id, setting), c.chunk_id
            for user in ("alice", "maria", "mallory"):
                for records in (day11["by_cid"], restricted):
                    assert contexts.may_read(c.chunk_id, user, records) == day11["may_read"](c.chunk_id, user, records)

    def test_expansion_matches_day_11_notebook(self, day11: dict, monkeypatch) -> None:
        structure = contexts.DocumentStructure(day11["chunks"])
        restricted = self.restricted(day11)
        for q in day11["queries3"]:
            ranked = day11["rankings"]["weighted"][q.id]
            for setting in ("off",) + contexts.SETTINGS:
                for records in (day11["by_cid"], restricted):
                    for budget, hits in ((300, 3), (150, 3), (40, 3), (300, 1), (300, 5)):
                        module_context = contexts.expand_sections(ranked, q.user, q.question, setting, records,
                                                                  structure, budget, hits)
                        nb_context = day11["expand_sections"](ranked, q.user, q.question, setting, records, budget, hits)
                        assert asdict(module_context) == asdict(nb_context), (q.id, setting, budget, hits)
        # Another directory: the module takes it as a setting, the notebook reads its global on every call
        other = {"alice": {"managers"}, "maria": set()}
        monkeypatch.setitem(day11, "DIRECTORY", other)
        for q in day11["queries3"][::5]:
            ranked = day11["rankings"]["weighted"][q.id]
            assert asdict(contexts.expand_sections(ranked, q.user, q.question, "parent", day11["by_cid"], structure,
                                                   directory=other)) == \
                   asdict(day11["expand_sections"](ranked, q.user, q.question, "parent", day11["by_cid"]))

    def test_end_to_end_matches_day_11_notebook(self, day11: dict, monkeypatch) -> None:
        texts = [c.text for c in day11["chunks"][:9]] + ["a", "Ünïcödé", day11["chunks"][0].text]
        module_cache, nb_cache = contexts.SentenceCache(day11["embedder"]), day11["SentenceCache"](day11["embedder"])
        assert np.array_equal(module_cache.embed(texts), nb_cache.embed(texts))
        assert np.array_equal(module_cache.embed_query("x?"), nb_cache.embed_query("x?"))
        assert module_cache.model == nb_cache.model
        reader = answering.ExtractiveReader(module_cache, min_score=day11["READER_THRESHOLD"])
        structure = contexts.DocumentStructure(day11["chunks"])
        natural = [q for q in day11["queries3"] if q.kind == "question"]

        def both(q, sent, gate):
            module_answer = contexts.answer_context(q.user, q.question, sent, day11["by_cid"],
                                                    dict(day11["rankings"]["weighted"][q.id]),
                                                    day11["rankings"]["dense"][q.id][0][1], reader, gate=gate)
            return module_answer, day11["answer_with"](q, sent, reader)

        for q in natural:
            top3 = tuple(cid for cid, _ in day11["rankings"]["weighted"][q.id][:3])
            expanded = contexts.expand_sections(day11["rankings"]["weighted"][q.id], q.user, q.question, "neighbours",
                                                day11["by_cid"], structure).sent
            for sent in (top3, expanded):
                module_answer, nb_answer = both(q, sent, day11["DENSE_GATE"])
                assert asdict(module_answer) == asdict(nb_answer), q.id
                assert contexts.why_refused(module_answer) == day11["why_refused"](nb_answer)
        # Other gates and budgets: the module takes them as settings, the notebook reads its globals
        for gate, budget in ((0.0, 300), (0.7, 300), (0.62, 60)):
            monkeypatch.setitem(day11, "DENSE_GATE", gate)
            monkeypatch.setitem(day11, "BUDGET", budget)
            for q in natural[::4]:
                top3 = tuple(cid for cid, _ in day11["rankings"]["weighted"][q.id][:3])
                module_answer = contexts.answer_context(q.user, q.question, top3, day11["by_cid"],
                                                        dict(day11["rankings"]["weighted"][q.id]),
                                                        day11["rankings"]["dense"][q.id][0][1], reader, gate=gate,
                                                        budget=budget)
                assert asdict(module_answer) == asdict(day11["answer_with"](q, top3, reader)), (q.id, gate, budget)
        # On the boundary (a gate exactly equal to the dense top score lets the question through in both), and a
        # context holding a chunk the asker may not read (both refuse before anything is sent)
        q = natural[0]
        top = day11["rankings"]["dense"][q.id][0][1]
        monkeypatch.setitem(day11, "DENSE_GATE", top)
        monkeypatch.setitem(day11, "BUDGET", 300)
        sent = tuple(cid for cid, _ in day11["rankings"]["weighted"][q.id][:3])
        on_the_line = contexts.answer_context(q.user, q.question, sent, day11["by_cid"],
                                              dict(day11["rankings"]["weighted"][q.id]), top, reader, gate=top)
        assert asdict(on_the_line) == asdict(day11["answer_with"](q, sent, reader)) and on_the_line.context is not None
        pay = "manager-pay-review-guide#overview"
        leaky = sent + (pay,)
        scores = {**dict(day11["rankings"]["weighted"][q.id]), pay: 0.1}
        assert _outcome(lambda: contexts.answer_context("alice", q.question, leaky, day11["by_cid"], scores, top, reader,
                                                        gate=top)) == \
               ("PermissionError", "a context holds a chunk alice may not read")
        alice_q = dataclasses.replace(q, user="alice")
        assert _outcome(lambda: day11["answer_with"](alice_q, leaky, reader))[0] == "PermissionError"


# ----------------------------------------------------------------------------- Day 12: the answer rubric
DAY12 = "day_12_answer_evaluation_and_regression_tests.ipynb"
GATE, READER_THRESHOLD = 0.62, 0.63                  # Day 07's measured values, as in Day 12's pipelines


def assignments(source: str, names) -> str:
    """Only the assignments to `names` in a cell: module-level constants that aren't literals (a set built from a
    string, a compiled pattern), which definitions() leaves out."""
    keep = [node for node in ast.parse(source).body if isinstance(node, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


@pytest.fixture(scope="class")
def day12(embedder: embeddings.Embedder):
    # The Day 12 notebook's setup and settings cells, then only what its sections 3, 5 and 11 define (the notebook also
    # answers 236 questions twice, reviews them and decides). The answers they grade, and the rankings and readers
    # closest_call() reads, are built here by shared code that earlier classes check.
    documents, _ = ingest(REPO_ROOT)
    chunks = [c for d in documents for c in chunking.section_chunks(d)]
    by_cid = {c.chunk_id: c for c in chunks}
    natural = [q for q in evaluation.load_queries(REPO_ROOT / "evals" / "questions-v4.jsonl") if q.kind == "question"]
    store = index.VectorIndex(embedder.model["repo"], embedder.revision, embedder.dimensions)
    index.sync(store, chunks, embedder.embed)
    rag = answering.Pipeline(store, embedder)
    bm25 = {user: BM25Index({c.chunk_id: c.text for c in chunks if set(c.allowed_groups) & groups})
            for user, groups in answering.DIRECTORY.items()}
    rankings = {"dense": {}, "weighted": {}}
    for q in natural:
        lexical = [(h.id, h.score) for h in bm25[q.user].search(q.question, len(chunks))]
        rankings["dense"][q.id] = [(r.id, r.score) for r in rag.retrieve(q.user, q.question, len(chunks))]
        rankings["weighted"][q.id] = fusion.weighted(lexical, rankings["dense"][q.id], fusion.ALPHA)
    cache = contexts.SentenceCache(embedder)
    readers = {"current": answering.ExtractiveReader(cache, min_score=READER_THRESHOLD),
               "names-v2": queries.NamesReader(cache, queries.unknown_names_v2, min_score=READER_THRESHOLD)}
    sent = {q.id: tuple(cid for cid, _ in rankings["weighted"][q.id][:3]) for q in natural}
    answers = {p: {q.id: contexts.answer_context(q.user, q.question, sent[q.id], by_cid, dict(rankings["weighted"][q.id]),
                                                 rankings["dense"][q.id][0][1], readers[p], gate=GATE)
                   for q in natural} for p in readers}
    nb = json.loads((NOTEBOOKS / DAY12).read_text(encoding="utf-8"))
    cells = {c.get("id"): "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
    source = "\n\n".join([cells["cell-02"], cells["cell-03"], definitions(cells["cell-06"]),
                           assignments(cells["cell-12"], {"STOPWORDS", "WORD"}), definitions(cells["cell-12"]),
                           definitions(cells["cell-18"]), assignments(cells["cell-34"], {"GRADE_FIELDS"}),
                           definitions(cells["cell-34"])])
    module = types.ModuleType(DAY12.removesuffix(".ipynb"))
    module.__dict__.update({"rankings": rankings, "ANSWERS": answers, "READERS": readers})
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(REPO_ROOT)
        mp.setitem(sys.modules, module.__name__, module)
        exec(compile(source, DAY12, "exec"), module.__dict__)
    ns = module.__dict__
    ns.update({"natural": natural, "by_cid": by_cid, "sent": sent, "answers": answers, "readers": readers,
               "rankings_": rankings})
    yield ns
    release(ns)


class TestDay12Rubric:
    @staticmethod
    def handmade(day12: dict) -> list:
        # Answers no pipeline gave: hand-written replies through the same steps, a model's own abstention, a gate
        # refusal, a blocked reply, a reply citing a chunk it wasn't sent, an unanswerable question answered
        by_qid = {q.id: q for q in day12["natural"]}
        dads, france, lost = by_qid["d01-01"], by_qid["d01-04"], by_qid["d03-13"]
        luna = ("Secondary caregivers receive 6 weeks of fully paid leave, and leave can be taken in up to two blocks "
                "within 12 months of the child's arrival. [parental-leave-policy#leave-entitlement]")
        replies = [luna, "Primary caregivers get 16 weeks of paid leave and secondary caregivers get 6 weeks "
                         "[parental-leave-policy#leave-entitlement].",
                   "Primary caregivers receive 18 weeks of fully paid leave [parental-leave-policy#leave-entitlement].",
                   "Leave is generous. Ask HR [parental-leave-policy#leave-entitlement].",
                   "Dads get time off [parental-leave-policy#leave-entitlement] [travel-policy#booking-travel]."]
        miscited = ("Report it within one hour to security@acme.example [travel-policy#safety-while-travelling].",
                    "Report it within one hour [security-incident-reporting#how-to-report]. Then tell the Security "
                    "Team [travel-policy#safety-while-travelling].")   # the words are in a source sent but not cited
        made = []
        for q in (dads, lost, by_qid["d06-02"], by_qid["d07-u01"]):
            for text in replies + list(miscited) + [answering.ABSTAIN]:
                a = contexts.answer_context(q.user, q.question, day12["sent"][q.id], day12["by_cid"],
                                            dict(day12["rankings_"]["weighted"][q.id]),
                                            day12["rankings_"]["dense"][q.id][0][1], answering.FixedReply(text),
                                            gate=GATE)
                made.append((q, a))
        gate = contexts.answer_context(france.user, france.question, day12["sent"][france.id], day12["by_cid"],
                                       dict(day12["rankings_"]["weighted"][france.id]),
                                       day12["rankings_"]["dense"][france.id][0][1], answering.FixedReply("x [y]."),
                                       gate=GATE)
        model_abstained = dataclasses.replace(made[7][1], reason="")       # a live model's own "I don't know"
        return made + [(france, gate), (dads, model_abstained)]

    def test_rubric_settings_match_day_12_notebook(self, day12: dict) -> None:
        assert rubric.STOPWORDS == day12["STOPWORDS"] and rubric.WORD.pattern == day12["WORD"].pattern
        assert (rubric.SUPPORT_MIN, rubric.COST) == (day12["SUPPORT_MIN"], day12["COST"])
        assert [f.name for f in dataclasses.fields(rubric.Grade)] == [f.name for f in dataclasses.fields(day12["Grade"])]
        assert rubric.GRADE_FIELDS == day12["GRADE_FIELDS"]
        texts = ["", "a", "The policy's 14-day limit (USD 1,000) applies; see https://it.acme.example/reset at 10:00.",
                 "Ünïcödé café, e-mail: people-team@acme.example!", "I can't do it, can I? No.",
                 "My laptop might have a virus. Should I switch it off? Now!", "One sentence only",
                 "Two. Sentences.", "  Leading space. And another one", "Ask HR or IT on your PC, ok? It's 2 km."]
        for text in texts:
            assert rubric.content_words(text) == day12["content_words"](text), text
            assert rubric.first_words(text) == day12["first_words"](text), text
        sources = [answering.Source("a", "T", "S", None, text, 0.5, "p") for text in texts[2:5]]
        for text in texts:
            for k in range(len(sources) + 1):
                assert rubric.support(text, sources[:k]) == day12["support"](text, sources[:k]), (text, k)

    def test_grades_match_day_12_notebook(self, day12: dict) -> None:
        graded = 0
        for p, by_qid in day12["answers"].items():
            for q in day12["natural"]:
                a = by_qid[q.id]
                for sent in (day12["sent"][q.id], (), tuple(q.answer_chunks) or ("none",)):
                    assert asdict(rubric.grade(a, q, sent)) == asdict(day12["grade"](a, q, sent)), (p, q.id, sent)
                    graded += 1
        made = self.handmade(day12)
        for q, a in made:
            assert asdict(rubric.grade(a, q, day12["sent"][q.id])) == asdict(day12["grade"](a, q, day12["sent"][q.id]))
        miscited = [rubric.grade(a, q, day12["sent"][q.id]) for q, a in made
                    if q.id == "d03-13" and a.status == "answered" and "within one hour" in a.text]
        assert [g.grounded for g in miscited] == [False, False] and made[-1][1].reason == "" and \
            rubric.grade(made[-1][1], made[-1][0], ()).refused_by == "the model"
        assert graded == 3 * 2 * 236
        stages = {rubric.grade(a, q, day12["sent"][q.id]).stage for p in day12["answers"] for q in day12["natural"]
                  for a in [day12["answers"][p][q.id]]}
        assert {"ok", "gate", "names rule: a sentence's first word", "names rule: a name", "citation check",
                "reader: answer in the context, not used", "answered: nothing answers it"} <= stages

    def test_values_and_comparisons_match_day_12_notebook(self, day12: dict, monkeypatch) -> None:
        for out in ("correct", "refused", "wrong", "answered"):
            for cost in (0, 1, 2, 3, 2.5):
                assert rubric.value(out, cost) == day12["value"](out, cost), (out, cost)
            assert rubric.value(out) == day12["value"](out)
        by_qid = {q.id: q for q in day12["natural"]}
        for q in (by_qid["d01-01"], by_qid["d07-u01"]):
            outs = ("correct", "refused", "wrong") if q.answerable else ("refused", "answered")
            for a_out in outs:
                for b_out in outs:
                    grades = {"a": {q.id: day12["Grade"](a_out, *[None] * 6, "ok")},
                              "b": {q.id: day12["Grade"](b_out, *[None] * 6, "ok")}}
                    monkeypatch.setitem(day12, "GRADES", grades)
                    assert rubric.better_or_worse(q, a_out, b_out) == day12["better_or_worse"](q, "a", "b"), \
                        (q.id, a_out, b_out)

    def test_closest_calls_and_entries_match_day_12_notebook(self, day12: dict, monkeypatch) -> None:
        for p, by_qid in day12["answers"].items():
            grades = {q.id: rubric.grade(by_qid[q.id], q, day12["sent"][q.id]) for q in day12["natural"]}
            monkeypatch.setitem(day12, "GRADES", {p: grades})
            for q in day12["natural"]:
                a = by_qid[q.id]
                call = rubric.closest_call(a, day12["readers"][p], day12["rankings_"]["dense"][q.id][0][1], GATE,
                                           day12["cutoff_gap"](q, 3))
                assert call == day12["closest_call"](q, p), (p, q.id)
                assert rubric.answer_entry(a, grades[q.id], call) == day12["answer_entry"](q, p), (p, q.id)
        # Other gates: the module takes the gate as an argument, the notebook reads its global
        q = day12["natural"][0]
        for gate in (0.0, 0.5, 0.7, day12["rankings_"]["dense"][q.id][0][1]):
            monkeypatch.setitem(day12, "GATE", gate)
            for p in day12["answers"]:
                a = day12["answers"][p][q.id]
                assert rubric.closest_call(a, day12["readers"][p], day12["rankings_"]["dense"][q.id][0][1], gate,
                                           day12["cutoff_gap"](q, 3)) == day12["closest_call"](q, p)

    def test_compare_answers_matches_day_12_notebook(self, day12: dict) -> None:
        accepted = json.loads((REPO_ROOT / "evals" / "baselines" / "day-12-answers-v4.json").read_text(encoding="utf-8"))
        pipelines = tuple(accepted["pipelines"])
        qid = "d03-01"

        def edited(change) -> dict:
            run = json.loads(json.dumps(accepted))
            change(run)
            return run

        def set_field(field, value, pipeline="names-v2", question=qid):
            return lambda run: run["questions"][question][pipeline].__setitem__(field, value)

        edits = [lambda run: None, set_field("status", "blocked"), set_field("text", "something else"),
                 set_field("cited", []), set_field("outcome", "wrong"), set_field("stage", "gate"),
                 set_field("support", 0.5, "current"), set_field("grounded", False),
                 set_field("closest_call", [0.0002, "gate"]),
                 lambda run: (set_field("closest_call", [0.0002, "gate"])(run), set_field("outcome", "wrong")(run)),
                 lambda run: (set_field("closest_call", [0.001, "gate"])(run), set_field("outcome", "wrong")(run)),
                 lambda run: (set_field("closest_call", [None, "x"])(run), set_field("outcome", "wrong")(run)),
                 lambda run: run["metrics"]["dev"]["current"].__setitem__("correct", 0),
                 lambda run: (set_field("closest_call", [0.0002, "gate"])(run), set_field("outcome", "wrong")(run),
                              run["metrics"]["dev"]["current"].__setitem__("correct", 0)),
                 lambda run: run["decisions"].__setitem__("D1 Day 10's names rule in the current pipeline", "no"),
                 lambda run: run["dataset"].__setitem__("version", 3),
                 lambda run: run["pipelines"]["current"].__setitem__("budget_words", 200),
                 lambda run: run["rubric"].__setitem__("grounded", "x"),
                 lambda run: run["questions"].__setitem__("new-question", run["questions"][qid]),
                 lambda run: run["questions"].pop(qid)]
        for i, change in enumerate(edits):
            new = edited(change)
            for old in (accepted, edited(set_field("text", "older"))):
                assert rubric.compare_answers(new, old, pipelines) == day12["compare_answers"](new, old), i
