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
copied from the Day 07 notebook into evaluation.py.
"""

import dataclasses
import gc
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from rag_from_scratch import answering, chunking, corpus, embeddings, evaluation, fusion, index, reranking
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
                assert margin == day09["rerank_margin"](name, q) == accepted["queries"][q.id][name]["margin"], (name, q.id)
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
