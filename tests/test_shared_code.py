"""The shared modules in src/rag_from_scratch/ are copies of code taught in earlier notebooks.

These tests run each notebook's own code (its code cells, in order, from the repo root) and
check that the shared module gives exactly the same results. If someone edits a notebook or a
module and the two drift apart, a test here fails.
"""

import dataclasses
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from rag_from_scratch import chunking, corpus, embeddings, index
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
        mp.setitem(sys.modules, module.__name__, module)   # dataclasses look up their module
        exec(compile(source, name, "exec"), module.__dict__)
    return module.__dict__


@pytest.fixture(scope="module")
def day01() -> dict:
    return run_notebook_code("day_01_rag_foundations.ipynb")


@pytest.fixture(scope="module")
def day03() -> dict:
    return run_notebook_code("day_03_chunking_and_structure.ipynb")


def test_ingest_module_matches_day_02_notebook() -> None:
    notebook = run_notebook_code("day_02_ingestion_and_metadata.ipynb")
    documents, quarantined = ingest(REPO_ROOT)
    assert [asdict(d) for d in documents] == [asdict(d) for d in notebook["documents"]]
    assert [asdict(q) for q in quarantined] == [asdict(q) for q in notebook["quarantined"]]
    assert len(documents) == 11 and len(quarantined) == 2


@pytest.mark.parametrize("question", [
    "How much is the home office allowance for remote employees?",
    "How many days do I have to submit an expense claim?",
    "How long do we keep application logs?",
    "How much time off do new dads get?",
    "What is Google's remote work policy?",
    "My account is locked, what now?",
    "How long is parental leave?",
])
def test_bm25_module_matches_day_01_notebook(question: str, day01: dict) -> None:
    chunks, searchable_text = day01["chunks"], day01["searchable_text"]
    shared = BM25Index({c.chunk_id: searchable_text(c) for c in chunks})
    expected = day01["index"].search(question, k=len(chunks))
    actual = shared.search(question, k=len(chunks))
    assert [(r.chunk.chunk_id, r.score, r.contributions) for r in expected] == \
           [(h.id, h.score, h.contributions) for h in actual]


def _as_dicts(chunks) -> list[dict]:
    return [asdict(c) for c in chunks]


def test_chunking_module_matches_day_03_notebook(day03: dict) -> None:
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


def test_probes_module_matches_day_03_notebook(day03: dict) -> None:
    assert [asdict(p) for p in DAY03_PROBES] == [asdict(q) for q in day03["QUESTIONS"]]
    assert len(DAY03_PROBES) == 13


EDGE_LINES = [
    "a|b", "a | b", "Header | Value", " | ", "no pipes here", "## Heading", "C++ & Rust: 2 langs?",
    "Is it fast? Yes! It is. Really... ok", "x" * 5, "",
]
EDGE_PARAGRAPH = ("Can you split this? Yes! Short one. Another sentence follows here, with a comma. "
                  "Is a question still a sentence? It should be. Done!")


def test_chunking_helpers_match_day_03_notebook_on_edge_cases(day03: dict) -> None:
    # The sample corpus does not exercise every rule (no "?" inside an oversize paragraph, no "|"
    # without spaces), so compare the helpers directly on small made-up inputs as well.
    for line in EDGE_LINES:
        assert chunking.is_table_row(line) == day03["is_table_row"](line), line
        assert chunking.count_words(line) == day03["count_words"](line), line
        assert chunking.slugify(line) == day03["slugify"](line), line
    table = ["Name | Limit | Note", "A | 1 | x", "B | 2 | y y y y", "C | 3 | z", "D | 4 | w w"]
    units = [[EDGE_PARAGRAPH], ["- a list item"], table, ["a|b without spaces"]]
    for room in (3, 6, 10, 25, 200):
        assert chunking.split_unit([EDGE_PARAGRAPH], room) == day03["split_unit"]([EDGE_PARAGRAPH], room)
        assert chunking.split_unit(table, room) == day03["split_unit"](table, room)
        assert chunking.pack(units, room) == day03["pack"](units, room)
    lines = ["Intro line.", *table, "After the table.", "x | y", "Last."]
    assert chunking.to_units(lines) == day03["to_units"](lines)
    body = "Preface text.\n## One\n" + "\n".join(lines) + "\n\n## Two?\n" + EDGE_PARAGRAPH
    assert chunking.split_sections(body) == day03["split_sections"](body)


# ----------------------------------------------------------------------------- Day 04: embeddings
@pytest.fixture(scope="module")
def day04() -> dict:
    # Runs the whole Day 04 notebook (downloads the model on first use, then embeds offline).
    return run_notebook_code("day_04_embeddings_and_similarity.ipynb")


@pytest.fixture(scope="module")
def embedder() -> embeddings.Embedder:
    return embeddings.Embedder(REPO_ROOT)


def test_embedding_settings_match_day_04_notebook(day04: dict, embedder: embeddings.Embedder) -> None:
    assert embeddings.MODEL == day04["MODEL"]
    assert embeddings.QUERY_INSTRUCTION == day04["QUERY_INSTRUCTION"]
    assert embeddings.MAX_TOKENS == day04["MAX_TOKENS"]
    assert embeddings.default_model_dir(REPO_ROOT) == day04["MODEL_DIR"] == embedder.model_dir
    assert embedder.revision == day04["MODEL"]["revision"] and embedder.dimensions == 384
    assert embedder.tokenizer.truncation == day04["tokenizer"].truncation
    assert embedder.tokenizer.padding == day04["tokenizer"].padding
    assert embedder.input_names == day04["INPUT_NAMES"]


LONG_TEXT = " ".join(f"word{i} policy approval" for i in range(700))   # over 512 tokens: truncation path
EDGE_TEXTS = ["", "x", "USD 220", "How much time off do new dads get?", LONG_TEXT, "Ünïcödé — ✓ 15:00"]


def test_embeddings_match_day_04_notebook(day04: dict, embedder: embeddings.Embedder) -> None:
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


def test_model_file_checks_match_day_04_notebook(day04: dict, tmp_path: Path, monkeypatch, capsys) -> None:
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
    assert embeddings.file_sha256(target) == day04["file_sha256"](target) == embeddings.MODEL["files"]["tokenizer.json"][0]

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
@pytest.fixture(scope="module")
def day05() -> dict:
    # Runs the whole Day 05 notebook (it saves outputs/day-05/index.npz, which Day 06 reads).
    return run_notebook_code("day_05_vector_search_and_lifecycle.ipynb")


def _results(results) -> list[tuple]:
    # Result objects from two modules: compare their fields, with the chunk as a dict
    return [(r.rank, r.score, r.id, asdict(r.chunk)) for r in results]


def _outcome(call):
    # A return value, or the error type and its full message: both versions must agree on either
    try:
        return ("ok", call())
    except Exception as error:
        return (type(error).__name__, str(error))


def _shared_index(day05: dict) -> index.VectorIndex:
    shared = index.VectorIndex(day05["index"].model, day05["index"].revision, day05["index"].dimensions)
    shared.upsert(day05["chunks"], day05["chunk_vectors"])
    return shared


SEARCH_QUESTIONS = [p.question for p in DAY03_PROBES] + [
    "How much time off do new dads get?", "When do pay rises take effect?", "How do I connect to the VPN?",
    "What is Google's remote work policy?", "What is the capital of France?", "My account is locked, what now?",
]


def test_index_settings_match_day_05_notebook(day05: dict) -> None:
    assert index.TOP_K == day05["TOP_K"] and index.INDEX_FORMAT == day05["INDEX_FORMAT"]
    for groups in ({"managers", "all-employees"}, ["people-team", "managers", "all-employees"], (), ["x"]):
        assert index.groups_filter(groups) == day05["groups_filter"](groups)   # a list in any order, sorted
    rng = np.random.default_rng(6)
    for n in (0, 1, 5, 44):
        for scores in (rng.standard_normal(n).astype(np.float32), rng.integers(0, 3, n).astype(np.float32)):  # ties
            for k in range(0, n + 3):
                assert np.array_equal(index.top_k(scores, k), day05["top_k"](scores, k)), (n, k)


def test_index_search_matches_day_05_notebook(day05: dict, embedder: embeddings.Embedder) -> None:
    notebook, shared = day05["index"], _shared_index(day05)
    assert shared.records == notebook.records and np.array_equal(shared.vectors, notebook.vectors)
    assert shared.row == notebook.row and len(shared) == len(notebook) == 44
    groups = day05["groups_filter"]
    filters = [None, *day05["FILTERS"].values(), groups({"managers"}), groups(set()), {"doc_id": "no-such-document"},
               {"allowed_groups": "managers"}, {"section": "Overview", "last_reviewed": {"gte": "2025-01-01"}}]
    for question in SEARCH_QUESTIONS:
        q = embedder.embed_query(question)
        for where in filters:
            for mode in ("pre", "post"):
                for k in (1, 3, 5, 44, 60):
                    assert _results(shared.search(q, k, where, mode)) == _results(notebook.search(q, k, where, mode)), \
                        (question, where, mode, k)
        assert _results(shared.search(q)) == _results(notebook.search(q))          # the defaults
    q = embedder.embed_query("x")
    assert _outcome(lambda: shared.search(q, filter_mode="both")) == \
           _outcome(lambda: notebook.search(q, filter_mode="both"))


def test_index_filters_match_day_05_notebook(day05: dict) -> None:
    conditions = [*day05["FILTERS"].values(), {"allowed_groups": "managers"}, {"allowed_groups": {"any": []}},
                  {"doc_id": "travel-policy", "allowed_groups": {"any": ["all-employees"]}},
                  {"last_reviewed": {"gte": "2026-04-15"}}, {"last_reviewed": "2026-04-15"},
                  {"allowed_groups": {"lt": 3}}, {"owner": "IT"}, {"last_reviewed": {"gte": None}}]
    for chunk in day05["chunks"]:
        for where in conditions:   # includes an unknown operator, a misspelled field and an impossible comparison
            assert _outcome(lambda: index.matches(chunk, where)) == _outcome(lambda: day05["matches"](chunk, where))


def test_index_file_matches_day_05_notebook(day05: dict, tmp_path: Path) -> None:
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


def test_index_sync_matches_day_05_notebook(day05: dict, embedder: embeddings.Embedder, tmp_path: Path) -> None:
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
    assert _outcome(lambda: index.plan_sync(a, [chunk, chunk])) == _outcome(lambda: day05["plan_sync"](b, [chunk, chunk]))
    assert _outcome(lambda: a.update_metadata(edited)) == _outcome(lambda: b.update_metadata(edited))
    assert _outcome(lambda: index.VectorIndex(model, revision, 384).upsert([chunk], np.zeros((1, 3)))) == \
           _outcome(lambda: day05["VectorIndex"](model, revision, 384).upsert([chunk], np.zeros((1, 3))))
    assert a.delete(["no-such-chunk", chunk.chunk_id]) == b.delete(["no-such-chunk", chunk.chunk_id]) == 1
