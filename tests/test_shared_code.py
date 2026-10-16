"""The shared modules in src/rag_from_scratch/ are copies of code taught in earlier notebooks.

These tests run each notebook's own code (its code cells, in order, from the repo root) and
check that the shared module gives exactly the same results. If someone edits a notebook or a
module and the two drift apart, a test here fails.
"""

import json
import sys
import types
from dataclasses import asdict
from pathlib import Path

import pytest

from rag_from_scratch import chunking
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
