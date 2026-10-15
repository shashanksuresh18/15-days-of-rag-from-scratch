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

from rag_from_scratch.bm25 import BM25Index
from rag_from_scratch.ingest import ingest

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
