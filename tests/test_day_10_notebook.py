"""Day 10 smoke tests.

1. Execute the notebook top to bottom in a fresh kernel, from two different working
   directories. The notebook's section 12 holds the behavioural assertions, so any failing
   check makes the cell (and this test) fail. It also checks that Day 09's six retrievers still
   reproduce Day 09's accepted baseline, that every pre-registered rule gives the value committed in
   evals/preregistration/day-10.md, and compares both runs (v1's 81 queries with today's two
   retrievers added, and v2's 122 queries) with Day 10's accepted baselines in evals/baselines/.
   Day 10 reuses Day 04's embedding model and Day 09's reranker (downloaded the first time if
   they are missing), hence the long timeout.
2. Check that running the notebook never changes the versioned files in evals/: a baseline is
   accepted by a person, never rewritten by a run.
3. Check that the committed copy of the notebook has no error outputs and no machine-specific
   paths, so it is safe to publish.
"""

import hashlib
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "day_10_query_understanding_and_transformations.ipynb"
PASSED = "All Day 10 sanity checks passed."
REPRODUCED = ("BM25, dense, RRF, weighted, weighted+rerank, RRF+rerank against evals/baselines/day-09-retrieval.json: "
              "no changes")
COMPARED = ["compared with evals/baselines/day-10-retrieval.json: no changes",
            "compared with evals/baselines/day-10-retrieval-v2.json: no changes"]


def _stream_text(nb) -> str:
    return "".join(
        output.get("text", "")
        for cell in nb.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )


def _evals_fingerprint() -> dict[str, str]:
    return {p.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((REPO_ROOT / "evals").rglob("*")) if p.is_file()}


@pytest.mark.parametrize("working_dir", [NOTEBOOK.parent, REPO_ROOT], ids=["from-notebooks", "from-repo-root"])
def test_notebook_runs_top_to_bottom(working_dir: Path) -> None:
    before = _evals_fingerprint()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,   # per cell; only a first-run model download takes long
        kernel_name="python3",
        resources={"metadata": {"path": str(working_dir)}},
    )
    client.execute()  # raises CellExecutionError at the first failing cell
    text = _stream_text(nb)
    assert PASSED in text
    assert REPRODUCED in text and all(line in text for line in COMPARED)
    assert _evals_fingerprint() == before, "the notebook must not write into evals/"


def test_committed_outputs_are_clean() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    errors = [
        output
        for cell in nb.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors, "Committed notebook contains error outputs; re-run it before committing."

    text = _stream_text(nb)
    assert PASSED in text, "Commit the notebook with its outputs."
    assert REPRODUCED in text and all(line in text for line in COMPARED), \
        "Commit the notebook run against the committed baselines."
    for marker in ("/home/", "/Users/", ":\\Users\\", "/root/", "/tmp/", "AppData"):
        assert marker not in text, f"Machine-specific path in saved outputs: {marker!r}"
