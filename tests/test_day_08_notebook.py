"""Day 08 smoke tests.

1. Execute the notebook top to bottom in a fresh kernel, from two different working
   directories. The notebook's section 11 holds the behavioural assertions, so any failing
   check makes the cell (and this test) fail. It also checks that BM25 and dense retrieval still
   reproduce Day 07's accepted baseline, and compares the whole run (with both hybrids) with Day 08's
   accepted baseline in evals/baselines/. Day 08 reuses Day 04's embedding model; if it isn't in
   models/ yet, the first run downloads it (133 MB, once), hence the long timeout.
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
NOTEBOOK = REPO_ROOT / "notebooks" / "day_08_hybrid_search_and_fusion.ipynb"
PASSED = "All Day 08 sanity checks passed."
REPRODUCED = "BM25 and dense against evals/baselines/day-07-retrieval.json: no changes"
COMPARED = "compared with evals/baselines/day-08-retrieval.json: no changes"


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
    assert REPRODUCED in text and COMPARED in text
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
    assert REPRODUCED in text and COMPARED in text, "Commit the notebook run against the committed baselines."
    for marker in ("/home/", "/Users/", ":\\Users\\", "/root/", "/tmp/", "AppData"):
        assert marker not in text, f"Machine-specific path in saved outputs: {marker!r}"
