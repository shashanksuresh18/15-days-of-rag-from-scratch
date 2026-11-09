"""Day 11 smoke tests.

1. Execute the notebook top to bottom in a fresh kernel, from two different working
   directories. The notebook's section 11 holds the behavioural assertions, so any failing
   check makes the cell (and this test) fail. It also checks that Day 10's five retrievers still
   reproduce Day 10's accepted v2 baseline, that the pre-registered rule gives the setting and the
   dev values committed in evals/preregistration/day-11.md, and compares the run on questions-v3
   with Day 11's accepted baseline in evals/baselines/. Day 11 reuses Day 04's embedding model
   (downloaded the first time if it is missing), hence the long timeout.
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
NOTEBOOK = REPO_ROOT / "notebooks" / "day_11_advanced_retrieval_section_expansion.ipynb"
PASSED = "All Day 11 sanity checks passed."
REPRODUCED = ("BM25, dense, weighted, BM25+map, weighted+map against evals/baselines/day-10-retrieval-v2.json: "
              "no changes")
COMPARED = "compared with evals/baselines/day-11-retrieval-v3.json: no changes"


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
    assert "off-switch: 'off' sends exactly the weighted fusion's top 3" in text and ": True" in text
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
