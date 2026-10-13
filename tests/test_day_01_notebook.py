"""Day 01 smoke tests.

1. Execute the notebook top to bottom in a fresh kernel, from two different working
   directories. The notebook's section 8 holds the behavioural assertions, so any failing
   check makes the cell (and this test) fail.
2. Check that the committed copy of the notebook has no error outputs and no
   machine-specific paths, so it is safe to publish.
"""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "day_01_rag_foundations.ipynb"


def _stream_text(nb) -> str:
    return "".join(
        output.get("text", "")
        for cell in nb.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )


@pytest.mark.parametrize("working_dir", [NOTEBOOK.parent, REPO_ROOT], ids=["from-notebooks", "from-repo-root"])
def test_notebook_runs_top_to_bottom(working_dir: Path) -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(working_dir)}},
    )
    client.execute()  # raises CellExecutionError at the first failing cell
    assert "All Day 01 sanity checks passed." in _stream_text(nb)


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
    for marker in ("/home/", "/Users/", ":\\Users\\", "/root/"):
        assert marker not in text, f"Machine-specific path in saved outputs: {marker!r}"
