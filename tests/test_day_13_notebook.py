"""Day 13 smoke tests.

1. Execute the notebook top to bottom in a fresh kernel, from two different working
   directories, with the live model switched off. The notebook's section 11 holds the behavioural
   assertions, so any failing check makes the cell (and this test) fail. It also checks that Day 12's accepted
   answers (both of its pipelines) are reproduced exactly, that the dev values are the ones committed in
   evals/preregistration/day-13.md, and compares the run on questions-v5 with Day 13's accepted baseline in
   evals/baselines/. Day 13 reuses Day 04's embedding model (downloaded the first time if it is missing), hence the
   long timeout. Its section 6 starts a local fake model server (127.0.0.1 only) and waits on it on purpose: a few
   seconds of the run are deliberate timeouts and deadlines.
2. Check that running the notebook never changes the versioned files in evals/: a baseline is
   accepted by a person, never rewritten by a run.
3. Check that the committed copy of the notebook has no error outputs, no machine-specific
   paths and no live-model output, so it is safe to publish.
"""

import hashlib
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "day_13_rag_operations_tracing_latency_cost.ipynb"
PASSED = "All Day 13 sanity checks passed."
REPRODUCED = "compared with evals/baselines/day-12-answers-v4.json: no changes"
COMPARED = "compared with evals/baselines/day-13-answers-v5.json: no changes"
LIVE_OFF = "Live model: off."


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
def test_notebook_runs_top_to_bottom(working_dir: Path, monkeypatch) -> None:
    monkeypatch.delenv("DAY13_LIVE_MODEL", raising=False)     # never call a real language model from a test
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
    assert PASSED in text and LIVE_OFF in text
    assert REPRODUCED in text and COMPARED in text
    assert "document text, question text or keys found: 0" in text
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
    assert LIVE_OFF in text, "Never commit a notebook executed with a live model."
    for marker in ("/home/", "/Users/", ":\\Users\\", "/root/", "/tmp/", "AppData", "sk-proj", "sk-live"):
        assert marker not in text, f"Machine-specific path or key in saved outputs: {marker!r}"
