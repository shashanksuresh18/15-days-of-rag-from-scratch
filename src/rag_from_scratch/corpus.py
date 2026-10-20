"""Day 05's "what if the documents changed?" helper, packaged for reuse by later days.

corpus_after(repo_root, change) copies data/sample to a temporary folder, lets `change` edit the
copy (edit a file, delete one, change a permission in access-control.json, ...), then runs Day 02's
ingestion and Day 03's chunker on the copy and returns the chunks. The files in data/sample are
never touched, so a notebook can demonstrate document changes and still run again from a clean
state.

Copied from notebooks/day_05_vector_search_and_lifecycle.ipynb (section 5). One change in shape:
the notebook read the repository root from a global variable; here it is the first argument.
tests/test_shared_code.py checks that it returns exactly the notebook's chunks.
"""

import shutil
import tempfile
from pathlib import Path

from .chunking import Chunk, section_chunks
from .ingest import ingest


def corpus_after(repo_root: Path, change) -> list[Chunk]:
    # Apply `change` to a temporary copy of data/sample, then ingest and chunk the copy. The originals are untouched.
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "data" / "sample"
        shutil.copytree(repo_root / "data" / "sample", sample)
        change(sample)
        docs, _ = ingest(Path(tmp))
    return [c for d in docs for c in section_chunks(d)]
