# Shared code

`rag_from_scratch/` holds code that more than one day of the series uses.

| Module | Taught in | What it does |
|---|---|---|
| [`ingest.py`](rag_from_scratch/ingest.py) | [Day 02 notebook](../notebooks/day_02_ingestion_and_metadata.ipynb) | Turns the sample files into `Document` records (text with `## ` headings, permissions, provenance, content hash) and a quarantine list |
| [`bm25.py`](rag_from_scratch/bm25.py) | [Day 01 notebook](../notebooks/day_01_rag_foundations.ipynb) | BM25 keyword retrieval over any `{id: text}` mapping, with per-term score contributions |

**How it grows.** Every technique is first written out and explained in the notebook that teaches it. When a *later* day needs it again, a copy moves here, and the later notebook imports it instead of repeating it. The teaching notebooks keep their own full copies.

**How we know the copies are right.** [`tests/test_shared_code.py`](../tests/test_shared_code.py) runs each original notebook's code and checks that the module returns identical results (Day 02: every record and quarantine entry; Day 01: every score and ranking for seven questions).

**How to import it.** Nothing to install. Notebooks find the repository root and add `src/` to `sys.path`; `pytest.ini` sets `pythonpath = src` for the tests. In your own script, run from the repository root with `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH="src"`).

This is teaching code, not a published library: no versioning or API stability promises.
