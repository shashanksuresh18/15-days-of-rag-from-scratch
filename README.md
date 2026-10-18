# 15 Days of Modern RAG from Scratch

A hands-on, 15-day series that teaches **Retrieval-Augmented Generation (RAG)** from first principles to a small, measured, security-aware application. Each day has a lesson, a runnable notebook, a LinkedIn post and a short visual explainer.

**Who it's for:** Python developers who want to understand how RAG works, why it fails, and what it takes to make it reliable in real organisations.

**How it's different:** every day starts from a transparent baseline, shows its evidence, and is measured before anything more complex is added. No framework is required to follow along. Where a tool is used, the lesson explains why and what the alternatives are.

> **Status:** Days 01–05 are available. Days 06–15 are planned in [`ROADMAP.md`](ROADMAP.md) and will be added one at a time.

## Day index

| Day | Topic | Lesson | Code | Visuals | Post | Status |
|---|---|---|---|---|---|---|
| 01 | RAG foundations | [LEARN.md](docs/day-01/LEARN.md) | [notebook](notebooks/day_01_rag_foundations.ipynb) | [slides](assets/day-01/) | [post](linkedin/day-01.md) | ✅ Verified 2026-10-13 |
| 02 | Ingestion, parsing and metadata | [LEARN.md](docs/day-02/LEARN.md) | [notebook](notebooks/day_02_ingestion_and_metadata.ipynb) | [slides](assets/day-02/) | [post](linkedin/day-02.md) | ✅ Verified 2026-10-13 |
| 03 | Chunking and document structure | [LEARN.md](docs/day-03/LEARN.md) | [notebook](notebooks/day_03_chunking_and_structure.ipynb) | [slides](assets/day-03/) | [post](linkedin/day-03.md) | ✅ Tested 2026-10-15 |
| 04 | Embeddings, lexical matching and similarity | [LEARN.md](docs/day-04/LEARN.md) | [notebook](notebooks/day_04_embeddings_and_similarity.ipynb) | [slides](assets/day-04/) | [post](linkedin/day-04.md) | ✅ Tested 2026-10-16 |
| 05 | Local vector search and data lifecycle | [LEARN.md](docs/day-05/LEARN.md) | [notebook](notebooks/day_05_vector_search_and_lifecycle.ipynb) | [slides](assets/day-05/) | [post](linkedin/day-05.md) | ✅ Tested 2026-10-18 |
| 06 | First end-to-end grounded RAG | — | — | — | — | Planned |
| 07 | Evaluation baseline | — | — | — | — | Planned |
| 08 | Hybrid search and rank fusion | — | — | — | — | Planned |
| 09 | Reranking and context selection | — | — | — | — | Planned |
| 10 | Query understanding and transformations | — | — | — | — | Planned |
| 11 | Advanced retrieval for harder questions | — | — | — | — | Planned |
| 12 | Answer evaluation and regression tests | — | — | — | — | Planned |
| 13 | Operations: tracing, latency and cost | — | — | — | — | Planned |
| 14 | Security, governance and freshness | — | — | — | — | Planned |
| 15 | Capstone application | — | — | — | — | Planned |

## Quick start

**You need:** Python **3.11 or newer**, Git, and about 600 MB of disk space (from Day 04: a 133 MB embedding model and its runtime). No API keys. Day 04 downloads its model from Hugging Face on first run; every other day runs offline.

```bash
git clone https://github.com/shashanksuresh18/15-days-of-rag-from-scratch.git
cd 15-days-of-rag-from-scratch

python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
jupyter lab notebooks/day_01_rag_foundations.ipynb   # or any other day's notebook
```

Already set up for an earlier day? Run `git pull` and `python -m pip install -r requirements.txt` again: later days may add a package (Day 02 adds `pypdf`; Day 03 adds nothing; Day 04 adds `numpy`, `onnxruntime`, `tokenizers` and `huggingface-hub`; Day 05 adds nothing). Day 04 needs Linux, Windows or an Apple silicon Mac: `onnxruntime` has no wheels for Intel Macs.

In JupyterLab choose **Kernel → Restart Kernel and Run All Cells**. Prefer VS Code? Open the notebook and select the `.venv` interpreter as the kernel.

Run the automated checks (they execute each notebook in a fresh kernel):

```bash
python -m pytest
```

## Day 01 at a glance

| | |
|---|---|
| **How to run** | Quick start above, then open `notebooks/day_01_rag_foundations.ipynb` |
| **What you build** | An API-free RAG baseline in pure Python: ingest 8 synthetic policy files → 34 citable chunks → BM25 index from scratch → explainable top-3 retrieval → cited extractive answer or "I don't know" → the grounded prompt an LLM would receive |
| **What was tested** | Notebook executed top to bottom and `pytest` passed on Linux (Python 3.11, 3.12, 3.13, 3.14) and Windows 11 (Python 3.11.9) on 2026-10-13, from both the repo root and `notebooks/`. GitHub Actions passes on Linux, Windows and macOS with Python 3.11 and 3.14 |
| **Known limitations** | Keyword matching only (no synonyms or stemming); abstention threshold chosen by eye on a tiny corpus; no LLM, so answers are copied sentences; no access control — every user sees every document; English-centric tokenizer and stopwords |
| **Cost** | Free. No network access, no paid services, no environment variables |

## Day 02 at a glance

| | |
|---|---|
| **How to run** | Quick start above, then open `notebooks/day_02_ingestion_and_metadata.ipynb` |
| **What you build** | An ingestion pipeline for 13 synthetic files (Markdown, HTML, PDF, scanned PDF, plain text): one adapter per format → normalised text with `## ` headings → descriptive, access and provenance metadata → 11 `Document` records in `outputs/day-02/documents.jsonl`, 2 files quarantined with reasons → change detection (added, content changed, metadata changed, removed) between runs |
| **What was tested** | Notebook executed top to bottom and `pytest` (8 tests) passed on Linux with Python 3.11, 3.12, 3.13 and 3.14 on 2026-10-13, from both the repo root and `notebooks/`; Day 01's tests still pass with the new files. GitHub Actions runs the same tests on Linux, Windows and macOS with Python 3.11 and 3.14 on every push |
| **Known limitations** | Heuristic cleaning tuned to clean sample files (de-hyphenation, 3-space table columns, numbered headings); no OCR (scanned PDFs are detected and quarantined); file names as document IDs, so renames look like delete + add; permissions are recorded but not yet enforced at query time (Days 05 and 14); no connectors, retries or monitoring |
| **Cost** | Free. Runs offline, no paid services, no environment variables. One new dependency: `pypdf` (pure Python, BSD-3-Clause) |

## Day 03 at a glance

| | |
|---|---|
| **How to run** | Quick start above, then open `notebooks/day_03_chunking_and_structure.ipynb` |
| **What you build** | Two chunkers for Day 02's 11 records: fixed-size word windows and structure-aware chunks (split at headings, never inside a sentence or table row, header row repeated when a table is split, `Title > Section` context header) → a boundary map of where each one cuts → the same BM25 retriever on 13 diagnostic questions, including the Singapore hotel-limit table question → measured effects of overlap, window size and headers → chunks that inherit permissions and hashes, written to `outputs/day-03/chunks.jsonl` |
| **What was tested** | Notebook executed top to bottom and `pytest` passed on Linux with Python 3.11, 3.12, 3.13 and 3.14 on 2026-10-15, from both the repo root and `notebooks/`. New `tests/test_shared_code.py` checks that `src/rag_from_scratch` returns exactly what the Day 01 and Day 02 notebooks return. GitHub Actions runs the same tests on Linux, Windows and macOS with Python 3.11 and 3.14 on every push |
| **Known limitations** | Sizes in words, not model tokens; only `## ` sections (no nested heading paths); assumes a table's first row is its header; 13 hand-written questions on 11 short documents are a diagnostic, not a benchmark; permissions are carried on every chunk but not yet enforced (Days 05 and 14) |
| **Cost** | Free. Runs offline, no paid services, no environment variables, no new dependencies |

## Day 04 at a glance

| | |
|---|---|
| **How to run** | Quick start above, then open `notebooks/day_04_embeddings_and_similarity.ipynb`. The first run downloads the model (133 MB) into `models/` |
| **What you build** | A dense retriever from first principles on Day 03's 44 chunks: a small open embedding model (`BAAI/bge-small-en-v1.5`, MIT) pinned to a commit and checked by SHA-256 → tokens, `[CLS]` pooling, normalisation and the attention mask, step by step → cosine vs dot product vs distance → BM25 and embeddings on the same 17 questions (Day 03's 13 and 4 of Day 01's failures) and 30 exact-value lookups → a threshold test for out-of-scope questions → Day 03's context headers re-measured per retriever → vectors with a manifest (model, revision, chunk hash, permissions) and a re-embedding cache, written to `outputs/day-04/` |
| **What was tested** | Notebook executed top to bottom and `pytest` passed on Linux with Python 3.11, 3.12, 3.13 and 3.14 on 2026-10-16, from both the repo root and `notebooks/`. With the model present, the notebook makes no network connections. The first download from Hugging Face could not be run in the test environment (the files came from a copy matching the published SHA-256 values); it has since been verified on Windows 11 (Python 3.11.9) and in GitHub Actions on Linux, Windows and macOS (runs 38915667595 and 38921548369). Day 03's chunker was copied to `src/` with a test that it matches the Day 03 notebook |
| **Known limitations** | One small English model; 17 diagnostic questions on 11 short documents, not a benchmark; brute-force search (no vector index until Day 05); scores can differ slightly across CPUs (checks use ranks and tolerances); no Intel Mac support in `onnxruntime`; permissions are carried with every vector but not yet enforced (Days 05 and 14) |
| **Cost** | Free. No API key, no paid services. One-time downloads: about 50 MB of Python packages and the 133 MB model. Needs internet only for that first download |

## Day 05 at a glance

| | |
|---|---|
| **How to run** | Quick start above, then open `notebooks/day_05_vector_search_and_lifecycle.ipynb`. Uses Day 04's model (downloaded once if it isn't in `models/` yet) |
| **What you build** | An exact vector index in plain NumPy for Day 03's 44 chunks: vectors + records + the model, keyed by `chunk_id` → metadata filters for scope, freshness and permissions, applied before or after the ranking → one-file persistence with an atomic swap, no pickle, and a refusal to load with the wrong model → a sync that follows document changes by `chunk_id` and `chunk_hash` (add, re-embed, update metadata, delete) with a guard against wiping the index → four demonstrations that a permission filter is not access control, and a sketch of a check that holds → a measurement of when exact search stops being enough |
| **What was tested** | Notebook executed top to bottom and `pytest` passed on Linux with Python 3.11, 3.12, 3.13 and 3.14 on 2026-10-18, from both the repo root and `notebooks/`. Day 04's embedding code was copied to `src/rag_from_scratch/embeddings.py` with a test that it returns exactly the Day 04 notebook's vectors; deliberate bugs in the copy and in the notebook were each caught. GitHub Actions runs the same tests on Linux, Windows and macOS on every push |
| **Known limitations** | Exact search only (fine to roughly 100,000 chunks, measured); the whole index file is rewritten on every save; filters are evaluated in Python; the sync assumes it receives the complete current corpus; permissions are copied into the index and are only as fresh as the last sync; `retrieve_for()` is a sketch without authentication, tenants or audit logging (Day 14) |
| **Cost** | Free. No new packages, no API key, no paid services. Runs offline once Day 04's model is in `models/` |

## Repository layout

```text
.
├── README.md                     you are here
├── ROADMAP.md                    the 15-day plan
├── VISUAL_STYLE_GUIDE.md         rules for the LinkedIn visuals
├── CLAUDE.md                     conventions for AI coding assistants working in this repo
├── requirements.txt              tested, pinned dependencies (Python 3.11+)
├── requirements-assets.txt       optional: only for re-rendering slide PNGs
├── LICENSE                       MIT
├── .env.example                  placeholders only; no day needs secrets so far
├── data/sample/acme-handbook/    synthetic corpus: md, html, pdf, txt (see data/sample/README.md)
├── data/sample/access-control.json  simulated permission export (Day 02)
├── docs/day-XX/LEARN.md          one lesson per day
├── notebooks/                    one runnable notebook per day
├── src/rag_from_scratch/         code reused by later days (ingestion, BM25, chunking, test questions, embeddings), see below
├── tests/                        smoke tests that execute the notebooks, shared-code and sample-data checks
├── linkedin/day-XX.md            post text, alt text, posting order
├── assets/day-XX/                slide PNGs + editable SVG sources + carousel PDF
├── scripts/slides/               reproducible slide build (SVG -> PNG)
├── scripts/data/                 generator for the synthetic sample PDFs (standard library only)
├── outputs/                      created when you run a notebook; ignored by Git
├── models/                       the embedding model downloaded by Day 04 (and later days); ignored by Git
└── .github/workflows/tests.yml   CI on Linux, Windows and macOS
```

`evals/` appears when evaluation starts on Day 07.

**Shared code (`src/rag_from_scratch/`).** Each technique is taught step by step in its own day's notebook. When a later day needs it again, a copy moves to `src/rag_from_scratch/` and later notebooks import it instead of repeating it (Day 03 imports Day 02's ingestion and Day 01's BM25; Day 04 also imports Day 03's chunker and its 13 questions; Day 05 also imports Day 04's embedding model). `tests/test_shared_code.py` runs the original notebooks' code and checks that the shared copy returns identical results. Nothing needs installing: notebooks add `src/` to the import path themselves, and `pytest.ini` does the same for the tests.

## Principles

- **Baseline first.** Every technique is compared with a simpler one on the same questions.
- **Show the evidence.** Answers carry source IDs; retrieval explains its scores.
- **Measure, don't assume.** No invented benchmarks; results come from code you can run.
- **Security is a design input.** Permissions belong in the retrieval and data layer, not in prompts. Retrieved documents are untrusted input.
- **Honest scope.** These are teaching implementations, not production systems. Each day lists what is still missing.

## Data and licence

- The sample corpus is **synthetic** and written for this repository; Acme Corp is fictional. The two sample PDFs are generated by [`scripts/data/make_sample_pdfs.py`](scripts/data/make_sample_pdfs.py). Details in [`data/sample/README.md`](data/sample/README.md).
- **Downloaded model (Day 04):** [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), MIT licence, is downloaded by the notebook at run time into `models/` and is not redistributed in this repository.
- **Third-party assets:** the slide tooling bundles two fonts (SIL Open Font License) and a subset of Lucide icons (ISC licence). Details and licence files are in [`scripts/slides/README.md`](scripts/slides/README.md).
- **Licence:** [MIT](LICENSE) for the code, lessons, slides and posts. The bundled fonts (SIL Open Font License) and icons (ISC) keep their own licences, listed above.

## Contributing and feedback

Found an error, a broken command or an unclear explanation? Please open an issue with your OS, Python version and the full error message.
