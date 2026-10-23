# 15 Days of Modern RAG from Scratch

A hands-on, 15-day series that teaches **Retrieval-Augmented Generation (RAG)** from first principles to a small, measured, security-aware application. Each day has a lesson, a runnable notebook, a LinkedIn post and a short visual explainer.

**Who it's for:** Python developers who want to understand how RAG works, why it fails, and what it takes to make it reliable in real organisations.

**How it's different:** every day starts from a transparent baseline, shows its evidence, and is measured before anything more complex is added. No framework is required to follow along. Where a tool is used, the lesson explains why and what the alternatives are.

> **Status:** Days 01–08 are available. Days 09–15 are planned in [`ROADMAP.md`](ROADMAP.md) and will be added one at a time.

## Day index

| Day | Topic | Lesson | Code | Visuals | Post | Status |
|---|---|---|---|---|---|---|
| 01 | RAG foundations | [LEARN.md](docs/day-01/LEARN.md) | [notebook](notebooks/day_01_rag_foundations.ipynb) | [slides](assets/day-01/) | [post](linkedin/day-01.md) | ✅ Verified 2026-10-13 |
| 02 | Ingestion, parsing and metadata | [LEARN.md](docs/day-02/LEARN.md) | [notebook](notebooks/day_02_ingestion_and_metadata.ipynb) | [slides](assets/day-02/) | [post](linkedin/day-02.md) | ✅ Verified 2026-10-13 |
| 03 | Chunking and document structure | [LEARN.md](docs/day-03/LEARN.md) | [notebook](notebooks/day_03_chunking_and_structure.ipynb) | [slides](assets/day-03/) | [post](linkedin/day-03.md) | ✅ Tested 2026-10-15 |
| 04 | Embeddings, lexical matching and similarity | [LEARN.md](docs/day-04/LEARN.md) | [notebook](notebooks/day_04_embeddings_and_similarity.ipynb) | [slides](assets/day-04/) | [post](linkedin/day-04.md) | ✅ Tested 2026-10-16 |
| 05 | Local vector search and data lifecycle | [LEARN.md](docs/day-05/LEARN.md) | [notebook](notebooks/day_05_vector_search_and_lifecycle.ipynb) | [slides](assets/day-05/) | [post](linkedin/day-05.md) | ✅ Tested 2026-10-18 |
| 06 | First end-to-end grounded RAG | [LEARN.md](docs/day-06/LEARN.md) | [notebook](notebooks/day_06_grounded_rag.ipynb) | [slides](assets/day-06/) | [post](linkedin/day-06.md) | ✅ Tested 2026-10-20 |
| 07 | Evaluation baseline | [LEARN.md](docs/day-07/LEARN.md) | [notebook](notebooks/day_07_evaluation_baseline.ipynb) · [evals/](evals/) | [slides](assets/day-07/) | [post](linkedin/day-07.md) | ✅ Tested 2026-10-22 |
| 08 | Hybrid search and rank fusion | [LEARN.md](docs/day-08/LEARN.md) | [notebook](notebooks/day_08_hybrid_search_and_fusion.ipynb) · [baseline](evals/baselines/day-08-retrieval.json) | [slides](assets/day-08/) | [post](linkedin/day-08.md) | ✅ Tested 2026-10-23 |
| 09 | Reranking and context selection | — | — | — | — | Planned |
| 10 | Query understanding and transformations | — | — | — | — | Planned |
| 11 | Advanced retrieval for harder questions | — | — | — | — | Planned |
| 12 | Answer evaluation and regression tests | — | — | — | — | Planned |
| 13 | Operations: tracing, latency and cost | — | — | — | — | Planned |
| 14 | Security, governance and freshness | — | — | — | — | Planned |
| 15 | Capstone application | — | — | — | — | Planned |

## Quick start

**You need:** Python **3.11 or newer**, Git, and about 600 MB of disk space (from Day 04: a 133 MB embedding model and its runtime). No API keys: Day 06 can optionally call a language model you configure, local or hosted, but runs without one. Day 04 downloads its model from Hugging Face on first run; every other day runs offline.

```bash
git clone https://github.com/shashanksuresh18/15-days-of-rag-from-scratch.git
cd 15-days-of-rag-from-scratch

python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
jupyter lab notebooks/day_01_rag_foundations.ipynb   # or any other day's notebook
```

Already set up for an earlier day? Run `git pull` and `python -m pip install -r requirements.txt` again: later days may add a package (Day 02 adds `pypdf`; Day 03 adds nothing; Day 04 adds `numpy`, `onnxruntime`, `tokenizers` and `huggingface-hub`; Days 05 to 08 add nothing). Day 04 needs Linux, Windows or an Apple silicon Mac: `onnxruntime` has no wheels for Intel Macs.

In JupyterLab choose **Kernel → Restart Kernel and Run All Cells**. Prefer VS Code? Open the notebook and select the `.venv` interpreter as the kernel.

Run the automated checks (they execute each notebook in a fresh kernel):

```bash
python -m pytest                          # everything
python -m pytest tests/test_evals.py      # one file at a time uses the least memory
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
| **What was tested** | Notebook executed top to bottom and `pytest` passed on Linux with Python 3.11, 3.12, 3.13 and 3.14 on 2026-10-18, from both the repo root and `notebooks/`. Day 04's embedding code was copied to `src/rag_from_scratch/embeddings.py` with a test that it returns exactly the Day 04 notebook's vectors; deliberate bugs in the copy and in the notebook were each caught. Verified after publication on Windows 11 (Python 3.11.9: 31 passed, 1 skipped; also offline) and in GitHub Actions on Linux, Windows and macOS (runs 39261318687 and 39430351485). CI runs the same tests on every push |
| **Known limitations** | Exact search only (fine to roughly 100,000 chunks, measured); the whole index file is rewritten on every save; filters are evaluated in Python; the sync assumes it receives the complete current corpus; permissions are copied into the index and are only as fresh as the last sync; `retrieve_for()` is a sketch without authentication, tenants or audit logging (Day 14) |
| **Cost** | Free. No new packages, no API key, no paid services. Runs offline once Day 04's model is in `models/` |

## Day 06 at a glance

| | |
|---|---|
| **How to run** | Quick start above, then open `notebooks/day_06_grounded_rag.ipynb`. It reads Day 05's saved index if present and syncs it, or builds it on a fresh clone |
| **What you build** | An end-to-end grounded RAG pipeline in five steps: retrieve as a specific user (permissions from a directory, filter always on) → a gate that abstains without calling anything when nothing is close → a budgeted context with citation ids and review dates → a prompt that keeps instructions apart from escaped, untrusted text (a planted note with fake tags shows why) → an API-free extractive reader, or any OpenAI-compatible model (Ollama locally, or a hosted one with the key in `.env`) → a citation check that blocks answers citing nothing, citing sources they weren't given, or changing numbers. Measured on 24 questions, including four that are cited, pass every check, and are still wrong |
| **What was tested** | Notebook executed top to bottom and `pytest` passed on Linux with Python 3.11, 3.12, 3.13 and 3.14 on 2026-10-20, from both the repo root and `notebooks/`, with and without Day 05's saved index. Day 05's index code was copied to `src/rag_from_scratch/index.py` (and `corpus.py`) with a test that it matches the Day 05 notebook; deliberate bugs in the copy and in the notebook were each caught. The optional live model was tested against local fake servers. Verified after publication on Windows 11 (Python 3.11.9: 43 passed, 1 skipped, one test file per process; also offline) and in GitHub Actions on Linux, Windows and macOS (runs 39481113937 and 39482676106). One live run with `gpt-6-luna` after publication (8 questions, about USD 0.0005): 6 of 8 right against the API-free reader's 4, by declining alice's pay question and the meeting-room question; its new-dads answer was cited, passed every check, and was still incomplete (LEARN.md, section 14). CI runs the same tests on every push |
| **Known limitations** | The API-free reader copies sentences: it can't judge whether one answers the question, so similar-but-wrong answers get through; its thresholds were chosen by eye on 15 questions (Day 07 measured them: the gate could be 0.62 instead of 0.50); the citation check verifies sources and numbers, not meaning, and splits a copied list number such as "3." into its own sentence (found on Day 07); escaping protects the prompt's structure, not the model's judgement; no retries, streaming or conversation history; the recorded outputs use the API-free reader |
| **Cost** | Free by default: no new packages, no API key. Optional live model: free with Ollama on your own computer; with OpenAI's `gpt-6-luna`, at most about USD 0.0044 for the notebook's eight live questions (prices checked 2026-10-20) |

## Day 07 at a glance

| | |
|---|---|
| **How to run** | Quick start above, then open `notebooks/day_07_evaluation_baseline.ipynb` (about 15 seconds). Uses Day 04's model (downloaded once if it isn't in `models/` yet) |
| **What you build** | An evaluation baseline before any optimisation: 81 labelled queries in [`evals/`](evals/) (who asks, the evidence, the chunks that hold it, a dev/test split, tied to a recorded snapshot of the documents and permissions) → a check that stops when documents change and lists the stale labels → BM25 and dense retrieval as each user, on the same queries and snapshot → hit rate@k, recall@k and MRR with 95% intervals and a paired comparison → every miss in the top 3 with a hand-written diagnosis → Day 06's thresholds measured with a rule on dev and reported on test → a committed baseline that every later run is compared with, near ties reported apart |
| **What was tested** | Notebook executed top to bottom and `pytest` passed on Linux with Python 3.11, 3.12, 3.13 and 3.14 on 2026-10-22, from both the repo root and `notebooks/`. Day 06's pipeline was copied to `src/rag_from_scratch/answering.py` with a test that it matches the Day 06 notebook, also with other thresholds; deliberate bugs in the copy and in the notebook were each caught. `tests/test_shared_code.py` now releases each day's notebook and model when its tests finish: its peak memory fell from 1,459 MB to 724 MB, the whole suite's from 1,632 MB to 766 MB (Linux, Python 3.13). Verified after publication on Windows 11 (Python 3.11.9: 56 passed, 1 skipped, one test file per process; peak working set of the shared-code tests 678 MB; also offline) and in GitHub Actions on Linux, Windows and macOS (runs 39748398969 and 39834132268). CI runs the same tests on every push |
| **Known limitations** | 81 queries written by one author: intervals are wide (94% means 79% to 98% on 31 questions); labels from evidence strings miss answers phrased differently from the evidence; the test split has 27 questions and can be used to decide something only once; answers are graded by fact strings, not judged (Day 12); the threshold rule uses only dev questions but wasn't pre-registered (an exploratory run had already shown the test split's scores; corrected 2026-10-23, LEARN.md section 9.2); the measured thresholds are recorded, not yet adopted; known failures (names rule, the checker's list numbers) are written down, not fixed |
| **Cost** | Free. No new packages, no API key, no paid services. Runs offline once Day 04's model is in `models/` |

## Day 08 at a glance

| | |
|---|---|
| **How to run** | Quick start above, then open `notebooks/day_08_hybrid_search_and_fusion.ipynb` (about 15 seconds). Uses Day 04's model (downloaded once if it isn't in `models/` yet) |
| **What you build** | Hybrid search from scratch on Day 07's 81 labelled queries and snapshot: a check that BM25 and dense still reproduce Day 07's baseline → why raw scores can't be added (the raw sum was BM25's top 3 for 62 of 71 queries) → Reciprocal Rank Fusion with a deterministic tie-break, and its k and window measured → lexical, dense and hybrid compared query by query with the sign test → RRF's two losses explained (a chunk only one retriever finds; a confident first place) → a weighted fusion with min-max scores, its weight chosen by a rule on the dev split and checked on the test split → every remaining miss diagnosed → one text or one per retriever, measured → permissions through fusion → a committed baseline with four retrievers for Days 09–11 |
| **What was tested** | Notebook executed top to bottom and `pytest` passed on Linux with Python 3.11, 3.12, 3.13 and 3.14 on 2026-10-23, from both the repo root and `notebooks/`. Day 07's evaluation harness was copied to `src/rag_from_scratch/evaluation.py` with a test that it matches the Day 07 notebook; deliberate bugs in the copy and in the notebook were each caught. GitHub Actions runs the same tests on Linux, Windows and macOS on every push |
| **Known limitations** | 81 queries written by one author; the fusion weight (0.7) was chosen on the dev split by a rule that wasn't pre-registered (exploratory runs had already shown test-split results), and the 15 test questions can't tell the weighted fusion and RRF apart (15 of 15 each); min-max normalisation depends on how many results each retriever returns; full rankings fused on 44 chunks (a large index fuses each retriever's top N); Day 06's pipeline still retrieves with dense alone until a later day switches it, measured |
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
├── .env.example                  placeholders only; Day 06's optional model settings (copy to .env, never commit it)
├── data/sample/acme-handbook/    synthetic corpus: md, html, pdf, txt (see data/sample/README.md)
├── data/sample/access-control.json  simulated permission export (Day 02)
├── docs/day-XX/LEARN.md          one lesson per day
├── notebooks/                    one runnable notebook per day
├── src/rag_from_scratch/         code reused by later days (ingestion, BM25, chunking, test questions, embeddings, vector index, answering, evaluation), see below
├── evals/                        labelled questions, the document snapshot they belong to, accepted baselines (Days 07 and 08)
├── tests/                        smoke tests that execute the notebooks, shared-code, sample-data and label checks
├── linkedin/day-XX.md            post text, alt text, posting order
├── assets/day-XX/                slide PNGs + editable SVG sources + carousel PDF
├── scripts/slides/               reproducible slide build (SVG -> PNG)
├── scripts/data/                 generator for the synthetic sample PDFs (standard library only)
├── outputs/                      created when you run a notebook; ignored by Git
├── models/                       the embedding model downloaded by Day 04 (and later days); ignored by Git
└── .github/workflows/tests.yml   CI on Linux, Windows and macOS
```

**Shared code (`src/rag_from_scratch/`).** Each technique is taught step by step in its own day's notebook. When a later day needs it again, a copy moves to `src/rag_from_scratch/` and later notebooks import it instead of repeating it (Day 03 imports Day 02's ingestion and Day 01's BM25; Day 04 also imports Day 03's chunker and its 13 questions; Day 05 also imports Day 04's embedding model; Day 06 also imports Day 05's vector index and its sync; Day 07 also imports Day 06's answering pipeline; Day 08 also imports Day 07's evaluation harness). `tests/test_shared_code.py` runs the original notebooks' code and checks that the shared copy returns identical results. Nothing needs installing: notebooks add `src/` to the import path themselves, and `pytest.ini` does the same for the tests.

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
