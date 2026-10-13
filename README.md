# 15 Days of Modern RAG from Scratch

A hands-on, 15-day series that teaches **Retrieval-Augmented Generation (RAG)** from first principles to a small, measured, security-aware application. Each day has a lesson, a runnable notebook, a LinkedIn post and a short visual explainer.

**Who it's for:** Python developers who want to understand how RAG works, why it fails, and what it takes to make it reliable in real organisations.

**How it's different:** every day starts from a transparent baseline, shows its evidence, and is measured before anything more complex is added. No framework is required to follow along. Where a tool is used, the lesson explains why and what the alternatives are.

> **Status:** Day 01 is published. Days 02–15 are planned in [`ROADMAP.md`](ROADMAP.md) and will be added one at a time.

## Day index

| Day | Topic | Lesson | Code | Visuals | Post | Status |
|---|---|---|---|---|---|---|
| 01 | RAG foundations | [LEARN.md](docs/day-01/LEARN.md) | [notebook](notebooks/day_01_rag_foundations.ipynb) | [slides](assets/day-01/) | [post](linkedin/day-01.md) | ✅ Verified 2026-10-13 |
| 02 | Ingestion, parsing and metadata | — | — | — | — | Planned |
| 03 | Chunking and document structure | — | — | — | — | Planned |
| 04 | Embeddings, lexical matching and similarity | — | — | — | — | Planned |
| 05 | Local vector search and data lifecycle | — | — | — | — | Planned |
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

**You need:** Python **3.11 or newer**, Git, and about 200 MB of disk space. No API keys.

```bash
git clone https://github.com/shashanksuresh18/15-days-of-rag-from-scratch.git
cd 15-days-of-rag-from-scratch

python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
jupyter lab notebooks/day_01_rag_foundations.ipynb
```

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
| **What was tested** | Notebook executed top to bottom and `pytest` passed on Python 3.11, 3.12, 3.13 and 3.14 (Linux, 2026-10-13), from both the repo root and `notebooks/`. Windows and macOS run in CI once the repository is on GitHub; not yet verified |
| **Known limitations** | Keyword matching only (no synonyms or stemming); abstention threshold chosen by eye on a tiny corpus; no LLM, so answers are copied sentences; no access control — every user sees every document; English-centric tokenizer and stopwords |
| **Cost** | Free. No network access, no paid services, no environment variables |

## Repository layout

```text
.
├── README.md                     you are here
├── ROADMAP.md                    the 15-day plan
├── VISUAL_STYLE_GUIDE.md         rules for the LinkedIn visuals
├── CLAUDE.md                     conventions for AI coding assistants working in this repo
├── requirements.txt              tested, pinned dependencies (Python 3.11+)
├── requirements-assets.txt       optional: only for re-rendering slide PNGs
├── .env.example                  placeholders only; Day 01 needs none
├── data/sample/acme-handbook/    synthetic corpus (see data/sample/README.md)
├── docs/day-01/LEARN.md          the lesson
├── notebooks/                    one runnable notebook per day
├── tests/                        smoke tests that execute the notebooks
├── linkedin/day-01.md            post text, alt text, posting order
├── assets/day-01/                slide PNGs + editable SVG sources
├── scripts/slides/               reproducible slide build (SVG -> PNG)
└── .github/workflows/tests.yml   CI on Linux, Windows and macOS
```

Folders such as `src/` and `evals/` appear when a day needs them (evaluation starts on Day 07).

## Principles

- **Baseline first.** Every technique is compared with a simpler one on the same questions.
- **Show the evidence.** Answers carry source IDs; retrieval explains its scores.
- **Measure, don't assume.** No invented benchmarks; results come from code you can run.
- **Security is a design input.** Permissions belong in the retrieval and data layer, not in prompts. Retrieved documents are untrusted input.
- **Honest scope.** These are teaching implementations, not production systems. Each day lists what is still missing.

## Data and licence

- The sample corpus is **synthetic** and written for this repository; Acme Corp is fictional. Details in [`data/sample/README.md`](data/sample/README.md).
- **Third-party assets:** the slide tooling bundles two fonts (SIL Open Font License) and a subset of Lucide icons (ISC licence). Details and licence files are in [`scripts/slides/README.md`](scripts/slides/README.md).
- **Licence:** _to be added before the repository is made public._

## Contributing and feedback

Found an error, a broken command or an unclear explanation? Please open an issue with your OS, Python version and the full error message.
