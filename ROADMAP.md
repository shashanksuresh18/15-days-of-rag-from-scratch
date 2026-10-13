# 15 Days of Modern, Enterprise-Relevant RAG from Scratch

**Status:** Recommended revised roadmap · **Reviewed:** October 2026  
**Audience:** Python developers who want to progress from a simple, explainable RAG baseline to a small, measured, security-aware application.  
**Outcome:** By Day 15, learners can explain the architecture, run the code, inspect failures, measure improvements, and identify what remains before enterprise production use.

> **Important:** The five historical RAG notebooks (Parts 1–18) are **learning references**, not the syllabus or verified current implementations. Claude may reorder, replace, or modernize their approaches. Verify current APIs, models, package versions, licensing, and official docs **when producing each lesson**; don't assume the October 2026 tool landscape will remain unchanged.
>
> **Keep it practical:** Teach the smallest working baseline first. Introduce advanced retrieval only if the task and evaluation justify its added complexity, latency, and cost. No single framework, vector database, or agentic design is mandatory.

## 15-day roadmap

| Day | Topic | What learners should understand | Runnable GitHub deliverable |
| --- | --- | --- | --- |
| **01** | **RAG fundamentals and when to use it** | Retrieval vs. generation, grounding, common failures, and when long-context prompting or ordinary search may suffice | Small **API-free** retriever that displays its evidence; draw a basic RAG pipeline |
| **02** | **Ingestion, document parsing, and metadata** | Ingest text/PDF-like sources; retain source, section, timestamps, ownership/authorization metadata, and provenance | Reproducible sample ingestion pipeline with stable document IDs; show a table/PDF extraction caveat |
| **03** | **Chunking and document structure** | Fixed/token/section-based splitting; overlap trade-offs; tables, headings, and lost context | Compare at least two chunking methods on real example queries and visualize differences |
| **04** | **Embeddings, lexical matching, and similarity** | Dense vs. keyword signals; embedding model choices; cosine/dot-product caveats; the importance of simple baselines | Run lexical and embedding-based retrieval on the same corpus; inspect where each fails. **Reuse Day 01's BM25 as the lexical baseline** rather than re-teaching it |
| **05** | **Local vector search and data lifecycle** | Indexing, metadata filters, persistence, updates, deletes, and why filters are not automatically authorization | Create/query a local vector index; demonstrate metadata filtering and document re-index/delete |
| **06** | **First end-to-end grounded RAG** | Retrieve → assemble context → generate; citation traceability, uncertainty, abstention, and prompt boundaries | Small working RAG Q&A with identifiable source references; configurable local/hosted model option |
| **07** | **Build an evaluation baseline early** | Label a small query set; inspect missing/irrelevant evidence; top-k, hit rate/recall@k, and why a metric needs ground truth | Version a **small labeled test set**, run baseline retrieval metrics and record errors. **Seed it with Day 01's recorded failure questions** (vocabulary mismatch, relevant-but-wrong, out-of-scope, stemming, answer at rank 3) and Day 02's table question |
| **08** | **Hybrid search and rank fusion** | BM25/keyword + vector search; score mismatch; Reciprocal Rank Fusion (RRF) | Compare lexical, dense, and hybrid retrieval on **Day 07's same queries** |
| **09** | **Reranking and context selection** | Candidate retrieval vs. reranking; cross-encoder trade-offs; deduplication, source diversity, and token budgets | Add optional reranker and show **measured** quality/latency changes vs. baseline |
| **10** | **Query understanding and transformations** | Rewrite, multi-query, decomposition, filtering/routing, failure modes, and when *not* to rewrite | Implement one query-improvement method and compare with a no-rewrite baseline |
| **11** | **Advanced retrieval for harder questions** | Parent-child/section-aware retrieval, multi-hop evidence needs, and *optional* agentic query planning; complexity vs. benefit | Prototype **one** advanced pattern with a clear off-switch, trace, and evaluation comparison |
| **12** | **End-to-end answer evaluation and regression tests** | Answer correctness/groundedness, citation quality, abstentions, judge limitations, and reproducible comparisons | Test retrieval **and** generated answers; produce a small regression report and documented rubric |
| **13** | **RAG operations: tracing, latency, and cost** | Request/step traces, cost/token visibility, p50/p95 latency, caching, timeouts, retries, and failure handling | Instrument the pipeline; produce a reproducible latency/cost breakdown and fallback demo |
| **14** | **Enterprise security, governance, and freshness** | Identity-aware document permissions, tenant boundaries, prompt injection from retrieved sources, PII, retention, updates/deletes, auditability | Demonstrate enforced user/document access checks **before generation**, a malicious-document test, and a security checklist |
| **15** | **Capstone: measured, documented RAG application** | Integration and honest engineering trade-offs; what a demo has vs. what production still needs | Runnable mini-app with setup docs, citations, evals, basic observability, authorization demo, tests, and a known-limitations section |

## Teaching phases

- **Days 01–05 — Core data/retrieval mechanics.** Understand the information pipeline; no black-box magic.
- **Days 06–07 — Working baseline + measurements.** Build grounded answers and a small eval set **before** optimizations.
- **Days 08–11 — Retrieval improvements based on evidence.** Experiment with hybrid retrieval, reranking, query transformation, and one advanced design.
- **Day 12 — Answer quality and regression protection.** Ensure retrieval improvements actually support better answers.
- **Days 13–14 — Enterprise engineering.** Demonstrate reliability, security, governance, and lifecycle operations.
- **Day 15 — Integrated capstone.** Deliver a usable educational app, not a claim of full production readiness.

## Daily publishing contract (required)

For **every day**, create the following under the public GitHub repository:

1. `docs/day-XX/LEARN.md` — concept, motivation, architecture, examples, where it fails, and exercises.
2. `notebooks/day_XX_<topic>.ipynb` **or** `src/day_XX_<topic>.py` — actually runnable, minimal reproducible implementation; provide sample inputs and realistic expected outputs.
3. `linkedin/day-XX.md` — engaging LinkedIn hook, what was learned/built, factual takeaways, repository link/placeholder, and alt text.
4. `assets/day-XX/` — **1–4 colorful, useful LinkedIn images** (preferred 1080 × 1350 PNG). Include cover, diagram, example/comparison, or key takeaways **only as needed**. Follow `VISUAL_STYLE_GUIDE.md`. If image generation/export isn't supported, say so and provide editable SVG or renderable diagram + export steps; **never claim nonexistent PNGs are ready**.
5. Updated README / day index, dependencies, `.env.example` where needed, relevant config, and a short **How to run / What was tested / Known limitations / Cost** section.
6. Starting Day 07, maintain and reuse `evals/` for comparison. Never invent benchmarks or pretend tests ran. Where possible include a quick smoke test or automated test.

**Before publishing:** Verify notebook/script execution; avoid secrets, copyrighted datasets of uncertain reuse status, invented benchmark numbers, and unverified current-version claims. If code can't be executed here, label it **unverified** and explain how the learner can test it.

## Recommended repository layout

```text
15-days-of-rag-from-scratch/
├── README.md
├── ROADMAP.md
├── VISUAL_STYLE_GUIDE.md
├── .gitignore
├── .env.example                 # placeholders only; no secrets
├── requirements.txt             # or a project lockfile; tested versions noted
├── data/sample/                 # redistributable tiny corpus + attribution
├── docs/day-01/LEARN.md
├── docs/day-02/LEARN.md
├── notebooks/
├── src/
├── evals/                      # small labeled test set, baselines, reports
├── tests/                      # smoke/regression tests
├── linkedin/                   # day-XX.md posts and image alt text
└── assets/day-01/              # shareable PNGs, optional editable SVGs
```

Not every folder needs content from Day 1. Grow the project day by day; don't invent missing implementations.

## Tooling and architectural guardrails

- **Python 3.11+:** every day targets Python 3.11 or newer (3.10 reached end of life on 2026-10-01, and the pinned Jupyter kernel needs 3.11).
- **Tool-agnostic first:** Teach retrieval and data flow independently of one framework; use thin integrations where they save time. Pin and test package versions in the working implementation; review official docs on each day.
- **Accessibility:** Prefer low-cost, locally reproducible examples. For any hosted LLM, clearly name the provider requirement, expected cost considerations, and a local/mock alternative when appropriate.
- **Evaluation discipline:** Use the same labeled queries and data snapshots for before/after tests. Report quality, latency, cost, and sample size; never infer causality from one cherry-picked example.
- **Security is not a late retrofit:** Carry source IDs and access metadata from ingestion; distinguish retrieval metadata filters from identity-enforced authorization. Treat documents as **untrusted data**. Day 14 demonstrates deeper controls.
- **No overclaiming:** A 15-day educational application is not automatically HIPAA/SOC 2 compliant, audited, scalable, or production-ready. Identify gaps explicitly.
- **Modern, not fashionable:** Agentic/iterative retrieval, multimodal RAG, knowledge graphs, and long-context alternatives are valuable **optional follow-on modules**, not default requirements for every project. Day 11 may introduce one of these if justified by the eval task.

## Optional post-series extensions

If the audience asks for more after Day 15, consider separate mini-series on **multimodal RAG** (images/tables/layout), **agentic RAG** (dynamic planning with a bounded budget), **GraphRAG / knowledge-graph retrieval**, **streaming/continuous ingestion**, **large-scale evaluation with human review**, and **production deployment/CI/CD**. These should be evaluated against simple baselines, not portrayed as universal upgrades.

## Reference material and maintenance

- Original notebooks cover numerous classic RAG concepts; they are reference material, not confirmed working code against current packages.
- Day 01 is implemented and verified (Python 3.11–3.14 on Linux, Windows 11, CI on Linux/Windows/macOS). Day 02 is implemented and was tested on 2026-10-13 (Python 3.11–3.14 on Linux; CI on every push). **Days 03–15 are planned, not implemented.**
- Day 02's ingestion output (`Document` records with `allowed_groups`, `content_hash` and `## ` headings) is the intended input for Day 03 (chunking), Day 05 (metadata filters, re-index and delete) and Day 14 (enforced permissions).
- **Image guidance:** `VISUAL_STYLE_GUIDE.md` already specifies a solid mobile-first visual format; keep it unless the branding preference changes.
- Revisit this roadmap when official documentation, supported models/APIs, evaluation findings, or reader feedback warrant it.

### Current official references for deeper research (verify again when generating daily lessons)

- Microsoft: [RAG and generative AI with Azure AI Search](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview) — classic and agentic retrieval trade-offs, governance.
- Microsoft: [Hybrid search in Azure AI Search](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview) — lexical + vector fusion.
- Microsoft: [Security trimming](https://learn.microsoft.com/en-us/azure/search/search-security-trimming-for-azure-search) — filtering results by authorization context.
- AWS: [Evaluate RAG sources in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation-kb.html) — retrieval-only and retrieve-and-generate evaluation.
- AWS: [Retrieval with Bedrock Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-how-retrieval.html) — citations, reranking, structured or iterative approaches.
