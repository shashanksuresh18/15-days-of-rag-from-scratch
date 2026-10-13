# LinkedIn — Day 01: RAG Foundations

**Status:** ready to post once the repository is pushed and the GitHub Actions run is green.
**Matches:** `docs/day-01/LEARN.md`, `notebooks/day_01_rag_foundations.ipynb` (outputs from the 2026-10-13 run).

---

## Post text

Ask a language model about your company's expense policy and, without the document, it can only guess or say it doesn't know.

Retrieval-Augmented Generation (RAG) adds a third option: look it up first, then answer from what you found.

I'm starting a 15-day series that builds RAG from first principles to a small, measured, security-aware app. Day 1 is the foundation, and it needs no API key.

What I built: a retriever in plain Python (standard library only) over 8 policy documents for a fictional company. It splits them into 34 citable sections, ranks them with BM25 (a classic keyword-scoring method) and returns either a cited answer or "I don't know".

Asked "How much is the home office allowance for remote employees?", it returns the right sentence (USD 500) with its source ID, and shows why that section scored 13.43 while the runner-up scored 4.92.

The most useful part was breaking it on purpose:

1. "How much time off do new dads get?" → it abstains, although the answer is in the parental leave policy. The question and the answer share no keywords. If retrieval misses the evidence, nothing downstream can recover it.

2. "What is Google's remote work policy?" → it confidently returns Acme's policy. Matching words is not the same as answering the question, and a score threshold can't catch it.

3. "What is the capital of France?" → "I don't know." That is the behaviour you want, and it's worth protecting as the system grows.

What I'm taking into the next 14 days:
• Inspect the retrieved evidence before blaming the model or rewriting the prompt.
• Every answer should point to a source someone can open and check.
• Build a simple baseline first, so you can prove later changes actually help.
• Permissions belong in the retrieval layer. A line in the prompt is not access control.

RAG isn't always the right tool, either. A few short documents can go straight into the prompt, and exact lookups are often better served by search or SQL.

Notebook, lesson and tests are on GitHub: https://github.com/shashanksuresh18/15-days-of-rag-from-scratch

Day 2 is ingestion, parsing and metadata. Follow along if you build with LLMs and want the fundamentals without the hype.

#RAG #LLM #Python #AIEngineering

---

## Images — posting order, alt text and purpose

Upload in this order. LinkedIn's API documentation recommends alt text under 120 characters (maximum 4,086), so each alt text below leads with a short sentence that works on its own; paste the full text where the box allows.

| # | File | Teaches |
|---|---|---|
| 1 | `assets/day-01/slide-01-cover.png` | The whole RAG loop in six numbered steps, why a model alone can only guess about private documents, and one real cited answer. |
| 2 | `assets/day-01/slide-02-how-it-works.png` | The three stages: indexing (once), retrieval and generation (every question), with the real index statistics from the notebook. |
| 3 | `assets/day-01/slide-03-example.png` | One real question traced through keywords, BM25 scores, the "I don't know" threshold and the cited answer. |
| 4 | `assets/day-01/slide-04-takeaways.png` | Three ways the baseline breaks (real runs), and which later day addresses each. |

**Alt text 1 — cover**
> Hand-drawn diagram: how RAG answers a question in six numbered steps. You ask; the retriever searches your docs and gets the top 3 chunks; the question plus chunks go to the LLM, which generates an answer with its source. A red note says that without RAG the LLM has never seen Acme's policy and can only guess. Real run below: Q: How much is the home office allowance for remote employees? A: a one-time home office allowance of USD 500, source remote-work-policy#equipment-and-home-office.

**Alt text 2 — how it works**
> Diagram of the three stages of RAG: indexing, retrieval and generation. Indexing, once: 8 docs are split on headings into 34 chunks and stored in a BM25 index, where "allowance" appears in 1 chunk, "vpn" in 6 and "acme" in 12, so rare words weigh more. Retrieval, every question: the question becomes keywords, chunks are scored with BM25, the top 3 are kept. Generation: a prompt of rules, evidence and question goes to the LLM (in Day 01, an API-free stand-in), which returns a cited answer, or "I don't know" if the best score is below 4.0.

**Alt text 3 — example**
> One real question traced through the Day 01 notebook. "How much is the home office allowance for remote employees?" is split into keywords; "much" appears in no document and adds 0, while "allowance" is rare and counts most. BM25 scores: equipment-and-home-office 13.43, security-while-remote 4.92, core-hours 4.29, all above the 4.0 threshold. The answer quotes the USD 500 allowance sentence with its source ID.

**Alt text 4 — takeaways**
> Three ways the baseline breaks, each a real run. 1: Wrong words — "How much time off do new dads get?" shares no keywords with the leave policy, scores 3.15 and the system says "I don't know". 2: Relevant is not answering — "What is Google's remote work policy?" matches Acme's policy at 5.57 and gives a wrong answer. 3: "I don't know" is a feature — "What is the capital of France?" matches nothing and the system correctly abstains.

## Posting options

- **Image post (simplest):** attach the four PNGs in the order above and add the alt text to each.
- **Swipeable document post:** upload `assets/day-01/day-01-carousel.pdf` (the same four slides as a 4-page PDF). Document posts may not offer per-page alt text, so keep the image descriptions in the post body or first comment.

## Pre-posting checklist

- [ ] Open https://github.com/shashanksuresh18/15-days-of-rag-from-scratch in a private/incognito window to confirm it's public and the README shows
- [ ] Repository is public, has a licence, and the GitHub Actions run is green
- [ ] Numbers in the post (8 documents, 34 sections, 13.43, 4.92, USD 500) still match the notebook output
- [ ] Alt text added to all four images
