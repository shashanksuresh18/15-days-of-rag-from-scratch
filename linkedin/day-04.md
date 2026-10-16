# LinkedIn — Day 04: Embeddings, Lexical Matching and Similarity

**Status:** ready to post once the `day-04` branch is merged into `main` and the GitHub Actions run is green (that run is also the first real download of the model on Linux, Windows and macOS).
**Matches:** `docs/day-04/LEARN.md`, `notebooks/day_04_embeddings_and_similarity.ipynb` (outputs from the 2026-10-16 run).

---

## Post text

15 Days of RAG from Scratch · Day 4/15: Embeddings, lexical matching and similarity

Keyword search and embeddings each found the evidence for 14 of 15 test questions. Not the same 14.

Day 1's keyword retriever (BM25) couldn't answer "How much time off do new dads get?", because the (synthetic) policy never says "dads". It says "primary caregivers" and "secondary caregivers". So Day 4 puts a small local embedding model (bge-small-en-v1.5, MIT licence, runs on a CPU) against BM25, on the same 44 chunks from Day 3.

What the notebook measured:

1. Same score, different misses. Embeddings found "new dads" (at rank 3) but missed "How soon must I report a lost laptop?": the answer sits in a section that never mentions laptops. BM25 found that one. Together they cover all 15.

2. Exact values are keyword territory. Every number in the documents as a query ("USD 220", "13 months"…): BM25 found 25 of 25 at rank 1, embeddings 14 of 25. For "USD 220", the embeddings returned chunks with other dollar amounts.

3. Similarity means "same topic", not "answers the question". "What is Google's remote work policy?" scored 0.710, above 5 answerable questions, including the Singapore hotel limit at 0.708. No threshold separates them. Unrelated sentences scored 0.30 to 0.50, never near 0.

4. One knob, opposite effects. Day 3's "Title > Section" header line helped BM25 (12 vs 9 questions at rank 1) and hurt embeddings (14 vs 15 found).

5. A vector belongs to its model. The model is pinned to a commit and checked by SHA-256, and every vector is stored with the model revision and its chunk's hash. One document edit re-embedded 1 of 44 chunks. A new model means re-embedding all of them.

17 questions on 11 short documents are a diagnostic, not a benchmark. Day 7 turns them into an evaluation set.

Runs locally: a notebook, a lesson and tests. Four new packages (about 50 MB) and a 133 MB model, downloaded once. No API key.

https://github.com/shashanksuresh18/15-days-of-rag-from-scratch

Day 5: these vectors go into a local index, with filters, updates and deletes.

#RAG #LLM #Python #AIEngineering

---

## Images — posting order, alt text and purpose

Upload in this order. LinkedIn's API documentation recommends alt text under 120 characters (maximum 4,086), so each alt text below leads with a short sentence that works on its own; paste the full text where the box allows.

| # | File | Teaches |
|---|---|---|
| 1 | `assets/day-04/slide-01-cover.png` | The core result: on the same chunks, keyword search and embeddings find the same number of answers but miss different questions. |
| 2 | `assets/day-04/slide-02-how-it-works.png` | How text becomes a vector (tokens, model, pooling, normalisation) and why a cosine score is relative, not a percentage. |
| 3 | `assets/day-04/slide-03-where-each-fails.png` | The two systematic failure tests: exact values (BM25's strength) and out-of-scope questions (no threshold separates them). |
| 4 | `assets/day-04/slide-04-knobs-and-models.png` | The same header knob helps one retriever and hurts the other, and what must be stored with every vector. |

**Alt text 1 — cover**
> Keyword search and embeddings both find 14 of 15 answers, missing different ones. Question: how much time off do new dads get? Left, BM25, matches words: its top 3 are sections of three unrelated policies (what not to do, purpose, your first day), each sharing one word with the question: off, time, new. No chunk says dads; the answer never reaches the top 3. Right, embeddings, meaning: the top 3 are all parental leave policy chunks, returning to work 0.72, who is eligible 0.67, leave entitlement 0.66, the answer at rank 3 (primary caregivers receive 16 weeks). Below, 15 answerable questions as dots for each retriever: BM25 misses "new dads", embeddings miss "lost laptop"; both score 14 of 15. Caption: different misses, together they find 15 of 15; that's the case for hybrid search on Day 08.

**Alt text 2 — how it works**
> Four steps turn text into 384 numbers, and the score is relative. 1, tokens: [CLS] how much time off do new dad ##s get ? [SEP]. 2, model: 12 tokens in, 12 vectors of 384 numbers out. 3, pool: keep the [CLS] vector, this model's rule; other models average. 4, normalise: divide by its length so every vector has length 1. Below, cosine similarity is the angle between the question and chunk vectors; with length 1, cosine equals the dot product. A 0 to 1 scale shows 9 pairs of unrelated sentences scoring 0.30 to 0.50 and "new dads" versus "secondary caregivers" at 0.62. Rank by score; it's not a percentage. Model: bge-small-en-v1.5, MIT licence, 33 million parameters, runs on a CPU.

**Alt text 3 — where each fails**
> BM25 finds exact values; embeddings can't say "no match". Top panel: every number in the documents used as a query, 25 in total (USD 220, 13 months, 15:00 and so on). BM25 finds 25 of 25, all at rank 1; embeddings find 14 of 25, 8 at rank 1. For USD 220, the Singapore hotel limit, the embeddings return chunks with USD 75, USD 500 and USD 1,000. Bottom panel: a dot plot of each question's top-1 cosine score from 0.4 to 0.9. Fifteen answerable questions score 0.626 to 0.856. Two the documents can't answer: France at 0.425, and Google's remote work policy at 0.710, with five answerable questions below it. A threshold that rejects Google's question also rejects Singapore's hotel limit at 0.708. BM25: France gets 0 hits and abstains; Google scores 6.21 and doesn't. Caption: similarity means same topic, not answers the question.

**Alt text 4 — knobs and models**
> The Title > Section header helps BM25 and hurts embeddings. Bars for 15 answerable questions: BM25 finds 14 with headers (12 at rank 1) and 13 without (9 at rank 1); embeddings find 14 with headers (12 at rank 1) and 15 without (13 at rank 1). Why: a shared header pulls a document's chunks together, average similarity 0.65 to 0.71, and "new dads" drops from rank 1 to 3. Below, a vector belongs to its model: store with every vector the model BAAI/bge-small-en-v1.5, the revision 5c38ec7c405e, pooling CLS with length 1, the chunk hash and the allowed groups. Edit one section: 1 of 44 chunks re-embedded. New model or revision: re-embed all 44. Pinned commit plus SHA-256, ONNX, no pickle. Model 133 MB, downloaded once, then it runs offline. Next: Day 05, local vector search and data lifecycle.

## Posting options

- **Image post (simplest):** attach the four PNGs in the order above and add the alt text to each.
- **Swipeable document post:** upload `assets/day-04/day-04-carousel.pdf` (the same four slides as a 4-page PDF). Document posts may not offer per-page alt text, so keep the image descriptions in the post body or first comment.

## Pre-posting checklist

- [ ] `day-04` merged into `main`, tag `day-04` pushed, GitHub Actions green on Linux, Windows and macOS (the first run downloads the model from Hugging Face on each OS; later runs use the cache)
- [ ] Open the repository in a private/incognito window and check the README index links to the Day 04 lesson, notebook, slides and post
- [ ] Numbers in the post (44 chunks, 14 of 15, 25 of 25 vs 14 of 25, 0.710 vs 0.708, 0.30 to 0.50, 12 vs 9, 14 vs 15, 1 of 44, about 50 MB, 133 MB) still match the notebook output
- [ ] Alt text added to all four images
