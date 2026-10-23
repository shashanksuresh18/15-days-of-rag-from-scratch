# LinkedIn — Day 08: Hybrid Search and Rank Fusion

**Status:** ready to post once the `day-08` branch is merged into `main`, the tag `day-08` is pushed and the GitHub Actions run is green. **Publish first, then post:** the post links to the repository.
**Matches:** `docs/day-08/LEARN.md`, `notebooks/day_08_hybrid_search_and_fusion.ipynb` (outputs from the 2026-10-23 run), `evals/baselines/day-08-retrieval.json`.

---

## Post text

15 Days of RAG from Scratch · Day 8/15: Hybrid search and rank fusion

Day 7 ended with a baseline: BM25 and embeddings each found the answer to 29 of 31 questions, but not the same 29, and each of their 15 misses had a written reason. The obvious next step is to combine them. The questions are how, and what the combination quietly loses.

Day 8 merges BM25 (keywords) and embeddings into one ranking, and measures it on Day 7's same 81 labelled queries, query by query.

Four results worth knowing:

1. Never add the raw scores. BM25 scores run from about 2 to 15, cosine similarities from 0.4 to 0.9. Added together, the top 3 was simply BM25's top 3 for 62 of 71 queries.

2. Reciprocal Rank Fusion uses positions only: each list adds 1/(60 + rank). It fixed all 11 exact values dense retrieval missed ("USD 220", "15 minutes") and the question both retrievers missed.

3. RRF has losses too, and you only see them per query. "New dads" fell from dense's rank 3 to rank 15: BM25 returned 12 unrelated chunks, and two weak votes beat one strong one. And 10 lookups lost first place, because a large BM25 lead becomes just "rank 1".

4. A weighted fusion (min-max scores, 0.7 dense + 0.3 BM25) avoided both losses and closed all 15 of Day 7's misses: every answerable query has its answer in the top 3. But the weight was chosen on dev queries, and on the 15 test questions it ties with RRF (15 of 15 each), so RRF stays as the option that needs no labels.

A correction, prompted by a reader's question on Day 7: were the rules fixed before the test questions were scored? Not quite. Day 7's threshold rule and today's weight rule use only dev queries, but exploratory runs had already printed test-split results before the rules were written down. So those test numbers are a check, not clean held-out proof. Both lessons now say so, and from Day 9 the rule is committed before any run touches the test split.

Also measured: a separate text per retriever (no gain once fused), the cost (BM25 plus fusion under 0.1 ms, next to about 10 ms to embed the question), and one retriever skipping the permission filter: a managers-only chunk entered an employee's fused ranking, at best rank 4.

Same caveat as Day 7: 81 questions written by one person, a baseline to compare against, not a benchmark.

Runs locally: a notebook, a committed baseline and tests. No new packages, no API key.

https://github.com/shashanksuresh18/15-days-of-rag-from-scratch

Day 9: a reranker, on the same 81 queries.

#RAG #LLM #Python #AIEngineering

---

## Images — posting order, alt text and purpose

Upload in this order. LinkedIn's API documentation recommends alt text under 120 characters (maximum 4,086), so each alt text below leads with a short sentence that works on its own; paste the full text where the box allows.

| # | File | Teaches |
|---|---|---|
| 1 | `assets/day-08/slide-01-cover.png` | How RRF turns two misses into a hit, and how four retrieval set-ups compare on the same 81 queries. |
| 2 | `assets/day-08/slide-02-how-rrf-works.png` | Why raw scores can't be added, and how RRF fuses positions instead, worked through on one question. |
| 3 | `assets/day-08/slide-03-what-rrf-throws-away.png` | RRF's two losses (single-retriever finds, confidence) and the weighted fusion that avoids them, with its caveat. |
| 4 | `assets/day-08/slide-04-before-you-ship.png` | Three checks before shipping hybrid search (indexed text, permissions, cost) and the decision for this corpus. |

**Alt text 1 — cover**
> Two rankings, one list: RRF turns two retriever misses into a hit. Top: the question "Will Acme pay for wine with a business dinner?" BM25 ranks the answer chunk (Meals) 6th and dense retrieval 4th, both outside the top 3. Both rankings go into Reciprocal Rank Fusion, which adds 1/(60 + rank) from each list; the hybrid ranking puts Meals 3rd, inside the top 3, because 1/66 + 1/64 beats chunks that only one list ranks high. Bottom: the same 81 labelled queries, four ways, as bars. BM25: 29 of 31 questions in the top 3, 30 of 30 lookups in the top 3, 30 of 30 lookups first. Dense: 29, 19 and 12. RRF: 30, 30 and 20. Weighted fusion: 31, 30 and 30. RRF has no weight to tune, but 10 lookups lose first place; the weighted fusion uses min-max scores, 0.7 dense plus 0.3 BM25, with the weight chosen on dev queries.

**Alt text 2 — how RRF works**
> Fuse positions, not scores: why raw scores can't be added, and RRF worked through. Top: the best score per query on one axis from 0 to 16. BM25 scores span 1.71 to 15.28; cosine similarities span 0.42 to 0.88, a sliver near zero. Added together, BM25 wins: the raw sum's top 3 equals BM25's for 62 of 71 queries. Middle: the formula, score equals the sum over lists of 1 divided by 60 plus the chunk's rank; rank 1 is worth 0.0164, rank 3 0.0159, rank 10 0.0143, not returned 0. The wine question as a table: expense Scope is first in BM25, dense and the hybrid; travel Purpose second in all three; expense Meals, the answer, is 6th in BM25, 4th in dense and 3rd in the hybrid; the VPN installation chunk is 9th, 5th and 4th. Bottom: ties are common, because ranks 2 and 5 score the same as 5 and 2, so ties are broken by chunk id; and k barely matters here, since k from 20 to 1000 gives identical counts on the 81 queries.

**Alt text 3 — what RRF throws away**
> What RRF throws away: both losses come from ignoring the scores. First, agreement beats a single find: for "How much time off do new dads get?" BM25 returns 12 chunks, none about parental leave, and each gets two votes, filling RRF ranks 1 to 12. Dense retrieval's top 3, the parental-leave chunks, get one vote each and land at 13, 14 and 15; the answer is 15th, outside the top 3. Second, how sure a retriever is gets lost: for the lookup "15 minutes", BM25's first chunk, the answer, scores 6.28 against 3.03 for its second, but dense ranks the answer 19th and RRF 2nd; 10 of 30 lookups lose first place this way. Bottom: keep the scores with a weighted fusion: rescale each list to 0 to 1, then 0.7 times dense plus 0.3 times BM25, with the weight chosen by a rule on 46 dev queries; the rule wasn't pre-registered, because exploratory runs had already shown test-split results. Result: 31 of 31 questions in the top 3 and 30 of 30 lookups first. On the 15 test questions RRF and the weighted fusion both find 15 of 15: a tie, and only a check, since those results had been seen while exploring.

**Alt text 4 — before you ship**
> Before you ship hybrid search: three checks and a decision. One, same text for both retrievers? Alone, dense retrieval finds 29 of 31 questions with the chunk's header line and 31 without it, a difference of two questions (p = 0.5); fused, no gain, so keep one text and one pipeline. Two, filter every retriever: with a shared, unfiltered BM25 index, a managers-only chunk enters 8 of alice's fused rankings, shown at rank 4 just below the top 3; ranking low is still a leak. Three, the second retriever is cheap: embedding the question takes about 10 milliseconds, BM25 plus fusion under 0.1 millisecond on 44 chunks, shown as a long bar and a sliver. Decision for this corpus: the weighted fusion, 0.7 dense plus 0.3 BM25, with no miss in the top 3; RRF kept as the alternative that needs no labels; never add raw scores from different retrievers. Next: Day 09 tests a reranker on the same 81 queries.

## Posting options

- **Image post (simplest):** attach the four PNGs in the order above and add the alt text to each.
- **Swipeable document post:** upload `assets/day-08/day-08-carousel.pdf` (the same four slides as a 4-page PDF). Document posts may not offer per-page alt text, so keep the image descriptions in the post body or first comment.

## Pre-posting checklist

- [ ] `day-08` merged into `main`, tag `day-08` pushed, GitHub Actions green on Linux, Windows and macOS. **Publish first, then post.**
- [ ] Open the repository in a private/incognito window and check the README index links to the Day 08 lesson, notebook, baseline, slides and post
- [ ] The correction paragraph is yours to keep or cut (it answers the reader's question on Day 7 publicly); if you cut it, the lessons still carry the correction
- [ ] Numbers in the post (29 of 31, 15 misses, 81 queries, 2 to 15, 0.4 to 0.9, 62 of 71, 11 exact values, rank 3 to 15, 12 chunks, 10 lookups, 0.7, 15 of 15, under 0.1 ms, about 10 ms, rank 4) still match the notebook output
- [ ] Alt text added to all four images
