# LinkedIn — Day 09: Reranking and Context Selection

**Status:** ready to post once the `day-09` branch is merged into `main`, the tag `day-09` is pushed and the GitHub Actions run is green. **Publish first, then post:** the post links to the repository.
**Matches:** `docs/day-09/LEARN.md`, `notebooks/day_09_reranking_and_context_selection.ipynb` (outputs from the 2026-10-28 run), `evals/preregistration/day-09.md`, `evals/baselines/day-09-retrieval.json`.

---

## Post text

15 Days of RAG from Scratch · Day 9/15: Reranking and context selection

Day 8's hybrid search put the answer in the top 3 for every answerable question. Two problems were left: the order inside the top 3, and five unanswerable questions whose chunks look relevant without answering them.

Day 9 adds a reranker: a small cross-encoder that reads the question and each candidate chunk together and scores the pair. It runs locally (ONNX, 91 MB, Apache-2.0).

This time the rules came first: before anything scored the test questions, a commit fixed how many candidates to rerank, a threshold on the score, and what each needed to be adopted.

Four results worth knowing:

1. Deeper isn't better. On dev questions, reranking more than the top 3 pushed an answer out: given more candidates, the reranker found one it liked more. So the rule picked 3: it can only reorder what the reader already gets.

2. On test questions, it reordered one for the worse: "Do I need a receipt for a USD 40 dinner?" The policy says "every expense over USD 25". That's arithmetic, not word matching. 13 answers first instead of 14, so by the criterion set in advance, it isn't adopted.

3. Relevant isn't answering. The five "similar but wrong" questions score among the answerable ones ("home internet" scores +3.15 against a home-office section about desks and monitors). A threshold chosen on dev refused none of them.

4. A reranker promotes whatever it's given. With one retriever skipping the permission filter (Day 8's leak test), a managers-only chunk went from rank 4 to rank 1 for an employee. Filter every retriever before you rerank.

Where it helped: reranking RRF's top 20 gave first place back to the 10 lookups RRF had demoted, and "new dads" went from rank 15 to 1.

Also measured: cost (30 to 40 ms for 3 candidates, a quarter of a second for 20, against 10 ms to embed the question on a 2-vCPU CPU), deduplication (it dropped a retitled copy but kept a stale version), and a one-chunk-per-document cap (three answers lost).

Same caveat: 81 questions written by one person, and the test split has now been used to decide, so the next decision that needs held-out evidence gets new test questions.

Runs locally: a notebook, the committed rules, a baseline and tests. No new packages, no API key.

https://github.com/shashanksuresh18/15-days-of-rag-from-scratch

Day 10: query transformations, on the same 81 queries.

#RAG #LLM #Python #AIEngineering

---

## Images — posting order, alt text and purpose

Upload in this order. LinkedIn's API documentation recommends alt text under 120 characters (maximum 4,086), so each alt text below leads with a short sentence that works on its own; paste the full text where the box allows.

| # | File | Teaches |
|---|---|---|
| 1 | `assets/day-09/slide-01-cover.png` | How a cross-encoder differs from a bi-encoder, one ranking it changes, and the result measured against criteria written down first. |
| 2 | `assets/day-09/slide-02-what-moved.png` | Every rank the reranker changed, grouped by why: it reads question and chunk together, scores each chunk alone, and doesn't reason. |
| 3 | `assets/day-09/slide-03-close-isnt-answers.png` | Why a relevance score can't tell "close" from "answers": the five similar-but-wrong questions against a gate set on dev. |
| 4 | `assets/day-09/slide-04-before-you-add-one.png` | Three checks before adding a reranker (latency, context selection, permissions) and today's two decisions. |

**Alt text 1 — cover**
> A second look, measured: a cross-encoder reranks the hybrid's candidates. Top: two ways to score a chunk. A bi-encoder (dense retrieval, Day 04) turns the question and the chunk into separate vectors and compares them; it is fast because the chunk vectors are made in advance. A cross-encoder reads the question and the chunk together and outputs one score; it is slow, one model run per pair. Middle: for "How much time off do new dads get?", dense retrieval ranks Returning to work first, Who is eligible second and Leave entitlement, the answer, third; the cross-encoder ranks Leave entitlement first, Who is eligible second and Returning to work third. Bottom: rules committed before the test split. Reranked against the hybrid alone: dev questions ranked first, 15 vs 13; test questions ranked first, 13 vs 14; reranking for later days: not adopted. One test question decided it, and the criterion was written down first.

**Alt text 2 — what moved**
> What moved, and why: every rank the reranker changed, in three groups. One, it reads question and chunk together: "How long is parental leave?" meets "16 weeks" and moves from rank 2 to 1; RRF's 10 demoted lookups, such as "USD 220", go from rank 2 or 3 back to 1; "How much time off do new dads get?", when reranking RRF's top 20, moves from rank 15 to 1. Two, it scores each chunk alone: "How soon must I report a lost laptop?" drops from rank 2 to 5 in RRF's top 20, because the How to report section never says laptop. Three, it doesn't reason: "Receipt for a USD 40 dinner?" drops from 1 to 2, because the rule says "over USD 25"; "Wine with a business dinner?" drops from 3 to 4 in RRF's top 20, because the chunk says "Alcohol". Note at the bottom: the last two need more than a reranker: context from another section, and a reader that knows 40 is more than 25 and wine is alcohol.

**Alt text 3 — close isn't answers**
> Close isn't answers: a gate on the reranker's score doesn't separate the five similar-but-wrong questions. A strip chart of each question's best reranker score, from −12 to +10, in two rows. Dev: most unanswerable questions between −12 and −8, one near −4.6 and one at +4.4 (Google's remote work policy); answerable questions from about +1 to +8, except new dads at −3.56, which sets the gate at −3.6, a dashed vertical line with refused to the left and answered to the right. Test: seven unanswerable questions between −12 and −8, but the five similar-but-wrong ones, marked with a cross, sit at −2.93, −0.62, −0.48, −0.37 and +3.15 (home internet), all above the gate and among the answerable ones; the receipt question, which is answerable, scores −7.38, below the gate. Bottom: the pre-registered test on the test split: the five, 0 refused; answerable, 1 refused (the receipt question); a web-search reranker judges "about", not "answers"; not adopted as a gate; answers are checked on Day 12.

**Alt text 4 — before you add a reranker**
> Before you add a reranker: three checks and the decision. One, every candidate is a model run: on the notebook's 2-vCPU machine, embedding the question took about 10 ms, reranking 3 candidates about 30 ms and reranking 20 about 240 ms, drawn as bars to scale. Two, selecting the context: a copy saved under another title is dropped; an old version with USD 400 instead of 500 is kept, because deduplication isn't freshness; one chunk per document loses 3 answers. Three, it promotes what it's given: with one unfiltered retriever, a managers-only chunk in an employee's pay question goes from rank 4 in RRF to rank 1 after reranking; filter every retriever, then rerank. Decisions, with criteria set in advance: reranking the hybrid's top 3 put 13 test questions first against 14; a gate on its score refused 1 answerable question and 0 of the 5; both not adopted, and the weighted fusion stays. Next: Day 10, query transformations.

## Posting options

- **Image post (simplest):** attach the four PNGs in the order above and add the alt text to each.
- **Swipeable document post:** upload `assets/day-09/day-09-carousel.pdf` (the same four slides as a 4-page PDF). Document posts may not offer per-page alt text, so keep the image descriptions in the post body or first comment.

## Pre-posting checklist

- [ ] `day-09` merged into `main`, tag `day-09` pushed, GitHub Actions green on Linux, Windows and macOS. **Publish first, then post.**
- [ ] Open the repository in a private/incognito window and check the README index links to the Day 09 lesson, notebook, rules, baseline, slides and post
- [ ] Numbers in the post still match the notebook output (top 3 for every answerable question, five questions, 91 MB, top 3, "over USD 25", 13 instead of 14, +3.15, none of them, rank 4 to rank 1, 10 lookups, rank 15 to 1, 30 to 40 ms, a quarter of a second for 20, 10 ms, three answers, 81 questions)
- [ ] Alt text added to all four images
