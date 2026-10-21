# LinkedIn — Day 07: An Evaluation Baseline

**Status:** ready to post once the `day-07` branch is merged into `main`, the tag `day-07` is pushed and the GitHub Actions run is green. **Publish first, then post:** the post links to the repository.
**Matches:** `docs/day-07/LEARN.md`, `notebooks/day_07_evaluation_baseline.ipynb` (outputs from the 2026-10-22 run), `evals/baselines/day-07-retrieval.json`.

---

## Post text

15 Days of RAG from Scratch · Day 7/15: Build an evaluation baseline early

Before improving a RAG system, you need a way to tell whether a change made it better. "It looked better on my three examples" isn't one.

Day 7 builds that baseline: 81 labelled queries, each with who asks, the evidence that answers it and the chunks that hold it, tied to a recorded snapshot of the documents and their permissions.

What the measurements showed:

1. Equal totals hide different failures. BM25 and dense retrieval each put the answer in the top 3 for 29 of 31 questions, but not the same 29: BM25 never finds "new dads" (the policy says caregivers), dense misses the lost laptop. 30 of 31 are found by at least one.

2. Small samples need counts. 29 of 31 is 94%, with a 95% interval from 79% to 98%. Compare question by question, not percentage by percentage.

3. Every miss gets a written reason. 15 misses: 11 exact values for dense ("USD 220"), 2 vocabulary for BM25, 1 answer in a neighbouring section, 1 broader chunk outranking the specific one. Each points at a fix a later day will test.

4. Labels go stale. Two ordinary edits (a stricter password rule, a policy made managers-only) produced 17 label problems. Permissions are part of the ground truth.

5. Thresholds chosen by eye don't hold. Chosen with a rule on old questions and reported on 27 new ones, Day 6's gate could be 0.62 instead of 0.50: it then refuses 6 of 12 unanswerable new questions instead of none. A names rule that was perfect on the old questions wrongly refused 2 answerable new ones.

81 questions written by one person: a baseline to compare against, not a benchmark.

Day 6 follow-up: one real run with gpt-6-luna (8 questions, about USD 0.0005) declined 2 questions the API-free reader got wrong. Details in the Day 6 lesson.

Runs locally: a notebook, versioned eval files and tests. No new packages, no API key.

https://github.com/shashanksuresh18/15-days-of-rag-from-scratch

Day 8: hybrid search, measured on these same 81 queries.

#RAG #LLM #Python #AIEngineering

---

## Images — posting order, alt text and purpose

Upload in this order. LinkedIn's API documentation recommends alt text under 120 characters (maximum 4,086), so each alt text below leads with a short sentence that works on its own; paste the full text where the box allows.

| # | File | Teaches |
|---|---|---|
| 1 | `assets/day-07/slide-01-cover.png` | Two retrievers with the same total fail on different questions; lookups separate them; every miss has a reason. |
| 2 | `assets/day-07/slide-02-labels-and-snapshot.png` | A label is evidence that depends on who asks, belongs to a snapshot of the documents, and comes with a dev/test split. |
| 3 | `assets/day-07/slide-03-every-miss.png` | Hit rate and MRR per retriever, small-sample caution, and the four reasons for the 15 misses with the day that will test a fix. |
| 4 | `assets/day-07/slide-04-thresholds.png` | Why a similarity score can't separate answerable from unanswerable questions, and what measuring Day 06's thresholds found. |

**Alt text 1 — cover**
> Measure before you improve: two retrievers, the same total, different misses. Top: 31 questions as two rows of dots, one per question, in the same order for both retrievers. BM25 finds the answer in the top 3 for 29 of 31, dense for 29 of 31. Three columns at the end are framed and numbered: 1, the lost laptop, only BM25 finds it; 2, new dads, only dense finds it; 3, wine (alcohol), both miss it. 30 of 31 are found by at least one retriever: combine them, Day 08. Middle: 30 lookups, a value typed as the query such as USD 220 or 15 minutes: BM25 finds 30 of 30, dense 19 of 30. Bottom: 15 misses, each with a written reason: exact value 11 (dense), vocabulary 2 (BM25), other section 1 (dense), broader chunk 1 (dense); each points at a fix a later day will test on the same questions.

**Alt text 2 — labels and snapshot**
> A label is evidence: the right answer depends on the documents and on who asks. Top: the question "When do pay rises take effect?" has the evidence "take effect from 1 April", found only in manager-pay-review-guide#overview, which is managers-only. For maria, a manager, the label is that chunk: answerable. For alice, an employee, there is no chunk she may read: the correct answer is "I don't know". Middle, labels belong to a snapshot of 44 chunks (text hash and who may read each): changing the password rule from 14 to 16 characters breaks 2 labels, making the retention policy managers-only flips 4 of alice's labels; 17 label problems, and the notebook stops before measuring. Bottom: dev, 54 questions from Days 01 to 06, seen while building, to tune on (16 answerable, 8 not, 30 lookups); test, 27 questions written today before any retriever saw them, to report on only (15 answerable, 12 not).

**Alt text 3 — every miss**
> Every miss gets a reason: counts first, then the questions behind them. Top, answer in the top 3 and MRR@10: questions with BM25, 29 of 31, MRR 0.85; questions with dense, 29 of 31, 0.90; lookups with BM25, 30 of 30, 1.00; lookups with dense, 19 of 30, 0.56. Note: 94% on 31 questions means 79% to 98%; read counts, not decimals. Middle, 15 misses in the top 3 by reason: exact value, 11, dense on lookups, to be tested on Day 08 (hybrid); vocabulary, 2, BM25 on new dads and wine, Day 08 or 10; other section, 1, dense on the lost laptop, Day 11; broader chunk, 1, dense on wine, Day 09 (reranking). The tags say where a fix will be tested, not promised. Bottom, the wine question, "Will Acme pay for wine with a business dinner?": the answer is "Alcohol is not reimbursed." BM25 ranks it 6th (only "business" matches), dense 4th. A person sees that wine is alcohol; neither retriever does.

**Alt text 4 — thresholds**
> Thresholds, measured: Day 06 chose them by eye; now a rule on dev, reported on test. Top: a dot plot of each question's best retrieval score from 0.4 to 0.9, one row for dev and one for test, filled blue dots for answerable questions and hollow orange dots for unanswerable ones. The two groups overlap: a score isn't a label. Dashed lines mark Day 06's gate at 0.50 and the measured gate at 0.62. Middle: on the test split, gate 0.50 refuses 0 of 12 unanswerable questions; gate 0.62 refuses 6 of 12 and keeps all 15 answerable ones. The rule, written first: the highest gate that loses no answerable dev question. Bottom, the names rule, perfect on dev, wrong on test: on dev it refused Google, France, CEO and Wi-Fi with no false refusals; on test it refused 2 answerable questions: "Tokyo" (the Other cities row applies) and "Should" (the first word of a second sentence).

## Posting options

- **Image post (simplest):** attach the four PNGs in the order above and add the alt text to each.
- **Swipeable document post:** upload `assets/day-07/day-07-carousel.pdf` (the same four slides as a 4-page PDF). Document posts may not offer per-page alt text, so keep the image descriptions in the post body or first comment.

## Pre-posting checklist

- [ ] `day-07` merged into `main`, tag `day-07` pushed, GitHub Actions green on Linux, Windows and macOS. **Publish first, then post.**
- [ ] Open the repository in a private/incognito window and check the README index links to the Day 07 lesson, notebook, `evals/`, slides and post
- [ ] Numbers in the post (81 queries, 29 of 31, 79% to 98%, 15 misses, 17 label problems, 0.62, 6 of 12, 2 answerable refused, USD 0.0005) still match the notebook output and the Day 06 lesson
- [ ] Alt text added to all four images
