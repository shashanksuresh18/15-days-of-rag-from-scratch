# LinkedIn — Day 12: End-to-End Answer Evaluation and Regression Tests

**Status:** ready to post once the `day-12` branch is merged into `main`, the tag `day-12` is pushed and the GitHub Actions run is green. **Publish first, then post:** the post links to the repository.
**Matches:** `docs/day-12/LEARN.md`, `notebooks/day_12_answer_evaluation_and_regression_tests.ipynb` (outputs from the 2026-11-11 run), `evals/preregistration/day-12.md`, `evals/questions-v4.jsonl`, `evals/baselines/day-12-answers-v4.json`, `evals/reviews/day-12-answers.jsonl`, `evals/regressions.jsonl`.

---

## Post text

15 Days of RAG from Scratch · Day 12/15: End-to-end answer evaluation and regression tests

On 52 new held-out questions, the retriever had the answer in its top 3 every time. The pipeline answered 30 of them correctly.

A retrieval metric stops at the context. People read answers. So Day 12 grades the answers with a written rubric: correct against each question's facts, refused or not (and was that right), grounded in what it cites, citing the right source, and the stage where it was lost.

The rules were committed first, with 76 new test questions written before anything ran on them.

Four results worth knowing:

1. The stage says what to fix. Of the 22 lost answers, a refusal rule threw away 17: it read the first word of a second sentence ("Where do I…") as an unknown name. The reader lost 3, the citation check 2. Retrieval lost none.

2. Grounded isn't correct. Asked how much leave new dads get, a real model replied "Secondary caregivers receive 6 weeks", citing the right policy. Every word is in the source, and it's still wrong: the policy is gender-neutral. A groundedness score passes it.

3. Grade the grader. The rule-based grader agreed with a careful reading of the answers 177 times out of 180. The 3 misses were lists of table rows that include the right one. It works because this reader copies the documents' words; a model that paraphrases needs checking again.

4. Fixing one rule exposes the next failure. The fixed rule turned 10 refusals into correct answers and added 4 wrong ones. With a wrong answer costing twice a correct one (set in advance), that's +2: adopted, by the smallest margin, and only because a near tie was set apart. Counted, it's a draw. The new wrong answers come from the reader.

And every failure this series has recorded since Day 1 is now a regression test: 29 run on every push. Fixed ones must stay fixed; known ones fail loudly if they change.

Same caveat: small sets written by one person, and one reader of the answers.

Runs locally, no API key: a notebook, the committed rules, 76 new labelled questions, the reading of every answer, and the regression tests.

https://github.com/shashanksuresh18/15-days-of-rag-from-scratch

Day 13: tracing, latency and cost.

#RAG #LLM #Python #AIEngineering

---

## Images — posting order, alt text and purpose

Upload in this order. LinkedIn's API documentation recommends alt text under 120 characters (maximum 4,086), so each alt text below leads with a short sentence that works on its own; paste the full text where the box allows.

| # | File | Teaches |
|---|---|---|
| 1 | `assets/day-12/slide-01-cover.png` | Why answers need their own grade: retrieval found all 52 held-out answers, the pipeline answered 30 correctly, and the stage of each loss says what to fix. |
| 2 | `assets/day-12/slide-02-the-rubric.png` | The rubric's six parts, and why grounded and correct must be graded apart (four replies to the new-dads question). |
| 3 | `assets/day-12/slide-03-the-decision.png` | How a fix is judged by answers: Day 10's names rule on 76 new questions, a value per answer set in advance, and the near-tie caveat. |
| 4 | `assets/day-12/slide-04-regression-tests.png` | How recorded failures become regression tests (fixed, guard, known), and what to check before trusting an answer metric. |

**Alt text 1 — cover**
> Found isn't answered: retrieval found every held-out answer, the pipeline answered 30 of 52 correctly. Top: 52 new questions, the current pipeline; bars: answer in the top 3, 52; answered correctly, 30. Retrieval found every answer; users got 30 right ones. Middle: where the other 22 were lost, by stage: names rule, a second sentence's first word, 12 ("Where…" read as a name); names rule, a name, 5 (Chrome, Slack, Madrid…); the reader, 3 (copied another sentence); the citation check, 2 (a copied "1." split off). Retrieval lost none of them. Bottom: a grade that names the stage says what to fix: here, a refusal rule first, then the reader. A retrieval metric stops at the context; people read answers.

**Alt text 2 — the rubric**
> Grounded isn't correct: one grade, six parts, and four replies to the new-dads question. Top, the rubric: correct (every fact in the answer); refused by (gate, names, reader, check); context (was the answer sent?); cited (does it cite the answer?); grounded (words in the cited source); stage (where it was lost). Middle: "How much time off do new dads get?", four replies with ticks for grounded, cites it, correct: the reader, which copies a phased return: grounded yes, cites it no, correct no. A faithful paraphrase, 16 weeks and 6 weeks: yes, yes, yes. gpt-6-luna on Day 06, "Secondary caregivers: 6 weeks": yes, yes, no. A changed number, 18 weeks: blocked, never shown. The policy is gender-neutral: a dad who is the main carer gets 16 weeks. Bottom: a groundedness score passes 3 of 4, and two of those are wrong; keep correct and grounded apart. Grade the grader: 177 of 180 agree with a careful reading; the 3 misses are lists of table rows that include the right one.

**Alt text 3 — the decision**
> Fix one rule, meet the next: Day 10's names rule judged by answers on 76 new questions. Top: a sentence's first word read as a name. "It's my first day. Where do I pick up my laptop?" Day 06's rule: no source mentions "Where", refused. Day 10's rule: "Collect your laptop and security badge…". Middle: 15 outcomes changed of 76: 10 refused to correct (green dots), 1 refused to wrong, 3 not answerable to answered (red dots), 1 near tie set apart (grey). The new wrong answers are the reader's: the old rule hid them. Bottom: value, set in advance: correct +1, refused 0, wrong −2. 10 × (+1) + 4 × (−2) = +2: adopted, by the smallest margin; counting the near tie, +0. If wrong costs 1: +6; if it costs 3: −2. A decision needs a cost for wrong answers, stated first. Next to fix: the reader.

**Alt text 4 — regression tests**
> Every recorded failure, a test: 29 cases from Days 1 to 11, run on every push. Top, three cards: fixed, 9, must stay fixed (new dads: in the top 3); guard, 13, must always hold (no managers-only chunk for alice); known, 7, pinned as it fails (a copied "1." is blocked). Middle: how a known case works: the failure still happens, the test passes; it stops or changes shape, the test fails and says why; fixed on purpose? Move it to fixed; never delete it. Bottom: before you trust an answer metric: grade refusals too (refusing everything has no wrong answers); name the stage (so the grade says what to fix); check the grader against people, per generator; state the cost of a wrong answer against a refusal first; pin failures as tests (fixed, guard or known).

## Posting options

- **Image post (simplest):** attach the four PNGs in the order above and add the alt text to each.
- **Swipeable document post:** upload `assets/day-12/day-12-carousel.pdf` (the same four slides as a 4-page PDF). Document posts may not offer per-page alt text, so keep the image descriptions in the post body or first comment.

## Pre-posting checklist

- [ ] `day-12` merged into `main`, tag `day-12` pushed, GitHub Actions green on Linux, Windows and macOS. **Publish first, then post.**
- [ ] Open the repository in a private/incognito window and check the README index links to the Day 12 lesson, notebook, rules, new questions, answer baseline, regression cases, slides and post
- [ ] Numbers in the post still match the notebook output (52 of 52 in the top 3, 30 correct, 22 lost: 17 / 3 / 2, 177 of 180, 76 new test questions, 10 correct and 4 wrong, +2 at cost 2, 29 regression cases)
- [ ] "A careful reading" refers to the reviews in `evals/reviews/day-12-answers.jsonl`: skim a few before posting, since the post vouches for them
- [ ] Alt text added to all four images
