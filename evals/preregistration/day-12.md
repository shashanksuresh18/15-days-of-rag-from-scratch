# Day 12: rules and settings, committed before the held-out questions are scored

This file was committed on the `day-12` branch as its first commit, together with the new held-out questions
(`evals/questions-v4.jsonl` and `evals/manifest-v4.json`), **before any Day 12 run scored a v4 test question**,
exploratory runs included. The exploration scripts loaded `questions-v3.jsonl` only (all 190 of its queries count as
dev today, see below) and refused to start if a `questions-v4.jsonl` existed. The rules below were frozen first (a
local scratch file, 2026-11-11 07:53:09 UTC); the new questions were written after that, and their labels were derived
from their evidence by Day 07's rule, with no retriever, reader or pipeline run on them. The Day 12 notebook recomputes
every dev value below and refuses to go on (an `assert`) if it gets another one.

The commit's timestamp is local, so it shows the order only as far as you trust the author. To have a server witness
it, push this commit (or open a pull request with it) before running anything on the held-out questions.

## What was already known when this was written

- **The v3 test split is used** (`evals/README.md`, rule 8): its 68 questions decided Day 11's decision. The author had
  read them and every published result on them, including Day 11's end-to-end description: the names rule refused 18
  of the 22 refused answerable v3 test questions, 16 of them for the first word of a second sentence ("How", "Can").
- **Everything below was computed on dev, which is all 190 v3 queries (160 questions), before freezing**: the
  current pipeline's answers, the candidate's answers and the rubric's verdicts. The author read every dev answer
  that either pipeline gives. So every dev number here is **in-sample**.
- **The candidate isn't new.** Day 10 measured the same names rule on v2's test split and didn't adopt it (it had
  to lose nothing, and one unanswerable question was let through). It was recorded "for a later day to revisit with
  new questions"; today revisits it with answers graded instead of refusals counted.
- **The new test questions were written after the rules were frozen, by the same author, who knew them**, knows how
  Day 06's names rule fails (a capitalised word missing from the sources sent, including a sentence's first word) and
  knows the documents. The protocol: questions an employee might ask, in everyday words, none a paraphrase of a v1, v2
  or v3 question (the same rule asked about a new case is allowed); at least three answerable questions for every
  document alice may read and three for the managers' guide, asked by maria; about a third with two sentences and
  about one in seven naming a place, a product or a day, close to v3's shares (24 of 68 had two sentences), so that
  the test set doesn't lean towards the change; about a third unanswerable (v3: 16 of 68), because refusals are half of
  what an answer evaluation measures; written in one pass; labels derived from their evidence by Day 07's rule (whole
  words, readable by the asking user); none changed after being written. By reading them (no code was run on them):
  **26** of the 76 have two sentences, **11** name a place, a product or a day (Chrome, Slack, Singapore twice,
  London, Madrid, Tokyo, CCTV, WhatsApp, Canada, Saturday), **18** describe a situation first. Two facts are known
  weak spots of the grader and say so in their notes: d12-a24's "6 weeks" is also inside "16 weeks", and d12-a13's
  "5 business days" is in two rows of the table.
- What hadn't been computed: **anything for any v4 test question**: no ranking, no context, no answer, no grade.

## The data

`questions-v4.jsonl` (manifest `manifest-v4.json`, version 4, the same snapshot as v1, v2 and v3):

- **dev, 190 queries:** the 190 queries of v3, text and labels unchanged. v3's 68 test questions are dev from now on
  (rule 4: Day 11 used them to decide); each one's note says so. Of the 190, **160 are questions** (112 answerable,
  48 not) and 30 are lookups (an end-to-end answer is graded for questions only).
- **test, 76 new questions** (`origin: day12-new`): **52 answerable** (49 asked by alice, 3 by maria) and **24 not**
  (two of them, d12-u18 and d12-u19, are answered by the managers-only guide alice may not read; one, d12-u20, is
  asked by maria).

The earlier files don't change: Day 12 compares its retrieval run on v3's 190 queries with
`evals/baselines/day-11-retrieval-v3.json`, which refers to v3.

## The current pipeline: the no-change baseline

Day 11's end-to-end path, unchanged (`src/rag_from_scratch/contexts.py`, promoted today from the Day 11 notebook):

| Step | Setting |
|---|---|
| Retrieve | Day 08's weighted fusion (α = 0.7 on dense, min-max per retriever and query) over BM25 and dense retrieval (`BAAI/bge-small-en-v1.5`), each permission-filtered before fusion |
| Context | the weighted fusion's top 3 (Day 11's section expansion wasn't adopted), Day 06's 300-word budget |
| Gate | Day 07's measured gate: refuse without reading if the **dense** top score is under 0.62 |
| Reader | Day 06's extractive reader at Day 07's measured threshold (0.63), with **Day 06's names rule** |
| Check | Day 06's citation check: a reply that fails it is blocked (refused) |

On dev it reproduces Day 11's published end-to-end numbers exactly (v2's 122 queries: 45 correct / 6 wrong / 9 refused
answerable, 21 refused / 11 answered unanswerable; v3's test questions: 24 / 6 / 22 and 8 / 8).

## The candidate: Day 10's names rule

The same pipeline with **Day 10's names rule** (`unknown_names_v2`, promoted today to `queries.py` with Day 10's
`NamesReader`): every sentence's first word is skipped, and a gazetteer place (the G20 countries and capitals) counts
as known when a source mentions its type ("Use the Other cities row for any city not listed"). Nothing else changes.

## The rubric

Every question gets one verdict per pipeline, computed from its answer, its labels and its facts:

| Part | Rule |
|---|---|
| Outcome | Day 07's `outcome()`: refused (abstained or blocked), or for an answerable question correct (every fact is a case-insensitive substring of the answer) or wrong; for an unanswerable one, answered |
| Refused by | the step that refused: the gate, the names rule, the reader's threshold or the citation check (Day 11's `why_refused()`) |
| Context | whether a labelled chunk was among the sections sent (for a gate refusal: the sections that would have been sent) |
| Citations | whether the answer cites a labelled chunk; citation precision = cited chunks that are labelled / cited chunks |
| Grounded | the citation check passed and every sentence has at least 0.75 of its content words (lower case, three or more letters or a digit, not a stop word) in the sources it cites. A lexical stand-in for "every claim is supported", stated before any model was graded; it can't tell meaning |
| Stage | where an answer went wrong, in this order: for an answerable question refused: *gate*, *names rule: a sentence's first word* (every unknown "name" is the first word of a sentence after the first), *names rule: a name*, *reader threshold*, *citation check*; wrong: *retrieval: answer not in the context*, *reader: answer in the context, not used* (cites no labelled chunk), *reader: right source, wrong part* (cites a labelled chunk, facts missing); for an unanswerable question answered: *answered: restricted to others* (the answer is in a chunk this user may not read) or *answered: nothing answers it* |

Its limits, known before any v4 answer was seen: containment passes a list of table rows that includes the right one
(Tokyo, Berlin on dev) and a fact inside a longer one ("6 weeks" in "16 weeks"); it fails a correct paraphrase ("14
days" for "two weeks"); "grounded" is lexical. Every limit is reported with the author's reading of the answers
(below).

**Value of an answer** (for the decision): answerable correct **+1**, refused **0**, wrong **−2**; unanswerable refused
**0**, answered **−2**. A wrong answer costs twice what a correct one gains. That is an assumption, not a measurement:
Day 06's design already prefers refusing to guessing (its citation check fails closed), and a wrong policy answer can
make someone miss a deadline or break a security rule, while a refusal sends them to the policy owner. Costs of 1 and
3 are reported next to it.

**Near ties.** Every answer has a *closest call*: the smallest of the gaps that decided it, among the dense top
score and the gate, the fused scores at ranks 3 and 4 (which sections are sent), the best sentence and the reader's
threshold, the best sentence and the best one from another source (which source is copied), and each sentence of the
copied source and the reader's margin (whether it is copied too). A question whose outcome differs between the two
pipelines is set apart, as CPU rounding, if either pipeline's answer has a closest call under 0.001.

## Computed on dev (in-sample)

| | Answerable (112): correct / wrong / refused | Unanswerable (48): refused / answered | Value, cost 2 |
|---|---|---|---|
| current pipeline | 69 / 12 / 31 | 29 / 19 | 7 |
| candidate (Day 10's names rule) | 84 / 18 / 10 | 28 / 20 | 8 |

Question by question: the candidate is better for **15** (refused → correct) and worse for **7** (6 refused → wrong,
1 unanswerable refused → answered), sign test p = 0.13; the value changes by **+8** at cost 1, **+1** at cost 2 and
**−6** at cost 3. The current pipeline's 31 refused answerable questions: names rule 25 (19 for a sentence's first
word), gate 3, reader threshold 2, citation check 1. Its 12 wrong answers: 8 *reader: answer in the context, not
used*, 2 *reader: right source, wrong part*, 2 *retrieval*. Nothing is tuned today: no setting was chosen on dev.

## What will be reported, whatever it shows

1. Day 11's six retrievers (BM25, dense, the weighted fusion, `BM25+map`, `weighted+map`, `weighted+neighbours`)
   against `evals/baselines/day-11-retrieval-v3.json` on v3's 190 queries (expected: no changes), and Day 11's
   end-to-end numbers reproduced by the current pipeline.
2. **The rubric on dev (in-sample)** for the current pipeline: outcomes with Wilson intervals, refusals by step, stages,
   citations and groundedness; the recorded failures it starts from (Day 06's "right numbers, wrong meaning", Day 11's
   answers that ignore the answer section, the names rule's second-sentence refusals) and how it classifies them.
3. **The rubric against the author's reading** of every answered dev question (both pipelines): agreement, and every
   disagreement with its reason. The reading is committed (`evals/reviews/day-12-answers.jsonl`), tied to the exact
   answer text, so a changed answer makes its review stale.
4. **Dev (in-sample)**: the candidate against the current pipeline, question by question, with the sign test and the
   value at costs 1, 2 and 3.
5. **v4 test, the held-out comparison:** for both pipelines, the rubric's counts on the 76 new questions with Wilson
   intervals; question by question, every changed outcome with the sign test, near ties set apart; the value at cost 2
   (the decision) and at costs 1 and 3; every new question's stage under both; and as a description, BM25, dense and
   the weighted fusion's hit@1/3/5/10 and MRR@10 on the 52 answerable new questions.
6. The author's reading of every answered v4 test question, **written after the run** and marked so: a description of
   the grader's agreement on new answers, never used to decide.
7. **The regression suite**: every failure recorded in the series turned into a test (`tests/test_regressions.py`,
   cases in `evals/regressions.jsonl`), with what each pins: a fixed failure that must stay fixed, a property that must
   always hold, or a known failure pinned exactly so that any change is noticed.
8. The cost: the rubric's own time per answer, next to the pipeline's.

## Decision, and what it needs

- **D1, Day 10's names rule in the current pipeline:** later days use it if, on the 76 v4 test questions, the
  candidate's total value at cost 2 is **higher** than the current pipeline's (the sum over questions of the change in
  value is above 0), counting only questions whose outcome changed and isn't a near tie. Otherwise Day 06's names rule
  stays, and the candidate is recorded as measured, not adopted.

`answering.py` doesn't change today: an adopted rule is recorded for later days to switch on.

## The optional live run (a description, never a decision)

Off by default (`DAY12_LIVE_MODEL=1` or `LIVE_MODEL = True` switches it on); the tests never run it. One pass with the
model configured in `.env` (`RAG_CHAT_*`, Day 06's client), fed the current pipeline's contexts: the gate refuses
without calling the model, and a reply is checked by Day 06's citation check. At most **40 requests**: the six
recorded failures d01-01, d03-13, d10-a12, d10-a24, d11-a06 and d11-a40 (dev), then the first 24 answerable and the
first 10 unanswerable v4 test questions, in file order, that the gate lets through. Graded with the same rubric; the
author's reading of its answers is added after it has run. One run of one model: an observation, not a measurement.
