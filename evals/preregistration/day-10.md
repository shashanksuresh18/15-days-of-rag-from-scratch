# Day 10: rules and settings, committed before the held-out questions are scored

This file was committed on the `day-10` branch as its first commit, together with the new held-out questions
(`evals/questions-v2.jsonl` and `evals/manifest-v2.json`), **before any Day 10 run scored a v1 test-split query or
any v2 test query**, exploratory runs included. The exploration scripts loaded `questions-v1.jsonl`, dropped every
`test` row before building anything (asserted: 54 dev queries left) and never opened `questions-v2.jsonl`. The Day 10
notebook recomputes each rule on dev and refuses to go on (an `assert`) if it doesn't give the value written here.

The commit's timestamp is local, so it shows the order only as far as you trust the author. To have a server
witness it, push this commit (or open a pull request with it) before running anything on the held-out questions.

## What was already known when this was written

- **The v1 test split is used** (`evals/README.md`, rule 8). Before writing this file the author had read its 27
  questions, every published result on it (Days 07 to 09), and the four failures today aims at, all of them
  published: "wine" vs "alcohol" (d07-a08, the answer third in the weighted fusion), "dads" vs "caregivers" (d01-01,
  dev, third), and Day 06's names rule refusing "Tokyo" (d07-a01) and "Should" (d07-a11, the first word of a second
  sentence). Three of the map's entries below were written **from v1 test-split diagnoses**. So every v1 test-split
  number Day 10 reports is **in-sample**: a description of the questions the rules were aimed at, not evidence.
- **The new test questions were written after the rules below were frozen, by the same author, who knew them.**
  The protocol: questions an employee might ask, at least one for most documents, in everyday words, none a
  paraphrase of a v1 question; written in one pass; labels derived from their evidence by Day 07's rule (whole
  words, readable by the asking user); none changed after being written. By reading them (no code was run on them):
  **one** of the 41 contains a phrase from the map (d10-a17, "work from home"); four name a place (Berlin and Paris,
  which are in the gazetteer below, Toronto, which isn't, and Italy); three have two sentences.
- What hadn't been computed: **any Day 10 rule's output, retrieval or refusal, for any v1 test-split query or any
  v2 test query.**

## The data

`questions-v2.jsonl` (manifest `manifest-v2.json`, version 2, same snapshot as v1):

- **dev, 81 queries:** the 81 queries of v1, text and labels unchanged. The 27 v1 test questions are dev from now on
  (rule 4: Day 09 used them to decide); each one's note says so.
- **test, 41 new questions** (`origin: day10-new`): 29 answerable (28 asked by alice, 1 by maria) and 12 not (one of
  them, d10-u08, is answered by a managers-only guide alice may not read).

`questions-v1.jsonl` and `manifest.json` don't change: Day 10 still compares its run on v1's 81 queries with
`evals/baselines/day-09-retrieval.json`, which refers to v1.

## Fixed settings (not tuned)

| Setting | Value |
|---|---|
| Default retriever | Day 08's weighted fusion (α = 0.7 on dense, min-max per retriever and query), over BM25 (k1 = 1.2, b = 0.75) and dense retrieval (`BAAI/bge-small-en-v1.5`), each permission-filtered before fusion, both indexing the chunk text with its `Title > Section` header |
| Refusal comparisons | Day 07's measured dense gate, **0.62** on the dense top score, in front of the names rule (as Day 09's table did); the names rule reads the sources Day 06's pipeline would send (dense retrieval's top 3, assembled) |
| End-to-end description | Day 06's pipeline with Day 07's measured settings (gate 0.62, reader 0.63) |
| Near ties | a rank change where the labelled chunk was within 0.001 of a neighbour at the cut-off is reported apart, as CPU rounding |

## Transformation 1: a reviewed vocabulary map (query expansion)

Each entry links words an asker uses to the words the documents use, and cites where the gap was recorded. An
entry matches a **whole phrase**, case-insensitive. A match **appends** the documents' terms to the query; no word
is ever removed or replaced.

| The asker says | The documents say | Recorded in |
|---|---|---|
| dad, dads, father, fathers, mum, mums, mom, moms, mother, mothers | caregivers | d01-01 (dev): dads vs caregivers; with the singular, a synonym and the counterparts |
| time off | leave | d01-01 (dev): time off vs leave |
| wine, beer, spirits | alcohol | d07-a08 (v1 test, used): wine vs alcohol |
| pay for | reimbursed | d07-a08 (v1 test, used): pay for vs reimbursed |
| work from home, working from home, wfh | remotely, remote | d07-a03 (v1 test, used): work from home vs work remotely |
| virus | infected | d07-a11 (v1 test, used): virus vs infected |
| switch off, switch it off, shut down, shut it down, power off | turn off | d07-a11 (v1 test, used): switch off vs turn off |
| locked | locks, lockout | d01-02 (dev): locked vs locks (BM25 has no stemming) |

Plus **the abbreviations the documents define themselves**, harvested from the chunks with the pattern
`Long form (ABBR)`: on this corpus one, "Multi-factor authentication (MFA)", which gives two entries
(mfa → multi-factor authentication, and multi-factor authentication → mfa).

**Routing:** an exact-value query is never expanded. *Exact* = no `?`, at most 4 words, and a digit, `@`, `#` or
`://` in it. On dev this marks all 30 lookups and none of the 24 questions.

## Rule 1: where the expansion goes

*Two variants: expand only BM25's query (**lexical**), or BM25's and dense retrieval's (**both**). On dev (the 16
answerable dev questions and the 30 lookups, 46 in all), choose the variant that puts a labelled chunk in the
weighted fusion's top 3 for the most queries, then first for the most, then the higher MRR@10; a tie goes to
**lexical** (it changes one input instead of two).*

Computed on dev only (the weighted fusion):

| Variant | top 3 (of 46) | first | MRR@10 | changed on dev |
|---|---|---|---|---|
| no expansion | 46 | 43 | 0.9601 | |
| lexical | 46 | 44 | 0.9746 | d01-01: 3 → 1 |
| both | 46 | 44 | 0.9746 | d01-01: 3 → 1 |

A tie on every count: **the rule chooses lexical.** The new retrievers are named `BM25+map` (BM25 on the expanded
query) and `weighted+map` (the weighted fusion of `BM25+map` and dense retrieval on the original question).

## Transformation 2: the names rule, as query analysis

Day 06's reader (and Day 07's refusal table) abstains when a capitalised word or acronym in the question, other
than its first word, appears in none of the sources. The new rule changes two things, one for each recorded failure:

1. **Every sentence's first word is skipped**, not only the question's (Day 06's `SENTENCE` split). For "Should".
2. **Places are typed.** A stand-in gazetteer, chosen by an external rule so that no place was picked because a
   question mentions it: **the G20's member countries and their capitals** (Argentina, Australia, Brazil, Canada,
   China, France, Germany, India, Indonesia, Italy, Japan, Mexico, Russia, Saudi Arabia, South Africa, South Korea,
   Turkey, United Kingdom, United States; Buenos Aires, Canberra, Brasilia, Ottawa, Beijing, Paris, Berlin, New Delhi,
   Jakarta, Rome, Tokyo, Mexico City, Moscow, Riyadh, Pretoria, Seoul, Ankara, London, Washington). Places are found
   as whole phrases first. A name counts as known if a source mentions it (whole word or phrase, case-insensitive),
   **or** if it is a gazetteer place and a source mentions its type word (`city`/`cities`, `country`/`countries`):
   an instance of a category the sources cover ("Use the Other cities row for any city not listed"). For "Tokyo".

Everything else is Day 06's: the `NAME` pattern, the sources it reads, and abstaining when a name is unknown.

Computed on dev only (24 questions, with the dense gate 0.62 in front): Day 06's rule keeps 16 of 16 answerable and
refuses 8 of 8 unanswerable; **the new rule gives the same, 16 of 16 and 8 of 8** (dev has no two-sentence question,
and its only gazetteer place, France, stays unknown).

## A guard for any rewriter (fixed; shown on hand-written examples)

`check_rewrite(original, rewritten)` lists what a rewrite broke: a value of the original (number, amount, email,
URL, `#channel`) missing from the rewrite; a value or a name in the rewrite that the original doesn't have; a name
of the original missing; an empty rewrite. A rewrite with any problem is discarded and the original query is used.
It isn't tuned; it is shown on rewrites written by hand (labelled as such), not on a model's output.

**No language-model rewrite is run today.** It would need a model call per query, can't run in CI, and would have
to beat the map on these same rules to earn its place. Exercises show where it would plug in, behind the guard.

## What will be reported, whatever it shows

1. Day 09's six retrievers against `evals/baselines/day-09-retrieval.json` on v1's 81 queries (expected: no changes).
2. **v1, 81 queries:** `BM25+map` against BM25 and `weighted+map` against the weighted fusion, both as recorded in the
   Day 09 baseline, query by query, first place and top 3, with the sign test, in three parts: the 16 answerable dev
   questions, the 15 answerable v1 test questions (**used, and in-sample** for the map), and the 30 lookups. Every
   query whose rank changes, with a reason.
3. **v2 test, the held-out comparison:** `weighted+map` against the weighted fusion, and `BM25+map` against BM25, on
   the 29 answerable new questions, query by query, first place and top 3, with the sign test and Wilson intervals.
   Every v2 test miss (not in the top 3) of BM25, dense, the weighted fusion and both new retrievers, with a diagnosis.
4. **The names rule**, Day 06's and the new one, each behind the dense gate 0.62: answerable kept and unanswerable
   refused on dev (24), on the v1 test split (27, in-sample) and on the v2 test split (41); every question whose
   decision differs between the two rules, by name.
5. **End to end** (a description): Day 06's pipeline at Day 07's measured settings, with each names rule, on every
   split: answerable correct / wrong / refused, unanswerable refused / answered; every question whose outcome changes.
6. **When not to rewrite** (descriptions; they decide nothing): (a) the same map matched word by word instead of as
   phrases, on every split; (b) the map plus one more entry, "dinner, lunch, breakfast → meals", on every split;
   (c) how many lookups the routing keeps unexpanded (expected: all 30) and how many queries the map touches at all,
   per split; (d) unanswerable questions whose top chunk in the weighted fusion changes with the map; (e) the guard
   on the hand-written rewrites.
7. The cost: the expansion's median time per query, next to embedding the question.

## Decisions, and what each needs

- **D1, the map in the default retriever:** later days build on `weighted+map` if, on the 29 answerable v2 test
  questions, it loses **no** question against the weighted fusion (none drops from first place, none drops out of the
  top 3; near ties set apart) **and gains at least one** (first place or the top 3). Otherwise the weighted fusion stays
  the default and the map is recorded as measured, not adopted.
- **D2, the new names rule:** adopted (recorded for later days) if, on the 41 v2 test questions with the dense gate
  0.62 in front, it keeps **at least as many** of the 29 answerable questions as Day 06's rule, refuses **at least as
  many** of the 12 unanswerable ones, and is **better on at least one** of the two counts. Otherwise it is recorded as
  measured, not adopted.

Day 06's pipeline and `answering.py` don't change today: an adopted setting is recorded in the baseline for a later
day to switch on and re-measure. The new baselines are `evals/baselines/day-10-retrieval.json` (v1, comparable with
Day 09's) and `evals/baselines/day-10-retrieval-v2.json` (v2, the reference for later days).
