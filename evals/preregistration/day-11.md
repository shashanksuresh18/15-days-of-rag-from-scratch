# Day 11: rules and settings, committed before the held-out questions are scored

This file was committed on the `day-11` branch as its first commit, together with the new held-out questions
(`evals/questions-v3.jsonl` and `evals/manifest-v3.json`), **before any Day 11 run scored a v3 test question**,
exploratory runs included. The exploration scripts loaded `questions-v2.jsonl` only (all 122 of its queries count as
dev today, see below), refused to start if a `questions-v3.jsonl` existed, and no v3 file existed while they ran. The
rules below were frozen first (a local scratch file, 2026-11-09 09:22:51 UTC); the new questions were written after
that. The Day 11 notebook recomputes the rule on dev and refuses to go on (an `assert`) if it doesn't give the values
written here.

The commit's timestamp is local, so it shows the order only as far as you trust the author. To have a server witness
it, push this commit (or open a pull request with it) before running anything on the held-out questions.

## What was already known when this was written

- **The v2 test split is used** (`evals/README.md`, rule 8): its 41 questions decided Day 10's two decisions. Before
  writing this file the author had read them and every published result on them, including the two misses today aims
  at: the stolen phone (d10-a24, "My phone was stolen on a business trip. Who should I tell?", the answer in
  *How to report*, 6th in the weighted fusion, behind *Safety while travelling*, the three other incident sections and
  *Claiming travel costs*) and
  the lockout question (d10-a12, the answer in *Account lockout*, 5th; a vocabulary gap, but the answer sits next to
  *Resetting your password*, which is 2nd). Day 03's lost laptop (d03-13) has been in the weighted fusion's top 3
  (3rd) since Day 08. The rule below was computed on dev, which contains all three, and **its choice comes from one of
  them** (d10-a24: 90 against 89). So every dev number Day 11 reports is **in-sample**: a description of the questions
  the pattern was aimed at, not evidence.
- **The new test questions were written after the rules were frozen, by the same author, who knew them**, knew the
  shape the pattern targets (the situation in one section, what to do in another), and knows the documents' section
  order (it is in the files). The protocol: questions an employee might ask, at least four answerable ones for every
  document alice may read and two for the managers' guide (asked by maria), about half of them describing the situation
  first ("I'm still in my probation period. Can I work remotely?"), in everyday words, none a paraphrase of a v1 or v2
  question; written in one pass; labels derived from their evidence by Day 07's rule (whole words, readable by the
  asking user); none changed after being written. By reading them (no code was run on them): **25** of the 68 describe
  the situation first (tag `situation`), **24** have two sentences, **2** answerable ones have two answer chunks
  (d11-a06 and d11-a49: two policies say so), and **7** contain a phrase from Day 10's map (d11-a05, a20, a29, a31,
  a36, a38 and the unanswerable d11-u06; the map isn't used today).
- What hadn't been computed: **any Day 11 rule's output, and any retriever's ranking, for any v3 test question.**

## The data

`questions-v3.jsonl` (manifest `manifest-v3.json`, version 3, the same snapshot as v1 and v2):

- **dev, 122 queries:** the 122 queries of v2, text and labels unchanged. v2's 41 test questions are dev from now on
  (rule 4: Day 10 used them to decide); each one's note says so. Of the 122, **90 are answerable**: 31 v1 questions,
  30 lookups and 29 v2 questions.
- **test, 68 new questions** (`origin: day11-new`): **52 answerable** (50 asked by alice, 2 by maria) and **16 not**
  (one of them, d11-u13, is answered by the managers-only guide alice may not read).

The earlier files don't change: Day 11 compares its run on v2's 122 queries with
`evals/baselines/day-10-retrieval-v2.json`, which refers to v2.

## Fixed settings (not tuned)

| Setting | Value |
|---|---|
| Retriever | Day 08's weighted fusion (α = 0.7 on dense, min-max per retriever and query) over BM25 (k1 = 1.2, b = 0.75) and dense retrieval (`BAAI/bge-small-en-v1.5`), each permission-filtered before fusion, both indexing the chunk text with its `Title > Section` header. Day 10's map isn't used (not adopted) |
| Hits | the weighted fusion's top 3 (Day 06's `TOP_K`) |
| Budget | Day 06's 300 words, whole chunks; a chunk that doesn't fit is skipped and the next one tried (Day 06's `assemble()`). The top 3 always fits: no chunk has more than 75 words |
| Sections | each document's sections in their order in the document (Day 03's `section_chunks()`); a section's neighbours are the sections just before and just after it in the same document |
| Permissions | every added section is checked against the asking user's groups before it is sent. Finding a neighbour is a lookup, not a search, so the search's permission filter doesn't apply to it |
| Near ties | a difference is set apart, as CPU rounding, if the fused scores at a cut-off that decides it are within 0.001: ranks 3 and 4 (which sections are the hits), and, for the control, ranks *m* and *m* + 1 |
| End-to-end description | Day 07's measured dense gate (0.62, on the dense top score), Day 06's extractive reader at Day 07's measured threshold (0.63) with Day 06's names rule, and Day 06's citation check |

## The pattern: section expansion

The weighted fusion's three hits come first, in rank order. Then, for hit 1, hit 2 and hit 3 in turn, the sections the
setting adds around it. A section already sent is skipped, so is one the user may not read, and so is one that
doesn't fit the budget (the next one is tried). Settings:

| Setting | Adds around each hit |
|---|---|
| `off` | nothing: the three hits only, exactly the weighted fusion's top 3 (the **off-switch**) |
| `previous` | the section just before it |
| `next` | the section just after it |
| `neighbours` | the section just before it, then the one just after it |
| `parent` | the hit's other sections, in their order in the document |

Every query gets a **trace**: each section considered, why (hit *k* with its fused score, or previous, next or parent
of hit *k*), what happened to it (sent, already sent, not readable, over the budget), and the words used.

## Rule 1: the setting

*On dev's 90 answerable queries, choose the setting whose context holds a labelled chunk for the most queries; then
the one that sends the fewest words on average (to 0.1); then the earlier of previous, next, neighbours, parent.*

Computed on dev only:

| Setting | labelled chunk in the context (of 90) | mean words sent |
|---|---|---|
| off | 88 | 115.6 |
| previous | 89 | 182.2 |
| next | 89 | 179.1 |
| neighbours | 90 | 223.1 |
| parent | 89 | 245.6 |

**The rule chooses neighbours.** The context it gives is named `weighted+neighbours`.

## The control: the same number of sections, from further down the ranking

An expansion sends more text, and more text holds more answers whatever it is. So the expansion is compared not only
with the top 3 but with a **control** that sends as much from the ranking itself: for each query, the weighted
fusion's top *m*, where *m* is the number of sections `weighted+neighbours` sent for that query. On dev it sends about
as many words (mean: 207.4 against 209.8 on the v1 questions, 254.3 against 266.3 on the lookups, 207.8 against
207.1 on the v2 questions).

Computed on dev (in-sample; no near ties among these):

| Part | `weighted+neighbours` vs the top 3 | `weighted+neighbours` vs the control |
|---|---|---|
| v1 questions (31) | 31 vs 31; no difference | 31 vs 31; no difference |
| lookups (30) | 30 vs 30; no difference | 30 vs 30; no difference |
| v2 questions, used (29) | 29 vs 27; only `weighted+neighbours`: d10-a12, d10-a24 (p = 0.5) | 29 vs 28; only `weighted+neighbours`: d10-a24 (p = 1) |

## What will be reported, whatever it shows

1. Day 10's five retrievers (BM25, dense, the weighted fusion, `BM25+map`, `weighted+map`) against
   `evals/baselines/day-10-retrieval-v2.json` on v2's 122 queries (expected: no changes).
2. The off-switch: `off` sends exactly the weighted fusion's top 3, for every query.
3. **Dev, 122 queries (in-sample):** `weighted+neighbours` against the weighted fusion's top 3 as recorded in
   `day-10-retrieval-v2.json`, and against the control, query by query, with the sign test, in three parts (the 31 v1
   questions, the 30 lookups, the 29 v2 questions); the words sent.
4. **v3 test, the held-out comparison:** on the 52 answerable new questions, how many have a labelled chunk in the
   context of the top 3, of `weighted+neighbours` and of the control, with Wilson intervals; `weighted+neighbours`
   against the top 3 and against the control, query by query, with the sign test, near ties set apart; every new
   question whose labelled chunk is missing from any of the three contexts, with a diagnosis; Day 10's five
   retrievers on the 68 new questions (hit@1/3/5/10, MRR@10) as a description; the words sent.
5. The traces, for every query (saved with the run), and printed for the stolen phone.
6. **Permissions:** restricted chunks in any context (every strategy, every query); and a demonstration on an
   in-memory copy of the records with one section restricted to managers: what alice is sent by an unchecked lookup,
   and by the checked expansion.
7. Unanswerable questions: the words each strategy sends to the reader.
8. **End to end** (a description): Day 06's steps with the settings above, fed each of the three contexts, on dev and
   on the v3 test split: answerable correct / wrong / refused, unanswerable refused / answered; every question whose
   outcome changes between contexts.
9. The cost: the mean and largest number of words sent per strategy, and the expansion step's median time next to
   embedding the question.

## Decision, and what it needs

- **D1, section expansion in the default context:** later days send `weighted+neighbours`' context if, on the 52
  answerable v3 test questions, it holds a labelled chunk for **at least one** question that the control's context
  doesn't, and the control's holds one for **none** that `weighted+neighbours`' doesn't (near ties set apart).
  Otherwise the top 3 stays, and the expansion is recorded as measured, not adopted.

Against the top 3 alone, `weighted+neighbours` can't lose a question: it sends the top 3 first, and the top 3 always
fits the budget. That is why the decision is against the control: a gain from the documents' structure shouldn't be
confused with a gain from sending more text.

Day 06's pipeline and `answering.py` don't change today: an adopted setting is recorded for a later day to switch on
and re-measure. The new baseline is `evals/baselines/day-11-retrieval-v3.json` (v3, the reference for later days).
