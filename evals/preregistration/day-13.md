# Day 13: rules and settings, committed before the held-out questions are scored

This file was committed on the `day-13` branch as its first commit, together with the new held-out questions
(`evals/questions-v5.jsonl` and `evals/manifest-v5.json`), **before any Day 13 run scored a v5 test question**,
exploratory runs included. The exploration scripts loaded `questions-v4.jsonl` and the accepted Day 12 answers only
(all 266 v4 queries count as dev today, see below) and refused to start if a `questions-v5.jsonl` existed. The rules
below were frozen first (a local scratch file, 2026-11-12 21:11:27 UTC); the new questions were written after that, by
the protocol below, and their labels were derived from their evidence by Day 07's rule, with no retriever, reader or
pipeline run on them. The Day 13 notebook recomputes every dev value below and refuses to go on (an `assert`) if it gets
another one.

The commit's timestamp is local, so it shows the order only as far as you trust the author. To have a server witness
it, push this commit (or open a pull request with it) before running anything on the held-out questions.

## What Day 13 is about, and what in it can change an answer

Day 13 instruments the current pipeline: a trace of every request (ids, scores, decisions, timings, tokens; no document
text, no question text, no keys), latency per step (p50 and p95), a cost breakdown, caching, and timeouts, retries and a
fallback for the optional model client. Most of that must change **no** answer, and the notebook checks it rather than
decides it (below). Two things can change answers, and they are today's two decisions:

- **D1, a semantic answer cache:** serve the answer given to a *similar* earlier question instead of running the
  pipeline. It saves the whole pipeline (and, with a language model, the model call), and it can serve the wrong
  answer to a question that only looks similar.
- **D2, what to answer when the model fails:** the current pipeline's API-free reader (checked by the same citation
  check, and marked as the fallback's answer) instead of Day 06's "unavailable" message.

## What was already known when this was written

- **The v4 test split is used** (`evals/README.md`, rule 8): its 76 questions decided Day 12's decision. The author had
  read them and every published result on them, including the current pipeline's answer to each.
- **Everything below was computed on dev, before freezing**: the semantic cache's leave-one-out on the 236 dev
  questions, from the accepted Day 12 answers (`evals/baselines/day-12-answers-v4.json`, the `names-v2` pipeline), and
  the current pipeline's value on dev. So every dev number here is **in-sample**.
- **Dev has no paraphrase pairs.** Earlier protocols forbade writing a question that paraphrases an existing one, so
  dev can show what a semantic cache gets wrong, but hardly what it gets right: the nearest dev question in the same
  permission scope is at most **0.889** away in cosine for every dev question, and the closest pairs are questions with
  *different* answers ("Does being the primary or the secondary caregiver change how much parental leave I get?" and
  "How much parental leave does a secondary caregiver get?": 0.8815).
- **The new test questions were written after the rules were frozen, by the same author, who knew them**, including
  the threshold (0.89) and that questions differing in one detail embed close together (the dev curve below shows it).
  The protocol fixed beforehand **which** dev questions they are built from (every 8th, by position, see "The data"),
  and that every answerable source gets one paraphrase **and** one near neighbour. So the test set is a **stress test
  of the cache, not a sample of traffic**: half of the questions built from answerable sources are written to look like
  a cached question and need another answer. It can show that a cache serves a wrong answer; it can't say how often
  that would happen in real traffic, where nobody knows the share of near neighbours.
- What hadn't been computed: **anything for any v5 test question**: no ranking, no context, no answer, no grade, no
  cosine with a dev question.

## The data

`questions-v5.jsonl` (manifest `manifest-v5.json`, version 5, the same snapshot as v1 to v4):

- **dev, 266 queries:** the 266 queries of v4, text and labels unchanged. v4's 76 test questions are dev from now on
  (rule 4: Day 12 used them to decide); each one's note says so. Of the 266, **236 are questions** (164 answerable,
  72 not) and 30 are lookups (an end-to-end answer is graded for questions only).
- **test, 51 new questions** (`origin: day13-new`), written by this protocol: take v4's questions (not lookups) in file
  order; every 8th **answerable** one, starting with the first (positions 1, 9, ..., 161: **21** sources), gets a
  **paraphrase** (the same user, the same answer and facts, reworded as an employee might ask it) and a **near
  neighbour** (the same user, the source's wording with one detail changed so that the documents give a different
  answer, or none); every 8th **unanswerable** one (positions 1, 9, ..., 65: **9** sources) gets a paraphrase (still
  unanswerable). Each new question's note starts with "paraphrase of" or "near neighbour of" and its source's id; its
  first tag is `paraphrase` or `near-neighbour`. By reading (no code was run on them): **33 answerable** (21 paraphrases,
  12 near neighbours with a different answer) and **18 not** (9 near neighbours the documents don't answer, 9
  paraphrases); **49** asked by alice, **2** by maria (the managers' guide question and its near neighbour).

The earlier files don't change: Day 13 compares its answers on v4's 236 questions with
`evals/baselines/day-12-answers-v4.json`, which refers to v4.

## The current pipeline: the no-change baseline

Day 12's pipeline with the names rule Day 12 adopted:

| Step | Setting |
|---|---|
| Retrieve | Day 08's weighted fusion (α = 0.7 on dense, min-max per retriever and query) over BM25 and dense retrieval (`BAAI/bge-small-en-v1.5`), each permission-filtered before fusion |
| Context | the weighted fusion's top 3, Day 06's 300-word budget |
| Gate | Day 07's measured gate: refuse without reading if the **dense** top score is under 0.62 |
| Reader | Day 06's extractive reader at Day 07's measured threshold (0.63), with **Day 10's names rule** (`NamesReader` with `unknown_names_v2`; adopted on Day 12) |
| Check | Day 06's citation check: a reply that fails it is blocked (refused) |

On v4 it is the `names-v2` pipeline of `evals/baselines/day-12-answers-v4.json`. The notebook first reproduces that
whole file (both of its pipelines) and checks "no changes".

## What Day 13 adds that must not change an answer (checked, not decided)

1. **A trace of every request:** each step's start and duration, the ids and scores it produced, its decision, and the
   tokens counted or estimated; the user id and groups, the question's SHA-256 (not its text), the index fingerprint
   and the pipeline version. Checked on every traced request: no run of six words from any chunk, no question text, no
   key.
2. **Exact caches**, keyed on everything their result depends on: each sentence's vector (its exact text and the model
   revision; Day 11's cache), the question's vector (exact text, model revision, query instruction), and whole answers
   (the asker's groups, the exact question, the index fingerprint, the pipeline's version), with every cited chunk
   checked again against the asker's groups before a cached answer is served (Day 11's rule: a lookup by id isn't a
   search). Checked: every v5 question's answer is identical with and without them.
3. **Timeouts, retries, a circuit breaker and a fallback** for the optional model client (settings below). They act
   only when a model is configured and fails; they are demonstrated on local fake servers, never on test questions.

## D1: a semantic answer cache

| Part | Rule |
|---|---|
| Key | the dense retriever's query vector (bge-small with its query instruction), which the pipeline computes anyway |
| Scope | an entry is looked up only for an asker with **the same groups** and under **the same index fingerprint** |
| Entries | answered answers only (a refusal is not cached) |
| Served | the nearest entry in scope, if its cosine with the question is at least **θ**; its answer is served as it is, after the permission re-check |
| θ | the **lowest** value in steps of 0.01 from 0.80 to 1.00 at which, leave-one-out on the 236 dev questions (each looked up against every other dev question's answer in its scope), **no dev question's outcome is worse** than the pipeline's own (correct > refused > wrong; refused > answered for an unanswerable question) |
| On test | the cache holds the dev questions' answers only; each test question is looked up on its own and isn't added to it |
| Near ties | a test question whose outcome changes is set apart, as CPU rounding, if its cosine is within 0.001 of θ (flipping that call turns the hit into a miss, which gives back the pipeline's outcome) |

Computed on dev (in-sample), leave-one-out, against the current pipeline's own outcome:

| θ | hits | better | worse | value change at cost 1 / 2 / 3 |
|---|---|---|---|---|
| 0.80 | 53 | 2 | 14 | −24 / −38 / −52 |
| 0.85 | 16 | 0 | 4 | −8 / −12 / −16 |
| 0.88 | 6 | 0 | 1 | −2 / −3 / −4 |
| **0.89** | **0** | 0 | 0 | 0 |

So **θ = 0.89**: no dev question has a neighbour that close (the highest is 0.8890). At 0.88 the one change is
d12-a25 ("Does being the primary or the secondary caregiver change how much parental leave I get?") served
d12-a24's answer about secondary caregivers only: wrong.

## D2: what to answer when the model fails

The candidate: when a configured language model can't answer (after the retries below, or with the breaker open),
answer with the current pipeline's API-free reader on the same context, checked by the same citation check and marked
as the fallback's answer, instead of Day 06's "The answering service is unavailable right now." Refusing everything
is worth 0 under Day 12's value of an answer (answerable correct **+1**, refused **0**, wrong **−2**; unanswerable
refused **0**, answered **−2**), so the fallback is worth having only if the reader's total value is above 0.

Computed on dev (in-sample, the 236 questions): answerable 124 correct / 23 wrong / 17 refused, unanswerable 39
refused / 33 answered; value **+68** at cost 1, **+12** at cost 2, **−44** at cost 3.

## Settings that aren't decisions

Chosen before any test question existed, from documentation and Day 12's recorded live pass, and demonstrated on local
fake servers; none is tuned on any question.

| Setting | Value (real use) | Why |
|---|---|---|
| Attempts | 3 (2 retries) | the OpenAI Python library's default (`DEFAULT_MAX_RETRIES = 2`) |
| Per-attempt timeout | 10 s (connect and each read) | about 4× the slowest of Day 12's 40 recorded requests (2.63 s); too few requests to estimate a p99.9 |
| Deadline for the model step | 30 s, whatever the attempts | a timeout on each read isn't a deadline: a server that sends a byte now and then never trips it |
| Backoff | full jitter: a random wait between 0 and min(8 s, 0.5 s × 2^retry) | jitter spreads retries out (AWS Builders' Library); 0.5 s and 8 s are the OpenAI library's initial and maximum delays |
| Retry-After | honoured if it fits in what is left of the deadline; otherwise give up now | OpenAI's error-code guide says to wait at least as long as it asks |
| Retried | connection errors, timeouts, HTTP 408, 409, 429 and 5xx | as the OpenAI library retries |
| Not retried | 401, 403, other 4xx, a 429 whose error code is a billing or quota one (`insufficient_quota`, `credit_balance_exhausted`, spend or usage limits), a reply that isn't JSON or is empty, `x-should-retry: false` | retrying can't fix them (OpenAI's error-code guide) |
| Circuit breaker | opens after 3 failed requests in a row; for 30 s every request goes straight to the fallback; then one trial request | fails fast during an outage instead of waiting for timeouts on every request |
| Fake-server demonstration | the same logic, times divided by 20 (timeout 0.5 s, deadline 1.5 s, backoff 0.025 to 0.4 s, cool-down 1.5 s) | so the notebook runs in seconds |

## What will be reported, whatever it shows

1. Day 12's answers reproduced: the whole of `evals/baselines/day-12-answers-v4.json` (both pipelines, 236 questions)
   rebuilt and compared (expected: no changes), and the regression suite (`tests/test_regressions.py`).
2. **Traces**: one request shown step by step; every traced request checked for document text, question text and keys.
3. **Latency** per step, p50 and p95 (nearest rank), on the 236 dev questions: cold (empty caches), warm (sentence
   vectors cached, as a running service would have them) and repeated (answer cache), on the machine that runs the
   notebook (a description: never pinned by a test); and, as recorded on the user's PC on 2026-11-12, the model call's
   p50 and p95 from Day 12's 40 live requests.
4. **Cost**: tokens per request by part (instructions, sources, question), the estimate (characters / 4) against the
   40 token counts Day 12's live pass recorded, cost per request and per 1,000 at the prices checked on 2026-11-12, and
   what the gate and a cache save.
5. **Exact caches**: no answer changed (asserted), hits, time saved; two key mistakes shown in memory: a key without the
   asker's groups serves a managers-only answer to an employee; a key without the index fingerprint keeps serving a
   policy's old value after the document changes.
6. **D1 on dev (in-sample)** as above, and **on the 51 v5 test questions**: hits by question type, every changed outcome
   with the sign test, near ties set apart, the value change at costs 1, 2 and 3; and, as a description never used to
   decide, the same counts for every θ from 0.80 to 1.00.
7. **D2 on the 51 v5 test questions**: the current pipeline's outcomes with Wilson intervals, its stages, and its value
   at costs 1, 2 (the decision) and 3.
8. **Failures on local fake servers**: a slow server (per-attempt timeout), a trickling server (a timeout isn't a
   deadline), rate limits with and without Retry-After, a billing 429, a 401, an outage (retries, then the fallback),
   the breaker during an outage, a malformed reply; each with its attempts, waits and outcome in the trace.
9. The new failures recorded since Day 12, added to `evals/regressions.jsonl`.

## Decisions, and what they need

- **D1, the semantic cache (θ = 0.89):** later days use it if, on the 51 v5 test questions, it serves **at least one**
  answer from the cache **and makes no outcome worse** (after near ties are set apart). A cache exists to save time and
  money; it must never cost an answer. Otherwise answers are cached only exactly (item 2 above), and the semantic cache
  is recorded as measured, not adopted.
- **D2, the fallback:** later days answer with the API-free reader when the model fails if, on the 51 v5 test
  questions, the current pipeline's total value at cost 2 is **above 0**. Otherwise they show Day 06's "unavailable"
  message. Costs 1 and 3 are reported next to it.

`answering.py` doesn't change today: an adopted rule is recorded for later days to switch on.

## The optional live pass (a description, never a decision)

Off by default (`DAY13_LIVE_MODEL=1` or `LIVE_MODEL = True` switches it on); the tests never run it. One pass through
the resilient client (the real-use settings above) with the model configured in `.env` (`RAG_CHAT_*`), traced. At most
**20 requests**: from the questions Day 12's live pass asked, its 6 recorded failures, then its first 9 answerable and
its first 5 unanswerable ones, in its order, so each reply can be compared with the one the same model gave on
2026-11-12. Reported: each request's trace (time per step, tokens, attempts), the model step's p50 and p95, tokens and
cost, how many replies are identical to Day 12's, and the outcomes. One run of one model: an observation.
