# LinkedIn — Day 13: RAG Operations: Tracing, Latency and Cost

**Status:** ready to post once the `day-13` branch is merged into `main`, the tag `day-13` is pushed and the GitHub Actions run is green. **Publish first, then post:** the post links to the repository.
**Matches:** `docs/day-13/LEARN.md`, `notebooks/day_13_rag_operations_tracing_latency_cost.ipynb` (outputs from the 2026-11-12 run), `evals/preregistration/day-13.md`, `evals/questions-v5.jsonl`, `evals/baselines/day-13-answers-v5.json`, `evals/reviews/day-12-live-gpt-6-luna.jsonl`, `evals/regressions.jsonl`.

---

## Post text

15 Days of RAG from Scratch · Day 13/15: Tracing, latency and cost

A user asks: "How soon must I report a lost laptop?" The pipeline answers wrongly. Which step lost it?

The trace for that request says it without a word of text: the right section was sent to the reader, and another one was cited. Retrieval did its job; the reader didn't. Day 13 records a trace like that for every request: ids, scores, decisions and timings. No document text, no question text, no keys, checked on all 287 traces.

Five things the measurements showed:

1. The slowest step depends on which request you measure. Cold, the reader was 86% of the time (it embedded every sentence it read). With those vectors computed once at index time, a request went from about 120 ms to about 12 ms. A hosted model call took about 1.1 s: roughly 90 times that.

2. A cache key is everything the answer depends on. Leave out the asker's permission groups and an employee is served a manager's answer. Leave out the index version and the cache keeps saying "14 characters" after the policy changed to 16.

3. Similar isn't the same. A semantic cache (serve the answer to a similar earlier question) was tested on 51 new questions with rules written first. It served 10 answers and made one worse: a question the documents don't answer got another question's wrong answer. Not adopted.

4. A timeout isn't a deadline. Against a server that trickles bytes, the old client with a 0.5 s timeout waited 2.44 s. The new one gives up at its deadline, retries only what can pass (not a wrong key or an empty balance), and with a circuit breaker sent 9 calls to a failing server instead of 24.

5. Choose the fallback by value. When the model fails, answering with a weaker local reader looked helpful. Scored in advance (a wrong answer costs twice a right one), it was −18 on the new questions, mostly confident answers to unanswerable questions. "Unavailable" won.

Same caveat: small sets written by one person, one machine's timings, and a test set built to stress a cache, not to sample real traffic.

Runs locally, no API key: a notebook, the committed rules, local fake servers that fail on purpose, and every result above.

https://github.com/shashanksuresh18/15-days-of-rag-from-scratch

Day 14: security, governance and freshness.

#RAG #LLM #Python #AIEngineering

---

## Images — posting order, alt text and purpose

Upload in this order. LinkedIn's API documentation recommends alt text under 120 characters (maximum 4,086), so each alt text below leads with a short sentence that works on its own; paste the full text where the box allows.

| # | File | Teaches |
|---|---|---|
| 1 | `assets/day-13/slide-01-cover.png` | What a trace holds (ids, scores, decisions, times; no text), and how one trace shows which step lost an answer. |
| 2 | `assets/day-13/slide-02-latency-and-cost.png` | Why the slowest step depends on the request measured (cold, warm, repeated, a model call), and where a model call's money goes. |
| 3 | `assets/day-13/slide-03-cache-keys.png` | What an answer-cache key must hold, what leaving out a part does, and why a semantic cache wasn't adopted. |
| 4 | `assets/day-13/slide-04-failures-and-fallback.png` | How a model client should treat each failure (retry, wait, give up), what a circuit breaker saves, and choosing the fallback by value. |

**Alt text 1 — cover**
> Trace every request: one trace shows where the time went and which step lost the answer. Top: one request, one trace, "How soon must I report a lost laptop?", cold, on one machine. A timeline of bars: embed_query 15.0 ms (the question to a vector); retrieve, three steps, 0.4 ms (BM25, dense, fusion); gate 0.0 ms (0.6758 at least 0.62: pass); context 0.1 ms (3 sections, about 497 tokens); generate 102.7 ms (the reader); check 0.1 ms (citations ok); total 118.3 ms. The trace holds ids, scores, decisions and times, no text. The reader is 86% of a cold request; with vectors saved at index time, 1%. Middle: the trace says which step lost it: sent, security-incident-reporting#how-to-report, ticked; cited, travel-policy#safety-while-travelling, crossed. Retrieval found the answer; the reader picked another sentence. 18 answerable questions show this pattern. Bottom: 287 traces, no text in any of them: the question is a hash; documents, names and keys stay out; checked: no 6 words of a document, no 4 of the question.

**Alt text 2 — latency and cost**
> Time and money, per step: the slowest step changes with the request, and input tokens drive the cost. Top, latency, one run on one machine, p50 of the whole request and its biggest step: cold, nothing cached, 119.56 ms, generate 86%; warm, sentence vectors saved once, 12.10 ms, embed_query 96%; repeat, the answer cache, 0.01 ms, the lookup; a model, Day 12's calls, about 91 times warm, 1.10 s, the model call. Measure each kind. p95s don't add up: the steps' 171.33 ms against the requests' 167.05 ms, cold. Bottom, cost per 1,000 requests that reach a model: gpt-6-luna, USD 0.10 and 0.50 per million tokens, USD 0.059; gpt-6.1-sol, USD 2 and 10 per million tokens, USD 1.187. The API-free reader has no token price, about 12 ms of CPU per warm request. The gate refuses 29 of 236 before any model call; input is 76% of the cost and the sources are 58% of the input; the provider's count is about 0.91 times our estimate of characters divided by 4; the longest prompt is about 543 tokens, and prompt caching starts at 1,024. Prices checked 2026-11-12; tokens counted by the provider on 40 requests.

**Alt text 3 — cache keys**
> Get the cache key right: exact caches changed no answer; a missing part of the key did. Top, the answer cache's key: groups, question, index fingerprint, pipeline version, plus a re-check that the user may still read every cited source. Middle, leave one part out: no groups, leak: maria, a manager, asks; alice asks it too; alice gets a managers-only answer. No fingerprint, stale: the password policy goes from 14 to 16 characters; the cache still says 14. Bottom, a semantic cache decided in advance: serve a past answer if the cosine is at least 0.89, the lowest value with no dev answer worse. On 51 new questions: 9 same outcome, 0 better, 1 worse. "If I give money to a charity, will Acme match it?": nothing answers it; the pipeline refuses, the cache served an answer. Value −2 at cost 2: not adopted; similar isn't the same.

**Alt text 4 — failures and fallback**
> Plan for the model to fail: timeouts, retries, a deadline and a breaker, tested on a fake server. Top, what the client does, real settings (the demo runs 20 times faster): 503, 500, 429: retry after a random wait (full jitter); 429 with Retry-After: wait as told, if it fits the deadline; 401, no credit, not JSON: no retry, it won't pass; slow server: 10 s per attempt, 3 attempts; trickling server: a 30 s deadline for the whole call; still failing: the fallback, and the trace says why. A timeout is per read: Day 06's client waited 2.44 s with a 0.5 s timeout. Middle, a circuit breaker in an outage of 8 requests: retries only, 24 calls; with a breaker, 9 calls. 3 failed requests open it; after a cool-down, one good trial closes it. Bottom, which fallback, decided by value: the API-free reader instead of "unavailable", on 51 new questions: 20 correct, 7 wrong, 12 unanswerable ones answered; correct +1, refused 0, wrong −2; value −18: not adopted; say "unavailable" instead.

## Posting options

- **Image post (simplest):** attach the four PNGs in the order above and add the alt text to each.
- **Swipeable document post:** upload `assets/day-13/day-13-carousel.pdf` (the same four slides as a 4-page PDF). Document posts may not offer per-page alt text, so keep the image descriptions in the post body or first comment.

## Pre-posting checklist

- [ ] `day-13` merged into `main`, tag `day-13` pushed, GitHub Actions green on Linux, Windows and macOS. **Publish first, then post.**
- [ ] Open the repository in a private/incognito window and check the README index links to the Day 13 lesson, notebook, rules, new questions, accepted run, live reviews, slides and post
- [ ] Numbers in the post still match the notebook output (287 traces, 86%, about 120 ms and 12 ms, about 1.1 s, 51 new questions, 10 served and 1 worse, 2.44 s with a 0.5 s timeout, 24 and 9 calls, −18) and the lesson
- [ ] The timings are one run on one machine; the post says "about" for each, and the lesson says which machine
- [ ] Alt text added to all four images
