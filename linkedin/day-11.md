# LinkedIn — Day 11: Advanced Retrieval for Harder Questions

**Status:** ready to post once the `day-11` branch is merged into `main`, the tag `day-11` is pushed and the GitHub Actions run is green. **Publish first, then post:** the post links to the repository.
**Matches:** `docs/day-11/LEARN.md`, `notebooks/day_11_advanced_retrieval_section_expansion.ipynb` (outputs from the 2026-11-09 run), `evals/preregistration/day-11.md`, `evals/questions-v3.jsonl`, `evals/baselines/day-11-retrieval-v3.json`.

---

## Post text

15 Days of RAG from Scratch · Day 11/15: Advanced retrieval for harder questions

Day 10 ended with a question no better word could fix: "My phone was stolen on a business trip. Who should I tell?" The retriever found the right policy and the wrong sections. The answer was in the section next to one it sent.

So Day 11 builds one "advanced" pattern from scratch: section expansion. After retrieving, also send the sections just before and after each hit in its document. It has an off-switch, a trace that says why every section was sent, and a 300-word budget.

The rules were committed first, with 68 new test questions written before anything ran on them.

Four results worth knowing:

1. Against the original top 3 it can't lose: it sends the top 3 first. That makes "it found more" meaningless on its own, because more text always finds more.

2. So it was also compared with a control: the same number of sections, taken from further down the ranking. On the 52 answerable new questions, the top 3 held 49 answers, the expansion 51, the control 52. Not adopted. All three misses were ranked 5th, and one was in another document, which neighbours can't reach.

3. A lookup is not a search. The retriever filters what it searches, but neighbours are fetched by position. With one section restricted to managers, an unchecked lookup would have handed it to an employee for 10 of her 155 queries. Every added section needs its own permission check.

4. In the context isn't answered. For none of the four questions where the expansion added the answer section did the simple API-free reader answer correctly: it copied the sentence closest to the question, from another section, or refused. The three outcomes that did change changed through an unrelated refusal rule.

When does it earn its place? When chunks are smaller than the unit of meaning, or answers span neighbouring chunks. Here the chunks are whole sections and the ranking already had every answer in its top 5.

Same caveat: small sets written by one person who knew the pattern, and the new test set has now decided too.

Runs locally, no API key: a notebook, the committed rules, 68 new labelled questions, baselines and tests.

https://github.com/shashanksuresh18/15-days-of-rag-from-scratch

Day 12: evaluating answers, not just what was retrieved.

#RAG #LLM #Python #AIEngineering

---

## Images — posting order, alt text and purpose

Upload in this order. LinkedIn's API documentation recommends alt text under 120 characters (maximum 4,086), so each alt text below leads with a short sentence that works on its own; paste the full text where the box allows.

| # | File | Teaches |
|---|---|---|
| 1 | `assets/day-11/slide-01-cover.png` | Why read around the hit (the stolen phone's answer is the section before hit 3), and the pre-registered result against a size-matched control: not adopted. |
| 2 | `assets/day-11/slide-02-how-it-works.png` | How section expansion works: hits first, a lookup around each hit, a permission check, a budget, a per-query trace and an off-switch. |
| 3 | `assets/day-11/slide-03-the-control.png` | Why a change that adds context needs a control: the same number of sections from further down the ranking reached every answer the expansion did, and one more. |
| 4 | `assets/day-11/slide-04-before-you-ship.png` | What any step that adds context needs before it ships: a permission check on lookups, a cost, an answer check, an off-switch, a trace, a budget and a control. |

**Alt text 1 — cover**
> Read around the hit, measured: section expansion against a control on 52 new questions. Top: the answer was next door. For "My phone was stolen on a business trip. Who should I tell?", the Security Incident Reporting policy's four sections in document order: What counts, ranked 4th and not sent; How to report, ranked 6th, the answer; What not to do, hit 3; After you report, hit 2. An arrow from hit 3 back to How to report: previous of hit 3, added. The top 3 misses the answer; with neighbours it is in the context. Middle: 52 new questions, written first; answer in the context: the top 3, 49 (110 words); top 3 plus neighbours, 51 (199 words); as many sections from further down the ranking, 52 (199 words). Bottom: the bar, set in advance, was to beat the control. It found 2 the top 3 missed; the control found those and 1 more. Section expansion: not adopted. More context always finds more: compare with the same amount.

**Alt text 2 — hits first, then their neighbours**
> Hits first, then their neighbours: section expansion with an off-switch and a trace. Four numbered steps: 1, search, filtered: the weighted fusion's top 3 are the hits; 2, look up around each hit: the section before and after, in the same document; 3, may this user read it? A lookup isn't a search, so check again; 4, does it fit 300 words? Whole sections; skip what doesn't fit. Middle: the trace for the stolen phone: hit 1 Safety while travelling, sent; hit 2 After you report, sent; hit 3 What not to do, sent; previous of hit 3, How to report, sent, the answer; next of hit 3, After you report, already sent. 5 sections, 188 words; the hits are always sent first. Bottom left: the off-switch, EXPANSION = "off", gives exactly Day 10's top 3 for every query. Bottom right: which neighbours, a dev rule committed before the test: both found 90 of 90; previous, next or the whole parent found 89.

**Alt text 3 — same words, chosen two ways**
> Same words, chosen two ways: the control sends as many sections from further down the ranking. Two cards: around the hits, top 3 plus neighbours, 51 of 52 answers in the context, 199 words; further down, the top m where m is the number of sections sent, 52 of 52, 199 words. The top 3 alone: 49 of 52, 110 words. Middle: the top 3's three misses, each ranked 5th, with ticks for around and down: a train ticket I paid for, Submitting a claim, next to hit 1: both reach it. I clicked a suspicious link, How to report, next to hit 1: both. A taxi from the airport, in another document, the expense policy: only further down reaches it. Neighbours can't leave the hits' document; the ranking can. Bottom: why the control: against the top 3 an expansion can't lose, because it sends the top 3 first; only a control says whether structure helped, or size.

**Alt text 4 — a lookup is not a search**
> A lookup is not a search: what to check before you ship any step that adds context. Top: one section restricted to managers, in memory, run as alice: the search returns it for 0 of 155 queries; an unchecked neighbour lookup would hand it to her for 10 of 155; the checked expansion, 0. Neighbours, parents and linked documents need their permissions checked again. Middle left, cost: words 114 to 214 and tokens 137 to 259, on average per answerable query; the lookup itself takes microseconds. Middle right, in context is not answered: for the suspicious-link question How to report is now sent, but the reader copies "Do not try to investigate…"; Day 12 grades answers. Bottom: every advanced retrieval step needs an off-switch (back to the old context, no deploy), a trace per query (why each section was sent), a permission check (on everything it fetches), a budget (and what it skipped, logged) and a control (the same amount, chosen simply).

## Posting options

- **Image post (simplest):** attach the four PNGs in the order above and add the alt text to each.
- **Swipeable document post:** upload `assets/day-11/day-11-carousel.pdf` (the same four slides as a 4-page PDF). Document posts may not offer per-page alt text, so keep the image descriptions in the post body or first comment.

## Pre-posting checklist

- [ ] `day-11` merged into `main`, tag `day-11` pushed, GitHub Actions green on Linux, Windows and macOS. **Publish first, then post.**
- [ ] Open the repository in a private/incognito window and check the README index links to the Day 11 lesson, notebook, rules, new questions, baseline, slides and post
- [ ] Numbers in the post still match the notebook output (68 new test questions, 52 answerable, 49 / 51 / 52, ranked 5th, one in another document, 10 of 155, 300 words, three outcomes changed)
- [ ] Alt text added to all four images
