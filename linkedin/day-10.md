# LinkedIn — Day 10: Query Understanding and Transformations

**Status:** ready to post once the `day-10` branch is merged into `main`, the tag `day-10` is pushed and the GitHub Actions run is green. **Publish first, then post:** the post links to the repository.
**Matches:** `docs/day-10/LEARN.md`, `notebooks/day_10_query_understanding_and_transformations.ipynb` (outputs from the 2026-10-30 run), `evals/preregistration/day-10.md`, `evals/questions-v2.jsonl`, `evals/baselines/day-10-retrieval.json` and `day-10-retrieval-v2.json`.

---

## Post text

15 Days of RAG from Scratch · Day 10/15: Query understanding and transformations

Four failures left over from Days 1 to 9 start in the question, not in the documents. "New dads" never meets "caregivers", "wine" never meets "alcohol", and a rule that refuses unknown names refused "Tokyo" and "Should" (the first word of a second sentence).

Day 10 reads the question before searching (sentences, values, names, places, exact values) and adds one transformation: a reviewed vocabulary map. Every entry is tied to a recorded failure, and it adds the documents' words to the keyword query instead of replacing the user's.

The rules came first again, committed with 41 new test questions before anything ran on them (the old test set had been used to decide).

Four results worth knowing:

1. On the questions it was written from, the map works: "new dads" goes from rank 3 to 1, and nothing gets worse. On the 41 new questions it touched one, already first. No gain, no loss: not adopted. A map fixes the gaps someone recorded, and only those.

2. Matched word by word instead of as phrases, the same map drifts: "for" adds "reimbursed", "it" adds "turn off", "one-time" adds "leave". It touched 11 new questions instead of 1: one answer moved up, one lost first place.

3. A true synonym can still hurt. "Dinner → meals" is correct English, but "Do I need a receipt for a USD 40 dinner?" needs the receipt rule, not the meal limit, and it lost first place.

4. Fixing the names rule moved the failure downstream. It kept 26 of 29 answerable new questions instead of 22 (Berlin, Paris, "Do…?", "Who…?"). Of the 7 questions it stopped refusing, 4 were answered correctly, 2 wrongly, and one unanswerable question, refused before by accident, got a confident, cited non-answer. The bar set in advance was "lose nothing", so this isn't adopted either.

Also shown: a guard for any rewrite. It discarded a rewrite that dropped "USD 40" and one that turned "Tokyo" into "Japan", but kept "new dads → secondary caregivers": it checks form, not meaning. No LLM rewrite was run; it would have to beat the map under the same rules first.

Same caveat: small sets written by one person, and the new test set has now decided twice, so the next held-out decision needs new questions again.

Runs locally, no API key: a notebook, the committed rules, 41 new labelled questions, baselines and tests.

https://github.com/shashanksuresh18/15-days-of-rag-from-scratch

Day 11: questions whose answer is in the next section.

#RAG #LLM #Python #AIEngineering

---

## Images — posting order, alt text and purpose

Upload in this order. LinkedIn's API documentation recommends alt text under 120 characters (maximum 4,086), so each alt text below leads with a short sentence that works on its own; paste the full text where the box allows.

| # | File | Teaches |
|---|---|---|
| 1 | `assets/day-10/slide-01-cover.png` | A vocabulary gap the map fixes, and the pre-registered result: two ranks up on the questions it was written from, nothing on 41 new ones. |
| 2 | `assets/day-10/slide-02-read-the-question.png` | Query understanding: what sentences, names and places, values and exact values each allow, and where the map's words go. |
| 3 | `assets/day-10/slide-03-when-not-to-rewrite.png` | Three ways a change to the question goes wrong: word-level drift, a true synonym that hurts, and a rewrite that keeps the form but changes the meaning. |
| 4 | `assets/day-10/slide-04-follow-the-questions.png` | Fixing a refusal rule moves the failure to the next stage: what happened to the seven questions it stopped refusing, and the decision. |

**Alt text 1 — cover**
> Change the question, measured: a reviewed vocabulary map, tested on 41 new questions. Top: where the words don't meet. For "How much time off do new dads get?", BM25 searches for "much time off new dads get", the answer says "primary caregivers … leave", and they have no word in common. The map adds the documents' words, "+ caregivers leave", and the weighted fusion moves the answer from rank 3 to rank 1. Middle, for the default retriever, the weighted fusion: on the questions behind the map (v1, 81 queries, 6 touched), 2 ranks up and 0 down against Day 09's accepted baseline; on 41 new questions, written and committed first, 0 moved and 1 touched, against the same fusion without the map. Bottom: decisions with criteria set in advance: the map in the default retriever, not adopted; the fixed names rule, not adopted. A map fixes the gaps someone recorded, and only those.

**Alt text 2 — read the question first**
> Read the question first: what each part of a question allows. Four cards. Sentences: in "…a virus. Should I switch it off?", "Should" starts a sentence, so it is not a name; Day 06 refused it. Names and places: in "…per night in Tokyo?", Tokyo is a city, and a source says "any city not listed"; Day 06 refused it. Values: in "…a receipt for a USD 40 dinner?", USD 40 decides the answer, so any rewrite must keep it; the guard checks. Route: "USD 1,000" and "#it-help" are exact values and are never rewritten; 30 of 30 lookups. Bottom: then, for questions only, the map: the question plus "caregivers leave" goes to BM25, the question as asked goes to dense retrieval and the gate, and both feed the weighted fusion. Append, never replace; chosen on dev, committed first.

**Alt text 3 — when not to rewrite**
> When not to rewrite: three ways a change to the question goes wrong. One, matched word by word, the map drifts: "for" adds reimbursed, "it" adds turn off, "one-time" adds leave. "Can I give my one-time code to someone from IT?" gets "leave remotely remote turn off" and its answer drops from rank 1 to 2; 11 of 29 new questions touched instead of 1. Two, true in general, wrong for the question: "dinner → meals" makes "Do I need a receipt for a USD 40 dinner?" drop from 1 to 2, because it needs "every expense over USD 25", not the meal limit. Three, a guard checks form, not meaning, on rewrites written by hand: dropping "USD 40" is discarded; "Tokyo → Japan" is discarded; "new dads → secondary caregivers" is kept, although a dad can be the primary caregiver, 16 weeks, not 6. Bottom: never rewrite exact values or who is asking; permissions come from the login.

**Alt text 4 — fix a rule, follow its questions**
> Fix a rule, follow its questions: Day 06's names rule, fixed, on 41 new questions, with the dense gate 0.62 first. Dot rows: Day 06's rule keeps 22 of 29 answerable questions and refuses 6 of 12 unanswerable ones; the fixed rule, which skips every sentence's first word and types places, keeps 26 of 29 and refuses 5 of 12. Gained: Berlin, Paris, and questions starting a second sentence with "Do" and "Who"; lost: a "Does" question that had been refused by accident. Middle: the 7 questions it stopped refusing: Tokyo, the virus question, Berlin and "register the trip" are correct; Paris gets London's row, wrong; the stolen phone gets "report it straight away", wrong; changing teams gets a cited non-answer. A lifted refusal hands the question to the next stage. Bottom: the bar set in advance, lose nothing: 4 more answerable kept and 1 fewer refusal, so not adopted; the new test split has decided twice and is now used too; next, Day 11, the section next door.

## Posting options

- **Image post (simplest):** attach the four PNGs in the order above and add the alt text to each.
- **Swipeable document post:** upload `assets/day-10/day-10-carousel.pdf` (the same four slides as a 4-page PDF). Document posts may not offer per-page alt text, so keep the image descriptions in the post body or first comment.

## Pre-posting checklist

- [ ] `day-10` merged into `main`, tag `day-10` pushed, GitHub Actions green on Linux, Windows and macOS. **Publish first, then post.**
- [ ] Open the repository in a private/incognito window and check the README index links to the Day 10 lesson, notebook, rules, new questions, baselines, slides and post
- [ ] Numbers in the post still match the notebook output (41 new test questions, rank 3 to 1, touched one, 11 instead of 1, one up and one down, USD 40, 26 of 29 instead of 22, 7 questions: 4 correct, 2 wrong, 1 non-answer)
- [ ] Alt text added to all four images
