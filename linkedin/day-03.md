# LinkedIn — Day 03: Chunking and Document Structure

**Status:** ready to post once the `day-03` branch is merged into `main` and the GitHub Actions run is green.
**Matches:** `docs/day-03/LEARN.md`, `notebooks/day_03_chunking_and_structure.ipynb` (outputs from the 2026-10-15 run).

---

## Post text

15 Days of RAG from Scratch · Day 3/15: Chunking and document structure

A retriever never returns a document. It returns a chunk. And a chunk can come from exactly the right place and still not answer the question.

Day 3's example is a hotel-limit table in a (synthetic) travel policy. Cut it every 40 words, and the Singapore row "USD 220 | USD 60" lands in a different chunk from the header row that says which number is the hotel limit. Retrieval finds the table. The answer is gone.

So I compared two chunkers on Day 2's 11 records, with the same BM25 retriever and 13 test questions:

A. Fixed 40-word windows
B. Structure-aware: cut at headings, never inside a sentence or table row, repeat a table's header row if it has to be split, and put a "Title > Section" line on top

Complete evidence inside one retrieved chunk: A 9 of 13, B 13 of 13. At about the same reading cost: 110 vs 113 words per question.

Four results worth knowing:

1. Bigger windows are not steadily better. 50 words kept the table together, 60 split it again, 80 kept it, 120 split it. Where a fixed cut lands is luck, and one edit to the document can change it.

2. Overlap repaired 3 of A's 4 misses, but indexed 27% more text, and the software table still lost its header.

3. Section headers helped two questions and hurt one: every chunk of the parental leave policy now starts with "Parental Leave Policy", so BM25 found them harder to tell apart. Measured, not assumed.

4. Stable IDs pay off on updates. Adding one sentence to the top of the travel policy changed all 6 fixed-size chunks, but only 1 of 5 section chunks. Only changed chunks need re-embedding.

The 13 questions are diagnostic, not a benchmark: each probes one way chunking can fail. Day 7 turns them into a proper evaluation set.

Everything runs offline: a notebook, a lesson and tests. Code from Days 1 and 2 now lives in a small shared package, with a test that proves it still matches the original notebooks.

https://github.com/shashanksuresh18/15-days-of-rag-from-scratch

Day 4: embeddings. Can dense retrieval find "time off for new dads" in a policy that only says "parental leave"?

#RAG #LLM #Python #AIEngineering

---

## Images — posting order, alt text and purpose

Upload in this order. LinkedIn's API documentation recommends alt text under 120 characters (maximum 4,086), so each alt text below leads with a short sentence that works on its own; paste the full text where the box allows.

| # | File | Teaches |
|---|---|---|
| 1 | `assets/day-03/slide-01-cover.png` | The core failure at a glance: a fixed-size cut separates a table row from its header, a structure-aware chunk keeps it, and the 13-question result. |
| 2 | `assets/day-03/slide-02-how-it-works.png` | The four rules of structure-aware chunking, and how a long table is split between rows with its header repeated. |
| 3 | `assets/day-03/slide-03-knobs.png` | What window size, overlap and section headers really change, measured on the same 13 questions. |
| 4 | `assets/day-03/slide-04-updates.png` | Why stable chunk IDs make updates cheap, and why every chunk must carry its document's metadata. |

**Alt text 1 — cover**
> A table cut by a fixed-size chunker loses its header; a section chunk keeps it. Question: what is the hotel limit per night in Singapore? Left, A, cut every 40 words: chunk travel-policy#w080 holds the header row City, Hotel per night, Meals per day, then London and half of the New York row; a scissors line cuts the table; chunk travel-policy#w120 starts with "300 | USD 75", then Singapore USD 220 USD 60 and Other cities. Caption: 220 or 60? No column header. Right, B, cut at headings: chunk travel-policy#daily-limits-by-city starts with "Travel Policy > Daily limits by city" and holds the whole table, Singapore row highlighted. Caption: USD 220 per night, one chunk. Bottom: 13 test questions with Day 01's BM25, top 3: complete evidence in one chunk for A 9 of 13 (misses: 3 table questions and one that spans 2 sections), 110 words read; B 13 of 13, 113 words read. Retrieval found the table both times. Only one chunk could answer.

**Alt text 2 — how it works**
> Four rules for structure-aware chunking. 1, split at headings: a ## line starts a section, text before it becomes Overview. 2, keep units whole: paragraphs, list items and whole tables, never cut mid-row. 3, pack up to a limit: 80 words here; a section that is too long is split between units. 4, stamp every chunk: Title > Section, a stable ID, permission groups and hashes. Below, real output with the limit set to 30 words: the travel table split into part 3 (header row, London, New York, Singapore) and part 4 (header row again, Other cities USD 180 USD 50). Without the repeated header, USD 180 would mean nothing. Parts 1 and 2 are the paragraph above the table, split between sentences.

**Alt text 3 — knobs**
> Bar chart: bigger fixed-size windows are not steadily better. Questions with the complete evidence in one chunk, out of 13, by window size with no overlap: 20 words 6, 30 words 8, 40 words 9, 50 words 12, 60 words 9, 80 words 12, 100 words 12, 120 words 11. The Singapore row stays with its column header only at 50, 80 and 100 words. Words read per question rise from 59 to 284. A dashed line marks structure-aware chunks: 13 of 13 at 113 words read. Overlap of 10 words on 40-word windows repaired 3 of 4 misses, stored 27% more text, and left the software table without its header. Title > Section headers helped 2 questions, hurt 1 (parental leave from rank 2 to 3), and added 18 words read per question. Fixed cuts land by luck: measure the knobs.

**Alt text 4 — updates**
> One edit, how many chunks change? Adding 7 words to the top of travel-policy and re-chunking changed all 6 fixed 40-word chunks, w000 to w200, because the new words push every boundary along; all six need re-embedding and the citation travel-policy#w080 now points at different text. With structure-aware chunks only 1 of 5 changed: purpose; booking, daily limits, claiming and safety are unchanged and keep their IDs. Every chunk carries its document's metadata: chunk_id travel-policy#daily-limits-by-city, allowed_groups all-employees, doc_hash and chunk_hash. The manager pay guide is 1 chunk restricted to managers and people-team; filters need this on every chunk on Day 05, and recording permissions is not enforcing them, which is Day 14. Next: Day 04, embeddings and similarity.

## Posting options

- **Image post (simplest):** attach the four PNGs in the order above and add the alt text to each.
- **Swipeable document post:** upload `assets/day-03/day-03-carousel.pdf` (the same four slides as a 4-page PDF). Document posts may not offer per-page alt text, so keep the image descriptions in the post body or first comment.

## Pre-posting checklist

- [ ] `day-03` merged into `main`, tag `day-03` pushed, GitHub Actions green on Linux, Windows and macOS
- [ ] Open the repository in a private/incognito window and check the README index links to the Day 03 lesson, notebook, slides and post
- [ ] Numbers in the post (11 records, 13 questions, 9 vs 13, 110 vs 113 words, 27%, 6 of 6 vs 1 of 5) still match the notebook output
- [ ] Alt text added to all four images
