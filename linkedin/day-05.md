# LinkedIn — Day 05: Local Vector Search and the Data Lifecycle

**Status:** ready to post once the `day-05` branch is merged into `main`, the tag `day-05` is pushed and the GitHub Actions run is green.
**Matches:** `docs/day-05/LEARN.md`, `notebooks/day_05_vector_search_and_lifecycle.ipynb` (outputs from the 2026-10-18 run).

---

## Post text

15 Days of RAG from Scratch · Day 5/15: Local vector search and data lifecycle

Leave one argument out of a search call, and the managers-only pay guide comes back to an employee as the #1 result.

Day 5 builds a local vector index in plain NumPy for the 44 chunks from Day 3: one vector and one record per chunk (text, permissions, review date, hashes), plus the model that made the vectors. Then it adds what changing documents need: filters, a safe save, updates and deletes.

What the notebook measured:

1. A filter isn't access control. The permission filter that hid the (synthetic) pay guide from an employee failed four ways: left out of a call, groups taken from the request, permissions changed but not yet synced, and the index file read directly. It holds only with groups from the directory, a filter that can't be skipped, and fast permission syncs.

2. Where the filter runs matters. Filtering after ranking returned 2 of 3 results for the employee, and 0 of 3 for a search scoped to one document. Filtering before ranking returned 3 both times.

3. Deletes are the step that gets forgotten. After Day 2's document changes plus a renamed heading, a sync re-embedded 3 of 41 chunks and matched a full rebuild exactly. An index that only inserts and updates kept 46 chunks, and a retired VPN guide filled the whole top 3 for "How do I connect to the VPN?".

4. Save atomically. A save that failed halfway destroyed the index when written in place. Written to a temporary file and swapped in, the old index survived. Loading with a different model revision is refused.

5. No vector database yet. Exact search over 100,000 random vectors took a few milliseconds, less than the model needs to embed one question. The smallest approximate-search library checked didn't even import on Python 3.13.

A teaching index on 11 short documents, not a production system. Day 14 goes deeper on access control.

Runs locally: a notebook, a lesson and tests. No new packages, no API key.

https://github.com/shashanksuresh18/15-days-of-rag-from-scratch

Day 6: the first end-to-end grounded answers, with citations.

#RAG #LLM #Python #AIEngineering

---

## Images — posting order, alt text and purpose

Upload in this order. LinkedIn's API documentation recommends alt text under 120 characters (maximum 4,086), so each alt text below leads with a short sentence that works on its own; paste the full text where the box allows.

| # | File | Teaches |
|---|---|---|
| 1 | `assets/day-05/slide-01-cover.png` | A permission filter hides a restricted chunk only while everything around it is right; four concrete ways it leaks, and what makes it hold. |
| 2 | `assets/day-05/slide-02-how-it-works.png` | What a vector index stores (vectors, records, the model), and why filtering after ranking returns fewer results than filtering before. |
| 3 | `assets/day-05/slide-03-sync-and-deletes.png` | How a sync follows document changes cheaply, and what an index that never deletes returns. |
| 4 | `assets/day-05/slide-04-when-to-add-a-library.png` | Measured exact-search cost against the install cost of vector libraries, and why this series doesn't add one yet. |

**Alt text 1 — cover**
> A permission filter isn't access control: four ways it leaks. Top: the question "When do pay rises take effect?" and its best match, the managers-only chunk manager-pay-review-guide#overview, ranked first with a score of 0.70; an employee's search with the filter applied doesn't return it. Below, four red cards on the same index: 1, the filter is forgotten (search called without it): the pay guide is back at #1, with no error. 2, groups taken from the request: an employee who claims the managers group makes exactly the same call as a manager. 3, permissions stale: the travel policy was made managers-only but the index wasn't synced, so the employee still gets it (the fix: 5 records updated, nothing re-embedded). 4, the file itself: loading index.npz gives all 44 chunk texts with no filter, plus vectors that can leak text. Bottom, what makes it hold: groups from the directory, a filter that can't be skipped, nothing for an unknown user, permissions synced fast. Deeper on Day 14.

**Alt text 2 — how it works**
> A vector index is vectors plus records plus the model. Top: one file, index.npz, 99,192 bytes, holding a 44 by 384 float32 vector matrix, one record per row keyed by chunk_id, and the model, bge-small at revision 5c38ec7c405e. One record shown: chunk_id manager-pay-review-guide#overview, a chunk hash, allowed groups managers and people-team, last reviewed None. Bottom, filter before you rank. An employee asks "When do pay rises take effect?": without a filter the top 3 are 0.70 (the managers-only chunk), 0.59 and 0.58; post-filtering removes the first and returns 2 of 3; pre-filtering returns 3 of 3 (0.59, 0.58, 0.57). "How do I connect to the VPN?" scoped to the remote work policy: without a filter the top 3 are all VPN guide chunks (0.80, 0.76, 0.74); post-filtering returns 0 of 3; pre-filtering returns 3 of 3 (0.71, 0.51, 0.51).

**Alt text 3 — sync and deletes**
> Sync, don't just upsert: Day 02's document changes plus a renamed heading, applied to the saved index. Sync plan by chunk_id and chunk_hash: add 2 (a new policy and the renamed heading), re-embed 1 (USD 500 changed to USD 600), update metadata 11 (contractors added, new document version), delete 5 (the retired VPN guide and the old heading), unchanged 27 (including a line-ending-only re-save). 3 of 41 texts went through the model, with the same result as a rebuild. Below, for "How do I connect to the VPN?" after the VPN guide was retired: an upsert-only index has 46 chunks and returns three retired VPN guide chunks (0.80, 0.76, 0.74); the fully synced index has 41 chunks and returns the remote work policy's security-while-remote section (0.71), then the new-starter checklist and the password standard. Bottom: atomic save, where a save that fails halfway leaves the old index loadable (written in place, the file is broken); and a delete guard that refuses to delete 44 of 44 chunks when an ingest returns nothing.

**Alt text 4 — when to add a library**
> Do you need a vector database? Exact NumPy search, measured, against the cost of a library. Top, time for an exact top 3 per question on a 2-vCPU cloud VM: 44 chunks 0.007 ms, 1,000 chunks 0.031 ms, 10,000 chunks 0.378 ms, 100,000 chunks 3.878 ms (153.6 MB of vectors); a dashed line marks embedding the question with the model, 5.8 ms. Middle, install size on top of NumPy, measured on 2026-10-18: sqlite-vec 0.2 MB, voyager 0.8 MB (version 2.2.2 didn't import on Python 3.13), usearch 41 MB, faiss-cpu 64 MB, chromadb 311 MB with 83 packages, lancedb 354 MB with 15 packages. Bottom: even 100,000 chunks searched faster than the model embeds one question, so no library yet; what carries over to any vector database is the records, filters and sync. Next: Day 06, grounded answers with citations.

## Posting options

- **Image post (simplest):** attach the four PNGs in the order above and add the alt text to each.
- **Swipeable document post:** upload `assets/day-05/day-05-carousel.pdf` (the same four slides as a 4-page PDF). Document posts may not offer per-page alt text, so keep the image descriptions in the post body or first comment.

## Pre-posting checklist

- [ ] `day-05` merged into `main`, tag `day-05` pushed, GitHub Actions green on Linux, Windows and macOS (CI's Linux runner may now be Ubuntu 26)
- [ ] Open the repository in a private/incognito window and check the README index links to the Day 05 lesson, notebook, slides and post
- [ ] Numbers in the post (44 chunks, 2 of 3 and 0 of 3, 3 of 41, 46 chunks, 100,000 vectors in a few milliseconds) still match the notebook output
- [ ] Alt text added to all four images
