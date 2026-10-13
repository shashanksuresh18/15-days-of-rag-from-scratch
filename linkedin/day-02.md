# LinkedIn — Day 02: Ingestion, Parsing and Metadata

**Status:** ready to post once the `day-02` branch is merged into `main` and the GitHub Actions run is green.
**Matches:** `docs/day-02/LEARN.md`, `notebooks/day_02_ingestion_and_metadata.ipynb` (outputs from the 2026-10-13 run, pypdf 6.20.1).

---

## Post text

15 Days of RAG from Scratch · Day 2/15: Ingestion, parsing and metadata

Most RAG tutorials start at embeddings. Day 2 starts one step earlier, because retrieval can only find what ingestion kept.

I took the 8 tidy Markdown policies from Day 1 and added what real knowledge bases look like: an intranet HTML page, a PDF with a table, a scanned PDF and two plain-text files (all synthetic, with common quirks built in on purpose). Then I built one pipeline that turns all 13 files into the same kind of record: clean text, plus metadata about what it is, who may read it and where it came from.

Result: 11 records, and 2 files quarantined, each with a reason.

The most instructive part was the PDF table. pypdf's default extraction returned the cells column by column: all the cities, then all the hotel limits, then all the meal limits. The rows were gone, so "What's the hotel limit in Singapore?" no longer has an answer in the text. Layout mode kept each row on one line. Same file, same library, one parameter.

Four more lessons I'm carrying forward:

1. Measure your cleaning. Removing the repeated header and footer took 4 noise lines out of a 2-page PDF, and re-joining "reim-" / "bursement" made the phrase "claim reimbursement" findable again.

2. Permissions come from the system of record, never from the file. A dropped-in draft said "please share widely". It had no entry in the permission export, so it stays invisible to everyone. Fail closed.

3. A scanned PDF can look like a document and contain zero characters of text. Detect it and route it to OCR instead of indexing an empty record.

4. Keep two identifiers: a document ID says which document, a content hash says which version. Comparing them between runs tells you what to add, re-index, update or delete. A file re-saved with Windows line endings changed on disk but not in content, so it is skipped.

Everything runs offline: standard-library parsers plus pypdf, a notebook, a lesson and tests.

https://github.com/shashanksuresh18/15-days-of-rag-from-scratch

Day 3 is chunking: where you cut a document decides what retrieval can find.

#RAG #LLM #Python #AIEngineering

---

## Images — posting order, alt text and purpose

Upload in this order. LinkedIn's API documentation recommends alt text under 120 characters (maximum 4,086), so each alt text below leads with a short sentence that works on its own; paste the full text where the box allows.

| # | File | Teaches |
|---|---|---|
| 1 | `assets/day-02/slide-01-cover.png` | The whole ingestion pipeline at a glance: 13 files, four steps, 11 records with metadata, 2 quarantined with reasons. |
| 2 | `assets/day-02/slide-02-metadata-origins.png` | Where each metadata field should come from, and why permissions must come from the system of record and fail closed. |
| 3 | `assets/day-02/slide-03-pdf-tables.png` | Why PDF tables break in text extraction, shown with real pypdf output in default and layout mode. |
| 4 | `assets/day-02/slide-04-freshness.png` | Document ID versus content hash, and the five change outcomes an ingestion run must handle. |

**Alt text 1 — cover**
> Hand-drawn diagram of a RAG ingestion pipeline turning 13 files into 11 records. Left: 13 files arrive: 8 Markdown, 1 HTML page, 1 PDF, 1 scanned PDF, 2 plain text. Middle: one pipeline with four steps: parse (one adapter per format), clean (headers, hyphens, tables), check (permissions and metadata), stamp (ID, hash, parser). Right: one real record, travel-policy, with title Travel Policy, owner Finance, reviewed 2026-04-15, groups all-employees, parser pypdf 6.20.1 and a SHA-256 hash. Below: 2 quarantined files with reasons: scanned-visitor-policy.pdf has no text layer and needs OCR; draft-reorg-announcement.txt has no permission entry, so it fails closed. Note: retrieval can only find what ingestion kept.

**Alt text 2 — metadata origins**
> Diagram of where each metadata field comes from, in three zones. The document itself (front matter, meta tags, PDF properties) gives title, owner and last_reviewed: any author can edit it, so validate it; for example the intranet page's review is 470 days overdue. The system of record, the permission export, gives allowed_groups: authoritative, and missing means fail closed. A dropped-in draft says "Audience: all staff (please share widely)", the export has no entry, so it is visible to nobody. The pipeline adds doc_id, source_path, parser and content_hash, which trace any answer to one file, parser version and text version.

**Alt text 3 — PDF tables**
> One PDF table extracted two ways by pypdf 6.20.1. The table as seen: City, Hotel per night, Meals per day, with rows London USD 250 USD 75, New York USD 300 USD 75, Singapore USD 220 USD 60, Other cities USD 180 USD 50. Default plain mode follows the drawing order and returns one stream: all cities, then all hotel limits, then all meal limits, so rows are lost. Layout mode plus cleaning keeps each row on one line, such as "Singapore | USD 220 | USD 60". Two more traps: "claim reim- / bursement" is re-joined so the phrase is found, and the header and footer on each page go from 4 lines to 0.

**Alt text 4 — freshness**
> Diagram of document identity versus version. doc_id answers which document and stays when text changes, for example remote-work-policy. content_hash answers which version and changes with one character: the USD 500 and USD 600 sentences hash differently. A real run made 5 changes to a copy of the data and re-ingested: added annual-leave-policy, chunk and index; text changed remote-work-policy, re-chunk and re-index; permissions changed password-and-mfa-standard, update metadata only; removed vpn-access-guide, delete its chunks; unchanged 8 documents including one re-saved with Windows line endings, skip.

## Posting options

- **Image post (simplest):** attach the four PNGs in the order above and add the alt text to each.
- **Swipeable document post:** upload `assets/day-02/day-02-carousel.pdf` (the same four slides as a 4-page PDF). Document posts may not offer per-page alt text, so keep the image descriptions in the post body or first comment.

## Pre-posting checklist

- [ ] `day-02` merged into `main`, tag `day-02` pushed, GitHub Actions green on Linux, Windows and macOS
- [ ] Open the repository in a private/incognito window and check the README index links to the Day 02 lesson, notebook, slides and post
- [ ] Numbers in the post (8 Markdown policies, 13 files, 11 records, 2 quarantined, 4 header/footer lines, the Singapore row) still match the notebook output
- [ ] Alt text added to all four images
