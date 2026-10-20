"""Shared code for the 15 Days of RAG from Scratch series.

Each technique is first taught step by step inside its day's notebook. When a later day needs
it again, a copy moves here so the new notebook can import it instead of repeating it:

- ingest.py:   Day 02's ingestion pipeline (files -> Document records)
- bm25.py:     Day 01's BM25 keyword retriever
- chunking.py: Day 03's structure-aware chunker (Document records -> Chunk records)
- probes.py:   Day 03's 13 diagnostic questions with their evidence strings
- embeddings.py: Day 04's pinned local embedding model (Embedder: text -> unit-length vectors)
- index.py:   Day 05's exact vector index (VectorIndex, filters, save/load, sync)
- corpus.py:  Day 05's corpus_after(): chunks of a temporarily changed copy of the sample data

tests/test_shared_code.py checks each module against the notebook it came from.
"""
