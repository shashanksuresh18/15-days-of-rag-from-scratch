"""Day 01's BM25 keyword retriever, packaged for reuse by later days.

The scoring is the code taught in notebooks/day_01_rag_foundations.ipynb (Lucene-style IDF,
k1 = 1.2, b = 0.75, the same tokenizer and stopwords, no stemming). One change in shape: the
Day 01 class indexed Chunk objects, this one indexes any {id: text} mapping, so each lesson
decides what text goes into the index (for example with or without section headings).

tests/test_shared_code.py runs the Day 01 notebook's own code and checks that this module
gives identical scores and rankings on Day 01's chunks.

Standard library only.
"""

import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

BM25_K1 = 1.2    # term-frequency saturation (Apache Lucene's documented default)
BM25_B = 0.75    # document-length normalisation (Apache Lucene's documented default)

STOPWORDS = frozenset('''
a an and are as at be by can could do does for from had has have how i if in into is it its
me my of on or our s so t than that the their them then there these they this to was we were
what when where which who why will with would you your
'''.split())

TOKEN_PATTERN = re.compile(r"[^\W_]+")  # runs of Unicode letters/digits


def tokenize(text: str) -> list[str]:
    # Lowercase, split on non-alphanumerics, drop stopwords. Deliberately no stemming.
    return [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    id: str
    contributions: dict[str, float]  # how much each matched query term added to the score


class BM25Index:
    # A tiny in-memory BM25 index built from scratch.
    #   Index time: tokenize each text, count terms, build the inverted index.
    #   Query time: find candidates through the inverted index, score them, sort.

    def __init__(self, texts: Mapping[str, str], k1: float = BM25_K1, b: float = BM25_B):
        self.ids = list(texts)
        self.k1, self.b = k1, b
        tokens = [tokenize(texts[i]) for i in self.ids]
        self.term_counts = [Counter(t) for t in tokens]  # per text: term -> count
        self.lengths = [len(t) for t in tokens]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        postings: dict[str, list[int]] = {}               # the inverted index: term -> positions
        for i, counts in enumerate(self.term_counts):
            for term in counts:
                postings.setdefault(term, []).append(i)
        self.postings = postings

    def idf(self, term: str) -> float:
        n = len(self.postings.get(term, ()))
        N = len(self.ids)
        return math.log(1 + (N - n + 0.5) / (n + 0.5))

    def term_score(self, term: str, i: int) -> float:
        tf = self.term_counts[i].get(term, 0)
        if tf == 0:
            return 0.0
        length_norm = 1 - self.b + self.b * self.lengths[i] / self.avg_length
        return self.idf(term) * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        terms = list(dict.fromkeys(tokenize(query)))  # unique terms, in query order
        candidates = sorted({i for t in terms for i in self.postings.get(t, ())})
        scored = []
        for i in candidates:
            contributions = {t: self.term_score(t, i) for t in terms if t in self.term_counts[i]}
            scored.append((sum(contributions.values()), self.ids[i], contributions))
        scored.sort(key=lambda item: (-item[0], item[1]))  # tie-break on id => deterministic
        return [Hit(rank, score, item_id, contributions)
                for rank, (score, item_id, contributions) in enumerate(scored[:k], start=1)]
