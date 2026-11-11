"""Day 11's contexts, packaged for reuse by later days.

The code is the one taught in notebooks/day_11_advanced_retrieval_section_expansion.ipynb (sections 2, 3 and 8):
the documents' structure (each document's sections in order), section expansion with its per-query trace and its
permission check on every added section, and Day 06's answer steps fed a context chosen by the caller (Day 07's
gate on the dense top score, the budget, a reader, the citation check), with a sentence cache for the reader and
the reason a question was refused. Day 12 reuses them to reproduce Day 11's baseline and to run the current
pipeline end to end.

One change in shape: the notebook kept the sections, the records, the rankings, the directory and the settings in
global variables. Here they are arguments, with Day 06's directory, budget and top 3 as defaults:

    structure = DocumentStructure(chunks)                                  # notebook: SECTIONS and POSITION
    c = expand_sections(ranked, "alice", question, "neighbours", by_cid, structure)   # notebook: expand_sections()
    a = answer_context("alice", question, c.sent, by_cid, dict(fused), dense_top, reader, gate=0.62)
    why_refused(a)                                     # "gate", "names rule", "reader threshold", "citation check"

Day 11's Context dataclass is called SectionContext here, so it can't be confused with Day 06's Context
(the sources sent to a generator), which answer_context() builds with Day 06's assemble().

A lookup by position is not a search: expand_sections() checks every section it adds against the user's groups,
and answer_context() checks the whole context again before anything is sent (Day 11, section 6).

tests/test_shared_code.py runs the Day 11 notebook's own definitions and checks that this module returns identical
sections, contexts, traces, answers and reasons.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .answering import (ABSTAIN, BLOCKED, DIRECTORY, MAX_CONTEXT_WORDS, TOP_K, RagAnswer, assemble, check_citations,
                        count_words)
from .index import Result

SETTINGS = ("previous", "next", "neighbours", "parent")   # what an expansion may add around each hit ("off": nothing)


# ----------------------------------------------------------------------------- the documents' structure (section 2)
class DocumentStructure:
    """Each document's sections, in their order in the document (Day 03's chunks keep that order)."""

    def __init__(self, chunks):
        self.sections = defaultdict(list)        # document -> its sections, in order (notebook: SECTIONS)
        for c in chunks:
            self.sections[c.doc_id].append(c.chunk_id)
        self.position = {cid: (doc, i) for doc, ids in self.sections.items() for i, cid in enumerate(ids)}

    def where(self, cid: str) -> str:
        doc, i = self.position[cid]
        return f"section {i + 1} of {len(self.sections[doc])}"

    def around(self, cid: str, setting: str) -> list[tuple[str, str]]:
        # What a setting adds around one hit: a lookup in the document's structure, not a search
        doc, i = self.position[cid]
        ids = self.sections[doc]
        before, after = ids[max(0, i - 1):i], ids[i + 1:i + 2]
        return {"off": [],
                "previous": [(x, "previous") for x in before],
                "next": [(x, "next") for x in after],
                "neighbours": [(x, "previous") for x in before] + [(x, "next") for x in after],
                "parent": [(x, "parent") for x in ids if x != cid]}[setting]


# ----------------------------------------------------------------------------- section expansion (section 3)
@dataclass(frozen=True)
class Step:
    chunk: str                 # the section considered
    why: str                   # "hit 2", "previous of hit 3", "parent of hit 1", ...
    score: float | None        # its fused score, if the retriever ranked it
    words: int
    decision: str              # "sent", "already sent", "not readable" or "over budget"


@dataclass(frozen=True)
class SectionContext:
    question: str
    user: str
    setting: str
    sent: tuple[str, ...]      # what the reader gets, in this order: the hits, then what was added around them
    words: int
    trace: tuple[Step, ...]    # every section considered and what happened to it: the per-query trace


def may_read(cid: str, user: str, records, directory: dict | None = None) -> bool:
    # The same permission rule every retriever applies before ranking (Day 05): the user's groups from the directory
    directory = DIRECTORY if directory is None else directory
    return bool(set(records[cid].allowed_groups) & directory.get(user, set()))


def expand_sections(ranked, user: str, question: str, setting: str, records, structure: DocumentStructure,
                    budget: int = MAX_CONTEXT_WORDS, hits: int = TOP_K, directory: dict | None = None) -> SectionContext:
    # Section expansion: the hits first, then hit by hit the sections the setting adds around them. Skip a section
    # already sent, one the user may not read, and one that doesn't fit the budget (and try the next one).
    scores = dict(ranked)
    top = [cid for cid, _ in ranked[:hits]]
    candidates = [(cid, f"hit {k}") for k, cid in enumerate(top, start=1)] + \
                 [(x, f"{relation} of hit {k}") for k, cid in enumerate(top, start=1)
                  for x, relation in structure.around(cid, setting)]
    sent, steps, used = [], [], 0
    for cid, why in candidates:
        words = count_words(records[cid].text)
        if cid in sent:
            decision = "already sent"
        elif not may_read(cid, user, records, directory):   # a lookup isn't a search: the search's filter doesn't cover it
            decision = "not readable"
        elif used + words > budget:
            decision = "over budget"
        else:
            decision = "sent"
            sent.append(cid)
            used += words
        score = scores.get(cid)
        steps.append(Step(cid, why, None if score is None else round(score, 4), words, decision))
    return SectionContext(question, user, setting, tuple(sent), used, tuple(steps))


# ----------------------------------------------------------------------------- end to end with a context (section 8)
class SentenceCache:
    """The reader embeds every sentence of every source it gets, and contexts share most sources. This keeps
    each sentence's vector by its exact text, so it is embedded once (alone or in a batch gives the same vector)."""

    def __init__(self, embedder):
        self.embedder, self.vectors = embedder, {}
        self.model = embedder.model                       # what the reader may ask about the model

    def embed(self, texts):
        new = [t for t in dict.fromkeys(texts) if t not in self.vectors]
        if new:
            self.vectors.update(zip(new, self.embedder.embed(new)))
        return np.stack([self.vectors[t] for t in texts])

    def embed_query(self, question, *args, **kwargs):
        return self.embedder.embed_query(question, *args, **kwargs)


def answer_context(user: str, question: str, sent, records, fused_scores: dict, dense_top: float, reader, *,
                   gate: float, budget: int = MAX_CONTEXT_WORDS, directory: dict | None = None) -> RagAnswer:
    # Day 06's answer() with the context chosen by the caller: Day 07's gate on the dense top score (a fused score is
    # near 1 by construction), the permission check again before sending, the budget, the reader, the citation check
    directory = DIRECTORY if directory is None else directory
    groups = directory.get(user, set())
    if any(not set(records[cid].allowed_groups) & groups for cid in sent):
        raise PermissionError(f"a context holds a chunk {user} may not read")
    if dense_top < gate:
        return RagAnswer(user, question, "abstained", ABSTAIN, "below the dense gate: no generator called")
    context = assemble(question, [Result(i, fused_scores[cid], cid, records[cid]) for i, cid in enumerate(sent, 1)],
                       budget)
    reply = reader.generate(context)
    check = check_citations(reply.text, context)
    status = "abstained" if check.abstained else "blocked" if not check.passed else "answered"
    return RagAnswer(user, question, status, {"abstained": ABSTAIN, "blocked": BLOCKED}.get(status, reply.text),
                     reply.note, context, check, reply)


def why_refused(a: RagAnswer) -> str:
    # Which step refused: the gate, the names rule, the reader's threshold or the citation check
    if a.status == "blocked":
        return "citation check"
    return ("gate" if a.reason.startswith("below the dense gate") else "names rule" if a.reason.startswith("no source "
            "mentions") else "reader threshold" if a.reason.startswith("best sentence") else a.reason)
