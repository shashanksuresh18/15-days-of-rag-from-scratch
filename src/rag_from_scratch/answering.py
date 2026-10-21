"""Day 06's grounded answering pipeline, packaged for reuse by later days.

The code is the one taught in notebooks/day_06_grounded_rag.ipynb (sections 2, 3, 5, 6 and 7):
retrieve as a specific user (groups from a directory, the permission filter always on), a gate
that abstains without calling any generator, a budgeted context of whole chunks with citation ids,
the API-free extractive reader, the citation check, and answer(), which runs the steps and says
what happened (answered, abstained, blocked or unavailable).

One change in shape: the notebook kept the index, the embedder, the user directory and the
thresholds in global variables. Here they are arguments, with the notebook's values as defaults,
so a later day can measure the thresholds instead of editing globals (Day 07 does):

    rag = Pipeline(index, embedder)                       # DIRECTORY, MIN_RETRIEVAL_SCORE = 0.50, 300 words
    results = rag.retrieve("alice", question, k=10)       # as the notebook's retrieve()
    reader = ExtractiveReader(embedder)                   # min_score = 0.60, margin = 0.03, 3 sentences
    a = rag.answer("alice", question, reader)             # as the notebook's answer()

Day 06's prompt (INSTRUCTIONS, build_messages) and its ChatModel client stay in the notebook until a
later day needs them; answer() accepts any object with a generate(context) -> Reply method.

tests/test_shared_code.py runs the Day 06 notebook's own code and checks that this module returns
identical results, contexts, replies, checks and answers, also with other threshold values.
"""

import re
from dataclasses import dataclass, field

import numpy as np

from .embeddings import Embedder
from .index import Result, VectorIndex, groups_filter

TOP_K = 3                       # chunks retrieved per question, as on Days 01, 03, 04 and 05
MIN_RETRIEVAL_SCORE = 0.50      # gate: if no chunk scores this high, abstain without calling any generator
MAX_CONTEXT_WORDS = 300         # budget for the sources sent to a generator; whole chunks only
ABSTAIN = "I don't know based on the provided documents."   # the one reply that means "no answer"
BLOCKED = "I couldn't verify an answer against the documents, so I won't guess."
UNAVAILABLE = "The answering service is unavailable right now. Please try again later."

# Stands in for your identity provider, read on every request (Day 05, section 6)
DIRECTORY = {"alice": {"all-employees"}, "maria": {"all-employees", "managers"}}


# ----------------------------------------------------------------------------- step 2: assemble (Day 06, section 3)
@dataclass(frozen=True)
class Source:
    """One retrieved chunk, as the generator reads it and as a citation points to it."""
    id: str                   # the chunk_id, used as the citation
    title: str
    section: str
    last_reviewed: str | None
    text: str                 # unchanged chunk text
    score: float              # retrieval score, for the log
    source_path: str          # shown to the user next to the answer; not sent to the model


@dataclass(frozen=True)
class Context:
    question: str
    sources: tuple[Source, ...]
    dropped: tuple[str, ...] = ()     # retrieved but not sent: a repeated text, or over the budget


def count_words(text: str) -> int:
    return len(text.split())


def assemble(question: str, results: list[Result], max_words: int = MAX_CONTEXT_WORDS) -> Context:
    # Best first, whole chunks only, no repeated text, and skip any chunk that would exceed the budget.
    sources, dropped, seen, used = [], [], set(), 0
    for r in results:
        c = r.chunk
        if c.chunk_hash in seen or used + count_words(c.text) > max_words:
            dropped.append(r.id)
            continue
        seen.add(c.chunk_hash)
        used += count_words(c.text)
        sources.append(Source(c.chunk_id, c.title, c.section, c.last_reviewed, c.text, r.score, c.source_path))
    return Context(question, tuple(sources), tuple(dropped))


# ----------------------------------------------------------------------------- step 4: generate (Day 06, section 5)
@dataclass(frozen=True)
class Reply:
    text: str
    generator: str
    note: str = ""                                  # why it answered or abstained, when the generator can say
    usage: dict = field(default_factory=dict)       # tokens in and out, when a model reports them


SENTENCE = re.compile(r"(?<=[.!?])\s+")
LIST_ITEM = re.compile(r"^(\d+\.|-)\s")
NAME = re.compile(r"\b(?:[A-Z][a-z]+(?:-[A-Z][a-z]+)*|[A-Z]{2,})\b")   # Google, Wi-Fi, CEO


def answer_units(source: Source) -> list[str]:
    # The pieces an extractive answer can copy: sentences, list items, and table rows paired with their header.
    header = f"{source.title} > {source.section}"
    lines = [line for line in source.text.splitlines() if line.strip() and line != header]
    units, i = [], 0
    while i < len(lines):
        if " | " in lines[i]:                          # a table: pair every row with the header row
            j = i
            while j < len(lines) and " | " in lines[j]:
                j += 1
            head = lines[i].split(" | ")
            units += [" | ".join(f"{h}: {cell}" for h, cell in zip(head, row.split(" | "))) for row in lines[i + 1:j]]
            i = j
        else:
            units += [lines[i]] if LIST_ITEM.match(lines[i]) else [s for s in SENTENCE.split(lines[i]) if s]
            i += 1
    return units


def unknown_names(question: str, sources) -> list[str]:
    # Names and acronyms in the question (except its first word) that none of the sources mentions.
    text = " ".join(s.text for s in sources).lower()
    names = [n for n in NAME.findall(question) if not question.startswith(n)]
    return [n for n in names if not re.search(rf"\b{re.escape(n.lower())}\b", text)]


MIN_SENTENCE_SCORE = 0.60   # below this, no piece is close enough to the question
SENTENCE_MARGIN = 0.03      # also copy pieces of the same section that score within this of the best one
MAX_SENTENCES = 3


class ExtractiveReader:
    """API-free stand-in for a language model: copies the best-matching sentences from the sources, with citations."""
    name = "extractive reader (API-free)"

    def __init__(self, embedder: Embedder, min_score: float = MIN_SENTENCE_SCORE, margin: float = SENTENCE_MARGIN,
                 max_sentences: int = MAX_SENTENCES):
        self.embedder = embedder
        self.min_score, self.margin, self.max_sentences = min_score, margin, max_sentences

    def scored_units(self, context: Context) -> list[tuple[float, Source, str]]:
        # Every sentence or table row of every source, scored against the question, best first.
        units = [(s, u) for s in context.sources for u in answer_units(s)]
        vectors = self.embedder.embed([f"{s.title} > {s.section}: {u}" for s, u in units])
        scores = vectors @ self.embedder.embed_query(context.question)
        return [(float(scores[j]), *units[j]) for j in np.argsort(-scores, kind="stable")]

    def generate(self, context: Context) -> Reply:
        if not context.sources:
            return Reply(ABSTAIN, self.name, "no sources")
        missing = unknown_names(context.question, context.sources)
        if missing:                                    # the question is about something the sources never mention
            return Reply(ABSTAIN, self.name, f"no source mentions {', '.join(missing)}")
        ranked = self.scored_units(context)
        best_score, best_source, _ = ranked[0]
        if best_score < self.min_score:                # nothing in the sources is close enough
            return Reply(ABSTAIN, self.name, f"best sentence scores {best_score:.2f} < {self.min_score}")
        picked = [u for score, s, u in ranked if s.id == best_source.id and score >= best_score - self.margin]
        order = answer_units(best_source)
        picked = sorted(picked[:self.max_sentences], key=order.index)        # in the document's order
        text = " ".join(f"{u.rstrip('.')} [{best_source.id}]." for u in picked)
        return Reply(text, self.name, f"best sentence scores {best_score:.2f} >= {self.min_score}")


# ----------------------------------------------------------------------------- step 5: check (Day 06, section 6)
CITATION = re.compile(r"\[([^\[\]]+)\]")
NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")


@dataclass(frozen=True)
class Claim:
    text: str                    # one sentence of the answer, without its citations
    cited: tuple[str, ...]       # the ids it cites
    problems: tuple[str, ...]    # empty if every check passed


@dataclass(frozen=True)
class Check:
    abstained: bool
    claims: tuple[Claim, ...]

    @property
    def passed(self) -> bool:
        return self.abstained or (bool(self.claims) and not self.problems)

    @property
    def problems(self) -> list[str]:
        return [p for c in self.claims for p in c.problems]


def split_claims(text: str) -> list[str]:
    # Sentences (and list lines); a citation written after the full stop is moved back to its sentence.
    pieces = [p.strip() for line in text.splitlines() for p in SENTENCE.split(line.strip().lstrip("-*• "))]
    claims = []
    for piece in filter(None, pieces):
        leading = re.match(r"(?:\[[^\[\]]+\]\s*)+", piece)
        if leading and claims:                        # "...leave. [id] Next sentence": the [id] belongs before
            claims[-1] += " " + leading.group().strip()
            piece = piece[leading.end():].strip()
        if piece:
            claims.append(piece)
    return claims


def check_citations(text: str, context: Context) -> Check:
    if ABSTAIN.rstrip(".").lower() in text.replace("\u2019", "'").lower():   # the generator found no answer
        return Check(True, ())
    sources = {s.id: s for s in context.sources}
    claims = []
    for sentence in split_claims(text):
        cited = tuple(dict.fromkeys(i.strip() for group in CITATION.findall(sentence)
                                    for i in re.split(r"[,;]", group) if i.strip()))
        claim = re.sub(r"\s+([.!?,;:])", r"\1", CITATION.sub("", sentence)).strip()
        if not re.search(r"\w", claim):
            continue                                  # nothing left but punctuation
        problems = [] if cited else ["no citation"]
        problems += [f"[{i}] was not among the sources sent" for i in cited if i not in sources]
        known = " ".join(sources[i].text for i in cited if i in sources)
        if known:
            in_sources = set(NUMBER.findall(known))
            problems += [f"{n} is not in the cited source" for n in dict.fromkeys(NUMBER.findall(claim))
                         if n not in in_sources]
        claims.append(Claim(claim, cited, tuple(problems)))
    return Check(False, tuple(claims))


# ----------------------------------------------------------------------------- the whole pipeline (Day 06, section 7)
class GeneratorError(RuntimeError):
    """A generator could not produce a reply (network, authentication, limits). Never contains an API key."""


@dataclass(frozen=True)
class RagAnswer:
    user: str
    question: str
    status: str              # answered | abstained | blocked | unavailable
    text: str                # what the user sees
    reason: str
    context: Context | None = None
    check: Check | None = None
    reply: Reply | None = None


class Pipeline:
    """Day 06's retrieve() and answer(), with the index, the model, the directory and the thresholds as settings."""

    def __init__(self, index: VectorIndex, embedder: Embedder, directory: dict | None = None,
                 min_retrieval_score: float = MIN_RETRIEVAL_SCORE, max_context_words: int = MAX_CONTEXT_WORDS):
        self.index, self.embedder = index, embedder
        self.directory = DIRECTORY if directory is None else directory
        self.min_retrieval_score, self.max_context_words = min_retrieval_score, max_context_words

    def retrieve(self, user_id: str, question: str, k: int = TOP_K, from_index: VectorIndex | None = None) -> list[Result]:
        # The pipeline's only way into the index. `user_id` comes from the verified login, never from the request:
        # the groups come from the directory, the permission filter is always applied, an unknown user gets nothing.
        groups = self.directory.get(user_id, set())
        if not groups:
            return []
        store = self.index if from_index is None else from_index
        return store.search(self.embedder.embed_query(question), k, where=groups_filter(groups))

    def answer(self, user_id: str, question: str, generator, from_index: VectorIndex | None = None) -> RagAnswer:
        results = self.retrieve(user_id, question, from_index=from_index)                     # 1. as this user
        groups = self.directory.get(user_id, set())
        if any(not set(r.chunk.allowed_groups) & groups for r in results):                    #    checked again
            raise PermissionError(f"retrieval returned a chunk {user_id} may not read")       #    before sending
        if not results:                                                                       # 2. gate
            return RagAnswer(user_id, question, "abstained", ABSTAIN, "nothing retrieved: no generator called")
        if results[0].score < self.min_retrieval_score:
            return RagAnswer(user_id, question, "abstained", ABSTAIN,
                             f"best retrieval score {results[0].score:.2f} < {self.min_retrieval_score}: "
                             "no generator called")
        context = assemble(question, results, self.max_context_words)                          # 3.
        try:
            reply = generator.generate(context)                                               # 4.
        except GeneratorError as error:
            return RagAnswer(user_id, question, "unavailable", UNAVAILABLE, str(error), context)
        check = check_citations(reply.text, context)                                          # 5.
        if check.abstained:
            return RagAnswer(user_id, question, "abstained", ABSTAIN,
                             f"{reply.generator}: {reply.note or 'no answer in the sources'}", context, check, reply)
        if not check.passed:
            return RagAnswer(user_id, question, "blocked", BLOCKED, "; ".join(check.problems), context, check, reply)
        return RagAnswer(user_id, question, "answered", reply.text,
                         f"{reply.generator}: every sentence cites a source it was given", context, check, reply)


def cited_sources(a: RagAnswer) -> list[Source]:
    sources = {s.id: s for s in a.context.sources} if a.context else {}
    return [sources[i] for i in dict.fromkeys(i for c in (a.check.claims if a.check else ()) for i in c.cited)
            if i in sources]


class FixedReply:
    """A generator that always returns the same text: for testing the pipeline with hand-written replies."""
    name = "hand-written reply"

    def __init__(self, text: str):
        self.text = text

    def generate(self, context: Context) -> Reply:
        return Reply(self.text, self.name)


def grade(a: RagAnswer, facts) -> str:
    # Day 06's check-list grade: facts is a tuple of strings a correct answer contains, or None if the documents
    # don't answer the question for this user. Case-insensitive containment: a rough check, not a judgement.
    if facts is None:
        return "correctly abstained" if a.status == "abstained" else "answered, but nothing answers it"
    if a.status != "answered":
        return f"{a.status}, but the documents answer it"
    return "correct" if all(f.lower() in a.text.lower() for f in facts) else "wrong or incomplete"
