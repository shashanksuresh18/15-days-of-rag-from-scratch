"""Day 10's query analysis and reviewed vocabulary map, packaged for reuse by later days.

The code is the one taught in notebooks/day_10_query_understanding_and_transformations.ipynb (sections 3
and 4): analyse() reads a question before it is searched with (its sentences; its values: numbers,
amounts, emails, URLs, #channels; its names, with gazetteer places typed; whether it is an exact value
typed as the query), and expand() appends the documents' words for the asker's words, using a reviewed
map of whole phrases and the abbreviations the documents define themselves. Day 11 reuses it to
reproduce Day 10's two map retrievers (BM25+map and weighted+map) before changing anything.

Day 12 adds Day 10's revised names rule (section 8): unknown_names_v2(), which skips every sentence's first word
and counts a gazetteer place as known when a source mentions its type ("Use the Other cities row for any city not
listed"), and NamesReader, Day 06's extractive reader with the names rule as a setting. Day 10 measured the rule
and didn't adopt it; Day 12 measures it again, end to end, on new held-out questions.

One change in shape: the notebook's expand() read its entries from a global, ENTRIES, built from the
chunks. Here the entries are an argument, so a later day builds them from its own chunks:

    entries = VOCABULARY + defined_abbreviations(chunks)   # the notebook's ENTRIES
    e = expand(question, entries)                          # notebook: expand(question)
    e.text, e.matched, e.route                             # what BM25 searches for, and why

The map was measured and not adopted on Day 10 (it met one of 41 new questions, whose answer was already
first): the default retriever doesn't use it. Like any query transformation, it changes what is searched
for, never who is searching.

tests/test_shared_code.py runs the Day 10 notebook's own cells and checks that this module returns
identical analyses, harvested abbreviations, phrase matches, expansions, unknown names and answers.
"""

import re
from dataclasses import dataclass

from .answering import ABSTAIN, NAME, SENTENCE, ExtractiveReader, Reply, answer_units, unknown_names

# ----------------------------------------------------------------------------- query analysis (Day 10, section 3)
# A stand-in gazetteer (a list of place names with their types). Chosen by an external rule, the G20's member
# countries and their capitals, so that no place is here because a test question mentions it.
COUNTRIES = ["Argentina", "Australia", "Brazil", "Canada", "China", "France", "Germany", "India", "Indonesia", "Italy",
             "Japan", "Mexico", "Russia", "Saudi Arabia", "South Africa", "South Korea", "Turkey", "United Kingdom",
             "United States"]
CAPITALS = ["Buenos Aires", "Canberra", "Brasilia", "Ottawa", "Beijing", "Paris", "Berlin", "New Delhi", "Jakarta",
            "Rome", "Tokyo", "Mexico City", "Moscow", "Riyadh", "Pretoria", "Seoul", "Ankara", "London", "Washington"]
PLACES = {**{c: "country" for c in COUNTRIES}, **{c: "city" for c in CAPITALS}}
PLACE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in sorted(PLACES, key=len, reverse=True)) + r")\b")
VALUE = re.compile(r"https?://\S+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+|#[\w-]+|USD \d[\d,]*(?:\.\d+)?|\d+(?:[.,:]\d+)*")


@dataclass(frozen=True)
class QueryAnalysis:
    question: str
    sentences: tuple[str, ...]
    values: tuple[str, ...]                    # numbers, amounts, emails, URLs, #channels: any rewrite must keep them
    names: tuple[tuple[str, str | None], ...]  # (name, place type or None): capitalised words, acronyms, places
    exact: bool                                # an exact value typed as the query: never expanded or rewritten


def question_names(question: str) -> tuple[tuple[str, str | None], ...]:
    # Day 06's NAME pattern, but skip the first word of EVERY sentence (Day 06 skipped only the question's first word),
    # and find gazetteer places first, as whole phrases ("New Delhi" is one name, not two)
    found = []
    for sentence in SENTENCE.split(question.strip()):
        places = [(m.start(), m.end(), m.group()) for m in PLACE.finditer(sentence)]
        for m in NAME.finditer(sentence):
            if m.start() == 0 or any(start <= m.start() < end for start, end, _ in places):
                continue                  # a sentence's first word is capitalised anyway; a place is handled whole
            found.append((m.group(), None))
        found += [(p, PLACES[p]) for start, _, p in places if start != 0]
    return tuple(found)


def analyse(question: str) -> QueryAnalysis:
    values = tuple(v.rstrip(".,!?") for v in VALUE.findall(question))
    exact = "?" not in question and len(question.split()) <= 4 and bool(re.search(r"\d|@|#|://", question))
    return QueryAnalysis(question, tuple(s for s in SENTENCE.split(question.strip()) if s), values,
                         question_names(question), exact)


# ----------------------------------------------------------------------------- the reviewed map (Day 10, section 4)
VOCABULARY = [   # (what the asker may say, what the documents say, where the gap was recorded)
    (("dad", "dads", "father", "fathers", "mum", "mums", "mom", "moms", "mother", "mothers"), ("caregivers",),
     "d01-01 (dev): dads vs caregivers; with the singular, a synonym and the counterparts"),
    (("time off",), ("leave",), "d01-01 (dev): time off vs leave"),
    (("wine", "beer", "spirits"), ("alcohol",), "d07-a08 (v1 test, used): wine vs alcohol"),
    (("pay for",), ("reimbursed",), "d07-a08 (v1 test, used): pay for vs reimbursed"),
    (("work from home", "working from home", "wfh"), ("remotely", "remote"),
     "d07-a03 (v1 test, used): work from home vs work remotely"),
    (("virus",), ("infected",), "d07-a11 (v1 test, used): virus vs infected"),
    (("switch off", "switch it off", "shut down", "shut it down", "power off"), ("turn off",),
     "d07-a11 (v1 test, used): switch off vs turn off"),
    (("locked",), ("locks", "lockout"), "d01-02 (dev): locked vs locks (BM25 has no stemming)"),
]
DEFINED = re.compile(r"\b([A-Z][a-z]+(?:[- ][a-z]+){0,4}) \(([A-Z]{2,})\)")    # "Long form (ABBR)"


def defined_abbreviations(chunks) -> list[tuple]:
    # The abbreviations the documents define themselves, both ways: no one has to write these entries
    pairs = sorted({m.groups() for c in chunks for m in DEFINED.finditer(c.text)})
    return [entry for long, short in pairs for entry in (
        ((short.lower(),), (long.lower(),), f"defined in the documents: {long} ({short})"),
        ((long.lower(),), (short.lower(),), f"defined in the documents: {long} ({short})"))]


def contains(text: str, phrase: str) -> bool:
    # A whole word or phrase, case-insensitive: "dad" is in "new dad?" but not in "Baghdad"
    return re.search(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)", text.lower()) is not None


@dataclass(frozen=True)
class Expansion:
    text: str                                      # what BM25 searches for: the question, then the added terms
    matched: tuple[tuple[str, tuple[str, ...]], ...]   # (phrase found in the question, terms it added)
    route: str                                     # "expanded", "no match" or "exact value: not expanded"


def expand(question: str, entries) -> Expansion:
    # Query expansion with a reviewed map: APPEND the documents' words; never remove or replace the asker's
    if analyse(question).exact:
        return Expansion(question, (), "exact value: not expanded")
    matched = [(next(p for p in phrases if contains(question, p)), terms) for phrases, terms, _ in entries
               if any(contains(question, p) for p in phrases)]
    added = list(dict.fromkeys(t for _, terms in matched for t in terms))
    return Expansion(f"{question} {' '.join(added)}" if added else question, tuple(matched),
                     "expanded" if added else "no match")


# ----------------------------------------------------------------------------- the revised names rule (Day 10, section 8)
TYPE_WORDS = {"city": ("city", "cities"), "country": ("country", "countries")}


def unknown_names_v2(question: str, sources) -> list[str]:
    # Day 06's names rule with the two changes: every sentence's first word skipped, and a gazetteer place counts as
    # known if a source mentions its TYPE ("Use the Other cities row for any city not listed")
    text = " ".join(s.text for s in sources)
    unknown = [name for name, kind in question_names(question)
               if not contains(text, name) and not (kind and any(contains(text, w) for w in TYPE_WORDS[kind]))]
    return list(dict.fromkeys(unknown))


class NamesReader(ExtractiveReader):
    """Day 06's extractive reader with the names rule as a setting. The rest of generate() is Day 06's, unchanged."""

    def __init__(self, embedder, names_rule=unknown_names, **settings):
        super().__init__(embedder, **settings)
        self.names_rule = names_rule

    def generate(self, context):
        if not context.sources:
            return Reply(ABSTAIN, self.name, "no sources")
        missing = self.names_rule(context.question, context.sources)
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
