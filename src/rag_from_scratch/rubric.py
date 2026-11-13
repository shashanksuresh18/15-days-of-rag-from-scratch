"""Day 12's answer rubric, packaged for reuse by later days.

The code is the one taught in notebooks/day_12_answer_evaluation_and_regression_tests.ipynb (sections 3, 5 and 11):
content words and a lexical support score, one verdict per answer (Grade: the outcome against the question's facts,
the step that refused, whether a labelled chunk was sent and cited, citation precision, support, groundedness and the
stage that lost the answer), the value of an answer to the person asking (a wrong answer costs twice what a correct
one gains), the closest call that decided an answer, and the comparison of a run with an accepted answer baseline.
Day 13 reuses them to reproduce Day 12's baseline and to grade what a cache serves.

One change in shape: the notebook kept the answers, the readers, the rankings and the settings in global variables.
Here they are arguments:

    g = grade(answer, q, sent)                         # sent: the chunk ids the context held (or would have held)
    value(g.outcome)                                   # +1, 0 or -2 (cost 2)
    better_or_worse(q, g_new.outcome, g_old.outcome)   # notebook: better_or_worse(q, "names-v2", "current")
    closest_call(answer, reader, dense_top, gate, sections_gap)   # notebook: closest_call(q, pipeline)
    compare_answers(run, accepted, ("current", "names-v2"))       # notebook: compare_answers(run, accepted)

Its limits (Day 12, section 3): containment passes a list of table rows that includes the right one and a fact inside
a longer one ("6 weeks" in "16 weeks"), fails a correct paraphrase, and "grounded" is lexical.

tests/test_shared_code.py runs the Day 12 notebook's own definitions and checks that this module gives identical
verdicts, values, closest calls and comparisons.

Standard library only.
"""

import dataclasses
import math
import re
from dataclasses import dataclass

from .answering import SENTENCE
from .contexts import why_refused
from .evaluation import TIE, outcome

SUPPORT_MIN = 0.75      # "grounded": every sentence has at least this share of its words in its cited sources
COST = 2                # a wrong answer costs twice what a correct one gains (Day 12's pre-registered assumption)

# ----------------------------------------------------------------------------- the rubric (section 3)
STOPWORDS = set("""a an and are as at be but by can could do does for from had has have how i if in into is it its
may me must my no not of on or our should so than that the their them then there these they this to up was we were
what when where which who why will with would you your""".split())
WORD = re.compile(r"[a-z0-9]+(?:[.,:/@-][a-z0-9]+)*")      # words, numbers, amounts, emails, URLs, times


def content_words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower())
            if w not in STOPWORDS and (len(w) >= 3 or any(ch.isdigit() for ch in w))]


def support(claim: str, sources) -> float:
    # A lexical stand-in for "the cited sources support this sentence": the share of its content words they contain
    claim_words = content_words(claim)
    known = set(content_words(" ".join(s.text for s in sources)))
    return sum(w in known for w in claim_words) / len(claim_words) if claim_words else 1.0


@dataclass(frozen=True)
class Grade:
    outcome: str                      # correct | wrong | refused (answerable); refused | answered (unanswerable)
    refused_by: str | None            # gate | names rule | reader threshold | citation check | the model (live)
    label_in_context: bool | None     # answerable: was a labelled chunk sent (or about to be, for a gate refusal)?
    cites_label: bool | None          # answered and answerable: does it cite a labelled chunk?
    citation_precision: float | None  # answered and answerable: labelled chunks cited / chunks cited
    support: float | None             # answered: the lowest support of any of its sentences
    grounded: bool | None             # answered: passed the citation check and every sentence has support >= 0.75
    stage: str                        # "ok", or where it went wrong


GRADE_FIELDS = tuple(f.name for f in dataclasses.fields(Grade))       # outcome, refused_by, ..., stage


def first_words(question: str) -> set[str]:
    # The first word of every sentence after the first (Day 06's names rule skips only the question's first word)
    return {s.split()[0] for s in SENTENCE.split(question.strip())[1:] if s.split()}


def found(sent, q) -> bool:
    return any(cid in q.answer_chunks for cid in sent)


def grade(a, q, sent) -> Grade:
    # The rubric: one verdict per answer, from the answer, the context it was given, the labels and the facts
    out = outcome(a, q)
    refused_by = (why_refused(a) or "the model") if a.status != "answered" else None     # "the model": it abstained
    label_in = found(sent, q) if q.answerable else None
    cites_label = precision = sup = grounded = None
    if a.status == "answered":
        sources = {s.id: s for s in a.context.sources}
        cited = list(dict.fromkeys(i for c in a.check.claims for i in c.cited))
        sup = round(min(support(c.text, [sources[i] for i in c.cited if i in sources]) for c in a.check.claims), 3)
        grounded = a.check.passed and sup >= SUPPORT_MIN
        if q.answerable:
            cites_label = bool(set(cited) & set(q.answer_chunks))
            precision = round(sum(i in q.answer_chunks for i in cited) / len(cited), 3)
    if q.answerable:
        if out == "correct":
            stage = "ok"
        elif out == "refused":
            if refused_by == "names rule":
                names = set(a.reason.removeprefix("no source mentions ").split(", "))
                stage = "names rule: a sentence's first word" if names <= first_words(q.question) else "names rule: a name"
            else:
                stage = refused_by
        elif not label_in:
            stage = "retrieval: answer not in the context"
        elif cites_label:
            stage = "reader: right source, wrong part"
        else:
            stage = "reader: answer in the context, not used"
    else:
        stage = "ok" if out == "refused" else \
            "answered: restricted to others" if q.restricted_chunks else "answered: nothing answers it"
    return Grade(out, refused_by, label_in, cites_label, precision, sup, grounded, stage)


def value(out: str, cost: float = COST) -> float:
    # What an answer is worth to the person asking (pre-registered): correct +1, refused 0, wrong or answered -cost
    return {"correct": 1, "refused": 0, "wrong": -cost, "answered": -cost}[out]


# ----------------------------------------------------------------------------- comparing two pipelines (section 5)
def better_or_worse(q, outcome_a: str, outcome_b: str) -> int:
    # +1 if outcome a is better for the person asking than outcome b, -1 if worse, 0 if the same
    rank = {"correct": 2, "refused": 1, "wrong": 0} if q.answerable else {"refused": 1, "answered": 0}
    return (rank[outcome_a] > rank[outcome_b]) - (rank[outcome_a] < rank[outcome_b])


def closest_call(a, reader, dense_top: float, gate: float, sections_gap: float) -> tuple[float, str]:
    # The smallest gap that decided this answer: another CPU's rounding (gaps under TIE) could flip it. sections_gap:
    # the fused scores' gap between the last section sent and the next one (the notebook's cutoff_gap(q, 3)).
    calls = [(abs(dense_top - gate), "gate"), (sections_gap, "which sections are sent")]
    if a.context is not None and a.reason.startswith("best sentence"):
        ranked = reader.scored_units(a.context)
        best, source = ranked[0][0], ranked[0][1].id
        calls.append((abs(best - reader.min_score), "reader threshold"))
        other = next((s for s, src, _ in ranked if src.id != source), None)
        if other is not None:
            calls.append((best - other, "which source is copied"))
        calls += [(abs(s - (best - reader.margin)), "which sentences are copied")
                  for s, src, _ in ranked[1:] if src.id == source]
    return min(calls)


# ----------------------------------------------------------------------------- the answer baseline (section 11)
def answer_entry(a, g: Grade, call: tuple[float, str]) -> dict:
    # One pipeline's entry for one question in an answer baseline
    gap, what = call
    return {"status": a.status, "text": a.text, "reason": a.reason,
            "cited": list(dict.fromkeys(i for c in a.check.claims for i in c.cited)) if a.check else [],
            **dataclasses.asdict(g), "closest_call": [round(gap, 4) if gap != math.inf else None, what]}


def compare_answers(new: dict, old: dict, pipelines) -> tuple[list[str], list[str]]:
    # What changed since the accepted answer baseline: (changes, near ties). Statuses, answer texts, citations and every
    # field of the verdict must match exactly; a change where either run's closest call is under TIE is CPU rounding.
    if new["dataset"] != old["dataset"]:
        return ["different labels or a different snapshot: these two runs can't be compared"], []
    changes = ["pipeline settings or the rubric changed"] if (new["pipelines"], new["rubric"]) != \
        (old["pipelines"], old["rubric"]) else []
    ties = []
    for qid, now in new["questions"].items():
        then = old["questions"].get(qid)
        if then is None:
            changes.append(f"{qid}: not in the baseline")
            continue
        for p in pipelines:
            fields = [f for f in ("status", "text", "cited", *GRADE_FIELDS) if now[p][f] != then[p][f]]
            if fields:
                moved = f"{qid} {p}: {', '.join(fields)} changed ({then[p]['outcome']} -> {now[p]['outcome']})"
                calls = [x[p]["closest_call"][0] for x in (now, then) if x[p]["closest_call"][0] is not None]
                (ties if calls and min(calls) < TIE else changes).append(moved)
    if new["metrics"] != old["metrics"]:
        (ties if ties and not changes else changes).append("metrics changed")
    if new["decisions"] != old["decisions"]:
        (ties if ties and not changes else changes).append("decisions changed")
    return changes, ties
