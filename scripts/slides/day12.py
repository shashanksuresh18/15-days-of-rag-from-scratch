"""Build the Day 12 LinkedIn carousel (whiteboard style) as editable SVGs in assets/day-12/.

    python scripts/slides/day12.py                    # writes the SVGs
    python scripts/slides/render.py assets/day-12     # renders PNGs + PDF

Every number and ID on the slides comes from the executed Day 12 notebook or the answer baseline it is compared with.
`check_facts()` confirms each quoted line appears in the notebook's saved outputs before anything is written, and
`from_baseline()` recounts slide 1's bars, slide 3's dots and the regression counts from the committed files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from kit import COLORS, INK, M, TEXT, W, Slide, check_contrast

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "day-12"
NOTEBOOK = ROOT / "notebooks" / "day_12_answer_evaluation_and_regression_tests.ipynb"
BASELINE = ROOT / "evals" / "baselines" / "day-12-answers-v4.json"
REGRESSIONS = ROOT / "evals" / "regressions.jsonl"
TOTAL = 4
DAY = "12"
PAPER_WHITE = "#FFFFFF"

FACTS = [
    "BM25, dense, weighted, BM25+map, weighted+map, weighted+neighbours against evals/baselines/day-11-retrieval-v3.json: "
    "no changes",
    "weighted                       49     52     52     52    0.97   100% (93% to 100%)",
    "v4 test, the current pipeline (52 answerable, 24 not):\n  correct                   30 of 52    58% (44% to 70%)",
    "  answerable, not correct: names rule: a sentence's first word 12, names rule: a name 5, reader: answer in the "
    "context, not used 2, citation check 2, reader: right source, wrong part 1\n  a labelled chunk was in the context "
    "for 52 of 52 answerable questions",
    "the extractive reader (today's pipeline)    wrong     passed       1.0      True        False  reader: answer in the "
    "context, not used",
    "a faithful paraphrase (hand-written)        correct   passed     0.833      True         True  ok",
    "gpt-6-luna on Day 06 (recorded)             wrong     passed       1.0      True         True  reader: right source, "
    "wrong part",
    "a changed number (hand-written)             refused   blocked        -         -            -  citation check",
    "Dev, both pipelines: 122 distinct answered (question, answer) pairs, 122 reviewed; the rubric agrees with the "
    "reading on 120",
    "58 distinct answered (question, answer) pairs, 58 reviewed; the rubric agrees with the reading on 57",
    "alice: It's my first day. Where do I pick up my laptop?",
    "  why: no source mentions Where",
    "v4 test (held-out): Day 10's names rule against the current pipeline, 76 questions | outcome changed for 15 | near "
    "ties set apart: d12-a01 | better 10, worse 4, sign test p = 0.18 | change in value: cost 1: +6, cost 2: +2, cost 3: -2",
    "  counting the near ties too: better 10, worse 5, change in value: cost 1: +5, cost 2: +0, cost 3: -5",
    "the value at cost 2 changes by +2 -> adopted: later days use Day 10's names rule",
    "the outcome with and without the borderline sentence: wrong and wrong",
    "29 regression cases in evals/regressions.jsonl: 13 guard, 9 fixed, 7 known",
    "compared with evals/baselines/day-12-answers-v4.json: no changes",
    "All Day 12 sanity checks passed.",
]


def notebook_outputs() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code"
                   for o in c.get("outputs", []))


def check_facts() -> None:
    outputs = notebook_outputs()
    missing = [f for f in FACTS if f not in outputs]
    if missing:
        raise SystemExit(f"Slide facts not found (re-run the notebook?): {missing}")


def from_baseline() -> dict:
    # Slide 1's bars, slide 3's dots and example, and slide 4's counts, recounted from the committed files
    b = json.loads(BASELINE.read_text(encoding="utf-8"))
    test = {qid: e for qid, e in b["questions"].items() if e["split"] == "test"}
    answerable = {qid: e for qid, e in test.items() if e["retrieval"]["first_relevant"] is not None}
    in_top3 = sum(e["retrieval"]["first_relevant"] <= 3 for e in answerable.values())
    stages = Counter(e["current"]["stage"] for e in answerable.values())
    order = {"correct": 2, "refused": 1, "wrong": 0, "answered": 0}
    changed = {qid: (e["current"]["outcome"], e["names-v2"]["outcome"]) for qid, e in test.items()
               if e["current"]["outcome"] != e["names-v2"]["outcome"]}
    near = b["decisions"]["v4 test"]["near"]
    kinds = Counter("near tie" if qid in near else f"{a} -> {c}" for qid, (a, c) in changed.items())
    example = b["questions"]["d11-a17"]
    cases = Counter(json.loads(line)["status"] for line in REGRESSIONS.read_text(encoding="utf-8").splitlines() if line)
    facts = {"answerable": len(answerable), "in_top3": in_top3, "correct": stages["ok"],
             "first_word": stages["names rule: a sentence's first word"], "a_name": stages["names rule: a name"],
             "reader": stages["reader: answer in the context, not used"] + stages["reader: right source, wrong part"],
             "check": stages["citation check"], "kinds": kinds, "delta": b["decisions"]["v4 test"]["delta"],
             "decision": b["decisions"]["D1 Day 10's names rule in the current pipeline"],
             "example": (example["current"]["reason"], example["names-v2"]["outcome"], example["names-v2"]["text"]),
             "cases": cases}
    expected = {"answerable": 52, "in_top3": 52, "correct": 30, "first_word": 12, "a_name": 5, "reader": 3, "check": 2,
                "kinds": {"refused -> correct": 10, "refused -> wrong": 1, "refused -> answered": 3, "near tie": 1},
                "delta": {"1": 6, "2": 2, "3": -2}, "decision": "adopted", "cases": {"guard": 13, "fixed": 9, "known": 7}}
    if {k: facts[k] for k in expected} != expected or facts["example"][:2] != ("no source mentions Where", "correct") \
            or not facts["example"][2].startswith("Collect your laptop and security badge from reception"):
        raise SystemExit(f"the slides' numbers don't match the baseline: {facts}")
    return facts


def heading(s: Slide, title: str, subtitle: str, size: int = 54) -> None:
    s.text(M, 190, title, size, 800, INK)
    s.text(M, 238, subtitle, 25, 500, TEXT["muted"])


def tcolor(color: str) -> str:
    return TEXT.get(color, INK) if color != "yellow" else INK


def zone_title(s: Slide, y: float, icon: str, color: str, title: str, right: str | None = None) -> None:
    s.icon(icon, M + 38, y + 40, 28, COLORS[color][0] if color != "gray" else TEXT["gray"], sw=2.4)
    s.text(M + 64, y + 49, title, 22, 800, tcolor(color), max_width=W - 2 * M - 90 - (300 if right else 0))
    if right:
        s.text(W - M - 28, y + 49, right, 19, 700, TEXT["muted"], anchor="end", max_width=300)


def mark(s: Slide, x: float, y: float, good: bool, size: int = 20) -> None:
    s.icon("check" if good else "x", x, y, size, COLORS["green" if good else "red"][0], sw=3.2)


def bar_row(s: Slide, y: float, label: str, value: int, total: int, color: str, x0: float, full: float,
            label_width: float, note: str | None = None) -> None:
    s.text(M + 30, y + 26, label, 20, 800, INK, max_width=label_width)
    s.box(x0, y, full, 36, "gray", PAPER_WHITE, r=10, sw=1.8, jitter=0.4)
    if value:
        s.box(x0, y, max(full * value / total, 24), 36, color, "fill", r=10, sw=2.2, jitter=0.6)
    s.text(x0 + max(full * value / total, 24) + 12, y + 26, f"{value}", 20, 800, INK, max_width=60)
    if note:
        s.text(W - M - 30, y + 26, note, 17, 700, TEXT["muted"], anchor="end", max_width=200)


# ----------------------------------------------------------------------------- slide 1
def slide_cover(f: dict) -> Slide:
    s = Slide(1, TOTAL, DAY, "Found isn't answered", seed=121)
    heading(s, "Found isn't answered", "Day 12: grade the answers, and say where each one was lost.")

    zy = 274
    s.box(M, zy, W - 2 * M, 300, "blue", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "search", "blue", f"{f['answerable']} NEW QUESTIONS, THE CURRENT PIPELINE",
               right="written first")
    full = W - 2 * M - 400
    bar_row(s, zy + 100, "answer in the top 3", f["in_top3"], f["answerable"], "blue", M + 330, full, 290)
    bar_row(s, zy + 170, "answered correctly", f["correct"], f["answerable"], "green", M + 330, full, 290)
    s.text(M + 30, zy + 270, "retrieval found every answer; users got 30 right ones", 19, 800, TEXT["blue"],
           max_width=W - 2 * M - 60)

    zy = 596
    lost = f["answerable"] - f["correct"]
    s.box(M, zy, W - 2 * M, 420, "orange", "wash", r=22, sw=2.8)
    zone_title(s, zy, "funnel", "orange", f"WHERE THE OTHER {lost} WERE LOST", right="the stage")
    rows = [("names rule: a 2nd sentence's", "first word", f["first_word"], "red", "“Where…” read as a name"),
            ("names rule: a name", "", f["a_name"], "red", "Chrome, Slack, Madrid…"),
            ("the reader", "", f["reader"], "violet", "copied another sentence"),
            ("the citation check", "", f["check"], "gray", "a copied “1.” split off")]
    full = W - 2 * M - 360 - 290
    for i, (label, label2, value, color, note) in enumerate(rows):
        ry = zy + 96 + i * 78
        s.text(M + 30, ry + 20, label, 19, 800, INK, max_width=330)
        if label2:
            s.text(M + 30, ry + 46, label2, 19, 800, INK, max_width=330)
        s.box(M + 360, ry, full, 34, "gray", PAPER_WHITE, r=10, sw=1.8, jitter=0.4)
        s.box(M + 360, ry, full * value / lost, 34, color, "fill", r=10, sw=2.2, jitter=0.6)
        s.text(M + 360 + full * value / lost + 12, ry + 25, f"{value}", 20, 800, INK, max_width=50)
        s.text(W - M - 30, ry + 25, note, 16, 700, TEXT["muted"], anchor="end", max_width=240)
    s.text(M + 30, zy + 404, "retrieval lost none of them", 19, 800, TEXT["orange"], max_width=W - 2 * M - 60)

    zy = 1038
    s.box(M, zy, W - 2 * M, 200, "yellow", "fill", r=22, sw=3)
    s.icon("clipboard-check", M + 40, zy + 44, 30, TEXT["orange"], sw=2.4)
    s.text(M + 70, zy + 54, "A GRADE THAT NAMES THE STAGE SAYS WHAT TO FIX", 22, 800, INK, max_width=W - 2 * M - 100)
    s.text(M + 30, zy + 104, "here: a refusal rule first, then the reader", 20, 700, INK, max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 148, "a retrieval metric stops at the context; people read answers", 20, 800, TEXT["teal"],
           max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 2
def slide_rubric() -> Slide:
    s = Slide(2, TOTAL, DAY, "Grounded isn't correct", seed=122)
    heading(s, "Grounded isn't correct", "One grade, six parts, written down before the test.")

    zy = 274
    s.box(M, zy, W - 2 * M, 290, "violet", "wash", r=22, dashed=True, sw=2.8)
    zone_title(s, zy, "clipboard-check", "violet", "THE RUBRIC")
    parts = [("check", "correct", "every fact in the answer"), ("ban", "refused by", "gate, names, reader, check"),
             ("search", "context", "was the answer sent?"), ("file-check", "cited", "does it cite the answer?"),
             ("book-open", "grounded", "words in the cited source"), ("funnel", "stage", "where it was lost")]
    half = (W - 2 * M - 60) / 2
    for i, (icon, title, note) in enumerate(parts):
        x = M + 30 + (i % 2) * (half + 0)
        ry = zy + 104 + (i // 2) * 60
        s.icon(icon, x + 14, ry - 7, 24, COLORS["violet"][0], sw=2.3)
        s.text(x + 40, ry, title, 20, 800, INK, max_width=150)
        s.text(x + 170, ry, note, 17, 700, TEXT["muted"], max_width=half - 180)
    s.text(M + 30, zy + 272, "facts, labels and stages come from the labelled questions", 17, 700, TEXT["muted"],
           max_width=W - 2 * M - 60)

    zy = 586
    s.box(M, zy, W - 2 * M, 420, "blue", "wash", r=22, sw=2.8)
    zone_title(s, zy, "message-square", "blue", "“HOW MUCH TIME OFF DO NEW DADS GET?”", right="four replies")
    cols = [(W - M - 330, "grounded"), (W - M - 200, "cites it"), (W - M - 80, "correct")]
    for x, label in cols:
        s.text(x, zy + 104, label, 17, 800, TEXT["blue"], anchor="middle", max_width=120)
    rows = [("the reader", "copies a phased return", (True, False, False)),
            ("a faithful paraphrase", "“16 weeks … 6 weeks”", (True, True, True)),
            ("gpt-6-luna (Day 06)", "“Secondary caregivers: 6 weeks”", (True, True, False)),
            ("a changed number", "“18 weeks”: blocked", None)]
    for i, (who, what, marks) in enumerate(rows):
        ry = zy + 148 + i * 60
        s.text(M + 30, ry, who, 20, 800, INK, max_width=W - 2 * M - 420)
        s.text(M + 30, ry + 28, what, 17, 700, TEXT["muted"], max_width=W - 2 * M - 420)
        if marks is None:
            s.text(W - M - 200, ry + 6, "never shown", 18, 800, TEXT["gray"], anchor="middle", max_width=300)
            continue
        for (x, _), good in zip(cols, marks):
            mark(s, x, ry + 2, good, 22)
    s.text(M + 30, zy + 404, "the policy is gender-neutral: a dad who is the main carer gets 16 weeks", 17, 700,
           TEXT["muted"], max_width=W - 2 * M - 60)

    zy = 1028
    s.box(M, zy, W - 2 * M, 210, "yellow", "fill", r=22, sw=3)
    s.icon("scale", M + 40, zy + 44, 30, TEXT["orange"], sw=2.4)
    s.text(M + 70, zy + 54, "A GROUNDEDNESS SCORE PASSES 3 OF 4", 22, 800, INK, max_width=W - 2 * M - 100)
    s.text(M + 30, zy + 102, "two of those are wrong: keep correct and grounded apart", 20, 700, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 144, "grade the grader: 177 of 180 agree with a careful reading", 20, 800, TEXT["teal"],
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 184, "the 3 misses: lists of table rows that include the right one", 17, 700, TEXT["muted"],
           max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 3
def slide_decision(f: dict) -> Slide:
    s = Slide(3, TOTAL, DAY, "Fix one rule, meet the next", seed=123)
    heading(s, "Fix one rule, meet the next", "Day 10's names rule, judged by answers on 76 new questions.")

    zy = 274
    s.box(M, zy, W - 2 * M, 236, "teal", "wash", r=22, sw=2.8)
    zone_title(s, zy, "eye-off", "teal", "A SENTENCE'S FIRST WORD, READ AS A NAME")
    s.text(M + 30, zy + 96, "“It's my first day. Where do I pick up my laptop?”", 20, 800, INK, max_width=W - 2 * M - 60)
    mark(s, M + 46, zy + 140, False, 20)
    s.text(M + 72, zy + 148, "Day 06's rule: no source mentions “Where” → refused", 19, 700, INK,
           max_width=W - 2 * M - 110)
    mark(s, M + 46, zy + 190, True, 20)
    s.text(M + 72, zy + 198, "Day 10's rule: “Collect your laptop and security badge…”", 19, 700, INK,
           max_width=W - 2 * M - 110)

    zy = 532
    k = f["kinds"]
    s.box(M, zy, W - 2 * M, 300, "violet", "wash", r=22, dashed=True, sw=2.8)
    zone_title(s, zy, "scale", "violet", f"{sum(k.values())} OUTCOMES CHANGED", right="of 76 new questions")
    groups = [("refused → correct", k["refused -> correct"], "green", True),
              ("refused → wrong", k["refused -> wrong"], "red", False),
              ("not answerable → answered", k["refused -> answered"], "red", False),
              ("near tie, set apart", k["near tie"], "gray", None)]
    for i, (label, n, color, good) in enumerate(groups):
        ry = zy + 98 + i * 50
        for j in range(n):
            s.ellipse(M + 46 + j * 34, ry - 7, 13, 13, color, "fill", sw=2.2, jitter=0.6)
        s.text(M + 420, ry, f"{n}  {label}", 20, 800, tcolor(color) if color != "gray" else TEXT["muted"],
               max_width=W - 2 * M - 450)
    s.text(M + 30, zy + 286, "the new wrong answers are the reader's: the old rule hid them", 18, 700, TEXT["muted"],
           max_width=W - 2 * M - 60)

    zy = 854
    d = f["delta"]
    s.box(M, zy, W - 2 * M, 384, "yellow", "fill", r=22, sw=3)
    s.icon("calculator", M + 40, zy + 44, 30, TEXT["orange"], sw=2.4)
    s.text(M + 70, zy + 54, "VALUE, SET IN ADVANCE", 22, 800, INK, max_width=W - 2 * M - 100)
    s.chip(M + 30, zy + 80, "correct +1 · refused 0 · wrong −2", "orange", size=19)
    s.text(M + 30, zy + 168, f"10 × (+1) + 4 × (−2) = {d['2']:+d}: {f['decision']}", 24, 800, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 210, "by the smallest margin: counting the near tie, +0", 19, 700, TEXT["red"],
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 252, f"if wrong costs 1: {d['1']:+d}  ·  if it costs 3: {d['3']:+d}", 19, 700, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 304, "a decision needs a cost for wrong answers, stated first", 20, 800, TEXT["teal"],
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 346, "next to fix: the reader", 19, 800, TEXT["violet"], max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 4
def slide_regressions(f: dict) -> Slide:
    s = Slide(4, TOTAL, DAY, "Every recorded failure, a test", seed=124)
    heading(s, "Every recorded failure, a test", "29 cases from Days 01–11, run on every push.")
    c = f["cases"]
    cards = [("circle-check", "green", "FIXED", c["fixed"], "must stay fixed", "new dads: in the top 3"),
             ("shield-check", "blue", "GUARD", c["guard"], "must always hold", "no managers-only chunk for alice"),
             ("bug", "red", "KNOWN", c["known"], "pinned as it fails", "a copied “1.” is blocked")]
    width = (W - 2 * M - 2 * 20) / 3
    zy = 274
    for i, (icon, color, title, n, rule, example) in enumerate(cards):
        x = M + i * (width + 20)
        s.box(x, zy, width, 290, color, "wash", r=22, sw=2.8)
        s.icon(icon, x + 38, zy + 42, 28, COLORS[color][0], sw=2.4)
        s.text(x + 64, zy + 52, title, 22, 800, tcolor(color), max_width=width - 80)
        s.text(x + 26, zy + 140, f"{n}", 56, 800, INK, max_width=width - 52)
        s.text(x + 26, zy + 186, rule, 18, 800, INK, max_width=width - 52)
        s.paragraph(x + 26, zy + 222, example, 16, 700, width - 52, 22, TEXT["muted"])

    zy = 586
    s.box(M, zy, W - 2 * M, 250, "red", "wash", r=22, dashed=True, sw=2.8)
    zone_title(s, zy, "bug", "red", "HOW A KNOWN CASE WORKS")
    s.text(M + 30, zy + 100, "the failure still happens", 20, 800, INK, max_width=420)
    s.arrow(M + 470, zy + 93, M + 560, zy + 93, INK, sw=2.6, head=12)
    s.text(M + 580, zy + 100, "the test passes", 20, 800, TEXT["green"], max_width=W - 2 * M - 610)
    s.text(M + 30, zy + 154, "it stops, or changes shape", 20, 800, INK, max_width=420)
    s.arrow(M + 470, zy + 147, M + 560, zy + 147, INK, sw=2.6, head=12)
    s.text(M + 580, zy + 154, "the test fails, and says why", 20, 800, TEXT["red"], max_width=W - 2 * M - 610)
    s.text(M + 30, zy + 214, "fixed on purpose? move it to fixed. never delete it", 19, 800, TEXT["red"],
           max_width=W - 2 * M - 60)

    zy = 858
    s.box(M, zy, W - 2 * M, 380, "gray", "wash", r=22, sw=2.6)
    s.icon("list-ordered", M + 40, zy + 44, 28, TEXT["gray"], sw=2.4)
    s.text(M + 70, zy + 52, "BEFORE YOU TRUST AN ANSWER METRIC", 22, 800, INK, max_width=W - 2 * M - 100)
    checklist = [("ban", "grade refusals too", "refusing everything has no wrong answers"),
                 ("funnel", "name the stage", "so the grade says what to fix"),
                 ("users", "check the grader", "against people, per generator"),
                 ("calculator", "state the cost first", "of a wrong answer vs a refusal"),
                 ("bug", "pin failures as tests", "fixed, guard or known")]
    for i, (icon, title, note) in enumerate(checklist):
        ry = zy + 116 + i * 56
        s.icon(icon, M + 50, ry - 7, 24, TEXT["gray"], sw=2.3)
        s.text(M + 84, ry, title, 21, 800, INK, max_width=320)
        s.text(M + 410, ry, note, 18, 700, TEXT["muted"], max_width=W - 2 * M - 440)
    return s


def main() -> None:
    check_contrast()
    check_facts()
    f = from_baseline()
    OUT.mkdir(parents=True, exist_ok=True)
    slides = [("slide-01-cover", slide_cover(f)), ("slide-02-the-rubric", slide_rubric()),
              ("slide-03-the-decision", slide_decision(f)), ("slide-04-regression-tests", slide_regressions(f))]
    for name, slide in slides:
        path = OUT / f"{name}.svg"
        path.write_text(slide.svg(), encoding="utf-8")
        print("wrote", path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
