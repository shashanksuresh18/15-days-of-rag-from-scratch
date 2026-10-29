"""Build the Day 10 LinkedIn carousel (whiteboard style) as editable SVGs in assets/day-10/.

    python scripts/slides/day10.py                    # writes the SVGs
    python scripts/slides/render.py assets/day-10     # renders PNGs + PDF

Every number and ID on the slides comes from the executed Day 10 notebook or the files it writes.
`check_facts()` confirms each quoted line appears in the notebook's saved outputs before anything is
written. The rank moves on slide 1 are counted from the committed baselines (Day 09's and Day 10's) and the
dots on slide 4 from the committed v2 baseline's names-rule outcomes, each checked against the notebook's
printed lines.
"""

from __future__ import annotations

import json
from pathlib import Path

from kit import COLORS, INK, M, TEXT, W, Slide, check_contrast

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "day-10"
NOTEBOOK = ROOT / "notebooks" / "day_10_query_understanding_and_transformations.ipynb"
BASELINES = ROOT / "evals" / "baselines"
TOTAL = 4
DAY = "10"
PAPER_WHITE = "#FFFFFF"

FACTS = [
    "BM25, dense, RRF, weighted, weighted+rerank, RRF+rerank against evals/baselines/day-09-retrieval.json: no changes",
    "  BM25's words for it:  much time off new dads get\n  shared with parental-leave-policy#leave-entitlement: none",
    "weighted+map  d01-01   dev       3 -> 1   caregivers leave",
    "weighted+map  d07-a08  test      3 -> 2   alcohol reimbursed",
    "v2's 41 new test questions: 1 expanded, 40 without a match",
    "weighted+map vs weighted, v2 test questions (29), labelled chunk first: both 24 | only weighted+map 0 | only weighted 0",
    "weighted+map vs weighted, v2 test questions (29), labelled chunk in the top 3: both 27 | only weighted+map 0 | "
    "only weighted 0",
    "d07-a11  My laptop might have a virus. Should I switch it off?    sentences 2 | values - | names -",
    "d07-a01  What is the hotel limit per night in Tokyo?              sentences 1 | values - | names ['Tokyo (city)']",
    "d07-a02  Do I need a receipt for a USD 40 dinner?                 sentences 1 | values ['USD 40']",
    "(c) lookups left unexpanded by the routing: 30 of 30",
    "(e) restricted chunks in any top 10 (8 retrievers on v1's queries, 5 on the new ones): 0",
    "    v2 test questions          touches 11 of 29  gains d10-a05@1                losses d10-a28@1",
    "word by word d10-a28  Can I give my one-time code to someone from IT?            adds: leave remotely remote turn off"
    "    rank 1 -> 2",
    "+ meals      d07-a02  Do I need a receipt for a USD 40 dinner?                   adds: meals"
    "                             rank 1 -> 2",
    "guard: discarded: drops USD 40",
    "guard: discarded: drops the name Tokyo; adds the name Japan",
    "commits to one reading: dads can be primary caregivers     guard: kept",
    "Day 06 names rule                  16 of 16 / 8 of 8          13 of 15 / 6 of 12          22 of 29 / 6 of 12",
    "new names rule                     16 of 16 / 8 of 8          15 of 15 / 6 of 12          26 of 29 / 5 of 12",
    "unknown names: Day 06 ['Does'] -> new []",
    "Day 06 names rule       v2 test         19 / 3 / 7  (of 29)                     6 / 6  (of 12)",
    "new names rule          v2 test         21 / 5 / 3  (of 29)                     5 / 7  (of 12)",
    "d07-a01  What is the hotel limit per night in Tokyo?                             refused -> correct",
    "d07-a11  My laptop might have a virus. Should I switch it off?                   refused -> correct",
    "d10-a01  What's the hotel budget per night in Berlin?                            refused -> correct",
    "d10-a02  How much can I spend on food per day in Paris?                          refused -> wrong\n"
    "           City: London | Hotel per night: USD 250 | Meals per day: USD 75 [travel-policy#daily-limits-by-city].",
    "d10-a23  I'm travelling abroad next month. Do I need to register the trip?       refused -> correct",
    "d10-a24  My phone was stolen on a business trip. Who should I tell?              refused -> wrong\n"
    "           Report a lost or stolen laptop or phone straight away",
    "d10-u12  I'm moving to a new team. Does my remote work arrangement carry over?   refused -> answered",
    "D1, the map in the default retriever: on the 29 answerable v2 test questions it gains nothing and loses nothing",
    "-> not adopted: the weighted fusion stays the default",
    "D2, the new names rule: on the 41 v2 test questions it keeps 26 of 29 answerable (Day 06's rule: 22) and refuses 5 "
    "of 12 unanswerable (Day 06's rule: 6) -> recorded as measured, not adopted",
    "compared with evals/baselines/day-10-retrieval.json: no changes",
    "compared with evals/baselines/day-10-retrieval-v2.json: no changes",
    "All Day 10 sanity checks passed.",
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


def v1_moves() -> tuple[int, int]:
    # Slide 1: how many v1 ranks the map moved in the default retriever (the weighted fusion) against Day 09's accepted
    # file, up and down
    day09 = json.loads((BASELINES / "day-09-retrieval.json").read_text(encoding="utf-8"))["queries"]
    day10 = json.loads((BASELINES / "day-10-retrieval.json").read_text(encoding="utf-8"))["queries"]
    up = down = 0
    for qid, per in day10.items():
        for name, base in (("weighted+map", "weighted"),):
            before, after = day09[qid][base]["first_relevant"], per[name]["first_relevant"]
            if before != after:
                better = after is not None and (before is None or after < before)
                up, down = up + better, down + (not better)
    if (up, down) != (2, 0):
        raise SystemExit(f"slide 1 says 2 ranks up and none down; the baselines give {up} and {down}")
    return up, down


def names_outcomes() -> dict[str, tuple[int, int, int, int]]:
    # Slide 4: the names rules on the v2 test split, from the committed v2 baseline (checked against the notebook)
    outcomes = json.loads((BASELINES / "day-10-retrieval-v2.json").read_text(encoding="utf-8"))["thresholds"][
        "names_rule"]["outcomes"]
    result = {rule: tuple(outcomes[f"{rule}, v2 test"].values()) for rule in ("Day 06 names rule", "new names rule")}
    if result != {"Day 06 names rule": (22, 29, 6, 12), "new names rule": (26, 29, 5, 12)}:
        raise SystemExit(f"slide 4's dots don't match the baseline: {result}")
    return result


def heading(s: Slide, title: str, subtitle: str, size: int = 54) -> None:
    s.text(M, 190, title, size, 800, INK)
    s.text(M, 238, subtitle, 25, 500, TEXT["muted"])


def tcolor(color: str) -> str:
    return TEXT.get(color, INK) if color != "yellow" else INK


def zone_title(s: Slide, y: float, icon: str, color: str, title: str, right: str | None = None) -> None:
    s.icon(icon, M + 38, y + 40, 28, COLORS[color][0] if color != "gray" else TEXT["gray"], sw=2.4)
    s.text(M + 64, y + 49, title, 22, 800, tcolor(color), max_width=W - 2 * M - 90 - (280 if right else 0))
    if right:
        s.text(W - M - 28, y + 49, right, 19, 700, TEXT["muted"], anchor="end", max_width=280)


def mark(s: Slide, x: float, y: float, good: bool, size: int = 20) -> None:
    s.icon("check" if good else "x", x, y, size, COLORS["green" if good else "red"][0], sw=3.2)


# ----------------------------------------------------------------------------- slide 1
def slide_cover(moves: tuple[int, int]) -> Slide:
    s = Slide(1, TOTAL, DAY, "Change the question, measured", seed=101)
    heading(s, "Change the question, measured", "Day 10: read the question, then expand it with a reviewed map.")

    # where the words don't meet
    zy = 274
    s.box(M, zy, W - 2 * M, 380, "blue", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "search", "blue", "WHERE THE WORDS DON'T MEET", right="BM25, Day 01")
    s.text(M + 30, zy + 92, "“How much time off do new dads get?”", 22, 800, INK, max_width=W - 2 * M - 60)
    col = (W - 2 * M - 84) / 2
    lx, rx = M + 30, M + 54 + col
    s.text(lx, zy + 136, "BM25 searches for", 18, 700, TEXT["muted"], max_width=col)
    s.box(lx, zy + 150, col, 54, "gray", PAPER_WHITE, r=12, sw=2, jitter=0.5)
    s.text(lx + 16, zy + 185, "much time off new dads get", 18, 500, INK, "mono", max_width=col - 30)
    s.text(rx, zy + 136, "the answer says", 18, 700, TEXT["muted"], max_width=col)
    s.box(rx, zy + 150, col, 54, "gray", PAPER_WHITE, r=12, sw=2, jitter=0.5)
    s.text(rx + 16, zy + 185, "primary caregivers … leave", 18, 500, INK, "mono", max_width=col - 30)
    mark(s, W / 2, zy + 177, False, 22)
    s.text(W / 2, zy + 236, "words in common: none", 19, 800, TEXT["red"], anchor="middle", max_width=300)
    # the map's fix
    s.icon("book-a", M + 52, zy + 300, 30, COLORS["blue"][0], sw=2.4)
    s.text(M + 82, zy + 296, "the map adds the documents' words:", 19, 700, INK, max_width=420)
    s.chip(M + 82, zy + 312, "+ caregivers leave", "blue", size=19)
    s.text(W - M - 30, zy + 296, "weighted fusion", 18, 700, TEXT["muted"], anchor="end", max_width=200)
    w = s.chip(W - M - 60, zy + 312, "#1", "green", size=21, anchor="end")
    s.arrow(W - M - 60 - w - 52, zy + 332, W - M - 60 - w - 12, zy + 332, INK, sw=2.4, head=10)
    s.chip(W - M - 60 - w - 60, zy + 312, "#3", "gray", size=21, anchor="end")
    mark(s, W - M - 40, zy + 332, True)

    # two test sets
    zy = 696
    half = (W - 2 * M - 24) / 2
    cards = [(M, "green", "the questions behind it", "v1: 81 queries, 6 touched", f"{moves[0]} ranks up",
              f"{moves[1]} down", "vs Day 09's accepted baseline"),
             (M + half + 24, "orange", "41 new questions", "written and committed first", "0 moved", "1 touched",
              "vs the same fusion without the map")]
    for x, color, title, sub, big, small, note in cards:
        s.box(x, zy, half, 288, color, "wash", r=22, sw=2.8)
        s.text(x + 26, zy + 52, title.upper(), 21, 800, tcolor(color), max_width=half - 52)
        s.text(x + 26, zy + 84, sub, 18, 700, TEXT["muted"], max_width=half - 52)
        s.text(x + 26, zy + 170, big, 48, 800, INK, max_width=half - 52)
        s.text(x + 26, zy + 226, small, 30, 800, TEXT["muted"], max_width=half - 52)
        s.text(x + 26, zy + 264, note, 17, 700, TEXT["muted"], max_width=half - 52)
    s.text(W / 2, zy - 12, "the default retriever, the weighted fusion", 17, 700, TEXT["muted"], anchor="middle",
           max_width=500)

    # the decision
    zy = 1002
    s.box(M, zy, W - 2 * M, 236, "yellow", "fill", r=22, sw=3)
    s.icon("scale", M + 40, zy + 44, 30, TEXT["orange"], sw=2.4)
    s.text(M + 70, zy + 54, "DECISIONS, CRITERIA SET IN ADVANCE", 23, 800, INK, max_width=W - 2 * M - 100)
    rows = [("the map in the default retriever", "not adopted"), ("the fixed names rule", "not adopted")]
    for i, (label, value) in enumerate(rows):
        y = zy + 104 + i * 36
        s.text(M + 30, y, label, 21, 700, INK, max_width=560)
        s.text(W - M - 30, y, value, 21, 800, TEXT["red"], anchor="end", max_width=300)
    s.text(M + 30, zy + 206, "a map fixes the gaps someone recorded, and only those", 20, 800, TEXT["teal"],
           max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 2
def slide_read() -> Slide:
    s = Slide(2, TOTAL, DAY, "Read the question first", seed=102)
    heading(s, "Read the question first", "Query understanding: what each part of a question allows.")
    rows = [
        ("list-ordered", "violet", "SENTENCES", "“…a virus. Should I switch it off?”",
         "“Should” starts a sentence: not a name", "Day 06 refused it", True),
        ("map-pin", "teal", "NAMES AND PLACES", "“…per night in Tokyo?”  Tokyo: a city",
         "a source says “any city not listed”", "Day 06 refused it", True),
        ("hash", "orange", "VALUES", "“…a receipt for a USD 40 dinner?”",
         "USD 40 decides it: any rewrite keeps it", "the guard checks", True),
        ("route", "blue", "ROUTE", "USD 1,000   ·   #it-help",
         "an exact value: never rewritten", "30 of 30 lookups", True),
    ]
    y = 274
    for icon, color, title, example, effect, note, good in rows:
        s.box(M, y, W - 2 * M, 168, color, "wash", r=22, sw=2.6)
        s.node(M + 60, y + 58, icon, color, None, size=72, icon_size=36)
        s.text(M + 114, y + 54, title, 22, 800, tcolor(color), max_width=420)
        s.text(W - M - 28, y + 54, note, 18, 700, TEXT["muted"], anchor="end", max_width=300)
        s.text(M + 114, y + 94, example, 20, 800, INK, max_width=W - 2 * M - 150)
        mark(s, M + 126, y + 128, good, 18)
        s.text(M + 146, y + 134, effect, 19, 700, INK, max_width=W - 2 * M - 180)
        y += 182
    # where the map goes
    s.box(M, y, W - 2 * M, 1238 - y, "gray", "wash", r=22, sw=2.4, dashed=True)
    s.text(M + 30, y + 50, "THEN, FOR QUESTIONS ONLY: THE MAP", 21, 800, INK, max_width=W - 2 * M - 60)
    s.chip(M + 30, y + 72, "question + caregivers leave", "blue", size=18)
    s.arrow(M + 360, y + 92, M + 420, y + 92, INK, sw=2.4, head=10)
    s.text(M + 430, y + 99, "BM25", 19, 800, INK, max_width=80)
    s.chip(M + 30, y + 124, "question as asked", "gray", size=18)
    s.arrow(M + 260, y + 144, M + 420, y + 144, INK, sw=2.4, head=10)
    s.text(M + 430, y + 151, "dense + the gate", 19, 800, INK, max_width=190)
    s.arrow(M + 640, y + 118, M + 700, y + 118, INK, sw=2.4, head=10)
    s.text(M + 710, y + 112, "weighted", 19, 800, INK, max_width=140)
    s.text(M + 710, y + 136, "fusion", 19, 800, INK, max_width=140)
    s.text(M + 30, y + 206, "append, never replace; chosen on dev, committed first", 18, 700, TEXT["muted"],
           max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 3
def slide_not() -> Slide:
    s = Slide(3, TOTAL, DAY, "When not to rewrite", seed=103)
    heading(s, "When not to rewrite", "Three ways a change to the question goes wrong.")
    # 1. word by word
    y = 274
    s.box(M, y, W - 2 * M, 272, "red", "wash", r=22, sw=2.6)
    s.node(M + 60, y + 58, "whole-word", "red", None, size=72, icon_size=36)
    s.text(M + 114, y + 66, "MATCHED WORD BY WORD, THE MAP DRIFTS", 22, 800, TEXT["red"], max_width=W - 2 * M - 140)
    x = M + 30
    for label in ("for → reimbursed", "it → turn off", "one-time → leave"):
        x += s.chip(x, y + 104, label, "red", size=18) + 14
    s.text(M + 30, y + 186, "“Can I give my one-time code to someone from IT?”", 19, 800, INK,
           max_width=W - 2 * M - 160)
    s.text(M + 30, y + 216, "+ leave remotely remote turn off", 18, 500, TEXT["muted"], "mono", max_width=560)
    w = s.chip(W - M - 60, y + 194, "#2", "red", size=19, anchor="end")
    s.arrow(W - M - 60 - w - 50, y + 213, W - M - 60 - w - 10, y + 213, INK, sw=2.4, head=10)
    s.chip(W - M - 60 - w - 58, y + 194, "#1", "gray", size=19, anchor="end")
    mark(s, W - M - 40, y + 213, False)
    s.text(M + 30, y + 252, "11 of 29 new questions touched instead of 1", 18, 700, TEXT["muted"],
           max_width=W - 2 * M - 60)
    # 2. true and still wrong
    y = 568
    s.box(M, y, W - 2 * M, 218, "orange", "wash", r=22, sw=2.6)
    s.node(M + 60, y + 58, "book-a", "orange", None, size=72, icon_size=36)
    s.text(M + 114, y + 66, "TRUE IN GENERAL, WRONG FOR THE QUESTION", 22, 800, TEXT["orange"], max_width=W - 2 * M - 140)
    s.chip(M + 30, y + 100, "dinner → meals", "orange", size=18)
    s.text(M + 30, y + 176, "“Do I need a receipt for a USD 40 dinner?”", 19, 800, INK, max_width=W - 2 * M - 160)
    s.text(M + 30, y + 204, "needs “every expense over USD 25”, not the meal limit", 18, 700, TEXT["muted"],
           max_width=W - 2 * M - 160)
    w = s.chip(W - M - 60, y + 160, "#2", "red", size=19, anchor="end")
    s.arrow(W - M - 60 - w - 50, y + 179, W - M - 60 - w - 10, y + 179, INK, sw=2.4, head=10)
    s.chip(W - M - 60 - w - 58, y + 160, "#1", "gray", size=19, anchor="end")
    mark(s, W - M - 40, y + 179, False)
    # 3. the guard
    y = 808
    s.box(M, y, W - 2 * M, 280, "violet", "wash", r=22, sw=2.6)
    s.node(M + 60, y + 58, "shield-check", "violet", None, size=72, icon_size=36)
    s.text(M + 114, y + 66, "A GUARD CHECKS FORM, NOT MEANING", 22, 800, TEXT["violet"], max_width=W - 2 * M - 140)
    s.text(W - M - 28, y + 66, "rewrites written by hand", 17, 700, TEXT["muted"], anchor="end", max_width=260)
    guard = [("drops “USD 40”", "discarded", True), ("Tokyo → Japan", "discarded", True),
             ("new dads → secondary caregivers", "kept", False)]
    for i, (label, verdict, good) in enumerate(guard):
        ry = y + 130 + i * 44
        s.text(M + 36, ry, label, 20, 700, INK, max_width=520)
        s.text(W - M - 70, ry, verdict, 20, 800, TEXT["green"] if good else TEXT["red"], anchor="end", max_width=200)
        mark(s, W - M - 44, ry - 7, good, 18)
    s.text(M + 36, y + 262, "a dad can be the primary caregiver: 16 weeks, not 6", 18, 700, TEXT["muted"],
           max_width=W - 2 * M - 70)
    # never
    y = 1110
    s.box(M, y, W - 2 * M, 128, "gray", "wash", r=22, sw=2.4, dashed=True)
    s.icon("lock-keyhole", M + 40, y + 44, 26, TEXT["gray"], sw=2.4)
    s.text(M + 66, y + 52, "NEVER REWRITE", 21, 800, INK, max_width=300)
    s.text(M + 30, y + 92, "exact values · who is asking: permissions come from the login", 19, 700, INK,
           max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 4
def dots(s: Slide, x: float, y: float, total: int, filled: int, color: str, step: float = 25) -> None:
    for i in range(total):
        s.ellipse(x + i * step, y, 9, 9, color, "fill" if i < filled else PAPER_WHITE, sw=2.2, jitter=0.4)


def slide_follow(outcomes: dict[str, tuple[int, int, int, int]]) -> Slide:
    s = Slide(4, TOTAL, DAY, "Fix a rule, follow its questions", seed=104)
    heading(s, "Fix a rule, follow its questions", "Day 06's names rule, fixed, on 41 new questions.")
    zy = 274
    s.box(M, zy, W - 2 * M, 360, "teal", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "map-pin", "teal", "SENTENCE STARTS SKIPPED, PLACES TYPED", right="dense gate 0.62 first")
    for i, (rule, label) in enumerate((("Day 06 names rule", "Day 06's rule"), ("new names rule", "fixed rule"))):
        kept, answerable, refused, unanswerable = outcomes[rule]
        ry = zy + 112 + i * 110
        s.text(M + 30, ry, label, 20, 800, INK, max_width=200)
        s.text(M + 230, ry, f"answerable kept: {kept} of {answerable}", 18, 700, TEXT["blue"], max_width=320)
        dots(s, M + 240, ry + 30, answerable, kept, "blue", step=24)
        s.text(W - M - 30, ry + 72, f"unanswerable refused: {refused} of {unanswerable}", 18, 700, TEXT["orange"],
               anchor="end", max_width=330)
        dots(s, M + 240, ry + 66, unanswerable, refused, "orange", step=24)
    s.text(M + 30, zy + 330, "+ Berlin, Paris, “Do…?”, “Who…?”     − “Does…?” was refused by accident", 18, 700,
           TEXT["muted"], max_width=W - 2 * M - 60)
    # the lifted refusals
    zy = 658
    s.box(M, zy, W - 2 * M, 330, "red", "wash", r=22, sw=2.8)
    s.icon("triangle-alert", M + 40, zy + 44, 28, COLORS["red"][0], sw=2.4)
    s.text(M + 70, zy + 52, "THE 7 QUESTIONS IT STOPPED REFUSING", 22, 800, TEXT["red"], max_width=W - 2 * M - 100)
    lifted = [("Tokyo, the virus question, Berlin, register the trip", "4 correct", True),
              ("Paris: London's row", "wrong", False),
              ("stolen phone: “report it straight away”", "wrong", False),
              ("changing teams: a cited non-answer", "answered", False)]
    for i, (label, verdict, good) in enumerate(lifted):
        ry = zy + 112 + i * 52
        mark(s, M + 46, ry - 7, good, 20)
        s.text(M + 72, ry, label, 20, 700, INK, max_width=W - 2 * M - 300)
        s.text(W - M - 30, ry, verdict, 20, 800, TEXT["green"] if good else TEXT["red"], anchor="end", max_width=200)
    s.text(M + 30, zy + 314, "a lifted refusal hands the question to the next stage", 19, 800, TEXT["teal"],
           max_width=W - 2 * M - 60)
    # the decision
    zy = 1012
    s.box(M, zy, W - 2 * M, 226, "yellow", "fill", r=22, sw=3)
    s.icon("scale", M + 40, zy + 44, 30, TEXT["orange"], sw=2.4)
    s.text(M + 70, zy + 54, "THE BAR, SET IN ADVANCE: LOSE NOTHING", 23, 800, INK, max_width=W - 2 * M - 100)
    lines = [("4 more answerable kept, 1 fewer refusal: not adopted", 700, INK),
             ("the new test split decided twice: now it's used too", 700, INK),
             ("next: Day 11, the section next door", 800, TEXT["orange"])]
    for i, (line, weight, color) in enumerate(lines):
        s.text(M + 30, zy + 106 + i * 38, line, 21, weight, color, max_width=W - 2 * M - 60)
    return s


def main() -> None:
    check_contrast()
    check_facts()
    moves = v1_moves()
    outcomes = names_outcomes()
    OUT.mkdir(parents=True, exist_ok=True)
    slides = [("slide-01-cover", slide_cover(moves)), ("slide-02-read-the-question", slide_read()),
              ("slide-03-when-not-to-rewrite", slide_not()), ("slide-04-follow-the-questions", slide_follow(outcomes))]
    for name, slide in slides:
        path = OUT / f"{name}.svg"
        path.write_text(slide.svg(), encoding="utf-8")
        print("wrote", path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
