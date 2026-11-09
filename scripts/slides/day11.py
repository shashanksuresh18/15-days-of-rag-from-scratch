"""Build the Day 11 LinkedIn carousel (whiteboard style) as editable SVGs in assets/day-11/.

    python scripts/slides/day11.py                    # writes the SVGs
    python scripts/slides/render.py assets/day-11     # renders PNGs + PDF

Every number and ID on the slides comes from the executed Day 11 notebook or the baseline it is compared with.
`check_facts()` confirms each quoted line appears in the notebook's saved outputs before anything is written, and
`held_out()` recounts the three bars on slide 1 from the committed Day 11 baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

from kit import COLORS, INK, M, TEXT, W, Slide, check_contrast

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "day-11"
NOTEBOOK = ROOT / "notebooks" / "day_11_advanced_retrieval_section_expansion.ipynb"
BASELINE = ROOT / "evals" / "baselines" / "day-11-retrieval-v3.json"
TOTAL = 4
DAY = "11"
PAPER_WHITE = "#FFFFFF"

FACTS = [
    "BM25, dense, weighted, BM25+map, weighted+map against evals/baselines/day-10-retrieval-v2.json: no changes",
    "d10-a24 My phone was stolen on a business trip. Who should I tell?",
    "    2  0.654  security-incident-reporting#after-you-report             section 4 of 4  sent",
    "    3  0.568  security-incident-reporting#what-not-to-do               section 3 of 4  sent",
    "    4  0.499  security-incident-reporting#what-counts-as-an-incident   section 1 of 4  ",
    "    6  0.452  security-incident-reporting#how-to-report                section 2 of 4  <- the answer",
    "  previous of hit 3   security-incident-reporting#how-to-report                 0.452   33 words  sent   <- labelled",
    "  next of hit 3       security-incident-reporting#after-you-report              0.654   37 words  already sent",
    "  sent 5 sections, 188 words",
    "off-switch: 'off' sends exactly the weighted fusion's top 3 for all 122 queries: True",
    "previous                                    89             182.2",
    "next                                        89             179.1",
    "neighbours                                  90             223.1",
    "parent                                      89             245.6",
    "v3 test questions, top 3                 52          49   94% (84% to 98%)           110",
    "v3 test questions, weighted+neighbours   52          51   98% (90% to 100%)          199",
    "v3 test questions, control               52          52   100% (93% to 100%)         199",
    "weighted+neighbours vs top 3, v3 test questions (52), labelled chunk in the context: both 49 | "
    "only weighted+neighbours 2 | only top 3 0 | neither 1 | sign test p = 0.5",
    "weighted+neighbours vs control, v3 test questions (52), labelled chunk in the context: both 51 | "
    "only weighted+neighbours 0 | only control 1 | neither 0 | sign test p = 1",
    "weighted+neighbours vs control, v2 questions (used) (29), labelled chunk in the context: both 28 | "
    "only weighted+neighbours 1 | only control 0",
    "  top 3                 d11-a06  fused #5   other section   I bought a train ticket for a client visit",
    "  top 3                 d11-a09  fused #5   other document  Can I claim a taxi from the airport to my hotel?",
    "  top 3                 d11-a40  fused #5   other section   I clicked a link in a suspicious email.",
    "  weighted+neighbours   d11-a09  fused #5   other document",
    "security-incident-reporting#after-you-report restricted to managers (in memory): the search returns it to alice "
    "for 0 of her 155 queries; an unchecked lookup of the neighbours hands it to her for 10; the checked expansion for 0",
    "  next of hit 2       security-incident-reporting#after-you-report                      37 words  not readable",
    "answerable (dev + test), words         mean 114, max 165     mean 214, max 300     mean 217, max 346",
    "answerable (dev + test), tokens        mean 137, max 195     mean 259, max 378     mean 262, max 426",
    "  d11-a40  I clicked a link in a suspicious email. What should I do now?\n"
    "           wrong -> wrong: Do not try to investigate the incident yourself",
    "questions whose outcome changes between contexts (3):",
    "why answerable v3 test questions were refused: top 3: names rule 18,",
    "-> not adopted: the top 3 stays; the expansion is recorded as measured",
    "compared with evals/baselines/day-11-retrieval-v3.json: no changes",
    "All Day 11 sanity checks passed.",
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


def held_out() -> tuple[int, int, int, int]:
    # Slide 1's bars, recounted from the committed baseline: answerable v3 test questions with a labelled chunk in the
    # context of the top 3, of the expansion and of the control
    m = json.loads(BASELINE.read_text(encoding="utf-8"))["metrics"]["v3 test questions, weighted+neighbours"]
    counts = (m["n"], m["top 3 in context"], m["in context"], m["control in context"])
    if counts != (52, 49, 51, 52):
        raise SystemExit(f"slide 1's bars don't match the baseline: {counts}")
    return counts


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


def section_strip(s: Slide, y: float, cells, height: int = 112) -> list[tuple[float, float]]:
    # One document's sections, left to right in document order: (label, sub, colour, fill); returns the centres
    gap, x = 14, M + 30
    width = (W - 2 * M - 60 - gap * (len(cells) - 1)) / len(cells)
    centres = []
    for label, sub, color, fill in cells:
        s.box(x, y, width, height, color, fill, r=16, sw=2.6)
        s.text(x + width / 2, y + 44, label, 19, 800, INK, anchor="middle", max_width=width - 16)
        s.text(x + width / 2, y + 80, sub, 17, 700, tcolor(color) if color != "gray" else TEXT["muted"],
               anchor="middle", max_width=width - 16)
        centres.append((x + width / 2, y))
        x += width + gap
    return centres


# ----------------------------------------------------------------------------- slide 1
def slide_cover(counts: tuple[int, int, int, int]) -> Slide:
    s = Slide(1, TOTAL, DAY, "Read around the hit, measured", seed=111)
    heading(s, "Read around the hit, measured", "Day 11: send the sections next to each hit, then test it fairly.")

    zy = 274
    s.box(M, zy, W - 2 * M, 404, "blue", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "rows-3", "blue", "THE ANSWER WAS NEXT DOOR", right="Day 10's held-out miss")
    s.text(M + 30, zy + 94, "“My phone was stolen on a business trip. Who should I tell?”", 21, 800, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 132, "Security Incident Reporting, in document order", 18, 700, TEXT["muted"],
           max_width=W - 2 * M - 60)
    cells = [("What counts", "#4: not sent", "gray", "wash"), ("How to report", "#6: the answer", "green", "fill"),
             ("What not to do", "#3: hit 3", "blue", "fill"), ("After you report", "#2: hit 2", "blue", "fill")]
    centres = section_strip(s, zy + 152, cells)
    (ax, ay), (bx, _) = centres[1], centres[2]
    s.arrow(bx - 20, ay + 140, ax + 20, ay + 140, COLORS["green"][0], sw=3, bend=-0.35, head=14)
    s.text((ax + bx) / 2, ay + 196, "previous of hit 3: added", 19, 800, TEXT["green"], anchor="middle",
           max_width=330)
    s.text(M + 30, zy + 382, "the top 3: answer missing   ·   + neighbours: in the context", 19, 700, INK,
           max_width=W - 2 * M - 60)

    zy = 700
    n, top3, expanded, control = counts
    s.box(M, zy, W - 2 * M, 300, "violet", "wash", r=22, sw=2.8)
    zone_title(s, zy, "scale", "violet", f"{n} NEW QUESTIONS: ANSWER IN THE CONTEXT?", right="written first")
    bars = [("the top 3", top3, "110 words", "gray"), ("top 3 + neighbours", expanded, "199 words", "blue"),
            ("as many, further down", control, "199 words", "green")]
    full = W - 2 * M - 330 - 150
    for i, (label, value, words, color) in enumerate(bars):
        by = zy + 96 + i * 62
        s.text(M + 30, by + 26, label, 20, 800, INK, max_width=290)
        s.box(M + 330, by, full, 36, "gray", PAPER_WHITE, r=10, sw=1.8, jitter=0.4)
        s.box(M + 330, by, full * value / n, 36, color, "fill", r=10, sw=2.2, jitter=0.6)
        s.text(M + 330 + full * value / n - 12, by + 26, f"{value}", 20, 800, INK, anchor="end", max_width=60)
        s.text(W - M - 30, by + 26, words, 18, 700, TEXT["muted"], anchor="end", max_width=130)
    s.text(M + 30, zy + 282, "as many, further down: as many sections as the expansion sent, from the ranking", 17, 700,
           TEXT["muted"], max_width=W - 2 * M - 60)

    zy = 1022
    s.box(M, zy, W - 2 * M, 216, "yellow", "fill", r=22, sw=3)
    s.icon("scale", M + 40, zy + 44, 30, TEXT["orange"], sw=2.4)
    s.text(M + 70, zy + 54, "THE BAR, SET IN ADVANCE: BEAT THE CONTROL", 23, 800, INK, max_width=W - 2 * M - 100)
    s.text(M + 30, zy + 106, "it found 2 the top 3 missed; the control found those and 1 more", 20, 700, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 144, "section expansion: not adopted", 21, 800, TEXT["red"], max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 188, "more context always finds more: compare with the same amount", 20, 800, TEXT["teal"],
           max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 2
def slide_how() -> Slide:
    s = Slide(2, TOTAL, DAY, "Hits first, then their neighbours", seed=112)
    heading(s, "Hits first, then their neighbours", "Section expansion, with an off-switch and a trace.")
    steps = [("search", "blue", "SEARCH, FILTERED", "the weighted fusion's top 3: the hits"),
             ("rows-3", "teal", "LOOK UP AROUND EACH HIT", "the section before and after, same document"),
             ("lock-keyhole", "red", "MAY THIS USER READ IT?", "a lookup isn't a search: check again"),
             ("ruler", "orange", "FITS 300 WORDS?", "whole sections; skip what doesn't fit")]
    y = 274
    for i, (icon, color, title, note) in enumerate(steps):
        s.box(M, y, W - 2 * M, 96, color, "wash", r=20, sw=2.4)
        s.badge(M + 40, y + 48, i + 1, color, r=20, size=21)
        s.icon(icon, M + 96, y + 48, 30, COLORS[color][0], sw=2.4)
        s.text(M + 130, y + 42, title, 21, 800, tcolor(color), max_width=W - 2 * M - 160)
        s.text(M + 130, y + 74, note, 19, 700, INK, max_width=W - 2 * M - 160)
        if i < len(steps) - 1:
            s.arrow(W / 2, y + 98, W / 2, y + 118, INK, sw=2.4, head=9)
        y += 120

    zy = 762
    s.box(M, zy, W - 2 * M, 300, "violet", "wash", r=22, dashed=True, sw=2.8)
    zone_title(s, zy, "scroll-text", "violet", "THE TRACE: WHY EACH SECTION IS THERE", right="the stolen phone")
    rows = [("hit 1", "Safety while travelling", "sent", True), ("hit 2", "After you report", "sent", True),
            ("hit 3", "What not to do", "sent", True), ("previous of hit 3", "How to report", "sent: the answer", True),
            ("next of hit 3", "After you report", "already sent", None)]
    for i, (why, section, decision, good) in enumerate(rows):
        ry = zy + 98 + i * 38
        s.text(M + 30, ry, why, 18, 500, TEXT["violet"], "mono", max_width=260)
        s.text(M + 300, ry, section, 19, 800 if i == 3 else 700, INK, max_width=360)
        s.text(W - M - 64, ry, decision, 19, 800 if i == 3 else 700,
               TEXT["green"] if good and i == 3 else TEXT["muted"], anchor="end", max_width=260)
        if good is not None:
            mark(s, W - M - 40, ry - 7, good, 18)
    s.text(M + 30, zy + 286, "5 sections, 188 words: the hits are always sent first", 18, 700, TEXT["muted"],
           max_width=W - 2 * M - 60)

    zy = 1084
    half = (W - 2 * M - 24) / 2
    s.box(M, zy, half, 154, "gray", "wash", r=22, sw=2.4)
    s.icon("power", M + 38, zy + 40, 26, TEXT["gray"], sw=2.4)
    s.text(M + 64, zy + 48, "THE OFF-SWITCH", 20, 800, INK, max_width=half - 90)
    s.chip(M + 26, zy + 70, 'EXPANSION = "off"', "gray", size=17)
    s.text(M + 26, zy + 138, "exactly Day 10's top 3, every query", 17, 700, TEXT["muted"], max_width=half - 52)
    x = M + half + 24
    s.box(x, zy, half, 154, "teal", "wash", r=22, sw=2.4)
    s.text(x + 26, zy + 48, "WHICH NEIGHBOURS? A DEV RULE", 20, 800, TEXT["teal"], max_width=half - 52)
    s.text(x + 26, zy + 84, "both: 90 of 90 found", 19, 800, INK, max_width=half - 52)
    s.text(x + 26, zy + 112, "previous, next, parent: 89", 18, 700, TEXT["muted"], max_width=half - 52)
    s.text(x + 26, zy + 138, "committed before the test", 16, 700, TEXT["muted"], max_width=half - 52)
    return s


# ----------------------------------------------------------------------------- slide 3
def slide_control() -> Slide:
    s = Slide(3, TOTAL, DAY, "Same words, chosen two ways", seed=113)
    heading(s, "Same words, chosen two ways", "The control: as many sections, from further down the ranking.")
    half = (W - 2 * M - 24) / 2
    cards = [(M, "blue", "AROUND THE HITS", "top 3 + neighbours", "51 of 52", "199 words"),
             (M + half + 24, "green", "FURTHER DOWN", "top m, m = sections sent", "52 of 52", "199 words")]
    zy = 274
    for x, color, title, sub, big, words in cards:
        s.box(x, zy, half, 232, color, "wash", r=22, sw=2.8)
        s.text(x + 26, zy + 50, title, 22, 800, tcolor(color), max_width=half - 52)
        s.text(x + 26, zy + 82, sub, 18, 700, TEXT["muted"], max_width=half - 52)
        s.text(x + 26, zy + 162, big, 46, 800, INK, max_width=half - 52)
        s.text(x + 26, zy + 208, words, 20, 700, TEXT["muted"], max_width=half - 52)
    s.text(W / 2, zy + 268, "the top 3 alone: 49 of 52, 110 words", 19, 700, TEXT["muted"], anchor="middle",
           max_width=600)

    zy = 574
    s.box(M, zy, W - 2 * M, 452, "orange", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "search", "orange", "THE TOP 3'S THREE MISSES", right="each ranked 5th")
    s.text(W - M - 160, zy + 96, "around", 18, 800, TEXT["blue"], anchor="middle", max_width=120)
    s.text(W - M - 60, zy + 96, "down", 18, 800, TEXT["green"], anchor="middle", max_width=100)
    misses = [("a train ticket I paid for", "Submitting a claim: next to hit 1", True, True),
              ("I clicked a suspicious link", "How to report: next to hit 1", True, True),
              ("a taxi from the airport", "another document: expense policy", False, True)]
    for i, (q, where, around, down) in enumerate(misses):
        ry = zy + 146 + i * 92
        s.text(M + 30, ry, q, 20, 800, INK, max_width=W - 2 * M - 280)
        s.text(M + 30, ry + 34, where, 18, 700, TEXT["muted"], max_width=W - 2 * M - 280)
        mark(s, W - M - 160, ry + 8, around, 24)
        mark(s, W - M - 60, ry + 8, down, 24)
    s.text(M + 30, zy + 416, "neighbours can't leave the hits' document; the ranking can", 19, 800, TEXT["orange"],
           max_width=W - 2 * M - 60)

    zy = 1048
    s.box(M, zy, W - 2 * M, 190, "yellow", "fill", r=22, sw=3)
    s.icon("scale", M + 40, zy + 44, 30, TEXT["orange"], sw=2.4)
    s.text(M + 70, zy + 54, "WHY THE CONTROL", 23, 800, INK, max_width=W - 2 * M - 100)
    s.text(M + 30, zy + 104, "vs the top 3 it can't lose: it sends the top 3 first", 20, 700, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 144, "only a control says whether structure helped, or size", 20, 800, TEXT["teal"],
           max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 4
def slide_ship() -> Slide:
    s = Slide(4, TOTAL, DAY, "A lookup is not a search", seed=114)
    heading(s, "A lookup is not a search", "Before you ship any step that adds context.")

    zy = 274
    s.box(M, zy, W - 2 * M, 266, "red", "wash", r=22, sw=2.8)
    zone_title(s, zy, "file-lock", "red", "ONE SECTION RESTRICTED TO MANAGERS", right="in memory, as alice")
    rows = [("the search returns it", "0 of 155", True), ("an unchecked neighbour lookup", "10 of 155", False),
            ("the checked expansion", "0", True)]
    for i, (label, value, good) in enumerate(rows):
        ry = zy + 104 + i * 46
        mark(s, M + 46, ry - 7, good, 20)
        s.text(M + 72, ry, label, 20, 700, INK, max_width=520)
        s.text(W - M - 30, ry, value, 20, 800, TEXT["green"] if good else TEXT["red"], anchor="end", max_width=200)
    s.text(M + 30, zy + 246, "neighbours, parents, linked docs: check permissions again", 19, 800, TEXT["red"],
           max_width=W - 2 * M - 60)

    zy = 562
    half = (W - 2 * M - 24) / 2
    s.box(M, zy, half, 230, "orange", "wash", r=22, sw=2.6)
    s.icon("gauge", M + 38, zy + 40, 26, COLORS["orange"][0], sw=2.4)
    s.text(M + 64, zy + 48, "COST", 21, 800, TEXT["orange"], max_width=half - 90)
    s.text(M + 26, zy + 100, "words: 114 → 214", 22, 800, INK, max_width=half - 52)
    s.text(M + 26, zy + 140, "tokens: 137 → 259", 22, 800, INK, max_width=half - 52)
    s.text(M + 26, zy + 178, "mean per answerable query", 16, 700, TEXT["muted"], max_width=half - 52)
    s.text(M + 26, zy + 206, "the lookup itself: microseconds", 16, 700, TEXT["muted"], max_width=half - 52)
    x = M + half + 24
    s.box(x, zy, half, 230, "violet", "wash", r=22, sw=2.6)
    s.icon("message-square", x + 38, zy + 40, 26, COLORS["violet"][0], sw=2.4)
    s.text(x + 64, zy + 48, "IN CONTEXT ≠ ANSWERED", 21, 800, TEXT["violet"], max_width=half - 90)
    s.text(x + 26, zy + 92, "the suspicious-link question", 18, 800, INK, max_width=half - 52)
    s.text(x + 26, zy + 122, "How to report: now sent", 18, 700, INK, max_width=half - 52)
    s.text(x + 26, zy + 152, "the reader copies:", 18, 700, INK, max_width=half - 52)
    s.text(x + 26, zy + 182, "“Do not try to investigate…”", 18, 800, TEXT["red"], max_width=half - 52)
    s.text(x + 26, zy + 208, "Day 12 grades answers", 16, 700, TEXT["muted"], max_width=half - 52)

    zy = 814
    s.box(M, zy, W - 2 * M, 424, "gray", "wash", r=22, dashed=True, sw=2.6)
    s.icon("shield-check", M + 40, zy + 44, 28, TEXT["gray"], sw=2.4)
    s.text(M + 70, zy + 52, "EVERY ADVANCED RETRIEVAL STEP NEEDS", 22, 800, INK, max_width=W - 2 * M - 100)
    checklist = [("power", "an off-switch", "back to the old context, no deploy"),
                 ("scroll-text", "a trace per query", "why each section was sent"),
                 ("lock-keyhole", "a permission check", "on everything it fetches"),
                 ("ruler", "a budget", "and what it skipped, logged"),
                 ("scale", "a control", "the same amount, chosen simply")]
    for i, (icon, title, note) in enumerate(checklist):
        ry = zy + 118 + i * 68
        s.icon(icon, M + 50, ry - 7, 26, TEXT["gray"], sw=2.3)
        s.text(M + 84, ry, title, 21, 800, INK, max_width=330)
        s.text(M + 420, ry, note, 19, 700, TEXT["muted"], max_width=W - 2 * M - 450)
    return s


def main() -> None:
    check_contrast()
    check_facts()
    counts = held_out()
    OUT.mkdir(parents=True, exist_ok=True)
    slides = [("slide-01-cover", slide_cover(counts)), ("slide-02-how-it-works", slide_how()),
              ("slide-03-the-control", slide_control()), ("slide-04-before-you-ship", slide_ship())]
    for name, slide in slides:
        path = OUT / f"{name}.svg"
        path.write_text(slide.svg(), encoding="utf-8")
        print("wrote", path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
