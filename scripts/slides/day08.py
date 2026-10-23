"""Build the Day 08 LinkedIn carousel (whiteboard style) as editable SVGs in assets/day-08/.

    python scripts/slides/day08.py                    # writes the SVGs
    python scripts/slides/render.py assets/day-08     # renders PNGs + PDF

Every number and ID on the slides comes from the executed Day 08 notebook or the files it writes.
`check_facts()` confirms each quoted line appears in the notebook's saved outputs before anything is
written. The table on the cover is read from the committed baseline (evals/baselines/day-08-retrieval.json)
and checked against the notebook's printed table; the timing card is checked for its order of magnitude
only, because timings differ from machine to machine.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from kit import COLORS, INK, M, TEXT, W, Slide, check_contrast

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "day-08"
NOTEBOOK = ROOT / "notebooks" / "day_08_hybrid_search_and_fusion.ipynb"
BASELINE = ROOT / "evals" / "baselines" / "day-08-retrieval.json"
TOTAL = 4
DAY = "08"
PAPER_WHITE = "#FFFFFF"

FACTS = [
    "best score of each query: BM25 1.71 to 15.28 | dense 0.42 to 0.88",
    "raw sum: top 3 identical to BM25's (same chunks, same order) for 62 of the 71 queries",
    "#1    0.0328  expense-reimbursement-policy#scope                    #1 -> 1/61    #1 -> 1/61",
    "#2    0.0323  travel-policy#purpose                                 #2 -> 1/62    #2 -> 1/62",
    "#3    0.0308  expense-reimbursement-policy#meals                    #6 -> 1/66    #4 -> 1/64   <- the answer",
    "#4    0.0299  vpn-access-guide#installing-the-client                #9 -> 1/69    #5 -> 1/65",
    "ties that decide whether a labelled chunk is in the top 3: 0",
    "RRF, k = 20                26  30  0.90    20  30  0.82",
    "RRF, k = 1000              26  30  0.90    20  30  0.82",
    "questions, BM25       31     24     29     29     30      0.94    0.85   94% (79% to 98%)",
    "questions, dense      31     26     29     31     31      0.94    0.90   94% (79% to 98%)",
    "questions, RRF        31     26     30     30     30      0.97    0.90   97% (84% to 99%)",
    "questions, weighted   31     27     31     31     31      1.00    0.92   100% (89% to 100%)",
    "lookups, BM25         30     30     30     30     30      1.00    1.00   100% (89% to 100%)",
    "lookups, dense        30     12     19     26     29      0.62    0.56   63% (46% to 78%)",
    "lookups, RRF          30     20     30     30     30      1.00    0.82   100% (89% to 100%)",
    "lookups, weighted     30     30     30     30     30      1.00    1.00   100% (89% to 100%)",
    "the 12 chunks BM25 returns fill RRF's ranks 1 to 12",
    "#13   0.0164  parental-leave-policy#returning-to-work             not returned    #1 -> 1/61",
    "#14   0.0161  parental-leave-policy#who-is-eligible               not returned    #2 -> 1/62",
    "#15   0.0159  parental-leave-policy#leave-entitlement             not returned    #3 -> 1/63   <- the answer",
    "    #1  6.283  password-and-mfa-standard#account-lockout   <- the answer",
    "    #2  3.030  remote-work-policy#core-hours",
    "  dense ranks the answer #19, RRF #2",
    "lookups BM25 ranks first and RRF doesn't: 10",
    "0.7           13  16  0.89          30  30  1.00        46  0.96  <- the rule's choice",
    "answerable questions, test (15): BM25 11/14 | dense 13/14 | RRF 13/15 | weighted 14/15",
    "dense, header                              26  29  0.90    12  19  0.56",
    "dense, no header                           27  31  0.92    12  19  0.57",
    "RRF: BM25 header, dense header             26  30  0.90    20  30  0.82   <- today",
    "RRF: BM25 header, dense no header          26  30  0.90    19  29  0.79",
    "weighted: BM25 header, dense header        27  31  0.92    30  30  1.00   <- today",
    "weighted: BM25 header, dense no header     28  31  0.95    30  30  1.00",
    "dense without the header, questions, top 3: gains 2 ['d03-13', 'd07-a08'], loses 0 [], sign test p = 0.5",
    "restricted chunks in any top 10 (all 5 retrievers, 81 queries): 0",
    "RRF: 8 queries (alice) get a chunk the asker may not read into the fused ranking; best rank 4",
    "BM25 and dense against evals/baselines/day-07-retrieval.json: no changes",
    "compared with evals/baselines/day-08-retrieval.json: no changes",
    "All Day 08 sanity checks passed.",
]


def notebook_outputs() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code"
                   for o in c.get("outputs", []))


def check_facts() -> dict[str, float]:
    outputs = notebook_outputs()
    missing = [f for f in FACTS if f not in outputs]
    if missing:
        raise SystemExit(f"Slide facts not found (re-run the notebook?): {missing}")
    # Slide 4's timing card states orders of magnitude only ("about 10 ms", "under 0.1 ms"): check them
    medians = {name: float(ms) for name, ms in re.findall(r"^(.+?)\s+median\s+([\d.]+) ms", outputs, re.M)}
    dense = medians["dense (embed the question + search)"]
    cheap = [medians["BM25 (full ranking)"], medians["RRF fusion"], medians["weighted fusion"]]
    if not (5 <= dense <= 20 and max(cheap) < 0.1):
        raise SystemExit(f"timings don't match the slide's wording: {medians}")
    return medians


def cover_rows() -> list[tuple[str, int, int, int]]:
    # (retriever, questions in the top 3, lookups in the top 3, lookups first), from the committed baseline
    metrics = json.loads(BASELINE.read_text(encoding="utf-8"))["metrics"]
    rows = [(name, metrics[f"questions, {name}"]["hit@3"], metrics[f"lookups, {name}"]["hit@3"],
             metrics[f"lookups, {name}"]["hit@1"]) for name in ("BM25", "dense", "RRF", "weighted")]
    if rows != [("BM25", 29, 30, 30), ("dense", 29, 19, 12), ("RRF", 30, 30, 20), ("weighted", 31, 30, 30)]:
        raise SystemExit(f"the baseline gives {rows}, not the notebook's printed table")
    return rows


def heading(s: Slide, title: str, subtitle: str, size: int = 54) -> None:
    s.text(M, 190, title, size, 800, INK)
    s.text(M, 238, subtitle, 25, 500, TEXT["muted"])


def tcolor(color: str) -> str:
    return TEXT.get(color, INK) if color != "yellow" else INK


def zone_title(s: Slide, y: float, icon: str, color: str, title: str, right: str | None = None) -> None:
    s.icon(icon, M + 38, y + 40, 28, COLORS[color][0] if color != "gray" else TEXT["gray"], sw=2.4)
    s.text(M + 64, y + 49, title, 22, 800, tcolor(color), max_width=W - 2 * M - 90 - (240 if right else 0))
    if right:
        s.text(W - M - 28, y + 49, right, 19, 700, TEXT["muted"], anchor="end", max_width=240)


def ladder(s: Slide, x: float, y: float, label: str, answer_rank: int, slots: int, color: str,
           w: float = 150, step: float = 36, found: bool | None = None) -> None:
    """A ranking drawn as a column of slots, with the answer's slot coloured."""
    s.text(x + w / 2, y - 14, label, 23, 800, INK, anchor="middle", max_width=w + 40)
    for i in range(slots):
        top = y + i * step
        is_answer = i + 1 == answer_rank
        s.box(x, top, w, 28, color if is_answer else "gray", "fill" if is_answer else "wash", r=8, sw=2,
              jitter=0.6)
        s.text(x + 14, top + 21, f"#{i + 1}", 18, 800 if is_answer else 600, INK if is_answer else TEXT["muted"],
               max_width=40)
        if is_answer:
            s.text(x + 52, top + 21, "Meals", 18, 800, INK, max_width=w - 76)
            if found is not None:
                s.icon("check" if found else "x", x + w - 16, top + 14, 18, COLORS["green" if found else "red"][0],
                       sw=3.2)


# ----------------------------------------------------------------------------- slide 1
def slide_cover(rows: list[tuple[str, int, int, int]]) -> Slide:
    s = Slide(1, TOTAL, DAY, "Two rankings, one list", seed=81)
    heading(s, "Two rankings, one list", "Day 08: hybrid search, measured on Day 07's 81 labelled queries.")

    # the wine question: two misses, one hit
    zy = 274
    s.box(M, zy, W - 2 * M, 428, "blue", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "search", "blue", "THE QUESTION BOTH RETRIEVERS MISSED")
    s.text(M + 30, zy + 92, "“Will Acme pay for wine with a business dinner?”", 22, 700, INK,
           max_width=W - 2 * M - 60)
    ly = zy + 160
    s.box(M + 16, ly - 50, 354, 264, "gray", PAPER_WHITE, r=16, sw=2.2, dashed=True, jitter=0.6)
    ladder(s, M + 26, ly, "BM25", 6, 6, "violet", found=False)
    ladder(s, M + 204, ly, "dense", 4, 6, "violet", found=False)
    # the fusion step
    fx, fy = M + 410, ly + 56
    s.arrow(M + 376, fy + 48, fx - 6, fy + 48, INK, sw=2.6, head=12)
    s.box(fx, fy, 190, 96, "yellow", "fill", r=18, sw=2.6)
    s.icon("merge", fx + 30, fy + 30, 26, TEXT["orange"], sw=2.4)
    s.text(fx + 54, fy + 38, "RRF", 22, 800, INK, max_width=120)
    s.text(fx + 95, fy + 76, "Σ 1/(60 + rank)", 17, 600, INK, family="mono", anchor="middle", max_width=176)
    s.arrow(fx + 196, fy + 48, M + 636, fy + 48, INK, sw=2.6, head=12)
    ladder(s, M + 644, ly, "hybrid", 3, 6, "green", found=True)
    # "top 3" bracket
    bx = M + 806
    s.line(bx, ly + 2, bx + 12, ly + 2, INK, sw=2.4, bow=0)
    s.line(bx + 12, ly + 2, bx + 12, ly + 100, INK, sw=2.4, bow=0.1)
    s.line(bx, ly + 100, bx + 12, ly + 100, INK, sw=2.4, bow=0)
    s.text(bx + 24, ly + 58, "top 3", 20, 800, TEXT["green"], max_width=110)
    s.text(M + 30, zy + 408, "#6 and #4: both fairly high. Agreement lifts it to #3: 1/66 + 1/64.", 20, 700,
           TEXT["teal"], max_width=W - 2 * M - 60)

    # the measured result
    zy = 730
    s.box(M, zy, W - 2 * M, 508, "gray", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "list-ordered", "gray", "THE SAME 81 QUERIES, 4 WAYS")
    cols = [("31 questions", "in the top 3", 31, "blue"), ("30 lookups", "in the top 3", 30, "orange"),
            ("30 lookups", "ranked first", 30, "violet")]
    x0, cw, bar = M + 196, 248, 150
    for j, (top, bottom, _, color) in enumerate(cols):
        s.text(x0 + j * cw, zy + 100, top, 19, 800, tcolor(color), max_width=cw - 20)
        s.text(x0 + j * cw, zy + 124, bottom, 17, 600, TEXT["muted"], max_width=cw - 20)
    for i, (name, questions, lookups, first) in enumerate(rows):
        y = zy + 180 + i * 60
        s.text(M + 30, y + 8, name, 24, 800, INK, max_width=150)
        for j, (value, (_, _, n, color)) in enumerate(zip((questions, lookups, first), cols)):
            x = x0 + j * cw
            s.box(x, y - 14, bar, 28, "gray", PAPER_WHITE, r=8, sw=1.8, jitter=0.5)
            if value:
                s.box(x, y - 14, bar * value / n, 28, color, "fill", r=8, sw=2, jitter=0.5)
            full = value == n
            s.text(x + bar + 10, y + 8, f"{value}/{n}", 20, 800, INK if full else TEXT["muted"], max_width=84)
    s.text(M + 30, zy + 440, "RRF: no weight to tune, but 10 lookups lose first place", 20, 700, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 478, "weighted: min-max scores, 0.7 dense + 0.3 BM25, weight chosen on dev", 20, 700,
           INK, max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 2
def slide_rrf() -> Slide:
    s = Slide(2, TOTAL, DAY, "How RRF fuses rankings", seed=82)
    heading(s, "Fuse positions, not scores", "Two scales that never meet, and a fix that uses ranks only.")

    # score mismatch on one axis
    zy = 274
    s.box(M, zy, W - 2 * M, 300, "orange", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "ruler", "orange", "WHY NOT ADD THE SCORES?", "best score per query")
    x_left, x_right, top_value = M + 170, W - M - 50, 16.0

    def px(v: float) -> float:
        return x_left + v / top_value * (x_right - x_left)

    for k, (name, lo, hi, color) in enumerate([("BM25", 1.71, 15.28, "orange"), ("cosine", 0.42, 0.88, "blue")]):
        y = zy + 108 + k * 62
        s.text(M + 30, y + 8, name, 23, 800, INK, max_width=120)
        s.box(px(lo), y - 15, max(px(hi) - px(lo), 10), 30, color, "fill", r=8, sw=2, jitter=0.5)
        label = f"{lo:.2f} to {hi:.2f}"
        if px(hi) - px(lo) > 240:
            s.text((px(lo) + px(hi)) / 2, y + 8, label, 19, 800, INK, anchor="middle", max_width=px(hi) - px(lo))
        else:
            s.text(px(hi) + 14, y + 8, label, 19, 800, INK, max_width=220)
    axis_y = zy + 206
    s.line(x_left, axis_y, x_right, axis_y, INK, sw=2, bow=0.1)
    for v in (0, 4, 8, 12, 16):
        s.line(px(v), axis_y, px(v), axis_y + 8, INK, sw=2, bow=0)
        s.text(px(v), axis_y + 32, str(v), 17, 700, TEXT["muted"], anchor="middle", max_width=40)
    s.text(M + 30, zy + 278, "added together, BM25 wins: the raw sum = BM25's top 3 for 62 of 71 queries", 20, 800,
           TEXT["orange"], max_width=W - 2 * M - 60)

    # the formula and the worked example
    zy = 604
    s.box(M, zy, W - 2 * M, 446, "blue", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "sigma", "blue", "RECIPROCAL RANK FUSION")
    s.text(M + 30, zy + 108, "score(chunk) = Σ  1 / (60 + its rank in each list)", 26, 600, INK, family="mono",
           max_width=W - 2 * M - 60)
    worth = [("#1", "0.0164"), ("#3", "0.0159"), ("#10", "0.0143"), ("not returned", "0")]
    x = M + 30
    for label, value in worth:
        w = s.chip(x, zy + 134, f"{label} → {value}", "blue", size=19)
        x += w + 14
    # worked example: the wine question
    s.text(M + 30, zy + 230, "chunk (wine question)", 19, 800, TEXT["muted"], max_width=300)
    for j, head in enumerate(("BM25", "dense", "hybrid")):
        s.text(M + 470 + j * 150, zy + 230, head, 19, 800, TEXT["muted"], anchor="middle", max_width=130)
    example = [("expense › Scope", 1, 1, 1, False), ("travel › Purpose", 2, 2, 2, False),
               ("expense › Meals", 6, 4, 3, True), ("vpn › installing", 9, 5, 4, False)]
    for i, (chunk, b, d, h, answer) in enumerate(example):
        y = zy + 276 + i * 40
        if answer:
            s.box(M + 20, y - 27, W - 2 * M - 40, 38, "green", "fill", r=10, sw=2, jitter=0.5)
        s.text(M + 30, y, chunk, 21, 800 if answer else 600, INK, max_width=330)
        for j, value in enumerate((b, d, h)):
            s.text(M + 470 + j * 150, y, f"#{value}", 21, 800 if answer and j == 2 else 600, INK, anchor="middle",
                   max_width=80)
        if answer:
            s.icon("check", W - M - 52, y - 8, 22, COLORS["green"][0], sw=3.2)
    s.text(M + 30, zy + 434, "the answer: neither retriever's first, both fairly high", 19, 700, TEXT["teal"],
           max_width=W - 2 * M - 60)

    # two small facts
    zy = 1080
    cw = (W - 2 * M - 20) / 2
    cards = [("copy", "violet", "TIES ARE COMMON", ["(2, 5) and (5, 2) score the", "same: break by chunk id"]),
             ("gauge", "teal", "k BARELY MATTERS HERE", ["k = 20 to 1000: identical", "counts on 81 queries"])]
    for k, (icon, color, title, lines) in enumerate(cards):
        x = M + k * (cw + 20)
        s.box(x, zy, cw, 158, color, "wash", r=20, sw=2.6)
        s.icon(icon, x + 36, zy + 38, 26, COLORS[color][0], sw=2.4)
        s.text(x + 62, zy + 46, title, 20, 800, tcolor(color), max_width=cw - 80)
        for i, line in enumerate(lines):
            s.text(x + 24, zy + 92 + i * 34, line, 19, 600, INK, max_width=cw - 48)
    return s


# ----------------------------------------------------------------------------- slide 3
def slide_losses() -> Slide:
    s = Slide(3, TOTAL, DAY, "What RRF throws away", seed=83)
    heading(s, "What RRF throws away", "Both losses come from ignoring the scores.")

    # 1. agreement beats a single find
    zy = 274
    s.box(M, zy, W - 2 * M, 322, "blue", "wash", r=22, sw=2.8)
    s.badge(M + 38, zy + 40, 1, "yellow", r=16, size=18)
    s.text(M + 66, zy + 48, "AGREEMENT BEATS A SINGLE FIND", 22, 800, TEXT["blue"], max_width=600)
    s.text(M + 30, zy + 92, "“How much time off do new dads get?”", 22, 700, INK, max_width=W - 2 * M - 60)
    slot_w, gap, sx, sy = 52, 7, M + 34, zy + 150
    for i in range(15):
        x = sx + i * (slot_w + gap)
        dense_only = i >= 12
        answer = i == 14
        s.box(x, sy, slot_w, 52, "red" if answer else ("violet" if dense_only else "gray"),
              "fill" if dense_only else "wash", r=8, sw=2, jitter=0.5)
        s.text(x + slot_w / 2, sy + 33, f"{i + 1}", 18, 800 if dense_only else 600,
               INK if dense_only else TEXT["muted"], anchor="middle", max_width=slot_w - 4)
        if answer:
            s.icon("x", x + slot_w - 9, sy + 9, 14, COLORS["red"][0], sw=3.4)
    # the top 3, which is what a reader of three chunks gets
    top3_w = 3 * (slot_w + gap) - gap
    s.line(sx, sy - 9, sx + top3_w, sy - 9, COLORS["gray"][0], sw=2.4, bow=0)
    s.text(sx + top3_w / 2, sy - 18, "top 3", 18, 800, TEXT["muted"], anchor="middle", max_width=top3_w)
    s.line(sx, sy + 70, sx + 12 * (slot_w + gap) - gap, sy + 70, COLORS["gray"][0], sw=2.4, bow=0.1)
    s.text(sx, sy + 102, "BM25's 12 chunks: 2 votes each", 19, 700, INK, max_width=400)
    s.text(sx, sy + 130, "none about parental leave", 18, 600, TEXT["muted"], max_width=400)
    rx = sx + 12 * (slot_w + gap)
    s.line(rx, sy + 70, rx + 3 * (slot_w + gap) - gap, sy + 70, COLORS["violet"][0], sw=2.4, bow=0.1)
    s.text(W - M - 30, sy + 102, "dense's top 3: 1 vote", 19, 700, INK, anchor="end", max_width=330)
    s.text(W - M - 30, sy + 130, "the answer ends up #15", 18, 800, TEXT["red"], anchor="end", max_width=330)

    # 2. confidence ignored
    zy = 620
    s.box(M, zy, W - 2 * M, 256, "orange", "wash", r=22, sw=2.8)
    s.badge(M + 38, zy + 40, 2, "yellow", r=16, size=18)
    s.text(M + 66, zy + 48, "HOW SURE A RETRIEVER IS GETS LOST", 22, 800, TEXT["orange"], max_width=640)
    s.text(M + 30, zy + 92, "lookup: “15 minutes”", 22, 700, INK, max_width=400)
    bars = [("BM25 #1 (the answer)", 6.283, "orange"), ("BM25 #2", 3.030, "gray")]
    for i, (label, value, color) in enumerate(bars):
        y = zy + 136 + i * 44
        s.text(M + 30, y + 7, label, 18, 700, INK, max_width=230)
        s.box(M + 260, y - 13, value / 6.283 * 300, 26, color, "fill", r=8, sw=2, jitter=0.5)
        s.text(M + 270 + value / 6.283 * 300, y + 7, f"{value:.2f}", 18, 800, INK, max_width=70)
    s.text(W - M - 30, zy + 143, "dense: #19", 21, 800, INK, anchor="end", max_width=220)
    s.arrow(W - M - 120, zy + 158, W - M - 120, zy + 186, INK, sw=2.4, head=10)
    s.text(W - M - 30, zy + 214, "RRF: #2", 21, 800, TEXT["red"], anchor="end", max_width=220)
    s.text(M + 30, zy + 230, "a lead of 2x becomes just “rank 1”: 10 of 30 lookups lose first place", 19, 700,
           TEXT["muted"], max_width=W - 2 * M - 60)

    # the fix that keeps the scores
    zy = 902
    s.box(M, zy, W - 2 * M, 336, "green", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "scale", "green", "KEEP THE SCORES: A WEIGHTED FUSION")
    s.text(M + 30, zy + 100, "rescale each list to 0..1, then  0.7 × dense + 0.3 × BM25", 22, 700, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 138, "the weight: chosen by a rule on 46 dev queries (not pre-registered)", 19, 600,
           TEXT["muted"], max_width=W - 2 * M - 60)
    results = [("questions in the top 3", "31/31", "green"), ("lookups ranked first", "30/30", "green")]
    x = M + 30
    for label, value, color in results:
        w = s.tag(x, zy + 202, f"{value}  {label}", color, size=20)
        x += w + 18
    s.text(M + 30, zy + 258, "15 test questions: RRF 15/15, weighted 15/15 in the top 3", 20, 800, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 296, "a tie, and only a check: test results were seen while exploring", 19, 700, TEXT["teal"],
           max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 4
def slide_checks(medians: dict[str, float]) -> Slide:
    s = Slide(4, TOTAL, DAY, "Before you ship hybrid search", seed=84)
    heading(s, "Before you ship hybrid", "Three checks and a decision, from today's measurements.")

    boxes = [
        ("files", "violet", "SAME TEXT FOR BOTH RETRIEVERS?", [
            ("alone, dense prefers no header", INK, 700),
            ("(2 questions, p = 0.5); fused, no gain", INK, 700),
            ("keep one text, one pipeline", TEXT["violet"], 800)]),
        ("lock-keyhole", "red", "FILTER EVERY RETRIEVER", [
            ("shared BM25 index: a managers-only", INK, 700),
            ("chunk in 8 of alice's rankings", INK, 700),
            ("ranking low is still a leak", TEXT["red"], 800)]),
        ("timer", "teal", "THE SECOND RETRIEVER IS CHEAP", [
            ("embedding the question: ~10 ms", INK, 700),
            ("BM25 + fusion: < 0.1 ms (44 chunks)", INK, 700),
            ("the model call dominates", TEXT["teal"], 800)]),
    ]
    zy = 274
    for k, (icon, color, title, lines) in enumerate(boxes):
        y = zy + k * 226
        s.box(M, y, W - 2 * M, 204, color, "wash", r=22, sw=2.8)
        s.node(M + 70, y + 102, icon, color, None, size=92, icon_size=48)
        s.text(M + 142, y + 54, title, 22, 800, tcolor(color), max_width=W - 2 * M - 170)
        for i, (line, ink, weight) in enumerate(lines):
            s.text(M + 142, y + 98 + i * 38, line, 20, weight, ink, max_width=440)
        if k == 0:      # dense alone with and without the header line, questions in the top 3
            lx = W - M - 262
            s.text(lx, y + 44, "dense alone, questions", 16, 700, TEXT["muted"], max_width=240)
            for r, (label, value) in enumerate((("header", 29), ("no header", 31))):
                top = y + 60 + r * 44
                s.box(lx, top, 160 * value / 31, 26, "violet", "fill", r=7, sw=1.8, jitter=0.4)
                s.text(lx + 10, top + 19, label, 16, 800, INK, max_width=150)
                s.text(lx + 170, top + 19, f"{value}/31", 16, 800, INK, max_width=70)
            s.text(lx, y + 176, "fused: no change", 16, 700, TEXT["muted"], max_width=240)
        if k == 1:      # alice's fused ranking with the leaked chunk at #4, just below the top 3
            lx = W - M - 262
            for r in range(5):
                top = y + 30 + r * 31
                leak = r == 3
                s.box(lx, top, 172, 26, "red" if leak else "gray", "fill" if leak else PAPER_WHITE, r=7,
                      sw=1.8, jitter=0.4)
                s.text(lx + 12, top + 19, f"#{r + 1}", 16, 800 if leak else 600, INK if leak else TEXT["muted"],
                       max_width=40)
                if leak:
                    s.text(lx + 44, top + 19, "managers", 16, 800, INK, max_width=96)
                    s.icon("lock-keyhole", lx + 154, top + 13, 16, COLORS["red"][0], sw=2.6)
            s.line(lx + 182, y + 31, lx + 192, y + 31, INK, sw=2, bow=0)
            s.line(lx + 192, y + 31, lx + 192, y + 117, INK, sw=2, bow=0.1)
            s.line(lx + 182, y + 117, lx + 192, y + 117, INK, sw=2, bow=0)
            s.text(lx + 200, y + 80, "top 3", 16, 800, TEXT["muted"], max_width=W - M - lx - 200)
        if k == 2:      # the cost of each step, to scale
            dense_ms, cheap_ms = medians["dense (embed the question + search)"], max(
                medians["BM25 (full ranking)"], medians["RRF fusion"], medians["weighted fusion"])
            scale = 240 / dense_ms
            for r, (label, ms) in enumerate((("embed + search", dense_ms), ("BM25 + fusion", cheap_ms))):
                top = y + 52 + r * 62
                s.text(W - M - 270, top - 6, label, 16, 700, TEXT["muted"], max_width=200)
                s.box(W - M - 270, top, max(ms * scale, 4), 24, "teal", "fill", r=6, sw=1.8, jitter=0.4)

    zy = 966
    s.box(M, zy, W - 2 * M, 272, "yellow", "fill", r=22, sw=3)
    s.icon("scale", M + 40, zy + 44, 30, TEXT["orange"], sw=2.4)
    s.text(M + 70, zy + 54, "DECISION FOR THIS CORPUS", 23, 800, INK, max_width=W - 2 * M - 100)
    s.text(M + 30, zy + 106, "weighted fusion (0.7 dense + 0.3 BM25): no miss in the top 3", 21, 700, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 146, "RRF kept as the no-labels alternative: nothing to tune", 21, 700, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 186, "never add raw scores from different retrievers", 21, 700, INK, max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 236, "next: Day 09 tests a reranker on the same 81 queries", 20, 800, TEXT["orange"],
           max_width=W - 2 * M - 60)
    return s


def main() -> None:
    check_contrast()
    medians = check_facts()
    rows = cover_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    slides = [("slide-01-cover", slide_cover(rows)), ("slide-02-how-rrf-works", slide_rrf()),
              ("slide-03-what-rrf-throws-away", slide_losses()), ("slide-04-before-you-ship", slide_checks(medians))]
    for name, slide in slides:
        path = OUT / f"{name}.svg"
        path.write_text(slide.svg(), encoding="utf-8")
        print("wrote", path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
