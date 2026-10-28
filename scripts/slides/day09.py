"""Build the Day 09 LinkedIn carousel (whiteboard style) as editable SVGs in assets/day-09/.

    python scripts/slides/day09.py                    # writes the SVGs
    python scripts/slides/render.py assets/day-09     # renders PNGs + PDF

Every number and ID on the slides comes from the executed Day 09 notebook or the files it writes.
`check_facts()` confirms each quoted line appears in the notebook's saved outputs before anything is
written. The score strip on slide 3 is read from the committed baseline (evals/baselines/day-09-retrieval.json)
and the questions file, and checked against the notebook's printed strip; the timing card is drawn to scale
from the notebook's printed medians and checked for its order of magnitude only, because timings differ
from machine to machine.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from kit import COLORS, INK, M, TEXT, W, Slide, check_contrast

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "day-09"
NOTEBOOK = ROOT / "notebooks" / "day_09_reranking_and_context_selection.ipynb"
BASELINE = ROOT / "evals" / "baselines" / "day-09-retrieval.json"
QUESTIONS = ROOT / "evals" / "questions-v1.jsonl"
TOTAL = 4
DAY = "09"
PAPER_WHITE = "#FFFFFF"
GATE = -3.6
FIVE = ("d07-u01", "d07-u03", "d07-u04", "d07-u11", "d07-u12")

FACTS = [
    "BM25, dense, RRF and weighted against evals/baselines/day-08-retrieval.json: no changes",
    "model card example: 8.6071 and -4.3201 (the card prints 8.607138 and -4.320078)",
    "parental-leave-policy#returning-to-work          #1  0.721             #1          #3   -4.651",
    "parental-leave-policy#who-is-eligible            #2  0.672             #2          #2   -4.217",
    "parental-leave-policy#leave-entitlement          #3  0.665             #3          #1   -3.565   <- the answer",
    "weighted            46 / 0.9855     45 / 0.9837     45 / 0.9837     45 / 0.9826     45 / 0.9826   -> N = 3",
    "RRF                 45 / 0.9674     44 / 0.9620     44 / 0.9609     45 / 0.9826     45 / 0.9826   -> N = 20",
    "weighted+rerank vs weighted, dev questions (16), labelled chunk first: both 13 | only weighted+rerank 2 | only weighted 0",
    "weighted+rerank vs weighted, test questions (15), labelled chunk first: both 13 | only weighted+rerank 0 | only weighted 1",
    "RRF+rerank vs RRF, lookups (30), labelled chunk first: both 20 | only RRF+rerank 9 | only RRF 0 | neither 0 | sign test p = 0.0039 | near ties set apart: d04-v10",
    "weighted+rerank  d03-04   dev      2 -> 1   'how long' meets '16 weeks'",
    "RRF+rerank       d01-01   dev     15 -> 1",
    "RRF+rerank       d03-13   dev      2 -> 5   each chunk scored alone: How to report never mentions a laptop",
    "weighted+rerank  d07-a02  test     1 -> 2   USD 40 vs 'over USD 25' is arithmetic",
    "RRF+rerank       d07-a08  test     3 -> 4   'wine' never appears; Alcohol does",
    "weakest answerable dev question: d01-01 \"How much time off do new dads get?\" at -3.5646 -> gate -3.6",
    "lowest answerable test questions: d07-a02 -7.38",
    "the gate refuses 0 of the 5 and 1 of the 15 answerable test questions -> does not separate them",
    "a cap of 1 per document loses: d03-13, d07-a02, d07-a08",
    "Day 06's budget, 300 words, is about 363 of these tokens",
    "  #1  +6.901  remote-work-policy#equipment-and-home-office        USD 500",
    "  #2  +6.864  remote-work-policy-2025#equipment-and-home-office   USD 400",
    "  #3  +6.845  remote-work-policy-copy#equipment-and-home-office   USD 500",
    "  allowances in it: USD 400 and USD 500 | dropped: remote-work-policy-copy#equipment-and-home-office (repeated text)",
    "RRF #4 -> reranked #1",
    "RRF #6 -> reranked #1",
    "D1, reranking in the retriever: test questions first 13 (weighted alone 14), in the top 3 15 of 15 -> not adopted",
    "D2, the reranker gate: refuses 1 of the 15 answerable test questions and 1 of the 6 unanswerable test questions Day 07's gate and names rule let through -> recorded as measured, not adopted",
    "compared with evals/baselines/day-09-retrieval.json: no changes",
    "All Day 09 sanity checks passed.",
]
TIMING_STEPS = ("dense (embed the question + search)", "rerank 3 candidates", "rerank 20 candidates")


def notebook_outputs() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code"
                   for o in c.get("outputs", []))


def check_facts() -> dict[str, float]:
    outputs = notebook_outputs()
    missing = [f for f in FACTS if f not in outputs]
    if missing:
        raise SystemExit(f"Slide facts not found (re-run the notebook?): {missing}")
    # Slide 4 labels each bar with its median rounded to tens ("~10 ms") and draws the bars to scale; the post and the
    # lesson say "about 10 ms", "30 to 40 ms" and "a quarter of a second": check the magnitudes match that wording
    medians = {name: float(ms) for name, ms in re.findall(r"^(.+?)\s+median\s+([\d.]+) ms", outputs, re.M)}
    dense, three, twenty = (medians[s] for s in TIMING_STEPS)
    if not (5 <= dense <= 15 and 28 <= three <= 45 and 200 <= twenty <= 300):
        raise SystemExit(f"timings don't match the slide's wording: {medians}")
    return medians


def strip_points() -> dict[str, list[tuple[str, float, bool]]]:
    # Each natural question's best reranker score, from the committed baseline: (id, score, answerable) per split
    queries = {q["id"]: q for q in map(json.loads, QUESTIONS.read_text(encoding="utf-8").splitlines())}
    entries = json.loads(BASELINE.read_text(encoding="utf-8"))["queries"]
    points = {"dev": [], "test": []}
    for qid, q in queries.items():
        if q["kind"] == "question":
            points[q["split"]].append((qid, entries[qid]["weighted+rerank"]["top_score"], bool(q["answer_chunks"])))
    # Cross-check with the notebook's printed strip (bins of 1.0): the same dots in the same bins
    outputs = notebook_outputs()
    printed = re.findall(r"^([+-]\s?\d+) to\s+[+-]\s?\d+   (.*)$", outputs, re.M)
    for low, row in printed:
        b = int(low.replace(" ", ""))
        dev_cells, test_cells = row[:26].strip(), row[26:].strip()
        for split, cells in (("dev", dev_cells), ("test", test_cells)):
            mine = [p for p in points[split] if math.floor(p[1]) == b]
            want = "".join("●" for p in mine if p[2]) + "".join("○" for p in mine if not p[2])
            if want != cells:
                raise SystemExit(f"strip bin {b} {split}: baseline gives {want!r}, the notebook printed {cells!r}")
    if len(points["dev"]) != 24 or len(points["test"]) != 27 or not printed:
        raise SystemExit("the strip should have 24 dev and 27 test questions")
    return points


def heading(s: Slide, title: str, subtitle: str, size: int = 54) -> None:
    s.text(M, 190, title, size, 800, INK)
    s.text(M, 238, subtitle, 25, 500, TEXT["muted"])


def tcolor(color: str) -> str:
    return TEXT.get(color, INK) if color != "yellow" else INK


def zone_title(s: Slide, y: float, icon: str, color: str, title: str, right: str | None = None) -> None:
    s.icon(icon, M + 38, y + 40, 28, COLORS[color][0] if color != "gray" else TEXT["gray"], sw=2.4)
    s.text(M + 64, y + 49, title, 22, 800, tcolor(color), max_width=W - 2 * M - 90 - (260 if right else 0))
    if right:
        s.text(W - M - 28, y + 49, right, 19, 700, TEXT["muted"], anchor="end", max_width=260)


def ladder(s: Slide, x: float, y: float, label: str, labels: list[str], answer: int, color: str,
           w: float = 300, step: float = 40) -> None:
    """A short ranking drawn as slots, with the answer's slot coloured and ticked."""
    s.text(x, y - 16, label, 21, 800, INK, max_width=w)
    for i, name in enumerate(labels):
        top = y + i * step
        is_answer = i == answer
        s.box(x, top, w, 32, color if is_answer else "gray", "fill" if is_answer else PAPER_WHITE, r=8, sw=2,
              jitter=0.6)
        s.text(x + 14, top + 23, f"#{i + 1}", 18, 800 if is_answer else 600, INK if is_answer else TEXT["muted"],
               max_width=40)
        s.text(x + 56, top + 23, name, 18, 800 if is_answer else 600, INK, max_width=w - 96)
        if is_answer:
            s.icon("check", x + w - 20, top + 16, 18, COLORS["green"][0], sw=3.2)


# ----------------------------------------------------------------------------- slide 1
def slide_cover() -> Slide:
    s = Slide(1, TOTAL, DAY, "A second look at the candidates", seed=91)
    heading(s, "A second look, measured", "Day 09: a cross-encoder reranks the hybrid's candidates.")

    # two ways to score a chunk
    zy = 274
    s.box(M, zy, W - 2 * M, 372, "blue", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "search", "blue", "TWO WAYS TO SCORE A CHUNK")
    half = (W - 2 * M - 60) / 2
    # bi-encoder: two separate vectors
    bx = M + 24
    s.box(bx, zy + 80, half, 268, "gray", PAPER_WHITE, r=18, sw=2.2, jitter=0.6)
    s.text(bx + 20, zy + 120, "bi-encoder (dense, Day 04)", 21, 800, TEXT["blue"], max_width=half - 40)
    for k, (label, icon) in enumerate((("question", "message-circle"), ("chunk", "file-text"))):
        y = zy + 168 + k * 70
        s.icon(icon, bx + 38, y, 24, COLORS["blue"][0], sw=2.4)
        s.text(bx + 60, y + 7, label, 19, 700, INK, max_width=110)
        s.arrow(bx + 160, y, bx + 214, y, INK, sw=2.2, head=10)
        s.chip(bx + 222, y - 16, "vector", "blue", size=17)
    s.text(bx + 20, zy + 310, "made apart, then compared", 18, 700, TEXT["muted"], max_width=half - 40)
    s.text(bx + 20, zy + 336, "fast: chunk vectors made in advance", 17, 600, TEXT["muted"], max_width=half - 40)
    # cross-encoder: one input, one score
    cx = M + 36 + half
    s.box(cx, zy + 80, half, 268, "violet", "fill", r=18, sw=2.4, jitter=0.6)
    s.text(cx + 20, zy + 120, "cross-encoder (today)", 21, 800, TEXT["violet"], max_width=half - 40)
    s.box(cx + 20, zy + 146, half - 40, 46, "gray", PAPER_WHITE, r=10, sw=2, jitter=0.5)
    s.text(cx + (half) / 2, zy + 176, "question + chunk", 19, 800, INK, anchor="middle", max_width=half - 60)
    s.arrow(cx + half / 2, zy + 196, cx + half / 2, zy + 228, INK, sw=2.4, head=10)
    s.icon("cpu", cx + half / 2, zy + 250, 30, COLORS["violet"][0], sw=2.4)
    s.text(cx + half / 2 + 26, zy + 258, "one score", 19, 800, INK, max_width=150)
    s.text(cx + 20, zy + 310, "read together, word by word", 18, 700, INK, max_width=half - 40)
    s.text(cx + 20, zy + 336, "slow: one model run per pair", 17, 700, TEXT["violet"], max_width=half - 40)

    # the new dads example
    zy = 672
    s.box(M, zy, W - 2 * M, 284, "green", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "list-ordered", "green", "SAME THREE CHUNKS, ANOTHER ORDER")
    s.text(M + 30, zy + 88, "“How much time off do new dads get?”", 21, 700, INK, max_width=W - 2 * M - 60)
    names = {"r": "Returning to work", "e": "Who is eligible", "a": "Leave entitlement"}
    ladder(s, M + 30, zy + 146, "dense", [names["r"], names["e"], names["a"]], 2, "green", w=410)
    ladder(s, W - M - 440, zy + 146, "cross-encoder", [names["a"], names["e"], names["r"]], 0, "green", w=410)
    s.arrow(M + 452, zy + 206, W - M - 452, zy + 206, INK, sw=2.6, head=12)

    # the measured, pre-registered result
    zy = 980
    s.box(M, zy, W - 2 * M, 258, "yellow", "fill", r=22, sw=3)
    s.icon("scale", M + 40, zy + 44, 30, TEXT["orange"], sw=2.4)
    s.text(M + 70, zy + 54, "RULES COMMITTED BEFORE THE TEST SPLIT", 23, 800, INK, max_width=W - 2 * M - 100)
    s.text(W - M - 30, zy + 96, "reranked vs hybrid alone", 17, 700, TEXT["muted"], anchor="end", max_width=300)
    rows = [("dev questions ranked first", "15 vs 13"), ("test questions ranked first", "13 vs 14"),
            ("reranking for later days", "not adopted")]
    for i, (label, value) in enumerate(rows):
        y = zy + 126 + i * 34
        s.text(M + 30, y, label, 21, 700, INK, max_width=520)
        s.text(W - M - 30, y, value, 21, 800, TEXT["red"] if i == 2 else INK, anchor="end", max_width=300)
    s.text(M + 30, zy + 228, "one test question decided it; the criterion was written down first", 18, 700,
           TEXT["teal"], max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 2
def rank_move(s: Slide, right: float, y: float, before: str, after: str, good: bool) -> None:
    """Two rank chips with a drawn arrow between them, right-aligned at `right`."""
    color = "green" if good else "red"
    w2 = s.chip(right, y - 20, after, color, size=19, anchor="end")
    x_after = right - w2
    s.arrow(x_after - 52, y - 4, x_after - 10, y - 4, INK, sw=2.4, head=10)
    s.chip(x_after - 60, y - 20, before, "gray", size=19, anchor="end")
    s.icon("check" if good else "x", right + 22, y - 4, 18, COLORS[color][0], sw=3.2)


def slide_moves() -> Slide:
    s = Slide(2, TOTAL, DAY, "What moved, and why", seed=92)
    heading(s, "What moved, and why", "Every rank the reranker changed, in three groups.")
    cards = [
        ("check", "green", "IT READS QUESTION AND CHUNK TOGETHER", [
            ("“How long is parental leave?”", "it meets “16 weeks”", "#2", "#1", True),
            ("RRF's 10 demoted lookups (“USD 220”)", "the chunk holding the value goes first", "#2-3", "#1", True),
            ("“How much time off do new dads get?”", "reranking RRF's top 20", "#15", "#1", True)]),
        ("layers", "orange", "IT SCORES EACH CHUNK ALONE", [
            ("“How soon must I report a lost laptop?”", "“How to report” never says laptop · RRF, top 20", "#2", "#5", False)]),
        ("calculator", "red", "IT DOESN'T REASON", [
            ("“Receipt for a USD 40 dinner?”", "the rule: “over USD 25”", "#1", "#2", False),
            ("“Wine with a business dinner?”", "the chunk says “Alcohol” · RRF, top 20", "#3", "#4", False)]),
    ]
    y = 274
    for icon, color, title, rows in cards:
        h = 116 + 78 * len(rows)
        s.box(M, y, W - 2 * M, h, color, "wash", r=22, sw=2.8)
        s.node(M + 60, y + 58, icon, color, None, size=72, icon_size=36)
        s.text(M + 114, y + 66, title, 22, 800, tcolor(color), max_width=W - 2 * M - 140)
        for i, (question, why, before, after, good) in enumerate(rows):
            ry = y + 140 + i * 78
            s.text(M + 36, ry, question, 20, 800, INK, max_width=W - 2 * M - 330)
            s.text(M + 36, ry + 30, why, 18, 600, TEXT["muted"], max_width=W - 2 * M - 330)
            rank_move(s, W - M - 64, ry, before, after, good)
        y += h + 22
    s.box(M, y, W - 2 * M, 1238 - y, "gray", "wash", r=18, sw=2.2, dashed=True)
    s.text(M + 30, y + 44, "the last two need more than a reranker: context from another", 19, 700, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, y + 74, "section, and a reader that knows 40 > 25 and wine is alcohol", 19, 700, INK,
           max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 3
def slide_strip(points: dict[str, list[tuple[str, float, bool]]]) -> Slide:
    s = Slide(3, TOTAL, DAY, "Close isn't answers", seed=93)
    heading(s, "Close isn't answers", "Each question's best reranker score, and a gate set on dev.")

    zy = 274
    s.box(M, zy, W - 2 * M, 640, "violet", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "scale", "violet", "BEST RERANKER SCORE PER QUESTION")
    x_left, x_right, lo, hi = M + 130, W - M - 40, -12.0, 10.0

    def px(v: float) -> float:
        return x_left + (v - lo) / (hi - lo) * (x_right - x_left)

    rows = {"dev": zy + 170, "test": zy + 410}
    for split, cy in rows.items():
        s.text(M + 30, cy + 8, split, 24, 800, INK, max_width=90)
        s.line(x_left, cy, x_right, cy, COLORS["gray"][0], sw=1.6, bow=0)
        placed: list[tuple[float, float]] = []
        for qid, score, answerable in sorted(points[split], key=lambda p: p[1]):
            x, level = px(score), 0
            while any(abs(x - ox) < 20 and oy == level for ox, oy in placed):
                level = -level + (1 if level <= 0 else 0)          # 0, 1, -1, 2, -2, ...
            placed.append((x, level))
            y = cy - level * 21
            five = qid in FIVE
            if answerable:
                s.ellipse(x, y, 9, 9, "blue", "fill", sw=2.2, jitter=0.4)
            else:
                s.ellipse(x, y, 9, 9, "red" if five else "orange", PAPER_WHITE, sw=3.0 if five else 2.4, jitter=0.4)
                if five:
                    s.icon("x", x, y, 11, COLORS["red"][0], sw=3.4)
    # the gate
    gx = px(GATE)
    s.line(gx, zy + 96, gx, zy + 520, INK, sw=2.4, dashed=True, bow=0)
    s.text(gx - 10, zy + 116, "gate −3.6", 19, 800, INK, anchor="end", max_width=160)
    s.text(gx - 10, zy + 140, "refused ←", 17, 700, TEXT["muted"], anchor="end", max_width=160)
    s.text(gx + 10, zy + 140, "→ answered", 17, 700, TEXT["muted"], max_width=160)
    # axis
    ay = zy + 548
    s.line(x_left, ay, x_right, ay, INK, sw=2, bow=0.1)
    for v in (-10, -5, 0, 5, 10):
        s.line(px(v), ay, px(v), ay + 8, INK, sw=2, bow=0)
        s.text(px(v), ay + 32, f"{v:+d}" if v else "0", 17, 700, TEXT["muted"], anchor="middle", max_width=50)
    # annotations for the test row
    test_y = rows["test"]
    s.text(px(3.15), test_y - 74, "home internet +3.15", 17, 800, TEXT["red"], anchor="middle", max_width=210)
    s.line(px(3.15), test_y - 64, px(3.15), test_y - 14, TEXT["red"], sw=1.8, bow=0)
    s.text(px(-7.38), test_y + 64, "receipt −7.38", 17, 800, TEXT["blue"], anchor="middle", max_width=180)
    dev_y = rows["dev"]
    s.text(px(-3.56) + 14, dev_y + 64, "new dads −3.56 sets the gate", 17, 800, TEXT["blue"], max_width=300)
    s.line(px(-3.56) + 4, dev_y + 12, px(-3.56) + 30, dev_y + 44, TEXT["blue"], sw=1.8, bow=0)
    s.line(px(-7.38), test_y + 12, px(-7.38), test_y + 44, TEXT["blue"], sw=1.8, bow=0)
    # legend
    ly = zy + 612
    s.ellipse(M + 44, ly - 6, 9, 9, "blue", "fill", sw=2.2, jitter=0.4)
    s.text(M + 62, ly, "answerable", 18, 700, INK, max_width=150)
    s.ellipse(M + 234, ly - 6, 9, 9, "orange", PAPER_WHITE, sw=2.4, jitter=0.4)
    s.text(M + 252, ly, "unanswerable", 18, 700, INK, max_width=170)
    s.ellipse(M + 444, ly - 6, 9, 9, "red", PAPER_WHITE, sw=3.0, jitter=0.4)
    s.icon("x", M + 444, ly - 6, 11, COLORS["red"][0], sw=3.4)
    s.text(M + 462, ly, "the five “similar but wrong”", 18, 800, TEXT["red"], max_width=360)

    zy = 940
    s.box(M, zy, W - 2 * M, 298, "yellow", "fill", r=22, sw=3)
    s.icon("triangle-alert", M + 40, zy + 44, 30, TEXT["orange"], sw=2.4)
    s.text(M + 70, zy + 54, "THE PRE-REGISTERED TEST, ON THE TEST SPLIT", 23, 800, INK, max_width=W - 2 * M - 100)
    lines = [("the five: 0 refused", 800, TEXT["red"]),
             ("answerable: 1 refused (the receipt question)", 700, INK),
             ("a web-search reranker judges “about”, not “answers”", 700, INK),
             ("not adopted as a gate; answers are checked on Day 12", 700, TEXT["teal"])]
    for i, (line, weight, color) in enumerate(lines):
        s.text(M + 30, zy + 108 + i * 44, line, 21, weight, color, max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 4
def slide_checks(medians: dict[str, float]) -> Slide:
    s = Slide(4, TOTAL, DAY, "Before you add a reranker", seed=94)
    heading(s, "Before you add a reranker", "Three checks and the decision, from today's measurements.")
    zy = 274
    # 1. cost grows with depth
    s.box(M, zy, W - 2 * M, 214, "teal", "wash", r=22, sw=2.8)
    s.node(M + 64, zy + 66, "timer", "teal", None, size=80, icon_size=40)
    s.text(M + 124, zy + 74, "EVERY CANDIDATE IS A MODEL RUN", 22, 800, TEXT["teal"], max_width=W - 2 * M - 150)
    labels = {"dense (embed the question + search)": "embed the question",
              "rerank 3 candidates": "rerank 3", "rerank 20 candidates": "rerank 20"}
    scale = 470 / medians["rerank 20 candidates"]
    for i, step in enumerate(TIMING_STEPS):
        y = zy + 118 + i * 30
        s.text(M + 40, y + 6, labels[step], 18, 700, INK, max_width=210)
        s.box(M + 250, y - 10, max(medians[step] * scale, 6), 22, "teal", "fill", r=6, sw=1.8, jitter=0.4)
        s.text(M + 262 + medians[step] * scale, y + 6, f"~{round(medians[step], -1):.0f} ms", 17, 800, INK,
               max_width=120)
    # 2. selection
    zy = 512
    s.box(M, zy, W - 2 * M, 214, "violet", "wash", r=22, sw=2.8)
    s.node(M + 64, zy + 66, "file-clock", "violet", None, size=80, icon_size=40)
    s.text(M + 124, zy + 74, "SELECTING THE CONTEXT", 22, 800, TEXT["violet"], max_width=W - 2 * M - 150)
    sel = [("check", "green", "a copy under another title: dropped"),
           ("x", "red", "an old version, USD 400 not 500: kept (dedupe isn't freshness)"),
           ("x", "red", "one chunk per document: 3 answers lost")]
    for i, (icon, color, line) in enumerate(sel):
        y = zy + 124 + i * 30
        s.icon(icon, M + 54, y - 6, 18, COLORS[color][0], sw=3.2)
        s.text(M + 76, y, line, 19, 700, INK, max_width=W - 2 * M - 110)
    # 3. leaks are promoted
    zy = 750
    s.box(M, zy, W - 2 * M, 214, "red", "wash", r=22, sw=2.8)
    s.node(M + 64, zy + 66, "lock-keyhole", "red", None, size=80, icon_size=40)
    s.text(M + 124, zy + 74, "IT PROMOTES WHAT IT'S GIVEN", 22, 800, TEXT["red"], max_width=W - 2 * M - 150)
    s.text(M + 40, zy + 130, "one unfiltered retriever: a managers-only", 19, 700, INK, max_width=470)
    s.text(M + 40, zy + 160, "chunk in an employee's pay question", 19, 700, INK, max_width=470)
    s.text(M + 40, zy + 192, "filter every retriever, then rerank", 19, 800, TEXT["red"], max_width=470)
    for k, (label, rank) in enumerate((("RRF", 4), ("reranked", 1))):
        lx = W - M - 400 + k * 206
        s.text(lx, zy + 52, label, 17, 800, TEXT["muted"], max_width=150)
        for r in range(4):
            top = zy + 66 + r * 32
            leak = r + 1 == rank
            s.box(lx, top, 160, 26, "red" if leak else "gray", "fill" if leak else PAPER_WHITE, r=6, sw=1.8,
                  jitter=0.3)
            s.text(lx + 10, top + 19, f"#{r + 1}", 16, 800 if leak else 600, INK if leak else TEXT["muted"],
                   max_width=40)
            if leak:
                s.text(lx + 44, top + 19, "managers", 16, 800, INK, max_width=92)
                s.icon("lock-keyhole", lx + 146, top + 13, 14, COLORS["red"][0], sw=2.6)
        if k == 0:
            s.arrow(lx + 168, zy + 126, lx + 198, zy + 126, INK, sw=2.4, head=10)

    # the decision
    zy = 988
    s.box(M, zy, W - 2 * M, 250, "yellow", "fill", r=22, sw=3)
    s.icon("scale", M + 40, zy + 44, 30, TEXT["orange"], sw=2.4)
    s.text(M + 70, zy + 54, "DECISIONS, CRITERIA SET IN ADVANCE", 23, 800, INK, max_width=W - 2 * M - 100)
    lines = [("rerank the hybrid's top 3: 13 vs 14 first on test", 700, INK),
             ("a gate on its score: 1 answerable refused, 0 of the 5", 700, INK),
             ("both: not adopted; the weighted fusion stays", 800, TEXT["red"]),
             ("next: Day 10, query transformations", 800, TEXT["orange"])]
    for i, (line, weight, color) in enumerate(lines):
        s.text(M + 30, zy + 104 + i * 38, line, 21, weight, color, max_width=W - 2 * M - 60)
    return s


def main() -> None:
    check_contrast()
    medians = check_facts()
    points = strip_points()
    OUT.mkdir(parents=True, exist_ok=True)
    slides = [("slide-01-cover", slide_cover()), ("slide-02-what-moved", slide_moves()),
              ("slide-03-close-isnt-answers", slide_strip(points)), ("slide-04-before-you-add-one", slide_checks(medians))]
    for name, slide in slides:
        path = OUT / f"{name}.svg"
        path.write_text(slide.svg(), encoding="utf-8")
        print("wrote", path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
