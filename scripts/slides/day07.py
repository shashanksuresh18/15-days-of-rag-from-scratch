"""Build the Day 07 LinkedIn carousel (whiteboard style) as editable SVGs in assets/day-07/.

    python scripts/slides/day07.py                    # writes the SVGs
    python scripts/slides/render.py assets/day-07     # renders PNGs + PDF

Every number and ID on the slides comes from the executed Day 07 notebook or the files it reads.
`check_facts()` confirms each quoted line appears in the notebook's saved outputs before anything is
written. The dot rows (which question each retriever finds) and the score plot are read from the
committed baseline (evals/baselines/day-07-retrieval.json) and the labelled questions, and checked
against the counts in the notebook's outputs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from kit import COLORS, INK, M, TEXT, W, Slide, check_contrast

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "day-07"
NOTEBOOK = ROOT / "notebooks" / "day_07_evaluation_baseline.ipynb"
BASELINE = ROOT / "evals" / "baselines" / "day-07-retrieval.json"
QUESTIONS = ROOT / "evals" / "questions-v1.jsonl"
TOTAL = 4
DAY = "07"
PAPER_WHITE = "#FFFFFF"

FACTS = [
    "81 labelled queries in evals/questions-v1.jsonl",
    "questions, answerable          16    15     31",
    "questions, not answerable       8    12     20",
    "lookups                        30     0     30",
    "questions, BM25    31     24     29     29     30      0.94    0.85   94% (79% to 98%)",
    "questions, dense   31     26     29     31     31      0.94    0.90   94% (79% to 98%)",
    "lookups, BM25      30     30     30     30     30      1.00    1.00   100% (89% to 100%)",
    "lookups, dense     30     12     19     26     29      0.62    0.56   63% (46% to 78%)",
    "relevant chunk in the top 3: both 28 | only BM25 1 | only dense 1 | neither 1",
    "relevant chunk in the top 3: both 19 | only BM25 11 | only dense 0 | neither 0",
    "exact value: 11 | vocabulary: 2 | other section: 1 | broader chunk: 1",
    "d06-02 | alice | question | dev | day06-check",
    "  evidence:          \"take effect from 1 April\"",
    "  restricted_chunks: manager-pay-review-guide#overview",
    "documents now: 44 chunks, snapshot 3f8344d00b8b...",
    "the same labels after two document changes: 17 problems",
    "  #6  2.21  expense-reimbursement-policy#meals   <- the answer   matched: business",
    "  #4  0.60  expense-reimbursement-policy#meals   <- the answer",
    "Meals during business travel are reimbursed up to USD 75 per day. Alcohol is not reimbursed.",
    "0.50                16 of 16                  1 of 8                15 of 15                 0 of 12  <- Day 06",
    "0.62                16 of 16                  6 of 8                15 of 15                 6 of 12  <- measured",
    "weakest answerable dev question: d03-11 \"What training must new starters complete?\" at 0.626",
    "dev: refuses 4 of 8 unanswerable questions (Google, France, CEO, Wi-Fi) and 0 of 16 answerable ones",
    "test: refuses 0 of 12 unanswerable questions (none) and 2 of 15 answerable ones",
    "no source mentions Tokyo",
    "no source mentions Should",
    "unanswerable test questions still answered with the measured settings: 5",
    "compared with evals/baselines/day-07-retrieval.json: no changes",
    "All Day 07 sanity checks passed.",
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
    # Slide 2's counts: the two-edit demo breaks 2 labels and flips 4 of alice's to "restricted"
    counted = {"no chunk holds its evidence any more": 2, "restricted [], now ['data-retention-policy": 4}
    for line, n in counted.items():
        if outputs.count(line) != n:
            raise SystemExit(f"expected {n} lines containing {line!r} in the notebook outputs")


def load() -> tuple[dict, list[dict]]:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]
    return baseline, rows


def found(baseline: dict, qid: str, name: str) -> bool:
    rank = baseline["queries"][qid][name]["first_relevant"]
    return rank is not None and rank <= baseline["top_k"]


def dot_rows(baseline: dict, rows: list[dict], kind: str) -> list[tuple[str, bool, bool]]:
    # (id, BM25 found, dense found), ordered: both, only BM25, only dense, neither
    picked = [(r["id"], found(baseline, r["id"], "BM25"), found(baseline, r["id"], "dense"))
              for r in rows if r["answer_chunks"] and r["kind"] == kind]
    order = {(True, True): 0, (True, False): 1, (False, True): 2, (False, False): 3}
    picked.sort(key=lambda item: order[item[1], item[2]])
    counts = Counter(order[b, d] for _, b, d in picked)
    expected = {"question": [28, 1, 1, 1], "lookup": [19, 11, 0, 0]}[kind]
    if [counts[i] for i in range(4)] != expected:
        raise SystemExit(f"{kind}: baseline gives {counts}, the notebook printed {expected}")
    return picked


def score_points(baseline: dict, rows: list[dict]) -> list[tuple[str, bool, float]]:
    # (split, answerable, best dense retrieval score) for the 51 natural questions
    points = [(r["split"], bool(r["answer_chunks"]), baseline["queries"][r["id"]]["dense"]["top_score"])
              for r in rows if r["kind"] == "question"]
    if Counter((s, a) for s, a, _ in points) != {("dev", True): 16, ("dev", False): 8, ("test", True): 15,
                                                  ("test", False): 12}:
        raise SystemExit("unexpected question counts")
    return points


def heading(s: Slide, title: str, subtitle: str, size: int = 54) -> None:
    s.text(M, 190, title, size, 800, INK)
    s.text(M, 238, subtitle, 25, 500, TEXT["muted"])


def tcolor(color: str) -> str:
    return TEXT.get(color, INK) if color != "yellow" else INK


def dot(s: Slide, cx: float, cy: float, ok: bool, r: float = 11) -> None:
    color = "green" if ok else "red"
    s.ellipse(cx, cy, r, r, color, "fill", sw=2.0, jitter=0.5)
    s.icon("check" if ok else "x", cx, cy, r * 1.15, COLORS[color][0], sw=3.4)


# ----------------------------------------------------------------------------- slide 1
def slide_cover(questions: list[tuple[str, bool, bool]], lookups: list[tuple[str, bool, bool]]) -> Slide:
    s = Slide(1, TOTAL, DAY, "Measure before you improve", seed=71)
    heading(s, "Measure before you improve", "Day 07: 81 labelled questions, two retrievers, every miss explained.")

    # questions: one dot per question, the same question in the same column for both retrievers
    zy = 274
    s.box(M, zy, W - 2 * M, 420, "blue", "wash", r=22, dashed=True, sw=3)
    s.icon("search", M + 38, zy + 40, 28, COLORS["blue"][0], sw=2.4)
    s.text(M + 64, zy + 49, "31 QUESTIONS: IS THE ANSWER IN THE TOP 3?", 22, 800, TEXT["blue"])
    x0, step = M + 182, 25.2
    for r, (name, index) in enumerate([("BM25", 1), ("dense", 2)]):
        y = zy + 116 + r * 66
        s.text(M + 28, y + 8, name, 24, 800, INK, max_width=100)
        hits = sum(item[index] for item in questions)
        for j, item in enumerate(questions):
            dot(s, x0 + j * step, y, item[index], r=10)
        s.text(M + 28, y + 34, f"{hits} of 31", 17, 700, TEXT["muted"], max_width=120)
    # frame the last three columns (the questions that tell the two apart) and key them with badges
    fx = x0 + 28 * step - 13
    s.box(fx, zy + 90, 2 * step + 26, 136, "gray", None, r=12, sw=2.4, dashed=True, jitter=0.6)
    for i in range(3):
        s.badge(x0 + (28 + i) * step, zy + 252, i + 1, "yellow", r=12, size=15)
    labels = [("lost laptop", "only BM25 finds it"), ("new dads", "only dense finds it"),
              ("wine (alcohol)", "both miss it")]
    for i, (q, who) in enumerate(labels):
        y = zy + 288 + i * 32
        s.badge(M + 470, y - 7, i + 1, "yellow", r=12, size=15)
        s.text(M + 492, y, f"{q}: {who}", 20, 700, INK, max_width=W - 2 * M - 520)
    s.text(M + 28, zy + 300, "same total,", 24, 800, TEXT["blue"], max_width=380)
    s.text(M + 28, zy + 336, "different misses", 24, 800, TEXT["blue"], max_width=380)
    s.text(M + 28, zy + 398, "30 of 31 found by at least one: combine them? → Day 08", 20, 700, TEXT["teal"],
           max_width=W - 2 * M - 56)

    # lookups
    zy = 724
    s.box(M, zy, W - 2 * M, 234, "orange", "wash", r=22, dashed=True, sw=3)
    s.icon("hash", M + 38, zy + 40, 28, COLORS["orange"][0], sw=2.4)
    s.text(M + 64, zy + 49, "30 LOOKUPS: A VALUE TYPED AS THE QUERY", 22, 800, TEXT["orange"])
    s.text(W - M - 28, zy + 49, "USD 220, 15 minutes…", 19, 700, TEXT["muted"], anchor="end", max_width=260)
    for r, (name, index) in enumerate([("BM25", 1), ("dense", 2)]):
        y = zy + 112 + r * 62
        s.text(M + 28, y + 8, name, 24, 800, INK, max_width=100)
        hits = sum(item[index] for item in lookups)
        for j, item in enumerate(lookups):
            dot(s, x0 + j * step, y, item[index], r=10)
        s.text(M + 28, y + 34, f"{hits} of 30", 17, 700, TEXT["muted"], max_width=120)

    # the misses, by reason
    zy = 988
    s.box(M, zy, W - 2 * M, 250, "violet", "wash", r=22, dashed=True, sw=3)
    s.icon("list-ordered", M + 38, zy + 40, 28, COLORS["violet"][0], sw=2.4)
    s.text(M + 64, zy + 49, "15 MISSES, EACH WITH A WRITTEN REASON", 22, 800, TEXT["violet"])
    reasons = [("exact value", 11, "orange", "dense"), ("vocabulary", 2, "blue", "BM25"),
               ("other section", 1, "teal", "dense"), ("broader chunk", 1, "violet", "dense")]
    for i, (label, n, color, who) in enumerate(reasons):
        x = M + 30 + (i % 2) * 470
        y = zy + 110 + (i // 2) * 64
        s.tag(x, y, f"{n}", color, size=24)
        s.text(x + 66, y, label, 23, 800, INK, max_width=250)
        s.text(x + 66 + 200, y, who, 20, 700, TEXT["muted"], max_width=120)
    s.text(M + 30, zy + 226, "each points at a fix a later day will test, on these same questions", 20, 700,
           TEXT["teal"], max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 2
def slide_labels() -> Slide:
    s = Slide(2, TOTAL, DAY, "A label is evidence, tied to a snapshot", seed=72)
    heading(s, "A label is evidence", "The right answer depends on the documents, and on who asks.")

    # one question, two users
    zy = 274
    s.box(M, zy, W - 2 * M, 412, "blue", "wash", r=22, sw=2.8)
    s.text(M + 30, zy + 50, "“When do pay rises take effect?”", 27, 800, INK, max_width=W - 2 * M - 60)
    s.icon("file-text", M + 44, zy + 100, 26, COLORS["violet"][0], sw=2.3)
    s.text(M + 70, zy + 108, "evidence", 20, 700, TEXT["muted"], max_width=120)
    s.chip(M + 168, zy + 84, "take effect from 1 April", "violet", size=20)
    s.arrow(M + 300, zy + 132, M + 300, zy + 172, INK, sw=2.6, head=12)
    s.icon("lock-keyhole", M + 44, zy + 206, 26, COLORS["red"][0], sw=2.3)
    s.text(M + 70, zy + 214, "only in", 20, 700, TEXT["muted"], max_width=120)
    s.chip(M + 168, zy + 190, "manager-pay-review-guide#overview", "red", size=19)
    s.text(W - M - 28, zy + 214, "managers only", 19, 700, TEXT["red"], anchor="end", max_width=200)
    lanes = [("user", "maria (manager)", "label: that chunk", "answerable", "green"),
             ("user-x", "alice (employee)", "label: none she may read", "correct: “I don't know”", "orange")]
    for k, (icon, who, label, verdict, color) in enumerate(lanes):
        x = M + 28 + k * 480
        y = zy + 254
        s.box(x, y, 444, 132, color, "fill" if False else "wash", r=18, sw=2.6)
        s.node(x + 52, y + 66, icon, color, None, size=76, icon_size=40)
        s.text(x + 104, y + 44, who, 22, 800, INK, max_width=330)
        s.text(x + 104, y + 78, label, 19, 700, INK, max_width=330)
        s.text(x + 104, y + 110, verdict, 19, 800, tcolor(color), max_width=330)

    # snapshot
    zy = 716
    s.box(M, zy, W - 2 * M, 250, "orange", "wash", r=22, dashed=True, sw=3)
    s.icon("fingerprint-pattern", M + 38, zy + 40, 28, COLORS["orange"][0], sw=2.4)
    s.text(M + 64, zy + 49, "LABELS BELONG TO A SNAPSHOT", 22, 800, TEXT["orange"])
    s.text(W - M - 28, zy + 49, "44 chunks: text hash + who may read", 19, 700, TEXT["muted"], anchor="end",
           max_width=380)
    edits = [("file-pen-line", "password rule: 14 → 16 characters", "2 labels break"),
             ("lock-keyhole", "retention policy → managers only", "4 of alice's labels flip")]
    for i, (icon, edit, effect) in enumerate(edits):
        y = zy + 106 + i * 50
        s.icon(icon, M + 46, y - 8, 26, COLORS["orange"][0], sw=2.3)
        s.text(M + 72, y, edit, 21, 700, INK, max_width=460)
        s.text(W - M - 28, y, effect, 21, 800, TEXT["red"], anchor="end", max_width=330)
    s.icon("triangle-alert", M + 46, zy + 206, 26, COLORS["red"][0], sw=2.4)
    s.text(M + 72, zy + 214, "17 label problems → the notebook stops before measuring", 21, 800, TEXT["red"],
           max_width=W - 2 * M - 100)

    # dev / test
    zy = 996
    cw = (W - 2 * M - 20) / 2
    splits = [("pencil-ruler", "gray", "DEV · 54", ["Days 01–06 questions", "seen while building:", "tune on these"],
               "16 answerable · 8 not · 30 lookups"),
              ("eye-off", "green", "TEST · 27", ["written today, before", "any retriever saw them:",
                                                  "report only, never tune"], "15 answerable · 12 not")]
    for k, (icon, color, title, lines, counts) in enumerate(splits):
        x = M + k * (cw + 20)
        s.box(x, zy, cw, 242, color, "wash", r=20, sw=2.8)
        s.icon(icon if icon != "pencil-ruler" else "ruler", x + 40, zy + 42, 28,
               COLORS[color][0] if color != "gray" else TEXT["gray"], sw=2.4)
        s.text(x + 66, zy + 51, title, 24, 800, tcolor(color), max_width=cw - 90)
        for i, line in enumerate(lines):
            s.text(x + 26, zy + 104 + i * 38, line, 21, 800 if i == 2 else 600, INK, max_width=cw - 52)
        s.text(x + 26, zy + 216, counts, 18, 700, TEXT["muted"], max_width=cw - 52)
    return s


# ----------------------------------------------------------------------------- slide 3
def slide_errors() -> Slide:
    s = Slide(3, TOTAL, DAY, "Every miss gets a reason", seed=73)
    heading(s, "Every miss gets a reason", "Counts first, then the questions behind them.")

    # metrics
    zy = 274
    s.box(M, zy, W - 2 * M, 286, "blue", "wash", r=22, dashed=True, sw=3)
    s.text(M + 30, zy + 48, "ANSWER IN THE TOP 3", 22, 800, TEXT["blue"])
    s.text(W - M - 28, zy + 48, "MRR@10", 22, 800, TEXT["blue"], anchor="end", max_width=160)
    rows = [("questions", "BM25", 29, 31, "0.85"), ("questions", "dense", 29, 31, "0.90"),
            ("lookups", "BM25", 30, 30, "1.00"), ("lookups", "dense", 19, 30, "0.56")]
    bar_x, bar_w = M + 290, 470
    for i, (kind, name, hits, n, mrr) in enumerate(rows):
        y = zy + 100 + i * 42
        s.text(M + 30, y, f"{kind}, {name}", 21, 700, INK, max_width=240)
        color = "blue" if kind == "questions" else "orange"
        s.box(bar_x, y - 22, bar_w * hits / n, 28, color, "fill", r=8, sw=2, jitter=0.6)
        s.text(bar_x + bar_w * hits / n + 12, y, f"{hits}/{n}", 21, 800, INK, max_width=90)
        s.text(W - M - 28, y, mrr, 21, 800, INK, anchor="end", max_width=100)
    s.text(M + 30, zy + 270, "94% on 31 questions = 79% to 98%: read counts, not decimals", 20, 700, TEXT["teal"],
           max_width=W - 2 * M - 60)

    # categories → a later day
    zy = 590
    s.text(M, zy + 20, "15 MISSES IN THE TOP 3, BY REASON", 22, 800, INK)
    cats = [("hash", "orange", "exact value", 11, "dense · lookups", "Day 08 hybrid"),
            ("whole-word", "blue", "vocabulary", 2, "BM25 · new dads, wine", "Day 08 / 10"),
            ("files", "teal", "other section", 1, "dense · lost laptop", "Day 11"),
            ("layers", "violet", "broader chunk", 1, "dense · wine", "Day 09 rerank")]
    for i, (icon, color, label, n, where, day) in enumerate(cats):
        y = zy + 60 + i * 92
        s.box(M, y, W - 2 * M, 78, color, "wash", r=16, sw=2.4)
        s.icon(icon, M + 40, y + 39, 28, COLORS[color][0], sw=2.4)
        s.text(M + 72, y + 34, label, 23, 800, tcolor(color), max_width=230)
        s.text(M + 72, y + 62, where, 18, 600, TEXT["muted"], max_width=260)
        for j in range(n):
            s.ellipse(M + 352 + j * 24, y + 39, 9, 9, color, "fill", sw=2, jitter=0.5)
        s.tag(W - M - 22, y + 48, day, color, size=19, anchor="end")
    s.text(M, zy + 440, "the tags say where a fix will be TESTED, not promised", 19, 700, TEXT["muted"],
           max_width=W - 2 * M)

    # the wine question
    zy = 1064
    s.box(M, zy, W - 2 * M, 174, "gray", PAPER_WHITE, r=18, sw=2.4)
    s.text(M + 28, zy + 42, "“Will Acme pay for wine with a business dinner?”", 22, 800, INK,
           max_width=W - 2 * M - 56)
    s.text(M + 28, zy + 80, "the answer: “Alcohol is not reimbursed.”", 21, 700, TEXT["violet"],
           max_width=W - 2 * M - 56)
    s.text(M + 28, zy + 118, "BM25 ranks it #6 (only “business” matches) · dense #4", 20, 700, INK,
           max_width=W - 2 * M - 56)
    s.text(M + 28, zy + 152, "a person sees wine = alcohol; neither retriever does", 20, 700, TEXT["muted"],
           max_width=W - 2 * M - 56)
    return s


# ----------------------------------------------------------------------------- slide 4
def slide_thresholds(points: list[tuple[str, bool, float]]) -> Slide:
    s = Slide(4, TOTAL, DAY, "Thresholds, measured", seed=74)
    heading(s, "Thresholds, measured", "Day 06 chose them by eye. A rule on dev, reported on test.")

    # strip plot of the best retrieval score
    zy = 274
    s.box(M, zy, W - 2 * M, 490, "gray", "wash", r=22, dashed=True, sw=3)
    s.text(M + 30, zy + 48, "BEST RETRIEVAL SCORE OF EACH QUESTION", 22, 800, INK)
    lo, hi = 0.40, 0.90
    x_left, x_right = M + 120, W - M - 40

    def px(v: float) -> float:
        return x_left + (v - lo) / (hi - lo) * (x_right - x_left)

    rows = {"dev": zy + 168, "test": zy + 310}
    for split, base in rows.items():
        s.text(M + 30, base + 8, split, 23, 800, INK, max_width=80)
        placed: list[tuple[float, float]] = []
        for _, answerable, score in sorted((p for p in points if p[0] == split), key=lambda p: (not p[1], p[2])):
            x = px(score)
            y = base
            k = 0
            while any(abs(x - qx) < 17 and abs(y - qy) < 17 for qx, qy in placed):   # stack like a beeswarm
                k += 1
                y = base + (k + 1) // 2 * 19 * (1 if k % 2 else -1)
            placed.append((x, y))
            if answerable:
                s.ellipse(x, y, 8.5, 8.5, "blue", "fill", sw=2.0, jitter=0.4)
            else:
                s.ellipse(x, y, 8.5, 8.5, "orange", PAPER_WHITE, sw=2.6, jitter=0.4)
    axis_y = zy + 386
    s.line(x_left, axis_y, x_right, axis_y, INK, sw=2.2, bow=0.1)
    for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        s.line(px(v), axis_y, px(v), axis_y + 8, INK, sw=2, bow=0)
        s.text(px(v), axis_y + 32, f"{v:.1f}", 18, 700, TEXT["muted"], anchor="middle", max_width=60)
    for value, color, label, dy in [(0.50, "gray", "Day 06: 0.50", 0), (0.62, "green", "measured: 0.62", 0)]:
        s.line(px(value), zy + 92, px(value), axis_y, COLORS[color][0], sw=3, dashed=True, bow=0.1)
        s.text(px(value), zy + 84, label, 19, 800, TEXT[color], anchor="middle", max_width=200)
    s.ellipse(M + 46, zy + 458, 8.5, 8.5, "blue", "fill", sw=2.0, jitter=0.4)
    s.text(M + 64, zy + 465, "answerable", 19, 700, INK, max_width=150)
    s.ellipse(M + 236, zy + 458, 8.5, 8.5, "orange", PAPER_WHITE, sw=2.6, jitter=0.4)
    s.text(M + 254, zy + 465, "not answerable", 19, 700, INK, max_width=200)
    s.text(W - M - 28, zy + 465, "they overlap: a score isn't a label", 19, 800, TEXT["orange"], anchor="end",
           max_width=380)

    # the gate on the test split
    zy = 790
    cw = (W - 2 * M - 20) / 2
    cards = [("gray", "gate 0.50 (Day 06)", "refuses 0 of 12", "unanswerable test questions"),
             ("green", "gate 0.62 (measured)", "refuses 6 of 12", "and keeps all 15 answerable")]
    for k, (color, title, big, small) in enumerate(cards):
        x = M + k * (cw + 20)
        s.box(x, zy, cw, 160, color, "wash", r=20, sw=2.8)
        s.icon("gauge", x + 38, zy + 42, 28, COLORS[color][0] if color != "gray" else TEXT["gray"], sw=2.4)
        s.text(x + 64, zy + 50, title, 21, 800, tcolor(color), max_width=cw - 84)
        s.text(x + 26, zy + 98, big, 27, 800, INK, max_width=cw - 52)
        s.text(x + 26, zy + 136, small, 19, 700, TEXT["muted"], max_width=cw - 52)
    s.text(M, zy + 194, "rule, written first: highest gate that loses no answerable dev question", 19, 700,
           TEXT["muted"], max_width=W - 2 * M)

    # the names rule
    zy = 1018
    s.box(M, zy, W - 2 * M, 220, "red", "wash", r=22, dashed=True, sw=3)
    s.icon("circle-help", M + 38, zy + 40, 28, COLORS["red"][0], sw=2.4)
    s.text(M + 64, zy + 49, "THE NAMES RULE: PERFECT ON DEV, WRONG ON TEST", 21, 800, TEXT["red"],
           max_width=W - 2 * M - 90)
    s.text(M + 30, zy + 96, "dev: refused Google, France, CEO, Wi-Fi · no false refusals", 20, 700, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 136, "test: refused 2 answerable questions:", 20, 800, INK, max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 176, "“Tokyo” (the Other cities row applies) · “Should” (2nd sentence)", 20, 700,
           TEXT["red"], max_width=W - 2 * M - 60)
    return s


def main() -> None:
    check_contrast()
    check_facts()
    baseline, rows = load()
    questions, lookups = dot_rows(baseline, rows, "question"), dot_rows(baseline, rows, "lookup")
    points = score_points(baseline, rows)
    OUT.mkdir(parents=True, exist_ok=True)
    slides = [("slide-01-cover", slide_cover(questions, lookups)), ("slide-02-labels-and-snapshot", slide_labels()),
              ("slide-03-every-miss", slide_errors()), ("slide-04-thresholds", slide_thresholds(points))]
    for name, slide in slides:
        path = OUT / f"{name}.svg"
        path.write_text(slide.svg(), encoding="utf-8")
        print("wrote", path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
