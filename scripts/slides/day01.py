"""Build the Day 01 LinkedIn carousel (whiteboard style) as editable SVGs in assets/day-01/.

    python scripts/slides/day01.py                    # writes the SVGs
    python scripts/slides/render.py assets/day-01     # renders PNGs + PDF

Every number and chunk ID on the slides comes from the executed Day 01 notebook.
`check_facts()` confirms each one appears in the notebook's saved outputs before anything is
written, so the slides cannot silently drift from the code.
"""

from __future__ import annotations

import json
from pathlib import Path

from kit import COLORS, INK, M, TEXT, W, Slide, check_contrast, measure, wrap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "day-01"
NOTEBOOK = ROOT / "notebooks" / "day_01_rag_foundations.ipynb"
TOTAL = 4

QUESTION = "How much is the home office allowance for remote employees?"
TOKENS = ["much", "home", "office", "allowance", "remote", "employees"]
TOP3 = [("remote-work-policy#equipment-and-home-office", 13.43),
        ("remote-work-policy#security-while-remote", 4.92),
        ("remote-work-policy#core-hours", 4.29)]
MIN_SCORE = 4.0
ANSWER = ("Employees approved for remote work receive a one-time home office allowance "
          "of USD 500 for a desk, chair or monitor.")
CITATION = "remote-work-policy#equipment-and-home-office"
FACTS = [
    QUESTION, ANSWER, f"[{CITATION}]", "score 13.43", "score  4.92", "score  4.29", "MIN_SCORE 4.0",
    f"Query terms: {TOKENS}",
    "8 documents -> 34 chunks",
    "'allowance' appears in 1 of 34 chunks", "'vpn' appears in 6 of 34 chunks",
    "'acme' appears in 12 of 34 chunks",
    "much                 0                 0",           # 'much' is in no chunk -> adds 0
    "best score 3.15 is below MIN_SCORE 4.0",            # new dads -> abstains
    "Terms shared with the question: none",
    "best score 5.57 >= MIN_SCORE 4.0",                  # Google -> confident wrong answer
    "no indexed chunk shares a term with the question",  # France -> abstains
]


def check_facts() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    outputs = "".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code"
                      for o in c.get("outputs", []))
    missing = [f for f in FACTS if f not in outputs]
    if missing:
        raise SystemExit(f"Slide facts not found in notebook outputs (re-run the notebook?): {missing}")


def heading(s: Slide, title: str, subtitle: str) -> None:
    s.text(M, 190, title, 54, 800, INK)
    s.text(M, 238, subtitle, 25, 500, TEXT["muted"])


def step_label(s: Slide, x: float, y: float, n: int, label: str, color: str, size=24) -> None:
    """Numbered step label: a small badge followed by text; (x, y) is the badge centre."""
    s.badge(x, y, n, color, r=17, size=20)
    s.text(x + 26, y + size * 0.36, label, size, 700, TEXT[color] if color in TEXT else INK)


# ----------------------------------------------------------------------------- slide 1
def slide_cover() -> Slide:
    s = Slide(1, TOTAL, "01", "How RAG answers a question", seed=11)
    heading(s, "How RAG answers a question",
            "Retrieval-Augmented Generation: look it up first, then answer.")

    r1, r2 = 410, 770
    xu, xr, xd = 150, 540, 930
    s.node(xu, r1, "user", "yellow", "YOU")
    s.node(xr, r1, "search", "blue", "RETRIEVER")
    s.node(xd, r1, "files", "green", "YOUR DOCS")
    s.node(xu, r2, "message-square", "teal", "ANSWER")
    s.node(xr, r2, "brain", "violet", "LLM")
    s.text(xr, r2 + 136, "(Day 01 uses an API-free stand-in)", 20, 500, TEXT["muted"], anchor="middle")

    s.arrow(222, r1, 468, r1, INK)
    step_label(s, 262, r1 - 34, 1, "ask", "yellow")
    s.arrow(612, r1 - 22, 858, r1 - 22, COLORS["green"][0])
    step_label(s, 650, r1 - 56, 2, "search", "green")
    s.arrow(858, r1 + 22, 612, r1 + 22, COLORS["green"][0])
    step_label(s, 636, r1 + 56, 3, "top 3 chunks", "green")
    s.arrow(xr, 532, xr, 698, COLORS["blue"][0])
    step_label(s, xr + 30, 600, 4, "question", "blue")
    s.text(xr + 56, 634, "+ top 3 chunks", 24, 700, TEXT["blue"])
    s.arrow(468, r2, 222, r2, COLORS["violet"][0])
    step_label(s, 268, r2 - 34, 5, "generate", "violet")
    s.arrow(xu, 698, xu, 532, COLORS["teal"][0])
    step_label(s, xu + 30, 600, 6, "answer", "teal")
    s.text(xu + 56, 634, "+ source", 24, 700, TEXT["teal"])

    # Without RAG
    bx, by, bw, bh = 700, 680, 320, 200
    s.box(bx, by, bw, bh, "red", "wash", r=18, dashed=True)
    s.icon("circle-x", bx + 38, by + 40, 34, COLORS["red"][0], sw=2.4)
    s.text(bx + 66, by + 50, "Without RAG", 26, 800, TEXT["red"])
    s.paragraph(bx + 22, by + 92, "The LLM alone has never seen Acme’s policy. It can only guess.",
                23, 500, bw - 44, 31, INK, max_lines=4)

    # Real run strip
    zy = 940
    s.box(M, zy, W - 2 * M, 296, "yellow", "wash", r=22, dashed=True)
    s.tag(M + 24, zy + 50, "REAL RUN · Day 01 notebook", "yellow", size=21)
    q_end = s.bubble(M + 24, zy + 80, W - 2 * M - 48, [f"Q: {QUESTION}"], "yellow", size=23,
                     weight=600, tail=None, pad=16)
    s.bubble(M + 24, q_end + 14, W - 2 * M - 48,
             ["A: “… a one-time home office allowance of USD 500 …”"], "teal", size=23, weight=700,
             tail=None, pad=16)
    s.text(M + 26, zy + 266, "source", 21, 700, TEXT["muted"])
    s.chip(M + 26 + measure("source", 21, 700) + 12, zy + 236, CITATION, "blue", size=19)
    return s


# ----------------------------------------------------------------------------- slide 2
def zone(s: Slide, y: float, h: float, color: str, n: int, title: str, note: str) -> None:
    s.box(M, y, W - 2 * M, h, color, "wash", r=26, dashed=True, sw=3)
    s.badge(M + 40, y + 42, n, color, r=20, size=22)
    tw = s.text(M + 70, y + 52, title, 30, 800, TEXT[color])
    s.text(M + 70 + tw + 14, y + 52, note, 22, 500, TEXT["muted"])


def caption(s: Slide, x: float, y: float, text: str, width: float = 220) -> None:
    s.paragraph(x, y, text, 20, 500, width, 25, TEXT["muted"], anchor="middle", max_lines=2)


def slide_pipeline() -> Slide:
    s = Slide(2, TOTAL, "01", "Index once, answer many times", seed=22)
    heading(s, "Index once, answer many times", "The three stages of RAG, as built in the Day 01 notebook.")

    # Zone 1 — indexing
    y1 = 268
    zone(s, y1, 336, "blue", 1, "INDEXING", "once, before any question")
    cy = y1 + 150
    s.node(150, cy, "files", "green", "8 DOCS", size=108, icon_size=56)
    caption(s, 150, cy + 136, "Markdown + metadata")
    s.arrow(212, cy, 286, cy, INK)
    s.node(350, cy, "scissors", "orange", "SPLIT", size=108, icon_size=56)
    caption(s, 350, cy + 136, "on ## headings")
    s.arrow(412, cy, 486, cy, INK)
    # fanned chunk cards
    for i, c in enumerate(["blue", "green", "orange"]):
        s.box(520 + i * 16, cy - 58 + i * 12, 82, 96, c, "fill", r=10, sw=2.6, jitter=1)
    for k in range(3):
        s.line(570 + 0, cy - 10 + k * 16, 618, cy - 10 + k * 16, INK, sw=2.4, bow=0.4)
    s.tag(574, cy + 88, "34 CHUNKS", "orange", size=22, anchor="middle")
    caption(s, 574, cy + 136, "each with a citable ID", width=250)
    s.arrow(662, cy, 716, cy, INK)
    # index card
    ix, iy, iw, ih = 728, y1 + 74, 274, 232
    s.box(ix, iy, iw, ih, "blue", "fill", r=16)
    s.text(ix + iw / 2, iy + 40, "BM25 INDEX", 24, 800, TEXT["blue"], anchor="middle")
    s.line(ix + 18, iy + 56, ix + iw - 18, iy + 56, COLORS["blue"][0], sw=2.4, bow=0.3)
    rows = [("allowance", "1 chunk"), ("vpn", "6 chunks"), ("acme", "12 chunks")]
    for i, (term, n) in enumerate(rows):
        yy = iy + 92 + i * 38
        s.text(ix + 16, yy, term, 19, 600, INK, family="mono")
        s.text(ix + iw - 16, yy, f"→ {n}", 19, 500, INK, family="mono", anchor="end", max_width=130)
    s.text(ix + iw / 2, iy + ih - 24, "rare word = more weight", 18, 700, TEXT["blue"], anchor="middle",
           max_width=iw - 32)

    # Zone 2 — retrieval
    y2 = 632
    zone(s, y2, 288, "green", 2, "RETRIEVAL", "every question")
    cy2 = y2 + 140
    s.node(150, cy2, "message-circle", "yellow", "QUESTION", size=104, icon_size=54)
    s.arrow(210, cy2, 276, cy2, INK)
    # keyword chips
    for i, word in enumerate(["home", "office", "allowance"]):
        w = measure(word, 21, 700) + 22
        s.box(350 - w / 2, cy2 - 64 + i * 40, w, 34, "green", "fill", r=12, sw=2.2, jitter=0.8)
        s.text(350, cy2 - 40 + i * 40, word, 21, 700, INK, anchor="middle")
    s.tag(350, cy2 + 86, "KEYWORDS", "green", size=22, anchor="middle")
    s.arrow(426, cy2, 494, cy2, INK)
    # score bars
    s.box(510, cy2 - 52, 128, 104, "teal", "fill", r=16)
    for i, frac in enumerate([1.0, 0.37, 0.32]):
        s.box(526, cy2 - 38 + i * 30, 12 + 84 * frac, 20, "teal", "#FFFFFF", r=6, sw=2, jitter=0.6)
    s.tag(574, cy2 + 86, "BM25 SCORE", "teal", size=22, anchor="middle")
    s.arrow(652, cy2, 790, cy2, INK)
    s.node(880, cy2, "list-ordered", "blue", "TOP 3", size=104, icon_size=54)
    # index -> scoring lookup
    s.arrow(ix + 40, iy + ih + 4, 610, cy2 - 60, COLORS["blue"][0], sw=3, bend=-0.15, dashed=True)
    s.text(726, y2 + 30, "look up terms", 20, 700, TEXT["blue"])

    # Zone 3 — generation (flows right to left)
    y3 = 948
    zone(s, y3, 290, "violet", 3, "GENERATION", "answer from the evidence")
    cy3 = y3 + 132
    s.arrow(880, cy2 + 120, 864, cy3 - 60, INK)
    s.node(860, cy3, "file-text", "violet", "PROMPT", size=104, icon_size=54)
    s.text(860, cy3 + 128, "rules + evidence + question", 19, 500, TEXT["muted"], anchor="middle", max_width=290)
    s.arrow(802, cy3, 702, cy3, INK)
    s.node(640, cy3, "brain", "violet", "LLM", size=104, icon_size=54)
    caption(s, 618, cy3 + 128, "Day 01: no API yet", width=190)
    s.arrow(582, cy3 - 8, 456, cy3 - 8, INK)
    s.node(400, cy3 - 8, "message-square", "teal", None, size=88, icon_size=46)
    s.text(400, cy3 + 66, "cited answer", 22, 800, TEXT["teal"], anchor="middle")
    s.text(282, cy3 + 2, "or", 28, 800, TEXT["muted"], anchor="middle")
    s.node(168, cy3 - 8, "ban", "red", None, size=88, icon_size=46)
    s.text(168, cy3 + 66, "“I don’t know”", 22, 800, TEXT["red"], anchor="middle")
    s.text(168, cy3 + 96, "if best score < 4.0", 19, 600, TEXT["red"], anchor="middle")
    return s


# ----------------------------------------------------------------------------- slide 3
def slide_example() -> Slide:
    s = Slide(3, TOTAL, "01", "What happens to one question", seed=33)
    heading(s, "What happens to one question", "A real run from the Day 01 notebook. No API, no embeddings.")

    # 1. the question
    s.node(108, 334, "user", "yellow", None, size=92, icon_size=50)
    lines = wrap(QUESTION, 26, 600, 760)
    s.bubble(180, 282, 840, lines, "yellow", size=26, weight=600, tail=None, pad=18)

    step_label(s, 120, 440, 1, "split into keywords", "green")
    s.arrow(80, 404, 80, 466, INK, sw=3)
    # 2. keyword chips
    x, cy = M, 520
    chip_pos = {}
    for word in TOKENS:
        w = measure(word, 25, 700) + 30
        color = "gray" if word == "much" else "green"
        s.box(x, cy - 28, w, 52, color, "fill", r=16, sw=2.6, jitter=1)
        s.text(x + w / 2, cy + 8, word, 25, 700, INK if word != "much" else TEXT["muted"], anchor="middle")
        chip_pos[word] = (x, w)
        x += w + 14
    mx, mw = chip_pos["much"]
    s.line(mx + 10, cy - 2, mx + mw - 10, cy + 4, COLORS["red"][0], sw=3.4)
    s.arrow(mx + 40, cy + 64, mx + 52, cy + 34, COLORS["red"][0], sw=2.6, head=14)
    s.text(mx + 4, cy + 92, "in no document: adds 0", 21, 700, TEXT["red"])
    ax, aw = chip_pos["allowance"]
    s.arrow(ax + aw / 2 + 30, cy + 64, ax + aw / 2, cy + 34, COLORS["green"][0], sw=2.6, head=14)
    s.text(ax - 40, cy + 92, "rare (1 of 34 chunks): counts most", 21, 700, TEXT["green"])

    step_label(s, 120, 676, 2, "score every chunk with BM25", "teal")
    s.arrow(80, 640, 80, 702, INK, sw=3)
    # 3. ranked chunks with bars
    scale = 610 / 14
    base_x = M + 6
    tx = base_x + MIN_SCORE * scale
    top = 740
    last_bottom = 0.0
    for i, (cid, score) in enumerate(TOP3):
        y0 = top + i * 96
        s.text(base_x, y0, f"#{i + 1} {cid}", 19, 600 if i == 0 else 500, INK if i == 0 else TEXT["muted"],
               family="mono")
        bw = score * scale
        s.box(base_x, y0 + 12, bw, 40, "teal", COLORS["teal"][1] if i == 0 else COLORS["gray"][1],
              r=10, sw=2.6, jitter=0.8, stroke=COLORS["teal"][0] if i == 0 else COLORS["gray"][0])
        s.text(base_x + bw + 14, y0 + 44, f"{score:.2f}", 28, 800, INK)
        last_bottom = y0 + 52
        s.line(tx, y0 + 16, tx, y0 + 60, COLORS["red"][0], sw=3.2, dashed=False, bow=0.2)
    s.text(tx - 6, last_bottom + 40, "4.0 = MIN_SCORE", 21, 800, TEXT["red"], anchor="middle")
    s.paragraph(810, top + 230, "If even #1 scored below the red line: “I don’t know”.", 21, 600, 210, 27,
                TEXT["red"], max_lines=3)

    step_label(s, 120, 1078, 3, "answer from the top chunk, with its source", "violet")
    s.arrow(80, 1042, 80, 1104, INK, sw=3)
    # 4. answer
    answer_lines = wrap(f"“{ANSWER}”", 23, 600, 860)
    bottom = s.bubble(M, 1112, W - 2 * M, answer_lines, "teal", size=23, weight=600, tail=None, pad=16,
                      line_height=30)
    s.text(W - M - measure(CITATION, 18, 500, "mono") - 26 - 12, bottom + 38, "source", 20, 700,
           TEXT["muted"], anchor="end")
    s.chip(W - M, bottom + 10, CITATION, "blue", size=18, anchor="end")
    return s


# ----------------------------------------------------------------------------- slide 4
def slide_failures() -> Slide:
    s = Slide(4, TOTAL, "01", "Three ways the Day 01 baseline breaks", seed=44)
    heading(s, "3 ways it breaks, on purpose", "Real runs from the notebook. Each one sets up a later day.")
    cases = [
        ("red", "Wrong words: evidence never found", "How much time off do new dads get?",
         ("file-text", "gray", "LEAVE POLICY", "0 shared keywords"),
         ("ban", "red", "I DON’T KNOW", "best 3.15 < 4.0"),
         "Later: embeddings, hybrid search, query rewriting (Days 04, 08, 10)"),
        ("orange", "Relevant is not the same as answering", "What is Google’s remote work policy?",
         ("file-text", "orange", "ACME POLICY", "matches: 5.57"),
         ("triangle-alert", "orange", "WRONG ANSWER", "Acme’s, not Google’s"),
         "Later: a reader that checks the evidence, plus an eval set (Days 06, 07)"),
        ("green", "“I don’t know” is a feature", "What is the capital of France?",
         ("ban", "gray", "NO MATCH", "0 keywords in docs"),
         ("circle-check", "green", "I DON’T KNOW", "the right call"),
         "Keep this behaviour as the system grows."),
    ]
    y, h, gap = 268, 300, 16
    for n, (color, title, q, result, outcome, later) in enumerate(cases, start=1):
        s.box(M, y, W - 2 * M, h, color, "wash", r=24, dashed=True, sw=3)
        s.badge(M + 40, y + 42, n, color, r=20, size=22)
        s.text(M + 70, y + 52, title, 29, 800, TEXT[color])
        cy = y + 140
        qlines = wrap(f"“{q}”", 22, 600, 250)
        qh = 32 + 22 * 0.95 + (len(qlines) - 1) * 29
        s.bubble(M + 22, cy - qh / 2, 290, qlines, "yellow", size=22, weight=600, tail=None, pad=16,
                 line_height=29)
        s.arrow(M + 322, cy, M + 384, cy, INK)
        s.node(510, cy, "search", "blue", "SEARCH", size=88, icon_size=46, label_size=19)
        s.arrow(562, cy, 624, cy, INK)
        icon, c, label, sub = result
        s.node(690, cy, icon, c, label, size=88, icon_size=46, label_size=19)
        s.text(690, cy + 116, sub, 19, 600, TEXT["muted"], anchor="middle")
        if n == 1:
            s.line(660, cy - 30, 720, cy + 30, COLORS["red"][0], sw=4)
            s.line(720, cy - 30, 660, cy + 30, COLORS["red"][0], sw=4)
        s.arrow(742, cy, 804, cy, INK)
        icon, c, label, sub = outcome
        s.node(890, cy, icon, c, label, size=88, icon_size=46, label_size=19)
        s.text(890, cy + 116, sub, 19, 600, TEXT["muted"], anchor="middle")
        s.text(M + 24, y + h - 22, "→ " + later, 21, 700, TEXT[color], max_width=W - 2 * M - 48)
        y += h + gap
    s.text(M, 1236, "Next → Day 02: ingestion, parsing and metadata", 26, 800, TEXT["teal"])
    return s


def main() -> None:
    check_contrast()
    check_facts()
    OUT.mkdir(parents=True, exist_ok=True)
    names = ["slide-01-cover", "slide-02-how-it-works", "slide-03-example", "slide-04-takeaways"]
    for name, build in zip(names, [slide_cover, slide_pipeline, slide_example, slide_failures]):
        path = OUT / f"{name}.svg"
        path.write_text(build().svg(), encoding="utf-8")
        print("wrote", path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
