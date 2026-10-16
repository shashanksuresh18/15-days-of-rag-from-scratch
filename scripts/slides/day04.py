"""Build the Day 04 LinkedIn carousel (whiteboard style) as editable SVGs in assets/day-04/.

    python scripts/slides/day04.py                    # writes the SVGs
    python scripts/slides/render.py assets/day-04     # renders PNGs + PDF

Every number, ID and score on the slides comes from the executed Day 04 notebook.
`check_facts()` confirms each one appears in the notebook's saved outputs (or, for the model's
pinned settings, in its code) before anything is written, so the slides cannot silently drift
from the code.
"""

from __future__ import annotations

import json
from pathlib import Path

from kit import COLORS, INK, M, TEXT, W, Slide, check_contrast

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "day-04"
NOTEBOOK = ROOT / "notebooks" / "day_04_embeddings_and_similarity.ipynb"
TOTAL = 4
DAY = "04"
PAPER_WHITE = "#FFFFFF"

QUESTION = "How much time off do new dads get?"
TOKENS = ['[CLS]', 'how', 'much', 'time', 'off', 'do', 'new', 'dad', '##s', 'get', '?', '[SEP]']
# BM25's top 3 for the "new dads" question: (chunk ID, score, the only term it matched)
BM25_DADS = [("security-incident-reporting#what-not-to-do", "3.51", "off"),
             ("remote-work-policy#purpose", "2.89", "time"),
             ("new-starter-it-checklist#your-first-day", "2.78", "new")]
BM25_LABELS = ["#what-not-to-do", "#purpose", "#your-first-day"]   # sections of 3 other policies
DENSE_DADS = [("parental-leave-policy#returning-to-work", "0.72"),
              ("parental-leave-policy#who-is-eligible", "0.67"),
              ("parental-leave-policy#leave-entitlement", "0.66")]
# (question, BM25 rank, embeddings rank) for the 15 answerable questions, in notebook order
RANKS = [
    ("What is the hotel limit per night in Singapore?", "1", "1"),
    ("What is the daily meal limit in New York?", "1", "1"),
    ("Who approves admin rights?", "1", "1"),
    ("How long is parental leave?", "3", "2"),
    ("How many failed sign-in attempts lock my account?", "1", "1"),
    ("How long are application logs kept?", "1", "1"),
    ("How much is the home office allowance?", "1", "1"),
    ("How long do I have to claim the home office allowance?", "1", "1"),
    ("Can I use my personal computer for work?", "1", "1"),
    ("Can I install software from the internet myself?", "1", "1"),
    ("What training must new starters complete?", "1", "1"),
    ("What happens after I report a security incident?", "1", "1"),
    ("How soon must I report a lost laptop?", "3", "-"),
    ("How much time off do new dads get?", "-", "3"),
    ("My account is locked, what now?", "1", "1"),
]
ANSWERABLE_TOP1 = [0.626, 0.667, 0.672, 0.676, 0.708, 0.721, 0.739, 0.746, 0.749, 0.758, 0.761, 0.769,
                   0.815, 0.824, 0.856]
OUT_OF_SCOPE = [(0.425, "capital of France"), (0.710, "Google's remote work policy")]
LOST = ["0.626  What training must new starters complete?", "0.667  Can I use my personal computer for work?",
        "0.672  Who approves admin rights?", "0.676  How soon must I report a lost laptop?",
        "0.708  What is the hotel limit per night in Singapore?"]
USD220_DENSE = ["expense-reimbursement-policy#meals", "remote-work-policy#equipment-and-home-office",
                "expense-reimbursement-policy#approval-and-payment"]
# (variant, found in top 3 of 15, at rank 1)
KNOBS = [("BM25, with headers", 14, 12), ("BM25, no headers", 13, 9),
         ("embeddings, with headers", 14, 12), ("embeddings, no headers", 15, 13)]
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

FACTS = [
    "11 documents -> 44 chunks",
    "model tokens: " + str(TOKENS),
    "shape (1, 12, 384): 1 text, 12 tokens, 384 numbers per token",
    "the one related pair (1 and 2): 0.62; the nine other pairs: 0.30 to 0.50",
    "SHA-256 828e1496d7fa... verified",
    *(f"#{i}  {score}  [{cid}]" for i, (cid, score, _) in enumerate(BM25_DADS, start=1)),
    *(f"BM25 #{i} [{cid}] matched only ['{term}']" for i, (cid, _, term) in enumerate(BM25_DADS, start=1)),
    *(f"#{i}  {score}  [{cid}]" for i, (cid, score) in enumerate(DENSE_DADS, start=1)),
    *(f"{q:<55}{b:>6}{e:>12}" for q, b, e in RANKS),
    f"{'What is Google' + chr(39) + 's remote work policy?':<55}{'(6.21)':>6}{'(0.71)':>12}",
    f"{'What is the capital of France?':<55}{'0 hits':>6}{'(0.42)':>12}",
    "BM25        14/15 answerable questions found in the top 3, 12 at rank 1",
    "embeddings  14/15 answerable questions found in the top 3, 12 at rank 1",
    "found by at least one of the two: 15/15",
    "15 answerable questions: " + " ".join(f"{s:.3f}" for s in ANSWERABLE_TOP1),
    "0.425 (What is the capital of France?)   0.710 (What is Google's remote work policy?)",
    "and also 5 of the 15 answerable ones",
    *LOST,
    "25 values       25/25 (25 at rank 1)    14/25 (8 at rank 1)",
    "USD 220\n  BM25\n    #1  5.95  [travel-policy#daily-limits-by-city]",
    *(f"[{cid}]" for cid in USD220_DENSE),
    *(f"{name:<42}{found:>12}/15{first:>11}" for name, found, first in KNOBS),
    "rank 1 -> 3   How much time off do new dads get?",
    "with headers  same document 0.71 | different documents 0.52",
    "no headers    same document 0.65 | different documents 0.52",
    "first run: 44 chunks embedded; after the edit: 1 of 44 re-embedded",
    "a million chunks would need 1.5 GB",
    "model file: 133 MB",
]
# Settings that live in the notebook's code or text rather than its outputs
SOURCE_FACTS = [
    f'"revision": "{REVISION}"', '"licence": "MIT"', "33 million parameters",
    'QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "',
    "This model uses the vector of the first token, `[CLS]`",
]


def check_facts() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    outputs = "".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code"
                      for o in c.get("outputs", []))
    source = "".join("".join(c["source"]) for c in nb["cells"])
    missing = [f for f in FACTS if f not in outputs] + [f for f in SOURCE_FACTS if f not in source]
    if missing:
        raise SystemExit(f"Slide facts not found in the notebook (re-run it?): {missing}")
    # The dense "USD 220" hits must be exactly the three chunks drawn on slide 3, in order
    block = outputs[outputs.index("USD 220\n"):]
    block = block[block.index("  embeddings"):]
    if [line.split("[")[1].split("]")[0] for line in block.splitlines()[1:4]] != USD220_DENSE:
        raise SystemExit("Slide 3's USD 220 hits differ from the notebook")


def heading(s: Slide, title: str, subtitle: str, size: int = 54) -> None:
    s.text(M, 190, title, size, 800, INK)
    s.text(M, 238, subtitle, 25, 500, TEXT["muted"])


def tcolor(color: str) -> str:
    return TEXT.get(color, INK) if color != "yellow" else INK


def icon_color(color: str) -> str:
    return COLORS[color][0] if color != "yellow" else TEXT["orange"]


def section_label(chunk_id: str) -> str:
    return "#" + chunk_id.split("#")[1]


# ----------------------------------------------------------------------------- slide 1
def hit_row(s: Slide, x: float, y: float, w: float, rank: int, label: str, right: str, color: str,
            highlight: str | None = None) -> None:
    if highlight:
        s.box(x - 6, y - 30, w + 12, 44, highlight, "fill", r=10, sw=2, jitter=0.6)
    s.text(x, y, f"#{rank}", 20, 800, INK, max_width=40)
    s.text(x + 44, y, label, 17, 600 if highlight else 500, INK, family="mono", max_width=w - 150)
    s.text(x + w, y, right, 20, 800, tcolor(color), anchor="end", max_width=110)


def slide_cover() -> Slide:
    s = Slide(1, TOTAL, DAY, "Same score, different misses", seed=41)
    heading(s, "Same score, different misses", "Day 04: keyword search vs embeddings on the same 44 chunks.")
    s.text(M, 304, f"Q: {QUESTION}", 25, 800, INK)

    zy, zh = 330, 432
    zw = (W - 2 * M - 20) / 2
    # --- BM25
    s.box(M, zy, zw, zh, "red", "wash", r=22, dashed=True, sw=3)
    s.icon("whole-word", M + 36, zy + 40, 32, COLORS["red"][0], sw=2.3)
    s.text(M + 64, zy + 50, "BM25 · MATCHES WORDS", 23, 800, TEXT["red"])
    s.text(M + 24, zy + 96, "top 3, and the one word each shares:", 20, 600, TEXT["muted"], max_width=zw - 48)
    s.box(M + 18, zy + 116, zw - 36, 210, "gray", PAPER_WHITE, r=12, sw=2.4)
    for i, (cid, _score, term) in enumerate(BM25_DADS):
        hit_row(s, M + 34, zy + 162 + i * 48, zw - 68, i + 1, BM25_LABELS[i], f"“{term}”", "red")
    s.text(M + 34, zy + 300, "sections of 3 unrelated policies", 16, 600, TEXT["muted"], max_width=zw - 68)
    s.icon("circle-x", M + 38, zy + 360, 30, COLORS["red"][0], sw=2.4)
    s.text(M + 64, zy + 369, "No chunk says “dads”.", 22, 800, TEXT["red"], max_width=zw - 80)
    s.text(M + 24, zy + 402, "The answer never reaches the top 3.", 20, 600, TEXT["muted"], max_width=zw - 48)

    # --- embeddings
    bx = M + zw + 20
    s.box(bx, zy, zw, zh, "green", "wash", r=22, dashed=True, sw=3)
    s.icon("brain", bx + 36, zy + 40, 32, COLORS["green"][0], sw=2.3)
    s.text(bx + 64, zy + 50, "EMBEDDINGS · MEANING", 23, 800, TEXT["green"])
    s.text(bx + 24, zy + 96, "top 3, cosine similarity:", 20, 600, TEXT["muted"], max_width=zw - 48)
    s.box(bx + 18, zy + 116, zw - 36, 210, "gray", PAPER_WHITE, r=12, sw=2.4)
    for i, (cid, score) in enumerate(DENSE_DADS):
        hit_row(s, bx + 34, zy + 162 + i * 48, zw - 68, i + 1, section_label(cid), score, "green",
                highlight="green" if i == 2 else None)
    s.text(bx + 34, zy + 300, "“Primary caregivers receive 16 weeks…”", 16, 600, TEXT["muted"], max_width=zw - 68)
    s.icon("circle-check", bx + 38, zy + 360, 30, COLORS["green"][0], sw=2.4)
    s.text(bx + 64, zy + 369, "Right policy. Answer at #3.", 22, 800, TEXT["green"], max_width=zw - 80)
    s.text(bx + 24, zy + 402, "All three: parental-leave-policy", 20, 600, TEXT["muted"], max_width=zw - 48)

    # --- 15 questions as dots, in notebook order
    ry = 812
    s.text(M, ry, "15 answerable questions · top 3 · same chunks", 24, 800, INK)
    s.text(M, ry + 30, "Where is the chunk with the complete evidence?", 21, 600, TEXT["muted"])
    step, x0 = 42, M + 150
    rows = [("BM25", "red", [b for _, b, _ in RANKS]), ("embed.", "green", [e for _, _, e in RANKS])]
    for i, (label, color, ranks) in enumerate(rows):
        y = ry + 100 + i * 74
        s.text(M, y + 9, label, 26, 800, tcolor(color), max_width=140)
        for j, rank in enumerate(ranks):
            cx = x0 + j * step
            if rank == "-":
                s.ellipse(cx, y, 16, 16, "red", "wash", sw=2.6, jitter=0.8)
                s.icon("x", cx, y, 20, COLORS["red"][0], sw=3)
            elif rank == "1":
                s.ellipse(cx, y, 16, 16, "green", "fill", sw=2.6, jitter=0.8)
            else:
                s.ellipse(cx, y, 16, 16, "green", "wash", sw=2.6, jitter=0.8)
                s.text(cx, y + 7, rank, 18, 800, TEXT["green"], anchor="middle", max_width=28)
        s.text(x0 + 15 * step + 4, y + 10, "14/15", 28, 800, INK)
    # name each miss, next to its own row
    s.text(x0 + 13 * step + 18, ry + 66, "new dads", 20, 800, TEXT["red"], anchor="end")
    s.text(x0 + 12 * step - 18, ry + 214, "lost laptop", 20, 800, TEXT["red"], anchor="end")
    s.text(x0 + 15 * step + 4, ry + 66, "found", 20, 700, TEXT["muted"])
    lx = W - M - 400
    s.ellipse(lx, ry + 246, 9, 9, "green", "fill", sw=2, jitter=0.5)
    s.text(lx + 16, ry + 252, "#1", 18, 700, TEXT["muted"])
    s.ellipse(lx + 72, ry + 246, 9, 9, "green", "wash", sw=2, jitter=0.5)
    s.text(lx + 88, ry + 252, "#2 or #3", 18, 700, TEXT["muted"])
    s.icon("x", lx + 214, ry + 246, 16, COLORS["red"][0], sw=3)
    s.text(lx + 230, ry + 252, "not in top 3", 18, 700, TEXT["muted"])

    s.bubble(M, 1090, W - 2 * M, ["Different misses: together they find 15/15.",
                                  "That's the case for hybrid search (Day 08)."],
             "yellow", size=24, weight=700, tail=None, pad=20)
    s.text(M, 1226, "Real run · Day 04 notebook · local model, free, no API key", 21, 600, TEXT["muted"])
    return s


# ----------------------------------------------------------------------------- slide 2
def step_row(s: Slide, y: float, n: int, icon: str, color: str, title: str, note: str, mono: bool = False) -> None:
    s.box(M, y, W - 2 * M, 104, color, "wash", r=18, sw=2.6, jitter=1.1)
    s.badge(M + 34, y + 52, n, color, r=20, size=22)
    s.node(M + 118, y + 52, icon, color, None, size=76, icon_size=40)
    s.text(M + 176, y + 44, title, 25, 800, tcolor(color))
    if mono:
        s.text(M + 176, y + 80, note, 17, 500, INK, family="mono", max_width=W - 2 * M - 196)
    else:
        s.text(M + 176, y + 80, note, 21, 600, TEXT["muted"], max_width=W - 2 * M - 196)


def slide_how() -> Slide:
    s = Slide(2, TOTAL, DAY, "Text in, 384 numbers out", seed=42)
    heading(s, "Text in, 384 numbers out", "bge-small-en-v1.5 · MIT · 33M parameters · runs on a CPU")

    steps = [("whole-word", "blue", "TOKENS", " ".join(TOKENS), True),
             ("cpu", "violet", "MODEL", "12 tokens in → 12 vectors of 384 numbers out", False),
             ("layers", "orange", "POOL", "keep the [CLS] vector: this model's rule (others average)", False),
             ("ruler", "teal", "NORMALISE", "divide by its length → every vector has length 1", False)]
    y = 280
    for i, (icon, color, title, note, mono) in enumerate(steps):
        step_row(s, y + i * 120, i + 1, icon, color, title, note, mono)

    # --- compare: cosine, and why the score is relative
    zy = 780
    s.box(M, zy, W - 2 * M, 410, "yellow", "wash", r=22, dashed=True, sw=3)
    s.text(M + 30, zy + 50, "COMPARE: COSINE SIMILARITY", 23, 800, INK)
    # angle sketch
    ox, oy = M + 70, zy + 300
    s.arrow(ox, oy, ox + 210, oy - 40, COLORS["blue"][0], sw=3.2, head=16)
    s.arrow(ox, oy, ox + 150, oy - 170, COLORS["green"][0], sw=3.2, head=16)
    s.text(ox + 218, oy - 34, "question", 20, 700, TEXT["blue"], max_width=120)
    s.text(ox + 158, oy - 172, "chunk", 20, 700, TEXT["green"], max_width=100)
    s.text(ox + 60, oy - 40, "θ", 26, 800, INK, max_width=30)
    s.text(M + 30, zy + 370, "length 1 → cosine = dot product", 20, 700, INK, max_width=380)
    # relative scores: a 0..1 scale
    sx, sw_, sy = M + 440, W - 2 * M - 480, zy + 200
    s.text(sx, zy + 100, "Scores are relative, not %", 22, 800, INK, max_width=sw_)
    s.line(sx, sy, sx + sw_, sy, INK, sw=2.6, bow=0.1)
    for v in (0, 0.5, 1):
        s.line(sx + v * sw_, sy - 8, sx + v * sw_, sy + 8, INK, sw=2.4, bow=0.1)
        s.text(sx + v * sw_, sy + 36, f"{v:g}", 18, 700, TEXT["muted"], anchor="middle", max_width=40)
    s.box(sx + 0.30 * sw_, sy - 22, 0.20 * sw_, 44, "gray", "fill", r=10, sw=2.2, jitter=0.6)
    s.ellipse(sx + 0.62 * sw_, sy, 13, 13, "green", "fill", sw=2.4, jitter=0.6)
    s.text(sx + 0.40 * sw_, sy - 36, "0.30–0.50", 20, 800, TEXT["muted"], anchor="middle", max_width=120)
    s.text(sx + 0.62 * sw_, sy - 30, "0.62", 20, 800, TEXT["green"], anchor="middle", max_width=60)
    s.paragraph(sx, sy + 86, "Gray: 9 pairs of unrelated sentences. Green: “new dads” vs “secondary caregivers”.",
                18, 600, sw_, 26, TEXT["muted"], max_lines=3)
    s.text(sx, zy + 370, "Rank by score. It's not a percentage.", 20, 800, TEXT["teal"], max_width=sw_)
    return s


# ----------------------------------------------------------------------------- slide 3
def slide_failures() -> Slide:
    s = Slide(3, TOTAL, DAY, "Where each one fails", seed=43)
    heading(s, "Where each one fails", "Two systematic tests on the same chunks, top 3.")

    # --- exact values
    zy = 280
    s.box(M, zy, W - 2 * M, 380, "blue", "wash", r=22, dashed=True, sw=3)
    s.icon("hash", M + 38, zy + 42, 30, COLORS["blue"][0], sw=2.3)
    s.text(M + 66, zy + 52, "EXACT VALUES · BM25'S HOME GROUND", 23, 800, TEXT["blue"])
    s.text(M + 30, zy + 92, "Every number in the documents as a query: 25 (USD 220, 13 months, 15:00…)",
           20, 600, TEXT["muted"], max_width=W - 2 * M - 60)
    bx, bw = M + 210, W - 2 * M - 360
    for i, (label, found, first, color) in enumerate([("BM25", 25, 25, "green"), ("embeddings", 14, 8, "orange")]):
        y = zy + 134 + i * 70
        s.text(M + 30, y + 30, label, 23, 800, INK, max_width=170)
        s.box(bx, y, bw, 44, "gray", PAPER_WHITE, r=10, sw=2.2, jitter=0.6)
        s.box(bx, y, bw * found / 25, 44, color, "fill", r=10, sw=2.4, jitter=0.6)
        s.text(bx + bw + 16, y + 30, f"{found}/25", 24, 800, tcolor(color), max_width=110)
        s.text(bx + 16, y + 30, f"{first} at #1", 20, 800, INK, max_width=bw * found / 25 - 20)
    s.text(M + 30, zy + 306, "“USD 220” (Singapore hotel) → embeddings return chunks with", 20, 700, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 340, "USD 75, USD 500 and USD 1,000. One amount looks like another.", 20, 700, INK,
           max_width=W - 2 * M - 60)

    # --- no such thing as no match: dot plot of top-1 scores
    zy = 690
    s.box(M, zy, W - 2 * M, 484, "red", "wash", r=22, dashed=True, sw=3)
    s.icon("ban", M + 38, zy + 42, 30, COLORS["red"][0], sw=2.3)
    s.text(M + 66, zy + 52, "NO SUCH THING AS “NO MATCH”", 23, 800, TEXT["red"])
    s.text(M + 30, zy + 92, "Embeddings always return 3. Top-1 cosine per question:", 20, 600, TEXT["muted"],
           max_width=W - 2 * M - 60)
    lo, hi = 0.40, 0.90
    ax0, ax1, ay = M + 50, W - M - 50, zy + 300

    def px(v: float) -> float:
        return ax0 + (v - lo) / (hi - lo) * (ax1 - ax0)

    s.line(ax0, ay, ax1, ay, INK, sw=2.6, bow=0.1)
    for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        s.line(px(v), ay - 7, px(v), ay + 7, INK, sw=2.2, bow=0.1)
        s.text(px(v), ay + 34, f"{v:.1f}", 18, 700, TEXT["muted"], anchor="middle", max_width=50)
    placed: list[tuple[float, int]] = []   # beeswarm: stack a dot on the lowest free level

    def level_for(x: float) -> int:
        level = 0
        while any(abs(x - px_) < 24 and lv == level for px_, lv in placed):
            level += 1
        placed.append((x, level))
        return level

    for v in ANSWERABLE_TOP1:
        y = ay - 22 - level_for(px(v)) * 26
        s.ellipse(px(v), y, 11, 11, "green", "fill", sw=2.2, jitter=0.5)
    for v, _label in OUT_OF_SCOPE:
        y = ay - 22 - level_for(px(v)) * 26
        s.ellipse(px(v), y, 11, 11, "red", "fill", sw=2.2, jitter=0.5)
        s.icon("x", px(v), y, 14, COLORS["red"][0], sw=3)
    # labels for the two out-of-scope questions
    s.text(px(0.425), ay - 64, "France 0.425", 20, 800, TEXT["red"], anchor="middle", max_width=150)
    gx = px(0.710)
    s.line(gx, ay - 140, gx, ay + 8, COLORS["red"][0], sw=2.4, dashed=True, bow=0.1)
    s.text(gx, ay - 150, "Google's policy 0.710", 20, 800, TEXT["red"], anchor="middle", max_width=230)
    s.text(px(0.64), ay - 112, "5 answerable", 20, 800, TEXT["green"], anchor="middle", max_width=170)
    s.text(px(0.64), ay - 90, "below it", 20, 800, TEXT["green"], anchor="middle", max_width=170)
    s.ellipse(M + 44, zy + 378, 9, 9, "green", "fill", sw=2, jitter=0.5)
    s.text(M + 60, zy + 384, "15 answerable questions", 18, 700, INK, max_width=280)
    s.ellipse(M + 340, zy + 378, 9, 9, "red", "fill", sw=2, jitter=0.5)
    s.text(M + 356, zy + 384, "2 the documents can't answer", 18, 700, INK, max_width=320)
    s.text(M + 30, zy + 422, "A threshold that rejects Google's also rejects Singapore's hotel limit (0.708).",
           20, 700, INK, max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 454, "BM25: France → 0 hits, abstains. Google → 6.21, doesn't.", 20, 700, TEXT["muted"],
           max_width=W - 2 * M - 60)
    s.text(M, 1222, "Similarity means “same topic”, not “answers the question”.", 23, 800, TEXT["teal"])
    return s


# ----------------------------------------------------------------------------- slide 4
def slide_knobs() -> Slide:
    s = Slide(4, TOTAL, DAY, "Measure per retriever, pin the model", seed=44)
    heading(s, "Same knob, opposite effects", "The “Title > Section” header line, on 15 answerable questions.",
            size=52)

    # grouped bars: found in top 3, with vs without headers
    cy, ch = 290, 400
    s.box(M, cy, W - 2 * M, ch, "violet", "wash", r=22, sw=2.8)
    s.icon("heading", M + 38, cy + 42, 30, COLORS["violet"][0], sw=2.3)
    s.text(M + 66, cy + 52, "FOUND IN THE TOP 3 (AND AT #1)", 22, 800, TEXT["violet"])
    base, unit = cy + 300, 12
    groups = [("BM25", KNOBS[0], KNOBS[1]), ("embeddings", KNOBS[2], KNOBS[3])]
    gw = (W - 2 * M - 120) / 2
    for g, (label, with_h, without) in enumerate(groups):
        gx = M + 60 + g * (gw + 40)
        for k, ((_, found, first), color, tag) in enumerate([(with_h, "violet", "with"), (without, "gray", "without")]):
            bx = gx + 30 + k * 170
            h = found * unit
            s.box(bx, base - h, 130, h, color, "fill", r=8, sw=2.4, jitter=0.7)
            s.text(bx + 65, base - h - 14, f"{found}", 26, 800, INK, anchor="middle", max_width=120)
            s.text(bx + 65, base - h + 30, f"#1: {first}", 20, 800, INK, anchor="middle", max_width=120)
            s.text(bx + 65, base + 30, tag, 20, 700, TEXT["muted"], anchor="middle", max_width=140)
        verdict, color = ("header helps", "green") if g == 0 else ("header hurts", "red")
        s.text(gx + 200, base + 68, f"{label}: {verdict}", 23, 800, TEXT[color], anchor="middle", max_width=gw)
    s.paragraph(M + 30, cy + ch + 40, "Why: a shared header pulls a document's chunks together (similarity 0.65 → "
                "0.71). “New dads” drops from #1 to #3.", 20, 700, W - 2 * M - 60, 28, INK, max_lines=2)

    # a vector belongs to its model
    zy = 800
    s.box(M, zy, W - 2 * M, 380, "blue", "wash", r=22, dashed=True, sw=3)
    s.icon("pin", M + 38, zy + 42, 30, COLORS["blue"][0], sw=2.3)
    s.text(M + 66, zy + 52, "A VECTOR BELONGS TO ITS MODEL", 23, 800, TEXT["blue"])
    fields = [("model", "BAAI/bge-small-en-v1.5"), ("revision", REVISION[:12] + "…"), ("pooling", "CLS, length 1"),
              ("chunk_hash", "only changed chunks"), ("allowed_groups", "travels with it")]
    fx, fw = M + 24, 540
    s.box(fx, zy + 80, fw, 236, "gray", PAPER_WHITE, r=12, sw=2.4)
    for i, (key, value) in enumerate(fields):
        y = zy + 122 + i * 42
        s.text(fx + 18, y, key, 17, 700, TEXT["blue"], family="mono", max_width=180)
        s.text(fx + fw - 18, y, value, 17, 500, INK, family="mono", anchor="end", max_width=fw - 210)
    nx = fx + fw + 26
    nw = W - M - nx - 20
    s.icon("refresh-cw", nx + 14, zy + 104, 24, COLORS["orange"][0], sw=2.4)
    s.paragraph(nx + 34, zy + 112, "Edit 1 section: 1 of 44 re-embedded.", 20, 700, nw - 34, 26, INK, max_lines=2)
    s.icon("triangle-alert", nx + 14, zy + 184, 24, COLORS["red"][0], sw=2.4)
    s.paragraph(nx + 34, zy + 192, "New model or revision: re-embed all 44.", 20, 700, nw - 34, 26, INK,
                max_lines=2)
    s.icon("fingerprint-pattern", nx + 14, zy + 264, 24, COLORS["blue"][0], sw=2.4)
    s.paragraph(nx + 34, zy + 272, "Pinned commit + SHA-256. ONNX, no pickle.", 20, 700, nw - 34, 26, INK,
                max_lines=2)
    s.text(M + 24, zy + 352, "Model 133 MB, once. Then it runs offline.", 20, 600, TEXT["muted"],
           max_width=W - 2 * M - 48)
    s.text(M, 1226, "Next → Day 05: local vector search and data lifecycle", 24, 800, TEXT["teal"])
    return s


def main() -> None:
    check_contrast()
    check_facts()
    OUT.mkdir(parents=True, exist_ok=True)
    names = ["slide-01-cover", "slide-02-how-it-works", "slide-03-where-each-fails", "slide-04-knobs-and-models"]
    for name, build in zip(names, [slide_cover, slide_how, slide_failures, slide_knobs]):
        path = OUT / f"{name}.svg"
        path.write_text(build().svg(), encoding="utf-8")
        print("wrote", path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
