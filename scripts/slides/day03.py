"""Build the Day 03 LinkedIn carousel (whiteboard style) as editable SVGs in assets/day-03/.

    python scripts/slides/day03.py                    # writes the SVGs
    python scripts/slides/render.py assets/day-03     # renders PNGs + PDF

Every number, ID and table line on the slides comes from the executed Day 03 notebook.
`check_facts()` confirms each one appears in the notebook's saved outputs before anything is
written, so the slides cannot silently drift from the code.
"""

from __future__ import annotations

import json
from pathlib import Path

from kit import COLORS, INK, M, TEXT, W, Slide, check_contrast, measure

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "day-03"
NOTEBOOK = ROOT / "notebooks" / "day_03_chunking_and_structure.ipynb"
TOTAL = 4
DAY = "03"
PAPER_WHITE = "#FFFFFF"

HEADER_ROW = "City | Hotel per night | Meals per day"
ROWS = ["London | USD 250 | USD 75", "New York | USD 300 | USD 75", "Singapore | USD 220 | USD 60",
        "Other cities | USD 180 | USD 50"]

# (question, rank with A, rank with B) in notebook order; "-" = no top-3 chunk holds the evidence
RANKS = [
    ("What is the hotel limit per night in Singapore?", "-", "1"),
    ("What is the daily meal limit in New York?", "-", "1"),
    ("Who approves admin rights?", "-", "1"),
    ("How long is parental leave?", "1", "3"),
    ("How many failed sign-in attempts lock my account?", "1", "1"),
    ("How long are application logs kept?", "1", "1"),
    ("How much is the home office allowance?", "1", "1"),
    ("How long do I have to claim the home office allowance?", "2", "1"),
    ("Can I use my personal computer for work?", "1", "1"),
    ("Can I install software from the internet myself?", "1", "1"),
    ("What training must new starters complete?", "1", "1"),
    ("What happens after I report a security incident?", "1", "1"),
    ("How soon must I report a lost laptop?", "-", "3"),
]
# (window words, chunks, complete out of 13, words read per question, Singapore row kept with header)
SWEEP = [(20, 76, 6, 59, "no"), (30, 52, 8, 87, "no"), (40, 42, 9, 110, "no"), (50, 34, 12, 139, "yes"),
         (60, 30, 9, 166, "no"), (80, 23, 12, 194, "yes"), (100, 21, 12, 242, "yes"), (120, 18, 11, 284, "no")]
DOC_HASH = "bf4bdf9b29cecca37f3f45b565a210bcda1f096d2a6be1fe5ca44d2ae7a90484"
CHUNK_HASH = "e13a0fb0b50f3a0e13d6b7d61e6813fd6159faf7dd67958b6fccee0450a9e086"
FIXED_IDS = ["w000", "w040", "w080", "w120", "w160", "w200"]
SECTIONS = [("purpose", True), ("booking", False), ("daily limits", False), ("claiming", False),
            ("safety", False)]

FACTS = [
    "11 documents -> 42 fixed-size chunks of up to 40 words",
    "11 documents -> 44 structure-aware chunks (sections longer than 80 words: 0)",
    "44 sections, 13-68 words each (median 29)",
    f"w080         {HEADER_ROW}",
    f"w080 ✂ w120  {ROWS[1]}",
    f"w120         {ROWS[2]}",
    f"[travel-policy#w120]  40 words\n    300 | USD 75\n    {ROWS[2]}",
    "Contains the column header 'Hotel per night'? False",
    "Travel Policy > Daily limits by city\n    Hotel and meal costs",
    "\n    ".join([HEADER_ROW] + ROWS),
    "[travel-policy#daily-limits-by-city-part-3]  29 words\n    Travel Policy > Daily limits by city\n    "
    + "\n    ".join([HEADER_ROW] + ROWS[:3]),
    "[travel-policy#daily-limits-by-city-part-4]  19 words\n    Travel Policy > Daily limits by city\n    "
    + "\n    ".join([HEADER_ROW, ROWS[3]]),
    "[travel-policy#daily-limits-by-city-part-1]  25 words",
    "[travel-policy#daily-limits-by-city-part-2]  16 words",
    "A. fixed 40 words       9/13 in one chunk | 11/13 across top 3 |   110 words read per question",
    "B. structure-aware     13/13 in one chunk | 13/13 across top 3 |   113 words read per question",
    *(f"{q:<54}{a:>3}{b:>4}" for q, a, b in RANKS),
    "29            7 of 8",
    *(f"{size:>7}{0:>8}{chunks:>8}{complete:>19}/13{words:>12}   {kept}" for size, chunks, complete, words, kept in SWEEP),
    "B. structure-aware (up to 80 words): 13/13, 113 words read",
    "A+ with 10-word overlap   12/13 in one chunk",
    "Overlap indexes 49 chunks instead of 42: 27% more words to embed and store.",
    "repaired       What is the hotel limit per night in Singapore?",
    "repaired       What is the daily meal limit in New York?",
    "repaired       How soon must I report a lost laptop?",
    "still missing  Who approves admin rights?",
    "B without headers     12/13 in one chunk | 12/13 across top 3 |    95 words read per question",
    "rank 2 -> 3   How long is parental leave?",
    "rank 2 -> 1   Can I install software from the internet myself?",
    "rank - -> 1   What happens after I report a security incident?",
    "Added 7 words at the start of travel-policy's Purpose section.",
    "A. fixed 40 words    6 of 6 chunks changed: " + ", ".join(FIXED_IDS),
    "B. structure-aware   1 of 5 chunks changed: purpose",
    '"chunk_id": "travel-policy#daily-limits-by-city"',
    f'"doc_hash": "{DOC_HASH}"', f'"chunk_hash": "{CHUNK_HASH}"',
    "manager-pay-review-guide -> 1 chunk(s), allowed_groups ('managers', 'people-team')",
    "['all-employees'] -> 43 of 44 chunks",
]


def check_facts() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    outputs = "".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code"
                      for o in c.get("outputs", []))
    missing = [f for f in FACTS if f not in outputs]
    if missing:
        raise SystemExit(f"Slide facts not found in notebook outputs (re-run the notebook?): {missing}")


def heading(s: Slide, title: str, subtitle: str, size: int = 54) -> None:
    s.text(M, 190, title, size, 800, INK)
    s.text(M, 238, subtitle, 25, 500, TEXT["muted"])


def tcolor(color: str) -> str:
    return TEXT.get(color, INK) if color != "yellow" else INK


def icon_color(color: str) -> str:
    return COLORS[color][0] if color != "yellow" else TEXT["orange"]


def mono_lines(s: Slide, x: float, y: float, lines: list[tuple[str, str | None]], size: int, step: float,
               width: float) -> float:
    """Monospace lines; a line with a colour gets a highlight bar behind it. Returns the next y."""
    for text, highlight in lines:
        if highlight:
            s.box(x - 8, y - size * 0.98, width + 16, size * 1.42, highlight, "fill", r=6, sw=1.8, jitter=0.5)
        s.text(x, y, text, size, 600 if highlight else 500, INK, family="mono", max_width=width)
        y += step
    return y


# ----------------------------------------------------------------------------- slide 1
def slide_cover() -> Slide:
    s = Slide(1, TOTAL, DAY, "Cut a table, lose the answer", seed=31)
    heading(s, "Cut a table, lose the answer", "Day 03: chunking. Same table, same retriever, two ways to cut.")

    s.text(M, 304, "Q: What is the hotel limit per night in Singapore?", 25, 800, INK)

    # --- A: fixed windows
    zx, zy, zw, zh = M, 330, 470, 500
    s.box(zx, zy, zw, zh, "red", "wash", r=22, dashed=True, sw=3)
    s.icon("scissors", zx + 36, zy + 40, 30, COLORS["red"][0], sw=2.3)
    s.text(zx + 62, zy + 50, "A · CUT EVERY 40 WORDS", 23, 800, TEXT["red"])

    cx, cw = zx + 20, zw - 40
    s.chip(cx, zy + 76, "travel-policy#w080", "gray", size=17)
    s.box(cx, zy + 118, cw, 140, "gray", PAPER_WHITE, r=12, sw=2.4)
    s.text(cx + 16, zy + 146, "… reimbursed up to the daily limits …", 17, 600, TEXT["muted"], max_width=cw - 30)
    mono_lines(s, cx + 16, zy + 180, [(HEADER_ROW, "yellow"), (ROWS[0], None), ("New York | USD", None)], 16, 30,
               cw - 32)
    # the cut
    cut_y = zy + 279
    s.line(cx - 6, cut_y, cx + cw + 6, cut_y, COLORS["red"][0], sw=2.6, dashed=True)
    s.icon("scissors", cx + cw - 22, cut_y, 30, COLORS["red"][0], sw=2.4)
    s.chip(cx, zy + 298, "travel-policy#w120", "gray", size=17)
    s.box(cx, zy + 340, cw, 110, "gray", PAPER_WHITE, r=12, sw=2.4)
    mono_lines(s, cx + 16, zy + 370, [("300 | USD 75", None), (ROWS[2], "red"), (ROWS[3], None)], 16, 30, cw - 32)
    s.icon("circle-help", cx + 16, zy + 472, 30, COLORS["red"][0], sw=2.4)
    s.text(cx + 42, zy + 481, "220 or 60? No column header.", 21, 800, TEXT["red"], max_width=cw - 42)

    # --- B: structure-aware
    bx = M + zw + 20
    bw = W - M - bx
    s.box(bx, zy, bw, zh, "green", "wash", r=22, dashed=True, sw=3)
    s.icon("heading", bx + 36, zy + 40, 30, COLORS["green"][0], sw=2.3)
    s.text(bx + 62, zy + 50, "B · CUT AT HEADINGS", 23, 800, TEXT["green"])
    dx, dw = bx + 20, bw - 40
    s.chip(dx, zy + 76, "travel-policy#daily-limits-by-city", "gray", size=16)
    s.box(dx, zy + 118, dw, 332, "gray", PAPER_WHITE, r=12, sw=2.4)
    s.text(dx + 16, zy + 152, "Travel Policy > Daily limits by city", 19, 800, TEXT["teal"], max_width=dw - 30)
    s.text(dx + 16, zy + 184, "Hotel and meal costs are reimbursed …", 17, 600, TEXT["muted"], max_width=dw - 30)
    mono_lines(s, dx + 16, zy + 228, [(HEADER_ROW, "yellow"), (ROWS[0], None), (ROWS[1], None), (ROWS[2], "green"),
                                      (ROWS[3], None)], 16, 40, dw - 32)
    s.icon("circle-check", dx + 16, zy + 472, 30, COLORS["green"][0], sw=2.4)
    s.text(dx + 42, zy + 481, "USD 220 per night. One chunk.", 21, 800, TEXT["green"], max_width=dw - 42)

    # --- results: 13 questions as dots, in notebook order
    ry = 878
    s.text(M, ry, "13 test questions · Day 01's BM25, top 3", 24, 800, INK)
    s.text(M, ry + 30, "Is the complete evidence inside one retrieved chunk?", 21, 600, TEXT["muted"])
    rows = [("A", "red", [a for _, a, _ in RANKS], "9/13", "110 words read"),
            ("B", "green", [b for _, _, b in RANKS], "13/13", "113 words read")]
    for i, (label, color, ranks, score, words) in enumerate(rows):
        y = ry + 106 + i * 60
        s.text(M + 10, y + 10, label, 30, 800, tcolor(color))
        for j, rank in enumerate(ranks):
            cx_dot = M + 70 + j * 44
            if rank == "-":
                s.ellipse(cx_dot, y, 15, 15, "red", "wash", sw=2.6, jitter=0.8)
                s.icon("x", cx_dot, y, 20, COLORS["red"][0], sw=3)
            else:
                s.ellipse(cx_dot, y, 15, 15, "green", "fill", sw=2.6, jitter=0.8)
        s.text(M + 70 + 13 * 44 + 6, y + 10, score, 28, 800, tcolor(color))
        s.text(W - M, y + 10, words, 21, 700, TEXT["muted"], anchor="end")
    # brackets naming A's misses
    by = ry + 74
    s.line(M + 55, by, M + 70 + 2 * 44 + 15, by, COLORS["red"][0], sw=2.2, bow=0.2)
    s.text(M + 55, by - 9, "3 table questions", 18, 800, TEXT["red"])
    lx = M + 70 + 12 * 44
    s.text(lx + 15, by - 9, "spans 2 sections", 18, 800, TEXT["red"], anchor="end")
    s.line(lx - 15, by, lx + 15, by, COLORS["red"][0], sw=2.2, bow=0.2)

    s.bubble(M, 1090, W - 2 * M, ["Retrieval found the table both times. Only one chunk could answer."],
             "yellow", size=25, weight=700, tail=None, pad=22)
    s.text(M, 1214, "Real run · Day 03 notebook · free, no API key", 21, 600, TEXT["muted"])
    return s


# ----------------------------------------------------------------------------- slide 2
def rule_row(s: Slide, y: float, n: int, icon: str, color: str, title: str, note: str) -> None:
    s.box(M, y, W - 2 * M, 104, color, "wash", r=18, sw=2.6, jitter=1.1)
    s.badge(M + 34, y + 52, n, color, r=20, size=22)
    s.node(M + 118, y + 52, icon, color, None, size=76, icon_size=40)
    s.text(M + 176, y + 44, title, 26, 800, tcolor(color))
    s.text(M + 176, y + 80, note, 21, 600, TEXT["muted"], max_width=W - 2 * M - 196)


def piece_card(s: Slide, x: float, y: float, w: float, chunk_id: str, rows: list[str]) -> None:
    s.chip(x, y, chunk_id, "gray", size=16)
    h = 70 + len(rows) * 32
    s.box(x, y + 44, w, h, "gray", PAPER_WHITE, r=12, sw=2.4)
    s.text(x + 14, y + 78, "Travel Policy > Daily limits by city", 18, 800, TEXT["teal"], max_width=w - 24)
    mono_lines(s, x + 14, y + 112, [(r, "yellow" if r == HEADER_ROW else None) for r in rows], 16, 32, w - 28)


def slide_rules() -> Slide:
    s = Slide(2, TOTAL, DAY, "Cut along the structure", seed=32)
    heading(s, "Cut along the structure", "Four rules. They work on Markdown, HTML and PDF alike.")

    rules = [("heading", "blue", "SPLIT AT HEADINGS", "## starts a section · text before it becomes “Overview”"),
             ("layers", "violet", "KEEP UNITS WHOLE", "paragraphs, list items, whole tables · never cut mid-row"),
             ("package", "orange", "PACK UP TO A LIMIT", "80 words here · too long? split between units"),
             ("tag", "teal", "STAMP EVERY CHUNK", "Title > Section · stable ID · groups · hashes")]
    y = 286
    for i, (icon, color, title, note) in enumerate(rules):
        rule_row(s, y + i * 122, i + 1, icon, color, title, note)

    # a table that has to be split
    zy = 800
    s.box(M, zy, W - 2 * M, 436, "yellow", "wash", r=22, dashed=True, sw=3)
    s.icon("table", M + 38, zy + 40, 30, TEXT["orange"], sw=2.3)
    s.text(M + 66, zy + 50, "TABLE TOO LONG? SPLIT BETWEEN ROWS,", 23, 800, INK)
    s.text(M + 66, zy + 80, "REPEAT THE HEADER ROW", 23, 800, INK)
    s.text(W - M - 24, zy + 50, "real output, limit 30 words", 18, 700, TEXT["muted"], anchor="end")
    pw = (W - 2 * M - 60) / 2
    piece_card(s, M + 20, zy + 104, pw, "…#daily-limits-by-city-part-3", [HEADER_ROW] + ROWS[:3])
    piece_card(s, M + 40 + pw, zy + 104, pw, "…#daily-limits-by-city-part-4", [HEADER_ROW, ROWS[3]])
    s.text(M + 44 + pw, zy + 318, "Without the repeated header,", 21, 700, INK, max_width=pw - 10)
    s.text(M + 44 + pw, zy + 348, "“USD 180” would mean nothing.", 21, 700, INK, max_width=pw - 10)
    s.text(M + 20, zy + 416, "Parts 1–2: the paragraph above the table, split between sentences.", 19, 600,
           TEXT["muted"], max_width=W - 2 * M - 40)
    return s


# ----------------------------------------------------------------------------- slide 3
def slide_knobs() -> Slide:
    s = Slide(3, TOTAL, DAY, "Bigger isn't steadily better", seed=33)
    heading(s, "Bigger isn't steadily better", "Fixed windows, no overlap: complete evidence in one chunk, of 13.")

    # chart frame
    left, base, top = M + 140, 700, 340
    width = W - M - left - 10
    unit = (base - top) / 13
    s.line(left, base, left + width, base, INK, sw=3, bow=0.2)
    s.line(left, base, left, top - 20, INK, sw=3, bow=0.2)
    for v in (0, 5, 10, 13):
        s.text(left - 14, base - v * unit + 8, str(v), 20, 700, TEXT["muted"], anchor="end")
    # structure-aware reference line
    ref_y = base - 13 * unit
    s.line(left, ref_y, left + width, ref_y, COLORS["green"][0], sw=2.8, dashed=True, bow=0.1)
    s.text(left + width, ref_y - 14, "structure-aware: 13/13 at 113 words read", 20, 800, TEXT["green"],
           anchor="end")
    slot = width / len(SWEEP)
    bar_w = slot * 0.6
    for i, (size, _chunks, complete, words, kept) in enumerate(SWEEP):
        cx = left + slot * (i + 0.5)
        color = "blue" if kept == "yes" else "gray"
        s.box(cx - bar_w / 2, base - complete * unit, bar_w, complete * unit, color, "fill", r=6, sw=2.4, jitter=0.9)
        s.text(cx, base - complete * unit + 30, str(complete), 22, 800, INK, anchor="middle", max_width=bar_w)
        s.text(cx, base + 34, str(size), 22, 800, INK, anchor="middle", max_width=slot)
        s.icon("check" if kept == "yes" else "x", cx, base + 72, 26,
               COLORS["green"][0] if kept == "yes" else COLORS["red"][0], sw=3)
        s.text(cx, base + 118, str(words), 20, 700, TEXT["muted"], anchor="middle", max_width=slot)
    for y, label in ((base + 34, "window"), (base + 80, "table kept"), (base + 118, "words read")):
        s.text(M, y, label, 20, 800, INK if label == "window" else TEXT["muted"], max_width=130)
    s.text(M, base + 160, "Blue bars and ✓: the Singapore row stays in one chunk with its column header.", 19, 600,
           TEXT["muted"], max_width=W - 2 * M)

    # two knobs
    ky = 880
    cards = [("copy", "orange", "OVERLAP (40 + 10 words)",
              ["Repaired 3 of 4 misses", "+27% text to embed and store", "Software table still headerless"]),
             ("heading", "violet", "“TITLE > SECTION” HEADER",
              ["Helped 2 questions", "Hurt 1: parental leave 2 → 3", "+18 words read per question"])]
    cw = (W - 2 * M - 20) / 2
    for i, (icon, color, title, lines) in enumerate(cards):
        x = M + i * (cw + 20)
        s.box(x, ky, cw, 230, color, "wash", r=20, sw=2.8)
        s.icon(icon, x + 38, ky + 42, 32, icon_color(color), sw=2.3)
        s.text(x + 66, ky + 52, title, 22, 800, tcolor(color), max_width=cw - 80)
        for j, line in enumerate(lines):
            s.text(x + 24, ky + 108 + j * 40, "• " + line, 21, 700, INK, max_width=cw - 40)
    s.text(M, 1176, "Fixed cuts land by luck. Measure the knobs; don't assume.", 24, 800, TEXT["teal"])
    return s


# ----------------------------------------------------------------------------- slide 4
def slide_updates() -> Slide:
    s = Slide(4, TOTAL, DAY, "One edit: how many chunks change?", seed=34)
    heading(s, "One edit: how many chunks change?", "Add 7 words to the top of travel-policy, then re-chunk.", size=48)

    # A: every window shifts
    ay = 290
    s.text(M, ay + 18, "A · FIXED 40 WORDS", 23, 800, TEXT["red"])
    s.text(W - M, ay + 18, "6 of 6 changed", 23, 800, TEXT["red"], anchor="end")
    tw = (W - 2 * M - 5 * 14) / 6
    for i, name in enumerate(FIXED_IDS):
        x = M + i * (tw + 14)
        s.box(x, ay + 40, tw, 84, "orange", "fill", r=14, sw=2.6)
        s.icon("refresh-cw", x + 28, ay + 82, 22, COLORS["orange"][0], sw=2.4)
        s.text(x + 44, ay + 90, name, 21, 600, INK, family="mono", max_width=tw - 50)
    s.arrow(M + 30, ay + 152, M + 260, ay + 152, COLORS["orange"][0], sw=3, head=14)
    s.text(M + 276, ay + 160, "the 7 new words push every boundary along", 21, 700, INK)
    s.text(M, ay + 200, "Re-embed all six. [travel-policy#w080] now cites different text.", 21, 600, TEXT["muted"])

    # B: one section changes
    by = 540
    s.text(M, by + 18, "B · STRUCTURE-AWARE", 23, 800, TEXT["green"])
    s.text(W - M, by + 18, "1 of 5 changed", 23, 800, TEXT["green"], anchor="end")
    sw_ = (W - 2 * M - 4 * 14) / 5
    for i, (name, changed) in enumerate(SECTIONS):
        x = M + i * (sw_ + 14)
        s.box(x, by + 40, sw_, 84, "orange" if changed else "green", "fill", r=14, sw=2.6)
        s.icon("refresh-cw" if changed else "check", x + 26, by + 82, 22,
               COLORS["orange"][0] if changed else COLORS["green"][0], sw=2.6)
        s.text(x + 44, by + 90, name, 21, 700, INK, max_width=sw_ - 50)
    s.icon("refresh-cw", M + 26, by + 158, 26, COLORS["orange"][0], sw=2.4)
    s.text(M + 50, by + 166, "Only the edited section needs re-embedding.", 21, 700, INK)
    s.text(M, by + 204, "Section IDs such as travel-policy#daily-limits-by-city keep their meaning.", 20, 600,
           TEXT["muted"], max_width=W - 2 * M)

    # metadata travels with every chunk
    zy = 800
    s.box(M, zy, W - 2 * M, 344, "blue", "wash", r=22, dashed=True, sw=3)
    s.icon("lock-keyhole", M + 38, zy + 42, 30, COLORS["blue"][0], sw=2.3)
    s.text(M + 66, zy + 52, "EVERY CHUNK CARRIES ITS DOCUMENT'S METADATA", 23, 800, TEXT["blue"])
    fields = [("chunk_id", "travel-policy#daily-limits-by-city"), ("allowed_groups", "all-employees"),
              ("doc_hash", DOC_HASH[:12] + "…"), ("chunk_hash", CHUNK_HASH[:12] + "…")]
    fx, fw = M + 24, 560
    s.box(fx, zy + 80, fw, 196, "gray", PAPER_WHITE, r=12, sw=2.4)
    for i, (key, value) in enumerate(fields):
        y = zy + 122 + i * 42
        s.text(fx + 18, y, key, 18, 700, TEXT["blue"], family="mono")
        s.text(fx + fw - 18, y, value, 18, 500, INK, family="mono", anchor="end", max_width=fw - 190)
    nx = fx + fw + 26
    s.paragraph(nx, zy + 116, "Manager pay guide: 1 chunk, groups managers + people-team.", 20, 700,
                W - M - nx - 20, 27, INK, max_lines=3)
    s.paragraph(nx, zy + 214, "Filters need this on every chunk (Day 05).", 20, 700, W - M - nx - 20, 27,
                TEXT["blue"], max_lines=2)
    s.text(M + 24, zy + 314, "Recording permissions is not enforcing them: that is Day 14.", 20, 600,
           TEXT["muted"], max_width=W - 2 * M - 48)
    s.text(M, 1214, "Next → Day 04: embeddings and similarity", 26, 800, TEXT["teal"])
    return s


def main() -> None:
    check_contrast()
    check_facts()
    OUT.mkdir(parents=True, exist_ok=True)
    names = ["slide-01-cover", "slide-02-how-it-works", "slide-03-knobs", "slide-04-updates"]
    for name, build in zip(names, [slide_cover, slide_rules, slide_knobs, slide_updates]):
        path = OUT / f"{name}.svg"
        path.write_text(build().svg(), encoding="utf-8")
        print("wrote", path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
