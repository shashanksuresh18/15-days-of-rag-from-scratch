"""Build the Day 02 LinkedIn carousel (whiteboard style) as editable SVGs in assets/day-02/.

    python scripts/slides/day02.py                    # writes the SVGs
    python scripts/slides/render.py assets/day-02     # renders PNGs + PDF

Every number, ID and extracted line on the slides comes from the executed Day 02 notebook.
`check_facts()` confirms each one appears in the notebook's saved outputs before anything is
written, so the slides cannot silently drift from the code.
"""

from __future__ import annotations

import json
from pathlib import Path

from kit import COLORS, INK, M, TEXT, W, Slide, check_contrast, measure

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "day-02"
NOTEBOOK = ROOT / "notebooks" / "day_02_ingestion_and_metadata.ipynb"
TOTAL = 4
DAY = "02"

HASH = "bf4bdf9b29cecca37f3f45b565a210bcda1f096d2a6be1fe5ca44d2ae7a90484"   # travel-policy body
HASH_500, HASH_600 = "9d0e7b74370d6210", "e6e17cd1b3d6f8db"                 # section 4 demo
TABLE = [("City", "Hotel per night", "Meals per day"),
         ("London", "USD 250", "USD 75"),
         ("New York", "USD 300", "USD 75"),
         ("Singapore", "USD 220", "USD 60"),
         ("Other cities", "USD 180", "USD 50")]
PLAIN_ORDER = [row[c] for c in range(3) for row in TABLE]                      # column by column
PAPER_WHITE = "#FFFFFF"

FACTS = [
    "13 files -> 11 documents + 2 quarantined",
    "no entry in the permission export: fail closed (visible to nobody)",
    "no extractable text (1 page, 1 image, 0 characters of text): scanned? run OCR first",
    "'Audience: all staff (please share widely)'",
    "permission export says: None",
    "review overdue: last reviewed 470 days before the snapshot",
    "managers, people-team",
    "pypdf 6.20.1 (layout mode)",
    f'"content_hash": "{HASH}"',
    '"owner": "Finance"', '"last_reviewed": "2026-04-15"', '"title": "Travel Policy"',
    "\n".join(PLAIN_ORDER),                                       # plain extraction, column order
    "\n".join(" | ".join(row) for row in TABLE),                  # ingested body rows
    "claim reim-\nbursement",
    "PDF   phrase 'claim reimbursement' is findable   False      True",
    "PDF   header/footer lines in the text                4         0",
    "PDF   'Singapore' and 'USD 220' on one line      False      True",
    "HTML  cookie-banner text present                  True     False",
    "HTML  words of text                                202       145",
    f"{HASH_500}…  Employees receive a home office allowance of USD 500.",
    f"{HASH_600}…  Employees receive a home office allowance of USD 600.",
    "added            annual-leave-policy",
    "content changed  remote-work-policy",
    "metadata changed password-and-mfa-standard",
    "removed          vpn-access-guide",
    "unchanged        8 documents",
    "file bytes changed = True, classified as unchanged",
    "['all-employees'] -> 10 documents",
    "['all-employees', 'managers'] -> 11 documents",
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


def file_tile(s: Slide, x: float, y: float, w: float, h: float, icon: str, color: str, label: str,
              size: int = 23) -> None:
    s.box(x, y, w, h, color, "fill", r=14, sw=2.6, jitter=1.0)
    s.icon(icon, x + 34, y + h / 2, 30, icon_color(color), sw=2.2)
    s.text(x + 62, y + h / 2 + size * 0.36, label, size, 700, INK, max_width=w - 72)


def step_box(s: Slide, x: float, y: float, w: float, h: float, n: int, color: str, title: str, note: str) -> None:
    s.box(x, y, w, h, color, "wash", r=16, sw=2.8)
    s.badge(x + 30, y + h / 2, n, color, r=18, size=21)
    s.text(x + 56, y + 32, title, 25, 800, tcolor(color), max_width=w - 62)
    s.text(x + 56, y + 60, note, 18, 600, TEXT["muted"], max_width=w - 62)


def field_row(s: Slide, x: float, y: float, key: str, value: str, color: str, width: float) -> None:
    s.text(x, y, key, 18, 700, tcolor(color), family="mono")
    s.text(x + width, y, value, 18, 500, INK, family="mono", anchor="end", max_width=width - 120)


# ----------------------------------------------------------------------------- slide 1
def slide_cover() -> Slide:
    s = Slide(1, TOTAL, DAY, "Messy files in, trusted records out", seed=21)
    heading(s, "Messy files in, trusted records out",
            "Day 02: ingestion, parsing and metadata, before any retrieval.", size=48)

    # Column 1: what arrived
    top = 336
    s.tag(M, 312, "13 FILES ARRIVE", "gray", size=21)
    tiles = [("file-text", "green", "8 × Markdown"), ("file-code", "blue", "1 × HTML page"),
             ("file-type", "orange", "1 × PDF"), ("file-image", "gray", "1 × scanned PDF"),
             ("file", "yellow", "2 × plain text")]
    for i, (icon, color, label) in enumerate(tiles):
        file_tile(s, M, top + 18 + i * 78, 266, 64, icon, color, label)

    # Column 2: the pipeline
    s.tag(378, 312, "ONE PIPELINE", "violet", size=21)
    steps = [("violet", "PARSE", "one adapter per format"), ("orange", "CLEAN", "headers, hyphens, tables"),
             ("red", "CHECK", "permissions + metadata"), ("blue", "STAMP", "ID, hash, parser")]
    for i, (color, title, note) in enumerate(steps):
        y = top + 18 + i * 98
        step_box(s, 376, y, 298, 80, i + 1, color, title, note)
        if i < 3:
            s.arrow(525, y + 82, 525, y + 96, INK, sw=2.6, head=10)
    s.arrow(334, 540, 372, 540, INK, sw=3.4)

    # Column 3: one real record
    cx, cy, cw, ch = 712, top + 18, 308, 376
    s.tag(cx, 312, "11 RECORDS", "green", size=21)
    s.arrow(680, 540, 707, 540, INK, sw=3.2, head=14)
    s.box(cx, cy, cw, ch, "gray", PAPER_WHITE, r=16, sw=2.8)
    s.text(cx + 18, cy + 40, "travel-policy", 22, 700, INK, family="mono")
    s.line(cx + 16, cy + 56, cx + cw - 16, cy + 57, COLORS["gray"][0], sw=2, bow=0.3)
    groups = [("blue", "WHAT", [("title", "Travel Policy"), ("owner", "Finance"), ("reviewed", "2026-04-15")]),
              ("red", "WHO", [("groups", "all-employees")]),
              ("violet", "WHERE", [("parser", "pypdf 6.20.1"), ("sha256", HASH[:12] + "…")])]
    y = cy + 94
    for color, label, rows in groups:
        s.text(cx + 18, y, label, 19, 800, tcolor(color))
        y += 27
        for key, value in rows:
            field_row(s, cx + 18, y, key, value, color, cw - 36)
            y += 27
        y += 12

    # Quarantine
    qy = 790
    s.box(M, qy, W - 2 * M, 200, "red", "wash", r=22, dashed=True, sw=3)
    s.icon("archive-x", M + 40, qy + 40, 34, COLORS["red"][0], sw=2.3)
    s.text(M + 70, qy + 50, "2 QUARANTINED, each with a reason", 27, 800, TEXT["red"])
    s.arrow(478, top + 18 + 3 * 98 + 84, 478, qy - 6, COLORS["red"][0], sw=3, head=14)
    s.text(494, qy - 16, "fails a check", 20, 700, TEXT["red"])
    rows = [("file-image", "scanned-visitor-policy.pdf", "no text layer: needs OCR"),
            ("file", "draft-reorg-announcement.txt", "no permission entry: fail closed")]
    for i, (icon, name, reason) in enumerate(rows):
        y = qy + 100 + i * 56
        s.icon(icon, M + 40, y, 28, COLORS["red"][0], sw=2.2)
        w = s.chip(M + 66, y - 20, name, "red", size=18)
        s.text(M + 66 + w + 14, y + 8, "→ " + reason, 22, 700, INK)

    # Takeaway
    by = 1036
    s.bubble(M, by, W - 2 * M, ["Retrieval can only find what ingestion kept.",
                                "So check what goes in, and say what was left out."],
             "yellow", size=27, weight=700, tail=None, pad=24, line_height=40)
    s.text(M, 1214, "Real run · Day 02 notebook · free, no API key", 21, 600, TEXT["muted"])
    return s


# ----------------------------------------------------------------------------- slide 2
def zone(s: Slide, y: float, h: float, color: str, icon: str, title: str, note: str) -> None:
    s.box(M, y, W - 2 * M, h, color, "wash", r=24, dashed=True, sw=3)
    s.node(M + 78, y + 104, icon, color, None, size=100, icon_size=52)
    s.text(M + 152, y + 56, title, 28, 800, tcolor(color))
    s.text(M + 152, y + 90, note, 21, 600, TEXT["muted"])


def chips(s: Slide, x: float, y: float, labels: list[str], color: str, gap: float = 10) -> float:
    for label in labels:
        x += s.chip(x, y, label, color, size=19) + gap
    return x


def slide_origins() -> Slide:
    s = Slide(2, TOTAL, DAY, "Where each field comes from", seed=22)
    heading(s, "Where each field comes from", "Who can change a field decides how far you can trust it.")

    # 1. The document
    y1, h1 = 276, 268
    zone(s, y1, h1, "blue", "file-pen-line", "THE DOCUMENT ITSELF", "front matter · <meta> tags · PDF properties")
    chips(s, M + 152, y1 + 116, ["title", "owner", "last_reviewed"], "blue")
    s.icon("triangle-alert", M + 166, y1 + 200, 26, COLORS["orange"][0], sw=2.3)
    s.text(M + 190, y1 + 208, "Any author can edit it. Validate it.", 22, 700, INK)
    s.icon("calendar-clock", M + 166, y1 + 240, 26, COLORS["orange"][0], sw=2.3)
    s.text(M + 190, y1 + 248, "e.g. intranet page: review overdue (470 days)", 21, 600, TEXT["muted"])

    # 2. The system of record
    y2, h2 = y1 + h1 + 20, 386
    zone(s, y2, h2, "red", "server", "THE SYSTEM OF RECORD", "the permission export, not the file")
    chips(s, M + 152, y2 + 116, ["allowed_groups"], "red")
    s.text(M + 152 + measure("allowed_groups", 19, 500, "mono") + 46, y2 + 140,
           "authoritative · missing = fail closed", 21, 700, TEXT["red"])
    # the draft that asks to be shared
    s.text(M + 30, y2 + 210, "A dropped-in draft says:", 21, 700, TEXT["muted"])
    s.bubble(M + 30, y2 + 226, 470, ["“Audience: all staff", "(please share widely)”"], "yellow", size=23,
             weight=700, tail=None, pad=16, line_height=31)
    s.text(M + 548, y2 + 210, "The export says:", 21, 700, TEXT["muted"])
    s.box(M + 548, y2 + 226, 196, 92, "gray", "fill", r=16, sw=2.6)
    s.text(M + 646, y2 + 282, "no entry", 25, 800, INK, anchor="middle")
    s.arrow(M + 752, y2 + 272, M + 794, y2 + 272, INK, sw=3)
    s.node(M + 850, y2 + 270, "eye-off", "red", None, size=96, icon_size=50)
    s.text(M + 850, y2 + 352, "visible to nobody", 20, 800, TEXT["red"], anchor="middle")

    # 3. The pipeline
    y3, h3 = y2 + h2 + 20, 1240 - (y2 + h2 + 20)
    zone(s, y3, h3, "violet", "cog", "THE PIPELINE", "added by your code on every run")
    x = chips(s, M + 152, y3 + 116, ["doc_id", "source_path", "parser", "content_hash"], "violet")
    s.text(M + 152, y3 + 208, "Traces any answer to one file, one parser", 22, 700, INK)
    s.text(M + 152, y3 + 238, "version and one exact version of the text.", 22, 700, INK)
    return s


# ----------------------------------------------------------------------------- slide 3
def draw_table(s: Slide, x: float, y: float, col_w: list[float], row_h: float, highlight: int | None) -> None:
    width = sum(col_w)
    n = len(TABLE)
    s.box(x, y, width, row_h, "gray", "fill", r=8, sw=2.2, jitter=0.6)
    if highlight is not None:
        s.box(x, y + highlight * row_h, width, row_h, "green", "fill", r=8, sw=2.2, jitter=0.6)
    for r in range(1, n):
        s.line(x, y + r * row_h, x + width, y + r * row_h, COLORS["gray"][0], sw=2, bow=0.2)
    cx = x
    for c in col_w[:-1]:
        cx += c
        s.line(cx, y, cx, y + n * row_h, COLORS["gray"][0], sw=2, bow=0.2)
    s.box(x, y, width, n * row_h, "gray", None, r=10, sw=2.8)
    for r, row in enumerate(TABLE):
        cx = x
        for c, cell in enumerate(row):
            s.text(cx + 14, y + r * row_h + row_h * 0.66, cell, 22, 800 if r == 0 else 600, INK,
                   max_width=col_w[c] - 20)
            cx += col_w[c]


def slide_pdf() -> Slide:
    s = Slide(3, TOTAL, DAY, "PDFs store ink, not tables", seed=23)
    heading(s, "PDFs store ink, not tables", "One page of travel-policy.pdf, extracted two ways by pypdf 6.20.1.")

    # What you see
    s.tag(M, 312, "WHAT YOU SEE", "gray", size=21)
    draw_table(s, M + 140, 334, [220, 230, 230], 44, highlight=3)

    # Default extraction: one cell per line, column by column
    zy, zh = 580, 470
    s.box(M, zy, 440, zh, "red", "wash", r=22, dashed=True, sw=3)
    s.text(M + 24, zy + 46, "DEFAULT (“plain”) MODE", 24, 800, TEXT["red"])
    s.text(M + 24, zy + 76, "follows the drawing order", 20, 600, TEXT["muted"])
    col_colors = ["green", "blue", "orange"]
    x, y, right = M + 24, zy + 100, M + 440 - 20
    for i, cell in enumerate(PLAIN_ORDER):
        w = measure(cell, 16, 500, "mono") + 26
        if x + w > right:
            x, y = M + 24, y + 48
        s.chip(x, y, cell, col_colors[i // 5], size=16)
        x += w + 8
    s.paragraph(M + 24, zy + 384, "One stream: all cities, then all hotel limits, then all meal limits.",
                20, 600, 392, 26, INK, max_lines=3)

    # Layout mode + cleaning
    gx = M + 470
    s.box(gx, zy, W - M - gx, zh, "green", "wash", r=22, dashed=True, sw=3)
    s.text(gx + 24, zy + 46, "LAYOUT MODE + CLEANING", 24, 800, TEXT["green"])
    s.text(gx + 24, zy + 76, "places text by position", 20, 600, TEXT["muted"])
    for r, row in enumerate(TABLE):
        line = " | ".join(row)
        color = "green" if row[0] == "Singapore" else "gray"
        s.chip(gx + 24, zy + 100 + r * 50, line, color, size=16)
    s.icon("circle-check", gx + 40, zy + 384, 30, COLORS["green"][0], sw=2.4)
    s.text(gx + 66, zy + 392, "Rows survive: Singapore → USD 220", 20, 700, TEXT["green"],
           max_width=W - M - gx - 96)

    # Two more traps
    ty = 1080
    traps = [("scissors", "orange", "claim reim- / bursement", "re-joined → phrase found"),
             ("eraser", "violet", "header + footer on each page", "4 lines → 0")]
    for i, (icon, color, what, fix) in enumerate(traps):
        x = M + i * 490
        s.box(x, ty, 470, 150, color, "wash", r=20, sw=2.8)
        s.icon(icon, x + 40, ty + 44, 34, COLORS[color][0], sw=2.3)
        s.text(x + 72, ty + 54, what, 23, 800, tcolor(color), max_width=380)
        s.text(x + 72, ty + 100, fix, 22, 700, INK, max_width=380)
    return s


# ----------------------------------------------------------------------------- slide 4
def slide_freshness() -> Slide:
    s = Slide(4, TOTAL, DAY, "Same document, new version", seed=24)
    heading(s, "Same document, new version", "Two IDs, two questions. Compare them on every run.")

    # identity vs version
    y = 286
    s.box(M, y, 450, 210, "blue", "wash", r=22, sw=2.8)
    s.icon("tag", M + 40, y + 44, 32, COLORS["blue"][0], sw=2.3)
    s.text(M + 70, y + 54, "WHICH document?", 26, 800, TEXT["blue"])
    w = s.text(M + 24, y + 98, "doc_id", 21, 700, TEXT["blue"], family="mono")
    s.text(M + 24 + w + 12, y + 98, "stays when the text changes", 21, 600, TEXT["muted"])
    s.chip(M + 24, y + 128, "remote-work-policy", "blue", size=19)

    x2 = M + 480
    s.box(x2, y, W - M - x2, 210, "violet", "wash", r=22, sw=2.8)
    s.icon("fingerprint-pattern", x2 + 40, y + 44, 32, COLORS["violet"][0], sw=2.3)
    s.text(x2 + 70, y + 54, "WHICH version?", 26, 800, TEXT["violet"])
    w = s.text(x2 + 24, y + 98, "content_hash", 21, 700, TEXT["violet"], family="mono")
    s.text(x2 + 24 + w + 12, y + 98, "changes with 1 character", 21, 600, TEXT["muted"])
    s.text(x2 + 24, y + 150, "USD 500", 20, 700, INK)
    s.chip(x2 + 130, y + 126, HASH_500 + "…", "violet", size=17)
    s.text(x2 + 24, y + 190, "USD 600", 20, 700, INK)
    s.chip(x2 + 130, y + 166, HASH_600 + "…", "violet", size=17)

    # the five outcomes
    s.text(M, 556, "Real run: 5 changes to a copy of the data, then re-ingest", 24, 800, INK)
    rows = [("file-plus", "green", "ADDED", "annual-leave-policy", "chunk + index"),
            ("file-pen-line", "orange", "TEXT CHANGED", "remote-work-policy", "re-chunk + re-index"),
            ("lock-keyhole", "violet", "PERMISSIONS CHANGED", "password-and-mfa-standard +contractors", "update metadata only"),
            ("file-x", "red", "REMOVED", "vpn-access-guide", "DELETE its chunks"),
            ("file-check", "gray", "UNCHANGED", "8 docs (one re-saved)", "skip")]
    y = 584
    for icon, color, kind, example, action in rows:
        s.box(M, y, W - 2 * M, 96, color, "wash", r=18, sw=2.6, jitter=1.1)
        s.icon(icon, M + 40, y + 48, 34, icon_color(color), sw=2.3)
        s.text(M + 74, y + 40, kind, 22, 800, tcolor(color))
        s.text(M + 74, y + 74, example, 19, 500, INK, family="mono")
        s.text(W - M - 24, y + 58, "→ " + action, 22, 800 if color == "red" else 700,
               TEXT["red"] if color == "red" else INK, anchor="end", max_width=330)
        y += 108

    s.paragraph(M, 1150, "One file was re-saved with Windows line endings: new bytes, same text, so it is skipped.",
                20, 600, W - 2 * M, 26, TEXT["muted"], max_lines=2)
    s.text(M, 1226, "Next → Day 03: chunking and document structure", 26, 800, TEXT["teal"])
    return s


def main() -> None:
    check_contrast()
    check_facts()
    OUT.mkdir(parents=True, exist_ok=True)
    names = ["slide-01-cover", "slide-02-metadata-origins", "slide-03-pdf-tables", "slide-04-freshness"]
    for name, build in zip(names, [slide_cover, slide_origins, slide_pdf, slide_freshness]):
        path = OUT / f"{name}.svg"
        path.write_text(build().svg(), encoding="utf-8")
        print("wrote", path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
