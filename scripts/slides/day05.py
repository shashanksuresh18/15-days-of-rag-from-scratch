"""Build the Day 05 LinkedIn carousel (whiteboard style) as editable SVGs in assets/day-05/.

    python scripts/slides/day05.py                    # writes the SVGs
    python scripts/slides/render.py assets/day-05     # renders PNGs + PDF

Every number and ID on the slides comes from the executed Day 05 notebook. `check_facts()` confirms
each one appears in the notebook's saved outputs before anything is written; the timing chart is read
from those outputs directly, and the library install sizes are checked against the table in
docs/day-05/LEARN.md, where they are documented with their date and platform.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from kit import COLORS, INK, M, TEXT, W, Slide, check_contrast

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "day-05"
NOTEBOOK = ROOT / "notebooks" / "day_05_vector_search_and_lifecycle.ipynb"
LEARN = ROOT / "docs" / "day-05" / "LEARN.md"
TOTAL = 4
DAY = "05"
PAPER_WHITE = "#FFFFFF"

PAY_Q = "When do pay rises take effect?"
VPN_Q = "How do I connect to the VPN?"
RESTRICTED = "manager-pay-review-guide#overview"
PLAN = [("add", 2, "green", "new policy · renamed heading"),
        ("re-embed", 1, "orange", "USD 500 → USD 600"),
        ("update metadata", 11, "blue", "contractors added · new doc version"),
        ("delete", 5, "red", "retired VPN guide · old heading"),
        ("unchanged", 27, "gray", "incl. a line-ending-only re-save")]
UPSERT_VPN = ["#troubleshooting", "#connecting", "#when-to-use-the-vpn"]
SYNC_VPN = [("0.71", "remote-work-policy#security-while-remote"), ("0.58", "new-starter-it-checklist#your-first-day"),
            ("0.57", "password-and-mfa-standard#multi-factor-authentication")]
# (package, MB installed on top of NumPy, packages added), from LEARN.md section 9
LIBRARIES = [("sqlite-vec", 0.2, 1), ("voyager", 0.8, 1), ("usearch", 41, 3), ("faiss-cpu", 64, 2),
             ("chromadb", 311, 83), ("lancedb", 354, 15)]

FACTS = [
    "11 documents -> 44 chunks",
    f"#1 0.70  [{RESTRICTED}]   <- only managers, people-team",
    "  employee, filter applied\n    #1 0.59  [expense-reimbursement-policy#approval-and-payment]",
    "  1. the filter is forgotten (where=None)\n    #1 0.70  [manager-pay-review-guide#overview]",
    "  2. an employee's request says groups = managers\n    #1 0.70  [manager-pay-review-guide#overview]",
    "  the permission changed, the index hasn't been synced yet\n    #1 0.71  [travel-policy#daily-limits-by-city]",
    "sync: 5 records updated, 0 texts embedded",
    "index.npz: 44 chunk texts and 44 vectors",
    "mallory: When do pay rises take effect?\n    (no results)",
    "post-filter: 2 of 3 results", "post-filter: 0 of 3 results", "pre-filter: 3 of 3 results",
    "    #1 0.59  [expense-reimbursement-policy#approval-and-payment]\n    #2 0.58  [parental-leave-policy#returning-to-work]\n"
    "    #3 0.57  [expense-reimbursement-policy#submitting-a-claim]",
    "    #1 0.80  [vpn-access-guide#troubleshooting]\n    #2 0.76  [vpn-access-guide#connecting]\n"
    "    #3 0.74  [vpn-access-guide#when-to-use-the-vpn]",
    "    #1 0.71  [remote-work-policy#security-while-remote]\n    #2 0.51  [remote-work-policy#core-hours]",
    "44 rows | vectors (44, 384) float32, 67,584 bytes | model BAAI/bge-small-en-v1.5 @ 5c38ec7c405e",
    "  chunk_hash     146c27100b941450...",
    "saved 44 chunks to outputs/day-05/index.npz (99,192 bytes):",
    *(f"{name:<16}{n:>3}" for name, n, _, _ in PLAN),
    "metadata updates by field: {'allowed_groups': 4, 'doc_hash': 7}",
    "texts sent through the model: 3 (a rebuild would embed all 41)",
    "upsert only: 46 chunks | full sync: 41 chunks | the documents: 41 chunks",
    *(f"[vpn-access-guide{label}]   <- no longer in any document" for label in UPSERT_VPN),
    *(f"  full sync\n    #1 {SYNC_VPN[0][0]}  [{SYNC_VPN[0][1]}]\n    #2 {SYNC_VPN[1][0]}  [{SYNC_VPN[1][1]}]\n"
      f"    #3 {SYNC_VPN[2][0]}  [{SYNC_VPN[2][1]}]",),
    "largest vector difference 0.0e+00",
    "refusing to delete 44 of 44 chunks",
    "the file on disk is broken: EOFError: No data left in file",
    "the file on disk still loads: 44 chunks",
]
LEARN_FACTS = [f"| `{name}` (" for name, _, _ in LIBRARIES] + \
              [f"| +{mb:g} MB, {n} |" for _, mb, n in LIBRARIES] + \
              ["the 2.2.2 Linux wheel for Python 3.12+ didn't import on Python 3.13"]


def notebook_outputs() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code"
                   for o in c.get("outputs", []))


def scale_table(outputs: str) -> tuple[list[tuple[int, float, float]], float]:
    # The section 7 measurements, exactly as saved: [(chunks, MB, ms)], embedding ms
    rows = [(int(n.replace(",", "")), float(mb), float(ms)) for n, mb, ms in
            re.findall(r"^\s+([\d,]+)\s+([\d.]+) MB\s+([\d.]+) ms$", outputs, re.M)]
    embed = float(re.search(r"embedding one question with the model: ([\d.]+) ms", outputs).group(1))
    if [n for n, _, _ in rows] != [44, 1_000, 10_000, 100_000]:
        raise SystemExit(f"Unexpected scale table in the notebook: {rows}")
    return rows, embed


def check_facts() -> None:
    outputs, learn = notebook_outputs(), LEARN.read_text(encoding="utf-8")
    missing = [f for f in FACTS if f not in outputs] + [f for f in LEARN_FACTS if f not in learn]
    if missing:
        raise SystemExit(f"Slide facts not found (re-run the notebook?): {missing}")
    if "All Day 05 sanity checks passed." not in outputs:
        raise SystemExit("The saved notebook did not pass its sanity checks")


def heading(s: Slide, title: str, subtitle: str, size: int = 54) -> None:
    s.text(M, 190, title, size, 800, INK)
    s.text(M, 238, subtitle, 25, 500, TEXT["muted"])


def tcolor(color: str) -> str:
    return TEXT.get(color, INK) if color != "yellow" else INK


def icon_color(color: str) -> str:
    return COLORS[color][0] if color != "yellow" else TEXT["orange"]


# ----------------------------------------------------------------------------- slide 1
def failure_card(s: Slide, x: float, y: float, w: float, h: float, n: int, icon: str, title: str,
                 code: str | None, lines: list[str], outcome: str) -> None:
    s.box(x, y, w, h, "red", "wash", r=20, sw=2.8, jitter=1.2)
    s.badge(x + 32, y + 38, n, "red", r=19, size=22)
    s.icon(icon, x + 78, y + 38, 30, COLORS["red"][0], sw=2.3)
    s.text(x + 104, y + 47, title, 22, 800, TEXT["red"], max_width=w - 120)
    ty = y + 92
    if code:
        s.text(x + 24, ty, code, 18, 500, INK, family="mono", max_width=w - 48)
        ty += 34
    for line in lines:
        s.text(x + 24, ty, line, 20, 600, TEXT["muted"], max_width=w - 48)
        ty += 30
    s.icon("circle-x", x + 38, y + h - 34, 28, COLORS["red"][0], sw=2.4)
    s.text(x + 62, y + h - 26, outcome, 21, 800, INK, max_width=w - 80)


def slide_cover() -> Slide:
    s = Slide(1, TOTAL, DAY, "A filter isn't access control", seed=51)
    heading(s, "A filter isn't access control", "Day 05: a local vector index, and one managers-only chunk.")

    # the restricted chunk
    y0 = 276
    s.box(M, y0, W - 2 * M, 176, "violet", "wash", r=22, sw=2.8)
    s.node(M + 74, y0 + 88, "lock-keyhole", "violet", None, size=100, icon_size=54)
    s.text(M + 146, y0 + 48, f"Q: {PAY_Q}", 24, 800, INK, max_width=W - 2 * M - 170)
    s.chip(M + 146, y0 + 66, RESTRICTED, "violet", size=19)
    s.text(W - M - 24, y0 + 94, "#1 · 0.70", 22, 800, TEXT["violet"], anchor="end", max_width=140)
    s.icon("circle-check", M + 162, y0 + 140, 26, COLORS["green"][0], sw=2.4)
    s.text(M + 184, y0 + 148, "only managers, people-team · employee + filter: not returned",
           20, 700, TEXT["green"], max_width=W - 2 * M - 210)

    # four failures
    s.text(M, 500, "Same index, same question. Four ways it leaks:", 24, 800, INK)
    cw, ch, gap = (W - 2 * M - 20) / 2, 262, 20
    cards = [
        (1, "eraser", "FILTER FORGOTTEN", "search(q)  # where=None",
         ["a debug path, a widget, an eval:", "no error, results look normal"], "pay guide back at #1"),
        (2, "user", "GROUPS FROM THE REQUEST", "groups = ['managers']",
         ["an employee claims a group:", "same call as a manager"], "pay guide back at #1"),
        (3, "calendar-clock", "PERMISSIONS STALE", None,
         ["travel policy made managers-only,", "index not synced yet", "(the fix: 5 records, 0 re-embeds)"],
         "employee still gets it"),
        (4, "file-lock", "THE FILE ITSELF", "np.load('index.npz')",
         ["all 44 texts, no filter runs,", "plus vectors that can leak text"], "pay guide text readable"),
    ]
    for i, (n, icon, title, code, lines, outcome) in enumerate(cards):
        x = M + (i % 2) * (cw + gap)
        y = 526 + (i // 2) * (ch + gap)
        failure_card(s, x, y, cw, ch, n, icon, title, code, lines, outcome)

    # what makes it hold
    zy = 1092
    s.box(M, zy, W - 2 * M, 146, "green", "wash", r=22, dashed=True, sw=3)
    s.icon("shield-check", M + 36, zy + 38, 30, COLORS["green"][0], sw=2.3)
    s.text(M + 62, zy + 47, "WHAT MAKES IT HOLD", 22, 800, TEXT["green"])
    s.text(W - M - 24, zy + 47, "deeper: Day 14", 20, 700, TEXT["muted"], anchor="end", max_width=200)
    items = ["groups from the directory", "filter can't be skipped", "unknown user → nothing",
             "permissions synced fast"]
    for i, item in enumerate(items):
        x = M + 28 + (i % 2) * 460
        y = zy + 92 + (i // 2) * 36
        s.icon("check", x + 10, y - 7, 22, COLORS["green"][0], sw=3)
        s.text(x + 30, y, item, 21, 700, INK, max_width=420)
    return s


# ----------------------------------------------------------------------------- slide 2
def slot_row(s: Slide, x: float, y: float, label: str, slots: list[tuple[str, str]], result: str,
             result_color: str) -> None:
    # slots: (text, kind) with kind in {"ok", "restricted", "removed", "other"}
    s.text(x, y + 28, label, 21, 800, INK, max_width=130)
    sx, sw_, gap = x + 140, 196, 14
    for j, (text, kind) in enumerate(slots):
        bx = sx + j * (sw_ + gap)
        if kind == "removed":
            s.box(bx, y, sw_, 42, "gray", PAPER_WHITE, r=10, sw=2.2, dashed=True, jitter=0.7)
            s.icon("x", bx + sw_ / 2, y + 21, 22, COLORS["red"][0], sw=3)
            continue
        color = {"ok": "green", "restricted": "violet", "other": "gray"}[kind]
        s.box(bx, y, sw_, 42, color, "fill" if kind != "other" else "wash", r=10, sw=2.2, jitter=0.7)
        if kind == "restricted":
            s.icon("lock-keyhole", bx + 22, y + 21, 20, COLORS["violet"][0], sw=2.4)
        s.text(bx + sw_ / 2 + (8 if kind == "restricted" else 0), y + 29, text, 19, 700, INK, anchor="middle",
               max_width=sw_ - 44)
    s.tag(W - M - 24, y + 30, result, result_color, size=21, anchor="end")


def slide_how() -> Slide:
    s = Slide(2, TOTAL, DAY, "Vectors + records + the model", seed=52)
    heading(s, "Vectors + records + the model", "What a vector index stores, and where the filter goes.")

    zy, zh = 274, 446
    s.box(M, zy, W - 2 * M, zh, "blue", "wash", r=22, dashed=True, sw=3)
    s.text(M + 30, zy + 48, "THE INDEX: ONE FILE, index.npz (99,192 bytes)", 22, 800, TEXT["blue"])
    parts = [("layers", "blue", "vectors", "44 × 384 float32"), ("table", "teal", "records", "1 per row, key chunk_id"),
             ("cpu", "violet", "the model", "bge-small @ 5c38ec7c405e")]
    for i, (icon, color, label, note) in enumerate(parts):
        cx = M + 150 + i * 330
        s.node(cx, zy + 136, icon, color, label, size=96, icon_size=50, label_size=22)
        s.text(cx, zy + 250, note, 19, 700, TEXT["muted"], anchor="middle", max_width=300)
        if i < 2:
            s.text(cx + 165, zy + 150, "+", 44, 800, INK, anchor="middle", max_width=40)
    # one record
    rx, ry, rw = M + 24, zy + 280, W - 2 * M - 48
    s.box(rx, ry, rw, 146, "gray", PAPER_WHITE, r=12, sw=2.4)
    fields = [("chunk_id", RESTRICTED), ("chunk_hash", "146c27100b941450..."),
              ("allowed_groups", "managers, people-team"), ("last_reviewed", "None")]
    for i, (key, value) in enumerate(fields):
        y = ry + 36 + i * 32
        s.text(rx + 20, y, key, 18, 700, TEXT["blue"], family="mono", max_width=200)
        s.text(rx + 230, y, value, 18, 500, INK, family="mono", max_width=rw - 250)

    # pre vs post filter
    zy = 748
    s.box(M, zy, W - 2 * M, 488, "orange", "wash", r=22, dashed=True, sw=3)
    s.icon("funnel", M + 38, zy + 40, 30, COLORS["orange"][0], sw=2.3)
    s.text(M + 64, zy + 49, "FILTER BEFORE YOU RANK", 22, 800, TEXT["orange"])
    s.text(M + 30, zy + 92, f"Employee asks: “{PAY_Q}”", 21, 700, INK, max_width=W - 2 * M - 60)
    x = M + 24
    slot_row(s, x, zy + 112, "no filter", [("0.70", "restricted"), ("0.59", "other"), ("0.58", "other")], "top 3", "gray")
    slot_row(s, x, zy + 164, "post-filter", [("", "removed"), ("0.59", "ok"), ("0.58", "ok")], "2 of 3", "red")
    slot_row(s, x, zy + 216, "pre-filter", [("0.59", "ok"), ("0.58", "ok"), ("0.57", "ok")], "3 of 3", "green")
    s.text(M + 30, zy + 300, f"“{VPN_Q}” · scope: remote work policy", 21, 700, INK, max_width=W - 2 * M - 60)
    slot_row(s, x, zy + 320, "no filter", [("0.80 VPN guide", "other"), ("0.76 VPN guide", "other"),
                                          ("0.74 VPN guide", "other")],
             "top 3", "gray")
    slot_row(s, x, zy + 372, "post-filter", [("", "removed"), ("", "removed"), ("", "removed")], "0 of 3", "red")
    slot_row(s, x, zy + 424, "pre-filter", [("0.71", "ok"), ("0.51", "ok"), ("0.51", "ok")], "3 of 3", "green")
    return s


# ----------------------------------------------------------------------------- slide 3
def slide_sync() -> Slide:
    s = Slide(3, TOTAL, DAY, "Sync, don't just upsert", seed=53)
    heading(s, "Sync, don't just upsert", "Day 02's changes + a renamed heading, applied to the saved index.")

    zy = 274
    s.box(M, zy, W - 2 * M, 392, "teal", "wash", r=22, dashed=True, sw=3)
    s.icon("refresh-cw", M + 38, zy + 40, 28, COLORS["teal"][0], sw=2.4)
    s.text(M + 64, zy + 49, "SYNC PLAN, BY chunk_id AND chunk_hash", 22, 800, TEXT["teal"])
    unit, bx = 11, M + 256
    for i, (name, n, color, reason) in enumerate(PLAN):
        y = zy + 84 + i * 52
        s.text(M + 30, y + 28, name, 21, 800, tcolor(color), max_width=200)
        s.box(bx, y + 4, max(n * unit, 14), 34, color, "fill", r=8, sw=2.2, jitter=0.6)
        s.text(bx + max(n * unit, 14) + 14, y + 30, str(n), 23, 800, INK, max_width=50)
        s.text(bx + max(n * unit, 14) + 58, y + 29, reason, 18, 600, TEXT["muted"],
               max_width=W - M - 24 - (bx + max(n * unit, 14) + 58))
    s.icon("cpu", M + 44, zy + 356, 28, COLORS["violet"][0], sw=2.4)
    s.text(M + 70, zy + 364, "3 of 41 texts through the model · same result as a rebuild",
           21, 800, INK, max_width=W - 2 * M - 100)

    # upsert only vs full sync
    zy = 706
    cw = (W - 2 * M - 20) / 2
    s.text(M, zy, f"Q: {VPN_Q}  (the VPN guide was retired)", 22, 800, INK)
    for k, (title, count, color, rows) in enumerate([
            ("UPSERT ONLY", "46 chunks", "red", [("0.80", "vpn-access-guide" + UPSERT_VPN[0]),
                                                ("0.76", "vpn-access-guide" + UPSERT_VPN[1]),
                                                ("0.74", "vpn-access-guide" + UPSERT_VPN[2])]),
            ("FULL SYNC", "41 chunks", "green", SYNC_VPN)]):
        x = M + k * (cw + 20)
        y = zy + 22
        s.box(x, y, cw, 282, color, "wash", r=20, sw=2.8)
        s.icon("circle-x" if color == "red" else "circle-check", x + 34, y + 38, 28, COLORS[color][0], sw=2.4)
        s.text(x + 58, y + 46, title, 22, 800, TEXT[color], max_width=200)
        s.text(x + cw - 22, y + 46, count, 22, 800, INK, anchor="end", max_width=140)
        for i, (score, cid) in enumerate(rows):
            ry = y + 68 + i * 56
            doc, section = cid.split("#")
            s.text(x + 22, ry + 22, f"#{i + 1} {score}", 19, 800, INK, max_width=90)
            s.text(x + 112, ry + 22, doc, 16, 500, TEXT["muted"], family="mono", max_width=cw - 130)
            s.text(x + 112, ry + 44, "#" + section, 16, 600, INK, family="mono", max_width=cw - 130)
        note = "retired guide fills the top 3" if color == "red" else "remote work policy's VPN advice"
        s.text(x + 22, y + 262, note, 19, 700, TEXT[color], max_width=cw - 44)

    # two guards
    zy = 1034
    cw = (W - 2 * M - 20) / 2
    guards = [("save", "blue", "ATOMIC SAVE", ["save fails halfway:", "naive → file broken (EOFError)",
                                                "atomic → old index loads (44)"]),
              ("trash-2", "orange", "DELETE GUARD", ["ingest returns nothing:", "“refusing to delete",
                                                      "44 of 44 chunks”"])]
    for k, (icon, color, title, lines) in enumerate(guards):
        x = M + k * (cw + 20)
        s.box(x, zy, cw, 194, color, "wash", r=20, sw=2.8)
        s.icon(icon, x + 34, zy + 38, 28, icon_color(color), sw=2.4)
        s.text(x + 58, zy + 46, title, 22, 800, tcolor(color), max_width=cw - 80)
        for i, line in enumerate(lines):
            s.text(x + 24, zy + 88 + i * 34, line, 20, 700 if i else 600, INK if i else TEXT["muted"],
                   max_width=cw - 48)
    return s


# ----------------------------------------------------------------------------- slide 4
def slide_scale(rows: list[tuple[int, float, float]], embed_ms: float) -> Slide:
    s = Slide(4, TOTAL, DAY, "Do you need a vector database?", seed=54)
    heading(s, "Do you need a vector DB?", "Exact NumPy search, measured, vs the cost of adding a library.")

    zy = 274
    s.box(M, zy, W - 2 * M, 390, "blue", "wash", r=22, dashed=True, sw=3)
    s.icon("timer", M + 38, zy + 40, 28, COLORS["blue"][0], sw=2.4)
    s.text(M + 64, zy + 49, "EXACT TOP 3, PER QUESTION", 22, 800, TEXT["blue"])
    s.text(W - M - 24, zy + 49, "2-vCPU cloud VM", 19, 700, TEXT["muted"], anchor="end", max_width=220)
    bx, bw = M + 214, W - 2 * M - 450
    top = max(embed_ms, rows[-1][2]) * 1.08

    def px(ms: float) -> float:
        return bx + ms / top * bw

    for i, (n, mb, ms) in enumerate(rows):
        y = zy + 80 + i * 60
        s.text(M + 30, y + 30, f"{n:,} chunks", 21, 800, INK, max_width=184)
        s.box(bx, y + 4, max(px(ms) - bx, 10), 38, "blue", "fill", r=8, sw=2.2, jitter=0.6)
        s.text(max(px(ms), bx + 10) + 14, y + 31, f"{ms:.3f} ms", 20, 800, INK, max_width=150)
        s.text(W - M - 24, y + 31, f"{mb:g} MB", 19, 700, TEXT["muted"], anchor="end", max_width=110)
    ex = px(embed_ms)
    s.line(ex, zy + 72, ex, zy + 318, COLORS["violet"][0], sw=2.6, dashed=True, bow=0.1)
    s.text(ex, zy + 352, f"embedding the question: {embed_ms:.1f} ms", 20, 800, TEXT["violet"], anchor="middle",
           max_width=420)

    zy = 688
    s.box(M, zy, W - 2 * M, 344, "orange", "wash", r=22, dashed=True, sw=3)
    s.icon("package", M + 38, zy + 40, 28, COLORS["orange"][0], sw=2.4)
    s.text(M + 64, zy + 49, "INSTALL SIZE ON TOP OF NUMPY", 22, 800, TEXT["orange"])
    s.text(W - M - 24, zy + 49, "measured 2026-10-18", 19, 700, TEXT["muted"], anchor="end", max_width=240)
    lx, lw = M + 190, W - 2 * M - 470
    biggest = max(mb for _, mb, _ in LIBRARIES)
    for i, (name, mb, n) in enumerate(LIBRARIES):
        y = zy + 76 + i * 44
        s.text(M + 30, y + 26, name, 20, 700, INK, family="mono", max_width=160)
        width = max(mb / biggest * lw, 8)
        s.box(lx, y + 4, width, 30, "orange", "fill", r=7, sw=2, jitter=0.5)
        label = f"{mb:g} MB · {n} package{'s' if n > 1 else ''}"
        s.text(lx + width + 14, y + 27, label, 19, 700, INK, max_width=W - M - 24 - (lx + width + 14))
        if name == "voyager":
            s.icon("triangle-alert", lx + width + 238, y + 19, 22, COLORS["red"][0], sw=2.4)
            s.text(lx + width + 256, y + 27, "2.2.2: no import on Py 3.13", 18, 700, TEXT["red"],
                   max_width=W - M - 24 - (lx + width + 256))

    if rows[-1][2] >= embed_ms:   # the bubble's claim must hold in the saved outputs
        raise SystemExit(f"100,000 vectors took {rows[-1][2]} ms, not less than embedding ({embed_ms} ms)")
    s.bubble(M, 1054, W - 2 * M, ["Even 100,000 chunks searched faster than the model",
                                  "embeds one question. No library yet.",
                                  "Carries over to any vector DB: records, filters, sync."],
             "yellow", size=23, weight=700, tail=None, pad=20)
    s.text(M, 1226, "Next → Day 06: grounded answers with citations", 22, 800, TEXT["teal"])
    return s


def main() -> None:
    check_contrast()
    check_facts()
    rows, embed_ms = scale_table(notebook_outputs())
    OUT.mkdir(parents=True, exist_ok=True)
    slides = [("slide-01-cover", slide_cover()), ("slide-02-how-it-works", slide_how()),
              ("slide-03-sync-and-deletes", slide_sync()),
              ("slide-04-when-to-add-a-library", slide_scale(rows, embed_ms))]
    for name, slide in slides:
        path = OUT / f"{name}.svg"
        path.write_text(slide.svg(), encoding="utf-8")
        print("wrote", path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
