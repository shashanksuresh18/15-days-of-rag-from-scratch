"""Build the two sample PDFs in data/sample/acme-handbook/ with the Python standard library only.

    python scripts/data/make_sample_pdfs.py            # (re)writes both PDFs
    python scripts/data/make_sample_pdfs.py --check    # fails if the committed PDFs differ

Why generate them instead of exporting from a word processor? So that every byte of the sample
data is explainable and reproducible, and nothing in it comes from a third party. Real PDFs come
from many different generators, each with its own quirks. These two are built to show the
quirks Day 02 teaches, on purpose:

travel-policy.pdf
  - two pages, with a running header and footer on every page
  - one word hyphenated across a line break ("reim-" / "bursement")
  - a table drawn one column at a time, so the drawing order differs from the reading order
    (some generators do this; PDF only records where text goes, not what is a row)
  - a document-information dictionary with Title plus two custom keys, Owner and LastReviewed

scanned-visitor-policy.pdf
  - one page holding a single greyscale image of a "scanned" page: it looks like lines of
    text, but there is no text layer, so text extraction returns nothing (OCR would be needed)

Content streams are not compressed, so you can open travel-policy.pdf in a text editor and read
the drawing instructions yourself, e.g. `BT /F1 10.5 Tf ... (London) Tj ET`.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "sample" / "acme-handbook"
PAGE_W, PAGE_H = 595, 842          # A4 in PDF points (1 pt = 1/72 inch)
LEFT = 72                          # 1-inch left margin


# ----------------------------------------------------------------------------- PDF writer
def pdf_string(s: str) -> str:
    """A PDF literal string: (text), with backslashes and parentheses escaped."""
    return "(" + s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"


def stream(content: bytes, extra: str = "") -> bytes:
    return f"<< /Length {len(content)}{extra} >>\nstream\n".encode("latin-1") + content + b"\nendstream"


def write_pdf(objects: list[bytes], info_obj: int) -> bytes:
    """Serialise numbered objects (object 1 = catalog) with a classic cross-reference table."""
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")   # binary comment marks the file as binary
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {info_obj} 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode("latin-1")
    return bytes(out)


def info_dict(**entries: str) -> bytes:
    return ("<< " + " ".join(f"/{k} {pdf_string(v)}" for k, v in entries.items()) + " >>").encode("latin-1")


# ----------------------------------------------------------------------------- travel policy
def text(x: float, y: float, s: str, font: str = "F1", size: float = 10.5, grey: float = 0.0) -> str:
    return f"BT /{font} {size} Tf {grey:g} g 1 0 0 1 {x:g} {y:g} Tm {pdf_string(s)} Tj ET"


def running_header_footer(page: int, pages: int) -> list[str]:
    return [
        text(LEFT, 806, "Acme Corp  |  Travel Policy  |  Internal", size=8.5, grey=0.4),
        f"0.75 G 0.6 w {LEFT} 798 m {PAGE_W - LEFT} 798 l S",
        f"0.75 G 0.6 w {LEFT} 52 m {PAGE_W - LEFT} 52 l S",
        text(LEFT, 38, f"Acme Corp - internal use only - Page {page} of {pages}", size=8.5, grey=0.4),
    ]


def paragraph(y: float, lines: list[str], leading: float = 15) -> tuple[list[str], float]:
    ops = [text(LEFT, y - i * leading, line) for i, line in enumerate(lines)]
    return ops, y - len(lines) * leading


TABLE_COLUMNS = [("City", 72, 160), ("Hotel per night", 232, 150), ("Meals per day", 382, 141)]
TABLE_ROWS = [
    ("London", "USD 250", "USD 75"),
    ("New York", "USD 300", "USD 75"),
    ("Singapore", "USD 220", "USD 60"),
    ("Other cities", "USD 180", "USD 50"),
]


def limits_table(top: float, row_h: float = 22) -> tuple[list[str], float]:
    width = sum(w for _, _, w in TABLE_COLUMNS)
    n_rows = len(TABLE_ROWS) + 1
    bottom = top - n_rows * row_h
    ops = [f"0.93 g {LEFT} {top - row_h:g} {width} {row_h} re f"]            # shaded header row
    for r in range(n_rows + 1):                                               # horizontal rules
        y = top - r * row_h
        ops.append(f"0.55 G 0.7 w {LEFT} {y:g} m {LEFT + width} {y:g} l S")
    for _, x, _ in TABLE_COLUMNS:                                             # vertical rules
        ops.append(f"0.55 G 0.7 w {x} {top:g} m {x} {bottom:g} l S")
    ops.append(f"0.55 G 0.7 w {LEFT + width} {top:g} m {LEFT + width} {bottom:g} l S")
    # The quirk: cells are drawn column by column (all cities, then all hotel limits, then
    # all meal limits). The page looks like a normal table; the drawing order does not.
    for c, (header, x, _) in enumerate(TABLE_COLUMNS):
        ops.append(text(x + 8, top - 15, header, font="F2"))
        for r, row in enumerate(TABLE_ROWS, start=1):
            ops.append(text(x + 8, top - r * row_h - 15, row[c]))
    return ops, bottom


def travel_policy_pages() -> list[str]:
    # Page 1
    ops = running_header_footer(1, 2)
    ops.append(text(LEFT, 752, "Travel Policy", font="F2", size=20))
    ops.append(text(LEFT, 714, "1. Purpose", font="F2", size=13))
    body, y = paragraph(694, [
        "This policy explains how Acme employees book business travel and claim reim-",
        "bursement for travel costs. It applies to all employees, and to contractors who",
        "travel at Acme's request.",
    ])
    ops += body
    ops.append(text(LEFT, y - 16, "2. Booking travel", font="F2", size=13))
    body, y = paragraph(y - 36, [
        "Book flights, trains and hotels through the travel portal at least 14 days before",
        "you travel. Economy class is standard for flights under six hours. Flights of six",
        "hours or more may be booked in premium economy with your manager's approval.",
    ])
    ops += body
    ops.append(text(LEFT, y - 16, "3. Daily limits by city", font="F2", size=13))
    body, y = paragraph(y - 36, [
        "Hotel and meal costs are reimbursed up to the daily limits below. Limits are per",
        "person and include taxes. Use the Other cities row for any city not listed.",
    ])
    ops += body
    table, _ = limits_table(y - 8)
    ops += table
    page1 = "\n".join(ops)

    # Page 2
    ops = running_header_footer(2, 2)
    ops.append(text(LEFT, 752, "4. Claiming travel costs", font="F2", size=13))
    body, y = paragraph(732, [
        "Claim travel costs in the expense system, as described in the Expense",
        "Reimbursement Policy, and attach itemised receipts. Costs above the daily limits",
        "need written approval from your manager before the trip.",
    ])
    ops += body
    ops.append(text(LEFT, y - 16, "5. Safety while travelling", font="F2", size=13))
    body, _ = paragraph(y - 36, [
        "Register every international trip in the travel portal, so the Security Team can",
        "reach you in an emergency. Report a lost or stolen laptop or phone straight away,",
        "as described in the Security Incident Reporting policy.",
    ])
    ops += body
    page2 = "\n".join(ops)
    return [page1, page2]


def build_travel_policy() -> bytes:
    pages = travel_policy_pages()
    # Object numbers: 1 catalog, 2 page tree, 3-4 fonts, then (page, contents) pairs, then info.
    first_page = 5
    kids = " ".join(f"{first_page + 2 * i} 0 R" for i in range(len(pages)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    ]
    for i, content in enumerate(pages):
        contents_obj = first_page + 2 * i + 1
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
                        f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
                        f"/Contents {contents_obj} 0 R >>").encode("latin-1"))
        objects.append(stream(content.encode("latin-1")))
    objects.append(info_dict(Title="Travel Policy", Owner="Finance", LastReviewed="2026-04-15",
                             Producer="scripts/data/make_sample_pdfs.py (synthetic sample)"))
    return write_pdf(objects, info_obj=len(objects))


# ----------------------------------------------------------------------------- scanned page
IMG_W, IMG_H = 300, 424            # low resolution on purpose: a quick office scan


def run_length_encode(data: bytes) -> bytes:
    """PDF RunLengthDecode encoding (ISO 32000, 7.4.5): a deterministic, dependency-free filter."""
    out, i, n = bytearray(), 0, len(data)
    while i < n:
        run = 1
        while i + run < n and run < 128 and data[i + run] == data[i]:
            run += 1
        if run > 1:
            out += bytes([257 - run, data[i]])
            i += run
            continue
        start = i
        while i < n and i - start < 128 and not (i + 1 < n and data[i + 1] == data[i]):
            i += 1
        out += bytes([i - start - 1]) + data[start:i]
    return bytes(out + b"\x80")


def scanned_page_pixels() -> bytes:
    rng = random.Random(2026)
    paper, ink = 244, 72
    px = [[paper] * IMG_W for _ in range(IMG_H)]

    def bar(x0: int, y0: int, w: int, h: int, value: int) -> None:
        for y in range(y0, min(y0 + h, IMG_H)):
            for x in range(x0, min(x0 + w, IMG_W)):
                px[y][x] = value

    for y in range(IMG_H):                       # scanner shadow along the left edge
        for x in range(6):
            px[y][x] = 190 + 9 * x
    bar(36, 34, 120, 7, ink)                     # title
    y = 62
    for block in (4, 5, 3, 6, 4):                # paragraphs of "words"
        bar(36, y, 70, 4, ink + 10)              # section heading
        y += 14
        for line in range(block):
            x, end = 36, 264 if line < block - 1 else 140 + int(rng.random() * 100)
            while x < end:
                w = 6 + int(rng.random() * 26)
                bar(x, y, min(w, end - x), 3, ink + int(rng.random() * 30))
                x += w + 4
            y += 9
        y += 12
    bar(36, y + 10, 60, 1, ink)                  # signature line
    return bytes(v for row in px for v in row)


def build_scanned_policy() -> bytes:
    image = run_length_encode(scanned_page_pixels())
    content = f"q {PAGE_W} 0 0 {PAGE_H} 0 0 cm /Im1 Do Q".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
         f"/Resources << /XObject << /Im1 5 0 R >> >> /Contents 4 0 R >>").encode("latin-1"),
        stream(content),
        stream(image, f" /Type /XObject /Subtype /Image /Width {IMG_W} /Height {IMG_H} "
                      f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /RunLengthDecode"),
        info_dict(Title="Visitor Policy", Owner="Facilities", LastReviewed="2026-02-02",
                  Producer="scripts/data/make_sample_pdfs.py (synthetic sample)"),
    ]
    return write_pdf(objects, info_obj=len(objects))


BUILDERS = {"travel-policy.pdf": build_travel_policy, "scanned-visitor-policy.pdf": build_scanned_policy}


def main(argv: list[str]) -> int:
    check = "--check" in argv
    stale = []
    for name, build in BUILDERS.items():
        path, data = OUT_DIR / name, build()
        if check:
            if not path.exists() or path.read_bytes() != data:
                stale.append(name)
            continue
        path.write_bytes(data)
        print(f"wrote {path.relative_to(ROOT).as_posix()} ({len(data):,} bytes)")
    if stale:
        print(f"Committed PDFs differ from the generator: {stale}. Run without --check to rebuild.")
        return 1
    if check:
        print("Sample PDFs match the generator.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
