"""Render every SVG in a folder to a PNG of the same name, check its size, and bundle the
slides into a PDF for LinkedIn document (swipeable carousel) posts.

    python scripts/slides/render.py assets/day-01

Uses resvg (via the resvg-py wheel), which needs no system graphics libraries. Fonts are
loaded from scripts/slides/fonts/, so every machine renders the same glyphs.

Rebuilding is byte-for-byte reproducible: the same SVGs give the same PNGs and the same PDF.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image

EXPECTED = {(1080, 1350), (1080, 1080)}
FONT_DIR = Path(__file__).resolve().parent / "fonts"

# A PDF records when it was created and modified. Pillow uses the current time by default,
# so every rebuild produced a different file. A fixed date (the series start) keeps rebuilt
# carousels byte-identical, so Git only shows a change when a slide really changed.
PDF_DATE = time.strptime("2026-10-13", "%Y-%m-%d")


def render_png(svg: Path) -> bytes:
    import resvg_py  # imported here so write_carousel() works with Pillow alone

    return resvg_py.svg_to_bytes(svg_path=str(svg), font_dirs=[str(FONT_DIR)])


def write_carousel(pngs: list[Path], pdf: Path) -> None:
    pages = [Image.open(png).convert("RGB") for png in pngs]
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=144,
                  creationDate=PDF_DATE, modDate=PDF_DATE)


def render(folder: Path) -> None:
    svgs = sorted(folder.glob("*.svg"))
    if not svgs:
        sys.exit(f"No SVG files in {folder}")
    for svg in svgs:
        png = svg.with_suffix(".png")
        png.write_bytes(render_png(svg))
        with Image.open(png) as img:
            if img.size not in EXPECTED:
                sys.exit(f"{png.name}: unexpected size {img.size}")
            print(f"{png.as_posix()}  {img.size[0]}x{img.size[1]}")

    pdf = folder / f"{folder.name}-carousel.pdf"
    write_carousel([svg.with_suffix(".png") for svg in svgs], pdf)
    print(f"{pdf.as_posix()}  {len(svgs)} pages")


if __name__ == "__main__":
    render(Path(sys.argv[1] if len(sys.argv) > 1 else "assets/day-01"))
