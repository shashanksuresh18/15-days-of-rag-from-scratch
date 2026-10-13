"""Render every SVG in a folder to a PNG of the same name, check its size, and bundle the
slides into a PDF for LinkedIn document (swipeable carousel) posts.

    python scripts/slides/render.py assets/day-01

Uses resvg (via the resvg-py wheel), which needs no system graphics libraries. Fonts are
loaded from scripts/slides/fonts/, so every machine renders the same glyphs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import resvg_py
from PIL import Image

EXPECTED = {(1080, 1350), (1080, 1080)}
FONT_DIR = Path(__file__).resolve().parent / "fonts"


def render(folder: Path) -> None:
    svgs = sorted(folder.glob("*.svg"))
    if not svgs:
        sys.exit(f"No SVG files in {folder}")
    for svg in svgs:
        png = svg.with_suffix(".png")
        png.write_bytes(resvg_py.svg_to_bytes(svg_path=str(svg), font_dirs=[str(FONT_DIR)]))
        with Image.open(png) as img:
            if img.size not in EXPECTED:
                sys.exit(f"{png.name}: unexpected size {img.size}")
            print(f"{png.as_posix()}  {img.size[0]}x{img.size[1]}")

    pages = [Image.open(svg.with_suffix(".png")).convert("RGB") for svg in svgs]
    pdf = folder / f"{folder.name}-carousel.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=144)
    print(f"{pdf.as_posix()}  {len(pages)} pages")


if __name__ == "__main__":
    render(Path(sys.argv[1] if len(sys.argv) > 1 else "assets/day-01"))
