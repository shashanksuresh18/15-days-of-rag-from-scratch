# Slide build

The LinkedIn images in `assets/day-XX/` are hand-drawn-style whiteboard diagrams, built in two steps so they are reproducible and editable.

1. A Python script (`dayXX.py`) writes each slide as an **SVG**. Text stays text, so you can change wording in Inkscape, Figma or a text editor.
2. `render.py` renders each SVG to a **1080 × 1350 PNG** with [resvg](https://github.com/linebender/resvg), checks the size, and bundles the slides into `day-XX-carousel.pdf` for LinkedIn document posts.

```bash
python -m pip install -r requirements-assets.txt
python scripts/slides/day01.py                    # writes assets/day-01/*.svg
python scripts/slides/render.py assets/day-01     # writes assets/day-01/*.png and the PDF
python scripts/slides/day02.py && python scripts/slides/render.py assets/day-02
python scripts/slides/day03.py && python scripts/slides/render.py assets/day-03
python scripts/slides/day04.py && python scripts/slides/render.py assets/day-04
python scripts/slides/day05.py && python scripts/slides/render.py assets/day-05
python scripts/slides/day06.py && python scripts/slides/render.py assets/day-06
python scripts/slides/day07.py && python scripts/slides/render.py assets/day-07
python scripts/slides/day08.py && python scripts/slides/render.py assets/day-08
python scripts/slides/day09.py && python scripts/slides/render.py assets/day-09
```

## How the sketch look works

`kit.py` draws every box, circle and arrow as a slightly wobbly pen stroke: two passes along a smoothed path with small, seeded random offsets. The seed is fixed per slide, so rebuilding produces identical images. It also provides the pieces the slides are made of: icon tiles (`node`), label pills (`tag`), speech bubbles, dashed zones, numbered badges and monospace ID chips.

## Built-in checks

- **Facts:** each `dayXX.py` refuses to build if any number, keyword or ID on its slides is missing from that day's executed notebook outputs.
- **Overflow:** every text element is measured with the bundled font files; a line that would not fit its box stops the build.
- **Contrast:** every text colour must reach at least 4.5:1 (WCAG AA) on white and on every pastel zone background.
- **Reproducible:** rebuilding gives byte-identical SVGs, PNGs and PDFs. `render.py` writes a fixed creation date into the carousel PDF instead of the build time, so Git only shows a slide as changed when it really changed. With `requirements-assets.txt` installed, `python -m pytest tests/test_slides.py` checks that each committed PDF matches its PNGs (the test is skipped otherwise).

## Bundled third-party assets

| Asset | Folder | Licence |
|---|---|---|
| [Shantell Sans](https://github.com/arrowtype/shantell-sans) variable font (hand-drawn text) | `fonts/` | SIL Open Font License 1.1 (`fonts/OFL-ShantellSans.txt`) |
| [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) variable font (IDs) | `fonts/` | SIL Open Font License 1.1 (`fonts/OFL-JetBrainsMono.txt`) |
| [Lucide](https://lucide.dev) icons (v1.56.0, a subset; copied unmodified from the `lucide-static` package) | `icons/` | ISC (`icons/LICENSE-lucide.txt`) |

The font files are unmodified copies from the Google Fonts repository; only the file names were simplified. `render.py` loads them directly, so you don't need to install anything system-wide. To edit an SVG by hand and see the same result, install Shantell Sans and JetBrains Mono locally (both are on Google Fonts).

## Branding

`kit.py` holds the shared palette, the header (`15 DAYS OF RAG`, `Day XX/15`) and the footer. Set `REPO_LABEL` to your repository URL once it is public, then rebuild every day's slides so the series stays consistent.
