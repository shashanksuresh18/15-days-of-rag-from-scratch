"""Checks on the committed LinkedIn carousels in assets/day-XX/.

Rebuilding a carousel PDF from the committed slide PNGs must reproduce the committed PDF byte
for byte, so a rebuild only shows up in Git when a slide really changed.

Needs Pillow from requirements-assets.txt. Learners who install only requirements.txt (and CI)
skip this test; maintainers who rebuild slides run it.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("PIL", reason="slide tooling not installed (pip install -r requirements-assets.txt)")

REPO_ROOT = Path(__file__).resolve().parents[1]
CAROUSELS = sorted(REPO_ROOT.glob("assets/day-*/day-*-carousel.pdf"))


def _load_render():
    path = REPO_ROOT / "scripts" / "slides" / "render.py"
    spec = importlib.util.spec_from_file_location("render", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("committed", CAROUSELS, ids=[p.parent.name for p in CAROUSELS])
def test_carousel_pdf_rebuilds_byte_identical(committed: Path, tmp_path: Path) -> None:
    render = _load_render()
    pngs = sorted(committed.parent.glob("slide-*.png"))
    first, second = tmp_path / committed.name, tmp_path / "again" / committed.name
    second.parent.mkdir()
    render.write_carousel(pngs, first)
    render.write_carousel(pngs, second)
    assert first.read_bytes() == second.read_bytes(), "two rebuilds differ: is a timestamp leaking in?"
    assert first.read_bytes() == committed.read_bytes(), (
        f"{committed.relative_to(REPO_ROOT).as_posix()} is out of date: "
        f"run python scripts/slides/render.py {committed.parent.relative_to(REPO_ROOT).as_posix()}"
    )
