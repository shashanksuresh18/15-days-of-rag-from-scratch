"""Whiteboard-style slide kit for the "15 Days of RAG" LinkedIn carousels.

The look: hand-drawn boxes and arrows, pastel fills, colour-coded dashed zones and line icons,
on a light dotted "whiteboard". Slides are written as editable SVG (text stays text) and
rendered to PNG by `render.py`.

Safety nets built in:
- every text element is measured with the real font files, and the build stops if a line
  would overflow the width it was given;
- every text colour is checked for WCAG AA contrast (4.5:1) on white;
- shapes use a seeded random generator, so the same script always draws the same wobble.

Fonts (bundled in ./fonts, SIL Open Font License): Shantell Sans (hand-drawn text) and
JetBrains Mono (IDs). Icons (bundled in ./icons): Lucide, ISC licence.
"""

from __future__ import annotations

import html
import math
import random
import re
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

HERE = Path(__file__).resolve().parent
FONT_DIR = HERE / "fonts"
ICON_DIR = HERE / "icons"
FONT_FILES = {"hand": FONT_DIR / "ShantellSans-VariableFont.ttf",
              "mono": FONT_DIR / "JetBrainsMono-VariableFont.ttf"}
FAMILY = {"hand": "'Shantell Sans', 'Comic Neue', 'Segoe Print', sans-serif",
          "mono": "'JetBrains Mono', Menlo, Consolas, monospace"}

# ------------------------------------------------------------------ canvas & brand
W, H = 1080, 1350        # LinkedIn 4:5 portrait
M = 60                   # side margin
SERIES = "15 DAYS OF RAG"
REPO_LABEL = "github.com/shashanksuresh18/15-days-of-rag-from-scratch"

INK = "#1E1E1E"
PAPER = "#FFFFFF"
# name: (stroke, fill, wash) — Open Color steps, the palette Excalidraw-style sketches use
COLORS = {
    "blue":   ("#1971C2", "#A5D8FF", "#E7F5FF"),
    "green":  ("#2F9E44", "#B2F2BB", "#EBFBEE"),
    "red":    ("#E03131", "#FFC9C9", "#FFF5F5"),
    "violet": ("#6741D9", "#D0BFFF", "#F3F0FF"),
    "orange": ("#E8590C", "#FFD8A8", "#FFF4E6"),
    "yellow": ("#F08C00", "#FFEC99", "#FFF9DB"),
    "teal":   ("#0C8599", "#96F2D7", "#E6FCF5"),
    "gray":   ("#868E96", "#E9ECEF", "#F8F9FA"),
}
# Darker steps for text, so coloured words stay readable (checked by check_contrast()).
TEXT = {"ink": INK, "muted": "#495057", "blue": "#1864AB", "green": "#247A37", "red": "#C92A2A",
        "violet": "#5F3DC4", "orange": "#C2410C", "teal": "#0B7285", "gray": "#495057"}


# ------------------------------------------------------------------ fonts & measuring
@lru_cache(maxsize=None)
def _font(family: str, weight: int, size: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(FONT_FILES[family]), size)
    axes = font.get_variation_axes()
    values = [weight if a["name"] == b"Weight" else a["default"] for a in axes]
    font.set_variation_by_axes(values)
    return font


def measure(s: str, size: int, weight: int = 400, family: str = "hand") -> float:
    return _font(family, weight, size).getlength(s)


def wrap(s: str, size: int, weight: int, max_width: float, family: str = "hand") -> list[str]:
    lines, line = [], ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        if measure(trial, size, weight, family) <= max_width:
            line = trial
        else:
            if not line:
                raise OverflowError(f"Word too long for {max_width:.0f}px: {word!r}")
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(fg: str, bg: str) -> float:
    a, b = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def check_contrast() -> None:
    for name, color in TEXT.items():
        for bg_name, bg in [("paper", PAPER)] + [(f"{k} wash", v[2]) for k, v in COLORS.items()]:
            if contrast(color, bg) < 4.5:
                raise ValueError(f"text colour {name} on {bg_name}: {contrast(color, bg):.2f} < 4.5")


# ------------------------------------------------------------------ icons
@lru_cache(maxsize=None)
def _icon_body(name: str) -> str:
    svg = (ICON_DIR / f"{name}.svg").read_text(encoding="utf-8")
    inner = re.search(r"<svg[^>]*>(.*)</svg>", svg, re.S).group(1)
    return re.sub(r"<!--.*?-->", "", inner, flags=re.S).strip()


# ------------------------------------------------------------------ the slide
class Slide:
    def __init__(self, number: int, total: int, day: str, title: str, seed: int):
        self.number, self.total, self.day, self.title = number, total, day, title
        self.rng = random.Random(seed)
        self.parts: list[str] = []

    # --- low-level -------------------------------------------------------------
    def raw(self, svg: str) -> None:
        self.parts.append(svg)

    def _n(self, amount: float) -> float:
        return self.rng.uniform(-amount, amount)

    @staticmethod
    def _smooth(points: list[tuple[float, float]]) -> str:
        """Catmull-Rom spline through the points, as an SVG cubic Bezier path."""
        p = [points[0]] + points + [points[-1]]
        d = f"M{p[1][0]:.1f},{p[1][1]:.1f}"
        for i in range(1, len(p) - 2):
            p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
            c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
            c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
            d += f" C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"
        return d

    def _stroke(self, d: str, color: str, sw: float, dash: str | None = None, opacity: float = 1) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' stroke-opacity="{opacity}"' if opacity != 1 else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
                          f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}{op}/>')

    # --- hand-drawn shapes -----------------------------------------------------
    def _rounded_rect_points(self, x, y, w, h, r, jitter):
        r = min(r, w / 2, h / 2)
        pts = []
        corners = [(x + w - r, y + r, -90), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90), (x + r, y + r, 180)]
        edges = [(x + r, y, x + w - r, y), (x + w, y + r, x + w, y + h - r),
                 (x + w - r, y + h, x + r, y + h), (x, y + h - r, x, y + r)]
        for (ex1, ey1, ex2, ey2), (cx, cy, a0) in zip(edges, corners):
            length = math.hypot(ex2 - ex1, ey2 - ey1)
            steps = max(2, int(length // 110) + 1)
            for k in range(steps):
                t = k / steps
                pts.append((ex1 + (ex2 - ex1) * t + self._n(jitter), ey1 + (ey2 - ey1) * t + self._n(jitter)))
            for k in range(3):
                a = math.radians(a0 + k * 45)
                pts.append((cx + r * math.cos(a) + self._n(jitter * 0.6), cy + r * math.sin(a) + self._n(jitter * 0.6)))
        return pts

    def box(self, x, y, w, h, color="gray", fill: str | None = "fill", r=18, sw=3.2, dashed=False,
            jitter=1.6, stroke: str | None = None) -> None:
        """Sketchy rounded rectangle. fill: 'fill' (pastel), 'wash' (very light), a hex colour, or None."""
        s, f, wash = COLORS[color]
        stroke = stroke or s
        if fill:
            fill_color = {"fill": f, "wash": wash}.get(fill, fill)
            self.parts.append(f'<rect x="{x + 2:.1f}" y="{y + 2:.1f}" width="{w - 4:.1f}" height="{h - 4:.1f}" '
                              f'rx="{max(r - 2, 0)}" fill="{fill_color}"/>')
        passes = 1 if dashed else 2
        for _ in range(passes):
            pts = self._rounded_rect_points(x, y, w, h, r, jitter)
            pts = pts + pts[:2]                     # overlap the start, like a pen stroke
            self._stroke(self._smooth(pts), stroke, sw if passes == 1 else sw * 0.8,
                         dash="14 11" if dashed else None)

    def ellipse(self, cx, cy, rx, ry, color="gray", fill: str | None = "fill", sw=3, jitter=1.5) -> None:
        s, f, wash = COLORS[color]
        if fill:
            fill_color = {"fill": f, "wash": wash}.get(fill, fill)
            self.parts.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx - 1}" ry="{ry - 1}" fill="{fill_color}"/>')
        for _ in range(2):
            start = self.rng.uniform(0, 360)
            pts = [(cx + rx * math.cos(math.radians(start + a)) + self._n(jitter),
                    cy + ry * math.sin(math.radians(start + a)) + self._n(jitter)) for a in range(0, 400, 30)]
            self._stroke(self._smooth(pts), s, sw * 0.8)

    def line(self, x1, y1, x2, y2, color: str = INK, sw=3, dashed=False, bow=1.0) -> None:
        length = math.hypot(x2 - x1, y2 - y1) or 1
        nx, ny = -(y2 - y1) / length, (x2 - x1) / length
        b = self._n(length * 0.012 * bow)
        pts = [(x1 + self._n(1), y1 + self._n(1)),
               (x1 + (x2 - x1) * 0.5 + nx * b, y1 + (y2 - y1) * 0.5 + ny * b),
               (x2 + self._n(1), y2 + self._n(1))]
        self._stroke(self._smooth(pts), color, sw, dash="12 10" if dashed else None)

    def arrow(self, x1, y1, x2, y2, color: str = INK, sw=3.2, bend=0.0, dashed=False, head=20) -> None:
        """Hand-drawn arrow. bend > 0 curves it to the left of travel, < 0 to the right."""
        length = math.hypot(x2 - x1, y2 - y1) or 1
        nx, ny = -(y2 - y1) / length, (x2 - x1) / length
        mx, my = (x1 + x2) / 2 + nx * bend * length, (y1 + y2) / 2 + ny * bend * length
        pts = []
        for k in range(6):
            t = k / 5
            px = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * mx + t ** 2 * x2
            py = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * my + t ** 2 * y2
            wob = 0 if k in (0, 5) else 1.2
            pts.append((px + self._n(wob), py + self._n(wob)))
        self._stroke(self._smooth(pts), color, sw, dash="12 10" if dashed else None)
        # arrowhead follows the curve's final direction (tangent at the end = end - control point)
        a = math.atan2(y2 - my, x2 - mx)
        for side in (-1, 1):
            ang = a + math.pi - side * math.radians(27)
            hx, hy = x2 + head * math.cos(ang), y2 + head * math.sin(ang)
            self.line(x2, y2, hx, hy, color, sw, bow=0.3)

    # --- text ------------------------------------------------------------------
    def text(self, x, y, s, size=28, weight=500, color=INK, family="hand", anchor="start",
             max_width: float | None = None) -> float:
        width = measure(s, size, weight, family)
        limit = max_width if max_width is not None else self._room(x, anchor)
        if width > limit + 0.5:
            raise OverflowError(f"Slide {self.number}: {s!r} is {width:.0f}px, limit {limit:.0f}px")
        anchor_attr = f' text-anchor="{anchor}"' if anchor != "start" else ""
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FAMILY[family]}" font-size="{size}" '
                          f'font-weight="{weight}" fill="{color}"{anchor_attr}>{html.escape(s, quote=False)}</text>')
        return width

    def paragraph(self, x, y, s, size, weight, max_width, line_height, color=INK, family="hand",
                  anchor="start", max_lines=6) -> float:
        rows = wrap(s, size, weight, max_width, family)
        if len(rows) > max_lines:
            raise OverflowError(f"Slide {self.number}: {s[:40]!r}... needs {len(rows)} lines > {max_lines}")
        for i, row in enumerate(rows):
            self.text(x, y + i * line_height, row, size, weight, color, family, anchor, max_width=max_width)
        return y + (len(rows) - 1) * line_height

    def _room(self, x, anchor):
        if anchor == "middle":
            return 2 * min(x - M, W - M - x)
        if anchor == "end":
            return x - M
        return W - M - x

    # --- composite pieces ------------------------------------------------------
    def icon(self, name, cx, cy, size, color, sw=2.0) -> None:
        s = size / 24
        self.parts.append(f'<g transform="translate({cx - size / 2:.1f},{cy - size / 2:.1f}) scale({s:.4f})" '
                          f'fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" '
                          f'stroke-linejoin="round">{_icon_body(name)}</g>')

    def node(self, cx, cy, icon, color, label=None, size=124, icon_size=64, label_size=24) -> None:
        """Icon in a sketchy pastel tile, with an optional tag label underneath."""
        self.box(cx - size / 2, cy - size / 2, size, size, color, "fill", r=24)
        self.icon(icon, cx, cy, icon_size, COLORS[color][0] if color != "yellow" else TEXT["orange"], sw=2.1)
        if label:
            self.tag(cx, cy + size / 2 + 34, label, color, size=label_size, anchor="middle")

    def tag(self, x, y, label, color, size=24, anchor="start", weight=800) -> float:
        """Pill-shaped label. (x, y) is the text baseline anchor."""
        w = measure(label, size, weight) + 30
        h = size + 20
        left = {"start": x, "middle": x - w / 2, "end": x - w}[anchor]
        self.box(left, y - size * 0.82 - 10, w, h, color, "fill", r=h / 2, sw=2.4, jitter=1.0)
        self.text(left + w / 2, y, label, size, weight, TEXT.get(color, INK) if color != "yellow" else INK,
                  anchor="middle", max_width=w)
        return w

    def chip(self, x, y, label, color="blue", size=20, anchor="start") -> float:
        """Monospace ID chip; (x, y) is the top-left (or top-centre/right per anchor)."""
        w = measure(label, size, 500, "mono") + 26
        h = size + 20
        left = {"start": x, "middle": x - w / 2, "end": x - w}[anchor]
        if left < M - 1 or left + w > W - M + 1:
            raise OverflowError(f"Slide {self.number}: chip {label!r} crosses the margin")
        self.box(left, y, w, h, color, "wash", r=10, sw=2.2, jitter=0.8)
        self.text(left + 13, y + h / 2 + size * 0.36, label, size, 500, TEXT.get(color, INK), "mono",
                  max_width=w)
        return w

    def badge(self, cx, cy, n, color, r=22, size=24) -> None:
        self.ellipse(cx, cy, r, r, color, "fill", sw=2.6, jitter=1.0)
        self.text(cx, cy + size * 0.36, str(n), size, 800, INK, anchor="middle", max_width=2 * r)

    def bubble(self, x, y, w, lines: list[str], color="yellow", size=26, weight=600, tail="left",
               line_height=None, pad=22) -> float:
        """Speech bubble with pre-wrapped lines; returns the bubble's bottom y."""
        line_height = line_height or size * 1.32
        h = pad * 2 + size * 0.95 + (len(lines) - 1) * line_height
        self.box(x, y, w, h, color, "fill", r=22)
        s = COLORS[color][0]
        if tail == "left":
            self.line(x + 34, y + h - 3, x + 18, y + h + 24, s, 3)
            self.line(x + 18, y + h + 24, x + 62, y + h - 3, s, 3)
        elif tail == "right":
            self.line(x + w - 34, y + h - 3, x + w - 18, y + h + 24, s, 3)
            self.line(x + w - 18, y + h + 24, x + w - 62, y + h - 3, s, 3)
        for i, row in enumerate(lines):
            self.text(x + pad, y + pad + size * 0.8 + i * line_height, row, size, weight, INK,
                      max_width=w - 2 * pad)
        return y + h

    # --- assembly --------------------------------------------------------------
    def svg(self) -> str:
        frame = Slide(self.number, self.total, self.day, self.title, seed=self.number * 101)
        frame.text(M, 96, SERIES, 26, 800, INK)
        frame.line(M, 108, M + measure(SERIES, 26, 800), 110, COLORS["teal"][0], sw=4, bow=0.5)
        frame.tag(W - M, 96, f"Day {self.day}/15", "yellow", size=24, anchor="end")
        frame.line(M, 1262, W - M, 1262, "#CED4DA", sw=2.2, bow=0.3)
        frame.text(M, 1306, REPO_LABEL, 24, 500, TEXT["muted"], max_width=760)
        frame.text(W - M, 1306, f"{self.number} / {self.total}", 26, 800, INK, anchor="end")
        body = "\n  ".join(frame.parts + self.parts)
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <title>{html.escape(self.title)}</title>
  <defs>
    <pattern id="dots" width="34" height="34" patternUnits="userSpaceOnUse">
      <circle cx="17" cy="17" r="1.6" fill="#DEE2E6"/>
    </pattern>
  </defs>
  <rect width="{W}" height="{H}" fill="{PAPER}"/>
  <rect width="{W}" height="{H}" fill="url(#dots)"/>
  {body}
</svg>
"""
