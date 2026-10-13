# LinkedIn Visual Style Guide — 15 Days of RAG

**Purpose:** Each day must include **1–4 educational images** that work as a single infographic or a compact swipeable carousel, not just a pretty decoration.

## Recommended format

- Preferred canvas: **1080 × 1350 px (4:5 portrait)**; square (1080 × 1080) only where a topic genuinely benefits from it.
- Deliver actual **PNG images** in `assets/day-XX/`. Retain editable `.svg`/design files where possible.
- Reserve comfortable margins; keep the main heading and diagram away from image edges. Prioritize contrast and legibility when displayed as a small mobile thumbnail.
- Consistent visual identity: `15 Days of RAG`, `Day XX/15`, topic name, a small optional GitHub/repo reference; no fake logos.
- **Series look (updated 2026-10-13): whiteboard sketch.** Light dotted background, hand-drawn boxes and arrows, pastel fills, colour-coded dashed zones and line icons, so the diagram explains the idea before anyone reads a sentence. Let icons, arrows and numbered steps carry the flow; keep text to short labels.
- Palette: Open Color strokes with pastel fills — blue, green, red, violet, orange, yellow, teal, gray — and darker steps of the same hues for text so every label reaches 4.5:1 contrast. Use colour consistently: one colour per stage or concept within a slide.
- Type and icons: Shantell Sans (hand-drawn, still legible at phone size) for all text, JetBrains Mono for IDs and code, Lucide line icons. All three are bundled in `scripts/slides/` with their licences.

## Storyboard options

| Slide | When used | Role | Example |
| --- | --- | --- | --- |
| 01 | Required | Cover / powerful concept | `Day 06 — How RAG answers with your documents` + mini pipeline |
| 02 | Optional | Step-by-step flow | `Question → Retrieve → Rank → Generate → Cite` |
| 03 | Optional | Concrete trade-off/example | `Dense vs keyword search` on sample queries |
| 04 | Optional | Practical takeaway | `Three things to check before production` |

Use just **1 or 2 slides** if more would add filler. A coherent 4-image carousel is fine for a complex topic. Each image should teach one point, with short explanatory text and accurate arrows/labels. Avoid tiny code, overcrowding, illegible handwriting fonts, jargon without context, decorative clip-art that explains nothing, and visually appealing but misleading diagrams. A sketch style is a way to make the mechanism obvious, not a licence to be vague: labels, numbers and arrows must match the code.

## Definition of done for an image set

- [ ] 1–4 actual PNG image files exist and are shareable; editable source included if feasible
- [ ] All image dimensions verified and slide sequence numbered clearly
- [ ] Colorful, clean, readable at mobile size, with high text/background contrast
- [ ] Technical diagram and facts checked against lesson and runnable example
- [ ] No hallucinated test results, fake charts, unsupported metrics, or sensitive information
- [ ] Spelling, layout, margins, labels, and crop-safe area checked
- [ ] Each image has descriptive filename, alt text, and purpose in `linkedin/day-XX.md`
- [ ] If actual image export is impossible, explicitly state that and provide an honest fallback (editable SVG + export method, or production-ready visual prompts)

## Suggested image naming

```
assets/day-XX/
  slide-01-cover.png
  slide-02-how-it-works.png
  slide-03-example.png
  slide-04-takeaways.png
```

Only create as many slides as needed for that day. Optional editable files can share the same basename with `.svg` extension.
