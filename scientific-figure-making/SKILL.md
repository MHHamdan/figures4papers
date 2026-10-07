---
name: scientific-figure-making
description: >-
  Covers publication-ready matplotlib figures for academic papers, slides, and
  reports—bars, trends, scatter, heatmaps, multi-panel layouts, and model
  architecture / pipeline diagrams (TikZ or PowerPoint replacements via
  figkit.Diagram)—with this repository’s house style, print/vector export
  conventions, and parity with figures4papers demos. Use when the user is
  finalizing or creating such figures in Python/matplotlib. Do not use for interactive dashboards or web viz (Plotly, Altair,
  Bokeh), exploratory-only plots without a publication target, dominant 3D or
  geographic mapping, or Illustrator/Figma-first infographic workflows.
---

# Scientific figure making

Open `references/` only as needed; do not preload every file. Start from the table below, then follow links inside the document you opened (and into `figure_*` code via [references/demos.md](references/demos.md)) instead of loading the full reference set up front.

## When to load this skill

- Matplotlib figures for **papers, slides, or reports** that must match **this repo’s publication look** (fonts, palette, spines, legends, export).
- Requests involving **grouped bars, trend lines, heatmaps, multi-panel grids**, or **PDF/SVG/high-DPI** output in a scientific-figure context.
- **Architecture, pipeline or method-overview diagrams** to be redrawn in Python → [references/diagrams.md](references/diagrams.md) and `figkit/`.
- References to **figures4papers** `figure_*` projects or “same style as the repo figures.”

## When not to load

- **Plotly, Altair, Bokeh**, or other interactive / web-first plotting.
- **EDA-only** plots where seaborn or pandas is enough until there is a publication target.
- Primary workflow is **3D, GIS**, or **non-matplotlib** tooling.
- **Illustrator / Figma–first** layout or infographic (not matplotlib data plots).

## Related files

| File | Open when |
|------|-----------|
| [references/tutorials.md](references/tutorials.md) | End-to-end walkthroughs (bar, trends, heatmap) |
| [references/api.md](references/api.md) | Function signatures, `PALETTE`, validation rules |
| [references/common-patterns.md](references/common-patterns.md) | Layout patterns, legend panel, print-safe bars |
| [references/design-theory.md](references/design-theory.md) | Typography, export policy, palette rationale |
| [references/demos.md](references/demos.md) | Canonical `figure_*` demo links in figures4papers |
| [references/diagrams.md](references/diagrams.md) | Architecture / pipeline diagrams with `figkit.Diagram` (mm canvas, print-size fonts) |

Shared code lives in `figkit/` at the repo root (`style.py`: venues, palettes, fonts, export; `diagram.py`: diagram primitives). Import it with `sys.path.insert(0, <repo root>)`.
