# Architecture and pipeline diagrams (`figkit.Diagram`)

Model-architecture, pipeline and method-overview figures drawn in matplotlib at
**print size** — a Python replacement for TikZ / PowerPoint diagrams.
Reference implementation: `figure_FasterHAND/plot_architecture.py`.

## Rules

1. **Build at print width, never rescale.** Pick the width from `figkit.VENUES`
   (`ieee["text"] = 181 mm` for `figure*`, `ieee["column"] = 88.9 mm` for `figure`).
   Include with `\includegraphics[width=\textwidth]{...pdf}` so 1 mm in the script = 1 mm on paper.
2. **Fonts in pt at print scale:** body 7 pt, secondary 6.2 pt, never below 6 pt
   (`FONT_PT`). Strokes 0.4–0.9 pt (`LINE_PT`).
3. **Units are millimetres**, origin bottom-left. Sketch the panel rectangles first,
   then fill them. Use `Diagram(..., debug_grid=True)` (`--debug`) for a 5 mm overlay while laying out.
4. **One colour per semantic role** (encoder / memory / decoder / output / each optional
   mechanism). Okabe-Ito (`OKABE_ITO`) is the default for diagrams; fills are tints
   (`tint(c, 0.05–0.18)`), strokes the full colour, text stays ink/muted.
5. **Colour is never the only cue:** base model = solid borders; optional / inference-only
   components = dashed borders + letter markers (`marker("A", ...)`) where they attach.
6. **Numbered panel headers** (`tag`) give the reading order 1 → 2 → 3; optional mechanisms
   get letters A, B.
7. **Show, don't only label:** real input thumbnails (`image`), 3-D slabs for CNN stages
   (`cuboid`), cell grids/strips for feature maps and sequences (`cells`), token chips
   (`chip`), masks as tiny cell glyphs. Keep exact hyper-parameters in sub-lines.
8. **Orthogonal arrows** with rounded corners (`arrow(p0, p1, p2, ..., r=1.2)`); feedback /
   optional flows dashed.
9. Export **PDF (vector, TrueType-embedded) + 600-dpi PNG** via `Diagram.save`. Then view the
   PNG and fix every overlap / overflow before finishing; compile the paper and check the page.

## API (figkit/diagram.py)

| Call | Draws |
|---|---|
| `Diagram(w_mm, h_mm, font_size=7, math="match"/"cm", debug_grid=False)` | canvas; `.ax`, `.fig` |
| `panel(x, y, w, h, color, fill, ls)` | background group; `ls=(0,(4,2))` for dashed |
| `box(x, y, w, h, title, sub, color, fill, ha="center"/"left")` | block with title + sub line |
| `tag(x, y, text, color, ha, va)` | filled pill header |
| `marker(x, y, "A", color)` | circled attachment letter |
| `arrow(*pts, color, ls, r, head)` | poly-line arrow, head at the last point |
| `line(*pts)` | plain poly-line |
| `cuboid(x, y, w, h, depth, color)` | 3-D slab (feature map) |
| `cells(x, y, nx, ny, cell, colors)` | grid/strip; colours per cell, per row or single |
| `chip(x, y, s, color, filled, square)` | token chip |
| `image(path, x, y, w)` | raster thumbnail with border |
| `oplus(x, y)`, `brace(x0, x1, y)` | ⊕ node, square bracket under a span |
| `stack_shadow(box, n)` | offset cards behind a repeated block (`×N`) |
| `extent(artist)`, `text_width(s)` | measure text in mm |

All shape calls return a `Box` with anchors `n s e west ne nw se sw c` and `at(fx, fy)`.
Math uses the text font by default (`$\mathbfit{f}^{1D}$`, `$\hat y_t$`); pass `math="cm"` to
match a Computer Modern body instead. No LaTeX install needed.

## Workflow for a new paper

1. `figure_<Paper>/plot_<name>.py` (copy the Faster-HAND script as a template), assets in
   `figure_<Paper>/assets/`, outputs in `figure_<Paper>/figures/`.
2. Read the paper's current figure source + caption, and the code it documents, so every block
   and number in the diagram is true.
3. Lay out panels → render → inspect PNG → iterate.
4. Copy the PDF into the paper's `figures/`, switch the LaTeX to `\includegraphics`, keep the
   caption consistent with what is drawn, compile, and look at the page.
