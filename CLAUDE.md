# figures4papers — instructions for Claude Code

Publication figures in Python for any paper. Point Claude here with:
"use ~/figures4papers to make Fig. N of <paper path>".

## Layout
- `figkit/` — shared library. `style.py`: venue widths (`VENUES`), `FONT_PT`, palettes
  (`PALETTE`, `OKABE_ITO`), `tint`/`shade`, `use_style`, `save`. `diagram.py`: `Diagram`
  millimetre canvas for architecture / pipeline figures.
- `figure_<Paper>/plot_<name>.py` → `figure_<Paper>/figures/<name>.{pdf,png}`; inputs in
  `figure_<Paper>/assets/`.
- `scientific-figure-making/` — the skill (also symlinked at `~/.claude/skills/`). Read
  `SKILL.md`, then `references/diagrams.md` for diagrams or `design-theory.md` for data plots.

## Rules
- Build every figure at its printed width (`VENUES[venue]["column"|"text"]` mm) with fonts in pt
  (7 body, >= 6 minimum); never rescale in LaTeX — `\includegraphics[width=\textwidth]`.
- Export vector PDF (TrueType embedded) + 600-dpi PNG. View the PNG and fix overlaps before
  finishing; when the paper source is available, copy the PDF in, compile, and check the page.
- Every number/label must match the paper text and the code it documents.
- Run scripts from the repo root: `python figure_<Paper>/plot_<name>.py [--debug]`.
- Reuse `figkit` helpers; add a new primitive to `figkit/diagram.py` only when it is generic.
