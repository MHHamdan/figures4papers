"""Axis A (density access) moves while axis B (mechanism) stays.

    python figure_ROSE/plot_axisA_moves.py [--debug]

Writes figure_ROSE/figures/axisA_moves.{pdf,png} (139.1 x 39.4 mm) for supplement/05_taxonomy.tex,
after fig:taxonomy. Rows are grounded in main/05_taxonomy.tex:16-17 (the four moves), 20-31
(axis A values), 42-47 (edge-case rules for B), 72-82 (composites, distillation, Parallel
WaveNet) and supplement/05_taxonomy.tex:44-46, 81-86, 112-120, 326-353.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figkit import INK, MUTED, shade  # noqa: E402
from rose_style import (DATAFLOW, GRADFLOW, GREY, INFERONLY, KIND, SMALL, TINY, W, access,  # noqa: E402
                        cli, diagram, kbox, kindword)

# column x-ranges (mm)
BX, BW = 0.4, 37.0           # before
OX0, OX1 = BX + BW, 80.0     # operation arrow span
AX, AW = 80.0, 37.0          # after
MX, MW = AX + AW + 2.2, None  # axis B column
MW = W - 0.4 - MX
ROW_GAP = 1.2

# (height, before cells, operation (top, bottom), flow style, after cells, B text, dashed)
# a cell is (access key or None, text)
ROWS = [
    (5.6, [("bound", "diffusion teacher (VLB)")],
     ("consistency / adversarial", "distillation"), GRADFLOW,
     [("noneR", "one- or few-step student")], "diffusion", False),
    (5.6, [(None, "no teacher: path regression")],
     ("direct few-step training", "(MeanFlow, IMM)"), GRADFLOW,
     [("noneR", "few-step sampler")], "diffusion", False),
    (5.6, [("bound", "diffusion model: discrete-chain VLB")],
     ("probability-flow evaluation", "trace estimator + ODE solve / datum"), INFERONLY,
     [("exactR", "likelihood of the ODE model")], "diffusion", False),
    (8.2, [("bound", "VQ-VAE encoder–decoder (VAE)"), ("exact", "prior over codes (AR)")],
     ("composition", "system label = sampling family"), DATAFLOW,
     [("exact", "VQ-VAE: exact over the codes"), ("bound", "bound once decoded to pixels")],
     "AR (sampling)", False),
    (5.6, [("exact", "WaveNet teacher (AR)")],
     ("Parallel WaveNet:", "probability density distillation"), GRADFLOW,
     [("exact", "IAF student: one sequential", "pass / dim on external data")], "AR → flow", True),
]


def cell(d, x, y, w, h, items, kind="ACCESS", ls="-"):
    kbox(d, x, y, w, h, kind=kind, ls=ls)
    n = len(items)
    for i, it in enumerate(items):
        yc = y + h - (i + 0.5) * h / n
        key, *lines = it
        tx = x + 1.4
        if key:
            access(d, x + 2.2, yc, key)
            tx = x + 4.1
        if len(lines) == 1:
            d.text(tx, yc, lines[0], size=SMALL, ha="left", color=INK if key else MUTED)
        else:
            d.text(tx, yc + 1.25, lines[0], size=SMALL, ha="left")
            d.text(tx, yc - 1.25, lines[1], size=SMALL, ha="left")


def build(debug):
    total = sum(r[0] for r in ROWS) + ROW_GAP * (len(ROWS) - 1)
    H = total + 4.0
    d = diagram(H, debug)
    top = total + 0.4
    kindword(d, BX, top + 0.5, "ACCESS", ha="left", text="ACCESS before")
    kindword(d, (OX0 + OX1) / 2, top + 0.5, "OBJECTIVE", ha="center", text="operation")
    kindword(d, AX, top + 0.5, "ACCESS", ha="left", text="ACCESS after")
    kindword(d, MX, top + 0.5, "MODEL", ha="left", text="MECHANISM (B)")

    y = top
    for h, before, (op1, op2), flow, after, mech, side in ROWS:
        y -= h
        ls = (0, (3, 1.6)) if side else "-"
        if side:  # contrast row: B changes, A stays
            d.line((BX, y + h + ROW_GAP / 2), (W - 0.4, y + h + ROW_GAP / 2), color=GREY, lw=0.4,
                   ls=(0, (1, 1)))
        cell(d, BX, y, BW, h, before, ls=ls)
        cell(d, AX, y, AW, h, after, ls=ls)
        ym = y + h / 2
        d.arrow((OX0 + 1.2, ym), (OX1 - 1.2, ym), **flow)
        d.text((OX0 + OX1) / 2, ym + 0.45, op1, size=TINY, va="bottom", color=INK)
        d.text((OX0 + OX1) / 2, ym - 0.45, op2, size=TINY, va="top", color=MUTED)
        kbox(d, MX, y, MW, h, kind="MODEL", ls=ls)
        if side:  # contrast: B changes while A stays exact
            d.text(MX + MW / 2, ym + 1.2, mech, size=SMALL, color=shade(KIND["OBJECTIVE"], 0.1),
                   weight="bold")
            d.text(MX + MW / 2, ym - 1.3, "contrast: A stays", size=TINY, color=GREY,
                   style="italic")
        else:
            d.text(MX + MW / 2, ym, mech, size=SMALL)
        y -= ROW_GAP
    d.save(HERE / "figures/axisA_moves")


if __name__ == "__main__":
    cli(build)
