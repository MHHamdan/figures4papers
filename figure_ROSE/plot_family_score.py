"""Score-based family: noise-conditional score fields (supplement fig:score_arch),
acmsmall \\textwidth (139.1 mm).

    python figure_ROSE/plot_family_score.py [--debug]

Writes figures/family_score.{pdf,png}. Redraws the TikZ figure of
paper/supplement/06e_ebm_score.tex: the arrow field is computed with numpy from exactly the
pgfmath expressions of the source (5 x 5 grid, two modes, weights exp(-d^2/(2 sigma^2 + 0.3)),
length min(0.45, 0.22/sigma * min(|v|, 1.2)/1.2 + 0.08)); one TikZ cm = S mm here.
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from figkit import INK, shade, tint  # noqa: E402
from rose_style import BODY, GREY, KIND, SANS, SMALL, W, access, cli, diagram, kindword  # noqa: E402

S = 10.0                       # mm per TikZ cm
H = 58.5
OX = (W - 13.0 * S) / 2 + 1.9 * S   # TikZ x=0 of the first panel
OY = 30.0                      # TikZ y=0
FIELD = shade(KIND["MODEL"], 0.35)
SAMP = shade(KIND["SAMPLER"], 0.25)
MODES = [(-0.9, -0.5, 0.32), (0.9, 0.7, 0.28)]
SIGMAS = [(1.2, r"large $\sigma_1$", 0.0), (0.6, r"medium $\sigma_2$", 4.6),
          (0.25, r"small $\sigma_3$", 9.2)]


def tk(x, y):
    return (OX + S * x, OY + S * y)


def field(s):
    """Arrow starts and vectors in TikZ cm, as the source's pgfmath loop."""
    out = []
    for px in (-1.6, -0.8, 0, 0.8, 1.6):
        for py in (-1.6, -0.8, 0, 0.8, 1.6):
            dxa, dya = -0.9 - px, -0.5 - py
            dxb, dyb = 0.9 - px, 0.7 - py
            da, db = np.hypot(dxa, dya), np.hypot(dxb, dyb)
            wa = np.exp(-da * da / (2 * s * s + 0.3))
            wb = np.exp(-db * db / (2 * s * s + 0.3))
            vx = (wa * dxa + wb * dxb) / (wa + wb + 0.001)
            vy = (wa * dya + wb * dyb) / (wa + wb + 0.001)
            nrm = np.hypot(vx, vy) + 0.001
            ln = min(0.45, 0.22 / s * min(nrm, 1.2) / 1.2 + 0.08)
            out.append((px, py, ln * vx / nrm, ln * vy / nrm))
    return out


def curve(d, pts, codes, ls="-", head=True, lw=0.9):
    path = MPath([tk(*p) for p in pts], codes)
    style = "-|>,head_length=0.36,head_width=0.22" if head else "-"
    d.ax.add_patch(FancyArrowPatch(path=path, arrowstyle=style, mutation_scale=10, color=SAMP, lw=lw,
                                   ls=ls, zorder=7, shrinkA=0, shrinkB=0, capstyle="butt"))


def build(debug):
    import matplotlib as mpl
    d = diagram(H, debug)
    mpl.rcParams["mathtext.cal"] = "cmsy10"   # calligraphic N as in the paper
    # header: family + density-access markers
    t = d.text(1.0, H - 3.0, "Score-based model", weight="bold", ha="left")
    x = d.extent(t).x1 + 2.6
    access(d, x, H - 3.0, "bound")
    access(d, x + 3.0, H - 3.0, "exactR")
    d.text(x + 5.2, H - 3.0, "density access: bound; exact via probability-flow ODE", size=SMALL,
           color=GREY, ha="left", family=SANS)

    for s, lab, dx in SIGMAS:
        x0, y0 = tk(dx - 1.9, -1.9)
        d.ax.add_patch(Rectangle((x0, y0), 3.8 * S, 3.8 * S, fc="white", ec=tint(INK, 0.4), lw=0.5,
                                 zorder=1))
        for mx, my, r in MODES:
            d.ax.add_patch(Circle(tk(dx + mx, my), r * S, fc=tint(GREY, 0.28), ec="none", zorder=2))
        for px, py, vx, vy in field(s):
            d.ax.add_patch(FancyArrowPatch(tk(dx + px, py), tk(dx + px + vx, py + vy),
                                           arrowstyle="-|>,head_length=0.22,head_width=0.13",
                                           mutation_scale=10, color=FIELD, lw=0.5, zorder=4,
                                           shrinkA=0, shrinkB=0))
        d.text(x0, y0 + 3.8 * S + 0.6, lab, ha="left", va="bottom")
    kindword(d, tk(9.2 + 1.9, 1.95)[0], tk(0, 1.95)[1] + 0.2, "MODEL",
             text=r"MODEL  $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x},\sigma_i)$")

    # annealed Langevin path spanning the three panels (source control points)
    C4, L = MPath.CURVE4, MPath.LINETO
    curve(d, [(1.5, 1.6), (0.5, 1.4), (-0.2, 0.6), (-0.3, 0.3)], [MPath.MOVETO, C4, C4, C4])
    curve(d, [(-0.3, 0.3), (4.6 - 0.3, 0.3)], [MPath.MOVETO, L], ls=(0, (3, 1.6)))
    curve(d, [(4.3, 0.3), (4.0, -0.1), (3.8, -0.3), (3.6, -0.4)], [MPath.MOVETO, C4, C4, C4])
    curve(d, [(3.6, -0.4), (9.2 - 0.9, -0.4)], [MPath.MOVETO, L], ls=(0, (3, 1.6)))
    # source's last segment is degenerate (8.3,-0.4) -> (8.3,-0.5): short drop onto the sample
    curve(d, [(8.3, -0.4), (8.3, -0.5)], [MPath.MOVETO, L], head=False)
    d.text(*tk(1.9, 1.95), r"$\mathbf{x}_0\sim\mathcal{N}(0,\sigma_1^2 I)$", color=SAMP, ha="right",
           va="bottom")
    d.ax.add_patch(Circle(tk(8.3, -0.5), 0.75, fc=SAMP, ec="none", zorder=8))
    d.line(tk(8.3, -0.5), tk(8.3, -1.98), color=SAMP, lw=0.5, z=7)
    d.text(*tk(8.3, -2.02), "sample", color=SAMP, va="top", z=9,
           bbox=dict(fc="white", ec="none", pad=0.6))
    eq = d.text(tk(4.6, 0)[0] + 6.0, 3.0,
           r"annealed Langevin dynamics: $\mathbf{x}\leftarrow\mathbf{x}+\frac{\kappa_i}{2}\,"
           r"\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x},\sigma_i)+\sqrt{\kappa_i}\,\boldsymbol{\epsilon}$,"
           r"  $\kappa_i\propto\sigma_i^2$, for $i=1,\dots,L$", va="center")
    kindword(d, d.extent(eq).x - 1.5, 3.0, "SAMPLER", va="center")
    d.save(HERE / "figures/family_score")


if __name__ == "__main__":
    cli(build)
