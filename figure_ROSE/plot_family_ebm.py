"""Energy-based family architecture (supplement fig:ebm_arch), acmsmall \\textwidth (139.1 mm).

    python figure_ROSE/plot_family_ebm.py [--debug]

Writes figures/family_ebm.{pdf,png}. Redraws the TikZ/pgfplots figure of
paper/supplement/06e_ebm_score.tex: left, the schematic energy curve (same function as the
pgfplots source) with the Langevin chain; right, the positive/negative-phase training loop.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from figkit import INK, MUTED, shade, tint  # noqa: E402
from rose_style import (DATAFLOW, GRADFLOW, GREY, KIND, SANS, SMALL, TINY, W, access, cli,  # noqa: E402
                        diagram, kbox, kindword)

H = 47.0
SAMP = shade(KIND["SAMPLER"], 0.25)
OBJ = KIND["OBJECTIVE"]


def energy(x):
    return (-1.1 * np.exp(-(x + 1.6) ** 2 / 0.35) - 0.8 * np.exp(-(x - 1.3) ** 2 / 0.5)
            + 0.15 * x ** 2 - 0.2)


def header(d, x, y, family, key, word):
    t = d.text(x, y, family, size=BODY_PT, weight="bold", ha="left", va="center")
    x1 = d.extent(t).x1 + 2.4
    access(d, x1, y, key)
    d.text(x1 + 2.2, y, word, size=SMALL, color=GREY, ha="left", va="center", family=SANS)


BODY_PT = 7.0


def energy_panel(d, x0, y0, w, h):
    """Inset matplotlib axes at (x0, y0, w, h) mm."""
    ax = d.fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    xs = np.linspace(-3.2, 3.2, 400)
    ax.plot(xs, energy(xs), color=shade(OBJ, 0.1), lw=0.9)
    chain = np.array([(2.9, 1.05), (2.45, 0.75), (2.0, 0.45), (1.75, 0.2), (1.55, -0.25),
                      (1.35, -0.62), (1.25, -0.85), (1.3, -0.9)])
    # source coordinates; the last points sit just under the curve there, so lift them onto it
    chain[:, 1] = np.maximum(chain[:, 1], energy(chain[:, 0]))
    ax.plot(chain[:, 0], chain[:, 1], color=SAMP, lw=0.8, zorder=4)
    ax.annotate("", xy=chain[-1], xytext=chain[-2], arrowprops=dict(
        arrowstyle="-|>,head_length=0.35,head_width=0.18", color=SAMP, lw=0.8, shrinkA=0, shrinkB=0))
    ax.plot(chain[:, 0], chain[:, 1], "o", ms=2.0, color=SAMP, mec="white", mew=0.3, zorder=5)
    ax.text(2.75, 1.12, "noise init", color=SAMP, ha="right", va="bottom", fontsize=SMALL)
    for xm in (-1.6, 1.3):
        ax.plot([xm], [energy(xm)], marker="o", ms=2.2, mfc=tint(GREY, 0.25), mec=GREY, mew=0.5,
                zorder=3)
    for xm in (-1.6, 1.3):
        ax.text(xm, energy(xm) - 0.12, "data mode", ha="center", va="top", fontsize=SMALL)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-1.4, 1.6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
    ax.plot(1, 0, ">k", ms=2.2, transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^k", ms=2.2, transform=ax.transAxes, clip_on=False)
    ax.set_xlabel(r"$\mathbf{x}$", fontsize=BODY_PT, labelpad=1.5)
    ax.set_ylabel(r"$E_{\boldsymbol{\theta}}(\mathbf{x})$", fontsize=BODY_PT, labelpad=1.5)
    ax.patch.set_alpha(0)


def build(debug):
    global H
    d = diagram(H, debug)
    mpl.rcParams["mathtext.cal"] = "cmsy10"

    header(d, 1.0, H - 3.0, "Energy-based model", "unnorm", "density access: unnormalised")

    # ---- left: energy surface and Langevin sampling (schematic)
    d.text(32.0, H - 8.0, "energy surface and Langevin sampling (schematic)", size=SMALL,
           color=INK)
    energy_panel(d, 7.0, 5.0, 50.0, H - 15.5)

    # ---- right: positive / negative phase
    bw, bh = 25.0, 7.6
    xL = 66.0
    xd = d.ellipse(xL, 31.0, bw, 8.6, fc=tint(INK, 0.06), ec=GREY, lw=0.55)
    d.text(xd.c[0], xd.c[1] + 1.4, r"$\mathbf{x}\sim p_{\mathrm{data}}$")
    d.text(xd.c[0], xd.c[1] - 1.7, "positive", size=SMALL, color=MUTED)

    xm = kbox(d, xL, 17.0, bw, bh, "$\\mathbf{x}$′$\\sim p_{\\boldsymbol{\\theta}}$", "negative (MCMC)",
              kind="SAMPLER")
    kindword(d, xm.x1, xm.y1 + 0.5, "SAMPLER")

    buf = d.cylinder(xL + 1.5, 1.5, bw - 3.0, 9.0, fc=tint(KIND["SAMPLER"], 0.08), ec=SAMP, lw=0.55,
                     cap=1.6)
    d.text(buf.c[0], buf.c[1] + 0.7, "replay buffer", size=SMALL)
    d.text(buf.c[0], buf.c[1] - 2.0, "(persistent chains)", size=SMALL, color=MUTED)
    d.arrow((xm.c[0], buf.y1 + 0.3), xm.s, color=SAMP, ls=(0, (3, 1.6)))

    E = kbox(d, 100.0, 24.0, 17.0, 10.0, r"$E_{\boldsymbol{\theta}}(\cdot)$",
             "CNN,\nscalar output", kind="OBJECTIVE")
    kindword(d, E.x1, E.y1 + 0.5, "OBJECTIVE")
    d.arrow(xd.e, (E.x - 3.0, xd.c[1]), (E.x - 3.0, E.c[1] + 2.0), (E.x, E.c[1] + 2.0), r=1.2,
            **DATAFLOW)
    d.arrow(xm.e, (E.x - 3.0, xm.c[1]), (E.x - 3.0, E.c[1] - 2.0), (E.x, E.c[1] - 2.0), r=1.2,
            **DATAFLOW)

    # -grad_x E drives Langevin (dashed, sampler colour)
    yl = 11.0
    d.arrow(E.s, (E.c[0], yl), (xm.x1 + 4.0, yl), (xm.x1 + 4.0, xm.c[1] - 1.8), (xm.x1, xm.c[1] - 1.8),
            color=SAMP, ls=(0, (3, 1.6)), r=1.2)
    d.text(E.c[0] + 1.2, yl, r"$-\nabla_{\mathbf{x}}E_{\boldsymbol{\theta}}$ drives Langevin",
           size=SMALL, color=shade(SAMP, 0.1), ha="left", va="center")

    upd = d.box(121.5, 21.5, 17.0, 15.0, None, fc="white", ec=shade(OBJ, 0.2), lw=0.55, r=0.6)
    d.text(upd.c[0], upd.y1 - 3.2, r"$\nabla_{\boldsymbol{\theta}}[E_{\boldsymbol{\theta}}(\mathbf{x})$",
           size=SMALL)
    d.text(upd.c[0], upd.y1 - 6.6, "$-\\,E_{\\boldsymbol{\\theta}}(\\mathbf{x}$′$)]$", size=SMALL)
    d.text(upd.c[0], upd.y + 3.6, "push down /", size=SMALL, color=MUTED)
    d.text(upd.c[0], upd.y + 1.5, "pull up", size=SMALL, color=MUTED)
    d.arrow(E.e, (upd.x, E.c[1]), **GRADFLOW)
    d.save(HERE / "figures/family_ebm")


if __name__ == "__main__":
    cli(build)
