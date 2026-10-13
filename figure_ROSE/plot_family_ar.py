"""Autoregressive family architectures (supplement fig:ar_arch), ACM CSUR \\textwidth.

    python figure_ROSE/plot_family_ar.py [--debug]

Writes figures/family_ar.{pdf,png}. Redraws the TikZ of paper/supplement/06a_autoregressive.tex:
(a) PixelCNN masked convolution, (b) WaveNet dilated causal convolution, (c) GPT causal mask.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from rose_style import (GREY, INK, KIND, MUTED, OKABE_ITO, SMALL, TINY, W, access, cli,  # noqa: E402
                        diagram, shade, tint)
from matplotlib.patches import Circle  # noqa: E402

H = 45.5
TOP = H - 5.0                       # panels below the figure header strip
C_VIS = tint(OKABE_ITO["blue"], 0.30)   # context visible to the conditional
C_CUR = OKABE_ITO["orange"]             # the variable being predicted
C_MASK = tint(INK, 0.16)
C_P = KIND["MODEL"]


def header(d, key, word):
    """Density-access marker + word, right-aligned in the header strip."""
    t = d.text(W - 0.5, H - 2.3, word, size=SMALL, ha="right")
    x = d.extent(t).x - 2.0
    access(d, x, H - 2.3, key)
    d.text(x - 2.0, H - 2.3, "density access", size=SMALL, color=MUTED, ha="right")


def panel(d, x, w, label):
    p = d.panel(x, 0, w, TOP, C_P, fill=0.04)
    d.tag(p.x + 1.2, p.y1 - 1.2, label, shade(C_P, 0.25), size=SMALL + 0.3, va="top", pad=0.25)
    return p


def swatch(d, x, y, fc, s=2.0):
    d.rect(x, y - s / 2, s, s, fc=fc, ec=GREY, lw=0.35, r=0.15, z=4)


def pixelcnn(d):
    p = panel(d, 0, 37.0, "(a)  PixelCNN")
    c, n = 4.0, 5
    gx, gy = p.c[0] - n * c / 2, 14.5
    cols = [[C_VIS if i * n + j < 12 else (C_CUR if i * n + j == 12 else "white") for j in range(n)]
            for i in range(n)]
    g = d.cells(gx, gy, n, n, c, cols, ec=tint(INK, 0.45), lw=0.35)
    # raster scan order over the visible pixels
    for i in range(3):
        y = g.y1 - (i + 0.5) * c
        x1 = g.x + (n - 0.35) * c if i < 2 else g.x + 1.65 * c
        d.arrow((g.x + 0.35 * c, y), (x1, y), color=shade(OKABE_ITO["blue"], 0.3), lw=0.35, head=1.8,
                z=6)
    d.text(g.x + 2.5 * c, g.y + 2.5 * c, r"$x_i$", size=SMALL, z=7)
    lx, ly = p.x + 2.5, 11.4
    swatch(d, lx, ly, C_CUR)
    d.text(lx + 3.0, ly, r"$x_i$", size=SMALL, ha="left")
    swatch(d, lx, ly - 3.4, C_VIS)
    d.text(lx + 3.0, ly - 3.4, r"$\mathbf{x}_{<i}$ visible to the kernel", size=SMALL, ha="left")
    d.text(lx + 3.0, ly - 6.1, "(raster order)", size=SMALL, ha="left", color=MUTED)
    swatch(d, lx, ly - 9.0, "white")
    d.text(lx + 3.0, ly - 9.0, "masked", size=SMALL, ha="left")


def wavenet(d):
    p = panel(d, 39.0, 61.0, "(b)  WaveNet")
    pitch, x0, ys = 5.4, p.x + 12.0, [15.5, 21.5, 27.5, 33.5]
    dil = [1, 2, 4]
    # receptive field of the last output
    rf = [set(range(8)), {1, 3, 5, 7}, {3, 7}, {7}]
    hot = shade(OKABE_ITO["blue"], 0.1)
    for l, dl in enumerate(dil):
        for t in range(8):
            for s in (t, t - dl):
                if s < 0:
                    continue
                on = t in rf[l + 1] and s in rf[l]
                d.line((x0 + s * pitch, ys[l]), (x0 + t * pitch, ys[l + 1]),
                       color=hot if on else tint(INK, 0.35), lw=0.7 if on else 0.35, z=3 + on)
        d.text(x0 + 7 * pitch + 3.6, (ys[l] + ys[l + 1]) / 2, rf"$d{{=}}{dl}$", size=SMALL, ha="left")
    for l, y in enumerate(ys):
        for t in range(8):
            out = l == 3
            fc = C_CUR if out else tint(OKABE_ITO["blue"], 0.6 if t in rf[l] else 0.18)
            d.ax.add_patch(Circle((x0 + t * pitch, y), 0.85, fc=fc, ec=shade(fc, 0.35) if out else
                                  shade(OKABE_ITO["blue"], 0.2), lw=0.4, zorder=6))
    d.text(x0 - 2.6, ys[0], "input", size=SMALL, color=MUTED, ha="right")
    d.text(x0 - 2.6, ys[3], "output", size=SMALL, color=MUTED, ha="right")
    d.text(p.c[0], 6.6, "receptive field doubles per layer;", size=SMALL)
    d.text(p.c[0], 3.9, "each output sees only past samples", size=SMALL)


def gpt(d):
    p = panel(d, 102.0, W - 102.0, "(c)  GPT")
    c, n = 3.5, 6
    gx, gy = p.x + 7.5, 13.5
    cols = [[C_VIS if j <= i else C_MASK for j in range(n)] for i in range(n)]
    g = d.cells(gx, gy, n, n, c, cols, ec="white", lw=0.5)
    d.text(g.x - 1.8, g.c[1], r"query $i$", size=SMALL, rotation=90)
    d.text(g.c[0], g.y - 1.8, r"key $j$", size=SMALL)
    d.text(g.x1 + 1.2, g.y + 1.0, r"$j\leq i$", size=SMALL, ha="left",
           color=shade(OKABE_ITO["blue"], 0.3))
    d.text(g.x1 + 1.2, g.y1 - 1.0, "masked", size=SMALL, ha="left", color=MUTED)
    x = p.x + 2.0
    d.text(x, 8.6, r"mask $j\leq i$: all $N$ conditionals", size=SMALL, ha="left")
    d.text(x, 5.9, "in one parallel pass; sampling", size=SMALL, ha="left")
    d.text(x, 3.2, "remains sequential with a KV cache", size=SMALL, ha="left")


def build(debug=False):
    d = diagram(H, debug)
    header(d, "exact", "exact likelihood")
    pixelcnn(d)
    wavenet(d)
    gpt(d)
    print("\n".join(map(str, d.save(HERE / "figures/family_ar"))))


if __name__ == "__main__":
    cli(build)
