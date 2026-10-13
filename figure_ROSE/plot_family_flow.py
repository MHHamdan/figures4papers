"""Normalizing-flow architectures (supplement fig:flow_arch), ACM CSUR \\textwidth.

    python figure_ROSE/plot_family_flow.py [--debug]

Writes figures/family_flow.{pdf,png}. Redraws the TikZ of paper/supplement/06d_flows.tex:
top, an affine coupling layer; bottom, the Real NVP multi-scale stack (flow/model.py).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from matplotlib.patches import Circle  # noqa: E402
from rose_style import (GREY, INK, KIND, MUTED, OKABE_ITO, SMALL, W, access, cli, diagram,  # noqa: E402
                        kbox, shade, tint)

H = 64.0
C_M = KIND["MODEL"]
C_A = tint(OKABE_ITO["blue"], 0.45)        # half that passes unchanged
C_B = tint(OKABE_ITO["yellow"], 0.4)      # half that is transformed
C_B2 = OKABE_ITO["yellow"]     # transformed half
TH = r"\mathbfit{\theta}"
SUM = r"{\sum}"


def header(d, key, word):
    t = d.text(W - 0.5, H - 2.3, word, size=SMALL, ha="right")
    x = d.extent(t).x - 2.0
    access(d, x, H - 2.3, key)
    d.text(x - 2.0, H - 2.3, "density access", size=SMALL, color=MUTED, ha="right")


def panel(d, y, h, label):
    p = d.panel(0, y, W, h, GREY, fill=0.025, lw=0.45)
    d.tag(p.x + 1.2, p.y1 - 1.2, label, GREY, size=SMALL + 0.3, va="top", pad=0.25)
    return p


def opbox(d, x, yc, w, title, sub=None, h=7.0):
    return d.box(x, yc - h / 2, w, h, title, sub, fc=tint(INK, 0.06), ec=GREY, lw=0.55, r=0.6,
                 title_size=SMALL if sub else None, title_weight="normal")


def op(d, x, y, s, r=1.6):
    d.ax.add_patch(Circle((x, y), r, fc="white", ec=INK, lw=0.5, zorder=6))
    d.text(x, y - 0.05, s, size=SMALL, z=7)
    return d.rect(x - r, y - r, 2 * r, 2 * r, fc="none", ec="none", z=0)


def vec(d, x, yc, cols, label, c=2.2, above=True):
    b = d.cells(x, yc - c / 2, len(cols), 1, c, [cols], ec="white", lw=0.45)
    d.text(b.c[0], (b.y1 + 1.7) if above else (b.y - 1.9), label, size=SMALL)
    return b


def coupling(d):
    panel(d, 29.5, 30.5, "affine coupling layer")
    ya, yb = 48.5, 37.5
    ym = (ya + yb) / 2
    h = vec(d, 3.5, ym, [C_A] * 4 + [C_B] * 4, r"$\mathbf{h}$")
    sp = opbox(d, h.x1 + 4.0, ym, 10.0, "split")
    ha = vec(d, 40.0, ya, [C_A] * 4, r"$\mathbf{h}_A$", above=False)
    hb = vec(d, 40.0, yb, [C_B] * 4, r"$\mathbf{h}_B$")
    net = kbox(d, 55.0, ya - 4.0, 22.0, 8.0, rf"$s_{{{TH}}},\,t_{{{TH}}}$", "(ResNet)", sub_size=SMALL)
    mul, add = op(d, 60.5, yb, r"$\odot$"), op(d, 72.0, yb, "$+$")
    hap = vec(d, 83.0, ya, [C_A] * 4, "$\\mathbf{h}$\u2032$_{A}$", above=False)
    hbp = vec(d, 83.0, yb, [C_B2] * 4, "$\\mathbf{h}$\u2032$_{B}$")
    cat = opbox(d, 98.0, ym, 12.0, "concat")
    hp = vec(d, cat.x1 + 4.5, ym, [C_A] * 4 + [C_B2] * 4, "$\\mathbf{h}$\u2032")
    d.arrow(h.e, sp.west)
    d.arrow(sp.e, (sp.x1 + 2.0, ym), (sp.x1 + 2.0, ya), ha.west, r=1.0)
    d.arrow((sp.x1 + 2.0, ym), (sp.x1 + 2.0, yb), hb.west, r=1.0)
    d.arrow(ha.e, net.west)
    yi = net.y1 + 2.8
    d.arrow(ha.n, (ha.c[0], yi), (hap.c[0], yi), (hap.c[0], hap.y1 + 0.3), r=1.2)
    d.text((ha.c[0] + hap.c[0]) / 2, yi + 1.3, "identity", size=SMALL, color=MUTED)
    d.arrow(hb.e, mul.west); d.arrow(mul.e, add.west); d.arrow(add.e, hbp.west)
    d.arrow((mul.c[0], net.y), mul.n); d.arrow((add.c[0], net.y), add.n)
    d.text(mul.c[0] - 1.0, (net.y + mul.y1) / 2, r"$\exp(s)$", size=SMALL, ha="right")
    d.text(add.c[0] + 1.0, (net.y + add.y1) / 2, "$t$", size=SMALL, ha="left")
    d.arrow(hap.e, (hap.x1 + 3.5, ya), (hap.x1 + 3.5, ym), cat.west, r=1.0)
    d.arrow(hbp.e, (hbp.x1 + 3.5, yb), (hbp.x1 + 3.5, ym), cat.west, r=1.0)
    d.arrow(cat.e, hp.west)
    d.text((mul.c[0] + add.c[0]) / 2, yb - 4.6, rf"$\log|\det J|={SUM}_j s_j$", size=SMALL)


def checker(d, x, y, n=4, c=1.0):
    return d.cells(x, y, n, n, c, [[C_A if (i + j) % 2 else C_B for j in range(n)] for i in range(n)],
                   ec="white", lw=0.25)


def multiscale(d):
    panel(d, 0.0, 27.5, "Real NVP multi-scale stack")
    yr, bw, gap = 12.0, 16.4, 3.3
    xs = [1.8 + k * (bw + gap) for k in range(7)]
    x = d.ellipse(xs[0], yr - 4.25, bw, 8.5, None, fc=tint(INK, 0.08), ec=GREY, lw=0.5)
    d.text(x.c[0], yr + 1.3, r"$\mathbf{x}$", size=None)
    d.text(x.c[0], yr - 1.6, r"$3{\times}32{\times}32$", size=SMALL)

    def net(k, t, s):
        return kbox(d, xs[k], yr - 4.25, bw, 8.5, t, s, sub_size=SMALL, title_size=SMALL)

    c1 = net(1, r"3$\times$ checker", "coupling")
    sq = opbox(d, xs[2], yr, bw, "squeeze", r"$12{\times}16{\times}16$", h=8.5)
    ch = net(3, r"3$\times$ channel", "coupling")
    fo = opbox(d, xs[4], yr, bw, "factor", r"out $\frac{1}{2}$", h=8.5)
    c2 = net(5, "scale 2", r"($6{\times}16{\times}16$)")
    c3 = net(6, "scale 3", "(final)")
    for a, b in zip([x, c1, sq, ch, fo, c2], [c1, sq, ch, fo, c2, c3]):
        d.arrow(a.e, b.west)
    for k, (b, s) in enumerate(zip((fo, c2, c3), ("1", "2", "3"))):
        zl = d.ellipse(b.c[0] - 4.5, 19.6, 9.0, 4.8, rf"$\mathbf{{z}}_{s}$", fc=tint(INK, 0.08), ec=GREY,
                       lw=0.5)
        d.arrow(b.n, zl.s)
    # mask glyphs: checkerboard, squeeze (2x2 -> 4 channels), channel-wise split
    checker(d, c1.c[0] - 2.0, c1.y1 + 1.4)
    g = d.cells(sq.c[0] - 4.6, sq.y1 + 1.9, 2, 2, 1.1, [[C_A, C_B], [C_B, C_A]], ec="white", lw=0.25)
    d.arrow((g.x1 + 0.4, g.c[1]), (g.x1 + 2.6, g.c[1]), lw=0.4, head=1.6)
    d.cells(g.x1 + 3.0, g.c[1] - 0.55, 4, 1, 1.1, [[C_A, C_B, C_B, C_A]], ec="white", lw=0.25)
    d.cells(ch.c[0] - 2.0, ch.y1 + 1.4, 4, 4, 1.0, [[C_A, C_A, C_B, C_B]] * 4, ec="white", lw=0.25)
    d.text(c2.c[0] - 4.0, 3.4,
           rf"$\log p_{{{TH}}}(\mathbf{{x}})={SUM}_i\log\mathcal{{N}}(\mathbf{{z}}_i;\mathbf{{0}},\mathbf{{I}})"
           rf"+{SUM}_k\log|\det J_k|$", size=SMALL)


def build(debug=False):
    d = diagram(H, debug)
    header(d, "exact", "exact likelihood")
    coupling(d)
    multiscale(d)
    print("\n".join(map(str, d.save(HERE / "figures/family_flow"))))


if __name__ == "__main__":
    cli(build)
