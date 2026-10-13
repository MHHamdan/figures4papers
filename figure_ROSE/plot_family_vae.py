"""VAE family architectures (supplement fig:vae_arch), ACM CSUR \\textwidth.

    python figure_ROSE/plot_family_vae.py [--debug]

Writes figures/family_vae.{pdf,png}. Redraws the TikZ of paper/supplement/06b_vae.tex:
(a) Gaussian VAE with the reparameterised sample, (b) VQ-VAE with nearest-code quantisation.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from rose_style import (GRADFLOW, GREY, INK, KIND, MUTED, OKABE_ITO, SMALL, W, access, cli,  # noqa: E402
                        diagram, kbox, shade, tint)

H = 51.5
C_M = KIND["MODEL"]
C_OBJ = shade(KIND["OBJECTIVE"], 0.15)
X, Z = r"\mathbf{x}", r"\mathbf{z}"
PHI, TH = r"\mathbfit{\phi}", r"\mathbfit{\theta}"
MU, SIG, EPS = r"\mathbfit{\mu}", r"\mathbfit{\sigma}", r"\mathbfit{\epsilon}"
CODE = [tint(OKABE_ITO[k], 0.6) for k in ("sky", "orange", "green", "yellow", "purple", "blue")]


def header(d, key, word):
    t = d.text(W - 0.5, H - 2.3, word, size=SMALL, ha="right")
    x = d.extent(t).x - 2.0
    access(d, x, H - 2.3, key)
    d.text(x - 2.0, H - 2.3, "density access", size=SMALL, color=MUTED, ha="right")


def panel(d, y, h, label):
    p = d.panel(0, y, W, h, GREY, fill=0.025, lw=0.45)
    d.tag(p.x + 1.2, p.y1 - 1.2, label, GREY, size=SMALL + 0.3, va="top", pad=0.25)
    return p


def data(d, xc, yc, s, w=9.0, h=5.6):
    return d.ellipse(xc - w / 2, yc - h / 2, w, h, s, fc=tint(INK, 0.08), ec=GREY, lw=0.5)


def op(d, x, y, s, r=1.6):
    from matplotlib.patches import Circle
    d.ax.add_patch(Circle((x, y), r, fc="white", ec=INK, lw=0.5, zorder=6))
    d.text(x, y - 0.05, s, size=SMALL, z=7)
    return d.rect(x - r, y - r, 2 * r, 2 * r, fc="none", ec="none", z=0)


def coder(d, x, yc, w, h, title, down=True):
    """Encoder / decoder block: kind box with a conv-stage glyph (shrinking or growing slabs)."""
    b = kbox(d, x, yc - h / 2, w, h)
    d.text(b.c[0], b.y1 - 2.1, title, size=SMALL)
    hs = [7.0, 5.2, 3.6] if down else [3.6, 5.2, 7.0]
    ds = [2.4, 3.4, 4.4] if down else [4.4, 3.4, 2.4]
    gw = 2 * 4.2 + 1.1 + ds[-1] * 0.55
    cx = b.c[0] - gw / 2
    for k, (hh, dd) in enumerate(zip(hs, ds)):
        d.cuboid(cx, b.y + 1.4, 1.1, hh, dd, C_M, z=4 + 0.1 * k)
        cx += 4.2
    return b


def gaussian_vae(d):
    p = panel(d, 26.5, 21.0, "(a)")
    ym, ys = 39.5, 31.5
    yc = (ym + ys) / 2
    x = data(d, 8.0, yc, f"${X}$")
    enc = coder(d, 17.0, yc, 22.0, 15.0, rf"encoder $q_{{{PHI}}}$")
    mu = kbox(d, 46.0, ym - 2.6, 9.0, 5.2, f"${MU}_{{{PHI}}}$")
    sg = kbox(d, 46.0, ys - 2.6, 9.0, 5.2, f"${SIG}_{{{PHI}}}$")
    pl, tm = op(d, 63.0, ym, "$+$"), op(d, 63.0, ys, r"$\odot$")
    eps = data(d, 76.5, ys, rf"${EPS}\sim\mathcal{{N}}(\mathbf{{0}},\mathbf{{I}})$", w=20.0)
    zc = d.cells(72.5, ym - 4.0, 1, 4, 2.0, [[tint(C_M, a)] for a in (0.35, 0.8, 0.55, 0.25)],
                 ec="white", lw=0.4)
    d.text(zc.x1 + 1.0, ym, f"${Z}$", size=None, ha="left")
    dec = coder(d, 86.0, ym - 1.0, 22.0, 13.0, rf"decoder $p_{{{TH}}}$", down=False)
    xh = data(d, 118.0, ym - 1.0, rf"$\hat{{{X}}}$")
    d.arrow(x.e, enc.west)
    d.arrow(enc.e, (43.5, yc), (43.5, ym), mu.west, r=1.0)
    d.arrow((43.5, yc), (43.5, ys), sg.west, r=1.0)
    d.arrow(mu.e, (pl.x, ym)); d.arrow(sg.e, (tm.x, ys))
    d.arrow(eps.west, (tm.x1, ys)); d.arrow((63.0, tm.y1), (63.0, pl.y))
    d.arrow((pl.x1, ym), (zc.x, ym)); d.arrow((zc.x1 + 3.6, ym), (dec.x, ym))
    d.arrow(dec.e, xh.west)
    d.text(zc.c[0] + 1.0, zc.y1 + 1.7, rf"${Z}={MU}+{SIG}\odot{EPS}$", size=SMALL)
    d.text(mu.c[0] - 1.0, mu.y1 + 1.8,
           rf"$D_{{\mathrm{{KL}}}}(q_{{{PHI}}}({Z}\mid{X})\,\Vert\,p({Z}))$", size=SMALL, color=C_OBJ)
    d.text(xh.c[0], xh.y - 2.0, rf"$\log p_{{{TH}}}({X}\mid{Z})$", size=SMALL, color=C_OBJ)


def vq_vae(d):
    p = panel(d, 0.0, 25.0, "(b)")
    yb = 11.0
    rng = np.random.default_rng(3)
    x = data(d, 8.0, yb, f"${X}$")
    enc = coder(d, 17.0, yb, 18.0, 13.0, "encoder")
    ze = d.cells(41.0, yb - 3.3, 3, 3, 2.2,
                 [[tint(INK, v) for v in rng.uniform(0.15, 0.6, 3)] for _ in range(3)], ec="white", lw=0.4)
    d.text(ze.c[0], ze.y1 + 1.7, rf"${Z}_e({X})$", size=SMALL)
    q = d.box(54.0, yb - 4.0, 31.0, 8.0, "nearest code",
              r"$k^\ast=\mathrm{arg\,min}_k\Vert\mathbf{z}_e-\mathbf{e}_k\Vert$",
              fc=tint(INK, 0.06), ec=GREY, lw=0.55, r=0.6, title_size=SMALL)
    idx = [[0, 3, 1], [2, 2, 5], [4, 0, 3]]
    zq = d.cells(90.0, yb - 3.3, 3, 3, 2.2, [[CODE[i] for i in r] for r in idx], ec="white", lw=0.4)
    d.text(zq.c[0], zq.y1 + 1.7, r"$\mathbf{z}_q=\mathbf{e}_{k^\ast}$", size=SMALL)
    dec = coder(d, 103.0, yb, 18.0, 13.0, "decoder", down=False)
    xh = data(d, 130.0, yb, rf"$\hat{{{X}}}$")
    # codebook: K learned vectors (columns)
    cb = d.cells(q.c[0] - 6.0, 19.8, 6, 1, 2.0, [CODE], ec="white", lw=0.4)
    kb = kbox(d, cb.x - 0.8, cb.y - 0.8, cb.w + 1.6, cb.h + 1.6, z=2)
    d.text(cb.x1 + 2.0, cb.c[1], r"codebook $\{\mathbf{e}_1,\dots,\mathbf{e}_K\}$", size=SMALL, ha="left")
    ym = (kb.y + q.y1) / 2  # two-headed (figkit's both=True fails for solid arrows)
    d.arrow((q.c[0], ym), (q.c[0], kb.y), head=2.6); d.arrow((q.c[0], ym), q.n, head=2.6)
    d.arrow(x.e, enc.west); d.arrow(enc.e, (ze.x, yb)); d.arrow((ze.x1, yb), q.west)
    d.arrow(q.e, (zq.x, yb)); d.arrow((zq.x1, yb), dec.west); d.arrow(dec.e, xh.west)
    # straight-through gradient: z_q -> z_e, around the quantiser
    yl = 3.4
    d.arrow(zq.s, (zq.c[0], yl), (ze.c[0], yl), (ze.c[0], ze.y), r=1.2, **GRADFLOW)
    d.text(q.c[0], yl - 1.6, "straight-through gradient", size=SMALL, color=C_OBJ,
           bbox=dict(fc=tint(GREY, 0.025), ec="none", pad=0.3))


def build(debug=False):
    d = diagram(H, debug)
    header(d, "bound", "evidence lower bound")
    gaussian_vae(d)
    vq_vae(d)
    print("\n".join(map(str, d.save(HERE / "figures/family_vae"))))


if __name__ == "__main__":
    cli(build)
