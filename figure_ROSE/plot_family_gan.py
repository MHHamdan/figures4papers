"""GAN family architectures (supplement fig:gan_arch), ACM CSUR \\textwidth.

    python figure_ROSE/plot_family_gan.py [--debug]

Writes figures/family_gan.{pdf,png}: one figure with panels (a) DCGAN, (b) StyleGAN,
(c) CycleGAN, redrawing the three TikZ subfigures of paper/supplement/06c_gan.tex.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from rose_style import (GRADFLOW, GREY, INK, KIND, MUTED, SMALL, W, access, cli, diagram,  # noqa: E402
                        kbox, shade, tint)

H = 88.0
C_G, C_D = KIND["MODEL"], KIND["OBJECTIVE"]
C_OBJ = shade(C_D, 0.15)
TH, PSI = r"\mathbfit{\theta}", r"\mathbfit{\psi}"
Z, X = r"\mathbf{z}", r"\mathbf{x}"
BH = 9.0   # module box height


def header(d, key, word):
    t = d.text(W - 0.5, H - 2.3, word, size=SMALL, ha="right")
    x = d.extent(t).x - 2.0
    access(d, x, H - 2.3, key)
    d.text(x - 2.0, H - 2.3, "density access", size=SMALL, color=MUTED, ha="right")


def panel(d, y, h, label):
    p = d.panel(0, y, W, h, GREY, fill=0.025, lw=0.45)
    d.tag(p.x + 1.2, p.y1 - 1.2, label, GREY, size=SMALL + 0.3, va="top", pad=0.25)
    return p


def data(d, xc, yc, s, w=9.0, h=5.6, fc=None):
    return d.ellipse(xc - w / 2, yc - h / 2, w, h, s, fc=fc or tint(INK, 0.08), ec=GREY, lw=0.5)


def module(d, x, yc, w, title, sub=None, kind="MODEL", slab=None, h=BH):
    """Kind box, text left; optional conv-tensor slab (height, thickness) on the right."""
    b = kbox(d, x, yc - h / 2, w, h, title, sub, kind=kind, ha="left", pad_x=1.4, sub_size=SMALL)
    if slab:
        sh, st = slab
        dep = sh * 0.6
        d.cuboid(b.x1 - 1.3 - st - dep * 0.55, yc - (sh + dep * 0.4) / 2, st, sh, dep, KIND[kind], z=5)
    return b


def flow(d, *boxes):
    for a, b in zip(boxes[:-1], boxes[1:]):
        d.arrow(a.e, b.west)


def dcgan(d):
    p = panel(d, 50.0, 34.0, "(a)  DCGAN")
    yg, yd, xw = 74.0, 59.0, 21.0
    z = data(d, 7.0, yg, f"${Z}$", w=8.0)
    g1 = module(d, 15.0, yg, xw, "linear", r"$4{\times}4{\times}512$", slab=(2.4, 2.8))
    g2 = module(d, 40.5, yg, xw, "convT", r"$8{\times}8$", slab=(3.6, 2.0))
    g3 = module(d, 66.0, yg, xw, "convT", r"$16{\times}16$", slab=(5.0, 1.3))
    g4 = module(d, 91.5, yg, xw, "convT + tanh", r"$32{\times}32$", slab=(6.4, 0.5))
    gz = data(d, 123.0, yg, f"$G({Z})$", w=12.0)
    flow(d, z, g1, g2, g3, g4, gz)
    d.text((g1.x + g4.x1) / 2, g1.y1 + 1.8, rf"generator $G_{{{TH}}}$", size=SMALL,
           color=shade(C_G, 0.35))
    xr = data(d, 11.5, yd, rf"${X}\sim p_{{\mathrm{{data}}}}$", w=17.0)
    d1 = module(d, 25.0, yd, xw, r"conv$\downarrow$", "LeakyReLU", kind="OBJECTIVE", slab=(5.0, 1.0))
    d2 = module(d, 50.5, yd, xw, r"conv$\downarrow$", r"$\times3$", kind="OBJECTIVE", slab=(3.0, 2.2))
    d3 = module(d, 76.0, yd, xw, "linear", rf"$D_{{{PSI}}}(\cdot)$", kind="OBJECTIVE")
    flow(d, xr, d1, d2, d3)
    rf = d.text(d3.x1 + 4.5, yd, "real / fake", size=None, ha="left")
    d.arrow(d3.e, (d3.x1 + 3.8, yd))
    d.text((d1.x + d3.x1) / 2, d1.y - 1.8, rf"discriminator $D_{{{PSI}}}$", size=SMALL, color=C_OBJ)
    ym = (g1.y + d1.y1) / 2
    d.arrow(gz.s, (gz.c[0], ym), (d1.c[0], ym), d1.n, r=1.2)
    # the generator sees D only through its input gradient at generated samples
    xe = W - 3.0
    d.arrow((d.extent(rf).x1 + 0.8, yd), (xe, yd), (xe, yg), gz.e, r=1.2, **GRADFLOW)
    d.text(xe - 1.0, (ym + yd) / 2 - 0.4, rf"$\nabla_{{{X}}}D_{{{PSI}}}$", size=SMALL, color=C_OBJ,
           ha="right")


def stylegan(d):
    p = panel(d, 23.0, 25.0, "(b)  StyleGAN")
    yb = 33.0
    z = data(d, 7.0, yb, f"${Z}$", w=8.0)
    mp = module(d, 15.0, yb, 25.0, "mapping", "8 FC layers")
    for k in range(8):   # eight fully connected layers
        d.rect(mp.x1 - 10.2 + k * 1.2, yb - 2.6, 0.62, 5.2, fc=tint(C_G, 0.6), ec=shade(C_G, 0.25),
               lw=0.3, r=0.15, z=5)
    w = d.cells(45.5, yb - 3.6, 1, 4, 1.8, [[tint(C_G, a)] for a in (0.3, 0.75, 0.5, 0.2)], ec="white",
                lw=0.4)
    d.text(w.c[0], w.y1 + 1.6, r"$\mathbf{w}$", size=None)
    c1 = module(d, 60.0, yb, 23.0, r"const $4{\times}4$", "mod-conv", slab=(2.6, 2.2))
    c2 = module(d, 88.0, yb, 24.0, "up + mod-conv", r"to $1024^2$", slab=(6.4, 0.6))
    gz = data(d, 123.0, yb, f"$G({Z})$", w=12.0)
    flow(d, z, mp)
    d.arrow(mp.e, (w.x, yb)); d.arrow((w.x1, yb), c1.west)
    d.text((w.x1 + c1.x) / 2 + 0.6, yb + 1.6, "style", size=SMALL)
    flow(d, c1, c2, gz)
    yr = c1.y - 2.2
    d.arrow(w.s, (w.c[0], yr), (c2.c[0], yr), c2.s, r=1.2)
    for c in (c1, c2):
        nz = data(d, c.c[0], 43.0, "noise", w=12.0, h=5.0)
        d.arrow(nz.s, c.n)


def cyclegan(d):
    panel(d, 0.0, 21.0, "(c)  CycleGAN")
    yc, gap = 10.0, 6.5
    xa = data(d, 26.0, yc, r"$x\in X$", w=13.0)
    G = kbox(d, xa.x1 + gap, yc - 3.5, 11.0, 7.0, "$G$")
    yb = data(d, G.x1 + gap + 8.5, yc, r"$G(x)\in Y$", w=17.0)
    F = kbox(d, yb.x1 + gap, yc - 3.5, 11.0, 7.0, "$F$")
    xc = data(d, F.x1 + gap + 8.0, yc, r"$F(G(x))$", w=16.0)
    dy = kbox(d, yb.c[0] - 5.5, 14.4, 11.0, 5.4, "$D_Y$", kind="OBJECTIVE")
    flow(d, xa, G, yb, F, xc)
    d.arrow(yb.n, dy.s)
    yl = 3.6
    d.arrow(xc.s, (xc.c[0], yl), (xa.c[0], yl), xa.s, r=1.2, **GRADFLOW)
    d.arrow((xc.c[0], xc.y - 1.0), xc.s, color=GRADFLOW["color"])   # second head (<->)
    d.text(yb.c[0], yl, r"cycle consistency $\Vert F(G(x))-x\Vert_1$", size=SMALL, color=C_OBJ,
           bbox=dict(fc=tint(GREY, 0.025), ec="none", pad=0.4))


def build(debug=False):
    d = diagram(H, debug)
    header(d, "none", "no density")
    dcgan(d)
    stylegan(d)
    cyclegan(d)
    print("\n".join(map(str, d.save(HERE / "figures/family_gan"))))


if __name__ == "__main__":
    cli(build)
