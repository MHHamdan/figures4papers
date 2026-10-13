"""Diffusion family architecture (supplement fig:diffusion_arch), acmsmall \\textwidth (139.1 mm).

    python figure_ROSE/plot_family_diffusion.py [--debug]

Writes figures/family_diffusion.{pdf,png}. Redraws the TikZ figure of
paper/supplement/06g_diffusion.tex: (a) forward/reverse chain, (b) the CIFAR-10 U-Net of
diffusion/model.py, (c) latent diffusion with a ControlNet side branch. The chain thumbnails
are one CelebA-HQ sample of paper/figures/samples_ddpm1000_celeba_hq.png pushed through the
DDPM forward process q(x_t|x_0) (linear beta 1e-4..0.02, T = 1000) at t = 0, 1, 250, 1000.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from figkit import INK, MUTED, shade, tint  # noqa: E402
from rose_style import (DATAFLOW, GREY, KIND, PAPER, SANS, SMALL, TINY, W, access, cli,  # noqa: E402
                        diagram, kbox, kindword)

H = 105.5
SAMP = shade(KIND["SAMPLER"], 0.25)
MOD = shade(KIND["MODEL"], 0.3)
FROZEN = (0, (2.2, 1.2))
DATA = dict(fc=tint(INK, 0.06), ec=GREY, lw=0.55)
SKIP = dict(color=GREY, ls=(0, (2.2, 1.4)), lw=0.5)


def thumbs():
    img = mpl.image.imread(str(PAPER / "figures/samples_ddpm1000_celeba_hq.png"))[..., :3]
    r, c, step = 3, 3, 662 / 10
    x0 = img[int(r * step) + 2:int((r + 1) * step) - 1, int(c * step) + 2:int((c + 1) * step) - 1]
    x0 = x0 * 2 - 1
    betas = np.linspace(1e-4, 0.02, 1000)
    abar = np.cumprod(1 - betas)
    eps = np.random.default_rng(0).standard_normal(x0.shape)
    out = []
    for t in (0, 1, 250, 1000):
        xt = x0 if t == 0 else np.sqrt(abar[t - 1]) * x0 + np.sqrt(1 - abar[t - 1]) * eps
        out.append(np.clip((xt + 1) / 2, 0, 1))
    return out


def arc(d, a, b, rad, color, lw=0.6, head=True):
    d.ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}",
                                   arrowstyle="-|>,head_length=0.32,head_width=0.2" if head else "-",
                                   mutation_scale=10, color=color, lw=lw, shrinkA=0, shrinkB=0,
                                   zorder=6))


def ell(d, x, y, w, h, s, size=None):
    return d.ellipse(x - w / 2, y - h / 2, w, h, s, size=size, **DATA)


def header(d, y):
    t = d.text(1.0, y, "Diffusion model", weight="bold", ha="left")
    x = d.extent(t).x1 + 2.6
    access(d, x, y, "bound")
    access(d, x + 3.0, y, "exactR")
    d.text(x + 5.2, y, "density access: bound; exact via probability-flow ODE", size=SMALL,
           color=GREY, ha="left", family=SANS)


def sub(d, y, s):
    d.text(1.0, y, s, ha="left", va="center")


def chain(d, ytop):
    sub(d, ytop, r"$\mathbf{(a)}$ forward diffusion $q$ and learned reverse $p_{\boldsymbol{\theta}}$")
    s, w = 11.0, 9.5
    cx = lambda u: 12.0 + s * u                                   # noqa: E731
    yc = ytop - 12.5
    ims = thumbs()
    names = [r"$\mathbf{x}_0$", r"$\mathbf{x}_1$", r"$\mathbf{x}_t$", r"$\mathbf{x}_T$"]
    us = [0, 1.5, 4.3, 7.1]
    bx = []
    for u, im, nm in zip(us, ims, names):
        b = d.image(im, cx(u) - w / 2, yc - w / 2, w, border=GREY, lw=0.5)
        d.text(b.c[0], b.y1 - 0.45, nm, size=SMALL, ha="center", va="top", z=7,
               bbox=dict(boxstyle="round,pad=0.12,rounding_size=0.3", fc="white", ec="none",
                         alpha=0.85))
        bx.append(b)
    for u in (2.9, 5.7):
        d.text(cx(u), yc, r"$\cdots$")
    d.text(bx[3].x1 + 1.2, yc, r"$\approx\mathcal{N}(\mathbf{0},\mathbf{I})$", ha="left")
    up, dn = yc + 2.4, yc - 2.4
    G = tint(INK, 0.5)
    # forward q (fixed, grey) above; reverse p_theta (sampler) below, as the source's pieces
    segs = [(bx[0].x1, bx[1].x), (bx[1].x1, cx(2.6)), (cx(3.2), bx[2].x), (bx[2].x1, cx(5.4)),
            (cx(6.0), bx[3].x)]
    for a, b in segs:
        arc(d, (a, up), (b, up), -0.8, G)
        arc(d, (b, dn), (a, dn), -0.8, SAMP, lw=0.9)
    d.text((bx[0].x1 + bx[1].x) / 2, yc + 6.6, r"$q(\mathbf{x}_1|\mathbf{x}_0)$", size=SMALL,
           color=MUTED, va="bottom")
    d.text((cx(6.0) + bx[3].x) / 2, yc + 6.6,
           r"$\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}+\sqrt{\beta_t}\,\boldsymbol{\epsilon}$", size=SMALL,
           color=MUTED, va="bottom")
    d.text((bx[0].x1 + bx[1].x) / 2, yc - 6.6, r"$p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)$",
           size=SMALL, color=SAMP, va="top")
    d.text((cx(6.0) + bx[3].x) / 2, yc - 6.6,
           r"$\boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_t,t)$ from $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$",
           size=SMALL, color=SAMP, va="top")
    kindword(d, bx[3].x1 + 1.2, yc - 6.6, "SAMPLER", ha="left", va="top",
             text="SAMPLER (reverse chain)")


def unet(d, ytop):
    sub(d, ytop, r"$\mathbf{(b)}$ U-Net denoiser $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t,t,\mathbf{c})$")
    dy, bh = 7.2, 5.0
    xd, xu = 32.0, 76.0
    y1 = ytop - 5.5
    rows = [(r"Res$\times$2", r"Res$\times$3", "32^2", 17), ("Res+Attn", "Res+Attn", "16^2", 19),
            (r"Res$\times$2", r"Res$\times$3", "8^2", 15), (r"Res$\times$2", r"Res$\times$3", "4^2", 13)]
    dn, upb = [], []
    for k, (td, tu, res, w) in enumerate(rows):
        y = y1 - k * dy
        dn.append(kbox(d, xd - w / 2, y - bh / 2, w, bh, f"{td}, ${res}$"))
        upb.append(kbox(d, xu - w / 2, y - bh / 2, w, bh, f"{tu}, ${res}$"))
    ym = y1 - 3.75 * dy
    mid = d.box((xd + xu) / 2 - 12, ym - bh / 2, 24, bh, "Res+Attn+Res", fc=tint(KIND["MODEL"], 0.3),
                ec=shade(KIND["MODEL"], 0.2), lw=0.55, r=0.6)
    for a, b in zip(dn[:-1], dn[1:]):
        d.arrow(a.s, b.n, head=2.6)
    for a, b in zip(upb[1:], upb[:-1]):
        d.arrow(a.n, b.s, head=2.6)
    d.arrow(dn[3].s, (xd, mid.c[1]), mid.west, r=1.2, head=2.6)
    d.arrow(mid.e, (xu, mid.c[1]), upb[3].s, r=1.2, head=2.6)
    for a, b in zip(dn, upb):
        d.arrow(a.e, b.west, head=2.6, **SKIP)
    d.text((dn[0].x1 + upb[0].x) / 2, dn[0].c[1] + 0.5, "skip concat. (dashed)", size=TINY,
           color=MUTED, va="bottom")
    xin = ell(d, dn[0].x - 8.5, dn[0].c[1], 9, 5.2, r"$\mathbf{x}_t$")
    d.arrow(xin.e, dn[0].west)
    out = ell(d, upb[0].x1 + 8.5, upb[0].c[1], 9, 5.2, r"$\hat{\boldsymbol{\epsilon}}$")
    d.arrow(upb[0].e, out.west)
    # conditioning inputs, fed to the blocks from the right (source: between the columns)
    tw = 37.0
    tx = W - 1.0 - tw
    temb = kbox(d, tx, dn[1].c[1] - 3.6, tw, 7.2, "$t$ \u2192 sinusoidal \u2192 MLP,",
                "added in every ResBlock")
    cemb = kbox(d, tx, dn[3].c[1] - 3.6, tw, 7.2, r"$\mathbf{c}$: class / text tokens",
                "via cross-attn. or adaLN")
    xb = upb[0].x1 + 3.5
    for b in (temb, cemb):
        d.arrow(b.west, (xb, b.c[1]), color=MOD)
    d.line((xb, upb[1].c[1] + 1.5), (xb, upb[3].c[1] - 3.2), color=MOD, lw=0.8)
    d.text(xb + 0.8, upb[3].c[1] - 3.2, "to ResBlocks", size=TINY, color=MOD, ha="left", va="bottom")
    kindword(d, temb.x1, temb.y1 + 0.5, "MODEL")
    return mid


def latent(d, ytop):
    sub(d, ytop, r"$\mathbf{(c)}$ latent diffusion with cross-attention; ControlNet side branch")
    s = 10.6
    X0, yc = 9.0, ytop - 15.5
    P = lambda u: X0 + s * u                                        # noqa: E731
    X = ell(d, P(0), yc, 8, 5.6, r"$\mathbf{x}$")
    E = kbox(d, P(1.3) - 4, yc - 3, 8, 6, r"$\mathcal{E}$", ls=FROZEN)
    Z = ell(d, P(2.6), yc, 8, 5.6, r"$\mathbf{z}_0$")
    Zt = ell(d, P(4.4), yc, 8, 5.6, r"$\mathbf{z}_t$")
    U = kbox(d, P(6.4) - 9, yc - 3.7, 18, 7.4, "latent U-Net", "/ DiT", ls=FROZEN)
    Zh = ell(d, P(8.6), yc, 8, 5.6, r"$\hat{\mathbf{z}}_0$")
    D = kbox(d, P(9.9) - 4, yc - 3, 8, 6, r"$\mathcal{D}$", ls=FROZEN)
    Xh = ell(d, P(11.1), yc, 8, 5.6, r"$\hat{\mathbf{x}}$")
    for a, b in [(X, E), (E, Z), (Zt, U), (Zh, D), (D, Xh)]:
        d.arrow(a.e, b.west)
    d.arrow(Z.e, Zt.west, color=GREY)
    d.text((Z.x1 + Zt.x) / 2, yc + 0.8, "noise", size=SMALL, color=MUTED, va="bottom")
    d.arrow(U.e, Zh.west, color=SAMP, lw=0.8)
    d.text((U.x1 + Zh.x) / 2, yc + 0.8, r"$T\!\to\!0$", size=SMALL, color=SAMP, va="bottom")
    T = kbox(d, P(6.4) - 13, yc + 7.0, 26, 5.4, r"text encoder $\tau(\mathbf{c})$", ls=FROZEN)
    d.arrow(T.s, U.n, color=MOD)
    d.text(U.c[0] + 1.0, (T.y + U.y1) / 2, "cross-attn.", size=SMALL, color=MOD, ha="left")
    C = kbox(d, P(6.4) - 15, yc - 17.0, 30, 7.4, "ControlNet:", "trainable copy of encoder")
    cin = ell(d, C.x - 15.0, C.c[1], 24, 6.4, "edge / depth / pose", size=SMALL)
    d.arrow(cin.e, C.west)
    d.arrow(C.n, U.s)
    d.text(U.c[0] + 1.0, (C.y1 + U.y) / 2, "zero conv", size=SMALL, ha="left")
    d.text(C.x1 + 6.0, C.y1 + 3.0,
           "base U-Net frozen;\n" + r"$\mathcal{E},\mathcal{D}$ pretrained" + "\n(KL or VQ regularised)",
           size=SMALL, color=MUTED, ha="left", va="top", linespacing=1.1)
    # legend for the dashed border
    lx, ly = C.x1 + 6.0, C.y + 1.2
    d.rect(lx, ly - 1.4, 5.0, 2.8, fc=tint(KIND["MODEL"], 0.13), ec=shade(KIND["MODEL"], 0.2),
           lw=0.55, ls=FROZEN, r=0.5)
    d.text(lx + 6.2, ly, "dashed = frozen / pretrained", size=TINY, color=MUTED, ha="left")


def build(debug):
    d = diagram(H, debug)
    mpl.rcParams["mathtext.cal"] = "cmsy10"
    header(d, H - 3.0)
    chain(d, H - 8.5)
    unet(d, H - 33.0)
    latent(d, H - 71.0)
    d.save(HERE / "figures/family_diffusion")


if __name__ == "__main__":
    cli(build)
