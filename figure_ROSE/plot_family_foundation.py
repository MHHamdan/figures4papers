"""Foundation generative model families (supplement fig:foundation_arch),
acmsmall \\textwidth (139.1 mm).

    python figure_ROSE/plot_family_foundation.py [--debug]

Writes figures/family_foundation.{pdf,png}. Redraws the TikZ figure of
paper/supplement/06h_foundation.tex: (a) decoder-only LLM, (b) vision-language model,
(c) text-to-image generator, (d) video generator. Density-access markers follow the same
file: overview row "What is tractable" (base LLMs exact; other classes inherit per component)
and the "Density access" column of the model tables (exact text likelihood given images;
latent bound only; exact over tokens).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from figkit import INK, MUTED, Box, shade, tint  # noqa: E402
from rose_style import (GREY, KIND, SANS, SMALL, TINY, W, access, cli, diagram, kbox,  # noqa: E402
                        kindword)

H = 107.5
PW, PH, GAP = 68.5, 49.0, 2.1
FROZEN = (0, (2.2, 1.2))
DATA = dict(fc=tint(INK, 0.06), ec=GREY, lw=0.55)
MOD = shade(KIND["MODEL"], 0.3)
BW, B1, B2, G = 40.0, 5.2, 6.8, 2.5      # main box width, 1-line / 2-line heights, gap


def panel(d, x, y, title, marks, word):
    d.rect(x, y, PW, PH, fc=tint(INK, 0.025), ec=tint(INK, 0.35), lw=0.45, r=1.4, z=0)
    d.text(x + 1.6, y + PH - 2.8, title, ha="left")
    mx, my = x + 3.0, y + PH - 6.4
    for k in marks:
        access(d, mx, my, k)
        mx += 3.0
    d.text(mx - 0.6, my, word, size=TINY, color=GREY, ha="left", family=SANS)
    return x + 23.0 + BW / 2          # main column centre


def stack(d, cx, y0, specs):
    """specs: (kind, title, sub, ls); kind 'm' = MODEL kbox, 'd' = data ellipse."""
    out, y = [], y0
    for kind, title, sub, ls in specs:
        h = B2 if sub else B1
        if kind == "d":
            b = d.ellipse(cx - BW / 2, y, BW, h, None, **DATA)
            d.text(b.c[0], b.c[1], title)
        elif kind == "a":           # alternative path: lighter fill
            b = d.box(cx - BW / 2, y, BW, h, title, sub, fc=tint(KIND["MODEL"], 0.05),
                      ec=shade(KIND["MODEL"], 0.2), lw=0.55, ls=ls, r=0.6)
        else:
            b = kbox(d, cx - BW / 2, y, BW, h, title, sub, ls=ls)
        out.append(b)
        y += h + G
    return out


def chain(d, boxes, styles=None):
    for k, (a, b) in enumerate(zip(boxes[:-1], boxes[1:])):
        kw = (styles or {}).get(k, {})
        d.arrow(a.n, b.s, head=2.6, **kw)


def side(d, target, w, title, kind="d", h=None):
    h = h or (B2 if "\n" in title else B1 + 0.6)
    x = target.x - 3.0 - w
    if kind == "d":
        b = d.ellipse(x, target.c[1] - h / 2, w, h, None, **DATA)
    else:
        b = kbox(d, x, target.c[1] - h / 2, w, h, None, kind=kind)
    d.text(b.c[0], b.c[1], title, size=SMALL, linespacing=1.05)
    d.arrow(b.e, target.west, head=2.6, color=MOD if kind == "d" else shade(KIND[kind], 0.25))
    return b


def build(debug):
    d = diagram(H, debug)
    t = d.text(1.0, H - 3.0, "Foundation generative models", weight="bold", ha="left")
    d.text(d.extent(t).x1 + 2.4, H - 3.0,
           "density access: exact for base LLMs; other classes inherit it per component",
           size=SMALL, color=GREY, ha="left", family=SANS)
    xa, xb = 0.0, PW + GAP
    yt, yb = H - 6.5 - PH, H - 6.5 - 2 * PH - GAP
    y0 = 6.8

    # (a) decoder-only LLM
    cx = panel(d, xa, yt, r"$\mathbf{(a)}$ decoder-only LLM", ["exact"], "exact token likelihood")
    emb, b1, b2, out = stack(d, cx, yt + y0, [("m", "token emb. + RoPE", None, "-"),
                                              ("m", "causal MHA (GQA)", None, "-"),
                                              ("m", "gated MLP / MoE", None, "-"),
                                              ("m", "tied output softmax", None, "-")])
    # x L pre-norm blocks: group behind the two block rows, moved up to make room for the tag
    g = Box(b1.x - 2.0, b1.y - 1.2, BW + 4.0, b2.y1 - b1.y + 2.4)
    d.stack_shadow(g, n=2, step=(0.7, 0.7), color=KIND["MODEL"], r=1.2, z=0.5)
    d.rect(g.x, g.y, g.w, g.h, fc=tint(KIND["MODEL"], 0.05), ec=shade(KIND["MODEL"], 0.2), lw=0.45,
           r=1.2, z=0.6)
    d.text(g.x - 1.0, g.c[1], r"$\times L$ pre-norm" + "\nblocks", size=SMALL, ha="right",
           color=MOD, linespacing=1.05)
    chain(d, [emb, b1, b2, out])
    d.text(cx, yt + 3.3, r"$x_1\dots x_{i-1}\ \to\ p(x_i\mid\mathbf{x}_{<i})$", size=SMALL)

    # (b) VLM
    cx = panel(d, xb, yt, r"$\mathbf{(b)}$ vision-language model", ["exact"],
               "exact text likelihood given images")
    vit, conn, llm, ans = stack(d, cx, yt + y0, [
        ("m", "ViT encoder", "(frozen)", FROZEN),
        ("m", "connector: linear /", "Q-Former / resampler", "-"),
        ("m", "LLM", "(LoRA or gated cross-attn.)", "-"),
        ("d", "answer tokens", None, "-")])
    chain(d, [vit, conn, llm, ans])
    side(d, llm, 15.0, "prompt")
    d.text(cx, yt + 3.3, "image / frames", size=SMALL)

    # (c) text-to-image
    cx = panel(d, xa, yb, r"$\mathbf{(c)}$ text-to-image generator", ["bound", "exact"],
               "latent bound only; exact over tokens (AR)")
    te, gen, ar, dec = stack(d, cx, yb + y0, [
        ("m", "text encoder", "CLIP / T5 (frozen)", FROZEN),
        ("m", "latent U-Net / DiT", "(diffusion or RF)", "-"),
        ("a", "or: AR / masked", "transformer on VQ tokens", "-"),
        ("m", "VAE / VQ decoder", None, "-")])
    chain(d, [te, gen, ar, dec], {1: dict(ls=(0, (2.2, 1.4)))})
    cfg = side(d, gen, 15.0, "CFG\nscale $w$", kind="SAMPLER")
    kindword(d, cfg.c[0], cfg.y1 + 0.5, "SAMPLER", ha="center")
    d.text(cx, yb + 3.3, r"caption $\mathbf{c}$", size=SMALL)

    # (d) video
    cx = panel(d, xb, yb, r"$\mathbf{(d)}$ video generator", ["bound"], "latent bound only")
    v3d, pat, dit, vdec = stack(d, cx, yb + y0, [
        ("m", "video autoencoder", "(e.g. causal 3D VAE)", "-"),
        ("d", "space-time patches", None, "-"),
        ("m", "DiT: joint or factorised", "space-time attention", "-"),
        ("m", "3D decoder → frames", None, "-")])
    chain(d, [v3d, pat, dit, vdec])
    side(d, dit, 15.0, "text\ncond.")
    d.text(cx, yb + 3.3, "clip / image", size=SMALL)
    d.save(HERE / "figures/family_foundation")


if __name__ == "__main__":
    cli(build)
