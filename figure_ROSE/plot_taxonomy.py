"""Route-based taxonomy (Section 4, fig:taxonomy of the ROSE survey), at acmsmall \\textwidth.

    python figure_ROSE/plot_taxonomy.py [--debug]

Redraw of paper/figures/tikz/taxonomy.tex. Left: the ROSE chain for one trained component,
one object-kind colour per box. Right: one row per mechanism (axis B) with signal and divergence,
density access (axis A, one marker shape per value) and sampler/NFE; backbone (C) and modality (D)
chips. Bottom: a latent text-to-image composite with one (A, B, C, D) tuple per component.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from rose_style import (BODY, GREY, INK, KIND, SANS, SMALL, W, access, cli,  # noqa: E402
                        diagram, kbox, shade, tint)
from figkit import OKABE_ITO  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

H = 56.0
NOTE = 6.5                       # TikZ \fontsize{6.5}: notes, sub-lines, tuples
FROZEN_LS = (0, (2.4, 1.3))


def runs(d, x, y, parts, ha="left", va="baseline"):
    """Left-to-right text runs [(text, dict(size, weight, color, dy))]; returns the width (mm)."""
    ws = [d.text_width(t, size=o.get("size", BODY), weight=o.get("weight", "normal"))
          for t, o in parts]
    x0 = {"left": x, "center": x - sum(ws) / 2, "right": x - sum(ws)}[ha]
    for (t, o), w in zip(parts, ws):
        d.text(x0, y + o.get("dy", 0.0), t, size=o.get("size", BODY), weight=o.get("weight", "normal"),
               color=o.get("color", INK), ha="left", va=va)
        x0 += w
    return sum(ws)


def sup(base, exp, rest=""):
    """'10^3'-style run list with the exponent kept at 6 pt (TikZ \\figsup)."""
    out = [(base, {}), (exp, dict(size=6.0, dy=0.95))]
    return out + ([(rest, {})] if rest else [])


def chain(d):
    """Left: the ROSE chain, route -> evaluation; bold initials spell R-O-S-E."""
    items = [("MODEL", "Route", "mechanism (axis B)", True),
             ("OBJECTIVE", "Objective", "signal, divergence", True),
             ("ACCESS", "Reportable quantities", "density access (axis A)", False),
             ("SAMPLER", "Sampler", "NFE per sample", True),
             ("COST", "Cost", "training, per sample", False),
             ("METRIC", "Evaluation", "admissible metrics", True)]
    x, w, bh = 2.7, 22.9, 5.5
    top, bot = H - 0.35, 15.0
    gap = (top - bot - 6 * bh) / 5
    boxes = []
    for k, (kind, title, sub, rose) in enumerate(items):
        y = top - bh - k * (bh + gap)
        b = kbox(d, x, y, w, bh, kind=kind)
        tc = shade(KIND[kind], 0.45)
        parts = ([(title[0], dict(weight="bold", color=tc, size=7.6)), (title[1:], {})] if rose
                 else [(title, {})])
        runs(d, b.c[0], y + bh - 2.35, parts, ha="center")
        d.text(b.c[0], y + 1.05, sub, size=SMALL, color=GREY, va="baseline")
        boxes.append(b)
    for a, b in zip(boxes, boxes[1:]):
        d.arrow(a.s, b.n, head=2.4, lw=0.5)
    # light "ROSE chain" label: thin bracket + rotated sans word
    d.line((1.9, boxes[0].y1 - 0.3), (1.3, boxes[0].y1 - 0.3), (1.3, boxes[-1].y + 0.3),
           (1.9, boxes[-1].y + 0.3), color=tint(GREY, 0.55), lw=0.4)
    d.ax.add_patch(Rectangle(
        (0.55, (top + bot) / 2 - 6.5), 1.5, 13.0, fc="white", ec="none", zorder=5.5))
    d.text(1.3, (top + bot) / 2, "ROSE chain", size=SMALL, color=GREY, rotation=90,
           family=SANS)
    return boxes


def matrix(d):
    """Right: B x A matrix, legend, and the C/D chip rows. Returns the x extent."""
    xL, xR = 27.2, W - 0.3
    fam = ["autoregressive", "latent variable (VAE)", "adversarial (GAN)", "invertible flow",
           "energy-based", "diffusion / score / FM"]
    sig = [r"exact NLL $\cdot$ forward KL", r"ELBO $\cdot$ forward-KL bound",
           r"critic $\cdot$ JS or Wasserstein", r"exact NLL $\cdot$ forward KL",
           r"MLE via MCMC $\cdot$ forward KL", r"regression $\cdot$ weighted ELBO"]
    smp = [[(r"$D$ sequential", {})], [("1 (decoder)", {})], [("1", {})], [("1 (inverse)", {})],
           sup("10", "2", " MCMC steps"),
           sup("10–10", "3", "; 1–8 distilled/direct")]
    w_fam = max(d.text_width(s) for s in fam)
    w_sig = max(d.text_width(s) for s in sig)
    xS = xL + 0.4 + w_fam + 2.0
    dx = 7.5                                     # marker column pitch
    xa = xS + w_sig + 1.6 + 3.0
    xc = [xa + i * dx for i in range(4)]
    xN = xc[-1] + 3.2
    w_smp = d.text_width("10–10") + d.text_width("3", size=6.0) + d.text_width("; 1–8 distilled/direct")
    print(f"matrix slack: {xR - (xN + w_smp):.2f} mm")

    # column heads, each with a rule in the kind colour of the chain box it instantiates
    yr, yh = H - 0.45, H - 3.55
    for x0, x1, kind in [(xL, xS - 1.2, "MODEL"), (xS, xa - 3.6, "OBJECTIVE"),
                         (xa - 3.0, xc[-1] + 3.0, "ACCESS"), (xN, xR, "SAMPLER")]:
        d.line((x0, yr), (x1, yr), color=KIND[kind], lw=1.6)
    d.text(xL, yh, "B: route", weight="bold", ha="left")
    d.text(xS, yh, r"signal $\cdot$ divergence", weight="bold", ha="left")
    d.text((xc[0] + xc[-1]) / 2, yh, "A: density access", weight="bold")
    d.text(xN, yh, r"sampler $\cdot$ NFE", weight="bold", ha="left")
    for x, t in zip(xc, ["exact", "bound", "unnorm.", "none"]):
        d.text(x, H - 7.0, t, size=SMALL, color=GREY)

    ytop, rh = H - 8.6, 3.75
    d.line((xL, ytop), (xR, ytop), color=GREY, lw=0.4)
    ys = []
    for i in range(6):
        y = ytop - (i + 0.5) * rh
        ys.append(y)
        if i % 2 == 0:
            d.rect(xL, y - rh / 2, xR - xL, rh, fc="#F2F2F2", ec="none", r=0, z=1)
        yt = y - 0.15                           # optical centring (descenders)
        d.text(xL + 0.4, yt, fam[i], ha="left")
        d.text(xS, yt, sig[i], ha="left")
        runs(d, xN, yt, smp[i], va="center")
        for x in xc:
            d.ax.add_patch(Circle(
                (x, y), 0.25, fc="#B3B3B3", ec="none", zorder=4))
    for i, col, key in [(0, 0, "exact"), (1, 1, "bound"), (2, 3, "none"), (3, 0, "exact"),
                        (4, 2, "unnorm"), (5, 1, "bound"), (5, 0, "exactR"), (5, 3, "noneR")]:
        access(d, xc[col], ys[i] + (0.1 if key == "unnorm" else 0), key)
    ybot = ytop - 6 * rh
    d.line((xL, ybot), (xR, ybot), color=GREY, lw=0.4)

    # marker legend
    yl = ybot - 2.3
    leg = [("exact", r"exact $\log p$"), ("bound", "bound"), ("unnorm", "unnormalised"),
           ("none", "none (sampler only)"), ("exactR", "via ODE solve"),
           ("noneR", "few-step sampler")]
    gw = {"unnorm": 3.4, "none": 3.1, "noneR": 3.1}
    ws = [gw.get(k, 2.8) + d.text_width(t, size=NOTE) for k, t in leg]
    pitch = (xR - xL - sum(ws)) / (len(leg) - 1)
    x = xL
    for (key, t), w in zip(leg, ws):
        g = gw.get(key, 2.8)
        access(d, x + (g - 0.6) / 2, yl + (0.1 if key == "unnorm" else 0), key)
        d.text(x + g, yl, t, size=NOTE, color=GREY, ha="left")
        x += w + pitch

    # axes C and D: orthogonal chip rows
    chips = {"C: backbone": ["C1 masked/dilated CNN", "C2 recurrent", "C3 U-Net/ResNet",
                             "C4 transformer", "C5 state-space", "C6 other"],
             "D: modality": ["D1 image", "D2 text", "D3 audio", "D4 video", "D5 scientific",
                             "D6 multimodal", "D7 other"]}
    x0 = xL + max(d.text_width(k, weight="bold") for k in chips) + 1.8
    for (head, labels), y in zip(chips.items(), (yl - 4.4, yl - 8.4)):
        d.text(xL, y, head, weight="bold", ha="left")
        x = x0
        for s in labels:
            w = d.text_width(s, size=SMALL) + 1.3
            d.rect(x, y - 1.55, w, 3.1, fc="white", ec=GREY, lw=0.45, r=0.35, z=3)
            d.text(x + w / 2, y - 0.05, s, size=SMALL)
            x += w + 0.65
        print(f"{head} slack: {xR - (x - 0.65):.2f} mm")
    return yl - 8.4 - 1.55


def composite(d, ytop):
    """Bottom strip: latent text-to-image composite, one (A, B, C, D) tuple per component."""
    xR = W - 0.3
    d.line((0, ytop), (xR, ytop), color=INK, lw=0.6)
    yh = ytop - 2.0
    t = d.text(0, yh, "Composite (latent text-to-image):", weight="bold", ha="left")
    d.text(d.extent(t).x1 + 1.2, yh, "one (A, B, C, D) tuple per component", size=NOTE,
           color=GREY, ha="left")
    d.text(xR, yh, "solid: trained; dashed: frozen; arrows: inference data flow", size=NOTE,
           color=GREY, ha="right")

    nodes = [("datum", "prompt", None),
             ("frozen", "CLIP text encoder", "(–, –, C4, D2)"),
             ("learned", "latent denoiser", "(bound, diffusion, C3, D1)"),
             ("learned", "VAE decoder", "(none, VAE-type, C3, D1)"),
             ("datum", "image", None)]
    ws = [d.text_width(s) + 3.4 if kind == "datum"
          else max(d.text_width(s), d.text_width(tu, size=NOTE)) + 2.4 for kind, s, tu in nodes]
    gap = 4.5                                    # arrow length; components share the rest
    extra = (xR - 0.1 - sum(ws) - gap * (len(nodes) - 1)) / 3
    ws = [w + (extra if kind != "datum" else 0) for (kind, _, _), w in zip(nodes, ws)]
    print(f"composite arrow gap: {gap:.2f} mm")
    bh, yc = 6.6, (ytop - 4.0) / 2 + 0.1
    x, boxes = 0.1, []
    sky = OKABE_ITO["sky"]
    for (kind, s, tu), w in zip(nodes, ws):
        if kind == "datum":
            b = d.ellipse(x, yc - 2.2, w, 4.4, s, fc="#F5F5F5", ec=INK, lw=0.5)
        else:
            b = d.rect(x, yc - bh / 2, w, bh, fc=tint(sky, 0.18 if kind == "learned" else 0.06),
                       ec=INK, lw=0.5, ls="-" if kind == "learned" else FROZEN_LS, r=0.55)
            d.text(b.c[0], yc + 1.35, s)
            d.text(b.c[0], yc - 1.45, tu, size=NOTE)
        boxes.append(b)
        x += w + gap
    for a, b in zip(boxes, boxes[1:]):
        d.arrow((a.x1 + 0.25, yc), (b.x - 0.25, yc), head=2.4, lw=0.5)


def build(debug=False):
    d = diagram(H, debug)
    chain(d)
    ymat = matrix(d)
    composite(d, min(ymat, 15.0) - 1.4)
    out = d.save(HERE / "figures/taxonomy")
    print(*out, sep="\n")


if __name__ == "__main__":
    cli(build)
