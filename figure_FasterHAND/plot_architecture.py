"""Faster-HAND architecture (Fig. 2), drawn at IEEE \\textwidth (181 mm).

    python figure_FasterHAND/plot_architecture.py [--debug]

Writes figures/architecture.{pdf,png}. Include in LaTeX with
    \\begin{figure*}[t] \\centering
      \\includegraphics[width=\\textwidth]{figures/architecture.pdf} ...
Shapes follow the trained page configuration (d = 256, 8 decoder layers,
4 heads, FFN 256, self-attention band w = 100, |V| = 100).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figkit import FONT_PT, IEEE, INK, MUTED, OKABE_ITO, Box, Diagram, shade, tint  # noqa: E402

# Okabe-Ito roles (colour-blind safe); solid = base model, dashed = optional mechanism.
C_ENC = OKABE_ITO["blue"]
C_MEM = "#9A4D8E"            # violet: feature grid / visual memory
C_DEC = "#1C2E62"            # navy: decoder
C_OUT = OKABE_ITO["green"]
C_A = OKABE_ITO["orange"]
C_B = OKABE_ITO["vermillion"]
SM, TINY = FONT_PT["small"], FONT_PT["tiny"]

W, H = IEEE["text"], 99.5
PANEL_Y, PANEL_H = 33.0, 66.0
MEM_ROWS = [tint(C_MEM, 0.35), tint(C_MEM, 0.6), tint(C_MEM, 0.85)]


def header(d, panel, num, label, color):
    d.tag(panel.x + 1.4, panel.y1 - 1.4, f"{num}  {label}", color, size=FONT_PT["body"], va="top")


def encoder(d):
    p = d.panel(0, PANEL_Y, 88.5, PANEL_H, C_ENC, fill=0.045)
    header(d, p, "1", "Visual encoder", C_ENC)

    # page image
    img = d.image(str(HERE / "assets/read2016_page_test_11.jpg"), 3.0, 62.0, 17.5)
    d.text(img.c[0], img.y - 2.2, "page image", size=SM)
    d.text(img.c[0], img.y - 4.6, r"$3\times H\times W$", size=TINY, color=MUTED)

    # FCN: 6 conv blocks (spatial shrinks, channels grow) + 4 depthwise-separable blocks
    yc, x = 76.0, 25.0
    d.arrow((img.x1 + 0.8, yc), (x - 0.9, yc))
    conv_h = [20, 16.5, 13.5, 11, 9, 7.5]
    conv_t = [0.8, 1.1, 1.4, 1.7, 2.0, 2.3]
    conv_d = [9, 8, 7, 6.5, 6, 6]
    xs = []
    for k, (h, t, dep) in enumerate(zip(conv_h, conv_t, conv_d)):
        d.cuboid(x, yc - h / 2, t, h, dep, C_ENC, z=3 + 0.1 * k)
        xs.append((x, x + t)); x += t + 1.5
    x += 1.0
    for k in range(4):
        d.cuboid(x, yc - 3.75, 2.6, 7.5, 6, shade(C_ENC, 0.25), z=4 + 0.1 * k)
        xs.append((x, x + 2.6)); x += 2.6 + 1.2
    fcn_x1 = xs[-1][1] + 6 * 0.55

    yb = yc - conv_h[0] / 2 - 1.2
    m = d.brace(xs[0][0], xs[5][1] + 1.0, yb)
    d.text(m[0], m[1] - 2.0, r"6$\times$ conv block", size=SM)
    d.text(m[0], m[1] - 4.4, r"3$\to$128 ch", size=TINY, color=MUTED)
    m = d.brace(xs[6][0], xs[9][1] + 1.0, yb)
    d.text(m[0], m[1] - 2.0, r"4$\times$ DS block", size=SM)
    d.text(m[0], m[1] - 4.4, r"128$\to$256 ch", size=TINY, color=MUTED)

    # feature grid f2D, rows coloured so the row-major flatten below is visible
    cell = 2.0
    gx = fcn_x1 + 4.5
    d.arrow((fcn_x1 + 0.6, yc), (gx - 0.9, yc))
    grid = d.cells(gx, yc - 1.5 * cell, 8, 3, cell, MEM_ROWS, ec="white", lw=0.4)
    d.text(grid.c[0], grid.y1 + 4.6, r"feature grid $\mathbfit{f}^{2D}$", size=SM)
    d.text(grid.c[0], grid.y1 + 2.0, r"$256\times\frac{H}{32}\times\frac{W}{8}$", size=TINY, color=MUTED)

    # + 2D positional encoding
    op = d.oplus(grid.c[0], 54.0)
    d.arrow(grid.s, (op.c[0], op.y1 + 0.15))
    pe_v = np.sin(np.add.outer(np.arange(3) * 1.3, np.arange(8) * 0.8))
    pe_c = [[tint(INK, 0.15 + 0.35 * (v + 1)) for v in row] for row in pe_v]
    pe = d.cells(op.x - 3.2 - 8 * 1.2, op.c[1] - 1.8, 8, 3, 1.2, pe_c, ec="white", lw=0.25)
    d.arrow((pe.x1 + 0.4, op.c[1]), (op.x - 0.1, op.c[1]))
    d.text(pe.c[0] - 0.6, pe.y - 2.0, "2D sinusoidal PE", size=TINY, color=MUTED)

    # flatten -> visual memory f1D (rows laid end to end)
    sc = 1.8
    seg = [MEM_ROWS[0]] * 8 + [MEM_ROWS[1]] * 8 + [MEM_ROWS[2]] * 8
    sx1 = 84.5
    sx0 = sx1 - 25 * sc
    mem = d.cells(sx0, 40.2, 25, 1, sc, [seg[:12] + [tint(INK, 0.0)] + seg[12:]], ec="white", lw=0.35)
    d.text(sx0 + 12.5 * sc, mem.c[1] - 0.05, r"$\cdots$", size=SM, color=MUTED)
    d.arrow((op.c[0], op.y - 0.15), (op.c[0], mem.y1 + 0.5))
    d.text(op.c[0] + 1.2, (op.y + mem.y1) / 2, "flatten", size=TINY, color=MUTED, ha="left")
    d.text(sx0 - 1.5, mem.c[1] + 1.1, r"visual memory $\mathbfit{f}^{1D}$", size=SM, ha="right",
           color=shade(C_MEM, 0.35))
    d.text(sx0 - 1.5, mem.c[1] - 1.5, r"$T_{\mathrm{enc}}\times 256$",
           size=TINY, ha="right", color=MUTED)
    return p, mem


def decoder(d, mem):
    p = d.panel(91.5, PANEL_Y, 48, PANEL_H, C_DEC, fill=0.045)
    header(d, p, "2", "Autoregressive decoder", C_DEC)
    bx, bw = 97.0, 37.0
    cx = bx + bw / 2

    d.text(cx, 36.0, r"previous symbols $y_{<t}$", size=SM, color=INK)
    emb = d.box(bx, 39.5, bw, 5.6, "symbol embedding", r"$+$ 1D sinusoidal PE", color=C_DEC)
    d.arrow((cx, 37.3), (cx, emb.y))

    # one decoder layer, repeated x8 (post-norm)
    g = Box(bx - 2.2, 48.5, bw + 4.4, 33.5)
    d.stack_shadow(g, n=2, step=(0.8, 0.8), color=C_DEC, r=1.6, z=0.6)
    d.panel(g.x, g.y, g.w, g.h, C_DEC, fill=0.07, z=0.7)
    d.tag(g.x1 - 1.6, g.y1, r"decoder layer $\times$8", C_DEC, size=SM, ha="right", va="center", pad=0.25)
    d.arrow(emb.n, (cx, g.y))

    def addnorm(y):
        return d.box(bx + 6, y, bw - 12, 2.4, "add & LayerNorm", color=tint(C_DEC, 0.5),
                     fill=0.12, title_size=TINY, title_weight="normal", lw=0.4, r=0.6, z=3)

    y = g.y + 2.0
    sa = d.box(bx, y, bw, 5.6, "causal self-attention", r"4 heads, band $w{=}100$", color=C_DEC,
               ha="left", pad_x=1.6)
    # band-mask glyph: lower-triangular with a finite band
    n, c = 6, 0.62
    mask = [[(C_DEC if 0 <= i - j < 3 else tint(C_DEC, 0.1)) for j in range(n)] for i in range(n)]
    d.cells(sa.x1 - 1.4 - n * c, sa.c[1] - n * c / 2, n, n, c, mask, ec="white", lw=0.2, z=5)
    an1 = addnorm(sa.y1 + 0.8)
    ca = d.box(bx, an1.y1 + 1.4, bw, 5.6, r"cross-attention to $\mathbfit{f}^{1D}$", "4 heads, full memory",
               color=C_DEC, ha="left", pad_x=1.6)
    an2 = addnorm(ca.y1 + 0.8)
    ff = d.box(bx, an2.y1 + 1.4, bw, 5.6, "feed-forward", r"$256\to256\to256$, ReLU", color=C_DEC,
               ha="left", pad_x=1.6)
    an3 = addnorm(ff.y1 + 0.8)
    for a, b in [(sa, an1), (an1, ca), (ca, an2), (an2, ff), (ff, an3)]:
        d.arrow((cx, a.y1), (cx, b.y), lw=0.5, head=2.6)

    head = d.box(bx, 87.0, bw, 6.0, "output projection", r"$256\to|V|{=}100$, softmax", color=C_DEC)
    d.arrow((cx, g.y1), head.s)
    d.text(cx + 1.0, (g.y1 + head.y) / 2, r"$h_t$", size=SM, ha="left")
    d.marker(cx - 2.6, (g.y1 + head.y) / 2, "B", C_B, ls=(0, (2, 1)))

    # visual memory -> cross-attention keys/values; marker A on the path
    xr = 89.9
    d.arrow(mem.e, (xr, mem.c[1]), (xr, ca.c[1]), ca.west, color=shade(C_MEM, 0.2), lw=0.7, r=1.2)
    d.text(92.4, ca.c[1] + 2.0, r"$K,V$", size=TINY, color=shade(C_MEM, 0.35), ha="left")
    d.marker(xr, (mem.c[1] + ca.c[1]) / 2, "A", C_A, ls=(0, (2, 1)))
    return p, head, emb


def stream(d, head):
    p = d.panel(142.5, PANEL_Y, W - 142.5, PANEL_H, C_OUT, fill=0.05)
    header(d, p, "3", "Text + layout stream", shade(C_OUT, 0.1))
    x0, pitch = p.x + 3.0, 3.15
    rows = [list("pn203N") + ["s", "a", "1", "7", "A"],
            ["b"] + list("Hiemit") + ["…", "B"],
            ["S", "s", "b", "…", "B", "S", "P"]]
    ys = [83.0, 79.2, 75.4]
    layout_open, layout_close = set("pnsba"), set("PNSBA")
    for r, y in zip(rows, ys):
        for k, t in enumerate(r):
            x = x0 + k * pitch
            if t in layout_open:
                d.chip(x, y, t, C_OUT)
            elif t in layout_close:
                d.chip(x, y, t, C_OUT, filled=True)
            elif t == "…":
                d.text(x, y, r"$\cdots$", size=SM, color=MUTED)
            else:
                d.chip(x, y, t, INK, square=True)
    # the symbol emitted at this step, appended to the stream below
    yt = head.e[1]
    xt = p.x + 4.5
    d.rect(xt - 1.4, yt - 1.4, 2.8, 2.8, fc="white", ec=C_OUT, lw=0.6, ls=(0, (1.6, 0.8)), r=0.5)
    d.text(xt, yt, r"$\hat y_t$", size=TINY)
    d.arrow(head.e, (xt - 1.6, yt))
    d.text(xt + 2.4, yt + 0.9, "emitted symbol, appended", size=TINY, color=MUTED, ha="left")
    d.text(xt + 2.4, yt - 1.4, "to the stream in reading order", size=TINY, color=MUTED, ha="left")

    # key
    ky = 65.0
    d.text(p.x + 3.0, ky + 2.6, "layout tokens: open / close", size=TINY, color=MUTED, ha="left")
    for k, (o, name) in enumerate([("p", "page"), ("n", "page number"), ("s", "section"),
                                   ("b", "body"), ("a", "annotation")]):
        y = ky - k * 3.3
        d.chip(p.x + 3.6, y, o, C_OUT)
        d.chip(p.x + 7.0, y, o.upper(), C_OUT, filled=True)
        d.text(p.x + 9.6, y, name, size=SM, ha="left")
    y = ky - 5 * 3.3
    d.chip(p.x + 3.6, y, "a", INK, square=True)
    d.chip(p.x + 7.0, y, "7", INK, square=True)
    d.text(p.x + 9.6, y, "characters", size=SM, ha="left")
    return p


def feedback(d, p_out, emb):
    x = p_out.x + 26.0
    y = 36.0
    d.arrow((x, 42.0), (x, y), (emb.c[0] + 13.5, y), color=MUTED, ls=(0, (3, 1.6)), r=1.2)
    d.text(x - 1.0, y + 1.5, r"fed back as $y_{<t+1}$", size=TINY, color=MUTED, ha="right")


def option_a(d):
    p = d.panel(0, 0, 56.0, 30.0, C_A, fill=0.06, ls=(0, (4, 2)))
    d.tag(p.x + 1.4, p.y1 - 1.4, "A  Shared visual K/V", shade(C_A, 0.1), va="top")
    sc = 1.6
    mem = d.cells(3.0, 11.8, 6, 1, sc, [[MEM_ROWS[k // 2] for k in range(6)]], ec="white", lw=0.3)
    d.text(mem.c[0], mem.y - 2.0, r"$\mathbfit{f}^{1D}$", size=SM, color=shade(C_MEM, 0.35))
    kv = d.box(16.5, 9.6, 12.5, 7.0, r"$W_K,\,W_V$", "one pair", color=shade(C_A, 0.1), fill=0.18)
    d.arrow(mem.e, kv.west, lw=0.6)
    xs, top = 37.0, 22.8
    for i in range(8):
        y = top - i * 2.2
        b = d.box(xs, y - 0.85, 6.0, 1.7, None, color=C_DEC, lw=0.45, r=0.4)
        d.text(b.c[0], b.c[1] - 0.03, str(i + 1), size=TINY - 0.8, color=C_DEC)
        d.arrow(kv.e, (b.x - 0.2, b.c[1]), lw=0.4, head=2.0, color=shade(C_A, 0.25))
    d.text(xs + 7.4, top - 3.5 * 2.2 + 1.2, "cross-attn", size=TINY, ha="left")
    d.text(xs + 7.4, top - 3.5 * 2.2 - 1.2, "layers 1–8", size=TINY, ha="left")
    d.text(3.0, 2.9, r"$K,V$ computed once, reused by all 8 layers", size=TINY,
           color=MUTED, ha="left")


def check(d, x, y, ok, s=0.9):
    if ok:
        d.line((x - s, y), (x - 0.25 * s, y - 0.7 * s), (x + s, y + 0.8 * s), color=OKABE_ITO["green"], lw=0.8)
    else:
        d.line((x - 0.7 * s, y - 0.7 * s), (x + 0.7 * s, y + 0.7 * s), color=C_B, lw=0.8)
        d.line((x - 0.7 * s, y + 0.7 * s), (x + 0.7 * s, y - 0.7 * s), color=C_B, lw=0.8)


def option_b(d):
    p = d.panel(59.0, 0, W - 59.0, 30.0, C_B, fill=0.05, ls=(0, (4, 2)))
    d.tag(p.x + 1.4, p.y1 - 1.4, "B  Speculative decoding", C_B, va="top")
    d.text(p.x + 39.0, p.y1 - 3.2, "inference only; output identical to greedy decoding", size=TINY,
           color=shade(C_B, 0.2), ha="left", style="italic")
    yc = 14.5
    d.text(p.x + 4.0, yc, r"$h_t$", size=SM)
    b1 = d.box(p.x + 8.0, yc - 4.5, 30.0, 9.0, "draft", "$m{-}1$ MLP heads propose\n" r"$\hat y_{t+1},\dots,\hat y_{t+m-1}$",
               color=C_B, fill=0.1)
    d.arrow((p.x + 5.6, yc), b1.west)
    # drafted symbols
    tx = b1.x1 + 4.0
    for k, t in enumerate("ers"):
        d.rect(tx + k * 3.1 - 1.3, yc + 1.6 - 1.3, 2.6, 2.6, fc="white", ec=C_B, lw=0.5, ls=(0, (1.6, 0.8)), r=0.5)
        d.text(tx + k * 3.1, yc + 1.55, t, size=SM)
    d.arrow((b1.x1, yc - 1.6), (b1.x1 + 13.0, yc - 1.6), color=C_B)
    b2 = d.box(b1.x1 + 13.0, yc - 4.5, 23.0, 9.0, "verify", "all drafts in one\ndecoder pass", color=C_B, fill=0.1)
    tx = b2.x1 + 4.0
    for k, (t, ok) in enumerate(zip("ers", (True, True, False))):
        d.chip(tx + k * 3.1, yc + 1.6, t, INK, square=True)
        check(d, tx + k * 3.1, yc + 4.6, ok, s=0.75)
    d.arrow((b2.x1, yc - 1.6), (b2.x1 + 13.0, yc - 1.6), color=C_B)
    b3 = d.box(b2.x1 + 13.0, yc - 4.5, 27.0, 9.0, "accept", "longest agreeing prefix,\nroll back the rest",
               color=C_B, fill=0.1)
    d.arrow(b3.s, (b3.c[0], 3.0), (b1.c[0], 3.0), b1.s, color=C_B, ls=(0, (3, 1.6)), r=1.2)
    d.text((b1.c[0] + b3.c[0]) / 2, 4.3, "resume after the last accepted symbol", size=TINY, color=shade(C_B, 0.2))


def main(debug=False):
    d = Diagram(W, H, font_size=FONT_PT["body"], debug_grid=debug)
    _, mem = encoder(d)
    _, head, emb = decoder(d, mem)
    p_out = stream(d, head)
    feedback(d, p_out, emb)
    option_a(d)
    option_b(d)
    out = d.save(HERE / "figures/architecture")
    print("\n".join(map(str, out)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="overlay a 5 mm grid")
    main(ap.parse_args().debug)
