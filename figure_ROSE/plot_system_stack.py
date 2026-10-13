"""Generative-AI system stack (fig:system_stack, Sec. 6 of the ROSE survey), at acmsmall
\\textwidth (139.1 mm). Redraw of paper/figures/tikz/system_stack.tex.

    python figure_ROSE/plot_system_stack.py [--debug]

Writes figure_ROSE/figures/system_stack.{pdf,png}. Three lanes so that training objectives,
inference-time conditioning and the external environment are not confused. Node shape = kind of
component: solid = learned and updated; dashed = frozen; plain rectangle = fixed operation;
cylinder = external store; double border = environment; ellipse = datum. Small squares = evidence
status of an anchor (filled established, half observed, open untested), as in tab:stack.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from matplotlib.patches import Ellipse, FancyBboxPatch  # noqa: E402

from figkit import Box, register_font_files, shade, tint  # noqa: E402
from rose_style import (BODY, FAMILY, GRADFLOW, GREY, KIND, OKABE_ITO, TINY, W, cli,  # noqa: E402
                        diagram, status)

H = 54.5
HEAD = 6.6                      # lane heads (TikZ lanehead)
NOTE = 6.5                      # TikZ note / \figsmall size
INKC = "black"
VERM = shade(OKABE_ITO["vermillion"], 0.2)
CAL = "Libertinus Math"         # script T (\mathcal T); Libertine has no calligraphic alphabet
register_font_files(Path.home() / ".TinyTeX/texmf-dist/fonts/opentype/public/libertinus-fonts"
                    / "LibertinusMath-Regular.otf")

# style.tex node kinds
LEARNED = dict(fc=tint(OKABE_ITO["sky"], 0.18), ec=INKC, ls="-")
FROZEN = dict(fc=tint(OKABE_ITO["sky"], 0.06), ec=INKC, ls=(0, (5, 3)))
FIXED = dict(fc="white", ec=INKC, ls="-")
DATUM_FC = "#F5F5F5"            # black!4
LW = 0.5
PITCH = 2.55                    # line pitch inside a node (mm)

# ---------------------------------------------------------------- rich text (subscripts >= 6 pt)
TH = "θ"


def seg(s, **o):
    return (s, o)


def sub(s, bold=False, upright=False):
    """Subscript kept at 6 pt (style.tex \\figsub)."""
    return seg(s, size=TINY, dy=-0.62, style="normal" if upright else "italic",
               weight="bold" if bold else "normal")


def it(s, bold=False):
    return seg(s, style="italic", weight="bold" if bold else "normal")


def bf(s):
    return seg(s, weight="bold")


def cal(s="\U0001D4AF"):
    return seg(s, family=CAL)


def rich(d, x, y, segs, ha="center", size=BODY, color=INKC, z=7):
    """Lay out text segments on one baseline at y; returns the extent Box."""
    arts, wsum = [], 0.0
    for s in segs:
        s, o = (s, {}) if isinstance(s, str) else s
        a = d.ax.text(0, y + o.get("dy", 0), s, fontsize=o.get("size", size), ha="left",
                      va="baseline", color=o.get("color", color), zorder=z,
                      fontstyle=o.get("style", "normal"), fontweight=o.get("weight", "normal"),
                      family=o.get("family", FAMILY))
        wd = d.extent(a).w if s.strip() else d.text_width("x" + s + "x", size=o.get("size", size)) \
            - d.text_width("xx", size=o.get("size", size))
        arts.append((a, wd)); wsum += wd
    x0 = {"center": x - wsum / 2, "left": x, "right": x - wsum}[ha]
    for a, wd in arts:
        a.set_x(x0); x0 += wd
    return Box(x0 - wsum, y - 0.6, wsum, 2.4)


def width(d, segs, size):
    b = rich(d, 0, -50, segs, size=size)
    for a in list(d.ax.texts):
        if a.get_position()[1] < -40 or a.get_position()[1] < -40.0 + 0:
            a.remove()
    return b.w


def lines_at(d, cx, cy, lines, sizes):
    n = len(lines)
    for i, (ln, sz) in enumerate(zip(lines, sizes)):
        rich(d, cx, cy + (n - 1) / 2 * PITCH - i * PITCH - 0.78, ln, size=sz)


def node(d, cx, cy, lines, kind=LEARNED, w=None, h=6.2, minw=0.0, pad=1.3, lw=LW, sizes=None):
    """Rectangle node with 1-2 centred lines (second line at the note size)."""
    sizes = sizes or [BODY] + [NOTE] * (len(lines) - 1)
    w = w or max(minw, max(width(d, ln, s) for ln, s in zip(lines, sizes)) + 2 * pad)
    d.ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                  boxstyle="round,pad=0,rounding_size=0.55", fc=kind["fc"],
                                  ec=kind["ec"], lw=lw, ls=kind["ls"], zorder=3))
    lines_at(d, cx, cy, lines, sizes)
    return Box(cx - w / 2, cy - h / 2, w, h)


def datum(d, cx, cy, lines, w=None, h=None):
    sizes = [BODY] * len(lines)
    w = w or max(width(d, ln, s) for ln, s in zip(lines, sizes)) * 1.18 + 2.6
    h = h or (4.2 if len(lines) == 1 else 6.6)
    d.ax.add_patch(Ellipse((cx, cy), w, h, fc=DATUM_FC, ec=INKC, lw=LW, zorder=3))
    lines_at(d, cx, cy, lines, sizes)
    return Box(cx - w / 2, cy - h / 2, w, h)


def store(d, cx, cy, lines, w, h=6.4, cap=1.3):
    b = d.cylinder(cx - w / 2, cy - h / 2, w, h, fc=DATUM_FC, ec=INKC, lw=LW, cap=cap)
    lines_at(d, cx, cy - cap / 2 + 0.15, lines, [BODY, NOTE])
    return b


def env(d, cx, cy, lines, w, h=6.2, gap=0.55):
    for k, (fc, g) in enumerate((("white", gap), (DATUM_FC, 0.0))):
        d.ax.add_patch(FancyBboxPatch((cx - w / 2 - g, cy - h / 2 - g), w + 2 * g, h + 2 * g,
                                      boxstyle="round,pad=0,rounding_size=0.2", fc=fc, ec=INKC,
                                      lw=LW, zorder=3 + 0.1 * k))
    lines_at(d, cx, cy, lines, [BODY] * len(lines))
    return Box(cx - w / 2 - gap, cy - h / 2 - gap, w + 2 * gap, h + 2 * gap)


def note(d, x, y, segs, ha="center", color=GREY, size=NOTE):
    return rich(d, x, y, segs, ha=ha, size=size, color=color)


def lab(d, x, y, segs, ha="center", color=INKC):
    return rich(d, x, y, segs, ha=ha, size=NOTE, color=color)


def st_after(d, b, key, gap=0.9):
    """Evidence-status square right after a text extent."""
    status(d, b.x1 + gap + 0.8, b.y + 0.6 + 0.85, key)


def arrow(d, *pts, **kw):
    return d.arrow(*pts, lw=LW, head=2.5, **({"color": INKC} | kw))


# ---------------------------------------------------------------- layout
LANES = [(0.15, 57.9, KIND["OBJECTIVE"]), (59.3, 98.6, KIND["MODEL"]), (100.0, W - 0.15, GREY)]
YT, YB = H - 0.6, 8.0
RA, RB, RC, RD = 44.3, 34.9, 25.5, 16.3
Y_OUT = 11.6


def build(debug):
    d = diagram(H, debug)
    # lanes ----------------------------------------------------------------------------------
    heads = [("TRAINING OBJECTIVE", ["stages that update the weights ", it(TH, True)]),
             ("INFERENCE-TIME CONDITIONING", ["changes the context ", bf("c"), ", not ",
                                              it(TH, True)]),
             ("EXTERNAL ENVIRONMENT", ["transition ", cal(), ", not differentiable"])]
    for (xa, xb, c), (hd, sb) in zip(LANES, heads):
        d.ax.add_patch(FancyBboxPatch((xa, YB), xb - xa, YT - YB, boxstyle="square,pad=0",
                                      fc="#F7F7F7", ec="none", zorder=0))
        d.line((xa, YT), (xb, YT), color=c, lw=1.6)
        d.text(xa + 0.8, YT - 3.1, hd, size=HEAD, weight="bold", ha="left", va="baseline",
               color=INKC)
        note(d, xa + 0.8, YT - 6.0, sb, ha="left")

    # training lane --------------------------------------------------------------------------
    XT = 44.6
    pt = node(d, XT, RA, [["Pretraining"], ["chain-rule MLE on a corpus"]], minw=25.5)
    sft = node(d, XT, RB, [["Supervised fine-tuning"], ["conditional MLE on demos"]], w=pt.w)
    pol = node(d, XT, RC, [["Preference optimisation"],
                           ["policy ", it("π"), sub(TH, True), ": reward − ", it("β"), "·KL"]],
               w=pt.w)
    ref = node(d, 8.7, RC, [["reference ", it("π"), sub("ref", upright=True)], ["frozen"]], kind=FROZEN,
               minw=15)
    rm = node(d, 33.0, RD, [["reward model ", it("r"), sub("ψ", True)],
                            ["fit to comparisons"]])
    ver = node(d, 50.7, RD, [["verifier"], ["tests, answers"]], kind=FIXED, pad=1.0)
    note(d, pt.x - 1.0, RA - 0.8, ["anchor: none"], ha="right")

    arrow(d, pt.s, sft.n)
    b = lab(d, XT - 4.2, (pt.y + sft.y1) / 2 - 0.8, ["init. from pretrained ", it(TH, True)],
            ha="right")
    st_after(d, b, "untested", gap=1.0)
    arrow(d, sft.s, pol.n)
    arrow(d, sft.west, (ref.c[0], sft.c[1]), ref.n, ls=(0, (5, 3)))
    lab(d, 24.0, sft.c[1] + 0.7, ["copy"])
    d.line(ref.e, pol.west, color=INKC, lw=1.2)
    kb = lab(d, (ref.x1 + pol.x) / 2 - 0.9, RC + 0.7, ["KL anchor"])
    st_after(d, kb, "established", gap=0.6)
    for bx in (rm, ver):
        arrow(d, bx.n, (bx.c[0], pol.y), **GRADFLOW)
    lab(d, rm.c[0] - 0.8, (rm.y1 + pol.y) / 2 - 0.85, ["reward"], ha="right", color=VERM)
    sb = lab(d, ver.c[0] + 0.8, (rm.y1 + pol.y) / 2 - 0.85, ["score"], ha="left", color=VERM)
    st_after(d, sb, "observed", gap=0.5)

    # inference lane -------------------------------------------------------------------------
    idx = store(d, 65.1, RB, [["index"], ["external"]], w=9.6)
    ctx = datum(d, 86.3, RB, [["context ", bf("c")]])
    dep = node(d, ctx.c[0], RC, [["aligned ", it("π"), sub(TH, True)], ["weights fixed"]],
               kind=FROZEN, minw=20)
    smp = node(d, 80.0, RD, [["sampler: chain of thought,"], ["test-time search; not a density"]],
               kind=dict(FIXED, ec=shade(OKABE_ITO["orange"], 0.2)), lw=0.6, pad=1.0)
    arrow(d, idx.e, ctx.west)
    lab(d, (idx.x1 + ctx.x) / 2, RB + 0.8, ["passages"])
    note(d, LANES[1][0] + 0.8, idx.y - 2.4, ["frozen or refreshed"], ha="left")
    arrow(d, ctx.s, dep.n)
    arrow(d, pol.e, dep.west)
    lab(d, pol.x1 + 0.62 * (dep.x - pol.x1), RC + 0.8, ["weights"])
    arrow(d, (smp.c[0], dep.y), smp.n)

    # environment lane: agent loop -----------------------------------------------------------
    act = datum(d, 108.3, RC, [["action ", it("a"), sub("t")]])
    en = env(d, 127.8, RC, [["tool runtime,"], ["environment ", cal()]], w=17.0)
    obs = datum(d, 127.8, RB, [["observation"], [it("o"), sub("t")]], w=17.5)
    his = datum(d, act.c[0], RB, [["history"], [it("h"), sub("t")]], w=11.0)
    arrow(d, dep.e, act.west)
    arrow(d, act.e, en.west)
    arrow(d, en.n, obs.s)
    arrow(d, obs.west, his.e)
    arrow(d, his.west, ctx.e)
    note(d, (act.c[0] + en.c[0]) / 2 - 1.0, (RB + RC) / 2 - 0.45, ["agent loop"])
    nb = note(d, LANES[2][0] + 0.8, RD + 0.4, ["scaffold fixed; its role as an"], ha="left")
    nb2 = note(d, LANES[2][0] + 0.8, RD + 0.4 - 2.75, ["anchor is untested"], ha="left")
    st_after(d, nb2, "untested")
    arrow(d, en.s, (en.c[0], Y_OUT), (ver.c[0], Y_OUT), ver.s, **GRADFLOW)
    lab(d, smp.c[0], Y_OUT - 2.55, ["outcome reward, only when the agent is trained by RL "
                                    "(updates ", it(TH, True), ")"], color=VERM)

    # legend ---------------------------------------------------------------------------------
    y1, y2 = 4.95, 1.6
    x = 0.4
    for kind in ("learned", "frozen", "fixed operation", "external store", "environment",
                 "datum"):
        if kind == "external store":
            d.cylinder(x, y1 - 1.6, 3.4, 3.2, fc=DATUM_FC, ec=INKC, lw=LW, cap=0.9); wd = 3.4
        elif kind == "environment":
            env(d, x + 2.3, y1, [], w=3.4, h=2.0, gap=0.45); wd = 4.6; x += 0.0
        elif kind == "datum":
            d.ax.add_patch(Ellipse((x + 2.2, y1), 4.4, 2.6, fc=DATUM_FC, ec=INKC, lw=LW,
                                   zorder=3)); wd = 4.4
        else:
            k = {"learned": LEARNED, "frozen": FROZEN, "fixed operation": FIXED}[kind]
            d.ax.add_patch(FancyBboxPatch((x, y1 - 1.3), 4.4, 2.6,
                                          boxstyle="round,pad=0,rounding_size=0.45", fc=k["fc"],
                                          ec=k["ec"], lw=LW, ls=k["ls"], zorder=3)); wd = 4.4
        b = note(d, x + wd + 1.0, y1 - 0.8, ["learned, updated" if kind == "learned" else kind],
                 ha="left")
        x = b.x1 + 3.6
    arrow(d, (0.4, y2), (5.0, y2))
    b = note(d, 6.0, y2 - 0.8, ["data flow"], ha="left")
    arrow(d, (b.x1 + 3.6, y2), (b.x1 + 8.2, y2), **GRADFLOW)
    b = note(d, b.x1 + 9.2, y2 - 0.8, ["training signal"], ha="left")
    b = note(d, b.x1 + 6.0, y2 - 0.8, ["evidence that an anchor matters:"], ha="left")
    for key in ("established", "observed", "untested"):
        status(d, b.x1 + 2.3, y2, key)
        b = note(d, b.x1 + 3.8, y2 - 0.8, [key], ha="left")
        b = Box(b.x + 1.8, b.y, b.w, b.h)

    out = d.save(HERE / "figures/system_stack")
    print("wrote", out)


if __name__ == "__main__":
    cli(build)
