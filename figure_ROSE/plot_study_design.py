"""Design of the common-protocol empirical study (ROSE survey, Section 8, experiments E1-E5).

    python figure_ROSE/plot_study_design.py [--debug]

Writes figure_ROSE/figures/study_design.{pdf,png} at acmsmall \\textwidth (139.1 mm).
Content follows paper/main/08_benchmark.tex; every number that has a \\newcommand in
paper/tables/macros_*.tex or paper/preamble.tex is read from there at build time.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from figkit import INK, MUTED, shade, tint  # noqa: E402
from rose_style import (BODY, GREY, KIND, PAPER, SANS, SMALL, TINY, W, access, cli,  # noqa: E402
                        diagram, kbox)

H = 49.0


# ----------------------------------------------------------------------------- macros
def load_macros():
    """``\\newcommand{\\name}{value}`` (brace-free values) from the paper's macro files."""
    files = sorted((PAPER / "tables").glob("macros_*.tex")) + [PAPER / "preamble.tex"]
    rx = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}")
    out = {}
    for f in files:
        for name, val in rx.findall(f.read_text()):
            out[name] = val.strip()
    return out


M = load_macros()


def m(name, thousands=False):
    v = M[name]
    if thousands and v.isdigit():
        v = f"{int(v):,}"
    return v


# ----------------------------------------------------------------------------- helpers
LH_S = SMALL * 0.3528 * 1.18     # line pitch of a small line (mm)


def pill(d, x, y, s, kind, ha="right", filled=True, size=TINY):
    c = KIND.get(kind, GREY)
    return d.text(x, y, s, size=size, color="white" if filled else shade(c, 0.3), ha=ha, va="center",
                  weight="bold", family=SANS, z=9,
                  bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.35",
                            fc=shade(c, 0.15) if filled else "white", ec=shade(c, 0.15), lw=0.45))


def fit(d, s, size, maxw, what=""):
    """Warn when a (multi-line) label is wider than the room it is drawn in."""
    for line in s.split("\n"):
        tw = d.text_width(line, size=size)
        if tw > maxw:
            print(f"OVERFLOW {what}: {tw:.1f} > {maxw:.1f} mm: {line!r}")


def header(d, p, num, label, kind):
    d.tag(p.x + 1.2, p.y1 - 1.1, f"{num}  {label}", shade(KIND[kind], 0.15), size=BODY, va="top")


# ----------------------------------------------------------------------------- panel 1
def models(d, x0, w, top, bot):
    p = d.panel(x0, bot, w, top - bot, KIND["MODEL"], fill=0.04)
    header(d, p, "1", "Models", "MODEL")
    rows = [  # (title, sub, access marker, dashed)
        ("VAE, β ∈ {1, 4}", f"{m('benchParamsVae')} M · MNIST, Fashion-MNIST", "bound", False),
        ("DCGAN, NS and WGAN-GP",
         f"{m('benchParamsGanSmall')} M (32²), {m('benchParamsGanLarge')} M (64²) · CIFAR-10;\n"
         "NS also on CelebA, CelebA-HQ", "none", False),
        ("RealNVP", f"{m('benchParamsFlow')} M · CIFAR-10, MNIST", "exact", False),
        ("DDPM (Ho et al.)", f"{m('benchParamsDdpm')} M · CIFAR-10 · 100k upd. (orig. 800k)\n"
         f"{m('benchParamsDdpmCeleb')} M · CelebA-HQ 64² · 60k, 120k upd.", "bound", False),
        ("GPT", f"{m('benchParamsGpt')} M · three text corpora", "exact", False),
        ("Stable Diffusion 1.5, frozen", f"{m('clipNPrompts', True)} COCO captions", None, True),
    ]
    gap, y = 0.5, top - 5.2
    hs = [5.35 + 2.35 * s.count("\n") for _, s, _, _ in rows]
    avail = y - (bot + 3.6)
    scale = (avail - gap * (len(rows) - 1)) / sum(hs)
    for (t, s, a, dashed), h in zip(rows, hs):
        h *= scale
        b = kbox(d, x0 + 1.2, y - h, w - 2.4, h, kind="MODEL", ls=(0, (2.2, 1.2)) if dashed else "-")
        d.text(b.x + 4.0, b.y1 - 1.8, t, size=BODY, ha="left", weight="medium")
        sb = d.text(b.x + 4.0, b.y1 - 2.95, s, size=TINY, ha="left", va="top", color=MUTED,
                    linespacing=1.1)
        if d.extent(sb).y < b.y + 0.2:
            print(f"OVERFLOW bottom {t}: {b.y + 0.2 - d.extent(sb).y:.2f} mm")
        fit(d, s, TINY, b.w - 4.7, t)
        if a:
            access(d, b.x + 2.1, b.y1 - 2.0, a)
        y -= h + gap
    # access legend
    ly, lx = bot + 1.75, x0 + 1.8
    d.text(lx, ly, "density access:", size=TINY, color=MUTED, ha="left")
    lx += d.text_width("density access:", size=TINY) + 1.9
    for key in ("exact", "bound", "none"):
        access(d, lx, ly + 0.05, key)
        d.text(lx + 1.6, ly, key, size=TINY, color=MUTED, ha="left")
        lx += 2.6 + d.text_width(key, size=TINY) + 1.4
    return p


# ----------------------------------------------------------------------------- panel 2
def baseline(d, x0, w, top, bot):
    p = d.panel(x0, bot, w, top - bot, KIND["METRIC"], fill=0.04)
    header(d, p, "2", "Baseline: E1", "METRIC")
    ix, iw = x0 + 1.2, w - 2.4
    # inputs
    bw, bh, yi = (iw - 0.8) / 2, 7.2, top - 13.2
    ins = []
    for k, (t, sub) in enumerate((("generated", "10k samples\n5k CelebA-HQ"),
                                  ("reference", "train images\nVAEs: test"))):
        b = d.box(ix + k * (bw + 0.8), yi, bw, bh, t, sub, color=GREY, fill=0.0, lw=0.5, r=0.6,
                  title_size=SMALL, sub_size=TINY, gap=0.05)
        fit(d, sub, TINY, bw - 0.6, t)
        ins.append(b)
    # torch-fidelity block with the two extractors
    tf_top, tf_h = yi - 2.2, 12.6
    tf = d.rect(ix, tf_top - tf_h, iw, tf_h, fc="white", ec=shade(KIND["METRIC"], 0.2), lw=0.5,
                r=0.6, z=2)
    d.text(tf.x + 1.0, tf.y1 - 1.55, "torch-fidelity", size=TINY, color=shade(KIND["METRIC"], 0.35),
           ha="left", style="italic")
    for b in ins:
        d.arrow(b.s, (b.s[0], tf.y1), lw=0.5)
    eh = 4.0
    for k, (net, met) in enumerate((("Inception-v3", "FID, KID"), ("VGG-16", "precision/recall"))):
        e = kbox(d, tf.x + 1.0, tf.y + 1.0 + (1 - k) * (eh + 0.8), tf.w - 2.0, eh, kind="METRIC")
        d.text(e.x + 1.0, e.c[1], net, size=SMALL, ha="left", weight="medium")
        d.text(e.x1 - 1.0, e.c[1], "→ " + met, size=TINY, ha="right", color=MUTED)
        fit(d, net + "  → " + met, TINY, e.w - 2.0, net)
    # run facts
    lines = ["per dataset: one split, one", "preprocessing, one reference set",
             "3 training seeds (mean ± s.d.)",
             f"eval. intervals: ≥ {m('uncReplicatesContrast')} resamples",
             "2 × RTX PRO 6000 Blackwell"]
    y = (tf.y + bot + 3.4) / 2 + LH_S * (len(lines) - 1) / 2
    for s in lines:
        d.text(ix + 0.4, y, s, size=TINY, ha="left", color=INK)
        fit(d, s, TINY, iw - 0.4, "facts")
        y -= LH_S
    pill(d, ix + 0.4, bot + 2.0, "RQ1", "ACCESS", ha="left")
    d.text(ix + 6.6, bot + 2.0, "→ results table", size=TINY, color=MUTED, ha="left",
           style="italic")
    return p


# ----------------------------------------------------------------------------- panel 3
def variations(d, x0, w, top, bot):
    p = d.panel(x0, bot, w, top - bot, GREY, fill=0.025)
    header(d, p, "3", "One choice changed at a time", "EVIDENCE")
    rows = [  # (id, kind, title, sub, RQ, decision-map question, reported in)
        ("E2", "METRIC", "Evaluation protocol",
         "resizer · JPEG · CLIP, DINOv2 · N 1k–5k · reference · metric\n"
         f"{m('protCifarConfigs')} / {m('protCelebConfigs')} sets → "
         f"{m('protCifarConditions')} / {m('protCelebConditions')} conditions (CIFAR-10 / CelebA-HQ)",
         "RQ3", None, "reversal plot"),
        ("E3", "COST", "Budget unit",
         f"iterations · FLOPs (DDPM iter. = {m('budFlopRatio')} DCGAN) · time · energy\n"
         "4 CIFAR-10 models × 2 seeds × 10 snapshots, 5k samples",
         "RQ3", "Q5", "budget curves"),
        ("E4", "SAMPLER", "Sampler, fixed weights",
         "DDIM, DPM-Solver++(2M), Heun · time grid · 10–100 NFE\n"
         f"EMA, CelebA-HQ 20k–120k / CIFAR-10 100k: {m('sampCelebConfigs')} / "
         f"{m('sampCifarConfigs')} configs",
         "RQ2", None, "NFE curves"),
        ("E5", "EVIDENCE", "Constructed defects",
         "training copies (exact, near) · 1 of 10 classes · 1-pixel blur\n"
         f"SD 1.5: 8 guidance scales, {m('clipNPrompts', True)} prompts · copy detector",
         None, None, "text, Suppl. S9"),
    ]
    gap, y = 0.8, top - 5.6
    h = (y - (bot + 1.2) - gap * (len(rows) - 1)) / len(rows)
    spine = x0 + 1.8
    tips = []
    for rid, kind, title, sub, rq, q, rep in rows:
        bx = x0 + 3.6
        b = kbox(d, bx, y - h, x0 + w - 1.2 - bx, h, kind=kind)
        c = shade(KIND[kind], 0.25)
        yt = b.y1 - 1.9
        d.text(b.x + 1.0, yt, rid, size=BODY, weight="bold", color=c, ha="left", family=SANS)
        tt = d.text(b.x + 5.0, yt, title, size=BODY, weight="bold", color=INK, ha="left")
        xx = d.extent(tt).x1 + 1.2
        for lab, k, filled in ((rq, "ACCESS", True), (q, "COST", False)):
            if lab:
                t = pill(d, xx, yt, lab, k, ha="left", filled=filled)
                xx = d.extent(t).x1 + 1.5
        r = d.text(b.x1 - 1.0, yt, "→ " + rep, size=TINY, color=MUTED, ha="right", style="italic")
        if d.extent(r).x < xx + 0.5:
            print(f"OVERLAP title row {rid}")
        sb = d.text(b.x + 1.0, b.y1 - 3.9, sub, size=TINY, color=INK, ha="left", va="top",
                    linespacing=1.12)
        fit(d, sub, TINY, b.w - 1.6, rid)
        if d.extent(sb).y < b.y + 0.3:
            print(f"OVERFLOW bottom {rid}")
        tips.append(b.west)
        y -= h + gap
    # branch: E1 -> each row
    d.line((spine, tips[0][1]), (spine, tips[-1][1]), lw=0.6, color=INK)
    for t in tips:
        d.arrow((spine, t[1]), (t[0], t[1]), lw=0.6)
    return p, spine


# ----------------------------------------------------------------------------- figure
def build(debug=False):
    d = diagram(H, debug)
    top, bot = H - 0.3, 0.3
    w1, w2, g = 43.4, 31.6, 1.6
    p1 = models(d, 0.2, w1, top, bot)
    p2 = baseline(d, p1.x1 + g, w2, top, bot)
    p3, spine = variations(d, p2.x1 + g, W - 0.2 - (p2.x1 + g), top, bot)
    yc = (top + bot) / 2 + 2.0
    d.arrow((p1.x1 - 0.5, yc), (p2.x + 0.5, yc), lw=0.7, head=3.6)
    d.line((p2.x1 - 0.5, yc), (spine, yc), lw=0.7)
    out = HERE / "figures" / "study_design"
    d.save(str(out))
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    cli(build)
