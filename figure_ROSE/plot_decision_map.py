"""ROSE survey, Sec. 7 decision map: what to check before comparing two generative models.

    python figure_ROSE/plot_decision_map.py [--debug]

Redraw of paper/figures/tikz/decision_map.tex at acmsmall \\textwidth (139.1 mm). Eight numbered
questions (Q1-Q8, cited by number in the text) in two rows of four; kind hue + kind word per card,
density-access markers of the taxonomy figure in Q1, reading-order arrows 1-4, 4->5 (wrap), 5-8.
Writes figures/decision_map.{pdf,png}.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from matplotlib.patches import Circle, Polygon  # noqa: E402

from figkit import INK, shade, tint  # noqa: E402
from rose_style import BODY, DATAFLOW, KIND, SMALL, W, access, cli, diagram, kindword  # noqa: E402

PT = 0.3528                      # mm per pt
QS = 6.8                         # question (header) size, bold (TikZ: 6.8 pt)
OS = 6.5                         # outcome size (TikZ: 6.5 pt)
LS_Q, LS_O = 1.08, 1.08          # line pitch / font size
GAP = 2.2                        # horizontal gap between cards (holds the reading-order arrow)
X0 = 0.3                         # side inset so outer strokes and kind words are not clipped
CW = (W - 2 * X0 - 3 * GAP) / 4  # card width
PAD = 0.9                        # inner padding
NUM_R = 1.7                      # numbered circle radius
KW_H = 2.3                       # band for the kind word above a card
TOP = 0.35                       # extra headroom above the first-row kind words
PARA_GAP = 0.4                   # extra space between outcomes
GLYPH_Y = 0.85                   # glyph centre above the baseline (mm)

# ---------------------------------------------------------------- content (verbatim from TikZ)
# Outcome = list of runs; run = (text, "i" italic | "r" roman) or ("@glyph", access key).
I, R = "i", "r"
CARDS = [
    (1, "ACCESS", "ACCESS", "What density does the deployed artefact provide?", [
        [("@", "exact"), ("@", "bound"), ("exact, bound", I), (": likelihood admissible; say which", R)],
        [("@", "unnorm"), ("unnormalised", I), (": AIS or likelihood ratios only", R)],
        [("@", "none"), ("none", I), (" (GAN, distilled sampler, system): skip Q3", R)]]),
    (2, "MODEL", "MODEL", "Which modality and conditioning?", [
        [("image", I), (": feature-space metrics; text-conditional adds a CLIPScore curve", R)],
        [("text", I), (": bits per byte; ", R), ("aligned or agentic", I),
         (": task success, with retrieval recall separate", R)]]),
    (3, "ACCESS", "ACCESS", "Likelihood at evaluation: by which estimator?", [
        [("exact", I), (": NLL or bits/dim, dequantisation and test set stated", R)],
        [("bound or estimate", I), (": name it (IWAE, VLB, PF-ODE, AIS); never rank across estimators", R)]]),
    (4, "SAMPLER", "SAMPLER", "Which sampler, NFE, guidance or truncation?", [
        [("fixed and reported", I), (": compare at equal NFE or as curves over NFE", R)],
        [("differs", I), (": the claim is about the model–sampler pair, not the family", R)]]),
    (5, "COST", "COST (TRAINING)", "Which training budget, matched how?", [
        [("matched", I), (" in updates, wall-clock, FLOPs or energy: name the unit; report the "
                          "quality–budget curve, a single point is descriptive", R)],
        [("unmatched", I), (": no family ranking", R)]]),
    (6, "METRIC", "METRIC", "Which metric is valid under a shared protocol?", [
        [("same", I), (" N, reference set, extractor, resizer and code: valid within the dataset", R)],
        [("resolved?", I), (" an interval on the difference excludes zero: rank; otherwise a tie", R)],
        [("any mismatch", I), (": recompute", R)]]),
    (7, "EVIDENCE", "EVIDENCE", "Which blind spots remain; what covers them?", [
        [("memorisation, saturation, preference", I),
         (": nearest-neighbour check, precision/recall, human subset", R)],
        [("not covered", I), (": report as a limitation", R)]]),
    (8, "COST", "COST (DEPLOYMENT)", "Which deployment cost is measured?", [
        [("throughput, memory, energy", I), (" at matched hardware, batch and precision", R)],
        [("not measured", I), (": the cost side of the comparison is missing", R)]]),
]


# ---------------------------------------------------------------- rich-text layout (local helper)
class Rich:
    """Greedy word wrap of mixed roman/italic/bold runs and inline glyphs, measured in mm."""

    def __init__(self, d):
        self.d, self._cache = d, {}

    def width(self, s, size, style):
        key = (s, size, style)
        if key not in self._cache:
            t = self.d.text(0, 0, s, size=size, weight="bold" if style == "b" else "normal",
                            style="italic" if style == "i" else "normal", ha="left", va="baseline")
            self._cache[key] = self.d.extent(t).w if s.strip() else self.d.text_width("a a", size) - 2 * self.d.text_width("a", size)
            t.remove()
        return self._cache[key]

    def words(self, runs, size):
        """Split runs into words; a word is a list of (text, style) segments or a glyph."""
        out, cur = [], []
        for txt, st in runs:
            if txt == "@":
                if cur:
                    out.append(cur); cur = []
                out.append([("@", st)])
                continue
            parts = txt.split(" ")
            for k, p in enumerate(parts):
                if k > 0 and cur:
                    out.append(cur); cur = []
                if p:
                    cur.append((p, st))
        if cur:
            out.append(cur)
        return out

    def wwidth(self, word, size):
        if word[0][0] == "@":
            return 2.7 if word[0][1] == "unnorm" else 2.2
        return sum(self.width(s, size, st) for s, st in word)

    def wrap(self, runs, size, width, first_indent=0.0, hang=0.0):
        sp = self.width(" ", size, "r")
        lines, line, x = [], [], first_indent
        for w in self.words(runs, size):
            ww = self.wwidth(w, size)
            nx = x + (sp if line else 0) + ww
            if line and nx > width + 1e-6:
                lines.append(line); line, x = [], hang
                nx = x + ww
            line.append((x if not line else x + sp, w))
            x = nx
        lines.append(line)
        return lines

    def draw(self, lines, x0, ytop, size, pitch, color=INK, bold=False):
        """Draw wrapped lines; first baseline sits one cap height under ytop. Returns bottom y."""
        cap = 0.68 * size * PT
        y = ytop - cap
        for line in lines:
            for x, w in line:
                if w[0][0] == "@":
                    access(self.d, x0 + x + self.wwidth(w, size) / 2 - 0.1, y + GLYPH_Y * size / 6.5, w[0][1])
                    continue
                xx = x0 + x
                for s, st in w:
                    st2 = "b" if bold else st
                    self.d.text(xx, y, s, size=size, color=color, ha="left", va="baseline",
                                weight="bold" if st2 == "b" else "normal",
                                style="italic" if st2 == "i" else "normal")
                    xx += self.width(s, size, st2)
            y -= pitch
        return y + pitch - 0.27 * size * PT       # descender of the last line


def bullet(d, x, y, color):
    """Hollow right-pointing triangle (TikZ $\\triangleright$), centred on the x-height."""
    s = 1.05
    d.ax.add_patch(Polygon([(x, y - s / 2), (x + s * 0.87, y), (x, y + s / 2)], closed=True,
                           fc="white", ec=color, lw=0.45, zorder=7, joinstyle="miter"))


def layout(rich):
    """Wrap every card; return per-card (header lines, outcomes lines, heights)."""
    tw = CW - 2 * PAD
    qp, op = QS * PT * LS_Q, OS * PT * LS_O
    hang_b = 1.75                               # bullet indent
    res = []
    for num, kind, word, q, outs in CARDS:
        hl = rich.wrap([(q, "b")], QS, tw, first_indent=2 * NUM_R + 0.5)
        ol = []
        for runs in outs:
            if runs[0][0] == "@":
                ol.append(rich.wrap(runs, OS, tw, hang=2.9))
            else:
                ol.append(rich.wrap(runs, OS, tw - hang_b, hang=0.0))
        hh = 2 * PAD - 0.2 + 0.68 * QS * PT + (len(hl) - 1) * qp + 0.27 * QS * PT
        bh = (2 * PAD + 0.68 * OS * PT + 0.27 * OS * PT + (sum(map(len, ol)) - 1) * op
              + (len(ol) - 1) * PARA_GAP)
        res.append(dict(hl=hl, ol=ol, hh=hh, bh=bh))
    return res


def build(debug=False):
    probe = diagram(58)
    L = layout(Rich(probe))
    import matplotlib.pyplot as plt
    plt.close(probe.fig)
    HH = [max(L[i]["hh"] for i in range(r * 4, r * 4 + 4)) for r in (0, 1)]
    BH = [max(L[i]["bh"] for i in range(r * 4, r * 4 + 4)) for r in (0, 1)]
    WRAP = 1.0                                  # wrap arrow runs this far below row 1
    H = TOP + KW_H + HH[0] + BH[0] + WRAP + KW_H + 0.4 + HH[1] + BH[1] + 0.35
    H = round(H, 2)
    d = diagram(H, debug)
    rich = Rich(d)
    qp, op = QS * PT * LS_Q, OS * PT * LS_O
    heads, bodies = [], []
    for i, (num, kind, word, q, outs) in enumerate(CARDS):
        row, col = divmod(i, 4)
        c = KIND[kind]
        x = X0 + col * (CW + GAP)
        ytop = H - TOP - KW_H if row == 0 else H - TOP - KW_H - HH[0] - BH[0] - WRAP - KW_H - 0.4
        hb = d.rect(x, ytop - HH[row], CW, HH[row], fc=tint(c, 0.13), ec=shade(c, 0.2), lw=0.55,
                    r=0.0, z=3)
        bb = d.rect(x, ytop - HH[row] - BH[row], CW, BH[row], fc="white", ec=shade(c, 0.2), lw=0.55,
                    r=0.0, z=2)
        heads.append(hb); bodies.append(bb)
        kindword(d, x + CW, ytop + 0.35, kind, text=word)
        # numbered circle on the first header line
        cy = ytop - PAD - 0.68 * QS * PT / 2 + 0.1
        d.ax.add_patch(Circle((x + PAD + NUM_R - 0.2, cy), NUM_R, fc=shade(c, 0.2), ec="none", zorder=7))
        d.text(x + PAD + NUM_R - 0.2, cy - 0.05, str(num), size=SMALL, color="white", weight="bold", z=8)
        rich.draw(L[i]["hl"], x + PAD, ytop - PAD + 0.2, QS, qp, bold=True)
        # outcomes
        y = ytop - HH[row] - PAD
        for runs, lines in zip(outs, L[i]["ol"]):
            if runs[0][0] == "@":
                yb = rich.draw(lines, x + PAD, y, OS, op)
            else:
                bullet(d, x + PAD + 0.05, y - 0.68 * OS * PT + 0.6, shade(c, 0.2))
                yb = rich.draw(lines, x + PAD + 1.75, y, OS, op)
            y = yb - (op - 0.95 * OS * PT) - PARA_GAP
    # reading order
    aw = dict(DATAFLOW, lw=0.7)
    for a, b in ((0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)):
        ya = heads[a].c[1] if a < 4 else heads[a].y1 - HH[1] / 2
        d.arrow((heads[a].x1, ya), (heads[b].x, ya), **aw)
    yw = bodies[3].y - WRAP / 2 - 0.15
    x5 = heads[4].x + 6.0
    d.arrow(bodies[3].s, (bodies[3].s[0], yw), (x5, yw), (x5, heads[4].y1), r=0.8, **aw)
    d.save(HERE / "figures/decision_map")
    print(f"{W:.1f} x {H:.2f} mm")


if __name__ == "__main__":
    cli(build)
