"""One score network, three likelihood statements (and none for its distilled student).

    python figure_ROSE/plot_likelihood_statements.py [--debug]

Writes figure_ROSE/figures/likelihood_statements.{pdf,png} (139.1 x 37 mm) for
supplement/07_evaluation.tex, Sec. "Likelihood-based evaluation". Every statement is quoted
from the paper: main/03_foundations.tex:152-160 and 176-180 (three statements, L_simple),
main/05_taxonomy.tex:29-31 (distilled / few-step student), main/07_evaluation.tex:41-52 and
supplement/07_evaluation.tex:35-47, 61-66 (dequantised bits/dim), decision_map.tex Q3.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figkit import INK, MUTED, shade  # noqa: E402
from rose_style import (BODY, DATAFLOW, GRADFLOW, KIND, SMALL, TINY, W, access, cli,  # noqa: E402
                        diagram, kbox, kindword)

H = 37.0
TOP, BOT = 33.6, 12.6          # branch boxes
BAR_Y, BAR_H = 1.0, 7.6        # bottom bar

BRANCHES = [  # (tag, title, lines, access key, kind)
    ("a", "Discrete-time chain",
     [r"$-\log p_\theta(\mathbf{x}_0)\leq L_T+\Sigma L_{t-1}+L_0$",
      "variational bound; bits/dim",
      "from the bound, not from",
      r"$\mathcal{L}_{\mathrm{simple}}$ (not itself a bound)"], "bound", "ACCESS"),
    ("b", "Reverse SDE",
     [r"with $\lambda(t)=g(t)^2$, training loss",
      r"+ terms free of $\theta$ upper-",
      "bounds the NLL of the",
      "reverse SDE"], "bound", "ACCESS"),
    ("c", "Probability-flow ODE",
     ["exact likelihood of the ODE",
      "model: trace estimator + one",
      "ODE solve per datum; differs",
      "from the SDE model unless",
      "the score is exact"], "exactR", "ACCESS"),
    ("d", "Distilled student",
     ["one- or few-step sampler:",
      "the teacher had a bound,",
      "the deployed sampler",
      "has none"], "noneR", "SAMPLER"),
]


def build(debug):
    d = diagram(H, debug)

    # source: one trained score network
    src = kbox(d, 0.4, BOT + 2.0, 17.0, TOP - BOT - 4.0, kind="MODEL")
    kindword(d, src.x, src.y1 + 0.5, "MODEL", ha="left")
    d.text(src.c[0], src.c[1] + 4.0, "one trained", size=BODY)
    d.text(src.c[0], src.c[1] + 0.8, "score network", size=BODY)
    d.text(src.c[0], src.c[1] - 3.6, r"$\epsilon_\theta\;/\;s_\theta$", size=BODY)

    x0, gap = 22.5, 1.5
    bw = (W - 0.4 - x0 - 3 * gap) / 4
    bus_y = BOT - 2.2
    d.line(src.s, (src.s[0], bus_y + 1.0), **DATAFLOW)
    for k, (tg, title, lines, key, kind) in enumerate(BRANCHES):
        x = x0 + k * (bw + gap)
        sam = kind == "SAMPLER"
        b = kbox(d, x, BOT, bw, TOP - BOT, kind=kind, ls=(0, (3, 1.6)) if sam else "-")
        if k == 0:
            kindword(d, x, TOP + 0.5, "ACCESS", ha="left",
                     text="ACCESS: three likelihood statements of one network")
        if sam:
            kindword(d, x, TOP + 0.5, "SAMPLER", ha="left", text="SAMPLER: distilled from it")
        access(d, x + 1.9, TOP - 2.6, key)
        d.text(x + 3.5, TOP - 2.6, f"({tg}) {title}", size=SMALL, ha="left", weight="bold")
        for i, s in enumerate(lines):
            d.text(x + 1.3, TOP - 6.4 - i * 3.0, s, size=SMALL, ha="left", color=INK)
        # bus below the boxes: solid = same network evaluated; dashed vermillion = distillation
        cx = x + bw / 2
        if sam:  # distillation branches off the bus after (c)
            px = x - gap - bw / 2
            d.arrow((px, bus_y), (cx, bus_y), (cx, BOT), r=1.0, **GRADFLOW)
            d.text((px + cx) / 2, bus_y, "distil", size=TINY, va="center",
                   color=shade(KIND["OBJECTIVE"], 0.2), bbox=dict(fc="white", ec="none", pad=0.6))
        else:
            d.arrow((src.s[0], bus_y + 1.0), (src.s[0], bus_y), (cx, bus_y), (cx, BOT), r=1.0,
                    **DATAFLOW)

    # bottom bar: the metric layer
    bar = kbox(d, 0.4, BAR_Y, W - 0.8, BAR_H, kind="METRIC")
    kindword(d, bar.x + 1.2, bar.y1 - 0.4, "METRIC", ha="left", va="top")
    d.text(bar.x + 11.0, bar.y + BAR_H * 0.70,
           r"Dequantised bits/dim is an upper bound on the code length per dimension, even when "
           r"$p_\theta(\mathbf{y})$ is exact; each bounds its own model.", size=SMALL, ha="left")
    d.text(bar.x + 11.0, bar.y + BAR_H * 0.27,
           r"Name the estimator (VLB, PF-ODE, IWAE, AIS); never rank across estimators. "
           r"(d): sample-based metrics only.", size=SMALL, ha="left", weight="bold")
    d.save(HERE / "figures/likelihood_statements")


if __name__ == "__main__":
    cli(build)
