"""Shared visual system for the ROSE survey figures (ACM CSUR, acmsmall).

Mirrors paper/figures/tikz/style.tex of ~/GenAI/generative-ai-survey so redrawn and new
figures read as one set: Okabe-Ito object-kind hues, one marker shape per density-access
value, one line style per arrow meaning. Import after ``sys.path.insert(0, <repo root>)``.
"""
from pathlib import Path

from figkit import FONT_PT, INK, MUTED, OKABE_ITO, VENUES, Diagram, register_font_files, shade, tint

W = VENUES["acmsmall"]["text"]          # 139.1 mm; include with \includegraphics[width=\textwidth]
PAPER = Path.home() / "GenAI/generative-ai-survey/paper"

# acmart body font, so figure text matches the page.
_LIB = Path.home() / ".TinyTeX/texmf-dist/fonts/opentype/public/libertine"
register_font_files(*(_LIB / f for f in ("LinLibertine_R.otf", "LinLibertine_RB.otf",
                                         "LinLibertine_RI.otf", "LinLibertine_RBI.otf",
                                         "LinBiolinum_R.otf", "LinBiolinum_RB.otf")))
FAMILY = "Linux Libertine O"
SANS = "Linux Biolinum O"               # kind words (ACCESS, SAMPLER, ...) as in the TikZ \sffamily

BODY, SMALL, TINY = FONT_PT["body"], FONT_PT["small"], FONT_PT["tiny"]

# Object kinds (style.tex kAccess ... kEvidence)
KIND = {
    "ACCESS": OKABE_ITO["blue"],        # density access / reportable quantities (axis A)
    "OBJECTIVE": OKABE_ITO["vermillion"],  # training signal and divergence
    "MODEL": OKABE_ITO["sky"],          # mechanism / route (axis B), modality, conditioning
    "SAMPLER": OKABE_ITO["orange"],     # sampler, NFE, guidance
    "METRIC": OKABE_ITO["green"],       # evaluation metric and protocol
    "COST": OKABE_ITO["purple"],        # training budget and deployment cost
    "EVIDENCE": "#555555",              # blind spots, evidence status
}
GREY = "#555555"

# Density-access markers: (shape, colour, hollow); size in mm as in style.tex.
ACCESS = {
    "exact": ("circle", OKABE_ITO["blue"], False, 2.0),
    "bound": ("square", OKABE_ITO["sky"], False, 1.8),
    "unnorm": ("triangle", OKABE_ITO["orange"], False, 2.7),
    "none": ("diamond", OKABE_ITO["vermillion"], False, 2.4),
    "exactR": ("circle", OKABE_ITO["blue"], True, 2.0),    # exact through an ODE solve
    "noneR": ("diamond", OKABE_ITO["vermillion"], True, 2.4),  # distilled / few-step sampler
}
# Evidence status of an anchor.
STATUS = {"established": ("square", "black", False), "observed": ("halfsquare", "black", False),
          "untested": ("square", "black", True)}

# Arrow meanings: data flow solid ink; training/gradient signal vermillion dashed;
# inference-only blue dotted; equivalence grey dash-dot.
DATAFLOW = dict(color=INK)
GRADFLOW = dict(color=OKABE_ITO["vermillion"], ls=(0, (3, 1.6)))
INFERONLY = dict(color=OKABE_ITO["blue"], ls=(0, (0.8, 1.2)), lw=0.7)
EQUIV = dict(color=GREY, ls=(0, (2, 1, 0.5, 1)))


def diagram(h, debug=False):
    return Diagram(W, h, font_size=BODY, family=FAMILY, math="match", debug_grid=debug)


def access(d, x, y, key):
    shape, c, hollow, size = ACCESS[key]
    ec = None if key in ("exact", "none", "exactR", "noneR") else "black"
    return d.glyph(x, y, shape, c, size=size, ec=ec, hollow=hollow)


def status(d, x, y, key, size=1.6):
    shape, c, hollow = STATUS[key]
    return d.glyph(x, y, shape, c, size=size, hollow=hollow)


def kbox(d, x, y, w, h, title=None, sub=None, kind="MODEL", ls="-", r=0.6, **kw):
    """Object-kind box: tint .13 fill, shaded stroke (style.tex kbox)."""
    c = KIND[kind]
    return d.box(x, y, w, h, title, sub, fc=tint(c, 0.13), ec=shade(c, 0.2), lw=0.55, ls=ls, r=r,
                 **kw)


def kindword(d, x, y, kind, ha="right", va="bottom", text=None):
    """Kind label (e.g. SAMPLER) in small sans, as above a TikZ kbox."""
    return d.text(x, y, text or kind, size=SMALL, color=shade(KIND.get(kind, GREY), 0.3), ha=ha,
                  va=va, family=SANS)


def cli(build):
    """Standard entry point: ``python figure_ROSE/plot_x.py [--debug]``."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="overlay a 5 mm grid")
    build(ap.parse_args().debug)
