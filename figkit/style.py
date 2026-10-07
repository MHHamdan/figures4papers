"""Shared publication style: page geometry, palettes, fonts, export.

All sizes are *print* sizes: figures are built at the width they are printed
(see ``IEEE`` / ``VENUES``), fonts are in pt and never scaled afterwards, so the
PDF can be included with ``\\includegraphics[width=\\textwidth]`` 1:1.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

MM = 1 / 25.4  # inches per mm

# --------------------------------------------------------------------------- geometry
# Printed widths in mm. ``column`` = \columnwidth, ``text`` = \textwidth.
VENUES = {
    "ieee": {"column": 88.9, "text": 181.0},       # IEEEtran journal (TPAMI, TIP, ...)
    "ieeeconf": {"column": 86.4, "text": 177.8},   # IEEE conference (CVPR/ICCV use ~82.5/172)
    "cvpr": {"column": 82.5, "text": 172.0},
    "neurips": {"column": 139.7, "text": 139.7},   # single column, 5.5 in
    "icml": {"column": 82.5, "text": 171.5},
    "springer": {"column": 84.0, "text": 174.0},   # LNCS-like two-column journals
    "elsevier": {"column": 90.0, "text": 190.0},
}
IEEE = VENUES["ieee"]

# Minimum legible sizes at print scale (IEEE asks for >= 6 pt after scaling).
FONT_PT = {"title": 7.5, "body": 7.0, "small": 6.2, "tiny": 6.0}
LINE_PT = {"thin": 0.4, "normal": 0.6, "thick": 0.9}

# --------------------------------------------------------------------------- palettes
# figures4papers house palette (data plots).
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
    "gray_1": "#767676", "gray_2": "#4D4D4D", "gray_3": "#272727",
}

# Okabe-Ito: colour-blind safe, preferred for diagrams with many semantic roles.
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}

INK = "#262626"      # default text / stroke
MUTED = "#6B6B6B"    # secondary text


def tint(color: str, amount: float) -> tuple:
    """Mix ``color`` with white; amount=0 -> white, 1 -> color (TikZ ``color!NN``)."""
    r, g, b = matplotlib.colors.to_rgb(color)
    return (1 - amount * (1 - r), 1 - amount * (1 - g), 1 - amount * (1 - b))


def shade(color: str, amount: float) -> tuple:
    """Mix ``color`` with black; amount=0 -> color, 1 -> black."""
    r, g, b = matplotlib.colors.to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


# --------------------------------------------------------------------------- fonts
_FONT_DIRS = [Path.home() / ".fonts", Path.home() / ".local/share/fonts",
              Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]
SANS_STACK = ["Inter", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]


def _register_system_fonts(families=("Inter", "Helvetica", "Arial", "Liberation Sans")) -> None:
    known = {f.name for f in font_manager.fontManager.ttflist}
    for d in _FONT_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*.[ot]tf"):
            stem = p.stem.replace("-", " ")
            if any(stem.startswith(fam) for fam in families) and stem.split()[0] not in known:
                try:
                    font_manager.fontManager.addfont(str(p))
                except Exception:  # unreadable font file
                    pass


def first_available(stack=SANS_STACK) -> str:
    _register_system_fonts()
    known = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in stack if f in known), "DejaVu Sans")


def use_style(font_size: float = FONT_PT["body"], family: str | None = None,
              math: str = "match") -> str:
    """Apply the print-scale publication style. Returns the font family used.

    math="match" typesets $...$ in the text font (custom mathtext), "cm" uses
    Computer Modern to match a LaTeX body, "stixsans" a neutral sans math font.
    """
    family = family or first_available()
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": font_size,
        "axes.linewidth": LINE_PT["normal"],
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,          # embed TrueType: editable, IEEE PDF eXpress safe
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
        "figure.dpi": 150,
    }
    if math == "match":
        rc.update({"mathtext.fontset": "custom", "mathtext.rm": family,
                   "mathtext.it": f"{family}:italic", "mathtext.bf": f"{family}:bold",
                   "mathtext.bfit": f"{family}:bold:italic", "mathtext.sf": family,
                   "mathtext.cal": f"{family}:italic", "mathtext.tt": "DejaVu Sans Mono"})
    else:
        rc["mathtext.fontset"] = math
    plt.rcParams.update(rc)
    return family


# --------------------------------------------------------------------------- export
def save(fig, out_base: str | os.PathLike, formats=("pdf", "png"), dpi: int = 600,
         tight: bool = False) -> list[Path]:
    """Save ``fig`` as ``out_base.<fmt>`` for each format; creates parent dirs.

    ``tight=False`` keeps the exact print size the figure was built at.
    """
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_base.with_suffix(f".{fmt}")
        kw = {"bbox_inches": "tight", "pad_inches": 0.01} if tight else {}
        fig.savefig(p, dpi=dpi, facecolor="white", **kw)
        paths.append(p)
    return paths
