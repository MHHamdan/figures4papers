"""Millimetre-based primitives for architecture / pipeline diagrams in matplotlib.

The canvas is one axes covering the whole figure with 1 data unit = 1 mm at
print size, origin bottom-left. Fonts are pt, line widths are pt, so the output
looks exactly like a TikZ picture drawn at natural size.

    d = Diagram(181, 90)
    a = d.box(10, 50, 30, 8, "encoder", sub="6 conv blocks", color="#0072B2")
    b = d.box(60, 50, 30, 8, "decoder", color="#1C2E62")
    d.arrow(a.e, b.w)
    d.save("figures/arch")

Every ``box``-like call returns a :class:`Box` with anchor points
(``n s e west ne nw se sw c``) and ``at(fx, fy)`` for fractional anchors.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch, Polygon,
                                Rectangle)
from matplotlib.path import Path as MPath

from . import style as S


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self): return self.x + self.w
    @property
    def y1(self): return self.y + self.h
    @property
    def c(self): return (self.x + self.w / 2, self.y + self.h / 2)
    @property
    def n(self): return (self.x + self.w / 2, self.y1)
    @property
    def s(self): return (self.x + self.w / 2, self.y)
    @property
    def e(self): return (self.x1, self.y + self.h / 2)
    @property
    def west(self): return (self.x, self.y + self.h / 2)  # ``w`` is the width
    @property
    def ne(self): return (self.x1, self.y1)
    @property
    def nw(self): return (self.x, self.y1)
    @property
    def se(self): return (self.x1, self.y)
    @property
    def sw(self): return (self.x, self.y)

    def at(self, fx: float, fy: float):
        """Point at fractional position (0,0)=sw, (1,1)=ne."""
        return (self.x + fx * self.w, self.y + fy * self.h)

    def expand(self, l=0.0, r=None, b=None, t=None) -> "Box":
        r = l if r is None else r
        b = l if b is None else b
        t = b if t is None else t
        return Box(self.x - l, self.y - b, self.w + l + r, self.h + b + t)

    @staticmethod
    def union(*boxes: "Box") -> "Box":
        x0 = min(b.x for b in boxes); y0 = min(b.y for b in boxes)
        x1 = max(b.x1 for b in boxes); y1 = max(b.y1 for b in boxes)
        return Box(x0, y0, x1 - x0, y1 - y0)


class Diagram:
    """A print-size canvas in millimetres."""

    def __init__(self, width_mm: float, height_mm: float, font_size: float = S.FONT_PT["body"],
                 family: str | None = None, math: str = "match", debug_grid: bool = False):
        self.family = S.use_style(font_size, family, math)
        self.W, self.H = width_mm, height_mm
        self.fig = plt.figure(figsize=(width_mm * S.MM, height_mm * S.MM))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width_mm)
        self.ax.set_ylim(0, height_mm)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fs = font_size
        if debug_grid:
            for v in range(0, int(width_mm) + 1, 5):
                self.ax.axvline(v, lw=0.2 if v % 10 else 0.4, color="#ff000040", zorder=100)
            for v in range(0, int(height_mm) + 1, 5):
                self.ax.axhline(v, lw=0.2 if v % 10 else 0.4, color="#ff000040", zorder=100)

    # ------------------------------------------------------------------ text
    def text(self, x, y, s, size=None, color=S.INK, ha="center", va="center", weight="normal",
             style="normal", rotation=0, z=6, linespacing=1.15, **kw):
        return self.ax.text(x, y, s, fontsize=size or self.fs, color=color, ha=ha, va=va,
                            fontweight=weight, fontstyle=style, rotation=rotation, zorder=z,
                            linespacing=linespacing, **kw)

    # ------------------------------------------------------------------ shapes
    def rect(self, x, y, w, h, fc="white", ec=S.INK, lw=S.LINE_PT["normal"], ls="-", r=1.0,
             z=2, alpha=1.0, hatch=None) -> Box:
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                           fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, alpha=alpha, hatch=hatch,
                           joinstyle="round")
        self.ax.add_patch(p)
        return Box(x, y, w, h)

    def box(self, x, y, w, h, title=None, sub=None, color=S.INK, fill=0.0, ls="-",
            lw=S.LINE_PT["normal"], r=1.0, title_size=None, sub_size=None, title_weight="medium",
            text_color=None, sub_color=None, gap=0.15, z=3, ha="center", pad_x=1.5) -> Box:
        """Rounded block with an optional title line and a smaller sub line."""
        b = self.rect(x, y, w, h, fc=S.tint(color, fill) if fill else "white", ec=color,
                      lw=lw, ls=ls, r=r, z=z)
        tx = {"center": b.c[0], "left": x + pad_x, "right": b.x1 - pad_x}[ha]
        ts, ss = title_size or self.fs, sub_size or S.FONT_PT["small"]
        tc = text_color or S.INK
        if title and sub:
            # place the two lines symmetric about the centre
            th, sh = ts * 0.3528, ss * 0.3528 * (sub.count("\n") + 1)
            span = th + sh + gap * th
            top = b.c[1] + span / 2
            self.text(tx, top - th / 2, title, size=ts, color=tc, weight=title_weight, ha=ha, z=z + 1)
            self.text(tx, top - th - gap * th - sh / 2, sub, size=ss, color=sub_color or S.MUTED,
                      ha=ha, z=z + 1)
        elif title:
            self.text(tx, b.c[1], title, size=ts, color=tc, weight=title_weight, ha=ha, z=z + 1)
        return b

    def panel(self, x, y, w, h, color, fill=0.05, ls="-", lw=S.LINE_PT["normal"], r=1.6, z=0) -> Box:
        """Background group / panel."""
        return self.rect(x, y, w, h, fc=S.tint(color, fill), ec=color, lw=lw, ls=ls, r=r, z=z)

    def tag(self, x, y, s, color, size=None, text_color="white", pad=0.32, weight="bold",
            ha="left", va="bottom", z=8):
        """Filled pill label (e.g. panel header) anchored at (x, y) by ``ha``/``va``."""
        return self.text(x, y, s, size=size or S.FONT_PT["title"], color=text_color, weight=weight,
                         ha=ha, va=va, z=z,
                         bbox=dict(boxstyle=f"round,pad={pad},rounding_size=0.45", fc=color,
                                   ec="none"))

    def marker(self, x, y, s, color, r=1.45, fill=None, ls="-", size=None, z=9):
        """Circled letter marker (e.g. attachment points A, B)."""
        self.ax.add_patch(Circle((x, y), r, fc=fill or "white", ec=color, lw=S.LINE_PT["normal"],
                                 ls=ls, zorder=z))
        self.text(x, y - 0.05, s, size=size or S.FONT_PT["small"], color=S.shade(color, 0.15),
                  weight="bold", z=z + 1)

    def chip(self, x, y, s, color, filled=False, r=1.35, size=None, z=6, square=False):
        """Token chip: circle (layout token) or rounded square (character)."""
        if square:
            self.rect(x - r, y - r, 2 * r, 2 * r, fc=S.tint(color, 0.12), ec=S.tint(color, 0.55),
                      lw=0.35, r=0.45, z=z)
            self.text(x, y - 0.05, s, size=size or S.FONT_PT["small"], color=S.INK, z=z + 1)
            return
        self.ax.add_patch(Circle((x, y), r, fc=S.tint(color, 0.85 if filled else 0.0) if filled
                                 else "white", ec=S.shade(color, 0.1), lw=0.5, zorder=z))
        self.text(x, y - 0.05, s, size=size or S.FONT_PT["small"],
                  color="white" if filled else S.shade(color, 0.3),
                  weight="bold" if filled else "medium", z=z + 1)

    def cells(self, x, y, nx, ny, cell, colors, ec="white", lw=0.3, z=4, gap=0.0) -> Box:
        """Grid of square cells; ``colors`` is a colour, a per-row list or an (ny, nx) array."""
        for j in range(ny):
            for i in range(nx):
                c = colors
                if isinstance(colors, (list, tuple)) and len(colors) == ny and not isinstance(colors[0], float):
                    c = colors[j]
                if isinstance(c, (list, tuple)) and len(c) == nx and not isinstance(c[0], float):
                    c = c[i]
                self.ax.add_patch(Rectangle((x + i * (cell + gap), y + (ny - 1 - j) * (cell + gap)),
                                            cell, cell, fc=c, ec=ec, lw=lw, zorder=z))
        return Box(x, y, nx * cell + (nx - 1) * gap, ny * cell + (ny - 1) * gap)

    def cuboid(self, x, y, w, h, depth, color, dx=0.55, dy=0.4, lw=0.4, z=3, alpha=1.0) -> Box:
        """3-D slab (CNN feature map): front face w x h at (x, y), extruded by ``depth``.

        Front face = ``tint(color, .55)``, top = lighter, side = darker. Returns the front face.
        """
        ox, oy = depth * dx, depth * dy
        ec = S.shade(color, 0.25)
        front = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True,
                        fc=S.tint(color, 0.55), ec=ec, lw=lw, zorder=z, alpha=alpha, joinstyle="round")
        top = Polygon([(x, y + h), (x + w, y + h), (x + w + ox, y + h + oy), (x + ox, y + h + oy)],
                      closed=True, fc=S.tint(color, 0.3), ec=ec, lw=lw, zorder=z, alpha=alpha,
                      joinstyle="round")
        side = Polygon([(x + w, y), (x + w + ox, y + oy), (x + w + ox, y + h + oy), (x + w, y + h)],
                       closed=True, fc=S.tint(color, 0.8), ec=ec, lw=lw, zorder=z, alpha=alpha,
                       joinstyle="round")
        for p in (side, top, front):
            self.ax.add_patch(p)
        return Box(x, y, w, h)

    def stack_shadow(self, b: Box, n=2, step=(0.9, 0.9), color=S.INK, fill=0.04, lw=0.45, r=1.6, z=0):
        """Draw ``n`` offset cards behind box ``b`` (a repeated block, e.g. ``x 8`` layers)."""
        for k in range(n, 0, -1):
            self.rect(b.x + k * step[0], b.y + k * step[1], b.w, b.h, fc=S.tint(color, fill),
                      ec=S.tint(color, 0.6), lw=lw, r=r, z=z - 0.1 * k)

    def image(self, path_or_array, x, y, w, h=None, border=S.INK, lw=0.5, z=3) -> Box:
        img = plt.imread(path_or_array) if isinstance(path_or_array, str) else path_or_array
        if h is None:
            h = w * img.shape[0] / img.shape[1]
        self.ax.imshow(img, extent=(x, x + w, y, y + h), zorder=z, interpolation="lanczos",
                       cmap="gray" if img.ndim == 2 else None)
        if border:
            self.ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=border, lw=lw, zorder=z + 0.1))
        return Box(x, y, w, h)

    def oplus(self, x, y, r=1.5, color=S.INK, lw=S.LINE_PT["normal"], z=6):
        self.ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=lw, zorder=z))
        self.ax.plot([x - r, x + r], [y, y], color=color, lw=lw, zorder=z + 0.1, solid_capstyle="butt")
        self.ax.plot([x, x], [y - r, y + r], color=color, lw=lw, zorder=z + 0.1, solid_capstyle="butt")
        return Box(x - r, y - r, 2 * r, 2 * r)

    def brace(self, x0, x1, y, depth=1.0, color=S.MUTED, lw=0.45, below=True, z=4):
        """Square bracket under (below=True) or over a horizontal span."""
        s = -1 if below else 1
        self.ax.plot([x0, x0, x1, x1], [y, y + s * depth, y + s * depth, y], color=color, lw=lw,
                     zorder=z, solid_joinstyle="miter")
        return ((x0 + x1) / 2, y + s * depth)

    # ------------------------------------------------------------------ connectors
    def arrow(self, *pts, color=S.INK, lw=S.LINE_PT["normal"], ls="-", head=3.2, z=5, both=False,
              r=0.0):
        """Poly-line arrow through ``pts`` (each an (x, y) tuple); head at the last point.

        ``r`` > 0 rounds the corners of orthogonal paths with that radius (mm).
        """
        pts = [tuple(map(float, p)) for p in pts]
        verts, codes = [pts[0]], [MPath.MOVETO]
        if r > 0 and len(pts) > 2:
            for a, b, c in zip(pts[:-2], pts[1:-1], pts[2:]):
                v1 = np.subtract(b, a); v2 = np.subtract(c, b)
                l1, l2 = np.hypot(*v1), np.hypot(*v2)
                rr = min(r, l1 / 2, l2 / 2)
                p_in = tuple(np.subtract(b, v1 / l1 * rr)); p_out = tuple(np.add(b, v2 / l2 * rr))
                verts += [p_in, b, p_out]; codes += [MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
            verts.append(pts[-1]); codes.append(MPath.LINETO)
        else:
            verts += pts[1:]; codes += [MPath.LINETO] * (len(pts) - 1)
        style = f"{'<' if both else ''}-|>,head_length={head * 0.12:.3f},head_width={head * 0.07:.3f}"
        p = FancyArrowPatch(path=MPath(verts, codes), arrowstyle=style, mutation_scale=10,
                            color=color, lw=lw, ls=ls, zorder=z, shrinkA=0, shrinkB=0,
                            joinstyle="round", capstyle="butt")
        self.ax.add_patch(p)
        return p

    def line(self, *pts, color=S.INK, lw=S.LINE_PT["normal"], ls="-", z=5):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt",
                     dash_capstyle="butt")

    # ------------------------------------------------------------------ utils
    def extent(self, artist) -> Box:
        """Bounding box of a drawn artist (e.g. a text) in mm."""
        bb = artist.get_window_extent(renderer=self.fig.canvas.get_renderer())
        x0, y0 = self.ax.transData.inverted().transform((bb.x0, bb.y0))
        x1, y1 = self.ax.transData.inverted().transform((bb.x1, bb.y1))
        return Box(x0, y0, x1 - x0, y1 - y0)

    def text_width(self, s, size=None, weight="normal") -> float:
        t = self.text(0, 0, s, size=size, weight=weight)
        w = self.extent(t).w
        t.remove()
        return w

    def save(self, out_base, formats=("pdf", "png"), dpi=600):
        return S.save(self.fig, out_base, formats=formats, dpi=dpi)
