"""figkit: print-size publication figures (data plots + architecture diagrams)."""
from . import style
from .diagram import Box, Diagram
from .style import (FONT_PT, IEEE, INK, LINE_PT, MM, MUTED, OKABE_ITO, PALETTE, VENUES,
                    register_font_files, save, shade, tint, use_style)

__all__ = ["Box", "Diagram", "style", "FONT_PT", "IEEE", "INK", "LINE_PT", "MM", "MUTED",
           "OKABE_ITO", "PALETTE", "VENUES", "register_font_files", "save", "shade", "tint", "use_style"]
