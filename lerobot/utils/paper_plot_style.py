"""Typography shared by the RA-L figures at their intended insertion widths.

The linked manuscript uses the Times family. Nimbus Roman is the installed
successor in that family; use compact 7 pt text at the
exported single/double-column widths.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path

from matplotlib import font_manager
from matplotlib.text import Text

FONT_FAMILY = "Nimbus Roman"
FONT_SIZE = 7.0
RC_PARAMS = {
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


@lru_cache(maxsize=16)
def embedded_font_file(source: str) -> str:
    """Use TrueType outlines for PDF embedding of the installed CFF fonts.

    Matplotlib otherwise wraps these OTF/CFF files as CIDFontType2, which PDF
    validators flag. Preserve the family, glyph order, metrics and layout tables;
    approximate cubic outlines within half a font unit (0.005 pt at 10 pt).
    """
    path = Path(source)
    if path.suffix.lower() != ".otf":
        return source
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont

    with TTFont(path) as font:
        if "CFF " not in font:
            return source
        digest = hashlib.sha256(b"truetype-outline-v2\0" + path.read_bytes()).hexdigest()[:12]
        cache = Path(__file__).resolve().parents[2] / "logs/figures/fonts"
        output = cache / f"{path.stem}-{digest}.ttf"
        if output.exists():
            return str(output)
        glyphs = {}
        for name, glyph in font.getGlyphSet().items():
            pen = TTGlyphPen(None)
            glyph.draw(Cu2QuPen(pen, max_err=.5, reverse_direction=True))
            glyphs[name] = pen.glyph()
        builder = FontBuilder(font["head"].unitsPerEm, isTTF=True)
        builder.setupGlyphOrder(font.getGlyphOrder())
        builder.setupCharacterMap(font.getBestCmap())
        builder.setupGlyf(glyphs)
        builder.setupHorizontalMetrics(font["hmtx"].metrics)
        builder.setupHorizontalHeader(ascent=font["hhea"].ascent, descent=font["hhea"].descent,
                                      lineGap=font["hhea"].lineGap)
        builder.setupNameTable({})
        builder.font["name"] = deepcopy(font["name"])
        builder.font["head"].macStyle = font["head"].macStyle
        builder.setupOS2()
        builder.font["OS/2"] = deepcopy(font["OS/2"])
        builder.setupPost(italicAngle=font["post"].italicAngle,
                          underlinePosition=font["post"].underlinePosition,
                          underlineThickness=font["post"].underlineThickness,
                          isFixedPitch=font["post"].isFixedPitch)
        for tag in ("kern", "GPOS", "GSUB", "GDEF"):
            if tag in font:
                builder.font[tag] = deepcopy(font[tag])
        cache.mkdir(parents=True, exist_ok=True)
        builder.save(output)
    return str(output)


def apply_font_files(fig) -> None:
    """Keep SVG family names and text styles, with valid embedded PDF fonts."""
    for text in fig.findobj(match=Text):
        properties = text.get_fontproperties().copy()
        source = font_manager.findfont(properties, fallback_to_default=False)
        properties.set_file(embedded_font_file(source))
        text.set_fontproperties(properties)


def font_metadata() -> dict:
    return {"family": FONT_FAMILY, "size_pt": FONT_SIZE,
            "regular_file": embedded_font_file(font_manager.findfont(
                font_manager.FontProperties(family=[FONT_FAMILY]), fallback_to_default=False)),
            "bold_file": embedded_font_file(font_manager.findfont(
                font_manager.FontProperties(family=[FONT_FAMILY], weight="bold"), fallback_to_default=False))}
