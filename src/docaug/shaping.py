"""Thai text shaping and rasterization, with a box for every cluster.

This is the load-bearing module. Everything else in the pipeline is arithmetic on
boxes; here is where the boxes come from.

Two libraries do the work. **HarfBuzz** shapes the text -- it reorders the
leading vowels เ แ โ ใ ไ to the left of their consonant, stacks marks, and
reports which input characters each output glyph came from. **FreeType**
rasterizes each shaped glyph at the offset HarfBuzz gave it.

The label is then read back *out of the raster*: as each glyph is blitted we
extend the bounding box of its cluster. So a cluster box describes ink that is
demonstrably on the page, and a mark that failed to render cannot silently
inflate a box. This is the asymmetry the whole method rests on -- we place the
text, so we know exactly what it says and exactly where it is.

Only one face is used per line. Mixed-script sources (CJK in a Thai face) would
need a per-run fallback; that is a deliberate omission, and the place to add it
is `_shape`, which already works run-at-a-time.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import freetype
import numpy as np
import uharfbuzz as hb
from PIL import Image

from .types import Cluster, Color, RenderedText

_IDENTITY = freetype.Matrix(0x10000, 0, 0, 0x10000)
_NO_SHIFT = freetype.Vector(0, 0)

DEFAULT_LEADING = 1.6
"""Line spacing as a multiple of the type size, for stacked Thai. Latin gets away
with ~1.2; Thai needs the extra room so line N's tone marks clear line N-1's
below-vowels."""


@dataclass(frozen=True, slots=True)
class Face:
    """One font face. Hashable and cheap to pass around; the expensive HarfBuzz
    and FreeType objects hang off the module caches below, keyed by this."""

    path: str
    index: int = 0

    @property
    def name(self) -> str:
        """The family name from the font's own name table, which is the only
        authoritative source; the filename is a convention."""
        family = (ft_face(self, 16).family_name or b"").decode("utf-8", "replace").strip()
        return family or Path(self.path).stem

    def covers(self, text: str) -> bool:
        """True when every character in `text` has a glyph -- no tofu boxes in
        the ground truth."""
        ft = ft_face(self, 16)
        return all(ft.get_char_index(ord(ch)) for ch in text)

    def missing(self, codepoints: list[int]) -> set[int]:
        ft = ft_face(self, 16)
        return {cp for cp in codepoints if not ft.get_char_index(cp)}


# HarfBuzz holds the whole font file in memory, so the Face object is cached once
# per file and only the sized Font is cached per (face, size). Without that split,
# auto-fitting a block -- which tries a dozen sizes -- would re-read the file a
# dozen times.
@functools.lru_cache(maxsize=64)
def _hb_face(face: Face) -> hb.Face:
    return hb.Face(Path(face.path).read_bytes(), face.index)


@functools.lru_cache(maxsize=512)
def _hb(face: Face, size: int) -> hb.Font:
    font = hb.Font(_hb_face(face))
    font.scale = (size, size)  # so HarfBuzz reports positions in pixels
    return font


@functools.lru_cache(maxsize=512)
def ft_face(face: Face, size: int) -> freetype.Face:
    ft = freetype.Face(face.path, index=face.index)
    ft.set_pixel_sizes(0, size)
    return ft


@contextmanager
def _sheared(ft: freetype.Face, slant: float) -> Iterator[None]:
    """Shear the glyph *outline* while rendering, so an oblique stays crisp
    instead of being a bilinear smear of an upright raster.

    The FreeType face is shared through the cache, so the transform is always
    reset -- including on the way out of an exception.
    """
    if not slant:
        yield
        return
    ft.set_transform(freetype.Matrix(0x10000, int(slant * 0x10000), 0, 0x10000), _NO_SHIFT)
    try:
        yield
    finally:
        ft.set_transform(_IDENTITY, _NO_SHIFT)


def glyph_bitmap(ft: freetype.Face, gid: int) -> tuple[np.ndarray, int, int]:
    """Rasterize one glyph -> (coverage, left_bearing, top_bearing)."""
    ft.load_glyph(gid, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
    bmp = ft.glyph.bitmap
    if not (bmp.width and bmp.rows):
        return np.zeros((0, 0), np.uint8), ft.glyph.bitmap_left, ft.glyph.bitmap_top
    buf = np.asarray(bmp.buffer, dtype=np.uint8)
    return (
        buf.reshape(bmp.rows, bmp.pitch)[:, : bmp.width],
        ft.glyph.bitmap_left,
        ft.glyph.bitmap_top,
    )


def shape_text(text: str, face: Face, size: int, *, monotone: bool = False):
    """Run HarfBuzz. `monotone` gives one cluster per character, which the glyph
    stitcher needs; the default merges a base and its marks into one cluster,
    which is what an OCR label wants."""
    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction, buf.script, buf.language = "ltr", "Thai", "th"
    if monotone:
        buf.cluster_level = 1  # HB_BUFFER_CLUSTER_LEVEL_MONOTONE_CHARACTERS
    hb.shape(
        _hb(face, size),
        buf,
        {"kern": True, "liga": True, "ccmp": True, "mark": True, "mkmk": True},
    )
    return buf.glyph_infos, buf.glyph_positions


def _dilate(alpha: np.ndarray, amount: float) -> np.ndarray:
    """Faux bold. Real weights come from the family when it has them; this is the
    fallback for a family that ships one weight."""
    for _ in range(int(round(amount))):
        alpha = np.maximum.reduce(
            [alpha, np.roll(alpha, 1, 0), np.roll(alpha, -1, 0),
             np.roll(alpha, 1, 1), np.roll(alpha, -1, 1)]
        )
    return alpha


def _colorize(alpha: np.ndarray, color: Color) -> Image.Image:
    rgba = np.zeros((*alpha.shape, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, "RGBA")


@dataclass(slots=True)
class ShapedLine:
    """One rasterized line. `baseline` is where the next line aligns to."""

    image: Image.Image
    clusters: list[Cluster]
    baseline: int

    @property
    def width(self) -> int:
        return self.image.width


def shape_line(
    text: str,
    face: Face,
    size: int = 64,
    *,
    color: Color = (0, 0, 0),
    weight: float = 0.0,
    slant: float = 0.0,
) -> ShapedLine:
    """Shape and rasterize a single line onto a transparent canvas."""
    ft = ft_face(face, size)
    infos, positions = shape_text(text, face, size)
    if not positions:
        return ShapedLine(Image.new("RGBA", (1, 1), (0, 0, 0, 0)), [], 0)

    ascent, descent = ft.size.ascender >> 6, -(ft.size.descender >> 6)
    # Tone marks routinely exceed the ascender and below-vowels the descender, so
    # give a full em of headroom at each end and crop to real ink afterwards.
    head = size
    height = ascent + descent + 2 * head
    lean = int(abs(slant) * height) + 1 if slant else 0
    width = max(1, sum(p.x_advance for p in positions)) + lean
    baseline = head + ascent

    alpha = np.zeros((height, width), np.uint8)
    bounds: dict[int, list[int]] = {}
    pen = lean // 2

    with _sheared(ft, slant):
        for info, pos in zip(infos, positions, strict=True):
            bmp, left, top = glyph_bitmap(ft, info.codepoint)
            gx, gy = pen + pos.x_offset + left, baseline - pos.y_offset - top
            pen += pos.x_advance
            if not bmp.size:
                continue
            gh, gw = bmp.shape
            x0, y0 = max(gx, 0), max(gy, 0)
            x1, y1 = min(gx + gw, width), min(gy + gh, height)
            if x1 <= x0 or y1 <= y0:
                continue
            patch = bmp[y0 - gy : y1 - gy, x0 - gx : x1 - gx]
            np.maximum(alpha[y0:y1, x0:x1], patch, out=alpha[y0:y1, x0:x1])
            # Extend this cluster's box by the ink we just laid down.
            box = bounds.setdefault(info.cluster, [x0, y0, x1, y1])
            box[0], box[1] = min(box[0], x0), min(box[1], y0)
            box[2], box[3] = max(box[2], x1), max(box[3], y1)

    if weight > 0:
        alpha = _dilate(alpha, weight)

    # A cluster id is the index of its first character, so consecutive ids
    # delimit the substring each box labels -- in logical order, marks included.
    starts = sorted(bounds)
    clusters = [
        Cluster(text[start:end], tuple(bounds[start]))
        for start, end in zip(starts, [*starts[1:], len(text)], strict=True)
    ]
    return ShapedLine(_colorize(alpha, color), clusters, baseline)


def line_width(text: str, face: Face, size: int) -> int:
    """Horizontal advance without rasterizing. Used by the fit loop, which asks
    this question far more often than it draws anything."""
    _, positions = shape_text(text, face, size)
    return sum(p.x_advance for p in positions)


def _tokenize(text: str) -> list[str]:
    """Split into breakable units. Thai has no inter-word spaces, so a dictionary
    tokenizer is the only way to find a legal break; without pythainlp installed
    we fall back to breaking between characters, which wraps in the wrong places
    but never corrupts a label."""
    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError:
        return list(text)

    tokens: list[str] = []
    for i, chunk in enumerate(text.split(" ")):
        if i:
            tokens.append(" ")
        if chunk:
            tokens.extend(word_tokenize(chunk, engine="newmm"))
    return tokens


def wrap(text: str, face: Face, size: int, max_width: int) -> list[str]:
    """Greedy wrap at Thai word boundaries.

    A line keeps the space it was broken at. The space draws nothing, so it costs
    the line no ink and no width, but it is part of the region's text -- dropping
    it is how a cluster stream stops reassembling into the label it came from.
    Everything that measures a line therefore measures it stripped.
    """
    lines: list[str] = []
    current = ""
    for token in _tokenize(text):
        candidate = current + token
        if current and line_width(candidate.rstrip(), face, size) > max_width:
            lines.append(candidate if token.isspace() else current)
            current = "" if token.isspace() else token
        else:
            current = candidate
    if current.strip():
        lines.append(current)
    return lines or [""]


def vertical_metrics(face: Face, size: int) -> tuple[int, int]:
    ft = ft_face(face, size)
    return ft.size.ascender >> 6, -(ft.size.descender >> 6)


def fit_lines(
    text: str, face: Face, box_w: int, box_h: int, leading: float, min_size: int, start: int
) -> tuple[int, list[str], int]:
    """Largest type size whose wrapped lines fit the box -> (size, lines, line_h).

    Height is checked first because it is cheap. Width is checked only for the
    sizes that already pass height, and it matters for *unbreakable* tokens: a
    single Thai word has no legal break point, so it stays on one line and runs
    out the side of a narrow box while the height check happily passes.
    """
    for size in range(start, min_size - 1, -1):
        lines = wrap(text, face, size, box_w)
        ascent, descent = vertical_metrics(face, size)
        line_h = max(round(size * leading), ascent + descent)
        if len(lines) * line_h > box_h:
            continue
        if max((line_width(ln.rstrip(), face, size) for ln in lines), default=0) <= box_w:
            return size, lines, line_h

    # Nothing fits. Return the floor and let the caller mark the region
    # `fit=False`: an overflowing region is still correctly labelled, and the
    # decision to drop it belongs to whoever is assembling the dataset.
    lines = wrap(text, face, min_size, box_w)
    ascent, descent = vertical_metrics(face, min_size)
    return min_size, lines, max(round(min_size * leading), ascent + descent)


def render_block(
    text: str,
    face: Face,
    box_w: int,
    box_h: int,
    *,
    color: Color = (0, 0, 0),
    leading: float = DEFAULT_LEADING,
    min_size: int = 6,
    start_size: int | None = None,
    align: str = "left",
    weight: float = 0.0,
    slant: float = 0.0,
) -> RenderedText:
    """Wrap and auto-fit `text` into a `box_w` x `box_h` region.

    `start_size` is the size to try first -- pass the measured source ink height
    to preserve the document's type hierarchy, or leave it out to fill the box.
    """
    start = max(min_size, start_size or int(box_h / leading))
    size, lines, line_h = fit_lines(text, face, box_w, box_h, leading, min_size, start)
    ascent, _ = vertical_metrics(face, size)

    pad = size  # headroom for marks that overshoot the first and last lines
    canvas = Image.new("RGBA", (box_w, len(lines) * line_h + 2 * pad), (0, 0, 0, 0))
    clusters: list[Cluster] = []
    for i, line in enumerate(lines):
        shaped = shape_line(line, face, size, color=color, weight=weight, slant=slant)
        x = max(0, (box_w - shaped.width) // 2) if align == "center" else 0
        y = pad + ascent + i * line_h - shaped.baseline
        canvas.alpha_composite(shaped.image, (x, y))
        clusters += [
            Cluster(c.text, (c.box[0] + x, c.box[1] + y, c.box[2] + x, c.box[3] + y))
            for c in shaped.clusters
        ]

    return _crop_to_ink(canvas, clusters, size)


def _crop_to_ink(canvas: Image.Image, clusters: list[Cluster], size: int) -> RenderedText:
    """Trim the vertical padding, keeping the boxes in step with the crop."""
    bbox = canvas.getbbox()
    if bbox:
        top = bbox[1]
        canvas = canvas.crop((0, top, canvas.width, bbox[3]))
        clusters = [
            Cluster(c.text, (c.box[0], c.box[1] - top, c.box[2], c.box[3] - top))
            for c in clusters
        ]
    return RenderedText(canvas, clusters, size)
