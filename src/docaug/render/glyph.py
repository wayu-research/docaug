"""Real-glyph handwriting: stitch text out of ink that a person actually wrote.

A handwriting *font* is still a font -- every 'ก' on the page is the same
outline. Sampling each character independently from a bank of real glyph
instances removes that regularity at its source.

**No glyph bank ships with this repository.** Handwriting instances are cut from
handwriting corpora, and their licences are not ours to pass on. What ships is
the renderer, the on-disk format, and `docs/glyph-bank.md`, which walks through
building your own. Without a bank, `--renderer glyph` says so and stops.

How a line is assembled: HarfBuzz shapes the text with the *shaping face* and is
used only for glyph **order** -- so leading vowels land left of their consonant
and marks follow their base. The font's own outlines are then thrown away, and
each character is drawn from the bank instead:

* **main band** -- sits on the baseline, scaled to its own true proportion so ญ
  keeps its descender and โ keeps its height rather than every glyph being
  squashed to one size, then the pen advances by its width plus a jittered gap;
* **above band** -- centred over the base and stacked upward, so a second mark
  goes above the first instead of on top of it;
* **below band** -- centred under the base.

Characters the bank has no instance for -- Latin, punctuation, digits, any Thai
it is missing -- fall back to the shaping face, scaled onto the bank's scale by
one shared factor. Scaling each fallback glyph to its own ink height instead
would inflate '-' and '.' into bars.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .. import thai
from ..config import Settings, get_settings
from ..fonts import FontBank
from ..shaping import (
    Face,
    fit_lines,
    ft_face,
    glyph_bitmap,
    shape_text,
    vertical_metrics,
)
from ..types import Cluster, Color, Page, RenderedText
from . import RENDERERS, RenderRequest

MARK_FRACTION = 0.45
"""A mark's ink is roughly 45% of the main-band height. Bank PNGs are tight ink
crops that carry no scale of their own, so this is what recovers one."""

REFERENCE_CHAR = "ก"
"""The x-height reference every other character is measured against."""

DOTTED_CIRCLE = "◌"
"""Combining marks are stored as `◌่`, `◌ี`, ... on disk. A bare mark makes a
filename that is invisible in a file browser and awkward in a shell."""


# --------------------------------------------------------------------------- #
# The bank
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class GlyphInstance:
    """One handwritten glyph: a tight ink mask, and the text height it came from."""

    ink: np.ndarray
    """uint8, 255 = ink, shaped (h, w)."""
    text_height: float
    """Main-band height of the source line, in this crop's pixels. Scaling by
    `target / text_height` puts the instance at the right size on a new line."""

    @property
    def size(self) -> tuple[int, int]:
        return self.ink.shape[1], self.ink.shape[0]


@dataclass
class GlyphBank:
    """`character -> [instances]`, loaded from a directory of PNGs.

    Layout on disk (see docs/glyph-bank.md)::

        bank/
          manifest.json     optional; {"chars": {"ก": 60, ...}}
          ก/0000.png        tight crop, dark ink on white
          ◌่/0000.png

    PNGs are stored dark-on-white so the bank is browsable; they are inverted to
    an ink mask on load.
    """

    root: Path
    instances: dict[str, list[GlyphInstance]] = field(default_factory=dict)

    def __len__(self) -> int:
        return sum(len(v) for v in self.instances.values())

    @classmethod
    def load(cls, root: Path | str) -> GlyphBank:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"glyph bank {root} does not exist. No bank ships with docaug -- "
                f"see docs/glyph-bank.md for the format and how to build one."
            )
        instances: dict[str, list[GlyphInstance]] = {}
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            char = directory.name.replace(DOTTED_CIRCLE, "")
            if not char:
                continue
            fraction = 1.0 if thai.band(char) == "main" else MARK_FRACTION
            loaded = [
                instance
                for path in sorted(directory.glob("*.png"))
                if (instance := _load_instance(path, fraction)) is not None
            ]
            if loaded:
                instances[char] = loaded
        if not instances:
            raise RuntimeError(f"glyph bank {root} contains no usable instances")
        return cls(root=root, instances=instances)

    def has(self, char: str) -> bool:
        return bool(self.instances.get(char))

    def sample(self, char: str, rng: random.Random) -> GlyphInstance | None:
        """Draw one instance. Each occurrence is drawn independently, so a
        repeated character is written differently each time -- the whole point."""
        pool = self.instances.get(char)
        return rng.choice(pool) if pool else None

    def coverage(self, text: str) -> float:
        chars = [c for c in text if c.strip() and thai.is_thai(c)]
        return sum(self.has(c) for c in chars) / len(chars) if chars else 0.0

    def summary(self) -> str:
        return f"GlyphBank({self.root}): {len(self.instances)} characters, {len(self)} instances"


def _load_instance(path: Path, fraction: float) -> GlyphInstance | None:
    gray = np.asarray(Image.open(path).convert("L"))
    if gray.size == 0:
        return None
    ink = (255 - gray).astype(np.uint8)
    return GlyphInstance(ink=ink, text_height=max(1.0, ink.shape[0] / fraction))


# --------------------------------------------------------------------------- #
# Font metrics used to place the real ink
# --------------------------------------------------------------------------- #

def _ink_extent(face: Face, size: int, char: str) -> tuple[int, int] | None:
    """(ink height, height above the baseline) for one character, in pixels."""
    ft = ft_face(face, size)
    gid = ft.get_char_index(ord(char)) if char else 0
    if not gid:
        return None
    bitmap, _left, top = glyph_bitmap(ft, gid)
    return (bitmap.shape[0], top) if bitmap.size else None


def char_proportions(face: Face, char: str, size: int = 64) -> tuple[float, float]:
    """(height, ascent) of `char` as multiples of the reference glyph's height.

    Bank instances are tight crops, so scaling them all to one height makes ก and
    โ the same size and flattens the line. These ratios put each character back
    at its true proportion, and say how much of it belongs above the baseline so
    ญ and ฐ can hang their descenders below it.
    """
    reference = _ink_extent(face, size, REFERENCE_CHAR)
    current = _ink_extent(face, size, char)
    if not reference or not current or reference[0] <= 0:
        return 1.0, 1.0
    height = min(max(current[0] / reference[0], 0.5), 2.4)
    ascent = min(max(current[1] / reference[0], 0.2), 2.4)
    return height, ascent


def reference_ink_height(face: Face, size: int) -> float:
    """Ink height of `ก` at `size` -- typically ~0.72 em, since an em box is
    taller than an x-height. Fallback glyphs are scaled by `size / this` so they
    stand at the same height as the stitched ones."""
    extent = _ink_extent(face, size, REFERENCE_CHAR)
    return float(extent[0]) if extent and extent[0] > 0 else 0.72 * size


def _paste_max(canvas: np.ndarray, ink: np.ndarray, x: int, y: int) -> None:
    """Composite by taking the brighter ink, so overlapping marks add rather than
    erase each other."""
    height, width = canvas.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + ink.shape[1]), min(height, y + ink.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    patch = ink[y0 - y : y1 - y, x0 - x : x1 - x]
    np.maximum(canvas[y0:y1, x0:x1], patch, out=canvas[y0:y1, x0:x1])


@dataclass(slots=True)
class _Placement:
    """Running state of the pen while a line is assembled."""

    pen: int
    base_center: float = 0.0
    base_bottom: float = 0.0
    above_top: float = 0.0


def stitch_line(
    text: str,
    bank: GlyphBank,
    face: Face,
    size: int,
    *,
    color: Color = (0, 0, 0),
    rng: random.Random,
    size_jitter: float = 0.10,
    baseline_jitter: float = 0.04,
) -> tuple[Image.Image, list[Cluster], int]:
    """Stitch one line -> (RGBA ink, cluster boxes, baseline y).

    `size` is the target main-glyph ink height, not an em size: bank instances
    are measured in ink, so that is the scale they share.
    """
    infos, positions = shape_text(text, face, size, monotone=True)
    if not positions:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), [], 0

    height = float(size)
    pad = size
    baseline = int(size * 2.4)  # headroom for ascenders and stacked marks
    canvas = np.zeros((size * 4, pad * 2 + int(1.6 * height * (len(text) + 2))), np.uint8)
    place = _Placement(pen=pad, above_top=baseline - height, base_bottom=baseline)
    ft = ft_face(face, size)
    fallback_scale = height / reference_ink_height(face, size)

    # Cluster id -> the end of the character span it covers. Ids are monotone
    # here because the shaper was asked for that, so one span ends where the
    # next begins.
    ids = sorted({i.cluster for i in infos if 0 <= i.cluster < len(text)})
    span_end = dict(zip(ids, [*ids[1:], len(text)], strict=True))

    boxes: dict[int, list[int]] = {}
    drawn: set[int] = set()
    for info, position in zip(infos, positions, strict=True):
        if not 0 <= info.cluster < len(text):
            continue
        # Several glyphs can share one cluster id: sara am (ำ) shapes into
        # nikhahit plus sara aa, and HarfBuzz merges it with a tone mark in front
        # of it. The bank is keyed by character, so the id is expanded back to
        # the characters it covers and each is drawn once, in logical order.
        # Keying the loop on the id instead drew the first character and lost the
        # rest -- น้ำ came out, and was labelled, as น้.
        for index in range(info.cluster, span_end[info.cluster]):
            if index in drawn:
                continue
            drawn.add(index)
            char = text[index]
            band = thai.band(char)

            instance = bank.sample(char, rng) if thai.is_thai(char) else None
            if instance is not None:
                box = _place_instance(canvas, instance, char, face, band, height, place,
                                      baseline, rng, size_jitter, baseline_jitter)
            else:
                box = _place_fallback(canvas, ft, char, band, height, place, baseline,
                                      fallback_scale, rng, size_jitter, baseline_jitter,
                                      position.x_advance)
            if box is None:
                continue
            current = boxes.setdefault(index, list(box))
            current[0], current[1] = min(current[0], box[0]), min(current[1], box[1])
            current[2], current[3] = max(current[2], box[2]), max(current[3], box[3])

    return _finish_line(canvas, text, boxes, color, baseline)


def _place_instance(canvas, instance, char, face, band, height, place, baseline,
                    rng, size_jitter, baseline_jitter) -> tuple | None:
    """Draw one real handwritten glyph and advance the pen."""
    proportion, ascent_ratio = char_proportions(face, char) if band == "main" else (1.0, 1.0)
    scale = proportion * height / instance.text_height
    scale *= 1 + rng.uniform(-1, 1) * size_jitter

    # Clamp per band. A bank crop can over- or under-grab its glyph, and an
    # unclamped outlier renders enormous and floating.
    limits = {"main": (0.62 * proportion, 1.35 * proportion),
              "above": (0.20, 0.55), "below": (0.20, 0.50)}[band]
    raw_h, raw_w = instance.ink.shape[0] * scale, instance.ink.shape[1] * scale
    target_h = min(max(raw_h, limits[0] * height), limits[1] * height)
    new_h = max(1, round(target_h))
    new_w = max(1, round(raw_w * target_h / max(1.0, raw_h)))
    ink = cv2.resize(instance.ink, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x, y = _band_position(band, new_w, new_h, height, place, baseline,
                          ascent_ratio / max(proportion, 1e-6), rng, baseline_jitter)
    _paste_max(canvas, ink, x, y)
    if band == "main":
        # Advance by the ink plus a jittered gap: real handwriting does not have
        # consistent side bearings.
        place.pen += new_w + round(0.10 * height * (1.0 + rng.uniform(-0.3, 0.4)))
    return (x, y, x + new_w, y + new_h)


def _place_fallback(canvas, ft, char, band, height, place, baseline, scale,
                    rng, size_jitter, baseline_jitter, advance) -> tuple | None:
    """Draw a character the bank lacks, from the shaping face."""
    gid = ft.get_char_index(ord(char)) if char else 0
    if not gid:
        return None
    bitmap, left, top = glyph_bitmap(ft, gid)
    if not bitmap.size:  # space and other zero-ink glyphs: advance only
        if band == "main":
            place.pen += max(round(0.25 * height), round(advance * scale))
        return None

    jittered = scale * (1 + rng.uniform(-1, 1) * size_jitter)
    new_w = max(1, round(bitmap.shape[1] * jittered))
    new_h = max(1, round(bitmap.shape[0] * jittered))
    ink = cv2.resize(bitmap, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if band == "main":
        x = place.pen + max(0, round(left * jittered))
        y = round(baseline - top * jittered + rng.uniform(-1, 1) * baseline_jitter * height)
        place.base_center, place.base_bottom, place.above_top = x + new_w / 2, y + new_h, y
        place.pen = x + max(round(advance * jittered), new_w)
    else:
        x, y = _band_position(band, new_w, new_h, height, place, baseline, 1.0, rng, 0.0)
    _paste_max(canvas, ink, x, y)
    return (x, y, x + new_w, y + new_h)


def _band_position(band, width, height_px, height, place, baseline, top_fraction,
                   rng, baseline_jitter) -> tuple[int, int]:
    """Where this glyph goes, and update the pen state for the next one."""
    if band == "main":
        x = place.pen
        # Sit on the baseline by the glyph's own ascent fraction, so a descender
        # (ญ ฐ ฏ) hangs below it instead of being lifted onto it.
        y = round(baseline - height_px * top_fraction
                  + rng.uniform(-1, 1) * baseline_jitter * height)
        place.base_center, place.base_bottom, place.above_top = x + width / 2, y + height_px, y
        return x, y
    x = round(place.base_center - width / 2)
    if band == "above":
        y = round(place.above_top - height_px + 0.18 * height)
        place.above_top = y  # the next mark stacks on top of this one
        return x, y
    return x, round(place.base_bottom - 0.12 * height)


def _finish_line(canvas, text, boxes, color, baseline):
    """Crop to ink and group the per-character boxes into cluster boxes."""
    ys, xs = np.nonzero(canvas)
    if not xs.size:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), [], 0
    x0, y0 = int(xs.min()), int(ys.min())
    cropped = canvas[y0 : int(ys.max()) + 1, x0 : int(xs.max()) + 1]

    rgba = np.zeros((*cropped.shape, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = cropped

    clusters: list[Cluster] = []
    for token, start, end in thai.cluster_spans(text):
        parts = [boxes[i] for i in range(start, end) if i in boxes]
        if not parts:
            # A space draws nothing but is part of the text. Attach it to the
            # previous cluster, which is what HarfBuzz does on the font path, so
            # both renderers produce labels that concatenate back to the input.
            # Anything else with no ink is genuinely not on the page, and
            # labelling it would be a lie.
            if token.isspace() and clusters:
                clusters[-1] = Cluster(clusters[-1].text + token, clusters[-1].box)
            continue
        clusters.append(
            Cluster(
                token,
                (min(p[0] for p in parts) - x0, min(p[1] for p in parts) - y0,
                 max(p[2] for p in parts) - x0, max(p[3] for p in parts) - y0),
            )
        )
    return Image.fromarray(rgba, "RGBA"), clusters, baseline - y0


# --------------------------------------------------------------------------- #
# The renderer
# --------------------------------------------------------------------------- #

class GlyphPageRenderer:
    """Fits stitched handwriting into a region, one page's worth."""

    def __init__(self, bank: GlyphBank, face: Face, settings: Settings,
                 rng: random.Random, min_coverage: float) -> None:
        self._bank, self._face, self._settings = bank, face, settings
        self._rng, self._min_coverage = rng, min_coverage

    @property
    def style(self) -> dict:
        return {"glyph_bank": str(self._bank.root), "shaping_font": self._face.name}

    def render(self, request: RenderRequest) -> RenderedText | None:
        if not request.text.strip():
            return None
        # Below this, the "handwriting" would be mostly font fallback; hand the
        # region to the next renderer in the chain instead of faking it.
        if self._bank.coverage(request.text) < self._min_coverage:
            return None

        settings = self._settings
        start = max(settings.min_size, request.start_size or int(request.height / settings.leading))
        size, lines, line_h = fit_lines(
            request.text, self._face, request.width, request.height,
            settings.leading, settings.min_size, start,
        )
        ascent, _ = vertical_metrics(self._face, size)

        pad = size
        canvas = Image.new("RGBA", (request.width, len(lines) * line_h + 2 * pad), (0, 0, 0, 0))
        clusters: list[Cluster] = []
        for i, line in enumerate(lines):
            ink, line_clusters, baseline = stitch_line(
                line, self._bank, self._face, size, color=request.color, rng=self._rng
            )
            y = pad + ascent + i * line_h - baseline
            canvas.alpha_composite(ink, (0, y))
            clusters += [
                Cluster(c.text, (c.box[0], c.box[1] + y, c.box[2], c.box[3] + y))
                for c in line_clusters
            ]
        if not clusters:
            return None
        return _fit_to_box(canvas, clusters, size, request.width, request.height)


def _fit_to_box(canvas, clusters, size, box_w, box_h) -> RenderedText:
    """Trim to ink, then scale down if the stitched line ran wider than the font
    line the fit loop measured. Boxes scale with it, so the labels stay exact."""
    bbox = canvas.getbbox()
    if bbox:
        canvas = canvas.crop(bbox)
        clusters = [
            Cluster(c.text, (c.box[0] - bbox[0], c.box[1] - bbox[1],
                             c.box[2] - bbox[0], c.box[3] - bbox[1]))
            for c in clusters
        ]
    scale = min(1.0, box_w / max(1, canvas.width), box_h / max(1, canvas.height))
    if scale < 1.0:
        canvas = canvas.resize(
            (max(1, round(canvas.width * scale)), max(1, round(canvas.height * scale))),
            Image.LANCZOS,
        )
        clusters = [Cluster(c.text, tuple(round(v * scale) for v in c.box)) for c in clusters]
        size = max(1, round(size * scale))
    return RenderedText(canvas, clusters, size)


@dataclass
class GlyphRenderer:
    """Stitches real handwriting glyphs, using `face` only for glyph order."""

    bank: GlyphBank
    face: Face
    settings: Settings
    min_coverage: float = 0.5

    def for_page(self, page: Page, rng: random.Random) -> GlyphPageRenderer:
        return GlyphPageRenderer(self.bank, self.face, self.settings, rng, self.min_coverage)


def _shaping_face(settings: Settings, font: str | None) -> Face:
    """A plain Thai text face for shaping and for fallback glyphs. Its outlines
    barely matter -- what matters is that it covers Thai, so HarfBuzz can order
    the glyphs."""
    bank = FontBank.load(settings.fonts_dir)
    if font:
        from .font import resolve_face

        return resolve_face(bank, font)
    handwriting = bank.by_category("handwriting")
    # A handwriting face keeps the fallback characters in keeping with the
    # stitched ones; a printed face beside real ink is immediately obvious.
    entry = (handwriting or bank.entries)[0]
    return entry.face(bank.root)


@RENDERERS.register("glyph")
def _glyph(
    settings: Settings | None = None,
    bank_dir: Path | str | None = None,
    font: str | None = None,
    min_coverage: float = 0.5,
) -> GlyphRenderer:
    settings = settings or get_settings()
    root = bank_dir or settings.glyph_bank_dir
    if root is None:
        raise RuntimeError(
            "the glyph renderer needs a handwriting glyph bank. Set "
            "DOCAUG_GLYPH_BANK_DIR in .env or pass --glyph-bank; no bank ships "
            "with docaug -- see docs/glyph-bank.md."
        )
    return GlyphRenderer(
        bank=GlyphBank.load(root),
        face=_shaping_face(settings, font),
        settings=settings,
        min_coverage=min_coverage,
    )
