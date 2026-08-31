"""The font renderer: sample a typeface per page, fit text into each region.

The style decisions -- which face, bold or not, upright or oblique, warped or
clean -- are made once per page, from that page's own RNG. That is why a run is
reproducible from a single seed and why re-running with a different seed gives a
genuinely different-looking corpus rather than the same pages in a new order.

Two of the probabilities are conditional rather than flat, because the
unconditional versions produce documents that do not exist. A handwriting face is
only drawn for a *sparse* page: a dense two-column report set in a script face is
not something anyone ever printed. And the per-instance warp is only offered to
handwriting faces, since jittering a printed face just makes it look broken.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .. import thai
from ..config import Settings, get_settings
from ..fonts import FontBank, FontEntry
from ..shaping import Face, render_block
from ..types import Page, RenderedText
from . import RENDERERS, RenderRequest
from .handwriting import render_warped, warp_strength


@dataclass(frozen=True, slots=True)
class PageStyle:
    """Every typographic decision for one page."""

    entry: FontEntry
    face: Face
    weight: float = 0.0
    """Faux-bold dilation, in pixels. 0 when a real bold face was available."""
    slant: float = 0.0
    warp: float = 0.0
    """Handwriting warp strength; 0 disables it."""

    def as_dict(self) -> dict:
        return {
            "font": self.entry.family,
            "font_file": self.entry.path,
            "category": self.entry.category,
            "weight": self.entry.weight,
            "faux_bold": self.weight,
            "slant": self.slant,
            "warp": self.warp,
        }


def sample_style(bank: FontBank, page: Page, rng: random.Random, settings: Settings) -> PageStyle:
    """Draw a page's typographic style from the bank."""
    sparse = len(page.regions) <= settings.handwriting_max_regions
    handwritten = sparse and rng.random() < settings.handwriting_prob

    categories = {"handwriting": 1.0} if handwritten else None
    entry = bank.sample(rng, categories=categories)

    bold = rng.random() < settings.bold_prob
    heavy = bank.variant(entry, bold=bold)
    # Only fall back to dilating the outline when the family has no real bold.
    faux_bold = 1.0 if bold and heavy is entry else 0.0

    warp = 0.0
    if handwritten and rng.random() < settings.augment_prob:
        warp = warp_strength(rng, settings.augment_min, settings.augment_max)

    return PageStyle(
        entry=heavy,
        face=heavy.face(bank.root),
        weight=faux_bold,
        slant=settings.slant if rng.random() < settings.italic_prob else 0.0,
        warp=warp,
    )


class FontPageRenderer:
    """Draws one page's regions in a settled `PageStyle`."""

    def __init__(self, style: PageStyle, settings: Settings, rng: random.Random) -> None:
        self._style = style
        self._settings = settings
        self._rng = rng

    @property
    def style(self) -> dict:
        return self._style.as_dict()

    def render(self, request: RenderRequest) -> RenderedText | None:
        if not request.text.strip():
            return None
        style, settings = self._style, self._settings
        common = dict(
            color=request.color,
            leading=settings.leading,
            min_size=settings.min_size,
            start_size=request.start_size,
            weight=style.weight,
            slant=style.slant,
        )
        if style.warp > 0:
            return render_warped(
                request.text, style.face, request.width, request.height,
                strength=style.warp, rng=self._rng, supersample=settings.supersample,
                **common,
            )
        return render_block(request.text, style.face, request.width, request.height, **common)


@dataclass
class FontRenderer:
    """Samples a typeface per page from `bank`."""

    bank: FontBank
    settings: Settings
    fixed: Face | None = None
    """Pin every page to one face. This is the single-typeface control condition;
    it is a much weaker teacher than the sampled bank, which is the point."""

    def for_page(self, page: Page, rng: random.Random) -> FontPageRenderer:
        if self.fixed is not None:
            entry = FontEntry(path=self.fixed.path, family=self.fixed.name, category="fixed")
            style = PageStyle(entry=entry, face=self.fixed)
        else:
            style = sample_style(self.bank, page, rng, self.settings)
        return FontPageRenderer(style, self.settings, rng)


def require_thai_coverage(face: Face, name: str) -> Face:
    """Refuse a face that cannot draw modern Thai.

    The bank filters its own entries on coverage when it loads, so this exists
    for the one face that never went through it: the one `--font <path>` points
    straight at. Without the check it rasterizes tofu boxes, and the labels call
    them words.
    """
    missing = face.missing([c for c in thai.required_codepoints() if c not in thai.RARE])
    if missing:
        sample = "".join(chr(c) for c in sorted(missing)[:8])
        raise RuntimeError(
            f"{name} cannot draw Thai: {len(missing)} required characters are missing "
            f"({sample}). Rendering it would put tofu boxes in the ground truth. Pick "
            f"a family from `docaug fonts`, or lay your own out by category, run "
            f"`docaug fonts scan --dir`, and point --fonts-dir at it."
        )
    return face


def resolve_face(bank: FontBank, font: str) -> Face:
    """`font` is a path to a font file, or a family name in the bank."""
    path = Path(font)
    if not path.is_file():
        return bank.find(font).face(bank.root)
    return require_thai_coverage(Face(str(path)), path.name)


@RENDERERS.register("font")
def _font(
    bank: FontBank | None = None,
    settings: Settings | None = None,
    font: str | None = None,
) -> FontRenderer:
    settings = settings or get_settings()
    bank = bank or FontBank.load(settings.fonts_dir)
    return FontRenderer(
        bank=bank, settings=settings, fixed=resolve_face(bank, font) if font else None
    )
