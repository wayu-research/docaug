"""Renderers: draw the replacement text.

A renderer is asked for one region at a time, but style is a *page* property --
a document does not change typeface every paragraph. So the protocol has two
levels: `for_page` samples the page's style once and returns a `PageRenderer`,
which then draws each region in that style.

    renderer = RENDERERS.create("font", bank=bank, settings=settings)
    page_renderer = renderer.for_page(page, rng)
    ink = page_renderer.render(RenderRequest(text, w, h, color=col))

`render` may return `None` to mean "not mine" -- a glyph stitcher does that for
a region it has no glyphs for. `chain` then falls through to the next renderer,
which is how real-glyph handwriting degrades to a handwriting face for the
characters a bank does not cover.

Built-in renderers:

===========  ==========================================================
``font``     Sample a face from the bank; optional handwriting-style warp.
``glyph``    Stitch real handwriting glyph instances from a bank on disk.
``chain``    Try renderers in order, first non-``None`` wins.
===========  ==========================================================
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..registry import Registry
from ..types import Color, Page, RenderedText


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """One region's worth of work."""

    text: str
    width: int
    height: int
    color: Color = (0, 0, 0)
    start_size: int | None = None
    """Type size to try first, from the measured source ink height. `None` means
    fill the box."""


@runtime_checkable
class PageRenderer(Protocol):
    """Draws regions of one page, in one settled style."""

    def render(self, request: RenderRequest) -> RenderedText | None: ...

    @property
    def style(self) -> dict:
        """A description of the sampled style, recorded in the page metadata."""


@runtime_checkable
class Renderer(Protocol):
    """Samples a per-page style and hands back something that can draw."""

    def for_page(self, page: Page, rng: random.Random) -> PageRenderer: ...


RENDERERS: Registry[Renderer] = Registry("renderer", entry_point_group="docaug.renderers")


class ChainPageRenderer:
    """First renderer that returns ink wins."""

    def __init__(self, members: list[PageRenderer]) -> None:
        self._members = members

    def render(self, request: RenderRequest) -> RenderedText | None:
        for member in self._members:
            if (ink := member.render(request)) is not None:
                return ink
        return None

    @property
    def style(self) -> dict:
        merged: dict = {}
        for member in reversed(self._members):  # earlier members win on conflict
            merged.update(member.style)
        return merged


@dataclass
class ChainRenderer:
    """Composes renderers, e.g. real glyphs with a handwriting face behind them."""

    members: list[Renderer]

    def for_page(self, page: Page, rng: random.Random) -> ChainPageRenderer:
        return ChainPageRenderer([m.for_page(page, rng) for m in self.members])


@RENDERERS.register("chain")
def _chain(members: list[Renderer]) -> ChainRenderer:
    return ChainRenderer(list(members))


from . import font, glyph  # noqa: E402,F401  (registers the built-ins)

__all__ = [
    "ChainRenderer",
    "PageRenderer",
    "RENDERERS",
    "RenderRequest",
    "Renderer",
]
