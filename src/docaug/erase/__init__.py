"""Erasers: remove the source text, keep the page.

An eraser takes the real page and its annotated regions and returns the page with
the text pixels gone -- and, for each region, the colour that text was. The
colour matters as much as the erasure: reading it back means the replacement text
inherits the document's own palette, so a blue heading stays a blue heading
instead of turning flat black.

Swap in your own with `@ERASERS.register("name")`; see docs/extending.md.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..registry import Registry
from ..types import ErasedPage, Page


@runtime_checkable
class Eraser(Protocol):
    """Removes source text from a page."""

    def __call__(self, page: Page) -> ErasedPage: ...


ERASERS: Registry[Eraser] = Registry("eraser", entry_point_group="docaug.erasers")

from . import adaptive  # noqa: E402,F401  (registers the built-ins)

__all__ = ["ERASERS", "Eraser"]
