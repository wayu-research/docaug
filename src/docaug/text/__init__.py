"""Text generators: decide what the reconstructed page will say.

A generator receives a page and returns one string per region, in order. Which
one you pick controls the *source domain* of the result:

=============  =========================================================
``keep``       Reuse the source text. In-domain reconstruction when your
               pages are already in the target language, and the only
               generator that needs no API key.
``translate``  Translate each region into Thai. Out-of-domain
               reconstruction from an English or Chinese corpus.
``synth``      Invent plausible Thai of a similar length and kind, from
               the region's category. Use when the source text is
               unreliable -- garbled OCR, a broken PDF text layer -- or
               when it must not be reproduced.
=============  =========================================================

Returning `""` for a region is allowed and means "leave it erased".
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from ..registry import Registry
from ..types import Page


@runtime_checkable
class TextGenerator(Protocol):
    """Produces the replacement text for every region of a page."""

    def __call__(self, page: Page, rng: random.Random) -> list[str]: ...


TEXT_GENERATORS: Registry[TextGenerator] = Registry(
    "text generator", entry_point_group="docaug.text_generators"
)

from . import providers  # noqa: E402,F401  (registers the built-ins)

__all__ = ["TEXT_GENERATORS", "TextGenerator"]
