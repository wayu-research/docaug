"""Page transforms: vary one property of the source page at a time.

The pipeline's default is to keep the source page as it is -- its background, its
figures, its stamps and scan noise, and its two-dimensional arrangement. Each
transform here removes exactly one of those properties, so a corpus rendered with
and without it differs in that property and nothing else. Same text, same boxes,
same reading order, same typefaces.

===================  =====================================================
``keep``             Nothing removed. The default.
``white-background`` Non-text pixels become white; regions stay put. Removes
                     page context -- figures, texture, colour, degradation.
``stack``            Regions restacked into a single column in reading order.
                     Removes two-dimensional structure.
===================  =====================================================

Transforms run before erasing, so a region still carries its original pixels when
the eraser samples its ink colour. They return a new `Page`; nothing is mutated.
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from PIL import Image

from .registry import Registry
from .types import Page, Region


@runtime_checkable
class PageTransform(Protocol):
    """Rewrites a page before it is reconstructed."""

    def __call__(self, page: Page, rng: random.Random) -> Page: ...


TRANSFORMS: Registry[PageTransform] = Registry(
    "transform", entry_point_group="docaug.transforms"
)


class Keep:
    """Identity. The control condition."""

    def __call__(self, page: Page, rng: random.Random) -> Page:
        return page


class WhiteBackground:
    """Blank everything outside the annotated regions.

    The regions keep their original pixels -- including their text, which has not
    been erased yet -- so the eraser can still read each one's ink colour. What
    disappears is the page around them.
    """

    def __call__(self, page: Page, rng: random.Random) -> Page:
        source = page.image.convert("RGB")
        canvas = Image.new("RGB", page.size, (255, 255, 255))
        for region in page.regions:
            canvas.paste(source.crop(region.box), region.box[:2])
        return page.evolve(image=canvas)


class StackVertical:
    """Restack the regions into one column, in reading order.

    Each region keeps its own pixels, size and text; only its position changes.
    A model trained on these pages sees Thai documents with no columns, no
    tables and no side-by-side structure.
    """

    def __init__(self, margin: int = 40, gap: int = 24) -> None:
        self.margin, self.gap = margin, gap

    def __call__(self, page: Page, rng: random.Random) -> Page:
        if not page.regions:
            return page
        source = page.image.convert("RGB")
        # The source's own order is the reading order, and it is the order the
        # unstacked corpus is labelled in. Re-sorting by position here would
        # interleave the columns of a two-column page, so the stacked corpus
        # would differ from its control in reading order as well as in layout --
        # two variables, and the point of this transform is to move one.
        ordered = list(page.regions)
        width = page.size[0]
        usable = max(1, width - 2 * self.margin)

        height = self.margin + sum(r.height + self.gap for r in ordered)
        canvas = Image.new("RGB", (width, max(height, self.margin)), (255, 255, 255))

        regions: list[Region] = []
        y = self.margin
        for region in ordered:
            crop = source.crop(region.box)
            if crop.width > usable:  # a wide region would run off the column
                scale = usable / crop.width
                crop = crop.resize(
                    (usable, max(1, round(crop.height * scale))), Image.LANCZOS
                )
            canvas.paste(crop, (self.margin, y))
            regions.append(
                Region(
                    box=(self.margin, y, self.margin + crop.width, y + crop.height),
                    text=region.text,
                    category=region.category,
                    ink_height=region.ink_height * (crop.width / max(1, region.width)),
                )
            )
            y += crop.height + self.gap
        return page.evolve(image=canvas, regions=regions)


TRANSFORMS.register("keep")(Keep)
TRANSFORMS.register("white-background")(WhiteBackground)
TRANSFORMS.register("stack")(StackVertical)
