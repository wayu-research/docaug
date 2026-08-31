"""The pipeline: erase the source text, render new text into the same regions.

Six stages, and every one of them is something you can replace::

    source -> transforms -> text -> erase -> render -> writers

The order is not arbitrary. Transforms run first, while a region still has its
original pixels, so the eraser can read its ink colour. Text generation runs
before erasing because it is the slow, networked stage and its results are
cached -- a failure there should cost nothing downstream. Erasing runs over the
whole page in one pass so the composite never lands on pixels that a later
erasure is about to overwrite.

Placement inside a region is worth spelling out. The text is fitted into a box
shrunk by `pad_fraction` on every side, because Thai is routinely longer than the
source it replaces and unpadded text bleeds over cell borders. It is centred
horizontally. Vertically it is centred only when the slack is small -- a
single-line cell -- and otherwise top-aligned, so a short paragraph sits at the
top of its box like real text instead of floating in the middle.

The output labels come from the renderer's own cluster boxes, offset into page
coordinates. Nothing is inferred from the source annotation except the region
rectangle it was allowed to draw in.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .config import Settings, get_settings
from .erase import Eraser
from .render import PageRenderer, Renderer, RenderRequest
from .sources import Source
from .sources.regions import downscale
from .text import TextGenerator
from .transforms import PageTransform
from .types import Cluster, Page, RegionLabel, SynthPage
from .writers import Writer

log = logging.getLogger(__name__)

CENTERING_SLACK = 12
"""Vertical slack, in pixels, below which a region is centred rather than
top-aligned. Above it, the region is a paragraph and text belongs at the top."""


@dataclass
class RunReport:
    """What a run did. Printed at the end and written next to the dataset."""

    pages: int = 0
    regions: int = 0
    fitted: int = 0
    empty: int = 0
    """Regions whose generator returned nothing -- erased, and left erased."""
    failed: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    """Pages dropped for overflowing -- see `Pipeline.min_fit_rate`."""

    @property
    def fit_rate(self) -> float:
        return self.fitted / self.regions if self.regions else 1.0

    def summary(self) -> str:
        lines = [
            f"pages     {self.pages}",
            f"regions   {self.regions} ({self.empty} empty)",
            f"fit rate  {self.fit_rate:.1%}",
        ]
        if self.rejected:
            lines.append(f"rejected  {len(self.rejected)} page(s) below the fit threshold")
        if self.failed:
            lines.append(f"failed    {len(self.failed)}: {', '.join(self.failed[:5])}")
        return "\n".join(lines)


@dataclass
class Pipeline:
    """One reconstruction run, assembled from swappable parts."""

    source: Source
    text: TextGenerator
    eraser: Eraser
    renderer: Renderer
    writers: list[Writer] = field(default_factory=list)
    transforms: list[PageTransform] = field(default_factory=list)
    settings: Settings = field(default_factory=get_settings)
    keep_source_image: bool = True
    """Carry the source page in the output metadata, so the preview writer can
    show the before and after. Costs memory on very large pages."""
    min_fit_rate: float = 0.0
    """Drop a page unless this fraction of its regions fitted. Text that still
    overflows at the minimum size is drawn anyway, over whatever the page had
    next to it, and the neighbouring labels then describe pixels that are no
    longer there. 1.0 rejects any page with such a region, which is what a
    training corpus wants; 0.0 keeps everything and leaves the decision to
    whoever reads `fit` out of the labels."""

    def run(self, pages: Iterable[Page] | None = None) -> RunReport:
        """Reconstruct every page and hand it to the writers."""
        report = RunReport()
        try:
            for index, page in enumerate(pages if pages is not None else self.source):
                try:
                    synthesized = self.process(page, index)
                except Exception:  # one bad page must not end the run
                    log.exception("page %s failed", page.id)
                    report.failed.append(page.id)
                    continue
                if synthesized.fit_rate < self.min_fit_rate:
                    log.info("page %s rejected: fit rate %.2f is below %.2f",
                             page.id, synthesized.fit_rate, self.min_fit_rate)
                    report.rejected.append(page.id)
                    continue
                for writer in self.writers:
                    writer.write(synthesized)
                report.pages += 1
                report.regions += len(synthesized.labels)
                report.fitted += sum(label.fit for label in synthesized.labels)
                report.empty += sum(not label.text.strip() for label in synthesized.labels)
        finally:
            for writer in self.writers:
                writer.close()
        return report

    def process(self, page: Page, index: int = 0) -> SynthPage:
        """Reconstruct one page. Deterministic in `index` and the run seed."""
        rng = random.Random(self.settings.rng_seed(index))
        page = downscale(page, self.settings.max_side)
        for transform in self.transforms:
            page = transform(page, rng)

        texts = self.text(page, rng)
        if len(texts) != len(page.regions):
            raise ValueError(
                f"{type(self.text).__name__} returned {len(texts)} texts "
                f"for {len(page.regions)} regions"
            )

        erased = self.eraser(page)
        page_renderer = self.renderer.for_page(page, rng)

        canvas = erased.image.convert("RGBA")
        labels: list[RegionLabel] = []
        for region, text, color in zip(page.regions, texts, erased.colors, strict=True):
            label, placed = self._reconstruct(page_renderer, region, text, color)
            labels.append(label)
            if placed is not None:
                ink, x, y = placed
                canvas.alpha_composite(ink, (x, y))

        return SynthPage(
            id=page.id,
            image=canvas.convert("RGB"),
            labels=labels,
            meta={**page.meta, "style": page_renderer.style, "index": index},
            source=page.image if self.keep_source_image else None,
        )

    def _reconstruct(self, page_renderer: PageRenderer, region, text, color):
        """Render one region -> (its label, and the ink to composite)."""
        settings = self.settings
        x1, y1, x2, y2 = region.box
        width, height = region.width, region.height
        label = RegionLabel(
            box=region.box, category=region.category, text=text, source_text=region.text
        )
        if not (text.strip() and width > 2 and height > 2):
            return label, None

        request = RenderRequest(
            text=text,
            width=max(4, round(width * (1 - settings.pad_fraction))),
            height=max(4, round(height * (1 - settings.pad_fraction))),
            color=color,
            start_size=round(region.ink_height * settings.size_scale) or None,
        )
        ink = page_renderer.render(request)
        if ink is None:
            return label, None

        x = x1 + max(0, (width - ink.width) // 2)
        slack = height - ink.height
        y = y1 + (
            slack // 2
            if 0 <= slack <= CENTERING_SLACK
            else max(1, round(height * settings.pad_fraction / 2))
        )

        label.size = ink.size
        label.fit = ink.width <= width + 1 and ink.height <= height + 1
        label.clusters = [
            Cluster(c.text, (c.box[0] + x, c.box[1] + y, c.box[2] + x, c.box[3] + y))
            for c in ink.clusters
        ]
        return label, (ink.image, x, y)

    def __iter__(self) -> Iterator[SynthPage]:
        """Reconstruct pages lazily, bypassing the writers. Handy in a notebook."""
        for index, page in enumerate(self.source):
            yield self.process(page, index)
