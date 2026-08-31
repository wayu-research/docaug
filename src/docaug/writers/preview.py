"""Preview writer: source and reconstruction, side by side.

Every failure this pipeline has is visible in about two seconds and invisible in
any metric -- a patch where the inpainting gave up, text overflowing a table
cell, a face whose tone marks collide. So a handful of previews get written by
default, and the cap exists because after twenty you have seen the problem.

The cluster overlay is the one to turn on when you doubt the labels: it draws
every box the dataset claims, on the pixels it claims them for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from ..types import SynthPage
from . import WRITERS


@dataclass
class PreviewWriter:
    """Writes `out/<id>.png` as `source | reconstruction`."""

    out: Path
    limit: int = 20
    boxes: bool = False
    """Draw every cluster box over the reconstruction."""
    gap: int = 12

    _written: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.out = Path(self.out)
        self.out.mkdir(parents=True, exist_ok=True)

    def write(self, page: SynthPage) -> None:
        if self._written >= self.limit:
            return
        source = page.source
        right = page.image.convert("RGB")
        if self.boxes:
            right = right.copy()
            draw = ImageDraw.Draw(right)
            for label in page.labels:
                draw.rectangle(label.box, outline=(220, 40, 40), width=2)
                for cluster in label.clusters:
                    draw.rectangle(cluster.box, outline=(40, 120, 220))

        if source is None:
            canvas = right
        else:
            left = source.convert("RGB")
            height = max(left.height, right.height)
            canvas = Image.new(
                "RGB", (left.width + self.gap + right.width, height), (245, 245, 245)
            )
            canvas.paste(left, (0, 0))
            canvas.paste(right, (left.width + self.gap, 0))

        canvas.save(self.out / f"{page.id}.png")
        self._written += 1

    def close(self) -> None:
        return None


@WRITERS.register("preview")
def _preview(out: Path | str, limit: int = 20, boxes: bool = False) -> PreviewWriter:
    return PreviewWriter(out=Path(out), limit=limit, boxes=boxes)
