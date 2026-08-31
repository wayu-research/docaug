"""Region utilities shared by every source.

Annotation granularity varies wildly between datasets -- some label whole
paragraphs, some label lines, some label individual words -- and it matters here
in a way it does not for detection training. Translating word boxes one at a time
is meaningless for Thai, which reorders freely and has no inter-word spaces, and
rendering each wrapped line of a paragraph separately gives every line its own
independently-fitted type size, which no real document does.

So sources normalize: group word boxes back into rows, then merge the rows of a
wrapped paragraph back into one block. A dataset that is already block-level
passes through both steps unchanged.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..types import Page, Region


def measure_ink_height(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    """Height of the dark pixels in a region: a proxy for the source type size.

    Feeding this to the renderer as `start_size` is what preserves a document's
    type hierarchy. Without it, every region is fitted to fill its own box and a
    footnote comes out the same size as the title.
    """
    x1, y1, x2, y2 = box
    crop = np.asarray(image.convert("L"))[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    rows = np.nonzero((crop < np.percentile(crop, 85) - 25).any(axis=1))[0]
    return float(rows[-1] - rows[0] + 1) if rows.size >= 2 else float(crop.shape[0])


def _union(regions: list[Region]) -> Region:
    sizes = [r.ink_height for r in regions if r.ink_height]
    return Region(
        box=(
            min(r.box[0] for r in regions), min(r.box[1] for r in regions),
            max(r.box[2] for r in regions), max(r.box[3] for r in regions),
        ),
        text=" ".join(r.text for r in regions).strip(),
        category=regions[0].category,
        # Keep the smallest observed size: a merged block is rendered at one
        # size, and overshooting is what makes text overflow its box.
        ink_height=min(sizes) if sizes else 0.0,
    )


def group_rows(regions: list[Region], column_gap: float = 30.0) -> list[Region]:
    """Rebuild text lines from word-level boxes.

    Boxes that overlap vertically and share a category are one row, ordered
    left to right. A large horizontal gap inside a row is a column or table-cell
    boundary, and splits it. Line-level input comes back unchanged.
    """
    rows: list[dict] = []
    for region in sorted(regions, key=lambda r: ((r.box[1] + r.box[3]) / 2, r.box[0])):
        _, y1, _, y2 = region.box
        for row in rows:
            if row["category"] != region.category:
                continue
            top, bottom = row["span"]
            overlap = max(0, min(y2, bottom) - max(y1, top))
            if overlap >= 0.5 * min(y2 - y1, bottom - top):
                row["members"].append(region)
                row["span"] = (min(top, y1), max(bottom, y2))
                break
        else:
            rows.append({"category": region.category, "span": (y1, y2), "members": [region]})

    out: list[Region] = []
    for row in rows:
        members = sorted(row["members"], key=lambda r: r.box[0])
        segment = [members[0]]
        for previous, current in zip(members, members[1:], strict=False):
            if current.box[0] - previous.box[2] > column_gap:
                out.append(_union(segment))
                segment = [current]
            else:
                segment.append(current)
        out.append(_union(segment))
    return out


def merge_wrapped(regions: list[Region]) -> list[Region]:
    """Merge the visual lines of a wrapped paragraph back into one block.

    Two regions belong together when they share a category, share a column (left
    edges within a few pixels, or strong horizontal overlap), and are stacked
    tightly enough that the gap reads as line spacing rather than a new row.

    The gap is compared against a *single* line's height, not the merged block's
    -- otherwise each merge inflates the threshold and the paragraph swallows the
    heading below it.
    """
    out: list[Region] = []
    previous_height = 0
    for region in sorted(regions, key=lambda r: (r.box[1], r.box[0])):
        x1, y1, x2, y2 = region.box
        height = y2 - y1
        if out:
            last = out[-1]
            lx1, _, lx2, ly2 = last.box
            overlap = max(0, min(lx2, x2) - max(lx1, x1))
            narrow = max(1, min(lx2 - lx1, x2 - x1))
            line_height = max(height, previous_height, 1)
            if (
                region.category == last.category
                and (abs(x1 - lx1) <= 8 or overlap / narrow > 0.6)
                and -3 <= y1 - ly2 <= 0.6 * line_height
            ):
                out[-1] = _union([last, region])
                previous_height = height
                continue
        out.append(region)
        previous_height = height
    return out


def normalize(regions: list[Region], *, column_gap: float = 30.0) -> list[Region]:
    """`group_rows` then `merge_wrapped`: the standard cleanup for a source."""
    return merge_wrapped(group_rows(regions, column_gap))


def prepare(
    image: Image.Image,
    regions: list[Region],
    *,
    group: bool = True,
    column_gap: float = 30.0,
) -> list[Region]:
    """The standard source pipeline: clip, measure, then group.

    Measuring comes *before* grouping, and the order is not cosmetic. Ink height
    is read from the dark pixels in a box, so measuring a merged three-line
    paragraph returns the height of all three lines and the renderer sets the
    paragraph in something like 40pt. Measure each line first, and the merge
    carries the smallest of them -- which is the size the block should be.
    """
    regions = clip(regions, image.size)
    regions = [
        region
        if region.ink_height
        else Region(region.box, region.text, region.category,
                    measure_ink_height(image, region.box))
        for region in regions
    ]
    return normalize(regions, column_gap=column_gap) if group else regions


def downscale(page: Page, max_side: int) -> Page:
    """Shrink a page and its regions together, so nothing drifts out of place."""
    width, height = page.size
    if max(width, height) <= max_side:
        return page
    scale = max_side / max(width, height)
    image = page.image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    regions = [
        Region(
            box=tuple(round(v * scale) for v in region.box),
            text=region.text,
            category=region.category,
            ink_height=region.ink_height * scale,
        )
        for region in page.regions
    ]
    return page.evolve(image=image, regions=regions)


def clip(regions: list[Region], size: tuple[int, int], min_side: int = 3) -> list[Region]:
    """Drop degenerate and empty regions, and keep the rest inside the page."""
    width, height = size
    out = []
    for region in regions:
        x1, y1, x2, y2 = region.box
        box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        if region.text.strip() and box[2] - box[0] >= min_side and box[3] - box[1] >= min_side:
            out.append(Region(box, region.text, region.category, region.ink_height))
    return out
