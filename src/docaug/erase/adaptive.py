"""The default eraser: local thresholding, then fill or inpaint.

Finding the ink is the whole problem, and the naive approach fails in a specific,
visible way. Thresholding a region against a global background level swallows any
uniformly coloured area that merely happens to be darker than the page -- a
region overlapping a grey banner masks the entire banner, and the fill then
paints it a flat colour, leaving an obvious rectangular patch. A *local* adaptive
threshold asks whether each pixel is darker than its own neighbourhood, which is
what "text" actually means, so the banner stays in the background where it
belongs.

Filling is then chosen by how busy the background is:

* near-uniform (white paper, a solid fill) -> flat-fill with the background
  colour, which is fast and exactly right;
* textured, gradient, or photographic -> `cv2.inpaint`, which reconstructs from
  the neighbours instead of stamping a flat patch over the texture.

Polarity is detected too, so light-on-dark text -- cover pages, coloured
sidebars -- is erased rather than having its background masked instead.

Known simplification: a horizontal rule inside a region (a form underline, a
table border) reads as ink and is erased with the text. Preserving page structure
under the replacement text is a natural thing to add as your own eraser.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..types import Color, ErasedPage, Page
from . import ERASERS

DARK_ON_LIGHT_THRESHOLD = 110
"""Median luminance above this means dark text on light paper."""

FLAT_FILL_STD = 8.0
"""Luminance spread of the non-text pixels below which the background counts as
uniform and a flat fill beats inpainting."""


def _ink_mask(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Locally-adaptive text mask -> (mask, dark_on_light)."""
    dark_on_light = float(np.median(gray)) >= DARK_ON_LIGHT_THRESHOLD
    # Window of roughly one text height, forced odd and kept in a sane range.
    block = int(np.clip(gray.shape[0] | 1, 11, 51)) | 1
    mode = cv2.THRESH_BINARY_INV if dark_on_light else cv2.THRESH_BINARY
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, block, 10)
    # Grow by a couple of pixels so anti-aliased glyph edges go too; a surviving
    # halo is what makes an erased page look erased.
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2), dark_on_light


def _sample_ink_color(rgb: np.ndarray, gray: np.ndarray, dark_on_light: bool) -> Color:
    """The colour of the text, taken from the darkest (or lightest) ink core.

    Sampling the whole mask would average in anti-aliased edge pixels and come
    back washed out, so we take a percentile of the core only.
    """
    if dark_on_light:
        core = gray < np.percentile(gray, 85) - 50
        percentile = 35
        default: Color = (0, 0, 0)
    else:
        core = gray > np.percentile(gray, 15) + 50
        percentile = 65
        default = (255, 255, 255)

    pixels = rgb[core]
    if pixels.shape[0] < 5:
        return default
    return tuple(int(np.percentile(pixels[:, c], percentile)) for c in range(3))


def erase_region(crop_rgb: np.ndarray) -> tuple[np.ndarray, Color]:
    """Erase the text in one region crop -> (clean crop, its text colour)."""
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    mask, dark_on_light = _ink_mask(gray)
    color = _sample_ink_color(crop_rgb, gray, dark_on_light)

    background = gray[mask == 0]
    if background.size == 0 or float(background.std()) < FLAT_FILL_STD:
        clean = crop_rgb.copy()
        clean[mask > 0] = np.percentile(
            crop_rgb.reshape(-1, 3), 80 if dark_on_light else 20, axis=0
        )
    else:
        clean = cv2.inpaint(crop_rgb, mask, 3, cv2.INPAINT_TELEA)
    return clean, color


class AdaptiveEraser:
    """Erases every annotated region of a page in place."""

    def __call__(self, page: Page) -> ErasedPage:
        canvas = np.array(page.image.convert("RGB"))
        colors: list[Color] = []
        for region in page.regions:
            x1, y1, x2, y2 = region.box
            crop = canvas[y1:y2, x1:x2]
            if crop.size == 0:
                colors.append((0, 0, 0))
                continue
            canvas[y1:y2, x1:x2], color = erase_region(crop)
            colors.append(color)
        return ErasedPage(Image.fromarray(canvas), colors)


class KeepEraser:
    """Leaves the page untouched. Useful for rendering onto blank stock, and as
    the control condition when you want to see what erasure is buying you."""

    def __call__(self, page: Page) -> ErasedPage:
        return ErasedPage(page.image.convert("RGB"), [(0, 0, 0)] * len(page.regions))


@ERASERS.register("adaptive")
def _adaptive() -> AdaptiveEraser:
    return AdaptiveEraser()


@ERASERS.register("none")
def _none() -> KeepEraser:
    return KeepEraser()
