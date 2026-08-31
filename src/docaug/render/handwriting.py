"""Per-instance warp: make a static face stop repeating itself.

A font draws every 'ก' identically. Handwriting does not, and a model trained on
the font version learns the font. So each cluster gets its own small affine
jitter, and the whole line gets a low-frequency elastic wobble.

The naive way to do this loses a third of the edge acutance -- warping a
1x raster resamples already-anti-aliased pixels, and the result looks soft in a
way real scans do not. The fix is to render crisp at N times the target size,
warp there, and area-downsample into place: the downsample *is* the
anti-aliasing. The warp magnitudes scale with glyph height and the downsample
divides that back out, so supersampling buys sharpness and changes nothing else.

Labels survive because the same warp is applied to an integer label map, and each
cluster box is re-derived from where its pixels actually landed. A box after
warping describes warped ink, not the ink we started with.

The elastic field uses a large sigma on purpose. A high-frequency field would
move a tone mark independently of the consonant under it, detaching a four-level
stack into unreadable pieces; a smooth field moves the whole stack together.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image

from ..shaping import DEFAULT_LEADING, Face, render_block
from ..types import Cluster, Color, RenderedText

MAX_STRENGTH = 1.1
"""Above this, neighbouring glyphs collide and their boxes stop being meaningful."""

_REFERENCE_HEIGHT = 100.0
"""Glyph height the magnitudes below were tuned at. Everything scales from it."""


def warp_strength(rng: random.Random, low: float = 0.4, high: float = MAX_STRENGTH) -> float:
    return rng.uniform(low, min(high, MAX_STRENGTH))


def _label_map(alpha: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    """Paint each cluster's ink with its own integer id, so the warp can carry
    identity along with pixels."""
    labels = np.zeros(alpha.shape, np.int32)
    for index, box in enumerate(boxes, start=1):
        if box is None:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        window = labels[y0:y1, x0:x1]
        window[alpha[y0:y1, x0:x1] > 30] = index
    return labels


def _boxes_from_labels(labels: np.ndarray, count: int) -> list[tuple | None]:
    """Re-derive each cluster's box from where its pixels ended up."""
    boxes: list[tuple | None] = []
    for index in range(1, count + 1):
        ys, xs = np.nonzero(labels == index)
        boxes.append(
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            if xs.size
            else None
        )
    return boxes


def _glyph_scale(boxes: list) -> float:
    heights = [b[3] - b[1] for b in boxes if b]
    return max(8.0, float(np.median(heights))) / _REFERENCE_HEIGHT if heights else 1.0


def jitter(alpha, labels, boxes, rng, strength, scale):
    """Rotate, scale and shift each cluster about its own centre.

    Horizontal drift is kept much tighter than vertical: sideways movement is
    what makes neighbours overlap and boxes ambiguous, while a wandering baseline
    is exactly what handwriting looks like.
    """
    height, width = alpha.shape
    out_alpha, out_labels = np.zeros_like(alpha), np.zeros_like(labels)
    for index in range(1, len(boxes) + 1):
        mask = labels == index
        ys, xs = np.nonzero(mask)
        if not xs.size:
            continue
        matrix = cv2.getRotationMatrix2D(
            (float(xs.mean()), float(ys.mean())),
            rng.uniform(-8, 8) * strength,
            1 + rng.uniform(-0.13, 0.13) * strength,
        )
        matrix[0, 2] += rng.uniform(-4, 4) * strength * scale
        matrix[1, 2] += rng.uniform(-13, 13) * strength * scale
        ink = np.where(mask, alpha, 0).astype(np.float32)
        moved = cv2.warpAffine(ink, matrix, (width, height), flags=cv2.INTER_LINEAR)
        moved_labels = cv2.warpAffine(
            (mask * index).astype(np.int32), matrix, (width, height), flags=cv2.INTER_NEAREST
        )
        out_alpha = np.maximum(out_alpha, moved.astype(alpha.dtype))
        out_labels = np.where(moved_labels > 0, moved_labels, out_labels)
    return out_alpha, out_labels


def elastic(alpha, labels, boxes, rng, strength, scale):
    """Simard-style elastic distortion with a deliberately smooth field."""
    height, width = alpha.shape
    sigma, magnitude = 16.0 * scale, 14.0 * strength * scale
    state = np.random.default_rng(rng.randrange(2**32))

    def field() -> np.ndarray:
        noise = state.uniform(-1, 1, (height, width)).astype(np.float32)
        smooth = cv2.GaussianBlur(noise, (0, 0), sigma)
        return smooth * (magnitude / (np.abs(smooth).max() + 1e-6))

    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x, map_y = xs + field(), ys + field()
    return (
        cv2.remap(alpha, map_x, map_y, cv2.INTER_LINEAR),
        cv2.remap(labels, map_x, map_y, cv2.INTER_NEAREST),
    )


PIPELINE = (jitter, elastic)
"""Warp operations, applied in order. Replace it to change the deformation
model -- the contract is `(alpha, labels, boxes, rng, strength, scale) ->
(alpha, labels)`."""


def _apply(alpha: np.ndarray, boxes: list, strength: float, rng: random.Random):
    scale = _glyph_scale(boxes)
    labels = _label_map(alpha, boxes)
    for operation in PIPELINE:
        alpha, labels = operation(alpha, labels, boxes, rng, strength, scale)
    return alpha, _boxes_from_labels(labels, len(boxes))


def render_warped(
    text: str,
    face: Face,
    box_w: int,
    box_h: int,
    *,
    strength: float,
    rng: random.Random,
    supersample: int = 3,
    color: Color = (0, 0, 0),
    leading: float = DEFAULT_LEADING,
    min_size: int = 8,
    start_size: int | None = None,
    align: str = "left",
    weight: float = 0.0,
    slant: float = 0.0,
) -> RenderedText:
    """`render_block` plus a per-instance warp. Same contract, softer glyphs."""
    scaled = supersample if strength > 0 else 1
    block = render_block(
        text, face, box_w * scaled, box_h * scaled,
        color=color, leading=leading, min_size=min_size,
        start_size=start_size * scaled if start_size else None,
        align=align, weight=weight, slant=slant,
    )
    if strength <= 0 or not block.clusters:
        return _downsample_plain(block, scaled)

    alpha = np.array(block.image)[:, :, 3]
    # Pad generously: jitter and elastic both push ink outside the original
    # bounds, and clipping it would truncate glyphs and their boxes with them.
    pad = int(block.height * 0.35) + 12
    padded = np.pad(alpha, pad)
    boxes = [(c.box[0] + pad, c.box[1] + pad, c.box[2] + pad, c.box[3] + pad)
             for c in block.clusters]
    warped, warped_boxes = _apply(padded, boxes, strength, rng)

    ys, xs = np.nonzero(warped)
    if not xs.size:
        return _downsample_plain(block, scaled)
    x0, y0 = int(xs.min()), int(ys.min())
    crop = warped[y0 : int(ys.max()) + 1, x0 : int(xs.max()) + 1]

    target = (max(1, round(crop.shape[1] / scaled)), max(1, round(crop.shape[0] / scaled)))
    down = cv2.resize(crop, target, interpolation=cv2.INTER_AREA)
    rgba = np.zeros((*down.shape, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = down

    clusters = [
        Cluster(
            cluster.text,
            tuple(round(v) for v in (
                (box[0] - x0) / scaled, (box[1] - y0) / scaled,
                (box[2] - x0) / scaled, (box[3] - y0) / scaled,
            )),
        )
        if box
        else cluster
        for cluster, box in zip(block.clusters, warped_boxes, strict=True)
    ]
    return RenderedText(Image.fromarray(rgba, "RGBA"), clusters, max(1, round(block.size / scaled)))


def _downsample_plain(block: RenderedText, scaled: int) -> RenderedText:
    """Bring an un-warped supersampled render back to target resolution."""
    if scaled == 1:
        return block
    block.image = block.image.resize(
        (max(1, block.width // scaled), max(1, block.height // scaled)), Image.LANCZOS
    )
    block.clusters = [
        Cluster(c.text, tuple(round(v / scaled) for v in c.box)) for c in block.clusters
    ]
    block.size = max(1, round(block.size / scaled))
    return block
