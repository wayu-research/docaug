"""The data that flows through the pipeline.

Every stage speaks these five types and nothing else, which is what makes the
stages swappable:

    Source   -> Page              (a real document page + its annotated regions)
    TextGen  -> list[str]         (the replacement text, one per region)
    Eraser   -> ErasedPage        (source ink removed, per-region ink colour kept)
    Renderer -> RenderedText      (ink bitmap + the box of every cluster in it)
    Writer   <- SynthPage         (the reconstructed page + its exact labels)

`Cluster` is the unit the labels are made of. One cluster is one Thai
orthographic cluster -- a base consonant plus whatever vowels and tone marks
stack on it -- because that is the smallest thing with a well-defined box. The
boxes are read back out of the rasterizer, so they describe pixels that are
actually on the page rather than pixels we hoped would be.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

Box = tuple[int, int, int, int]
"""(x1, y1, x2, y2) in pixels, top-left origin, x2/y2 exclusive."""

Color = tuple[int, int, int]
"""RGB, 0-255."""


@dataclass(frozen=True, slots=True)
class Region:
    """One annotated text region of a source page."""

    box: Box
    text: str
    category: str = "Text"
    ink_height: float = 0.0
    """Measured height of the source glyphs, in pixels. 0 means unknown, in which
    case the renderer fills the box instead of matching the source type size."""

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


@dataclass(slots=True)
class Page:
    """A source page: real pixels plus the regions we are allowed to rewrite."""

    id: str
    image: Image
    regions: list[Region] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    def evolve(self, **changes) -> Page:
        """Copy with fields replaced -- how page transforms stay non-mutating."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class Cluster:
    """One rendered orthographic cluster and the pixels it occupies."""

    text: str
    box: Box


@dataclass(slots=True)
class RenderedText:
    """Ink produced by a renderer, in its own coordinate frame.

    `image` is RGBA with a transparent background so it composites over the
    erased page without a visible patch. `clusters` are in `image` coordinates;
    the pipeline offsets them into page coordinates when it pastes.
    """

    image: Image
    clusters: list[Cluster]
    size: int
    """The type size actually used, in pixels. Reported for diagnostics."""

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


@dataclass(slots=True)
class ErasedPage:
    """A page with its source text removed."""

    image: Image
    """RGB, same size as the source page."""
    colors: list[Color]
    """The ink colour sampled from each region before erasing, so the replacement
    text inherits the document's own palette instead of a flat black."""


@dataclass(slots=True)
class RegionLabel:
    """The ground truth for one reconstructed region."""

    box: Box
    category: str
    text: str
    """What we rendered -- the OCR label."""
    source_text: str
    """What the source said, kept for provenance and for re-runs."""
    clusters: list[Cluster] = field(default_factory=list)
    size: int = 0
    fit: bool = True
    """False when the text overflowed its region even at the minimum type size.
    Overflowing pages are still labelled correctly, but they are the pages you
    want to drop first when tightening a dataset."""


@dataclass(slots=True)
class SynthPage:
    """A reconstructed page and its labels: one output example."""

    id: str
    image: Image
    labels: list[RegionLabel]
    meta: dict = field(default_factory=dict)
    """JSON-serializable provenance. Keep it that way -- writers serialize it."""
    source: Image | None = None
    """The page this was reconstructed from, kept for previews. Not serialized."""

    @property
    def fit_rate(self) -> float:
        return sum(x.fit for x in self.labels) / len(self.labels) if self.labels else 1.0
