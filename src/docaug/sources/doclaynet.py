"""DocLayNet, straight from the Hugging Face Hub.

DocLayNet ships per-region text, boxes and layout categories as ground truth,
which makes it the cheapest way to see the pipeline work end to end -- no
detection, no OCR, nothing to trust but the dataset's own labels.

Its granularity is inconsistent: some pages are labelled line by line and others
word by word. `sources.regions.normalize` handles both, which is why this module
is barely longer than the loader call.

Needs `pip install docaug[hub]`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from ..types import Page, Region
from . import SOURCES
from .regions import prepare

DEFAULT_DATASET = "pierreguillou/DocLayNet-small"

PARQUET_REVISION = "refs/convert/parquet"
"""DocLayNet is published behind a loading script, which recent `datasets`
refuses to run. Every Hub dataset also has an auto-converted Parquet branch, and
it holds the same rows, so that is what gets loaded -- with a fall back to the
plain call for anyone on an older `datasets` or a private mirror."""

DROP_CATEGORIES = frozenset({"Picture", "Formula"})
"""Picture has no text. Formula text is not meaningfully translatable, and
rendering Thai into a formula box would teach the model something false."""


@dataclass
class DocLayNetSource:
    """Streams annotated English document pages."""

    limit: int = 10
    skip: int = 0
    split: str = "train"
    dataset: str = DEFAULT_DATASET
    drop_categories: frozenset[str] = field(default_factory=lambda: DROP_CATEGORIES)

    def __iter__(self) -> Iterator[Page]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the doclaynet source needs the Hub extra: pip install 'docaug[hub]'"
            ) from exc

        try:
            data = load_dataset(self.dataset, split=self.split, revision=PARQUET_REVISION)
        except Exception:
            data = load_dataset(self.dataset, split=self.split)
        names = data.features["categories"].feature.names
        for index in range(self.skip, min(self.skip + self.limit, len(data))):
            example = data[index]
            image = example["image"].convert("RGB")
            regions = [
                Region(box=_to_xyxy(box, image.size), text=text.strip(), category=names[category])
                for text, box, category in zip(
                    example["texts"], example["bboxes_line"], example["categories"],
                    strict=True,
                )
                if text and text.strip() and names[category] not in self.drop_categories
            ]
            regions = prepare(image, regions)
            yield Page(
                id=str(example.get("page_hash", index))[:16],
                image=image,
                regions=regions,
                meta={
                    "source": self.dataset,
                    "doc_category": example.get("doc_category"),
                    "page_no": example.get("page_no"),
                },
            )


def _to_xyxy(box, size) -> tuple[int, int, int, int]:
    """DocLayNet boxes are COCO `[x, y, w, h]`."""
    x, y, w, h = box
    width, height = size
    return max(0, int(x)), max(0, int(y)), min(width, int(x + w)), min(height, int(y + h))


@SOURCES.register("doclaynet")
def _doclaynet(limit: int = 10, skip: int = 0, split: str = "train",
               dataset: str = DEFAULT_DATASET) -> DocLayNetSource:
    return DocLayNetSource(limit=limit, skip=skip, split=split, dataset=dataset)
