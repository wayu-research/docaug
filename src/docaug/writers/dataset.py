"""Dataset writers.

`dataset` is the archival form: page pixels, region boxes, and the box of every
orthographic cluster, all in page coordinates. It is deliberately more than any
one trainer needs, because re-deriving a cluster box later is impossible and
re-rendering the corpus to get it is expensive.

`pagejson` is one ready-to-train shape: a single JSON object per page with boxes
normalized to [0, 1000] and regions in reading order, which is what page-level
VLM OCR training consumes. If your trainer wants something else, write it from
`dataset` -- that is what it is for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..types import SynthPage
from . import WRITERS


def _as_record(page: SynthPage, image_path: str, clusters: bool) -> dict:
    return {
        "id": page.id,
        "image": image_path,
        "width": page.image.width,
        "height": page.image.height,
        "fit_rate": round(page.fit_rate, 4),
        "meta": page.meta,
        "regions": [
            {
                "box": list(label.box),
                "category": label.category,
                "text": label.text,
                "source_text": label.source_text,
                "size": label.size,
                "fit": label.fit,
                **(
                    {"clusters": [{"text": c.text, "box": list(c.box)} for c in label.clusters]}
                    if clusters
                    else {}
                ),
            }
            for label in page.labels
        ],
    }


@dataclass
class DatasetWriter:
    """`out/images/<id>.<fmt>` plus `out/labels.jsonl`."""

    out: Path
    image_format: str = "png"
    """PNG keeps the reconstruction exact. JPEG matches what a scanner or an
    inference pipeline would actually hand the model -- use it if you want the
    compression artefacts in the training distribution."""
    quality: int = 92
    clusters: bool = True
    """Cluster-level boxes roughly triple the label file. Worth it unless you are
    certain you will only ever train at region level."""

    _handle: object = field(default=None, init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.out = Path(self.out)
        (self.out / "images").mkdir(parents=True, exist_ok=True)
        self._handle = (self.out / "labels.jsonl").open("w", encoding="utf-8")

    def write(self, page: SynthPage) -> None:
        name = f"{page.id}.{self.image_format}"
        path = self.out / "images" / name
        if self.image_format in {"jpg", "jpeg"}:
            page.image.convert("RGB").save(path, quality=self.quality)
        else:
            page.image.save(path)
        record = _as_record(page, f"images/{name}", self.clusters)
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()  # a killed run still leaves a readable dataset
        self._count += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        (self.out / "dataset_info.json").write_text(
            json.dumps({"pages": self._count, "format": "docaug/v1"}, indent=2)
        )


@dataclass
class PageJsonWriter:
    """Page-level VLM training targets, boxes normalized to [0, 1000]."""

    out: Path
    scale: int = 1000

    _handle: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.out = Path(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self._handle = (self.out / "pages.jsonl").open("w", encoding="utf-8")

    def write(self, page: SynthPage) -> None:
        width, height = page.image.size
        blocks = [
            {
                "bbox": [
                    round(label.box[0] * self.scale / width),
                    round(label.box[1] * self.scale / height),
                    round(label.box[2] * self.scale / width),
                    round(label.box[3] * self.scale / height),
                ],
                "category": label.category,
                "text": label.text,
            }
            # Reading order is top-to-bottom, left-to-right. A source that knows
            # better should preserve its own order in the region list; this is
            # the fallback for one that does not.
            for label in sorted(page.labels, key=lambda x: (x.box[1], x.box[0]))
            if label.text.strip()
        ]
        self._handle.write(
            json.dumps({"id": page.id, "layout": blocks}, ensure_ascii=False) + "\n"
        )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@WRITERS.register("dataset")
def _dataset(out: Path | str, image_format: str = "png", clusters: bool = True) -> DatasetWriter:
    return DatasetWriter(out=Path(out), image_format=image_format, clusters=clusters)


@WRITERS.register("pagejson")
def _pagejson(out: Path | str) -> PageJsonWriter:
    return PageJsonWriter(out=Path(out))
