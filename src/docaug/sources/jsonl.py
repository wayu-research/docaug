"""The generic source: a folder of images and one JSONL file of annotations.

This is the format to convert your own corpus into. One object per page::

    {"id": "0001",
     "image": "images/0001.png",
     "regions": [{"box": [71, 54, 929, 103], "text": "Annual Report",
                  "category": "Title"}]}

`box` is `[x1, y1, x2, y2]` in pixels on that image. `category` is free-form and
is carried through to the labels unchanged; the DocLayNet vocabulary is a
reasonable default if you have no opinion. `image` is resolved relative to the
JSONL file's directory.

Anything else on the line is kept in the page's metadata, so a converter can pass
provenance through to the output without this module knowing about it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..types import Page, Region
from . import SOURCES
from .regions import prepare

RESERVED = frozenset({"id", "image", "regions"})


@dataclass
class JsonlSource:
    """Reads pages from `annotations.jsonl` next to an images directory."""

    path: Path
    """The JSONL file, or a directory containing `annotations.jsonl`."""
    limit: int | None = None
    skip: int = 0
    group: bool = True
    """Group word boxes into rows and merge wrapped lines. Turn it off when your
    annotations are already block-level and you want them respected exactly."""
    drop_categories: frozenset[str] = field(default_factory=lambda: frozenset({"Picture"}))
    """Categories with no text to reconstruct."""

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.is_dir():
            self.path = self.path / "annotations.jsonl"
        if not self.path.is_file():
            raise FileNotFoundError(f"no annotations at {self.path}; see docs/formats.md")

    def __iter__(self) -> Iterator[Page]:
        emitted = 0
        with self.path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                if index < self.skip:
                    continue
                if self.limit is not None and emitted >= self.limit:
                    return
                yield self._page(json.loads(line), index)
                emitted += 1

    def _page(self, record: dict, index: int) -> Page:
        image = Image.open(self.path.parent / record["image"]).convert("RGB")
        regions = [
            Region(
                box=tuple(int(v) for v in entry["box"]),
                text=(entry.get("text") or "").strip(),
                category=entry.get("category", "Text"),
                ink_height=float(entry.get("ink_height") or 0.0),
            )
            for entry in record.get("regions", [])
            if entry.get("category", "Text") not in self.drop_categories
        ]
        regions = prepare(image, regions, group=self.group)
        return Page(
            id=str(record.get("id", index)),
            image=image,
            regions=regions,
            meta={k: v for k, v in record.items() if k not in RESERVED},
        )


@SOURCES.register("jsonl")
def _jsonl(path: str | Path, limit: int | None = None, skip: int = 0,
           group: bool = True) -> JsonlSource:
    return JsonlSource(path=Path(path), limit=limit, skip=skip, group=group)
