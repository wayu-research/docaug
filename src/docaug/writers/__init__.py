"""Writers: turn reconstructed pages into a dataset on disk.

A run can have several writers, and usually does -- the dataset itself plus a
handful of side-by-side previews you can actually look at. Writers are context
managers, so a summary written on `close` is written even if the run is
interrupted.

===============  ======================================================
``dataset``      Images plus `labels.jsonl`: every region, every cluster
                 box, in page pixels. The lossless form -- convert from
                 this rather than re-rendering.
``pagejson``     One JSON object per page with boxes normalized to
                 [0, 1000], ready for VLM page-level training.
``preview``      Source and reconstruction side by side, for eyeballing.
                 Capped, because you do not need a thousand of them.
===============  ======================================================
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..registry import Registry
from ..types import SynthPage


@runtime_checkable
class Writer(Protocol):
    """Consumes reconstructed pages."""

    def write(self, page: SynthPage) -> None: ...

    def close(self) -> None: ...


WRITERS: Registry[Writer] = Registry("writer", entry_point_group="docaug.writers")

from . import dataset, preview  # noqa: E402,F401  (registers the built-ins)

__all__ = ["WRITERS", "Writer"]
