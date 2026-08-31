"""Sources: where the annotated pages come from.

A source yields `Page` objects -- a real page image plus the regions we are
allowed to rewrite. Annotations are *required*: this pipeline does not detect
text, it reconstructs pages whose text is already labelled. That is the whole
reason the output labels are exact rather than pseudo-labels with a detector's
errors baked into them.

Built-in sources:

===============  ======================================================
``jsonl``        A folder of images plus `annotations.jsonl`. See docs/formats.md.
``doclaynet``    DocLayNet from the Hugging Face Hub (needs `docaug[hub]`).
===============  ======================================================

Add your own with `@SOURCES.register("name")`; anything iterable of `Page` works.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from ..registry import Registry
from ..types import Page


@runtime_checkable
class Source(Protocol):
    """An iterable of annotated source pages."""

    def __iter__(self) -> Iterator[Page]: ...


SOURCES: Registry[Source] = Registry("source", entry_point_group="docaug.sources")

from . import doclaynet, jsonl  # noqa: E402,F401  (registers the built-ins)

__all__ = ["SOURCES", "Source"]
