"""A small, cached client for any OpenAI-compatible chat endpoint.

Two decisions here are worth more than the code.

**One request per region.** Batching every region of a page into one indexed
response is cheaper and is wrong: when the model drops or merges an entry, every
later translation slides onto the wrong box, and the result is a page of
confidently mislabelled text that looks fine. Keying the batch by id only moves
the failure -- a neighbour's text merges under a present key and overflows. One
request per region cannot slide, so that is what we do, in parallel.

**Everything is cached on disk, keyed by content.** Rendering is cheap and
deterministic; the text is neither. Caching by `(model, prompt)` means
re-rendering a corpus with different typefaces, or after a bug fix downstream,
costs nothing and produces the same text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from ..config import LLMSettings

log = logging.getLogger(__name__)
T = TypeVar("T")


def cache_key(*parts) -> str:
    """A stable short hash of anything JSON-serializable."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(json.dumps(part, ensure_ascii=False, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:20]


@dataclass
class DiskCache:
    """One JSON file per entry, under `root/namespace/<key>.json`."""

    root: Path

    def get_or_set(self, namespace: str, key: str, compute: Callable[[], T]) -> T:
        path = self.root / namespace / f"{key}.json"
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass  # a truncated write from an interrupted run: just redo it
        value = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return value


class LLMClient:
    """Chat completions with retries, a disk cache, and bounded parallelism."""

    def __init__(self, settings: LLMSettings, cache_dir: Path) -> None:
        self._settings = settings
        self._cache = DiskCache(cache_dir)
        self._client = None

    @property
    def model(self) -> str:
        return self._settings.model

    def _openai(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "LLM text generation needs the llm extra: pip install 'docaug[llm]'"
                ) from exc
            self._client = OpenAI(
                base_url=self._settings.base_url,
                api_key=self._settings.require_key(),
                timeout=self._settings.timeout,
            )
        return self._client

    def complete(self, system: str, user: str, *, namespace: str = "chat") -> str:
        """One cached completion. Returns "" if every attempt fails, so a single
        bad region degrades to an untranslated region instead of a dead run."""
        key = cache_key(self._settings.model, system, user)
        return self._cache.get_or_set(namespace, key, lambda: self._call(system, user))

    def _call(self, system: str, user: str) -> str:
        client = self._openai()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(self._settings.retries):
            try:
                response = client.chat.completions.create(
                    model=self._settings.model, messages=messages, temperature=0.2
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # network, rate limit, provider hiccup
                if attempt == self._settings.retries - 1:
                    log.warning("giving up on a completion after %d tries: %s",
                                self._settings.retries, exc)
                    return ""
                time.sleep(2**attempt)
        return ""

    def map(self, items: Sequence[T], work: Callable[[T], str]) -> list[str]:
        """Run `work` over `items` in parallel, preserving order."""
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=self._settings.concurrency) as pool:
            return list(pool.map(work, items))
