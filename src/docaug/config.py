"""Settings, read from the environment and `.env`.

Everything configurable lives here, in one object, with a `DOCAUG_` prefix. Copy
`.env.example` to `.env` in your working directory and edit it; nothing else
reads `os.environ` directly. `.env` is resolved relative to where you run
`docaug`, not to where the package is installed.

CLI flags win over `.env`, which wins over the defaults below. That ordering is
the only reason a flag exists at all -- every flag has a usable default, so the
shortest working invocation is `docaug run --source ... --out ...`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
BUNDLED_FONTS = PACKAGE_ROOT / "assets" / "fonts"
"""The open-licensed Thai faces shipped with the package. Inside the package
rather than beside it, so an installed wheel has them too."""


class LLMSettings(BaseSettings):
    """Any OpenAI-compatible chat endpoint: OpenRouter, vLLM, Ollama, ...

    Only the LLM-backed text generators (`translate`, `synth`) touch this; the
    `keep` generator makes no network calls, so a run that reuses the source
    text needs no key at all.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCAUG_LLM_", env_file=".env", extra="ignore"
    )

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    model: str = "google/gemini-2.5-flash"
    timeout: float = 120.0
    retries: int = 4
    concurrency: int = 8
    """Regions translated in parallel per page. One request per region: batching
    them into a single indexed response is what makes translations slide onto the
    wrong box when the model drops or merges an entry."""

    def require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "DOCAUG_LLM_API_KEY is empty. Set it in .env, or use "
                "`--text keep` to reuse the source text without an LLM."
            )
        return self.api_key


class Settings(BaseSettings):
    """Top-level configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DOCAUG_", env_file=".env", extra="ignore"
    )

    # --- paths ---------------------------------------------------------- #
    fonts_dir: Path = BUNDLED_FONTS
    """Directory scanned for `.ttf`/`.otf` faces. Defaults to the bundled
    open-licensed bank; point it at your own to widen typeface diversity."""
    glyph_bank_dir: Path | None = None
    """Real handwriting glyph bank. No bank ships with this repo -- see
    docs/glyph-bank.md for the on-disk format and how to build one."""
    cache_dir: Path = Path(".docaug-cache")
    """Disk cache for LLM responses, keyed by content. Re-rendering a corpus with
    different fonts costs nothing because the text is already cached."""

    # --- determinism ---------------------------------------------------- #
    seed: int = 20260626
    """Base seed. Each page derives its own RNG from it and its index, so pages
    render identically whether the run is sequential or parallel."""

    # --- ingestion ------------------------------------------------------ #
    max_side: int = 2600
    """Longest image side, in pixels. Larger pages are downscaled (regions
    scale with them) to bound render cost."""

    # --- typeface sampling ---------------------------------------------- #
    handwriting_prob: float = 0.10
    """P(a page uses a handwriting face) when rendering with the font bank."""
    handwriting_max_regions: int = 25
    """Handwriting only on sparse pages: a dense page set in a script face is
    not a document anyone would produce."""
    bold_prob: float = 0.10
    italic_prob: float = 0.01
    slant: float = 0.15
    """Shear applied when a page draws the (rare) oblique style."""

    # --- handwriting-style warp ----------------------------------------- #
    augment_prob: float = 0.50
    """P(per-instance warp | handwriting face). Static faces draw every 'ก'
    identically; real handwriting does not."""
    augment_min: float = 0.5
    augment_max: float = 1.1
    supersample: int = 3
    """Warping at 1x blurs the ink. Render at Nx, warp, then area-downsample --
    the downsample is the anti-aliasing."""

    # --- rendering ------------------------------------------------------ #
    leading: float = 1.35
    """Line spacing as a multiple of the type size. Thai needs more than Latin so
    a tone mark clears the below-vowel of the line above."""
    size_scale: float = 0.95
    """Rendered size relative to the measured source ink height. Slightly under
    1.0 because a Thai body looks heavier than Latin at the same pixel size."""
    pad_fraction: float = 0.10
    """Margin reserved inside each region so text never bleeds past a cell edge."""
    min_size: int = 6
    """Type size floor. Text still overflowing here marks the region `fit=False`."""

    llm: LLMSettings = Field(default_factory=LLMSettings)

    def rng_seed(self, index: int) -> int:
        """Per-page seed. The stride is prime so nearby pages do not correlate."""
        return self.seed + index * 7919


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings, loaded once."""
    return Settings()
