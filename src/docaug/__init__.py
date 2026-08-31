"""docaug -- document reconstruction for synthetic OCR supervision.

Take a real document page, erase its text, and render new text back into the same
regions. The reconstruction inherits the source page's layout, background and
degradation, and the labels are exact by construction: we placed the text, so we
know what it says and read its boxes back out of the rasterizer.

    from docaug import Pipeline, SOURCES, TEXT_GENERATORS, ERASERS, RENDERERS, WRITERS

    pipeline = Pipeline(
        source=SOURCES.create("doclaynet", limit=20),
        text=TEXT_GENERATORS.create("translate"),
        eraser=ERASERS.create("adaptive"),
        renderer=RENDERERS.create("font"),
        writers=[WRITERS.create("dataset", out="out/")],
    )
    report = pipeline.run()

Or from a shell: `docaug run --source doclaynet --limit 20 --out out/`.
"""

from .config import Settings, get_settings
from .erase import ERASERS, Eraser
from .fonts import FontBank
from .pipeline import Pipeline, RunReport
from .registry import Registry
from .render import RENDERERS, Renderer, RenderRequest
from .sources import SOURCES, Source
from .text import TEXT_GENERATORS, TextGenerator
from .transforms import TRANSFORMS, PageTransform
from .types import Cluster, Page, Region, RegionLabel, RenderedText, SynthPage
from .writers import WRITERS, Writer

__version__ = "0.1.0"

__all__ = [
    "Cluster", "ERASERS", "Eraser", "FontBank", "Page", "PageTransform",
    "Pipeline", "RENDERERS", "Region", "RegionLabel", "Registry",
    "RenderRequest", "RenderedText", "Renderer", "RunReport", "SOURCES",
    "Settings", "Source", "SynthPage", "TEXT_GENERATORS", "TRANSFORMS",
    "TextGenerator", "WRITERS", "Writer", "__version__", "get_settings",
]
