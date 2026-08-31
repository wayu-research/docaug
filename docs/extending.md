# Extending docaug

Every stage is a protocol plus a registry. Implement the protocol, register a
factory, and the CLI can address it by name -- including from another package,
without touching this one.

```
source -> transforms -> text -> erase -> render -> writers
```

## The protocols

| Stage | Registry | Contract |
|---|---|---|
| Source | `docaug.sources.SOURCES` | `__iter__() -> Iterator[Page]` |
| Transform | `docaug.transforms.TRANSFORMS` | `(page, rng) -> Page` |
| Text | `docaug.text.TEXT_GENERATORS` | `(page, rng) -> list[str]`, one per region |
| Eraser | `docaug.erase.ERASERS` | `(page) -> ErasedPage` |
| Renderer | `docaug.render.RENDERERS` | `for_page(page, rng) -> PageRenderer` |
| Writer | `docaug.writers.WRITERS` | `write(SynthPage)`, `close()` |

They are `typing.Protocol`s, so nothing needs to inherit from anything. A plain
function is a valid transform, a `list[Page]` is a valid source.

## In your own project

```python
# my_pkg/renderer.py
from docaug.render import RENDERERS, RenderRequest
from docaug.types import RenderedText


class NeonPageRenderer:
    def __init__(self, rng):
        self._rng = rng

    def render(self, request: RenderRequest) -> RenderedText | None:
        ...  # return None to pass the region to the next renderer in a chain

    @property
    def style(self) -> dict:
        return {"renderer": "neon"}


class NeonRenderer:
    def for_page(self, page, rng):
        return NeonPageRenderer(rng)


@RENDERERS.register("neon")
def build(**kwargs) -> NeonRenderer:
    return NeonRenderer()
```

Declare an entry point and it is discoverable without an import:

```toml
# pyproject.toml
[project.entry-points."docaug.renderers"]
neon = "my_pkg.renderer:build"
```

```bash
docaug list                      # neon is in the renderers row
docaug run --renderer neon ...
```

The groups are `docaug.sources`, `docaug.text_generators`, `docaug.erasers`,
`docaug.renderers`, `docaug.transforms`, `docaug.writers`.

## Two things to get right

**Style is a page property.** A renderer decides its typeface once, in
`for_page`, not per region -- documents do not change face every paragraph. Draw
from the `rng` you are handed and the run stays reproducible from its seed.

**Return `None`, do not guess.** A renderer that cannot draw a region should say
so. `ChainRenderer` then passes the region to the next one, which is how
real-glyph handwriting degrades to a handwriting face for characters no bank
covers.

## Passing options from the CLI

`--source-opt`, `--text-opt` and `--renderer-opt` take `key=value` pairs, JSON
decoded when they parse as JSON, and are passed to your factory. Only the keyword
arguments your factory actually declares are passed, so one set of CLI flags can
drive components with very different signatures.

```bash
docaug run --renderer neon --renderer-opt glow=0.8 --renderer-opt palette='["#f0f","#0ff"]'
```

## Using the library directly

The CLI is a thin wrapper; assembling a pipeline yourself is often clearer:

```python
from docaug import ERASERS, RENDERERS, SOURCES, TEXT_GENERATORS, WRITERS, Pipeline

pipeline = Pipeline(
    source=SOURCES.create("jsonl", path="corpus/"),
    text=TEXT_GENERATORS.create("translate"),
    eraser=ERASERS.create("adaptive"),
    renderer=RENDERERS.create("font"),
    writers=[WRITERS.create("dataset", out="out/")],
)
print(pipeline.run().summary())
```

Iterating a `Pipeline` reconstructs pages lazily and skips the writers, which is
what you want in a notebook:

```python
for page in pipeline:
    display(page.image)
```
