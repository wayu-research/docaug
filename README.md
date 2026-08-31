# docaug

**Document reconstruction for synthetic OCR supervision.** Take a real document
page, erase its text, and render new text back into the same regions. The
reconstruction inherits the source page's layout, background and degradation --
and the labels are exact, because we placed the text.

```bash
pip install -e '.[all]'
docaug run --source doclaynet --limit 20 --text translate --out out/
```

## The idea

Synthetic OCR data has an asymmetry worth exploiting. Labelling a document is
unreliable; *placing* text into one is exact. So instead of reading a page and
hoping the transcription is right, we take a page whose regions are already
annotated, remove the source text, and draw our own text into the same boxes. The
label is not a prediction -- it is what we asked the rasterizer to draw, with
boxes read back out of it.

Both obvious alternatives give up one half of that:

- **Crawl and OCR.** Real pages, unreliable labels. Thai in particular: extractors
  and OCR engines scramble the order of combining characters and drop tone marks
  off four-level stacks, so the errors land exactly on the hard cases.
- **Render from templates.** Exact labels, but a page that looks like a template.
  No scan noise, no stamps, no figures, no columns that a real document has.

Reconstruction keeps the real page *and* the exact labels. What it costs you is
that the source pages must already be annotated -- this pipeline does not detect
and does not OCR, on purpose.

## What comes out

```
out/
  images/<id>.png
  labels.jsonl        regions, text, and a box for every orthographic cluster
  previews/<id>.png   source | reconstruction, for eyeballing
  report.json
```

Thai is not the length of the source it replaces, so some regions do not fit even
at the smallest acceptable type size. Those are labelled `fit: false`, and the
ink that did not fit is drawn anyway, over whatever was beside it. Pass
`--min-fit-rate 1.0` to reject a page the moment one of its regions overflows.

Cluster boxes are the reason to use this rather than a region-level tool. One
cluster is a base consonant plus the vowels and tone marks stacked on it, and its
box comes from the pixels that were actually drawn:

```json
{"text": "รายงานประจำปี ๒๕๖๗", "box": [71, 54, 929, 103],
 "clusters": [{"text": "รา", "box": [74, 60, 105, 88]},
              {"text": "ย",  "box": [105, 60, 121, 88]}]}
```

Concatenating a region's cluster texts gives back its text exactly, in logical
order -- with the leading vowels เ แ โ ใ ไ back where Unicode puts them, not
where they are drawn. See [docs/formats.md](docs/formats.md).

## Install

```bash
pip install -e .                # core: shaping, erasing, rendering
pip install -e '.[all]'         # + Thai word breaking, LLM text, Hub datasets
```

| Extra | Brings |
|---|---|
| `thai` | `pythainlp`, for wrapping at Thai word boundaries |
| `llm` | `openai`, for `--text translate` and `--text synth` |
| `hub` | `datasets`, for `--source doclaynet` |

Configuration is entirely through `.env`; copy `.env.example` and edit. Nothing
in it is required for a `--text keep` run.

## Quickstart

```bash
# No API key: reuses the source text. Exercises everything but translation.
docaug run --source doclaynet --limit 10 --text keep --out out/ --previews 5 --boxes

# The real thing: needs DOCAUG_LLM_API_KEY in .env.
docaug run --source doclaynet --limit 100 --text translate --out out/ --pagejson

# Your own annotated corpus (see docs/formats.md for the JSONL shape).
docaug run --source jsonl --input corpus/ --text translate --out out/
```

Full walkthrough: [examples/doclaynet](examples/doclaynet).

## The controls

"Realism" is several separable properties, and you may want to vary them one at
a time. Each flag below removes exactly one and changes nothing else -- same
text, same boxes, same reading order, same typefaces -- so a pair of corpora
differs in one variable and is a clean A/B.

| Property | How to remove it |
|---|---|
| Page context (figures, texture, degradation) | `--transform white-background` |
| Two-dimensional structure | `--transform stack` |
| Typeface diversity | `--font Sarabun` (any family from `docaug fonts`) |
| Source-domain match | `--text keep` vs `--text translate` |
| Real handwriting glyphs | `--renderer font` vs `--renderer glyph` |

## How it fits together

```
source -> transforms -> text -> erase -> render -> writers
```

Every stage is a `typing.Protocol` plus a registry, so every stage is
replaceable:

```bash
docaug list
# sources           doclaynet, jsonl
# text generators   keep, synth, translate
# erasers           adaptive, none
# renderers         chain, font, glyph
# transforms        keep, stack, white-background
# writers           dataset, pagejson, preview
```

Registration is a decorator, and an entry point makes your component
discoverable from another package without an import here:

```python
from docaug.render import RENDERERS

@RENDERERS.register("neon")
def build(**kwargs):
    return NeonRenderer()
```

```toml
[project.entry-points."docaug.renderers"]
neon = "my_pkg.renderer:build"
```

Nothing needs to inherit from anything -- a plain function is a valid transform,
a `list[Page]` is a valid source. See [docs/extending.md](docs/extending.md).

The library is usable directly, and often clearer than the CLI:

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

## Fonts

25 open-licensed Thai faces across five categories ship inside the package --
SIL OFL and GPL-with-font-exception only, listed with their provenance in
[src/docaug/assets/fonts/LICENSES/SOURCES.md](src/docaug/assets/fonts/LICENSES/SOURCES.md).

```bash
docaug fonts          # what is in the bank, and every family name
```

Sampling is design-balanced, not face-balanced: a category, then a family, then a
face. Some families ship eighteen weights and some ship one, and drawing faces
uniformly would produce a corpus that is mostly one superfamily's grades.
Coverage is checked when the bank is scanned and again per string at render time,
so an incomplete face can never put a tofu box into the ground truth.

To widen the typeface distribution, lay your own collection out by category and
point the bank at it -- no need to write into the installed package:

```bash
mkdir -p myfonts/{sans,serif,handwriting,display,monospace}
cp /path/to/more/*.ttf myfonts/handwriting/
docaug fonts scan --dir myfonts     # measures Thai coverage, writes manifest.json
export DOCAUG_FONTS_DIR=myfonts     # or set it in .env
```

## Handwriting

Two routes, and the second is better.

**A handwriting typeface plus a per-instance warp** works out of the box. The
warp gives each cluster its own small affine jitter and the line a smooth elastic
wobble, rendered at 3x and area-downsampled so the ink stays crisp instead of
turning into a bilinear smear. The same warp is applied to a label map and every
box is re-derived from where its pixels landed, so the labels stay exact.

**Real glyph instances**, stitched from a bank of ink that people actually wrote,
remove the regularity at its source: a font draws every `ก` identically, and
sampling each occurrence independently does not.

```bash
docaug run --source jsonl --input corpus/ --out out/ \
  --renderer glyph --glyph-bank /path/to/bank
```

**No glyph bank ships with this repository.** A bank is thousands of crops of ink
cut out of handwriting corpora, and those corpora's licences are not ours to pass
on. [docs/glyph-bank.md](docs/glyph-bank.md) gives the on-disk format and three
routes to building one, from an afternoon of manual cropping to the full
detect / align / verify pipeline. Characters a bank lacks fall back to a
typeface, so a partial bank is useful immediately.

## Notes on the tricky parts

Most of this is arithmetic on boxes. Four things are not, and each is commented
where it lives:

- **`shaping.py`** -- HarfBuzz for order, FreeType for pixels, and cluster boxes
  accumulated during the blit rather than computed from advances. This is where
  the labels come from.
- **`erase/adaptive.py`** -- a *local* threshold, because a global one swallows
  any uniformly-coloured region darker than the page and leaves a flat patch
  where a grey banner used to be.
- **`render/handwriting.py`** -- warping at 1x costs about a third of the edge
  acutance; supersampling and area-downsampling is what buys it back.
- **`text/llm.py`** -- one request per region, never a batch. A batched, indexed
  response slides every later translation onto the wrong box the moment the model
  drops an entry, and the result looks completely fine.
- **`text/providers.py`** -- a region with no words in it is never sent. The
  prompt carries page context so the model can pick a register, and asked to
  translate a table cell holding `-` the model translates that context instead:
  a paragraph of Thai, in a cell four pixels wide, labelled as if it belonged
  there.

## Relation to the paper

docaug is the open implementation of the document-reconstruction pipeline behind
*How Far Can Synthetic Data Take Thai OCR?* and its model,
[Wayu-Paxa-OCR-Zero](https://huggingface.co/wayu-ai/wayu-paxa-ocr-zero)
(inference code: [wayu-research/wayu-ocr-inference](https://github.com/wayu-research/wayu-ocr-inference)).
The design is the paper's -- reconstruct annotated pages in place, fit-constrained
HarfBuzz rendering, cluster-level labels, one property removed at a time -- and the
controls the paper varies map onto the flags in *The controls* above. The paper's
corpus itself, though, was produced with an internal pipeline and assets that this
repository does not ship:

- **The typeface distribution.** The paper samples faces from a character-weighted
  profile measured on 8,000 public Thai PDF pages (693 families; the ten most
  common cover 80.9% of characters). Most of those families are proprietary and
  cannot be bundled. This repository ships 22 open families and samples them
  design-balanced; point `DOCAUG_FONTS_DIR` at your own bank to get closer.
- **The handwriting glyph bank.** The paper's bank holds 5,953 real glyph
  instances over 76 character classes, cut from the iApp Thai Handwriting Dataset
  and from public notebook pages. No bank ships here -- see
  [docs/glyph-bank.md](docs/glyph-bank.md) for the format and how to build one.
- **Two settings.** The paper rejects any page whose label overflows its region at
  the minimum type size, which is `--min-fit-rate 1.0` here (the default keeps the
  page and marks the region `fit: false`). Its handwriting-typeface variant puts a
  handwriting face on 24.0% of pages, half of them with the per-instance warp; the
  defaults here are `DOCAUG_HANDWRITING_PROB=0.10` and `DOCAUG_AUGMENT_PROB=0.50`.

The training pages are therefore not byte-reproducible from this code. The method
is, and the same page can be re-rendered under every control the paper reports.

## Development

```bash
pip install -e '.[dev,all]'
pytest -q          # 73 tests, no network
ruff check src tests
```

The tests that matter assert that a rendered region's cluster labels reassemble
its text exactly, and that every cluster box lands inside the region it belongs
to. If those hold, the dataset is telling the truth.

## Citation

```bibtex
@article{pipatanakul2026wayupaxaocr,
  title  = {How Far Can Synthetic Data Take Thai OCR?},
  author = {Pipatanakul, Kunat},
  year   = {2026}
}
```

## Licence

Apache-2.0. Bundled fonts keep their own licences -- see
[src/docaug/assets/fonts/LICENSES](src/docaug/assets/fonts/LICENSES).
