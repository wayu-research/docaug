# Data formats

## Input

The pipeline reconstructs pages whose text is **already annotated**. It does not
detect and does not OCR. That is the point: the output labels are exact because
we place the text ourselves, and a detector in front would smuggle its errors
into the ground truth.

### `jsonl` -- the format to convert your corpus into

```
corpus/
  annotations.jsonl
  images/
    0001.png
    0002.png
```

One JSON object per line:

```json
{
  "id": "0001",
  "image": "images/0001.png",
  "regions": [
    {"box": [71, 54, 929, 103], "category": "Page-header", "text": "Annual Report 2024"},
    {"box": [71, 142, 929, 388], "category": "Text", "text": "The board resolved ..."}
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | no | Output filename stem. Defaults to the line number. |
| `image` | yes | Path to the page, relative to the JSONL file. |
| `regions[].box` | yes | `[x1, y1, x2, y2]` in pixels on that image. |
| `regions[].text` | yes | The source text. Empty regions are skipped. |
| `regions[].category` | no | Free-form, carried through to the labels. Defaults to `Text`. |
| `regions[].ink_height` | no | Source glyph height in pixels. Measured from the image when absent. |

Any other top-level key is carried into the output page's `meta`, so a converter
can pass provenance through without this format knowing about it.

**Annotation granularity does not need to be consistent.** Word boxes are grouped
into rows and wrapped lines are merged back into blocks, because rendering each
wrapped line at its own independently-fitted size produces something no document
looks like. Block-level input passes through both steps unchanged. Turn it off
with `--source-opt group=false` if your annotations are exactly what you want.

### `doclaynet`

```bash
docaug run --source doclaynet --limit 100 --out out/
```

Streams DocLayNet from the Hub (`pip install 'docaug[hub]'`). Needs no
preparation, which makes it the fastest way to see the pipeline work.

## Output

```
out/
  images/<id>.png
  labels.jsonl        one object per page -- the archival form
  pages.jsonl         page-level training targets, with --pagejson
  previews/<id>.png   source | reconstruction, with --previews N
  dataset_info.json
  report.json         what the run did, and what it was configured with
```

### `labels.jsonl`

```json
{
  "id": "0001",
  "image": "images/0001.png",
  "width": 1025, "height": 1025,
  "fit_rate": 1.0,
  "meta": {"style": {"font": "Sarabun", "category": "sans", "warp": 0.0}, "index": 0},
  "regions": [
    {
      "box": [71, 54, 929, 103],
      "category": "Page-header",
      "text": "รายงานประจำปี ๒๕๖๗",
      "source_text": "Annual Report 2024",
      "size": 28,
      "fit": true,
      "clusters": [
        {"text": "ร", "box": [74, 60, 92, 88]},
        {"text": "า", "box": [92, 60, 105, 88]}
      ]
    }
  ]
}
```

Three fields deserve a note.

**`clusters`** are Thai orthographic clusters -- a base consonant with the
vowels and tone marks that stack on it -- and their boxes come from the
rasterizer, not from arithmetic. Concatenating `text` over a region's clusters
gives back the region's `text` exactly, in logical order. Marks are inside their
base's box, not beside it. Drop them with `--no-clusters` if you are certain you
will only train at region level; you cannot recover them later without
re-rendering.

**`fit`** is false when the text overflowed its region even at the minimum type
size. The region is still labelled correctly, but the ink it did not have room
for is drawn anyway, past the region's edge and over whatever the page had
next to it -- so a neighbour's labels can end up describing pixels that are no
longer there. `--min-fit-rate 1.0` rejects any page with such a region at write
time and lists it under `rejected` in `report.json`; the default of 0.0 keeps
everything and leaves the decision to you, with `fit_rate` telling you how much
a stricter pass would cost.

**`meta.style`** records the typeface decisions for the page, so you can slice a
corpus by typeface after the fact, or reproduce one page exactly.

### `pages.jsonl` (`--pagejson`)

One object per page, boxes normalized to `[0, 1000]` on both axes, regions in
reading order -- the shape page-level VLM OCR training consumes:

```json
{"id": "0001", "layout": [{"bbox": [69, 52, 906, 100], "category": "Page-header", "text": "..."}]}
```

If your trainer wants something else, generate it from `labels.jsonl`. That file
is the archival form and holds strictly more.
