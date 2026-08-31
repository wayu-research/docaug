# DocLayNet -> CLI -> output

The end-to-end example. DocLayNet ships per-region text, boxes and layout
categories as ground truth, so there is nothing to detect and nothing to trust
but the dataset's own labels -- which makes it the cleanest way to see what the
pipeline does.

```bash
pip install -e '.[hub]'
./examples/doclaynet/run.sh out 20
```

That writes:

```
out/smoke/          20 pages, source text reused  (no API key needed)
out/thai/           20 pages, translated into Thai (needs DOCAUG_LLM_API_KEY)
out/no-context/     page context removed
out/no-layout/      two-dimensional structure removed
out/one-typeface/   typeface diversity removed
```

Open `out/smoke/previews/*.png` first. Source on the left, reconstruction on the
right, every cluster box drawn. If the boxes sit on the ink, the labels are
right.

## What just happened

For each page, and for each annotated region on it:

1. **Erase.** A local adaptive threshold finds the text pixels -- not "everything
   darker than the page", which would swallow a grey banner whole -- and they are
   either flat-filled or inpainted depending on how busy the background is. The
   colour of the erased text is kept.
2. **Generate.** `--text translate` sends the region to an LLM, one request per
   region. `--text keep` reuses the source text and makes no network calls.
3. **Fit.** The text is shaped with HarfBuzz and the type size reduced until it
   fits the region, starting from the source's measured glyph height so the
   document's type hierarchy survives.
4. **Render.** Into a typeface sampled for the page, in the colour the source
   text was, with its cluster boxes read back out of the rasterizer.

The page keeps its real layout, real background and real degradation. The labels
are exact because we placed the text.

## Your own corpus

`--source doclaynet` is a convenience. The general path is `--source jsonl`,
which reads a folder of images and one JSONL file of annotations -- see
[docs/formats.md](../../docs/formats.md). Any annotated document corpus converts
to it in a few lines:

```python
import json
from pathlib import Path

with open("corpus/annotations.jsonl", "w", encoding="utf-8") as out:
    for page in my_corpus:
        out.write(json.dumps({
            "id": page.id,
            "image": f"images/{page.id}.png",
            "regions": [
                {"box": [r.x1, r.y1, r.x2, r.y2], "text": r.text, "category": r.label}
                for r in page.regions
            ],
        }, ensure_ascii=False) + "\n")
```

```bash
docaug run --source jsonl --input corpus/ --text translate --out out/
```
