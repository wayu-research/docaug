# Building a handwriting glyph bank

Real-glyph rendering is the one part of this pipeline the repository cannot give
you.

**No glyph bank ships here.** A bank is thousands of small crops of ink that
people wrote, cut out of handwriting corpora. Those corpora come with their own
licences, and redistributing pieces of them is not ours to do. The renderer, the
format, and this guide are here; the ink is yours to collect.

Everything else works without one. `--renderer font` includes handwriting faces
and a per-instance warp, which is a reasonable approximation. This document is
about closing the rest of the gap.

## Why a bank beats a handwriting font

A font draws every `ก` from the same outline. A page of it has a regularity that
no handwritten page has, and a model trained on it learns the font rather than
the language. Sampling each character occurrence independently from a pool of
real instances removes that regularity at its source: the same character appears
three times on a line in three different hands.

## The format

A bank is a directory of PNGs, one directory per character:

```
bank/
  ก/0000.png
  ก/0001.png
  ข/0000.png
  ◌่/0000.png      <- a combining mark, prefixed with U+25CC DOTTED CIRCLE
  ◌ุ/0000.png
```

Rules, all of them load-bearing:

* **One directory per character**, named with the character itself. Combining
  marks get a leading `◌` (U+25CC), the Unicode convention for showing a mark in
  isolation -- without it the directory name is an invisible standalone mark,
  which is miserable to work with in a shell or a file browser.
* **Tight crops.** Each PNG is the glyph's ink and nothing else, cropped to its
  bounding box. Padding is read as part of the glyph and throws off the scale.
* **Dark ink on white**, 8-bit grayscale. Stored that way so the bank is
  browsable; it is inverted to an ink mask on load.
* **Consistent scale within a directory is not required.** Instances are rescaled
  at render time -- main-band glyphs to the line's x-height, marks to a fraction
  of it (`MARK_FRACTION`, 0.45) -- because a tight crop carries no scale of its
  own.

Check what you built:

```bash
docaug glyphs --dir /path/to/bank
```

## What to put in it

Aim for coverage first and volume second:

* **Every character class you expect to render.** Thai has 44 consonants plus
  vowels, tone marks and digits; a bank that covers the consonants and the common
  marks already carries most of the ink on a page.
* **A few dozen instances per class is enough.** Past that you are mostly adding
  near-duplicates from the same few hands.
* **Writers matter more than instances.** A bank of 500 instances from twenty
  writers beats 5,000 from three, so sample round-robin across writers rather than
  taking whatever the largest contributor produced.

Characters the bank lacks fall back to a typeface, so a partial bank is useful
immediately. Consonants first: they are most of the ink on a page. Marks and Thai
digits next -- they are small and easy to get wrong, and a bank without them
renders visibly heavier marks than consonants.

## Getting the instances

Three routes, in increasing order of effort and quality.

### 1. Cut them by hand

For a first bank, this is faster than it sounds. Write out a charset sheet,
scan it, and crop. A hundred instances across the twenty most common consonants
is an afternoon and is enough to see whether real glyphs help on your data.

### 2. Segment line images you already have

If you have handwriting line images with transcriptions, connected components
plus the transcription gets you most of the way:

1. Binarize each line crop (Otsu or Sauvola).
2. Take connected components as candidate glyphs, and note each one's vertical
   band -- above, main, or below the x-height -- from the line's horizontal
   projection profile.
3. Align the main-band candidates left to right against the transcription's
   characters, monotonically. Marks attach to the nearest main-band candidate.
4. Keep the alignments where the counts agree exactly, and discard the whole line
   otherwise. Being ruthless here is cheap; a wrongly-labelled instance
   contaminates every page it is later drawn into.

Touching consonants merge into one component, and that is the main source of
error. Dropping any component wider than about 1.5 x-heights removes most of it.

### 3. Detect, align, verify

Higher yield, and considerably more machinery. The shape of it:

1. **Detection.** Train a class-agnostic glyph detector and run it over line
   crops. Keep a tight ink mask, the vertical band, and the line's text height for
   each candidate. This stage should be deliberately over-eager -- you are going
   to throw most of it away.
2. **Alignment and filtering.** Align candidates monotonically to the
   transcription, accounting for bands. Keep an identity only when a second
   signal agrees -- a glyph classifier, or an OCR engine reading the same
   position. Rare classes need the stricter signal, because a classifier trained
   on a long-tailed charset is unreliable exactly there. Size and per-class shape
   filters remove the fragments.
3. **Verification.** Show each surviving crop to a model that can read it, and
   keep it only when the character it reads matches the one alignment assigned.
4. **Assembly.** Group by character, cap the count per class, and sample
   round-robin across writers.

Expect the funnel to be steep, and want it to be. Precision at the instance level
is what the bank is for; recall you can buy with more input lines.

## Corpora to start from

* **iApp Thai Handwriting Dataset** -- `iapp/thai_handwriting_dataset` on the
  Hugging Face Hub. Line images with transcriptions. Check its licence before
  redistributing anything derived from it.
* **Your own scans.** A dozen writers copying a charset sheet gives a bank with
  more writer diversity than most public corpora, and no licence question at all.

Whatever you use, **exclude your evaluation writers**. A bank harvested from the
pages you evaluate on puts those writers' strokes into the training set, and the
resulting number is not measuring what you think it is.

## Using it

```bash
docaug run --source jsonl --input corpus/ --out out/ \
  --renderer glyph --glyph-bank /path/to/bank
```

Or set `DOCAUG_GLYPH_BANK_DIR` in `.env` and drop the flag.

The glyph renderer is chained behind a typeface renderer by default, so a region
the bank cannot cover is rendered with a handwriting face rather than dropped.
`--no-fallback` turns that off if you would rather see the gaps. `min_coverage`
(default 0.5) is the fraction of Thai characters a region needs before the
stitcher will claim it:

```bash
docaug run ... --renderer glyph --renderer-opt min_coverage=0.8
```
