#!/usr/bin/env bash
# DocLayNet -> docaug -> a Thai OCR training set.
#
# The shortest real example: an annotated English document corpus in, a Thai
# document corpus with exact labels out. Nothing to prepare, nothing to detect.
set -euo pipefail

OUT="${1:-out}"
LIMIT="${2:-20}"

# ---------------------------------------------------------------------------
# 1. Smoke test -- no API key. The source text is reused as-is, so the pages
#    come out English set in Thai typefaces. Useless as training data, but it
#    exercises erasing, fitting, rendering and labelling end to end.
# ---------------------------------------------------------------------------
docaug run \
  --source doclaynet --limit "$LIMIT" \
  --text keep \
  --out "$OUT/smoke" \
  --previews 5 --boxes

# ---------------------------------------------------------------------------
# 2. The real thing -- each region translated into Thai and rendered back into
#    its own box. Needs DOCAUG_LLM_API_KEY in .env; skipped without one so the
#    rest of the script still runs.
# ---------------------------------------------------------------------------
TEXT=translate
if ! grep -qs '^DOCAUG_LLM_API_KEY=.' .env && [ -z "${DOCAUG_LLM_API_KEY:-}" ]; then
  echo ">> DOCAUG_LLM_API_KEY is not set -- steps 2 and 3 will reuse the source"
  echo "   text instead of translating. Set it in .env for real Thai pages."
  TEXT=keep
fi

docaug run \
  --source doclaynet --limit "$LIMIT" \
  --text "$TEXT" \
  --out "$OUT/thai" \
  --pagejson --previews 5

# ---------------------------------------------------------------------------
# 3. One property at a time. Each run removes exactly one and changes nothing
#    else -- same text, same boxes, same reading order, same typefaces -- so any
#    two of these outputs differ in a single variable.
# ---------------------------------------------------------------------------
docaug run --source doclaynet --limit "$LIMIT" --text "$TEXT" \
  --transform white-background --out "$OUT/no-context"   # remove page context

docaug run --source doclaynet --limit "$LIMIT" --text "$TEXT" \
  --transform stack --out "$OUT/no-layout"               # remove 2D structure

docaug run --source doclaynet --limit "$LIMIT" --text "$TEXT" \
  --font Sarabun \
  --out "$OUT/one-typeface"                              # remove typeface diversity

echo "done -> $OUT"
