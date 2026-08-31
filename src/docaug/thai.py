"""Thai script taxonomy (Unicode U+0E00-U+0E7F) and locale mixing.

Thai stacks up to four levels inside a single orthographic cluster:

    level +2   tone mark / thanthakhat     ่ ้ ๊ ๋ ์
    level +1   above vowel                 ิ ี ึ ื ั ็ ํ
    level  0   base consonant              ก .. ฮ
    level -1   below vowel                 ุ ู ฺ

A cluster may carry a below vowel *and* an above vowel *and* a tone mark at
once. Handling that is what separates a renderer that produces usable Thai OCR
labels from one that silently drops marks.

The other trap is order. The leading vowels เ แ โ ใ ไ are *drawn* to the left of
their consonant but are *stored* before it in logical order ("เก" is U+0E40
U+0E01). The shaper does the visual reordering; labels must stay in logical
order, which is why they are read back from cluster ids rather than from the
left-to-right sequence of glyphs on the page.
"""

from __future__ import annotations

import re
from random import Random

CONSONANTS = [chr(c) for c in range(0x0E01, 0x0E2F)]
"""ก .. ฮ, including the vowel-like ฤ (U+0E24) and ฦ (U+0E26)."""

LEADING_VOWELS = ["เ", "แ", "โ", "ใ", "ไ"]
FOLLOWING_VOWELS = ["ะ", "า", "ๅ"]
SARA_AM = "ำ"
"""U+0E33. Shapes into two glyphs (nikhahit + sara aa) from one character."""
ABOVE_VOWELS = ["ิ", "ี", "ึ", "ื", "ั", "็", "ํ"]
BELOW_VOWELS = ["ุ", "ู", "ฺ"]
TONE_MARKS = ["่", "้", "๊", "๋"]
THANTHAKHAT = "์"
NIKHAHIT = "ํ"
YAMAKKAN = "๎"
THAI_DIGITS = [chr(c) for c in range(0x0E50, 0x0E5A)]
SYMBOLS = ["ฯ", "ๆ", "฿", "๏", "๚", "๛"]

ABOVE_CHARS = frozenset(ABOVE_VOWELS + TONE_MARKS + [THANTHAKHAT, NIKHAHIT, YAMAKKAN])
BELOW_CHARS = frozenset(BELOW_VOWELS)
COMBINING = ABOVE_CHARS | BELOW_CHARS
"""Zero-advance marks: they stack on a base rather than taking their own slot."""

RARE = frozenset(
    {
        0x0E3A,  # ฺ  phinthu (Pali virama)
        0x0E45,  # ๅ  lakkhangyao
        0x0E4E,  # ๎  yamakkan
        0x0E4F,  # ๏  fongman
        0x0E5A,  # ๚  angkhankhu
        0x0E5B,  # ๛  khomut
        0x0E3F,  # ฿  baht sign
    }
)
"""Archaic, liturgical and editorial marks. A face missing only these still
renders every modern Thai document, so the font bank tolerates their absence and
re-checks coverage per string instead."""


def is_thai(ch: str) -> bool:
    return "฀" <= ch <= "๿"


def band(ch: str) -> str:
    """Which vertical band a character occupies: above, below, or main."""
    if ch in ABOVE_CHARS:
        return "above"
    if ch in BELOW_CHARS:
        return "below"
    return "main"


LEADING_SET = frozenset(LEADING_VOWELS)


def cluster_spans(text: str) -> list[tuple[str, int, int]]:
    """Split `text` into orthographic clusters -> (substring, start, end).

    A cluster is a base character with whatever binds to it: the marks that stack
    on it, and any leading vowel that precedes it in logical order. Grouping this
    way is what makes a stitched line's labels line up with a font-rendered
    line's, since HarfBuzz merges the same characters into the same cluster.
    """
    spans: list[tuple[str, int, int]] = []
    i, n = 0, len(text)
    while i < n:
        start = i
        while i + 1 < n and text[i] in LEADING_SET:
            i += 1
        i += 1  # the base itself
        while i < n and text[i] in COMBINING:
            i += 1
        spans.append((text[start:i], start, i))
    return spans


def required_codepoints() -> list[int]:
    """The glyphs a usable Thai face must draw. Order-preserving, de-duplicated."""
    chars = (
        CONSONANTS
        + LEADING_VOWELS
        + FOLLOWING_VOWELS
        + [SARA_AM]
        + ABOVE_VOWELS
        + BELOW_VOWELS
        + TONE_MARKS
        + [THANTHAKHAT, NIKHAHIT, YAMAKKAN]
        + THAI_DIGITS
        + SYMBOLS
    )
    return list(dict.fromkeys(ord(c) for c in chars))


def stack_depth(cluster: str) -> int:
    """Vertical levels a cluster occupies, 1 (bare consonant) to 4."""
    depth = 1
    depth += any(c in BELOW_CHARS for c in cluster)
    depth += any(c in ABOVE_VOWELS for c in cluster)
    depth += any(c in TONE_MARKS or c == THANTHAKHAT for c in cluster)
    return depth


# --------------------------------------------------------------------------- #
# Locale mixing
# --------------------------------------------------------------------------- #
# Real Thai documents mix Common Era with Buddhist Era, and Arabic digits with
# Thai ones, within the same corpus. An LLM asked to translate will "helpfully"
# convert 2023 to 2566 some of the time and not others, which makes the mix a
# property of the model rather than of the dataset. So the translator is told to
# preserve numbers exactly, and the mix is sampled here instead -- once per page,
# so a page stays internally consistent.

_ARABIC_TO_THAI = str.maketrans({str(i): THAI_DIGITS[i] for i in range(10)})
_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")


def to_thai_digits(text: str) -> str:
    return text.translate(_ARABIC_TO_THAI)


def years_to_be(text: str) -> str:
    """Shift plausible CE years (1900-2099) to the Buddhist Era."""
    return _YEAR.sub(lambda m: str(int(m.group()) + 543), text)


def sample_locale(rng: Random, p_be: float = 0.30, p_thai_digits: float = 0.25) -> dict:
    """One locale choice per page, applied to every region on it."""
    return {"be": rng.random() < p_be, "thai_digits": rng.random() < p_thai_digits}


def apply_locale(text: str, choice: dict) -> str:
    if choice.get("be"):
        text = years_to_be(text)
    if choice.get("thai_digits"):
        text = to_thai_digits(text)
    return text
