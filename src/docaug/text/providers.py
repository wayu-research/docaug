"""The built-in text generators.

The translation prompt is worth reading before you change it. It insists on
preserving numbers exactly, which sounds like a limitation and is the opposite:
an LLM left to itself converts *some* years to the Buddhist Era and *some*
Arabic digits to Thai ones, so the locale mix of the dataset becomes a property
of the model rather than a thing you chose. Pin the model to literal numbers and
sample the mix afterwards, in `thai.sample_locale`, once per page.

The prompt also asks for length parity. Thai is not the same length as its
English source, and a region that comes back three times longer either shrinks to
an unreadable size or overflows its box.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass

from .. import thai
from ..config import Settings, get_settings
from ..types import Page
from . import TEXT_GENERATORS
from .llm import LLMClient

log = logging.getLogger(__name__)

TRANSLATE_SYSTEM = """\
You are translating text from a document into Thai to build an OCR training set.
Translate the text you are given into natural, fluent Thai.

RULES
1. Keep the Thai roughly the same visual length as the source. Do not pad, \
explain, annotate, or add anything the source does not say.
2. Preserve numbers, dates, years and digits EXACTLY as written. Do not convert \
years to the Buddhist era. Do not convert Arabic numerals to Thai numerals. Do \
not reformat amounts.
3. Leave proper nouns, codes, units, URLs and identifiers as they are when they \
would not normally be translated.
4. Output only the Thai translation: no quotes, no labels, no commentary, no \
romanization, no source text.\
"""

SYNTH_SYSTEM = """\
You write plausible Thai text for document OCR training data.

Given a document region's category and approximate length, write Thai that would \
naturally appear in that position of a real Thai document.

RULES
1. Match the requested length as closely as you can.
2. Use ordinary Thai orthography, including tone marks and vowels.
3. Output only the Thai text: no quotes, no labels, no commentary.\
"""

_REFUSAL = re.compile(
    r"(please\s+(provide|specify|supply)\s+(me\s+)?(with\s+)?(the\s+)?(text|content)"
    r"|there\s+is\s+no\s+text|i\s+need\s+the\s+text|no\s+text\s+(was\s+)?provided"
    r"|กรุณาระบุข้อความ|โปรดระบุข้อความ|ไม่มีข้อความ)",
    re.I,
)

_MOJIBAKE = set("ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßµ")
_WORD_RUN = re.compile(r"[A-Za-z]{3,}|[฀-๿]{3,}")
_WORDLIKE = re.compile(r"[A-Za-z฀-๿]{2,}")

LENGTH_TOLERANCE = 3
"""How many times its source a translation may run before it is not one."""


def looks_like_garbage(text: str) -> bool:
    """True when this is mis-decoded PDF glyphs rather than language.

    A PDF with a broken ToUnicode map extracts as spaced Latin-1 supplement
    characters ("Ã Ì À > Ê"). Sent to a translator it comes back either echoed
    verbatim or as a complaint -- and both then become *labels* on the page.

    Two signatures, and both must hold, because either alone over-fires: Thai
    prose quoting arithmetic trips the character test, and accented European
    words come close. So a line is garbage only when it looks like soup *and*
    carries no run of three or more real letters. Mojibake never has those; prose
    always does.
    """
    stripped = (text or "").strip()
    if len(stripped) < 3:
        return False
    if len(_WORD_RUN.findall(stripped)) >= 2:
        return False
    singles = sum(1 for token in stripped.split() if len(token) == 1 and token in _MOJIBAKE)
    density = sum(ch in _MOJIBAKE for ch in stripped) / len(stripped)
    return singles >= 3 or density > 0.25


def has_words(text: str) -> bool:
    """True when there is something here to translate.

    A table cell holding "-", "12,651" or "%" gives the model nothing to work
    on -- and the prompt hands it page context to set register with, so asked to
    translate a dash it translates the context instead. A paragraph of Thai then
    lands in a 4x10 cell, overflows across its neighbours, and the label says it
    belongs there. Regions like this keep their source text, which is already
    what a number or a rule should say.
    """
    return bool(_WORDLIKE.search(text))


def is_disproportionate(source: str, translation: str) -> bool:
    """True when a reply is too long to be a translation of `source`.

    Thai runs longer than English, but not several times longer. When it does,
    the model has answered with something other than the region -- the context,
    a gloss, the whole page -- and that answer would become a label.
    """
    return len(translation) > LENGTH_TOLERANCE * len(source) + 10


def _clean(raw: str) -> str:
    """Strip the wrappers models add despite being told not to."""
    text = (raw or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”„«»":
        text = text[1:-1].strip()
    return "" if _REFUSAL.search(text) else text


@dataclass
class KeepText:
    """Reuse the source text, optionally re-mixing digits and era."""

    locale: bool = True

    def __call__(self, page: Page, rng: random.Random) -> list[str]:
        if not self.locale:
            return [region.text for region in page.regions]
        choice = thai.sample_locale(rng)
        return [thai.apply_locale(region.text, choice) for region in page.regions]


@dataclass
class TranslateText:
    """Translate every region into Thai, one request each, in parallel."""

    client: LLMClient
    locale: bool = True

    def __call__(self, page: Page, rng: random.Random) -> list[str]:
        # A little page context helps the model pick register and terminology;
        # the region itself is still what gets translated.
        context = " | ".join(r.text for r in page.regions[:6])[:400]

        def translate(text: str) -> str:
            if not text.strip() or looks_like_garbage(text):
                return ""
            if not has_words(text):
                return text
            prompt = f"Document context: {context}\n\nTranslate into Thai:\n{text}"
            translated = _clean(
                self.client.complete(TRANSLATE_SYSTEM, prompt, namespace="translate")
            )
            if translated and is_disproportionate(text, translated):
                log.warning("discarding a %d-character reply for a %d-character region; "
                            "keeping the source text", len(translated), len(text))
                return text
            return translated

        results = self.client.map([r.text for r in page.regions], translate)
        if not self.locale:
            return results
        choice = thai.sample_locale(rng)
        return [thai.apply_locale(text, choice) for text in results]


@dataclass
class SynthText:
    """Invent Thai text of a similar length and kind for every region."""

    client: LLMClient
    locale: bool = True

    def __call__(self, page: Page, rng: random.Random) -> list[str]:
        def synthesize(region) -> str:
            length = len(region.text.strip())
            if not length:
                return ""
            prompt = (
                f"Document region category: {region.category}\n"
                f"Approximate length: {length} characters\n"
                f"Write the Thai text for this region."
            )
            return _clean(self.client.complete(SYNTH_SYSTEM, prompt, namespace="synth"))

        results = self.client.map(list(page.regions), synthesize)
        if not self.locale:
            return results
        choice = thai.sample_locale(rng)
        return [thai.apply_locale(text, choice) for text in results]


def _client(settings: Settings | None) -> LLMClient:
    settings = settings or get_settings()
    return LLMClient(settings.llm, settings.cache_dir)


@TEXT_GENERATORS.register("keep")
def _keep(settings: Settings | None = None, locale: bool = True) -> KeepText:
    return KeepText(locale=locale)


@TEXT_GENERATORS.register("translate")
def _translate(settings: Settings | None = None, locale: bool = True) -> TranslateText:
    return TranslateText(client=_client(settings), locale=locale)


@TEXT_GENERATORS.register("synth")
def _synth(settings: Settings | None = None, locale: bool = True) -> SynthText:
    return SynthText(client=_client(settings), locale=locale)
