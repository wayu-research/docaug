"""The text generators, with the network stubbed out."""

import random

import pytest

from docaug.text.llm import DiskCache, LLMClient, cache_key
from docaug.text.providers import (
    SynthText,
    TranslateText,
    _clean,
    has_words,
    is_disproportionate,
    looks_like_garbage,
)
from docaug.types import Page, Region


class StubClient(LLMClient):
    """An LLMClient whose completions come from a canned reply, not a provider."""

    def __init__(self, settings, cache_dir, reply="ข้อความภาษาไทย"):
        super().__init__(settings, cache_dir)
        self.reply = reply
        self.prompts = []

    def _call(self, system, user):
        self.prompts.append(user)
        return self.reply


@pytest.fixture
def client(settings, tmp_path):
    return StubClient(settings.llm, tmp_path / "cache")


def test_translate_returns_one_string_per_region(page, client, rng):
    result = TranslateText(client=client, locale=False)(page, rng)
    assert len(result) == len(page.regions)
    assert all(text == "ข้อความภาษาไทย" for text in result)


def test_one_request_per_region_not_one_batched_request(page, client, rng):
    """Batching regions into one indexed response is what slides every later
    translation onto the wrong box when the model drops an entry."""
    TranslateText(client=client, locale=False)(page, rng)
    assert len(client.prompts) == len(page.regions)


def test_responses_are_cached_by_content(page, client, rng):
    generator = TranslateText(client=client, locale=False)
    generator(page, rng)
    calls = len(client.prompts)
    generator(page, rng)  # same text, same model -> no new requests
    assert len(client.prompts) == calls


def test_locale_is_applied_once_per_page(page, settings, tmp_path, rng):
    client = StubClient(settings.llm, tmp_path / "cache", reply="ปี 2024")
    seen = {tuple(TranslateText(client=client)(page, random.Random(s))) for s in range(20)}
    # Every region of a page shares the page's locale choice, so each outcome is
    # uniform across the page.
    for outcome in seen:
        assert len(set(outcome)) == 1
    # ...and across seeds we see more than one outcome, i.e. it really is sampled.
    assert len(seen) > 1


def test_a_failed_completion_leaves_the_region_erased(page, settings, tmp_path, rng):
    client = StubClient(settings.llm, tmp_path / "cache", reply="")
    assert TranslateText(client=client, locale=False)(page, rng) == [""] * len(page.regions)


def test_synth_prompts_with_the_category_and_length(page, client, rng):
    SynthText(client=client, locale=False)(page, rng)
    assert "Title" in client.prompts[0]
    assert str(len(page.regions[0].text)) in client.prompts[0]


def test_mojibake_is_dropped_before_it_reaches_the_model(page, client, rng):
    """A PDF with a broken ToUnicode map extracts as Latin-1 soup. Translated, it
    comes back either echoed or as a complaint -- and both become labels."""
    assert looks_like_garbage("Ã Ì À > Ê Ï Ð")
    assert not looks_like_garbage("The board resolved to proceed")
    assert not looks_like_garbage("65 × 100 ÷ 80 = 81.25")  # real arithmetic
    assert not looks_like_garbage("§ 1 und § 363 BGB")      # real German law


def test_refusals_are_treated_as_failures_not_translations():
    assert _clean("Please provide the text you would like me to translate.") == ""
    assert _clean("กรุณาระบุข้อความที่ต้องการให้แปล") == ""
    assert _clean('"สวัสดีครับ"') == "สวัสดีครับ"


def test_cache_key_is_stable_and_content_addressed():
    assert cache_key("a", {"b": 1}) == cache_key("a", {"b": 1})
    assert cache_key("a", {"b": 1}) != cache_key("a", {"b": 2})


def test_disk_cache_survives_a_truncated_entry(tmp_path):
    cache = DiskCache(tmp_path)
    assert cache.get_or_set("ns", "k", lambda: "value") == "value"
    (tmp_path / "ns" / "k.json").write_text("{not json")
    assert cache.get_or_set("ns", "k", lambda: "recomputed") == "recomputed"


@pytest.mark.parametrize("text", ["-", "12,651", "%", "1.0", "N/A", "(4)"])
def test_a_region_with_no_words_in_it_is_not_worth_translating(text):
    assert not has_words(text)


@pytest.mark.parametrize("text", ["Total assets", "รายงาน", "Note 50"])
def test_a_region_with_words_in_it_is(text):
    assert has_words(text)


def test_a_region_with_nothing_to_translate_keeps_its_source_text(page, client, rng):
    """The prompt carries page context so the model can pick a register. Asked to
    translate a dash, the model translates that context instead, and a paragraph
    of Thai lands in a table cell four pixels wide. Regions like this never go
    out at all."""
    cells = Page(
        id=page.id,
        image=page.image,
        regions=[Region(box=r.box, text="-", category=r.category) for r in page.regions],
    )
    assert TranslateText(client=client, locale=False)(cells, rng) == ["-"] * len(page.regions)
    assert client.prompts == []


def test_a_reply_far_longer_than_its_region_is_discarded(page, settings, tmp_path, rng):
    """Whatever that reply is, it is not a translation of this region -- and
    without this check it would become the region's label."""
    client = StubClient(settings.llm, tmp_path / "cache", reply="ข้อความยาวมาก" * 40)
    result = TranslateText(client=client, locale=False)(page, rng)
    assert result == [r.text for r in page.regions]


def test_a_translation_of_a_reasonable_length_is_kept(page, client, rng):
    result = TranslateText(client=client, locale=False)(page, rng)
    assert result == [client.reply] * len(page.regions)
    assert not is_disproportionate("Annual Report", client.reply)
