"""The real-glyph stitcher, exercised against a stand-in bank (see conftest)."""

import pytest

from docaug import ERASERS, RENDERERS, WRITERS, Pipeline
from docaug.render import ChainRenderer, RenderRequest
from docaug.render.glyph import GlyphBank, stitch_line

SAMPLES = ["สวัสดีครับ", "ปุ๋ยเคมี", "ผู้ป่วยมีอาการไข้", "น้ำตาลทรายขาว"]


@pytest.fixture(scope="session")
def glyphs(glyph_bank_dir) -> GlyphBank:
    return GlyphBank.load(glyph_bank_dir)


def test_bank_loads_marks_and_consonants(glyphs):
    assert glyphs.has("ก")
    assert glyphs.has("่")  # a combining mark, stored as ◌่ on disk
    assert len(glyphs) > 100


@pytest.mark.parametrize("text", SAMPLES)
def test_stitched_labels_reassemble_the_text(text, glyphs, face, rng):
    _, clusters, _ = stitch_line(text, glyphs, face, 48, rng=rng)
    assert "".join(c.text for c in clusters) == text


def test_a_space_stays_in_the_label_even_though_it_draws_nothing(glyphs, face, rng):
    _, clusters, _ = stitch_line("ก ข", glyphs, face, 48, rng=rng)
    assert "".join(c.text for c in clusters) == "ก ข"


def test_sara_am_survives_a_mark_standing_in_front_of_it(glyphs, face, rng):
    """ำ shapes into two glyphs, and the shaper merges them with a tone mark on
    the same base -- so all three arrive under one cluster id. Walking the ids
    instead of the characters drew the base and dropped the vowel, out of the
    ink and out of the label at the same time: น้ำ came out as น้."""
    for text in ["น้ำ", "คำ", "สำหรับ", "น้ำใจ"]:
        _, clusters, _ = stitch_line(text, glyphs, face, 48, rng=rng)
        assert "".join(c.text for c in clusters) == text


def test_repeated_characters_are_written_differently(glyphs, face, rng):
    """The reason to stitch at all: a font draws every 'ก' identically."""
    widths = set()
    for _ in range(12):
        _, clusters, _ = stitch_line("กกก", glyphs, face, 48, rng=rng)
        widths.update(c.box[2] - c.box[0] for c in clusters)
    assert len(widths) > 1


def test_uncovered_characters_fall_back_instead_of_vanishing(glyphs, face, rng):
    _, clusters, _ = stitch_line("ก ABC", glyphs, face, 48, rng=rng)
    assert "".join(c.text for c in clusters) == "ก ABC"


def test_missing_bank_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="docs/glyph-bank.md"):
        GlyphBank.load(tmp_path / "absent")


def test_renderer_declines_a_region_it_cannot_cover(glyph_bank_dir, settings, page, rng):
    renderer = RENDERERS.create("glyph", settings=settings, bank_dir=glyph_bank_dir,
                                min_coverage=0.9)
    page_renderer = renderer.for_page(page, rng)
    # No Thai at all: coverage is zero, so the region belongs to another renderer.
    assert page_renderer.render(RenderRequest("Hello world", 300, 40)) is None


def test_chain_falls_back_to_a_typeface(glyph_bank_dir, settings, page, tmp_path):
    pipeline = Pipeline(
        source=[],
        text=lambda p, rng: ["สวัสดีครับ", "Latin only here", "ผู้ป่วยมีอาการไข้"],
        eraser=ERASERS.create("adaptive"),
        renderer=ChainRenderer(
            [
                RENDERERS.create("glyph", settings=settings, bank_dir=glyph_bank_dir),
                RENDERERS.create("font", settings=settings),
            ]
        ),
        writers=[WRITERS.create("dataset", out=tmp_path / "out")],
        settings=settings,
    )
    result = pipeline.process(page)
    assert all(label.clusters for label in result.labels)
    for label in result.labels:
        assert "".join(c.text for c in label.clusters) == label.text
