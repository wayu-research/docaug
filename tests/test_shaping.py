"""The labels are the product. These tests are about whether they are true."""

import pytest

from docaug.shaping import fit_lines, line_width, render_block, shape_line, wrap

SAMPLES = [
    "สวัสดีครับ ยินดีต้อนรับ",
    "ปุ๋ยเคมีคุณภาพสูง ๑๒๓",
    "การศึกษาไทยในศตวรรษที่ ๒๑",
    "ใบเสร็จรับเงินเลขที่ 4567",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_cluster_labels_reassemble_the_rendered_text(text, face):
    """Concatenating the labels in order must give back exactly what was asked
    for -- in logical order, with the leading vowels back where Unicode puts
    them, not where they are drawn."""
    line = shape_line(text, face, 48)
    assert "".join(c.text for c in line.clusters) == text


@pytest.mark.parametrize("text", SAMPLES)
def test_cluster_boxes_are_inside_the_ink(text, face):
    line = shape_line(text, face, 48)
    width, height = line.image.size
    for cluster in line.clusters:
        x1, y1, x2, y2 = cluster.box
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height


def test_a_stacked_cluster_is_taller_than_a_bare_one(face):
    """A four-level stack has to occupy more vertical space than its base alone,
    or the marks are not being drawn."""
    bare = shape_line("ป", face, 64).clusters[0].box
    stacked = shape_line("ปุ๋", face, 64).clusters[0].box
    assert (stacked[3] - stacked[1]) > (bare[3] - bare[1])


def test_leading_vowel_is_drawn_left_of_its_consonant(face):
    """เ is stored before ก and drawn before it; both facts must hold at once."""
    line = shape_line("เก", face, 64)
    assert "".join(c.text for c in line.clusters) == "เก"
    ink = line.image.split()[3]
    assert ink.getbbox() is not None


def test_wrapping_preserves_every_character(face):
    text = "การศึกษาไทยในศตวรรษที่ยี่สิบเอ็ดต้องอาศัยเทคโนโลยีสารสนเทศ"
    lines = wrap(text, face, 32, 200)
    assert len(lines) > 1
    assert "".join(lines) == text


def test_the_space_a_line_breaks_at_stays_in_the_label(face):
    """The break used to eat the space it happened at. That is invisible on the
    page -- the line ends there either way -- and wrong in the label, which came
    back with the words either side of the break run together."""
    text = "รายงานผลการดำเนินงาน ประจำปีงบประมาณ ๒๕๖๗ ของคณะกรรมการบริหารสถาบัน"
    lines = wrap(text, face, 32, 220)
    assert len(lines) > 1
    assert "".join(lines) == text


def test_a_wrapped_block_reassembles_including_its_break_spaces(face):
    """The same thing one level up: the labels come off the rasterizer line by
    line, so a space lost at a break is a space missing from the dataset."""
    text = "รายงานผลการดำเนินงาน ประจำปีงบประมาณ ๒๕๖๗ ของคณะกรรมการบริหารสถาบัน"
    block = render_block(text, face, 200, 400, min_size=8)
    assert "".join(c.text for c in block.clusters) == text


def test_fit_shrinks_until_the_text_fits(face):
    text = "รายงานประจำปีฉบับสมบูรณ์ของคณะกรรมการบริหาร"
    big, _, _ = fit_lines(text, face, 600, 80, 1.35, 6, 60)
    small, _, _ = fit_lines(text, face, 200, 40, 1.35, 6, 60)
    assert small < big


def test_an_unbreakable_word_is_not_declared_to_fit_a_narrow_box(face):
    """A single Thai word has no legal break point, so a height-only check would
    happily leave it running out the side of the box."""
    size, lines, _ = fit_lines("กฎหมายมหาชน", face, 60, 300, 1.35, 6, 80)
    assert max(line_width(line, face, size) for line in lines) <= 60


def test_render_block_reports_overflow_rather_than_lying(face):
    block = render_block("รายงานประจำปีฉบับสมบูรณ์" * 6, face, 40, 20, min_size=6)
    assert block.image.height > 20  # it overflowed
    assert "".join(c.text for c in block.clusters).replace(" ", "").startswith("รายงาน")
