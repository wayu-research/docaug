import random

from docaug import thai


def test_cluster_spans_reassemble_the_input():
    for text in ["สวัสดีครับ", "เก", "แล้ว", "ปุ๋ยเคมี", "ไทย 2024", ""]:
        assert "".join(token for token, _, _ in thai.cluster_spans(text)) == text


def test_a_leading_vowel_binds_to_the_consonant_that_follows_it():
    # เ is written left of ก but stored before it; one cluster, not two.
    assert [t for t, _, _ in thai.cluster_spans("เก")] == ["เก"]
    assert [t for t, _, _ in thai.cluster_spans("แล้ว")] == ["แล้", "ว"]


def test_marks_stack_onto_their_base():
    assert [t for t, _, _ in thai.cluster_spans("ปุ๋ย")] == ["ปุ๋", "ย"]
    assert thai.stack_depth("ก") == 1
    assert thai.stack_depth("ปุ๋") == 3  # base + below vowel + tone
    # The full four-level stack the renderer has to keep upright.
    assert thai.stack_depth("ก" + "ุ" + "ิ" + "่") == 4


def test_bands():
    assert thai.band("ก") == "main"
    assert thai.band("่") == "above"
    assert thai.band("ุ") == "below"


def test_locale_is_sampled_not_forced():
    assert thai.to_thai_digits("2024") == "๒๐๒๔"
    assert thai.years_to_be("in 2023 and 1999") == "in 2566 and 2542"
    # A page's choice is one draw, applied to every region on it.
    choice = thai.sample_locale(random.Random(0))
    assert set(choice) == {"be", "thai_digits"}


def test_required_codepoints_are_unique_and_cover_the_consonants():
    codepoints = thai.required_codepoints()
    assert len(codepoints) == len(set(codepoints))
    assert all(ord(c) in codepoints for c in thai.CONSONANTS)
