import json

from docaug import ERASERS, RENDERERS, TEXT_GENERATORS, WRITERS, Pipeline
from docaug.transforms import TRANSFORMS

THAI = [
    "รายงานประจำปี ๒๕๖๗",
    "ระบบสารสนเทศเพื่อการจัดการเอกสารภาษาไทย",
    "ผู้รับผิดชอบต้องจัดทำเอกสารให้ครบถ้วน",
]


class FixedText:
    """A text generator with no network: returns a canned Thai line per region."""

    def __call__(self, page, rng):
        return [THAI[i % len(THAI)] for i in range(len(page.regions))]


def build(settings, tmp_path, **kwargs):
    return Pipeline(
        source=[],
        text=FixedText(),
        eraser=ERASERS.create("adaptive"),
        renderer=RENDERERS.create("font", settings=settings),
        writers=[WRITERS.create("dataset", out=tmp_path / "out")],
        settings=settings,
        **kwargs,
    )


def test_reconstructs_a_page_and_labels_every_region(page, settings, tmp_path):
    result = build(settings, tmp_path).process(page)
    assert result.image.size == page.image.size
    assert len(result.labels) == len(page.regions)
    for label, region in zip(result.labels, page.regions, strict=True):
        assert label.source_text == region.text
        assert label.text in THAI
        # The label must reassemble the text we asked to render.
        assert "".join(c.text for c in label.clusters) == label.text


def test_cluster_boxes_land_inside_their_region(page, settings, tmp_path):
    result = build(settings, tmp_path).process(page)
    for label in result.labels:
        x1, y1, x2, y2 = label.box
        for cluster in label.clusters:
            cx1, cy1, cx2, cy2 = cluster.box
            # Boxes are in page coordinates and belong to the region they label,
            # with a pixel of slack for the anti-aliased edge.
            assert x1 - 2 <= cx1 and cx2 <= x2 + 2
            assert y1 - 2 <= cy1 and cy2 <= y2 + 2


def test_the_source_text_is_actually_gone(page, settings, tmp_path):
    """Erasing has to change the pixels inside a region -- otherwise the page
    carries both languages and every label is wrong."""
    import numpy as np

    result = build(settings, tmp_path).process(page)
    box = page.regions[0].box
    before = np.asarray(page.image.crop(box).convert("L"), float)
    after = np.asarray(result.image.crop(box).convert("L"), float)
    assert np.abs(before - after).mean() > 1.0


def test_a_run_is_reproducible_from_its_seed(page, settings, tmp_path):
    import numpy as np

    first = build(settings, tmp_path).process(page, index=3)
    second = build(settings, tmp_path).process(page, index=3)
    assert first.meta["style"] == second.meta["style"]
    assert np.array_equal(np.asarray(first.image), np.asarray(second.image))


def test_different_pages_get_different_styles(page, settings, tmp_path):
    pipeline = build(settings, tmp_path)
    styles = {pipeline.process(page, index=i).meta["style"]["font"] for i in range(12)}
    assert len(styles) > 1


def test_run_writes_a_dataset(page, settings, tmp_path):
    pipeline = build(settings, tmp_path)
    report = pipeline.run([page, page])
    assert report.pages == 2
    assert report.regions == 6

    lines = (tmp_path / "out" / "labels.jsonl").read_text().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["regions"][0]["clusters"]
    assert (tmp_path / "out" / "images" / f"{record['id']}.png").is_file()


def test_a_failing_page_does_not_end_the_run(page, settings, tmp_path):
    class Broken:
        def __call__(self, page, rng):
            raise ValueError("nope")

    pipeline = build(settings, tmp_path)
    pipeline.text = Broken()
    report = pipeline.run([page])
    assert report.pages == 0
    assert report.failed == ["fixture"]


def test_white_background_clears_pixels_outside_the_regions(page, settings, tmp_path):
    import numpy as np

    pipeline = build(settings, tmp_path, transforms=[TRANSFORMS.create("white-background")])
    result = pipeline.process(page)
    corner = np.asarray(result.image.crop((500, 300, 600, 400)))
    assert (corner == 255).all()


def test_stack_puts_every_region_in_one_column(page, settings, tmp_path):
    pipeline = build(settings, tmp_path, transforms=[TRANSFORMS.create("stack")])
    result = pipeline.process(page)
    lefts = {label.box[0] for label in result.labels}
    assert len(lefts) == 1
    tops = [label.box[1] for label in result.labels]
    assert tops == sorted(tops)


def test_stacking_keeps_the_source_reading_order(page, settings, tmp_path):
    """A two-column page is read down one column and then the other, so the
    order the source gives is the only reading order there is. Sorting the
    regions by position here would restack those columns interleaved -- and the
    unstacked corpus this one is compared against is labelled in source order."""
    reordered = page.evolve(regions=list(reversed(page.regions)))
    pipeline = build(settings, tmp_path, transforms=[TRANSFORMS.create("stack")])
    result = pipeline.process(reordered)
    assert [x.source_text for x in result.labels] == [r.text for r in reordered.regions]


class Overflowing:
    """A generator whose text cannot fit any region at the minimum size."""

    def __call__(self, page, rng):
        return ["ข้อความภาษาไทยที่ยาวเกินกว่าจะบรรจุลงในกรอบนี้ได้" * 20] * len(page.regions)


def test_a_page_whose_text_overflows_is_rejected_when_asked(page, settings, tmp_path):
    """Text that will not fit even at the minimum size is still drawn, over
    whatever the page had beside it -- so the neighbouring labels end up
    describing pixels that are no longer there. This is how a corpus refuses
    those pages instead of shipping them."""
    pipeline = build(settings, tmp_path, min_fit_rate=1.0)
    pipeline.text = Overflowing()
    report = pipeline.run([page])
    assert report.pages == 0
    assert report.rejected == ["fixture"]


def test_an_overflowing_page_is_kept_by_default(page, settings, tmp_path):
    """The default keeps everything and writes `fit` per region, so the decision
    stays with whoever is assembling the dataset."""
    pipeline = build(settings, tmp_path)
    pipeline.text = Overflowing()
    report = pipeline.run([page])
    assert report.pages == 1
    assert report.rejected == []
    assert report.fit_rate < 1.0


def test_keep_generator_needs_no_api_key(page, settings):
    generator = TEXT_GENERATORS.create("keep", settings=settings, locale=False)
    assert generator(page, __import__("random").Random(0)) == [r.text for r in page.regions]
