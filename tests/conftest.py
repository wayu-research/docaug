"""Fixtures built from the bundled fonts, so the suite needs no network."""

from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

from docaug import thai
from docaug.config import BUNDLED_FONTS, Settings
from docaug.fonts import FontBank
from docaug.render.glyph import DOTTED_CIRCLE
from docaug.shaping import ft_face, glyph_bitmap
from docaug.types import Page, Region


@pytest.fixture(scope="session")
def bank() -> FontBank:
    return FontBank.load(BUNDLED_FONTS)


@pytest.fixture(scope="session")
def face(bank: FontBank):
    return bank.by_category("sans")[0].face(bank.root)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(fonts_dir=BUNDLED_FONTS, cache_dir=tmp_path / "cache")


@pytest.fixture
def page() -> Page:
    """A three-region document page drawn from scratch."""
    image = Image.new("RGB", (600, 400), (250, 249, 245))
    draw = ImageDraw.Draw(image)
    regions = []
    for i, (text, box) in enumerate(
        [
            ("Annual Report", (40, 30, 400, 62)),
            ("Section one covers the results for the period.", (40, 90, 560, 118)),
            ("Total revenue increased by twelve percent.", (40, 140, 560, 168)),
        ]
    ):
        draw.rectangle(box, fill=(255, 255, 255))
        draw.text((box[0] + 4, box[1] + 6), text, fill=(20, 20, 20))
        regions.append(Region(box=box, text=text, category="Title" if not i else "Text",
                              ink_height=14.0))
    return Page(id="fixture", image=image, regions=regions)


@pytest.fixture(scope="session")
def glyph_bank_dir(tmp_path_factory, bank: FontBank):
    """A stand-in glyph bank: one 'instance' per handwriting face per character.

    Not a real bank -- real instances are cut from handwritten pages -- but it
    has the same shape on disk, which is what the stitcher is being tested on.
    """
    root = tmp_path_factory.mktemp("glyph_bank")
    faces = [entry.face(bank.root) for entry in bank.by_category("handwriting")]
    characters = thai.CONSONANTS + thai.ABOVE_VOWELS + thai.BELOW_VOWELS + thai.TONE_MARKS
    for char in characters:
        directory = root / (DOTTED_CIRCLE + char if char in thai.COMBINING else char)
        directory.mkdir(exist_ok=True)
        for index, face in enumerate(faces):
            ft = ft_face(face, 96)
            gid = ft.get_char_index(ord(char))
            if not gid:
                continue
            bitmap, _, _ = glyph_bitmap(ft, gid)
            if not bitmap.size:
                continue
            ys, xs = np.nonzero(bitmap)
            if not xs.size:
                continue
            crop = bitmap[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            Image.fromarray(255 - crop).save(directory / f"{index:03d}.png")
    return root


@pytest.fixture
def rng() -> random.Random:
    return random.Random(1234)
