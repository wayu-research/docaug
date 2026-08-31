"""The font bank: which typefaces exist, and how to draw one.

Typeface diversity is a property worth controlling deliberately, so sampling is
not an afterthought. Two rules shape it.

**Coverage is checked, twice.** A face joins the bank only if it draws every
character modern Thai actually uses; archaic marks (`thai.RARE`) are excused,
because rejecting on those would throw away most of the good faces. Then, at
sampling time, the specific string is re-checked against the specific face, so a
line that *does* use a rare mark still cannot end up as tofu in the labels.

**Sampling is design-balanced, not face-balanced.** Some families ship eighteen
weights and some ship one. Drawing faces uniformly would make a dataset that is
mostly one superfamily's grades. So the sampler picks a *category*, then a
*family*, then a face -- each distinct design gets a fair share, and handwriting
is not drowned by sans-serif weights.

The bundled bank is small and entirely OFL / GPL-with-font-exception. Point
`DOCAUG_FONTS_DIR` at a larger collection and rerun `docaug fonts scan` to widen
it; the manifest format is the contract, not this directory.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import freetype

from . import thai
from .config import BUNDLED_FONTS
from .shaping import Face

MANIFEST_NAME = "manifest.json"


def _fold(name: str) -> str:
    """Family-name join key: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


class UnknownFamily(LookupError):
    """Raised for a family the bank does not have, listing the ones it does."""

    def __init__(self, family: str, known: list[str]) -> None:
        super().__init__(family)
        self.family, self.known = family, known

    def __str__(self) -> str:
        return f"no font family {self.family!r} in the bank; available: {', '.join(self.known)}"

CATEGORY_WEIGHTS = {
    "sans": 0.42,
    "serif": 0.18,
    "handwriting": 0.25,
    "display": 0.10,
    "monospace": 0.05,
}
"""Sampling mix over categories. Handwriting is deliberately over-weighted
relative to its share of the bank: script faces are the scarce, useful kind."""

_WEIGHT_TOKENS = {
    "thin": 100, "extralight": 200, "ultralight": 200, "light": 300,
    "regular": 400, "normal": 400, "book": 400, "medium": 500,
    "semibold": 600, "demibold": 600, "bold": 700, "extrabold": 800,
    "ultrabold": 800, "black": 900, "heavy": 900,
}


@dataclass(frozen=True, slots=True)
class FontEntry:
    """One face in the bank."""

    path: str
    """Relative to the bank directory, so a manifest stays portable."""
    family: str
    category: str
    weight: int = 400
    italic: bool = False
    index: int = 0
    license: str = ""
    missing: frozenset[int] = frozenset()
    """Codepoints from `thai.required_codepoints()` this face cannot draw."""

    def face(self, root: Path) -> Face:
        return Face(str(root / self.path), self.index)

    def covers(self, text: str) -> bool:
        return not any(ord(ch) in self.missing for ch in text)


@dataclass
class FontBank:
    """A loaded, coverage-filtered collection of faces."""

    root: Path
    entries: list[FontEntry]
    _by_family: dict[tuple[str, str], list[FontEntry]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for entry in self.entries:
            self._by_family.setdefault((entry.category, entry.family), []).append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, root: Path | str, *, strict: bool = False) -> FontBank:
        """Read `manifest.json` from `root`, scanning the directory if absent.

        `strict` demands full coverage including the archaic marks; the default
        excuses `thai.RARE`.
        """
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"font directory {root} does not exist. The bundled bank lives at "
                f"{BUNDLED_FONTS}; set DOCAUG_FONTS_DIR to use another."
            )
        manifest = root / MANIFEST_NAME
        records = json.loads(manifest.read_text())["fonts"] if manifest.is_file() else scan(root)

        allowed = frozenset() if strict else thai.RARE
        entries = []
        for record in records:
            missing = frozenset(record.get("missing", ()))
            if missing - allowed:
                continue
            entries.append(FontEntry(**{**record, "missing": missing}))
        if not entries:
            raise RuntimeError(
                f"no usable Thai faces under {root}. Run `docaug fonts scan` after "
                f"adding fonts, or point DOCAUG_FONTS_DIR somewhere else."
            )
        return cls(root=root, entries=entries)

    # -- sampling --------------------------------------------------------- #
    def sample(
        self,
        rng: random.Random,
        *,
        text: str | None = None,
        categories: dict[str, float] | None = None,
    ) -> FontEntry:
        """Draw a face: category, then family, then face.

        `text` restricts the draw to faces that can render it. Raises only when
        *nothing* in the bank can, which means the text needs a character no
        installed face has.
        """
        weights = categories or CATEGORY_WEIGHTS
        present = sorted({c for c, _ in self._by_family})
        if not present:
            raise RuntimeError("font bank is empty")

        for _ in range(64):
            category = rng.choices(present, [weights.get(c, 0.03) for c in present])[0]
            families = [f for c, f in self._by_family if c == category]
            faces = self._by_family[(category, rng.choice(families))]
            if text is not None:
                faces = [f for f in faces if f.covers(text)]
            if faces:
                return rng.choice(faces)

        pool = [f for f in self.entries if text is None or f.covers(text)]
        if not pool:
            raise ValueError(f"no face in the bank covers {text!r}")
        return rng.choice(pool)

    def variant(self, entry: FontEntry, *, bold: bool = False) -> FontEntry:
        """The bold sibling of `entry` when the family has one, else `entry`.

        A real bold beats dilating a regular, so this is tried first and the
        faux-bold in the shaper is the fallback.
        """
        if not bold:
            return entry
        siblings = self._by_family.get((entry.category, entry.family), [])
        heavy = [f for f in siblings if f.weight >= 600 and f.italic == entry.italic]
        return max(heavy, key=lambda f: f.weight) if heavy else entry

    def find(self, family: str) -> FontEntry:
        """The face for a family name, matched loosely (case and spacing).

        Pinning a single typeface is the control condition for the diversity
        experiment, and naming a family is friendlier than naming a file inside
        an installed package.
        """
        key = _fold(family)
        matches = [f for f in self.entries if _fold(f.family) == key]
        if not matches:
            raise UnknownFamily(family, sorted({f.family for f in self.entries}))
        # The regular upright grade is what "the Sarabun face" means.
        return min(matches, key=lambda f: (f.italic, abs(f.weight - 400)))

    def families(self) -> list[str]:
        return sorted({f.family for f in self.entries})

    def by_category(self, category: str) -> list[FontEntry]:
        return [f for f in self.entries if f.category == category]

    def summary(self) -> str:
        counts = Counter(f.category for f in self.entries)
        families = {c: len({f.family for f in self.entries if f.category == c}) for c in counts}
        head = f"{len(self.entries)} faces / {len({f.family for f in self.entries})} families"
        rows = [
            f"  {c:<12} {counts[c]:>3} faces  {families[c]:>3} families"
            for c in sorted(counts)
        ]
        return "\n".join([f"FontBank({self.root}): {head}", *rows])


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #

def _describe(path: Path, root: Path) -> tuple[str, str, int, bool]:
    """(family, category, weight, italic) for one font file.

    The family and style come from the font's own name table, which is the only
    authoritative source -- filenames are a convention, not a contract. Weight is
    read from the style name, and falls back to a numeric filename suffix
    (`sarabun-700.ttf`) for the many Google Fonts files whose style name is just
    "Regular" across every grade.

    Category comes from the directory, because no font declares whether it is a
    display face or a handwriting one. That is the one thing you have to curate,
    and a directory tree is the least annoying place to record it::

        assets/fonts/<category>/<file>.ttf
    """
    relative = path.relative_to(root)
    category = relative.parts[0].lower() if len(relative.parts) > 1 else "sans"

    face = freetype.Face(str(path))
    family = (face.family_name or b"").decode("utf-8", "replace").strip() or path.stem
    style = (face.style_name or b"").decode("utf-8", "replace").strip().lower()
    italic = bool(face.style_flags & freetype.FT_STYLE_FLAG_ITALIC) or "oblique" in style

    weight = next((v for token, v in _WEIGHT_TOKENS.items() if token in style.replace(" ", "")), 0)
    if not weight:
        match = re.search(r"-(\d{3})i?$", path.stem)
        weight = int(match.group(1)) if match else 400
    return family, category, weight, italic


def scan(root: Path | str, *, license_note: str = "") -> list[dict]:
    """Walk `root` for font files and measure each one's Thai coverage."""
    root = Path(root)
    required = thai.required_codepoints()
    records = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".ttf", ".otf", ".ttc"}:
            continue
        family, category, weight, italic = _describe(path, root)
        try:
            missing = Face(str(path)).missing(required)
        except Exception:  # unreadable or non-font file with a font extension
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "family": family,
                "category": category,
                "weight": weight,
                "italic": italic,
                "index": 0,
                "license": license_note,
                "missing": sorted(missing),
            }
        )
    return records


def write_manifest(root: Path | str, records: list[dict] | None = None) -> Path:
    """Scan `root` and persist the manifest next to the fonts."""
    root = Path(root)
    records = scan(root) if records is None else records
    path = root / MANIFEST_NAME
    path.write_text(json.dumps({"fonts": records}, ensure_ascii=False, indent=1))
    return path
