"""Command line interface.

    docaug run --source doclaynet --limit 20 --out out/
    docaug run --source jsonl --input corpus/ --text translate --out out/
    docaug list
    docaug fonts scan

Every flag has a default, and every default can be moved into `.env` -- see
`.env.example`. Flags win over `.env`, which wins over the built-in defaults.

Components are addressed by the names they registered under, so an installed
plugin is usable here the moment it is installed. `--source-opt`, `--renderer-opt`
and friends pass arbitrary keyword arguments through to a component's factory,
which is how a third-party component takes options this CLI has never heard of.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from pathlib import Path

from .config import get_settings
from .erase import ERASERS
from .fonts import FontBank, UnknownFamily, write_manifest
from .pipeline import Pipeline
from .registry import UnknownComponent
from .render import RENDERERS, ChainRenderer
from .sources import SOURCES
from .text import TEXT_GENERATORS
from .transforms import TRANSFORMS
from .writers import WRITERS


def _parse_options(pairs: list[str] | None) -> dict:
    """`["a=1", "b=x"]` -> `{"a": 1, "b": "x"}`, values JSON-decoded when they can be."""
    options = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            options[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            options[key.strip()] = raw
    return options


def _build(registry, name: str, **kwargs):
    """Create a component, passing only the arguments its factory accepts.

    This is what lets one set of CLI flags drive components with different
    signatures -- `--limit` reaches a source that takes it and is quietly
    dropped for one that does not.
    """
    registry._load_plugins()  # so the signature below is the real factory's
    factory = registry._factories.get(name)
    if factory is None:
        return registry.create(name, **kwargs)  # raises with the list of names
    parameters = inspect.signature(factory).parameters
    accepts_any = any(p.kind is p.VAR_KEYWORD for p in parameters.values())
    allowed = kwargs if accepts_any else {k: v for k, v in kwargs.items() if k in parameters}
    return factory(**allowed)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def command_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.seed is not None:
        settings.seed = args.seed
    if args.fonts_dir:
        settings.fonts_dir = Path(args.fonts_dir)
    if args.glyph_bank:
        settings.glyph_bank_dir = Path(args.glyph_bank)

    source = _build(
        SOURCES, args.source,
        path=args.input, limit=args.limit, skip=args.skip,
        **_parse_options(args.source_opt),
    )
    text = _build(TEXT_GENERATORS, args.text, settings=settings,
                  **_parse_options(args.text_opt))
    eraser = _build(ERASERS, args.erase)
    renderer = _build(
        RENDERERS, args.renderer, settings=settings, font=args.font,
        bank_dir=settings.glyph_bank_dir, **_parse_options(args.renderer_opt),
    )
    # Real glyph banks never cover every character on a page. Falling back to a
    # face keeps a region readable instead of dropping it, which is why this is
    # the default rather than something you have to know to ask for.
    if args.renderer == "glyph" and args.fallback:
        renderer = ChainRenderer([renderer, _build(RENDERERS, "font", settings=settings)])

    transforms = [_build(TRANSFORMS, name) for name in args.transform]

    out = Path(args.out)
    writers = [
        _build(WRITERS, "dataset", out=out, image_format=args.format,
               clusters=not args.no_clusters)
    ]
    if args.pagejson:
        writers.append(_build(WRITERS, "pagejson", out=out))
    if args.previews:
        writers.append(
            _build(WRITERS, "preview", out=out / "previews",
                   limit=args.previews, boxes=args.boxes)
        )

    pipeline = Pipeline(
        source=source, text=text, eraser=eraser, renderer=renderer,
        writers=writers, transforms=transforms, settings=settings,
        min_fit_rate=args.min_fit_rate,
    )
    report = pipeline.run()
    (out / "report.json").write_text(
        json.dumps(
            {
                "pages": report.pages, "regions": report.regions,
                "fit_rate": report.fit_rate, "empty": report.empty,
                "failed": report.failed, "rejected": report.rejected,
                "config": {
                    "source": args.source, "text": args.text, "erase": args.erase,
                    "renderer": args.renderer, "transforms": args.transform,
                    "seed": settings.seed, "min_fit_rate": args.min_fit_rate,
                },
            },
            indent=2,
        )
    )
    print(report.summary())
    print(f"\nwrote {out}")
    return 0 if report.pages else 1


# --------------------------------------------------------------------------- #
# list / fonts
# --------------------------------------------------------------------------- #

def command_list(args: argparse.Namespace) -> int:
    for title, registry in [
        ("sources", SOURCES), ("text generators", TEXT_GENERATORS),
        ("erasers", ERASERS), ("renderers", RENDERERS),
        ("transforms", TRANSFORMS), ("writers", WRITERS),
    ]:
        print(f"{title:17} {', '.join(registry.names())}")
    return 0


def command_fonts(args: argparse.Namespace) -> int:
    settings = get_settings()
    root = Path(args.dir or settings.fonts_dir)
    if args.action == "scan":
        path = write_manifest(root)
        print(f"wrote {path}")
    bank = FontBank.load(root)
    print(bank.summary())
    print("\nfamilies: " + ", ".join(bank.families()))
    return 0


def command_glyphs(args: argparse.Namespace) -> int:
    from .render.glyph import GlyphBank

    settings = get_settings()
    root = args.dir or settings.glyph_bank_dir
    if root is None:
        print(
            "no glyph bank configured. Set DOCAUG_GLYPH_BANK_DIR in .env or pass "
            "--dir; no bank ships with docaug -- see docs/glyph-bank.md.",
            file=sys.stderr,
        )
        return 1
    bank = GlyphBank.load(root)
    print(bank.summary())
    for char, instances in sorted(bank.instances.items()):
        print(f"  {char}  {len(instances):>4}")
    return 0


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docaug", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="reconstruct a dataset")
    run.add_argument("--source", default="jsonl", help="see `docaug list`")
    run.add_argument("--input", help="path the source reads from (jsonl: the corpus dir)")
    run.add_argument("--out", required=True, help="output dataset directory")
    run.add_argument("--limit", type=int, help="stop after this many pages")
    run.add_argument("--skip", type=int, default=0, help="skip this many pages first")
    run.add_argument("--text", default="keep", help="keep | translate | synth")
    run.add_argument("--erase", default="adaptive", help="adaptive | none")
    run.add_argument("--renderer", default="font", help="font | glyph")
    run.add_argument("--font", metavar="FAMILY_OR_PATH",
                     help="pin every page to one typeface, by family name (see "
                          "`docaug fonts`) or path; the control condition")
    run.add_argument("--fonts-dir", help="override DOCAUG_FONTS_DIR")
    run.add_argument("--glyph-bank", help="handwriting glyph bank directory")
    run.add_argument("--no-fallback", dest="fallback", action="store_false",
                     help="do not fall back to a typeface for uncovered regions")
    run.add_argument("--transform", action="append", default=[],
                     help="repeatable: keep | white-background | stack")
    run.add_argument("--seed", type=int, help="override DOCAUG_SEED")
    run.add_argument("--min-fit-rate", type=float, default=0.0, metavar="RATE",
                     help="drop a page unless this fraction of its regions fitted; "
                          "1.0 rejects every page with text that overflowed its box")
    run.add_argument("--format", default="png", choices=["png", "jpg"])
    run.add_argument("--pagejson", action="store_true",
                     help="also write page-level VLM training targets")
    run.add_argument("--previews", type=int, default=8, help="side-by-side previews (0 = none)")
    run.add_argument("--boxes", action="store_true", help="draw cluster boxes on previews")
    run.add_argument("--no-clusters", action="store_true",
                     help="omit cluster boxes from labels.jsonl")
    for name in ("source", "text", "renderer"):
        run.add_argument(f"--{name}-opt", action="append", metavar="KEY=VALUE",
                         help=f"extra keyword argument for the {name}")
    run.set_defaults(func=command_run)

    listing = sub.add_parser("list", help="show every registered component")
    listing.set_defaults(func=command_list)

    fonts = sub.add_parser("fonts", help="inspect or rebuild the font manifest")
    fonts.add_argument("action", nargs="?", default="show", choices=["show", "scan"])
    fonts.add_argument("--dir", help="font directory (default: DOCAUG_FONTS_DIR)")
    fonts.set_defaults(func=command_fonts)

    glyphs = sub.add_parser("glyphs", help="inspect a handwriting glyph bank")
    glyphs.add_argument("--dir", help="bank directory (default: DOCAUG_GLYPH_BANK_DIR)")
    glyphs.set_defaults(func=command_glyphs)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except (RuntimeError, FileNotFoundError, ImportError, UnknownComponent,
            UnknownFamily) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
