# Bundled fonts

Every face here is redistributable. The table below is generated from
`manifest.json`, so it cannot drift from what is actually in the tree.

Two sources, two licences:

* **Google Fonts** -- SIL Open Font License 1.1 (`OFL-1.1.txt`).
  Upstream: <https://fonts.google.com>.
* **TLWG** (Thai Linux Working Group) -- GPL-2.0-or-later *with the font
  exception*, which is what makes an embedded or rendered document free of
  the GPL (`TLWG-copyright.txt`). Upstream:
  <https://linux.thai.net/projects/fonts-tlwg>.

The bank is deliberately small -- enough diversity to be useful, small
enough to clone. If you are building a real corpus, add more:

```bash
mkdir -p myfonts/{sans,serif,handwriting,display,monospace}
cp /path/to/more/*.ttf myfonts/handwriting/
docaug fonts scan --dir myfonts
export DOCAUG_FONTS_DIR=myfonts
```

Coverage is checked at scan time and again per string at render time, so an
incomplete face can never put a tofu box into the ground truth.

| Family | Category | Faces | Source | Licence |
|---|---|---:|---|---|
| Charm | handwriting | 1 | Google Fonts | OFL-1.1 |
| Charmonman | handwriting | 1 | Google Fonts | OFL-1.1 |
| Chonburi | display | 1 | Google Fonts | OFL-1.1 |
| Garuda | sans | 1 | TLWG | GPL-2.0+ with font exception |
| Itim | handwriting | 1 | Google Fonts | OFL-1.1 |
| Kanit | sans | 1 | Google Fonts | OFL-1.1 |
| Kinnari | serif | 1 | TLWG | GPL-2.0+ with font exception |
| Maitree | serif | 1 | Google Fonts | OFL-1.1 |
| Mali | handwriting | 1 | Google Fonts | OFL-1.1 |
| Norasi | serif | 1 | TLWG | GPL-2.0+ with font exception |
| Noto Sans Thai | sans | 2 | Google Fonts | OFL-1.1 |
| Noto Serif Thai | serif | 2 | Google Fonts | OFL-1.1 |
| Pattaya | display | 1 | Google Fonts | OFL-1.1 |
| Prompt | sans | 1 | Google Fonts | OFL-1.1 |
| Purisa | handwriting | 1 | TLWG | GPL-2.0+ with font exception |
| Sarabun | sans | 2 | Google Fonts | OFL-1.1 |
| Sawasdee | handwriting | 1 | TLWG | GPL-2.0+ with font exception |
| Sriracha | handwriting | 1 | Google Fonts | OFL-1.1 |
| Tlwg Mono | monospace | 1 | TLWG | GPL-2.0+ with font exception |
| Tlwg Typewriter | monospace | 1 | TLWG | GPL-2.0+ with font exception |
| Tlwg Typist | monospace | 1 | TLWG | GPL-2.0+ with font exception |
| Umpush | sans | 1 | TLWG | GPL-2.0+ with font exception |

22 families, 25 faces.
