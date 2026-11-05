"""Generate cached WOFF2 subsets from rendered Astro HTML and live Demo text.

The first tier covers the marked masthead/Demo plus recorded and dynamic UI
strings. Disjoint remaining glyphs are loaded later without replacing tier-one
characters. All generated files are ignored; build/dev entry points regenerate.
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import tempfile
import unicodedata

from fontTools import __version__ as fonttools_version, subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public/generated-fonts"
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class TextCollector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.page, self.critical = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        critical = attrs.get("data-font-priority") == "critical" or (bool(self.stack) and self.stack[-1][1])
        skip = tag in {"script", "style", "noscript"} or (bool(self.stack) and self.stack[-1][2])
        if not skip:
            for key in ("title", "aria-label", "data-citation"):
                if key in attrs:
                    self.page.append(attrs[key])
                    if critical:
                        self.critical.append(attrs[key])
        if tag not in VOID:
            self.stack.append((tag, critical, skip))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        if self.stack and not self.stack[-1][2]:
            self.page.append(data)
            if self.stack[-1][1]:
                self.critical.append(data)


def strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, (str, int, float)):
        yield str(value)


def characters(texts):
    return {ord(c) for text in texts for c in unicodedata.normalize("NFC", text) if not unicodedata.category(c).startswith("C")}


def collect(rendered):
    page, critical = [], []
    for path in sorted(rendered.rglob("*.html")):
        parser = TextCollector()
        parser.feed(path.read_text())
        page.extend(parser.page)
        critical.extend(parser.critical)
    dynamic_path = ROOT / ".font-render/dynamic-text.json"
    if dynamic_path.exists():
        dynamic = json.loads(dynamic_path.read_text())
        critical.extend(dynamic["critical"])
        page.extend(dynamic["page"])
    # Recorded language, memory, and all three tasks are covered before playback.
    for path in sorted((ROOT / "public/data").glob("*.json")):
        critical.extend(strings(json.loads(path.read_text())))
    a = characters(critical)
    return a, characters(page) | a


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cmap(path):
    with TTFont(path, lazy=True) as font:
        return set(font.getBestCmap() or {})


def sources():
    yield "display", "Libre Baskerville", "normal", "400 700", True, [ROOT / "public/fonts/LibreBaskerville-VariableFont_wght.woff2"]
    yield "display-italic", "Libre Baskerville", "italic", "400 700", True, [ROOT / "public/fonts/LibreBaskerville-Italic-VariableFont_wght.woff2"]
    yield "sans", "Source Sans 3", "normal", "200 900", True, [ROOT / "public/fonts/SourceSans3VF-Upright.ttf.woff2"]
    yield "sans-italic", "Source Sans 3", "italic", "200 900", False, [ROOT / "public/fonts/SourceSans3VF-Italic.ttf.woff2"]
    for style in ("normal", "italic"):
        files = list((ROOT / "node_modules/@fontsource-variable/source-serif-4/files").glob(f"source-serif-4-*-opsz-{style}.woff2"))
        files.sort(key=lambda p: (f"-latin-opsz-{style}." not in p.name, "-latin-ext-" not in p.name, p.name))
        yield f"serif-{style}", "Source Serif 4 Variable", style, "200 900", True, files
    files = list((ROOT / "node_modules/@fontsource-variable/jetbrains-mono/files").glob("jetbrains-mono-*-wght-normal.woff2"))
    files.sort(key=lambda p: ("-latin-wght-normal." not in p.name, "-latin-ext-" not in p.name, p.name))
    yield "mono", "JetBrains Mono Variable", "normal", "100 800", False, files


def unicode_range(values):
    ranges = []
    for value in sorted(values):
        if ranges and value == ranges[-1][1] + 1:
            ranges[-1][1] = value
        else:
            ranges.append([value, value])
    return ",".join(f"U+{a:X}" if a == b else f"U+{a:X}-{b:X}" for a, b in ranges)


def subset_font(source, required):
    options = subset.Options()
    options.flavor = "woff2"
    options.layout_features = ["*"]
    options.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14]  # Copyright and license notices.
    options.recalc_timestamp = False
    options.notdef_outline = True
    options.recommended_glyphs = True
    font = subset.load_font(str(source), options)
    sub = subset.Subsetter(options=options)
    sub.populate(unicodes=required)
    sub.subset(font)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "subset.woff2"
        subset.save_font(font, str(path), options)
        data = path.read_bytes()
        if not required <= cmap(path):
            raise RuntimeError(f"Generated subset lost glyphs: {source}")
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rendered", type=Path, default=ROOT / ".font-render")
    parser.add_argument("--check-rendered", type=Path)
    args = parser.parse_args()
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    critical, page = collect(args.check_rendered or args.rendered)
    if not critical:
        raise RuntimeError("No marked critical text in rendered HTML")
    plans, source_hashes = [], {}
    for slug, family, style, weight, is_critical, files in sources():
        covered = set()
        for index, source in enumerate(files):
            source_hashes[str(source.relative_to(ROOT))] = digest(source.read_bytes())
            supported = cmap(source) - covered
            covered |= supported
            first = supported & critical if is_critical else set()
            for tier, required in (("critical", first), ("page", (supported & page) - first)):
                if required:
                    plans.append((f"{slug}-{index}-{tier}", family, style, weight, tier, source, required))
    key = digest(json.dumps({"script": digest(Path(__file__).read_bytes()), "fonttools": fonttools_version, "sources": source_hashes, "critical": sorted(critical), "page": sorted(page)}, sort_keys=True).encode())
    manifest_path = OUTPUT / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    current = old.get("key") == key and all((OUTPUT / f["file"]).is_file() and digest((OUTPUT / f["file"]).read_bytes()) == f["sha256"] for f in old.get("fonts", []))
    if args.check_rendered:
        if not current:
            raise RuntimeError("Rendered text is not covered by the current font inputs; run npm run fonts")
        print("Verified generated font coverage against the final rendered page and Demo data.")
        return
    if current:
        print("Font subsets already current (rendered text, source fonts, generator, and output hashes checked).")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fonts = []
    for slug, family, style, weight, tier, source, required in plans:
        data = subset_font(source, required)
        sha = digest(data)
        file = f"{slug}.{sha[:12]}.woff2"
        (OUTPUT / file).write_bytes(data)
        fonts.append({"family": family, "style": style, "weight": weight, "tier": tier, "file": file, "range": unicode_range(required), "text": "".join(chr(c) for c in sorted(required)), "bytes": len(data), "sha256": sha})
    manifest = {"key": key, "criticalCharacters": len(critical), "pageCharacters": len(page), "fonts": fonts}
    temporary = OUTPUT / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(manifest_path)
    keep = {f["file"] for f in fonts}
    for path in OUTPUT.glob("*.woff2"):
        if path.name not in keep:
            path.unlink()
    print("Generated font subsets: " + ", ".join(f"{tier} {sum(f['bytes'] for f in fonts if f['tier'] == tier) / 1024:.1f} KiB" for tier in ("critical", "page")))


if __name__ == "__main__":
    main()
