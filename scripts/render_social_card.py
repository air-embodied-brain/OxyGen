"""Render the social card as font outlines; no browser/system-font dependency.

Text outlines use the site's bundled SIL OFL fonts. The Node entry point supplies
current page metadata, measured results, and palette, then rasterizes this SVG.
"""
import json
import sys
from functools import lru_cache
from html import escape
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen

ROOT = Path(__file__).resolve().parents[1]
DATA = json.load(sys.stdin)
P = DATA['palette']
PARTS = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">', '<!-- Text outlines: Libre Baskerville, Source Serif 4, Source Sans 3. SIL Open Font License 1.1. -->']
PATHS = {
    'display': ROOT / 'public/fonts/LibreBaskerville-VariableFont_wght.woff2',
    'sans': ROOT / 'public/fonts/SourceSans3VF-Upright.ttf.woff2',
    'serif-italic': ROOT / 'node_modules/@fontsource-variable/source-serif-4/files/source-serif-4-latin-opsz-italic.woff2',
}


@lru_cache(None)
def font(key, weight, optical):
    face = TTFont(PATHS[key])
    glyphs = face.getGlyphSet(location={'wght': weight, 'opsz': optical})
    return face, glyphs, face.getBestCmap(), face['head'].unitsPerEm


def width(text, size, family='sans', weight=400, tracking=0):
    _, glyphs, cmap, units = font(family, weight, size)
    return sum(glyphs[cmap.get(ord(c), cmap[32])].width for c in text) * size / units + max(0, len(text)-1)*tracking


def text(value, x, y, size, family='sans', weight=400, color='ink', tracking=0):
    _, glyphs, cmap, units = font(family, weight, size)
    scale = size / units
    offset = 0
    paths = []
    for char in value:
        if ord(char) not in cmap:
            raise ValueError(f"Font {family} does not cover {char!r}")
        glyph = glyphs[cmap[ord(char)]]
        pen = SVGPathPen(glyphs)
        glyph.draw(pen)
        if pen.getCommands():
            paths.append(f'<path transform="translate({offset:.3f} 0)" d="{pen.getCommands()}"/>')
        offset += glyph.width + tracking / scale
    PARTS.append(f'<g aria-label="{escape(value, quote=True)}" fill="{P.get(color,color)}" transform="translate({x} {y}) scale({scale} {-scale})">'+''.join(paths)+'</g>')


def rect(x,y,w,h,color,opacity=1):
    PARTS.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{P.get(color,color)}" opacity="{opacity}"/>')


def rule(x,y,w,color='hair'):
    rect(x,y,w,1,color)


rect(0,0,1200,630,'paper')
text(DATA['site']['conference'].upper()+' · '+DATA['site']['track'].upper(),60,42,15,weight=600,tracking=1.3)
arxiv='arXiv '+DATA['site']['arxiv']
text(arxiv,1140-width(arxiv,15),42,15,color='muted')
rule(60,62,1080,'ink')
text(DATA['site']['name'],56,188,110,'display',700)
lines=[]
for word in DATA['site']['paperTitle'].split():
    if lines and width(lines[-1]+' '+word,28,'serif-italic')<650:
        lines[-1]+=' '+word
    else:
        lines.append(word)
for i,line in enumerate(lines):
    text(line,60,243+i*35,28,'serif-italic',color='ink-soft')
text(DATA['site']['tagline'],60,367,25,weight=600,color='blue')

# Schematic, matching the site's blue action / red language palette.
rect(784,132,356,259,'paper-deep')
text('ISOLATED',806,164,13,weight=600,tracking=1,color='muted')
rect(806,190,55,8,'blue'); rect(866,190,250,8,'red')
text('language blocks the next action',806,222,15,color='muted')
text(DATA['site']['name'],806,268,17,weight=600,color='blue')
for i in range(4):
    x=806+i*79
    rect(x,304,32,8,'blue')
    for row in range(4):
        rect(x+36,289+row*9,35,6,'red',1-row*.18)
text('Shared KV. Batched language.',806,366,16,color='ink-soft')

rule(60,419,1080,'ink')
values=[(DATA['speedup']+'×','faster in both outputs'),(DATA['action'],'action frequency'),(DATA['language'],'language throughput')]
for i,(value,label) in enumerate(values):
    x=60+i*370
    if i: rect(x-24,442,1,103,'hair')
    if ' → ' in value:
        before, after = value.split(' → ')
        text(before,x,497,28,'display',color='muted')
        arrow_x = x + width(before,28,'display') + 12
        PARTS.append(f'<path d="M{arrow_x} 487h24m-7-6 7 6-7 6" fill="none" stroke="{P["muted"]}" stroke-width="1.5"/>')
        text(after,arrow_x+37,501,43,'display')
    else:
        text(value,x,501,47,'display')
    text(label,x,540,18,color='muted')
text('π0.5 · LIBERO · RTX 4090 · 30-token language requests',60,594,17,color='muted')
text('air-embodied-brain.github.io/OxyGen',806,594,14,color='muted')
PARTS.append('</svg>')
sys.stdout.write(''.join(PARTS))
