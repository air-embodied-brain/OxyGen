import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import sharp from 'sharp';
const html=readFileSync(new URL('../dist/index.html',import.meta.url),'utf8');
const meta = name => html.match(new RegExp(`<meta (?:name|property)="${name}" content="([^"]*)"`))?.[1];

test('social metadata points at the current image and declares its real dimensions',async()=>{
  const png=readFileSync(new URL('../public/social-card.png',import.meta.url));
  const dimensions=await sharp(png).metadata();
  assert.equal(dimensions.width,1200);assert.equal(dimensions.height,630);
  const hash=createHash('sha256').update(png).digest('hex').slice(0,12);
  assert.equal(new URL(meta('og:image')).searchParams.get('v'),hash);
  assert.equal(meta('twitter:image'),meta('og:image'));
  assert.equal(meta('og:image:width'),'1200');assert.equal(meta('og:image:height'),'630');
  assert.equal(meta('og:image:type'),'image/png');
  assert.ok(meta('og:image:alt'));assert.equal(meta('twitter:image:alt'),meta('og:image:alt'));
  assert.equal(meta('twitter:title'),meta('og:title'));
  assert.equal(meta('twitter:description'),meta('og:description'));
  assert.ok(meta('twitter:description').length<=200);
});

test('raster icons have declared sizes and production HTML contains no acceptance delay',async()=>{
  for(const [file,size] of [['favicon-32.png',32],['apple-touch-icon.png',180]]){
    const m=await sharp(new URL('../public/'+file,import.meta.url).pathname).metadata();
    assert.equal(m.width,size);assert.equal(m.height,size);
  }
  assert.ok(!html.includes('setTimeout(resolve, 5000)'));
  assert.ok(!/<source[^>]+\bsrc=/.test(html));
});
