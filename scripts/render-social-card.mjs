import { readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import sharp from 'sharp';

process.chdir(fileURLToPath(new URL('../', import.meta.url)));
async function dataModule(path) {
  const { outputText } = ts.transpileModule(readFileSync(path, 'utf8'), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } });
  return import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
}
const { site } = await dataModule('src/data/site.ts');
const { tradeoff } = await dataModule('src/data/paper-figures.ts');
const peak = tradeoff.libero.rtx4090.k1.at(-1);
const baseline = tradeoff.libero.rtx4090.baseline.at(-1);
const css = readFileSync('src/styles/tokens.css', 'utf8');
const palette = Object.fromEntries([...css.split(':root[data-theme="dark"]')[0].matchAll(/--([\w-]+):\s*(#[\da-f]+);/g)].map(match => [match[1], match[2]]));
const result = spawnSync('uv', ['run', '--locked', 'python', 'scripts/render_social_card.py'], {
  input: JSON.stringify({ site, palette, speedup: (peak[1]/baseline[1]).toFixed(1), action: `${Math.round(baseline[1])} → ${Math.round(peak[1])} Hz`, language: `${Math.round(baseline[2])} → ${Math.round(peak[2])} tok/s` }),
  maxBuffer: 10 * 1024 * 1024,
});
if (result.status !== 0) throw new Error(result.stderr.toString());
const png = await sharp(result.stdout).png({ compressionLevel: 9 }).toBuffer();
writeFileSync('public/social-card.png', png);
const icon = readFileSync('public/favicon.svg');
await sharp(icon).resize(32,32).png().toFile('public/favicon-32.png');
await sharp(icon).resize(142,142).extend({top:19,bottom:19,left:19,right:19,background:palette.paper}).flatten({background:palette.paper}).png().toFile('public/apple-touch-icon.png');
console.log(`Rendered social-card.png (1200×630, ${(png.length/1024).toFixed(1)} KiB), favicon-32.png, and apple-touch-icon.png.`);
