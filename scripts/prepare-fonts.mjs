import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import ts from "typescript";

const root = fileURLToPath(new URL("../", import.meta.url));
process.chdir(root);
function run(command, args, env = process.env) {
  const result = spawnSync(command, args, { stdio: "inherit", env });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

// Render the actual Astro templates before collecting text. No checked-in
// character list, generated fonts, or browser installation is needed by CI.
run(process.execPath, ["node_modules/astro/astro.js", "build", "--outDir", ".font-render"], {
  ...process.env, OXYGEN_FONT_SCAN: "1",
});

const dynamic = { critical: [], page: [] };
for (const dir of ["src/scripts", "src/data"]) {
  for (const file of readdirSync(dir).filter(file => file.endsWith(".ts"))) {
    const filename = path.join(dir, file);
    const source = ts.createSourceFile(filename, readFileSync(filename, "utf8"), ts.ScriptTarget.Latest);
    const strings = [];
    function visit(node) {
      if (ts.isStringLiteralLike(node) || ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) strings.push(node.text);
      ts.forEachChild(node, visit);
    }
    visit(source);
    dynamic.page.push(...strings);
    if (file.startsWith("demo") || file === "site.ts") dynamic.critical.push(...strings);
  }
}
mkdirSync(".font-render", { recursive: true });
writeFileSync(".font-render/dynamic-text.json", JSON.stringify(dynamic));
run("uv", ["run", "--locked", "python", "scripts/build_font_subsets.py", "--rendered", ".font-render"]);
