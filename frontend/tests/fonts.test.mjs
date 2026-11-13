import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const APP = resolve(dirname(fileURLToPath(import.meta.url)), "../app");

function sourceFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    return /\.(tsx?|jsx?)$/.test(entry.name) ? [path] : [];
  });
}

test("no page loads fonts from Google at build time", () => {
  const remote = sourceFiles(APP).filter((file) =>
    readFileSync(file, "utf8").includes("next/font/google"),
  );
  assert.deepEqual(remote, []);
});

test("every local font file the layout names is in the repo", () => {
  const layout = readFileSync(join(APP, "layout.tsx"), "utf8");
  const paths = [...layout.matchAll(/path:\s*"([^"]+\.woff2)"/g)].map((m) => m[1]);
  assert.ok(paths.length > 0, "the layout names no local font files");
  for (const path of paths) {
    assert.ok(existsSync(join(APP, path)), `missing font file ${path}`);
  }
});
