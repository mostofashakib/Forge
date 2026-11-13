import assert from "node:assert/strict";
import { test } from "node:test";
import {
  BAND_LABELS,
  MAX_ATTEMPTS,
  histogramRows,
  isActive,
  targetModelProblem,
} from "../lib/calibration.ts";

test("a target is provider:model with a provider calibration can drive", () => {
  assert.equal(targetModelProblem("ollama:qwen3.8:27b"), null);
  assert.equal(targetModelProblem("vllm:my-model"), null);
  assert.equal(targetModelProblem(" openai:gpt-5 "), null);
});

test("a target with no model, an unknown provider, or nothing at all is refused", () => {
  for (const bad of ["", "ollama:", "qwen3.8:27b", "gemini:pro", "random"]) {
    assert.ok(targetModelProblem(bad), `expected ${JSON.stringify(bad)} to be refused`);
  }
});

test("attempts are capped at five", () => {
  assert.equal(MAX_ATTEMPTS, 5);
});

test("histogram rows run from 0% to 100% with each share of the tasks", () => {
  const rows = histogramRows({ "1.0": 1, "0.0": 2, "0.5": 1 });

  assert.deepEqual(rows.map((r) => r.label), ["0%", "50%", "100%"]);
  assert.deepEqual(rows.map((r) => r.count), [2, 1, 1]);
  assert.equal(rows[0].share, 0.5);
});

test("an empty histogram has no rows and no division by zero", () => {
  assert.deepEqual(histogramRows({}), []);
  assert.deepEqual(histogramRows({ "0.0": 0, "1.0": 0 }).map((r) => r.share), [0, 0]);
});

test("only queued and running runs are polled", () => {
  assert.equal(isActive("queued"), true);
  assert.equal(isActive("running"), true);
  assert.equal(isActive("complete"), false);
  assert.equal(isActive("failed"), false);
});

test("every band has a label", () => {
  assert.deepEqual(Object.keys(BAND_LABELS).sort(), ["frontier", "suspect", "too_easy", "unsolved"]);
});
