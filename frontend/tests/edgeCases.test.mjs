import assert from "node:assert/strict";
import { test } from "node:test";
import { EDGE_CASE_LABELS, keepOffered, toggleEdgeCase } from "../lib/edgeCases.ts";

test("switching environments drops edge cases the new one does not offer", () => {
  assert.deepEqual(
    keepOffered(["missing_deps", "recovery"], ["boundary_conditions", "conflicting_state", "recovery"]),
    ["recovery"],
  );
});

test("an environment with no offer list keeps no edge cases", () => {
  assert.deepEqual(keepOffered(["recovery"], undefined), []);
});

test("toggling adds an edge case once and removes it on the second toggle", () => {
  const once = toggleEdgeCase([], "recovery");
  assert.deepEqual(once, ["recovery"]);
  assert.deepEqual(toggleEdgeCase(once, "recovery"), []);
});

test("toggling keeps the order edge cases were picked in", () => {
  assert.deepEqual(toggleEdgeCase(["recovery"], "boundary_conditions"), ["recovery", "boundary_conditions"]);
});

test("every edge case the backend offers has a label", () => {
  for (const name of ["boundary_conditions", "missing_deps", "conflicting_state", "recovery"]) {
    assert.ok(EDGE_CASE_LABELS[name], name);
  }
});
