import assert from "node:assert/strict";
import { test } from "node:test";
import { sandboxHost, sandboxUpstreamUrl } from "../lib/sandboxProxy.ts";

test("builds the upstream URL from the sandbox host, port, path, and query", () => {
  assert.equal(
    sandboxUpstreamUrl("10.0.0.5", 50646, ["forge", "reset"], "?seed=3&x=a%20b"),
    "http://10.0.0.5:50646/forge/reset?seed=3&x=a%20b",
  );
});

test("an empty path reaches the root", () => {
  assert.equal(sandboxUpstreamUrl("127.0.0.1", 8123, [], ""), "http://127.0.0.1:8123/");
  assert.equal(sandboxUpstreamUrl("127.0.0.1", 8123, undefined, ""), "http://127.0.0.1:8123/");
});

test("encodes segments so a slash inside one cannot add a path level", () => {
  assert.equal(sandboxUpstreamUrl("127.0.0.1", 8123, ["a/b", "c d"], ""), "http://127.0.0.1:8123/a%2Fb/c%20d");
});

test("rejects a dot-dot or dot segment", () => {
  assert.throws(() => sandboxUpstreamUrl("127.0.0.1", 8123, ["ui", "..", "admin"], ""), /path segment/);
  assert.throws(() => sandboxUpstreamUrl("127.0.0.1", 8123, ["."], ""), /path segment/);
});

test("rejects an invalid port", () => {
  for (const port of [0, -1, 65536, 1.5, Number.NaN]) {
    assert.throws(() => sandboxUpstreamUrl("127.0.0.1", port, [], ""), /port/);
  }
});

test("the sandbox host comes from FORGE_SANDBOX_HOST, falling back to loopback when unset or blank", () => {
  assert.equal(sandboxHost({ FORGE_SANDBOX_HOST: "10.0.0.5" }), "10.0.0.5");
  assert.equal(sandboxHost({ FORGE_SANDBOX_HOST: "   " }), "127.0.0.1");
  assert.equal(sandboxHost({}), "127.0.0.1");
});
