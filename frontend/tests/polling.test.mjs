import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import { test } from "node:test";
import { startPolling } from "../lib/polling.ts";

test("polls only after the preceding request finishes", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let calls = 0;
  let finish;
  const pending = new Promise((resolve) => { finish = resolve; });
  const stop = startPolling(async () => {
    calls += 1;
    await pending;
  }, 5000, assert.fail);

  t.mock.timers.tick(0);
  assert.equal(calls, 1);
  t.mock.timers.tick(15000);
  assert.equal(calls, 1);
  finish();
  await setImmediate();
  t.mock.timers.tick(4999);
  assert.equal(calls, 1);
  t.mock.timers.tick(1);
  assert.equal(calls, 2);
  stop();
});

test("cleanup cancels startup and aborts requests without scheduling another poll", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let calls = 0;
  const stopBeforeStart = startPolling(async () => { calls += 1; }, 10, assert.fail);
  stopBeforeStart();
  t.mock.timers.tick(20);
  assert.equal(calls, 0);

  let signal;
  let finish;
  const stop = startPolling(async (requestSignal) => {
    calls += 1;
    signal = requestSignal;
    await new Promise((resolve) => { finish = resolve; });
  }, 10, assert.fail);
  t.mock.timers.tick(0);
  stop();
  assert.equal(signal.aborted, true);
  finish();
  await setImmediate();
  t.mock.timers.tick(100);
  assert.equal(calls, 1);
});

test("reports request failures and continues polling", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const errors = [];
  let calls = 0;
  const stop = startPolling(async () => {
    calls += 1;
    throw new Error("offline");
  }, 10, (error) => errors.push(error.message));
  t.mock.timers.tick(0);
  await setImmediate();
  assert.deepEqual(errors, ["offline"]);
  t.mock.timers.tick(10);
  await setImmediate();
  assert.equal(calls, 2);
  stop();
});
