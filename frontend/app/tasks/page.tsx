"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { wsBase } from "@/lib/api";
import {
  apiJson,
  formatDate,
  type BatchSummary,
  type Settings,
  type TargetEnvironment,
} from "@/lib/taskFactory";

type Phase = "idle" | "running" | "done" | "error";

const MAX_TASKS = 20_000;

/** The task count typed into the form, or an error saying why it is not one. */
function parseCount(text: string): { count: number; error: string | null } {
  const count = Number(text);
  if (!text.trim() || !Number.isInteger(count) || count < 1 || count > MAX_TASKS) {
    return { count: 0, error: `Enter a whole number from 1 to ${MAX_TASKS.toLocaleString()}.` };
  }
  return { count, error: null };
}

const STAGES = [
  { key: "taxonomy", label: "Taxonomy" },
  { key: "writing", label: "Writer" },
  { key: "static", label: "Static check" },
  { key: "pass_k", label: "Golden pass^k" },
  { key: "review", label: "LLM review" },
  { key: "registry", label: "Registry" },
];

const FAMILY_LABEL: Record<TargetEnvironment["family"], string> = {
  state: "Seeded state",
  cli: "Shell",
  browser: "Browser pages",
};

export default function TaskFactoryPage() {
  const [envs, setEnvs] = useState<TargetEnvironment[]>([]);
  const [envsLoading, setEnvsLoading] = useState(true);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [envName, setEnvName] = useState("");
  const [countText, setCountText] = useState("20");
  const { count, error: countError } = parseCount(countText);
  const [k, setK] = useState(3);
  const [phase, setPhase] = useState<Phase>("idle");
  const [stage, setStage] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [batches, setBatches] = useState<BatchSummary[]>([]);
  const [lastBatchId, setLastBatchId] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const socketRef = useRef<WebSocket | null>(null);

  const loadBatches = useCallback(async (name: string) => {
    if (!name) return setBatches([]);
    try {
      setBatches(await apiJson<BatchSummary[]>(`/api/task-factory/batches?env_name=${encodeURIComponent(name)}`));
    } catch {
      setBatches([]);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      apiJson<TargetEnvironment[]>("/api/task-factory/environments"),
      apiJson<Settings>("/api/settings"),
    ])
      .then(([targets, current]) => {
        if (cancelled) return;
        setEnvs(targets);
        setSettings(current);
        const first = targets.find((t) => t.ready) ?? targets[0];
        if (first) setEnvName(first.name);
      })
      .catch((err: Error) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setEnvsLoading(false));
    return () => {
      cancelled = true;
      socketRef.current?.close();
    };
  }, []);

  useEffect(() => {
    const t = window.setTimeout(() => loadBatches(envName), 0);
    return () => window.clearTimeout(t);
  }, [envName, loadBatches]);

  const selected = envs.find((e) => e.name === envName);
  const validator = settings?.task_validator;
  const canStart = Boolean(selected?.ready && validator?.configured && !countError) && phase !== "running";

  function appendLog(line: string) {
    setLogs((prev) => [...prev.slice(-498), line]);
    requestAnimationFrame(() => {
      if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    });
  }

  async function handleStart() {
    setPhase("running");
    setStage(null);
    setLogs([]);
    setError(null);
    let batchId: string;
    try {
      ({ batch_id: batchId } = await apiJson<{ batch_id: string }>("/api/task-factory/batches", {
        method: "POST",
        body: JSON.stringify({ env_name: envName, count, k }),
      }));
    } catch (err) {
      setPhase("error");
      setError((err as Error).message);
      return;
    }
    setLastBatchId(batchId);
    loadBatches(envName);
    const ws = new WebSocket(`${wsBase()}/api/task-factory/ws/progress/${batchId}`);
    socketRef.current = ws;
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data) as Record<string, unknown>;
      const known = STAGES.some((s) => s.key === msg.stage);
      if (known && !msg.rejected) setStage(msg.stage as string);
      if (typeof msg.log === "string") appendLog(msg.log);
      if (msg.done) {
        setStage("registry");
        setPhase("done");
        loadBatches(envName);
        ws.close();
      }
      if (msg.error) {
        setPhase("error");
        setError(String(msg.error));
        loadBatches(envName);
        ws.close();
      }
    };
    ws.onerror = () => {
      setPhase((p) => (p === "running" ? "error" : p));
      setError((prev) => prev ?? "Lost the progress stream. The batch keeps running. Refresh the versions list.");
    };
  }

  const stageIndex = STAGES.findIndex((s) => s.key === stage);

  return (
    <div className="benchmark-run tasks-page">
      <header className="benchmark-run__hero">
        <div className="benchmark-run__hero-copy">
          <span className="benchmark-run__eyebrow">Task factory / 01</span>
          <h1>FORGE THE<br /><em>WORK.</em></h1>
          <p>
            Build a taxonomy, write executable tasks with golden solutions, prove each one passes every
            run, and have a model from another family judge it. Every batch is saved as a dated version.
          </p>
        </div>
        <div className="benchmark-run__readout" aria-label="Batch configuration">
          <div><span>Tasks</span><strong>{count ? count.toLocaleString() : "—"}</strong></div>
          <div><span>pass^k</span><strong>{String(k).padStart(2, "0")}</strong></div>
          <div><span>Versions</span><strong>{String(batches.filter((b) => b.version).length).padStart(2, "0")}</strong></div>
          <div className={`benchmark-run__state benchmark-run__state--${phase}`}>
            <span>System state</span><strong><i />{phase}</strong>
          </div>
        </div>
      </header>

      <div className="benchmark-workbench">
        <section className="benchmark-config">
          <div className="benchmark-panel__heading">
            <div><span>01</span><h2>New batch</h2></div>
            <p>Pick an environment and a size</p>
          </div>

          <div className="benchmark-field">
            <div className="benchmark-field__label">
              <span>Environment</span>
              {selected && <small>{FAMILY_LABEL[selected.family]}</small>}
            </div>
            {envsLoading ? (
              <p className="benchmark-domain-empty">Loading environments…</p>
            ) : envs.length === 0 ? (
              <p className="benchmark-domain-empty">
                No environments yet. <Link href="/environments/new">Create one</Link> first.
              </p>
            ) : (
              <div className="benchmark-domain-grid">
                {envs.map((env) => (
                  <label key={env.name} className={`benchmark-domain ${env.ready ? "" : "tasks-domain--idle"}`}>
                    <input
                      type="radio"
                      name="environment"
                      checked={envName === env.name}
                      disabled={phase === "running"}
                      onChange={() => setEnvName(env.name)}
                    />
                    <span className="benchmark-domain__check" aria-hidden="true">✓</span>
                    <span>
                      <strong>{env.name}</strong>
                      <small>{env.env_type.replace("_", "-")} · {env.ready ? "ready" : "stopped"}</small>
                    </span>
                  </label>
                ))}
              </div>
            )}
            {selected && !selected.ready && <p className="tasks-hint tasks-hint--warn">{selected.reason}</p>}
          </div>

          <label className="benchmark-field">
            <span className="benchmark-field__label">
              <span>Number of tasks</span>
              <small>1 to {MAX_TASKS.toLocaleString()}</small>
            </span>
            <input
              type="number" inputMode="numeric" min={1} max={MAX_TASKS} step={1}
              value={countText} disabled={phase === "running"}
              onChange={(e) => setCountText(e.target.value)}
              className="benchmark-input benchmark-input--mono"
            />
            {countError && <p className="tasks-hint tasks-hint--warn">{countError}</p>}
          </label>

          <div className="benchmark-field">
            <div className="benchmark-field__label">
              <span>Golden solution runs (pass^k)</span>
              <strong>{k}</strong>
            </div>
            <input
              type="range" min={1} max={10} value={k} disabled={phase === "running"}
              onChange={(e) => setK(Number(e.target.value))}
              className="benchmark-range"
            />
            <div className="benchmark-range__legend"><span>1 run</span><span>10 runs, all must pass</span></div>
          </div>

          <div className="benchmark-field tasks-models">
            <div>
              <span>Writer</span>
              <strong>{settings?.writer.model ?? "—"}</strong>
              <small>{settings?.writer.family} family</small>
            </div>
            <div>
              <span>Validator</span>
              <strong>{validator?.model ?? "Not set"}</strong>
              <small>{validator?.configured ? `${validator.family} family` : "different family required"}</small>
            </div>
            {validator && !validator.configured && (
              <p className="tasks-hint tasks-hint--warn">
                {validator.error} <Link href="/settings">Open settings</Link>
              </p>
            )}
          </div>

          {phase === "running" ? (
            <div className="benchmark-active-state">
              <span className="benchmark-active-state__pulse" />
              <span><strong>Creating tasks</strong><small>Agent runs on {envName} wait between rounds</small></span>
            </div>
          ) : (
            <button onClick={handleStart} disabled={!canStart} className="benchmark-launch">
              <span>Create {count ? count.toLocaleString() : ""} tasks</span><span aria-hidden="true">↗</span>
            </button>
          )}
        </section>

        <section className="benchmark-console">
          <div className="benchmark-console__bar">
            <div><i /><i /><i /></div>
            <span>worker://task-factory</span>
            {phase === "running" && <span className="benchmark-console__live"><i /> live</span>}
          </div>
          <ol className="tasks-stages">
            {STAGES.map((s, i) => (
              <li
                key={s.key}
                className={
                  phase === "done" || i < stageIndex ? "tasks-stage--done" : i === stageIndex ? "tasks-stage--active" : ""
                }
              >
                <span>0{i + 1}</span>{s.label}
              </li>
            ))}
          </ol>
          <div ref={logRef} className="benchmark-console__output scrollbar-thin">
            {logs.length === 0 ? (
              <div className="benchmark-console__idle">
                <span>&gt;_</span>
                <p>Factory idle.<br />Waiting for a batch<span className="animate-pulse">_</span></p>
              </div>
            ) : (
              logs.map((line, i) => (
                <p key={i}><span>{String(i + 1).padStart(3, "0")}</span>{line}</p>
              ))
            )}
          </div>
        </section>
      </div>

      {phase === "done" && lastBatchId && (
        <div className="benchmark-notice benchmark-notice--success">
          <div><p>Batch saved</p><span>{logs[logs.length - 1]}</span></div>
          <Link href={`/tasks/${lastBatchId}`}>Open batch →</Link>
        </div>
      )}
      {phase === "error" && error && (
        <div className="benchmark-notice benchmark-notice--error">
          <div><p>Batch failed</p><span>{error}</span></div>
          <button onClick={() => setPhase("idle")}>Dismiss →</button>
        </div>
      )}

      <section className="benchmark-config">
        <div className="benchmark-panel__heading">
          <div><span>02</span><h2>Versions{envName ? ` · ${envName}` : ""}</h2></div>
          <p>Newest first</p>
        </div>
        {batches.length === 0 ? (
          <p className="tasks-empty">No batches for this environment yet.</p>
        ) : (
          <div className="tasks-table-wrap">
            <table className="tasks-table">
              <thead>
                <tr>
                  <th>Version</th><th>Created</th><th>Status</th><th>Tasks</th><th>pass^k</th><th>Validator</th><th />
                </tr>
              </thead>
              <tbody>
                {batches.map((b) => (
                  <tr key={b.id}>
                    <td className="tasks-table__version">{b.version ? `v${b.version}` : "—"}</td>
                    <td>{formatDate(b.created_at)}</td>
                    <td><span className={`tasks-status tasks-status--${b.status}`}>{b.status}</span></td>
                    <td>{b.version ? `${b.delivered} / ${b.requested}` : `— / ${b.requested}`}</td>
                    <td>{b.pass_k}</td>
                    <td className="tasks-table__mono">{b.validator_model}</td>
                    <td><Link href={`/tasks/${b.id}`}>Open →</Link></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
