"use client";
import { useRef, useState } from "react";
import Link from "next/link";
import { API_BASE, wsBase } from "@/lib/api";

type Phase = "idle" | "running" | "done" | "error";

export default function BenchmarkRunPage() {
  const [domains, setDomains] = useState<string[]>(["email", "project_mgmt"]);
  const [depth, setDepth] = useState(5);
  const [seeds, setSeeds] = useState(5);
  const [outputDir, setOutputDir] = useState("benchmark_results");
  const [phase, setPhase] = useState<Phase>("idle");
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState<{ completed: number; total: number | null }>({
    completed: 0,
    total: null,
  });
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  function toggleDomain(domain: string, checked: boolean) {
    setDomains((prev) => checked ? [...prev, domain] : prev.filter((d) => d !== domain));
  }

  async function handleLaunch() {
    setPhase("running");
    setLogs([]);
    setProgress({ completed: 0, total: null });
    setError(null);

    let run_id = "";
    try {
      const res = await fetch(`${API_BASE}/api/benchmark/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ domains, depth, seeds, output_dir: outputDir }),
      });
      if (!res.ok) {
        const msg = await res.text();
        setPhase("error");
        setError(`Failed to create run: ${msg}`);
        return;
      }
      ({ run_id } = await res.json());
    } catch (e) {
      setPhase("error");
      setError("Could not reach the backend — is it running?");
      return;
    }

    const ws = new WebSocket(`${wsBase()}/api/benchmark/ws/progress/${run_id}`);

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data) as Record<string, unknown>;
      if (msg.total != null) setProgress((p) => ({ ...p, total: msg.total as number }));
      if (msg.progress != null) setProgress((p) => ({ ...p, completed: msg.progress as number }));
      if (msg.log) {
        const line = msg.log as string;
        setLogs((prev) => [...prev.slice(-998), line]);
        requestAnimationFrame(() => {
          if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
        });
      }
      if (msg.done) { setPhase("done"); ws.close(); }
      if (msg.error) { setPhase("error"); setError(msg.error as string); ws.close(); }
    };
    ws.onerror = () => {
      if (phase !== "done") { setPhase("error"); setError("WebSocket connection failed — is the backend running?"); }
    };
  }

  const pct = progress.total ? Math.round((progress.completed / progress.total) * 100) : 0;
  const isRunning = phase === "running";

  return (
    <div className="benchmark-run">
      <header className="benchmark-run__hero">
        <div className="benchmark-run__hero-copy">
          <span className="benchmark-run__eyebrow">Evaluation protocol / 01</span>
          <h1>RUN THE<br /><em>GAUNTLET.</em></h1>
          <p>Collect episodes across the task suite and turn raw trajectories into quality signals.</p>
        </div>
        <div className="benchmark-run__readout" aria-label="Current run configuration">
          <div><span>Domains</span><strong>{String(domains.length).padStart(2, "0")}</strong></div>
          <div><span>Difficulty</span><strong>0{depth}</strong></div>
          <div><span>Seeds / task</span><strong>{String(seeds).padStart(2, "0")}</strong></div>
          <div className={`benchmark-run__state benchmark-run__state--${phase}`}>
            <span>System state</span><strong><i />{phase}</strong>
          </div>
        </div>
      </header>

      <div className="benchmark-workbench">
        <section className="benchmark-config">
          <div className="benchmark-panel__heading">
            <div><span>01</span><h2>Run configuration</h2></div>
            <p>Define the evaluation envelope</p>
          </div>

          <div className="benchmark-field benchmark-field--domains">
            <div className="benchmark-field__label">
              <span>Target domains</span>
              <small>{domains.length} selected</small>
            </div>
            <div className="benchmark-domain-grid">
              {(["email", "project_mgmt"] as const).map((d) => (
                <label key={d} className="benchmark-domain">
                  <input
                    type="checkbox"
                    checked={domains.includes(d)}
                    disabled={isRunning}
                    onChange={(e) => toggleDomain(d, e.target.checked)}
                  />
                  <span className="benchmark-domain__check" aria-hidden="true">✓</span>
                  <span>
                    <strong>{d === "project_mgmt" ? "Project management" : "Email"}</strong>
                    <small>{d === "project_mgmt" ? "Planning · delivery" : "Inbox · compose"}</small>
                  </span>
                </label>
              ))}
            </div>
          </div>

          <div className="benchmark-field">
            <div className="benchmark-field__label">
              <span>Max difficulty</span>
              <strong>{depth} / 5</strong>
            </div>
            <input
              type="range" min={1} max={5} value={depth} disabled={isRunning}
              onChange={(e) => setDepth(Number(e.target.value))}
              className="benchmark-range"
            />
            <div className="benchmark-range__legend">
              <span>01 / Foundation</span>
              <span>05 / Full stress</span>
            </div>
          </div>

          <div className="benchmark-field-row">
            <label className="benchmark-field">
              <span className="benchmark-field__label"><span>Seeds per task</span></span>
              <input
                type="number" min={1} max={20} value={seeds} disabled={isRunning}
                onChange={(e) => setSeeds(Number(e.target.value))}
                className="benchmark-input"
              />
            </label>
            <label className="benchmark-field">
              <span className="benchmark-field__label"><span>Output directory</span></span>
              <input
                type="text" value={outputDir} disabled={isRunning}
                onChange={(e) => setOutputDir(e.target.value)}
                className="benchmark-input benchmark-input--mono"
              />
            </label>
          </div>

          {phase === "idle" && (
            <button
              onClick={handleLaunch}
              disabled={domains.length === 0}
              className="benchmark-launch"
            >
              <span>Launch benchmark run</span><span aria-hidden="true">↗</span>
            </button>
          )}
          {isRunning && (
            <div className="benchmark-active-state">
              <span className="benchmark-active-state__pulse" />
              <span><strong>Evaluation in progress</strong><small>Worker stream is connected</small></span>
            </div>
          )}
          {phase === "done" && (
            <p className="benchmark-complete-state">Run complete <span>✓</span></p>
          )}
        </section>

        <section className="benchmark-console">
          <div className="benchmark-console__bar">
            <div><i /><i /><i /></div>
            <span>worker://benchmark/output</span>
            {isRunning && (
              <span className="benchmark-console__live">
                <i /> live
              </span>
            )}
          </div>
          <div
            ref={logRef}
            className="benchmark-console__output scrollbar-thin"
          >
            {logs.length === 0 ? (
              <div className="benchmark-console__idle">
                <span>&gt;_</span>
                <p>System armed.<br />Waiting for worker output<span className="animate-pulse">_</span></p>
              </div>
            ) : (
              logs.map((line, i) => (
                <p key={i}><span>{String(i + 1).padStart(3, "0")}</span>{line}</p>
              ))
            )}
          </div>
          {phase !== "idle" && (
            <div className="benchmark-console__progress">
              <div>
                <span>
                  {progress.completed} / {progress.total ?? "?"} episodes
                </span>
                <strong>
                  {phase === "done" ? "100" : pct}%
                </strong>
              </div>
              <div className="benchmark-console__track">
                <div
                  style={{ width: `${phase === "done" ? 100 : pct}%` }}
                />
              </div>
            </div>
          )}
        </section>
      </div>

      {phase === "done" && (
        <div className="benchmark-notice benchmark-notice--success">
          <div>
            <p>Benchmark run complete</p>
            <span>
              Results saved to <code className="font-mono">{outputDir}/</code>
            </span>
          </div>
          <Link href="/benchmark/report">
            View Report →
          </Link>
        </div>
      )}

      {phase === "error" && error && (
        <div className="benchmark-notice benchmark-notice--error">
          <div><p>Run failed</p><span>{error}</span></div>
          <button
            onClick={() => setPhase("idle")}
          >
            Reset run →
          </button>
        </div>
      )}
    </div>
  );
}
