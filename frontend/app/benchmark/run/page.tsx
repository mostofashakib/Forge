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
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Benchmark Run</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Collect episodes across the task suite and compute environment quality metrics.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 items-start">
        {/* Config panel */}
        <div className="border rounded-lg p-4 space-y-4">
          <h2 className="text-sm font-medium">Configuration</h2>

          <div>
            <p className="text-xs text-muted-foreground mb-2">Domains</p>
            <div className="flex flex-col gap-1.5">
              {(["email", "project_mgmt"] as const).map((d) => (
                <label key={d} className="flex items-center gap-2 text-sm cursor-pointer">
                  <input
                    type="checkbox"
                    checked={domains.includes(d)}
                    disabled={isRunning}
                    onChange={(e) => toggleDomain(d, e.target.checked)}
                    className="rounded"
                  />
                  {d}
                </label>
              ))}
            </div>
          </div>

          <div>
            <div className="flex items-baseline justify-between mb-2">
              <p className="text-xs text-muted-foreground">Max difficulty</p>
              <span className="text-xs font-mono font-medium tabular-nums">{depth} / 5</span>
            </div>
            <input
              type="range" min={1} max={5} value={depth} disabled={isRunning}
              onChange={(e) => setDepth(Number(e.target.value))}
              className="w-full accent-primary"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>1 — easy only</span>
              <span>5 — all tasks</span>
            </div>
            <p className="text-xs text-muted-foreground/70 mt-1.5">
              Includes tasks with difficulty ≤ {depth}
            </p>
          </div>

          <div>
            <p className="text-xs text-muted-foreground mb-2">Seeds per task</p>
            <input
              type="number" min={1} max={20} value={seeds} disabled={isRunning}
              onChange={(e) => setSeeds(Number(e.target.value))}
              className="w-full border rounded-md px-3 py-1.5 text-sm bg-background"
            />
          </div>

          <div>
            <p className="text-xs text-muted-foreground mb-2">Output directory</p>
            <input
              type="text" value={outputDir} disabled={isRunning}
              onChange={(e) => setOutputDir(e.target.value)}
              className="w-full border rounded-md px-3 py-1.5 text-sm bg-background font-mono"
            />
          </div>

          {phase === "idle" && (
            <button
              onClick={handleLaunch}
              disabled={domains.length === 0}
              className="w-full bg-primary text-primary-foreground rounded-lg px-4 py-2 text-sm font-medium hover:bg-primary/90 disabled:opacity-50 transition-colors"
            >
              Launch Benchmark Run
            </button>
          )}
          {isRunning && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse shrink-0" />
              Running…
            </div>
          )}
          {phase === "done" && (
            <p className="text-sm text-green-600 font-medium">Run complete ✓</p>
          )}
        </div>

        {/* Log panel */}
        <div className="border rounded-lg overflow-hidden flex flex-col">
          <div className="px-3 py-2 border-b bg-muted/40 flex items-center justify-between">
            <span className="text-xs font-medium text-muted-foreground">Worker logs</span>
            {isRunning && (
              <span className="flex items-center gap-1 text-xs text-green-600">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                live
              </span>
            )}
          </div>
          <div
            ref={logRef}
            className="flex-1 overflow-y-auto p-3 font-mono text-xs min-h-52 max-h-80 bg-muted/10 space-y-0.5"
          >
            {logs.length === 0 ? (
              <p className="text-muted-foreground italic">Waiting for worker output…</p>
            ) : (
              logs.map((line, i) => (
                <p key={i} className="text-foreground/80 leading-5 whitespace-pre-wrap break-all">{line}</p>
              ))
            )}
          </div>
          {phase !== "idle" && (
            <div className="p-3 border-t space-y-1.5">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>
                  {progress.completed} / {progress.total ?? "?"} episodes
                </span>
                <span className="font-medium text-foreground">
                  {phase === "done" ? "100" : pct}%
                </span>
              </div>
              <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full transition-all duration-500"
                  style={{ width: `${phase === "done" ? 100 : pct}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {phase === "done" && (
        <div className="border border-green-200 bg-green-50 rounded-lg p-4 flex items-center justify-between">
          <div>
            <p className="text-sm text-green-700 font-semibold">Benchmark run complete!</p>
            <p className="text-xs text-green-600 mt-0.5">
              Results saved to <code className="font-mono">{outputDir}/</code>
            </p>
          </div>
          <Link href="/benchmark/report" className="text-sm font-medium text-green-700 hover:underline">
            View Report →
          </Link>
        </div>
      )}

      {phase === "error" && error && (
        <div className="border border-red-200 bg-red-50 rounded-lg p-4 space-y-1">
          <p className="text-sm text-red-600 font-medium">Run failed</p>
          <p className="text-xs text-red-500">{error}</p>
          <button
            onClick={() => setPhase("idle")}
            className="text-xs text-primary hover:underline mt-1 block"
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
