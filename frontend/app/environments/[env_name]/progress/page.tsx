"use client";
import { use, useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { wsBase } from "@/lib/api";

const AGENTS = [
  { id: "app_code",          label: "App Generator",            logPrefix: "[app-gen]" },
  { id: "instrumented_code", label: "Telemetry Instrumentation", logPrefix: null },
  { id: "state_bridge_code", label: "State Bridge",              logPrefix: null },
  { id: "policy_dsl",        label: "Policy Rules",              logPrefix: null },
  { id: "reward_fn_code",    label: "Reward Function",           logPrefix: null },
];

const STEP_PCT = Math.floor(100 / AGENTS.length);

export default function ProgressPage({
  params,
}: {
  params: Promise<{ env_name: string }>;
}) {
  const { env_name: envName } = use(params);
  const router = useRouter();
  const [done, setDone] = useState<Set<string>>(new Set());
  const [logs, setLogs] = useState<string[]>([]);
  const [agentStep, setAgentStep] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [finished, setFinished] = useState(false);
  const [connected, setConnected] = useState(false);
  const [startedAt] = useState(() => Date.now());
  const [elapsed, setElapsed] = useState(0);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (finished) return;
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => clearInterval(id);
  }, [finished, startedAt]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const onDone = useCallback(() => {
    setFinished(true);
    setTimeout(() => router.push(`/environments/${envName}`), 2000);
  }, [envName, router]);

  useEffect(() => {
    const ws = new WebSocket(`${wsBase()}/api/sandbox/ws/progress/${envName}`);
    ws.onopen = () => setConnected(true);
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data) as Record<string, unknown>;
      if (msg.log) {
        const line = msg.log as string;
        setLogs((prev) => [...prev.slice(-998), line]);
        // Update inline sub-step for the matching agent
        for (const agent of AGENTS) {
          if (agent.logPrefix && line.startsWith(agent.logPrefix)) {
            setAgentStep((prev) => ({ ...prev, [agent.id]: line.replace(agent.logPrefix + " ", "") }));
            break;
          }
        }
      }
      if (msg.artifact) {
        const id = msg.artifact as string;
        setDone((prev) => new Set([...prev, id]));
        setAgentStep((prev) => { const n = { ...prev }; delete n[id]; return n; });
      }
      if (msg.error) {
        setError(msg.error as string);
        ws.close();
      }
      if (msg.done && !msg.error) {
        onDone();
        ws.close();
      }
    };
    ws.onerror = () => {
      setConnected(false);
      setError("WebSocket connection failed — is the backend running?");
    };
    ws.onclose = () => setConnected(false);
    return () => ws.close();
  }, [envName, onDone]);

  const completedCount = done.size;
  const pct = finished ? 100 : Math.min(completedCount * STEP_PCT, 95);

  const estimatedTotal = 90;
  const remaining = Math.max(0, estimatedTotal - elapsed);
  const etaLabel = finished
    ? "Done"
    : elapsed < 5
    ? `~${estimatedTotal}s estimated`
    : `~${remaining}s remaining`;

  return (
    <div className="max-w-2xl mx-auto space-y-6 py-8">
      <div>
        <Link href="/environments" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
          ← All environments
        </Link>
        <div className="flex items-center gap-3 mt-3">
          <h1 className="text-2xl font-semibold tracking-tight">{envName}</h1>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
            finished ? "bg-green-100 text-green-700" :
            error    ? "bg-red-100 text-red-600" :
            connected ? "bg-blue-100 text-blue-700" :
                        "bg-yellow-100 text-yellow-700"
          }`}>
            {finished ? "ready" : error ? "error" : connected ? "building" : "connecting…"}
          </span>
        </div>
        <p className="text-sm text-muted-foreground mt-1">
          Agents running in parallel · {etaLabel} · {elapsed}s elapsed
        </p>
      </div>

      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>{completedCount} / {AGENTS.length} agents done</span>
          <span className="font-medium text-foreground">{pct}%</span>
        </div>
        <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
          <div
            className="h-full bg-primary rounded-full transition-all duration-700"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Agent checklist */}
        <div className="border rounded-lg divide-y">
          {AGENTS.map((a) => {
            const isDone = done.has(a.id);
            const step = agentStep[a.id];
            return (
              <div key={a.id} className="flex items-start gap-3 px-4 py-3">
                {isDone ? (
                  <span className="text-green-500 font-bold text-sm leading-5 shrink-0">✓</span>
                ) : (
                  <span className="text-muted-foreground text-sm leading-5 animate-spin inline-block shrink-0">⟳</span>
                )}
                <div className="min-w-0">
                  <span className={`text-sm block ${isDone ? "font-medium" : "text-muted-foreground"}`}>
                    {a.label}
                  </span>
                  {!isDone && step && (
                    <span className="text-xs text-blue-600 truncate block mt-0.5">{step}</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Live log stream */}
        <div className="border rounded-lg overflow-hidden flex flex-col">
          <div className="px-3 py-2 border-b bg-muted/40 flex items-center justify-between">
            <span className="text-xs font-medium text-muted-foreground">Worker logs</span>
            {connected && !finished && (
              <span className="flex items-center gap-1 text-xs text-green-600">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                live
              </span>
            )}
          </div>
          <div
            ref={logRef}
            className="flex-1 overflow-y-auto p-3 space-y-0.5 font-mono text-xs min-h-40 max-h-60 bg-muted/10"
          >
            {logs.length === 0 ? (
              <p className="text-muted-foreground italic">Waiting for worker output…</p>
            ) : (
              logs.map((line, i) => (
                <p key={i} className="text-foreground/80 leading-5 whitespace-pre-wrap break-all">
                  {line}
                </p>
              ))
            )}
          </div>
        </div>
      </div>

      {finished && (
        <p className="text-center text-sm text-green-600 font-medium">
          Environment ready — redirecting to hub…
        </p>
      )}

      {error && (
        <div className="border border-red-200 bg-red-50 rounded-lg p-4 space-y-2">
          <p className="text-sm text-red-600 font-medium">Build failed</p>
          <p className="text-xs text-red-500">{error}</p>
          <Link href="/environments" className="text-sm text-primary hover:underline block mt-1">
            Back to environments
          </Link>
        </div>
      )}
    </div>
  );
}
