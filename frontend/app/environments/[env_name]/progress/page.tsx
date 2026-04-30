"use client";
import { use, useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { API_BASE, wsBase } from "@/lib/api";

type EnvType = "general" | "cli" | "browser";

const AGENTS = [
  { id: "app_code",          label: "App Generator",             logPrefix: "[app-gen]" },
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
  const [envType, setEnvType] = useState<EnvType>("general");
  const [done, setDone] = useState<Set<string>>(new Set());
  const doneRef = useRef<Set<string>>(new Set());
  const [logs, setLogs] = useState<string[]>([]);
  const [agentStep, setAgentStep] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [finished, setFinished] = useState(false);
  const finishedRef = useRef(false);
  const [phase, setPhase] = useState<"connecting" | "building" | "docker" | "ready" | "error">("connecting");
  const [startedAt] = useState(() => Date.now());
  const [elapsed, setElapsed] = useState(0);
  const logRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch env_type so we can show the right progress UI
  useEffect(() => {
    fetch(`${API_BASE}/api/sandbox/${envName}`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : null)
      .then((d) => { if (d?.env_type) setEnvType(d.env_type as EnvType); })
      .catch(() => {});
  }, [envName]);

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

  const markDone = useCallback(() => {
    finishedRef.current = true;
    setFinished(true);
    setPhase("ready");
    if (pollRef.current) clearInterval(pollRef.current);
    setTimeout(() => router.push(`/environments/${envName}`), 2000);
  }, [envName, router]);

  // Poll the REST API until status = running (fallback when WS drops during Docker build)
  const startPolling = useCallback(() => {
    if (pollRef.current) return;
    setPhase("docker");
    pollRef.current = setInterval(async () => {
      if (finishedRef.current) { clearInterval(pollRef.current!); return; }
      try {
        const res = await fetch(`${API_BASE}/api/sandbox/${envName}`, { cache: "no-store" });
        if (!res.ok) return;
        const data = await res.json();
        if (data.status === "running") {
          markDone();
        } else if (data.status === "error") {
          clearInterval(pollRef.current!);
          setPhase("error");
          setError("Build failed — check the worker logs for details.");
        }
      } catch { /* network blip, retry */ }
    }, 5000);
  }, [envName, markDone]);

  useEffect(() => {
    const ws = new WebSocket(`${wsBase()}/api/sandbox/ws/progress/${envName}`);

    ws.onopen = () => setPhase("building");

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data) as Record<string, unknown>;

      if (msg.log) {
        const line = msg.log as string;
        setLogs((prev) => [...prev.slice(-998), line]);
        if (line.includes("building Docker image")) setPhase("docker");
        for (const agent of AGENTS) {
          if (agent.logPrefix && line.startsWith(agent.logPrefix)) {
            setAgentStep((prev) => ({ ...prev, [agent.id]: line.replace(agent.logPrefix + " ", "") }));
            break;
          }
        }
      }

      if (msg.artifact) {
        const id = msg.artifact as string;
        const next = new Set([...doneRef.current, id]);
        doneRef.current = next;
        setDone(new Set(next));
        setAgentStep((prev) => { const n = { ...prev }; delete n[id]; return n; });
      }

      if (msg.error) {
        setPhase("error");
        setError(msg.error as string);
        ws.close();
      }

      if (msg.done && !msg.error) {
        markDone();
        ws.close();
      }
    };

    ws.onerror = () => {
      if (finishedRef.current) return;
      // For general: poll only after all agents finished (WS drops during Docker build)
      // For cli/browser: no agents, so always fall back to polling on WS error
      const allAgentsDone = doneRef.current.size >= AGENTS.length;
      if (allAgentsDone || envType !== "general") {
        startPolling();
      } else {
        setPhase("error");
        setError("WebSocket connection failed — is the backend running?");
      }
    };

    ws.onclose = () => {
      if (finishedRef.current) return;
      const allAgentsDone = doneRef.current.size >= AGENTS.length;
      if (allAgentsDone || envType !== "general") {
        startPolling();
      }
    };

    return () => {
      ws.close();
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [envName, markDone, startPolling]);

  const completedCount = done.size;
  const pct = finished ? 100 : phase === "docker" ? 97 : Math.min(completedCount * STEP_PCT, 95);

  const phaseLabel: Record<typeof phase, string> = {
    connecting: "connecting…",
    building:   "building",
    docker:     "building Docker image…",
    ready:      "ready",
    error:      "error",
  };

  const estimatedTotal = 1450;
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
            phase === "ready"   ? "bg-green-100 text-green-700" :
            phase === "error"   ? "bg-red-100 text-red-600" :
            phase === "docker"  ? "bg-purple-100 text-purple-700" :
            phase === "building"? "bg-blue-100 text-blue-700" :
                                  "bg-yellow-100 text-yellow-700"
          }`}>
            {phaseLabel[phase]}
          </span>
        </div>
        <p className="text-sm text-muted-foreground mt-1">
          Agents running in parallel · {etaLabel} · {elapsed}s elapsed
        </p>
      </div>

      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          {envType === "general"
            ? <span>{completedCount} / {AGENTS.length} agents done</span>
            : <span>{envType === "cli" ? "Starting CLI container" : "Starting Browser container"}</span>
          }
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
        {/* Left panel: agent checklist (general) or simple step list (cli/browser) */}
        <div className="border rounded-lg divide-y">
          {envType === "general" ? (
            <>
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
              <div className="flex items-start gap-3 px-4 py-3">
                {finished ? (
                  <span className="text-green-500 font-bold text-sm leading-5 shrink-0">✓</span>
                ) : phase === "docker" ? (
                  <span className="text-muted-foreground text-sm leading-5 animate-spin inline-block shrink-0">⟳</span>
                ) : (
                  <span className="text-muted-foreground/40 text-sm leading-5 shrink-0">○</span>
                )}
                <div className="min-w-0">
                  <span className={`text-sm block ${finished ? "font-medium" : "text-muted-foreground"}`}>
                    Docker Build &amp; Launch
                  </span>
                  {phase === "docker" && !finished && (
                    <span className="text-xs text-purple-600 block mt-0.5">Building image and starting container…</span>
                  )}
                </div>
              </div>
            </>
          ) : (
            <>
              {[
                { label: "Pull Docker image", done: phase !== "connecting" },
                { label: "Start container", done: finished },
              ].map((step) => (
                <div key={step.label} className="flex items-start gap-3 px-4 py-3">
                  {step.done ? (
                    <span className="text-green-500 font-bold text-sm leading-5 shrink-0">✓</span>
                  ) : phase === "building" || phase === "docker" ? (
                    <span className="text-muted-foreground text-sm leading-5 animate-spin inline-block shrink-0">⟳</span>
                  ) : (
                    <span className="text-muted-foreground/40 text-sm leading-5 shrink-0">○</span>
                  )}
                  <span className={`text-sm ${step.done ? "font-medium" : "text-muted-foreground"}`}>
                    {step.label}
                  </span>
                </div>
              ))}
            </>
          )}
        </div>

        {/* Right panel: worker logs */}
        <div className="border rounded-lg overflow-hidden flex flex-col">
          <div className="px-3 py-2 border-b bg-muted/40 flex items-center justify-between">
            <span className="text-xs font-medium text-muted-foreground">Worker logs</span>
            {(phase === "building" || phase === "docker") && (
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
        <div className="border border-green-200 bg-green-50 rounded-lg p-4 flex items-center justify-between">
          <div>
            <p className="text-sm text-green-700 font-semibold">Environment ready!</p>
            <p className="text-xs text-green-600 mt-0.5">Redirecting to your environment hub…</p>
          </div>
          <Link
            href={`/environments/${envName}`}
            className="text-sm font-medium text-green-700 hover:underline"
          >
            Open now →
          </Link>
        </div>
      )}

      {phase === "error" && error && (
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
