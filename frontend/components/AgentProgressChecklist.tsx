"use client";
import { useEffect, useState } from "react";
import { wsBase } from "@/lib/api";

const AGENTS = [
  { id: "generation_plan", label: "Prompt Planner" },
  { id: "backend_code",    label: "Backend Builder" },
  { id: "app_code",        label: "App Assembly" },
  { id: "instrumented_code", label: "Telemetry Instrumentation" },
  { id: "state_bridge_code", label: "State Bridge (ContainerForgeEnv)" },
  { id: "policy_dsl",      label: "Policy Rules" },
  { id: "reward_fn_code",  label: "Reward Function" },
  { id: "review_report",   label: "Quality Reviewer" },
];

interface Props {
  envName: string;
  onDone: () => void;
  onError?: (msg: string) => void;
}

// Only present for environments built with a UI; a headless build never runs it.
const UI_AGENT = { id: "ui_code", label: "UI Builder" };

export function AgentProgressChecklist({ envName, onDone, onError }: Props) {
  const [agents, setAgents] = useState(AGENTS);
  const [done, setDone] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`${wsBase()}/api/sandbox/ws/progress/${envName}`);
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data) as Record<string, string | boolean>;
      if (msg.error) {
        setError(msg.error as string);
        onError?.(msg.error as string);
        ws.close();
        return;
      }
      if (typeof msg.ui_builder_enabled === "boolean") {
        setAgents(
          msg.ui_builder_enabled
            ? [AGENTS[0], AGENTS[1], UI_AGENT, ...AGENTS.slice(2)]
            : AGENTS,
        );
      }
      if (msg.artifact) {
        setDone((prev) => new Set([...prev, msg.artifact as string]));
      }
      if (msg.done) {
        onDone();
        ws.close();
      }
    };
    ws.onerror = () => {
      setError("WebSocket connection failed");
      onError?.("WebSocket connection failed");
    };
    return () => ws.close();
  }, [envName, onDone, onError]);

  return (
    <div className="space-y-3">
      {agents.map((a) => (
        <div key={a.id} className="flex items-center gap-3">
          {done.has(a.id) ? (
            <span className="text-green-500 text-lg font-bold">✓</span>
          ) : (
            <span className="text-gray-400 text-lg animate-spin inline-block">⟳</span>
          )}
          <span className={done.has(a.id) ? "text-gray-900 font-medium" : "text-gray-400"}>
            {a.label}
          </span>
        </div>
      ))}
      {error && (
        <p className="text-red-500 text-sm mt-2 p-2 bg-red-50 rounded">{error}</p>
      )}
    </div>
  );
}
