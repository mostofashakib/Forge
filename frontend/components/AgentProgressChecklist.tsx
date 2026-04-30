"use client";
import { useEffect, useState } from "react";
import { wsBase } from "@/lib/api";

const AGENTS = [
  { id: "app_code",        label: "App Generator" },
  { id: "instrumented_code", label: "Telemetry Instrumentation" },
  { id: "state_bridge_code", label: "State Bridge (ContainerForgeEnv)" },
  { id: "policy_dsl",      label: "Policy Rules" },
  { id: "reward_fn_code",  label: "Reward Function" },
];

interface Props {
  envName: string;
  onDone: () => void;
  onError?: (msg: string) => void;
}

export function AgentProgressChecklist({ envName, onDone, onError }: Props) {
  const [done, setDone] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`${wsBase()}/api/sandbox/ws/progress/${envName}`);
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data) as Record<string, string>;
      if (msg.error) {
        setError(msg.error);
        onError?.(msg.error);
        ws.close();
        return;
      }
      if (msg.artifact) {
        setDone((prev) => new Set([...prev, msg.artifact]));
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
      {AGENTS.map((a) => (
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
