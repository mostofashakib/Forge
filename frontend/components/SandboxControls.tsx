"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { API_BASE } from "@/lib/api";

interface Props {
  envName: string;
  status: string;   // "running" | "stopped" | "error" | "queued" | "building"
  hasSandbox: boolean;
}

export function SandboxControls({ envName, status, hasSandbox }: Props) {
  const router = useRouter();
  const [busy, setBusy] = useState<"start" | "stop" | "delete" | null>(null);

  const isRunning = status === "running";
  const isStopped = status === "stopped" || status === "error";
  const canStart  = hasSandbox && isStopped;
  const canStop   = hasSandbox && isRunning;

  async function handleStart() {
    setBusy("start");
    try {
      await fetch(`${API_BASE}/api/sandbox/${envName}/start`, { method: "POST" });
      router.refresh();
    } finally {
      setBusy(null);
    }
  }

  async function handleStop() {
    setBusy("stop");
    try {
      await fetch(`${API_BASE}/api/sandbox/${envName}/stop`, { method: "POST" });
      router.refresh();
    } finally {
      setBusy(null);
    }
  }

  async function handleDelete() {
    if (!confirm(`Delete environment "${envName}"? This cannot be undone.`)) return;
    setBusy("delete");
    try {
      const url = hasSandbox
        ? `${API_BASE}/api/sandbox/${envName}`
        : `${API_BASE}/api/envs/${envName}`;
      await fetch(url, { method: "DELETE" });
      router.push("/environments");
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="flex items-center gap-2">
      {canStart && (
        <button
          onClick={handleStart}
          disabled={busy !== null}
          className="px-3 py-1.5 border border-green-300 text-green-700 rounded-md text-sm hover:bg-green-50 transition-colors disabled:opacity-50"
        >
          {busy === "start" ? "Starting…" : "Start"}
        </button>
      )}
      {canStop && (
        <button
          onClick={handleStop}
          disabled={busy !== null}
          className="px-3 py-1.5 border rounded-md text-sm hover:bg-muted/30 transition-colors disabled:opacity-50"
        >
          {busy === "stop" ? "Stopping…" : "Stop"}
        </button>
      )}
      <button
        onClick={handleDelete}
        disabled={busy !== null}
        className="px-3 py-1.5 border border-red-200 text-red-600 rounded-md text-sm hover:bg-red-50 transition-colors disabled:opacity-50"
      >
        {busy === "delete" ? "Deleting…" : "Delete"}
      </button>
    </div>
  );
}
