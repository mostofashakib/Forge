"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { DataCollectionPanel } from "./_components/DataCollectionPanel";
import { NewRunModal } from "./_components/NewRunModal";
import { RunCard } from "./_components/RunCard";
import { AgentRun } from "./_components/types";

export default function AgentRunsPage() {
  const params = useParams<{ env_name: string }>();
  const envName = params.env_name;

  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNew, setShowNew] = useState(false);
  const [newRunObjective, setNewRunObjective] = useState<string | undefined>(undefined);
  const [selection, setSelection] = useState<Record<string, string[]>>({});

  const loadRuns = useCallback(async () => {
    try {
      const runsRes = await fetch(`${API_BASE}/api/sandbox/${envName}/agent-runs`);
      if (runsRes.ok) setRuns(await runsRes.json());
    } finally {
      setLoading(false);
    }
  }, [envName]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadRuns(), 0);
    return () => window.clearTimeout(timer);
  }, [loadRuns]);

  // One poll for the whole page while any run is active, however many there are.
  const anyRunActive = runs.some(r => r.status === "running" || r.status === "pending");
  useEffect(() => {
    if (!anyRunActive) return;
    const t = setInterval(() => void loadRuns(), 4000);
    return () => clearInterval(t);
  }, [anyRunActive, loadRuns]);

  function openNewRun(prefillObjective?: string) {
    setNewRunObjective(prefillObjective);
    setShowNew(true);
  }

  function handleSelectionChange(runId: string, episodeIds: string[]) {
    setSelection(prev => ({ ...prev, [runId]: episodeIds }));
  }

  const totalSelected = Object.values(selection).reduce((s, ids) => s + ids.length, 0);

  return (
    <div className="min-h-screen bg-gray-50" style={{ paddingBottom: totalSelected > 0 ? "140px" : "0" }}>
      {/* Top bar */}
      <div className="bg-white border-b px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link
            href={`/environments/${envName}`}
            className="text-sm text-gray-500 hover:text-gray-700"
          >
            ← {envName}
          </Link>
          <span className="text-gray-300">/</span>
          <h1 className="text-sm font-semibold text-gray-900">Agent Runs</h1>
        </div>
        <button
          onClick={() => openNewRun()}
          className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
        >
          + New Run
        </button>
      </div>

      {/* Info banner */}
      <div className="bg-blue-50 border-b border-blue-100 px-6 py-3">
        <p className="text-sm text-blue-700">
          Agent runs record full step-level trajectories (state, action, reward, objective score) for policy training.
          Each episode stops automatically when the agent diverges, hits a dead end, succeeds, or reaches max steps.
        </p>
      </div>

      {/* Content */}
      <div className="py-6 space-y-6">

        {/* Runs list */}
        {loading ? (
          <div className="text-sm text-gray-400 py-10 text-center">Loading runs…</div>
        ) : runs.length === 0 ? (
          <div className="text-center py-16">
            <p className="text-gray-500 mb-4">No agent runs yet.</p>
            <button
              onClick={() => openNewRun()}
              className="px-5 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
            >
              Launch your first run
            </button>
          </div>
        ) : (
          runs.map(run => (
            <RunCard
              key={run.id}
              run={run}
              envName={envName}
              onRefresh={loadRuns}
              onSelectionChange={handleSelectionChange}
            />
          ))
        )}
      </div>

      {showNew && (
        <NewRunModal
          envName={envName}
          onClose={() => setShowNew(false)}
          onCreated={loadRuns}
          defaultObjective={newRunObjective}
        />
      )}

      <DataCollectionPanel
        envName={envName}
        selection={selection}
        onClear={() => setSelection({})}
      />
    </div>
  );
}
