"use client";

import { useState } from "react";
import Link from "next/link";

export interface ReplayStatus {
  active: boolean;
  objective?: string;
  num_episodes?: number;
  episodes?: { index: number; num_commands: number }[];
}

export function SyntheticEpochCard({
  status,
  envName,
  onLaunch,
  launching,
}: {
  status: ReplayStatus;
  envName: string;
  onLaunch: (seedStart: number, numEpisodes: number) => void;
  launching: string | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const total = status.num_episodes ?? 0;

  return (
    <div className="border border-indigo-200 rounded-xl overflow-hidden">
      {/* Epoch header */}
      <div
        className="bg-indigo-50 px-5 py-4 flex items-center justify-between gap-4 cursor-pointer hover:bg-indigo-100/60 transition-colors"
        onClick={() => setExpanded((v) => !v)}
      >
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <span className="text-sm font-semibold text-indigo-900">Synthetic Epoch</span>
            <span className="px-1.5 py-0.5 bg-indigo-100 text-indigo-600 text-xs rounded font-medium">
              {total} episode{total !== 1 ? "s" : ""}
            </span>
          </div>
          <p className="text-xs text-indigo-600 font-mono truncate">{status.objective}</p>
        </div>
        <div className="flex items-center gap-3 shrink-0" onClick={(e) => e.stopPropagation()}>
          <button
            onClick={() => onLaunch(0, total)}
            disabled={launching !== null}
            className="px-3 py-1.5 text-xs font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
          >
            {launching === `0-${total}` ? "Launching…" : `Run all ${total} episodes →`}
          </button>
          <Link
            href={`/environments/${envName}/synthetic`}
            className="text-xs text-indigo-500 hover:text-indigo-700 underline"
          >
            Manage
          </Link>
          <span className="text-xs text-indigo-400">{expanded ? "▲" : "▼"}</span>
        </div>
      </div>

      {/* Episodes list */}
      {expanded && (
        <div className="divide-y divide-indigo-100">
          {status.episodes?.map((ep) => (
            <div key={ep.index} className="flex items-center justify-between px-5 py-3 hover:bg-indigo-50/40">
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-gray-700">Episode {ep.index + 1}</span>
                <span className="text-xs text-gray-400">{ep.num_commands} steps in trajectory</span>
              </div>
              <button
                onClick={() => onLaunch(ep.index, 1)}
                disabled={launching !== null}
                className="px-3 py-1.5 text-xs font-medium text-indigo-700 border border-indigo-300 rounded-lg hover:bg-indigo-50 disabled:opacity-50 transition-colors"
              >
                {launching === `${ep.index}-1` ? "Launching…" : "Run episode →"}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
