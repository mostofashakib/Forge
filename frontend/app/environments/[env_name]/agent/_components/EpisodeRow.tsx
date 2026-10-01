"use client";

import { useState } from "react";
import { createPortal } from "react-dom";
import { TrajectoryDrawer } from "./TrajectoryDrawer";
import { STATUS_BADGE, TERM_BADGE, badge, fmt, scoreBar } from "./format";
import { AgentEpisode } from "./types";

export function EpisodeRow({
  envName,
  runId,
  ep,
  selected,
  onToggle,
}: {
  envName: string;
  runId: string;
  ep: AgentEpisode;
  selected: boolean;
  onToggle: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <tr
        className={`hover:bg-gray-50 ${ep.status === "completed" ? "cursor-pointer" : ""} ${selected ? "bg-blue-50" : ""}`}
        onClick={() => ep.status === "completed" && setOpen(true)}
      >
        <td className="px-3 py-3" onClick={(e) => e.stopPropagation()}>
          {ep.status === "completed" && (
            <input
              type="checkbox"
              checked={selected}
              onChange={() => onToggle(ep.id)}
              className="w-3.5 h-3.5 accent-blue-600 cursor-pointer"
            />
          )}
        </td>
        <td className="px-4 py-3 text-sm text-gray-700">{ep.episode_index + 1}</td>
        <td className="px-4 py-3">{badge(STATUS_BADGE, ep.status)}</td>
        <td className="px-4 py-3 text-sm text-gray-700">{ep.total_steps}</td>
        <td className="px-4 py-3 text-sm text-gray-700">{ep.total_reward.toFixed(3)}</td>
        <td className="px-4 py-3 w-32">{scoreBar(ep.final_objective_score)}</td>
        <td className="px-4 py-3">{badge(TERM_BADGE, ep.termination_reason, "—")}</td>
        <td className="px-4 py-3 text-xs text-gray-400">{fmt(ep.completed_at)}</td>
        <td className="px-4 py-3 text-xs text-blue-600">
          {ep.status === "completed" && "View →"}
        </td>
      </tr>
      {open && createPortal(
        <TrajectoryDrawer
          envName={envName}
          runId={runId}
          episode={ep}
          onClose={() => setOpen(false)}
        />,
        document.body
      )}
    </>
  );
}
