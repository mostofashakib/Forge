"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import EpisodeTimeline from "@/components/EpisodeTimeline";
import RewardBreakdown from "@/components/RewardBreakdown";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface Step {
  step_index: number;
  action: string;
  reward: number;
  verifier_results: string;
  diff: string;
  terminated: boolean;
  truncated: boolean;
}

interface EpisodeReplayClientProps {
  episodeId: string;
  envName: string;
  steps: Step[];
  totalReward: number;
  passed: boolean;
}

export default function EpisodeReplayClient({
  episodeId,
  envName,
  steps,
  totalReward,
  passed,
}: EpisodeReplayClientProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const router = useRouter();

  const selectedStep = steps.find((s) => s.step_index === selectedIndex) ?? steps[0];

  const handleBranch = async (index: number) => {
    const res = await fetch(`${API}/api/episodes/${episodeId}/steps/${index}/branch`);
    if (!res.ok) return;
    const data = await res.json();
    router.push(`/environments/${envName}/graph?actions=${encodeURIComponent(JSON.stringify(data.actions))}`);
  };

  let action: Record<string, unknown> = {};
  let diff: Record<string, unknown> = {};
  try { action = JSON.parse(selectedStep?.action ?? "{}"); } catch { /* ignore */ }
  try { diff = JSON.parse(selectedStep?.diff ?? "{}"); } catch { /* ignore */ }

  return (
    <div className="flex h-screen bg-background text-foreground">
      {/* Left: Timeline */}
      <div className="w-64 border-r p-4 flex flex-col">
        <h1 className="text-sm font-semibold mb-2 truncate">{episodeId}</h1>
        <div className="text-xs text-muted-foreground mb-3">
          {passed ? "✓ Passed" : "✗ Failed"} · reward {totalReward.toFixed(3)}
        </div>
        <div className="flex-1 overflow-auto">
          <EpisodeTimeline
            steps={steps}
            selectedIndex={selectedIndex}
            onSelect={setSelectedIndex}
            onBranch={handleBranch}
          />
        </div>
      </div>

      {/* Right: Step Detail */}
      <div className="flex-1 p-6 overflow-auto">
        {selectedStep ? (
          <div className="space-y-6">
            <section>
              <h2 className="text-xs font-semibold text-muted-foreground uppercase mb-2">Action</h2>
              <pre className="text-xs bg-muted rounded p-3 overflow-x-auto">{JSON.stringify(action, null, 2)}</pre>
            </section>
            <section>
              <h2 className="text-xs font-semibold text-muted-foreground uppercase mb-2">State Diff</h2>
              <pre className="text-xs bg-muted rounded p-3 overflow-x-auto">{JSON.stringify(diff, null, 2)}</pre>
            </section>
            <section>
              <h2 className="text-xs font-semibold text-muted-foreground uppercase mb-2">Reward Breakdown</h2>
              <RewardBreakdown components={[]} total={selectedStep.reward} />
            </section>
          </div>
        ) : (
          <p className="text-muted-foreground">No steps recorded.</p>
        )}
      </div>
    </div>
  );
}
