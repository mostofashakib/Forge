"use client";

interface StepSummary {
  step_index: number;
  action: string;       // JSON string
  reward: number;
  terminated: boolean;
  truncated: boolean;
}

interface EpisodeTimelineProps {
  steps: StepSummary[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  onBranch: (index: number) => void;
}

export default function EpisodeTimeline({
  steps,
  selectedIndex,
  onSelect,
  onBranch,
}: EpisodeTimelineProps) {
  return (
    <div className="flex flex-col gap-1 h-full overflow-auto">
      <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
        Steps
      </div>
      {steps.map((step) => {
        const action = (() => {
          try {
            return JSON.parse(step.action);
          } catch {
            return { type: "?" };
          }
        })();
        const isSelected = step.step_index === selectedIndex;
        const hasFailed = step.terminated || step.truncated;

        return (
          <button
            key={step.step_index}
            onClick={() => onSelect(step.step_index)}
            className={`text-left rounded px-2 py-1.5 text-xs transition-colors ${
              isSelected
                ? "bg-blue-950 border border-blue-400"
                : "bg-muted hover:bg-muted/80"
            }`}
          >
            <div className={isSelected ? "text-blue-400" : hasFailed ? "text-red-400" : "text-muted-foreground"}>
              Step {step.step_index}
            </div>
            <div className="text-foreground/70 truncate">{action.type ?? "—"}</div>
          </button>
        );
      })}
      {selectedIndex >= 0 && (
        <button
          onClick={() => onBranch(selectedIndex)}
          className="mt-2 w-full bg-blue-700 hover:bg-blue-600 text-white rounded px-2 py-1 text-xs font-medium"
        >
          ⑂ Branch from here
        </button>
      )}
    </div>
  );
}
