"use client";
import { useState } from "react";

interface AuditLogEntry {
  id: number;
  episode_id: string;
  step_index: number;
  actor: string;
  action_type: string;
  rule_id: string;
  violation: string;
  severity: string;
  created_at: string;
}

interface ViolationTableProps {
  initialViolations: AuditLogEntry[];
}

const SEVERITY_STYLES: Record<string, string> = {
  high: "bg-destructive/15 text-destructive border border-destructive/30",
  medium: "bg-orange-500/15 text-orange-400 border border-orange-500/30",
  low: "bg-muted text-muted-foreground border border-border",
};

const FILTER_BUTTONS = ["all", "high", "medium", "low"] as const;

export default function ViolationTable({ initialViolations }: ViolationTableProps) {
  const [severityFilter, setSeverityFilter] = useState<string>("all");

  const violations = initialViolations.filter(
    (v) => severityFilter === "all" || v.severity === severityFilter
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground mr-1">Severity</span>
        {FILTER_BUTTONS.map((s) => (
          <button
            key={s}
            onClick={() => setSeverityFilter(s)}
            className={`px-3 py-1 rounded text-xs font-medium capitalize transition-colors ${
              severityFilter === s
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground hover:text-foreground hover:bg-muted/80"
            }`}
          >
            {s}
          </button>
        ))}
        <span className="ml-auto text-xs text-muted-foreground tabular-nums">
          {violations.length} result{violations.length !== 1 ? "s" : ""}
        </span>
      </div>

      {violations.length === 0 ? (
        <div className="rounded-lg border border-border px-4 py-8 text-center text-muted-foreground text-sm">
          No violations found.
        </div>
      ) : (
        <div className="rounded-lg border border-border overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                {["Episode", "Step", "Action", "Rule", "Violation", "Severity", "Time"].map((h) => (
                  <th key={h} className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {violations.map((v, i) => (
                <tr
                  key={v.id}
                  className={`hover:bg-muted/20 transition-colors ${i < violations.length - 1 ? "border-b border-border/50" : ""}`}
                >
                  <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground">
                    {v.episode_id.slice(0, 12)}…
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs tabular-nums text-muted-foreground">
                    {v.step_index}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs">{v.action_type}</td>
                  <td className="px-4 py-2.5 font-mono text-xs">{v.rule_id}</td>
                  <td className="px-4 py-2.5 text-xs text-muted-foreground max-w-xs truncate">
                    {v.violation}
                  </td>
                  <td className="px-4 py-2.5">
                    <span
                      className={`inline-block px-2 py-0.5 rounded text-xs font-medium capitalize ${
                        SEVERITY_STYLES[v.severity] ?? SEVERITY_STYLES.low
                      }`}
                    >
                      {v.severity}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground whitespace-nowrap">
                    {new Date(v.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
