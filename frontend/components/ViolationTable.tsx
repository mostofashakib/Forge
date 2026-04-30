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
  high: "bg-red-100 text-red-800",
  medium: "bg-orange-100 text-orange-800",
  low: "bg-gray-100 text-gray-600",
};

export default function ViolationTable({ initialViolations }: ViolationTableProps) {
  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const violations = initialViolations.filter(
    (v) => severityFilter === "all" || v.severity === severityFilter
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <label className="text-sm font-medium text-gray-700">Severity:</label>
        {["all", "high", "medium", "low"].map((s) => (
          <button
            key={s}
            onClick={() => setSeverityFilter(s)}
            className={`px-3 py-1 rounded text-sm font-medium capitalize ${
              severityFilter === s
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            {s}
          </button>
        ))}
      </div>

      {violations.length === 0 ? (
        <p className="text-gray-500 text-sm py-4">No violations found.</p>
      ) : (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                {["Episode", "Step", "Action", "Rule", "Violation", "Severity", "Time"].map((h) => (
                  <th
                    key={h}
                    className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {violations.map((v) => (
                <tr key={v.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-xs font-mono text-gray-600">{v.episode_id}</td>
                  <td className="px-4 py-3 text-sm text-center">{v.step_index}</td>
                  <td className="px-4 py-3 text-sm font-medium">{v.action_type}</td>
                  <td className="px-4 py-3 text-xs font-mono text-gray-600">{v.rule_id}</td>
                  <td className="px-4 py-3 text-sm text-gray-700 max-w-xs truncate" title={v.violation}>
                    {v.violation}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium capitalize ${
                        SEVERITY_STYLES[v.severity] ?? SEVERITY_STYLES.low
                      }`}
                    >
                      {v.severity}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-500">
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
