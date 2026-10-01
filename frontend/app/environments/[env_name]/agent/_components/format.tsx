"use client";

export const STATUS_BADGE: Record<string, string> = {
  pending:   "bg-yellow-100 text-yellow-700",
  running:   "bg-blue-100 text-blue-700",
  completed: "bg-green-100 text-green-700",
  failed:    "bg-red-100 text-red-600",
  stopped:   "bg-gray-100 text-gray-500",
};

export const TERM_BADGE: Record<string, string> = {
  success:   "bg-green-100 text-green-700",
  max_steps: "bg-yellow-100 text-yellow-700",
  diverged:  "bg-orange-100 text-orange-700",
  dead_end:  "bg-gray-100 text-gray-500",
  failed:    "bg-red-100 text-red-600",
};

export function badge(map: Record<string, string>, key: string | null, fallback = "") {
  const label = key ?? fallback;
  const cls = (key && map[key]) ?? "bg-gray-100 text-gray-500";
  return <span className={`px-2 py-0.5 rounded text-xs font-medium ${cls}`}>{label}</span>;
}

export function fmt(ts: string | null) {
  if (!ts) return "—";
  return new Date(ts).toLocaleString();
}

export function scoreBar(score: number) {
  const pct = Math.round(score * 100);
  const color = score >= 0.8 ? "bg-green-500" : score >= 0.5 ? "bg-yellow-400" : "bg-red-400";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-200 rounded overflow-hidden">
        <div className={`h-full ${color} rounded`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-600 w-8 text-right">{pct}%</span>
    </div>
  );
}
