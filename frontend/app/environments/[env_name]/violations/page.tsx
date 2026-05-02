import Link from "next/link";
import { API_BASE } from "@/lib/api";

interface AuditLog {
  id: string;
  episode_id: string;
  step_index: number | null;
  actor: string | null;
  action_type: string | null;
  rule_id: string | null;
  violation: boolean;
  severity: string | null;
  message: string | null;
  created_at: string;
}

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-red-100 text-red-700",
  high:     "bg-orange-100 text-orange-700",
  medium:   "bg-yellow-100 text-yellow-700",
  low:      "bg-gray-100 text-gray-500",
  info:     "bg-blue-100 text-blue-700",
};

function fmt(ts: string) {
  return new Date(ts).toLocaleString();
}

export default async function ViolationsPage({
  params,
}: {
  params: Promise<{ env_name: string }>;
}) {
  const { env_name } = await params;

  const logs: AuditLog[] = await fetch(
    `${API_BASE}/api/audit/?env_name=${encodeURIComponent(env_name)}&limit=200`,
    { cache: "no-store" }
  )
    .then((r) => (r.ok ? r.json() : []))
    .catch(() => []);

  const violations = logs.filter((l) => l.violation);
  const total = logs.length;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Policy Violations</h1>
          <p className="text-sm text-muted-foreground mt-1">
            {violations.length} violation{violations.length !== 1 ? "s" : ""} · {total} total audit entries
          </p>
        </div>
        <Link
          href={`/environments/${env_name}`}
          className="text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          ← {env_name}
        </Link>
      </div>

      {violations.length === 0 ? (
        <div className="border rounded-lg p-10 text-center text-sm text-muted-foreground">
          No policy violations recorded for this environment.
        </div>
      ) : (
        <div className="border rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/40 text-xs text-muted-foreground">
                <th className="px-4 py-2.5 text-left">Time</th>
                <th className="px-4 py-2.5 text-left">Episode</th>
                <th className="px-4 py-2.5 text-left">Step</th>
                <th className="px-4 py-2.5 text-left">Rule</th>
                <th className="px-4 py-2.5 text-left">Severity</th>
                <th className="px-4 py-2.5 text-left">Message</th>
              </tr>
            </thead>
            <tbody>
              {violations.map((log) => (
                <tr key={log.id} className="border-b last:border-0 hover:bg-muted/20">
                  <td className="px-4 py-3 text-xs text-muted-foreground whitespace-nowrap">
                    {fmt(log.created_at)}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                    {log.episode_id.slice(0, 8)}
                  </td>
                  <td className="px-4 py-3 text-xs">{log.step_index ?? "—"}</td>
                  <td className="px-4 py-3 font-mono text-xs">{log.rule_id ?? "—"}</td>
                  <td className="px-4 py-3">
                    {log.severity ? (
                      <span
                        className={`px-2 py-0.5 rounded text-xs font-medium ${
                          SEVERITY_COLORS[log.severity.toLowerCase()] ?? "bg-gray-100 text-gray-500"
                        }`}
                      >
                        {log.severity}
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-xs max-w-xs truncate">{log.message ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
