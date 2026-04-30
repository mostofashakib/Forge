import ViolationTable from "@/components/ViolationTable";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface SearchParams {
  env?: string;
  episode_id?: string;
  severity?: string;
}

export default async function ViolationsPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const params = await searchParams;
  const envName = params.env ?? "";

  const violations = envName
    ? await fetch(
        `${API}/api/audit/?env_name=${encodeURIComponent(envName)}${
          params.episode_id ? `&episode_id=${encodeURIComponent(params.episode_id)}` : ""
        }${params.severity ? `&severity=${encodeURIComponent(params.severity)}` : ""}`,
        { cache: "no-store" }
      )
        .then((r) => (r.ok ? r.json() : []))
        .catch(() => [])
    : [];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Policy Violations</h1>
        <p className="text-muted-foreground text-sm mt-1.5">
          Audit log of policy rule violations per episode and step.
        </p>
      </div>

      <form method="get" className="flex items-center gap-3">
        <Input
          name="env"
          defaultValue={envName}
          placeholder="Environment name"
          className="w-64"
        />
        <Button type="submit" variant="secondary">
          Filter
        </Button>
      </form>

      <ViolationTable initialViolations={violations} />
    </div>
  );
}
