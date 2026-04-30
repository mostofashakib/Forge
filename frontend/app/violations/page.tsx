import ViolationTable from "@/components/ViolationTable";

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
    <main className="max-w-7xl mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-bold">Policy Violations</h1>
      <form method="get" className="flex items-center gap-3">
        <input
          name="env"
          defaultValue={envName}
          placeholder="Environment name"
          className="border border-gray-300 rounded px-3 py-2 text-sm w-64"
        />
        <button
          type="submit"
          className="bg-blue-600 text-white px-4 py-2 rounded text-sm font-medium hover:bg-blue-700"
        >
          Filter
        </button>
      </form>
      <ViolationTable initialViolations={violations} />
    </main>
  );
}
