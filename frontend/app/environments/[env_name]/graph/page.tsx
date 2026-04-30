import EnvironmentGraph from "@/components/EnvironmentGraph";

import { API_BASE as API } from "@/lib/api";

async function getCompilerInput(envName: string) {
  try {
    const res = await fetch(`${API}/api/envs/${envName}/compiler-input`, { cache: "no-store" });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export default async function GraphPage({
  params,
  searchParams,
}: {
  params: Promise<{ env_name: string }>;
  searchParams: Promise<{ episode_id?: string }>;
}) {
  const { env_name } = await params;
  const { episode_id } = await searchParams;

  const compilerInput = await getCompilerInput(env_name);

  return (
    <div className="p-6 space-y-4">
      <div>
        <h1 className="text-xl font-bold">Environment Graph</h1>
        <p className="text-sm text-muted-foreground">
          {env_name}
          {episode_id && ` · ${episode_id} (live)`}
        </p>
      </div>
      <EnvironmentGraph
        envName={env_name}
        episodeId={episode_id}
        compilerInput={compilerInput}
      />
    </div>
  );
}
