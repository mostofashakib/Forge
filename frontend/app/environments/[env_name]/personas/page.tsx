import PersonaEditor from "@/components/PersonaEditor";
import { API_BASE } from "@/lib/api";

async function getPersonas(envName: string) {
  const res = await fetch(`${API_BASE}/api/envs/${envName}/personas`, {
    cache: "no-store",
  });
  if (!res.ok) return null;
  return res.json();
}

export default async function PersonasPage({
  params,
}: {
  params: Promise<{ env_name: string }>;
}) {
  const { env_name } = await params;
  const data = await getPersonas(env_name);

  if (!data) {
    return (
      <div className="text-center py-20">
        <p className="text-muted-foreground">
          No configuration found for environment &quot;{env_name}&quot;.
        </p>
        <p className="text-sm text-muted-foreground mt-2">
          Generate the environment first — simulated people are stored alongside its
          reward and observation settings.
        </p>
      </div>
    );
  }

  return <PersonaEditor envName={env_name} initial={data} />;
}
