import ConfigEditor from "@/components/ConfigEditor";

async function getConfig(envName: string) {
  const res = await fetch(
    `http://localhost:8000/api/envs/${envName}/config`,
    { cache: "no-store" }
  );
  if (!res.ok) return null;
  return res.json();
}

export default async function ConfigPage({
  params,
}: {
  params: Promise<{ env_name: string }>;
}) {
  const { env_name } = await params;
  const data = await getConfig(env_name);

  if (!data) {
    return (
      <div className="text-center py-20">
        <p className="text-muted-foreground">
          Config not found for environment &quot;{env_name}&quot;.
        </p>
        <p className="text-sm text-muted-foreground mt-2">
          Make sure the environment has been generated first.
        </p>
      </div>
    );
  }

  return <ConfigEditor envName={env_name} initialYaml={data.yaml} />;
}
