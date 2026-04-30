"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function PromptForm() {
  const router = useRouter();
  const [prompt, setPrompt] = useState("");
  const [projectName, setProjectName] = useState("");
  const [domain, setDomain] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.SyntheticEvent) {
    e.preventDefault();
    if (!prompt.trim() || !projectName.trim() || !domain.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/api/compile/extract`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, project_name: projectName, domain }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      router.push(`/compiler-review/${data.job_id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <Label htmlFor="project-name">Project Name</Label>
          <Input
            id="project-name"
            placeholder="zendesk_support_env"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
          />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="domain">Domain</Label>
          <Input
            id="domain"
            placeholder="support"
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
          />
        </div>
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="prompt">Workflow Description</Label>
        <Textarea
          id="prompt"
          rows={10}
          placeholder="Describe the workflow, entities, and tasks you want to model as an RL environment..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="resize-none font-mono text-sm"
        />
      </div>

      {error && (
        <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded px-3 py-2">
          {error}
        </p>
      )}

      <Button
        type="submit"
        disabled={loading || !prompt.trim() || !projectName.trim() || !domain.trim()}
        className="w-full sm:w-auto"
      >
        {loading ? "Extracting…" : "Extract Structure →"}
      </Button>
    </form>
  );
}
