"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface RolloutJob {
  id: string;
  env_name: string;
  task_name: string;
  agent_id: string;
  num_episodes: number;
  episodes_completed: number;
  status: string;
}

const STATUS_STYLES: Record<string, string> = {
  completed: "text-emerald-600",
  failed: "text-destructive",
  running: "text-primary",
  pending: "text-muted-foreground",
};

export default function RolloutLauncher() {
  const [envName, setEnvName] = useState("");
  const [taskName, setTaskName] = useState("");
  const [agentId, setAgentId] = useState("random");
  const [numEpisodes, setNumEpisodes] = useState(5);
  const [seedStart, setSeedStart] = useState(0);
  const [jobs, setJobs] = useState<RolloutJob[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.SyntheticEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API}/api/rollouts/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          env_name: envName,
          task_name: taskName,
          agent_id: agentId,
          num_episodes: numEpisodes,
          seed_start: seedStart,
        }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const { rollout_job_id } = await resp.json();
      const jobResp = await fetch(`${API}/api/rollouts/${rollout_job_id}`);
      if (jobResp.ok) {
        const job = await jobResp.json();
        setJobs((prev) => [job, ...prev]);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const refreshJob = async (jobId: string) => {
    const resp = await fetch(`${API}/api/rollouts/${jobId}`);
    if (resp.ok) {
      const updated = await resp.json();
      setJobs((prev) => prev.map((j) => (j.id === jobId ? updated : j)));
    }
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-5">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <Label htmlFor="env-name">Environment</Label>
            <Input
              id="env-name"
              placeholder="my_env"
              value={envName}
              onChange={(e) => setEnvName(e.target.value)}
              required
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="task-name">Task</Label>
            <Input
              id="task-name"
              placeholder="task_name"
              value={taskName}
              onChange={(e) => setTaskName(e.target.value)}
              required
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="agent-id">Agent</Label>
            <select
              id="agent-id"
              value={agentId}
              onChange={(e) => setAgentId(e.target.value)}
              className="h-9 w-full rounded-md border border-input bg-card px-3 py-2 text-sm text-foreground shadow-sm transition-colors outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/25"
            >
              <option value="random">Random</option>
              <option value="anthropic:claude-sonnet-4-6">Anthropic — claude-sonnet-4-6</option>
              <option value="openai:gpt-4o">OpenAI — gpt-4o</option>
              <option value="vllm:meta-llama/Llama-3-8b">vLLM — meta-llama/Llama-3-8b</option>
            </select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="num-episodes">Episodes</Label>
            <Input
              id="num-episodes"
              type="number"
              min={1}
              max={1000}
              value={numEpisodes}
              onChange={(e) => setNumEpisodes(Number(e.target.value))}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="seed-start">Seed Start</Label>
            <Input
              id="seed-start"
              type="number"
              min={0}
              value={seedStart}
              onChange={(e) => setSeedStart(Number(e.target.value))}
            />
          </div>
        </div>

        {error && (
          <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded px-3 py-2">
            {error}
          </p>
        )}

        <Button type="submit" disabled={loading}>
          {loading ? "Launching…" : "Launch Rollout →"}
        </Button>
      </form>

      {jobs.length > 0 && (
        <div className="rounded-lg border border-border overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                {["Job ID", "Env", "Task", "Agent", "Progress", "Status", ""].map((h) => (
                  <th key={h} className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {jobs.map((job, i) => (
                <tr
                  key={job.id}
                  className={`hover:bg-muted/20 transition-colors ${i < jobs.length - 1 ? "border-b border-border/50" : ""}`}
                >
                  <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground">{job.id.slice(0, 8)}</td>
                  <td className="px-4 py-2.5 text-xs">{job.env_name}</td>
                  <td className="px-4 py-2.5 text-xs">{job.task_name}</td>
                  <td className="px-4 py-2.5 text-xs font-mono">{job.agent_id}</td>
                  <td className="px-4 py-2.5 text-xs tabular-nums text-muted-foreground">
                    {job.episodes_completed}/{job.num_episodes}
                  </td>
                  <td className="px-4 py-2.5">
                    <span className={`text-xs font-medium ${STATUS_STYLES[job.status] ?? "text-muted-foreground"}`}>
                      {job.status}
                    </span>
                  </td>
                  <td className="px-4 py-2.5">
                    <button
                      onClick={() => refreshJob(job.id)}
                      className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                      ↻ Refresh
                    </button>
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
