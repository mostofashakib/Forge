"use client";
import { useState } from "react";

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

export default function RolloutLauncher() {
  const [envName, setEnvName] = useState("");
  const [taskName, setTaskName] = useState("");
  const [agentId, setAgentId] = useState("random");
  const [numEpisodes, setNumEpisodes] = useState(5);
  const [seedStart, setSeedStart] = useState(0);
  const [jobs, setJobs] = useState<RolloutJob[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
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
      <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow p-6 space-y-4">
        <h2 className="text-lg font-semibold">Launch Rollout</h2>
        {error && <p className="text-red-600 text-sm">{error}</p>}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Environment</label>
            <input
              className="mt-1 block w-full border border-gray-300 rounded px-3 py-2 text-sm"
              value={envName}
              onChange={(e) => setEnvName(e.target.value)}
              placeholder="my_env"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Task</label>
            <input
              className="mt-1 block w-full border border-gray-300 rounded px-3 py-2 text-sm"
              value={taskName}
              onChange={(e) => setTaskName(e.target.value)}
              placeholder="task_name"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Agent</label>
            <select
              className="mt-1 block w-full border border-gray-300 rounded px-3 py-2 text-sm"
              value={agentId}
              onChange={(e) => setAgentId(e.target.value)}
            >
              <option value="random">Random</option>
              <option value="anthropic:claude-sonnet-4-6">Anthropic claude-sonnet-4-6</option>
              <option value="openai:gpt-4o">OpenAI gpt-4o</option>
              <option value="vllm:meta-llama/Llama-3-8b">vLLM meta-llama/Llama-3-8b</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Episodes</label>
            <input
              type="number"
              min={1}
              max={1000}
              className="mt-1 block w-full border border-gray-300 rounded px-3 py-2 text-sm"
              value={numEpisodes}
              onChange={(e) => setNumEpisodes(Number(e.target.value))}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Seed Start</label>
            <input
              type="number"
              min={0}
              className="mt-1 block w-full border border-gray-300 rounded px-3 py-2 text-sm"
              value={seedStart}
              onChange={(e) => setSeedStart(Number(e.target.value))}
            />
          </div>
        </div>
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 text-white px-4 py-2 rounded text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Launching..." : "Launch Rollout"}
        </button>
      </form>

      {jobs.length > 0 && (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                {["Job ID", "Env", "Task", "Agent", "Progress", "Status", ""].map((h) => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {jobs.map((job) => (
                <tr key={job.id}>
                  <td className="px-4 py-3 text-xs font-mono text-gray-600">{job.id}</td>
                  <td className="px-4 py-3 text-sm">{job.env_name}</td>
                  <td className="px-4 py-3 text-sm">{job.task_name}</td>
                  <td className="px-4 py-3 text-sm">{job.agent_id}</td>
                  <td className="px-4 py-3 text-sm">
                    {job.episodes_completed}/{job.num_episodes}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                        job.status === "completed"
                          ? "bg-green-100 text-green-800"
                          : job.status === "failed"
                          ? "bg-red-100 text-red-800"
                          : job.status === "running"
                          ? "bg-blue-100 text-blue-800"
                          : "bg-gray-100 text-gray-800"
                      }`}
                    >
                      {job.status}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() => refreshJob(job.id)}
                      className="text-xs text-blue-600 hover:underline"
                    >
                      Refresh
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
