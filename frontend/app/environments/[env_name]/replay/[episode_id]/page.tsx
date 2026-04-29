import { notFound } from "next/navigation";
import EpisodeReplayClient from "@/components/EpisodeReplayClient";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface Step {
  step_index: number;
  action: string;
  reward: number;
  verifier_results: string;
  diff: string;
  terminated: boolean;
  truncated: boolean;
}

interface Episode {
  id: string;
  env_name: string;
  task_name: string;
  status: string;
  total_reward: number;
  passed: boolean;
  steps: Step[];
}

async function getEpisode(episodeId: string): Promise<Episode | null> {
  try {
    const res = await fetch(`${API}/api/episodes/${episodeId}`, { cache: "no-store" });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export default async function EpisodeReplayPage({
  params,
}: {
  params: Promise<{ env_name: string; episode_id: string }>;
}) {
  const { env_name, episode_id } = await params;
  const episode = await getEpisode(episode_id);

  if (!episode) notFound();

  return (
    <EpisodeReplayClient
      episodeId={episode_id}
      envName={env_name}
      steps={episode.steps}
      totalReward={episode.total_reward}
      passed={episode.passed}
    />
  );
}
