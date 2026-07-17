"use client";

import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

interface SandboxCapacity {
  activeCount: number | null;
  limit: number | null;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useSandboxCapacity(): SandboxCapacity {
  const [activeCount, setActiveCount] = useState<number | null>(null);
  const [limit, setLimit] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/sandbox/capacity`, {
        cache: "no-store",
        signal: AbortSignal.timeout(8000),
      });
      if (!response.ok) {
        throw new Error(`Capacity request failed (${response.status})`);
      }
      const data = await response.json() as { active_count: number; limit: number };
      setActiveCount(data.active_count);
      setLimit(data.limit);
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not reach the backend");
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);

  return { activeCount, limit, error, refresh };
}
