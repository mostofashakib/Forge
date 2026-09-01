"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";

interface RosterEntry {
  id: string;
  name: string;
  role?: string;
}

interface PersonaPayload {
  personas: {
    enabled: boolean;
    driver: string;
    roster: RosterEntry[];
    archetypes?: RosterEntry[];
    count?: number | null;
  };
}

/**
 * The on/off switch for simulated people, shown where runs are launched.
 *
 * This edits the environment, not the run: personas are resolved when the
 * environment is built, so the flag applies to this run and every later one
 * until it is changed again. The copy says so rather than implying a per-run
 * override that does not exist.
 */
export default function PersonaRunToggle({ envName }: { envName: string }) {
  const [payload, setPayload] = useState<PersonaPayload["personas"] | null>(null);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/envs/${envName}/personas`, { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : null))
      .then((body: PersonaPayload | null) => {
        if (!cancelled) {
          setPayload(body?.personas ?? null);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [envName]);

  async function toggle(next: boolean) {
    setPending(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/envs/${envName}/personas/enabled`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: next }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(typeof body.detail === "string" ? body.detail : "Could not change this.");
        return;
      }
      setPayload(body.personas);
    } catch {
      setError("Could not reach the API.");
    } finally {
      setPending(false);
    }
  }

  // An environment whose config could not be read has no cast to switch on.
  // Showing a dead control would be worse than showing nothing.
  if (loading || payload === null) return null;

  const roster = payload.roster ?? [];
  const castSize = payload.count ?? roster.length;
  const empty = roster.length === 0 && (payload.archetypes ?? []).length === 0;
  const names = roster
    .slice(0, 3)
    .map((p) => (p.role ? `${p.name} (${p.role})` : p.name))
    .join(", ");
  const overflow = roster.length > 3 ? ` +${roster.length - 3} more` : "";

  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2.5">
      <label className="flex cursor-pointer items-start gap-2.5">
        <input
          type="checkbox"
          checked={payload.enabled}
          disabled={pending || empty}
          onChange={(e) => toggle(e.target.checked)}
          className="mt-0.5 size-4 shrink-0 accent-blue-600 disabled:opacity-40"
        />
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-medium text-gray-900">
            Simulated people
          </span>
          <span className="mt-0.5 block text-xs leading-snug text-gray-600">
            {empty ? (
              <>
                Nobody is configured for this environment yet.{" "}
                <Link
                  href={`/environments/${envName}/personas`}
                  className="text-blue-600 underline underline-offset-2"
                >
                  Add someone
                </Link>{" "}
                to run with colleagues in the world.
              </>
            ) : payload.enabled ? (
              <>
                {castSize} {castSize === 1 ? "person" : "people"} will act alongside
                the agent{names ? `: ${names}${overflow}` : ""}. Decided by{" "}
                <code className="rounded bg-gray-200 px-1 text-[0.7rem]">
                  {payload.driver}
                </code>
                .
              </>
            ) : (
              <>
                {castSize} configured, currently switched off — the agent runs in an
                empty world.
              </>
            )}
          </span>
          {!empty && (
            <span className="mt-1 block text-[0.7rem] text-gray-500">
              This is an environment setting: it applies to this run and every later
              one until you change it.{" "}
              <Link
                href={`/environments/${envName}/personas`}
                className="text-blue-600 underline underline-offset-2"
              >
                Edit the cast
              </Link>
            </span>
          )}
        </span>
      </label>
      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
    </div>
  );
}
