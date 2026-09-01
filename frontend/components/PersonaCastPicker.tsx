"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

export interface CastPersona {
  id: string;
  name: string;
  role?: string;
  backstory?: string;
  goals?: string[];
  style?: string;
  traits: Record<string, number>;
  behavior: Record<string, unknown>;
}

export interface CastConfig {
  enabled: boolean;
  driver: string;
  max_actions_per_step: number;
  count?: number | null;
  roster: CastPersona[];
}

interface ArchetypeOption extends CastPersona {
  id: string;
}

export const EMPTY_CAST: CastConfig = {
  enabled: false,
  driver: "scripted",
  max_actions_per_step: 1,
  roster: [],
};

const DRIVERS = [
  {
    id: "scripted",
    label: "Scripted",
    hint: "Free and offline. People act at human-like times, but never say anything new.",
  },
  {
    id: "anthropic:claude-sonnet-5",
    label: "Claude Sonnet 5",
    hint: "A model decides each person's turn, in character.",
  },
  {
    id: "anthropic:claude-opus-5",
    label: "Claude Opus 5",
    hint: "A model decides each person's turn, in character.",
  },
];

/**
 * Choosing who will share the environment with the agent, at build time.
 *
 * Deliberately does not offer an action picker. The environment's actions do
 * not exist until it is generated, so there is nothing to bind to yet — this
 * step settles who is in the world, and what each of them may do is chosen
 * afterwards against the real generated surface. The copy says so, rather than
 * letting the omission read as a missing feature.
 */
export default function PersonaCastPicker({
  value,
  onChange,
  disabled,
}: {
  value: CastConfig;
  onChange: (next: CastConfig) => void;
  disabled?: boolean;
}) {
  const [archetypes, setArchetypes] = useState<ArchetypeOption[]>([]);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/persona-archetypes`, { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((body) => {
        if (!cancelled) setArchetypes(body.archetypes ?? []);
      })
      .catch(() => {
        if (!cancelled) setLoadError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const set = (next: Partial<CastConfig>) => onChange({ ...value, ...next });

  function toggleArchetype(option: ArchetypeOption) {
    const present = value.roster.some((p) => p.id === option.id);
    const roster = present
      ? value.roster.filter((p) => p.id !== option.id)
      : [...value.roster, structuredClone(option)];
    set({
      roster,
      // Turning the first person on turns the cast on; removing the last turns
      // it off, so there is never an "enabled" environment with nobody in it.
      enabled: roster.length > 0 && (value.enabled || !present),
      count: value.count && value.count < roster.length ? roster.length : value.count,
    });
  }

  const castSize = Math.max(value.count ?? value.roster.length, value.roster.length);

  return (
    <section className="cast-picker" aria-label="Simulated people">
      <div className="cast-picker__heading">
        <span>05</span>
        <div>
          <h2>Simulated people</h2>
          <p>
            Put colleagues, patients, or customers in the environment so the agent
            has to work with someone rather than alone. Optional — leave it empty
            for a world with nobody in it.
          </p>
        </div>
      </div>

      {loadError ? (
        <p className="cast-picker__note">
          Could not load the archetype library — is the backend running? You can add
          people after the build from the environment&apos;s Simulated People page.
        </p>
      ) : (
        <>
          <div className="cast-picker__grid" role="group" aria-label="Who is in the world">
            {archetypes.map((option) => {
              const on = value.roster.some((p) => p.id === option.id);
              return (
                <button
                  key={option.id}
                  type="button"
                  aria-pressed={on}
                  disabled={disabled}
                  onClick={() => toggleArchetype(option)}
                  className={`cast-card ${on ? "cast-card--on" : ""}`}
                >
                  <strong>{option.name}</strong>
                  <em>{option.role}</em>
                  <small>{option.backstory}</small>
                </button>
              );
            })}
          </div>

          {value.roster.length > 0 && (
            <div className="cast-picker__settings">
              <label>
                <span>Cast size</span>
                <input
                  type="number"
                  min={value.roster.length}
                  max={50}
                  value={castSize}
                  disabled={disabled}
                  onChange={(e) => set({ count: Number(e.target.value) })}
                />
                <small>
                  Above {value.roster.length}, the people you picked are cloned with
                  their own names to fill the room.
                </small>
              </label>
              <label>
                <span>Who decides their turns</span>
                <select
                  value={value.driver}
                  disabled={disabled}
                  onChange={(e) => set({ driver: e.target.value })}
                >
                  {DRIVERS.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.label}
                    </option>
                  ))}
                </select>
                <small>{DRIVERS.find((d) => d.id === value.driver)?.hint}</small>
              </label>
              <label>
                <span>Speakers per step</span>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={value.max_actions_per_step}
                  disabled={disabled}
                  onChange={(e) => set({ max_actions_per_step: Number(e.target.value) })}
                />
                <small>How many people may act on a single agent step.</small>
              </label>
            </div>
          )}

          {value.roster.length > 0 && (
            <p className="cast-picker__note">
              <strong>One step after the build:</strong> nobody can act until you
              choose what they&apos;re allowed to do. The environment&apos;s actions
              don&apos;t exist until it&apos;s generated, so you&apos;ll pick them from
              the real list on the Simulated People page — Forge will link you
              straight there.
            </p>
          )}
        </>
      )}
    </section>
  );
}
