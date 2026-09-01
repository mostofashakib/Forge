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
 * Writing the people who will share the environment with the agent.
 *
 * The library is a set of dispositions, not characters — a gatekeeper, someone
 * who answers half the question — so the author supplies the identity that
 * makes each one belong to *this* world. A template is a starting point that
 * is expected to be edited, which is why name, role, and background are
 * editable right here rather than on a later page, and why a blank person is
 * offered alongside them.
 *
 * Deliberately no action picker: the environment's actions do not exist until
 * it is generated. This step settles who is in the world; what each of them may
 * do is chosen afterwards against the real generated surface. The copy says so,
 * rather than letting the omission read as a missing feature.
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

  function withRoster(roster: CastPersona[]) {
    set({
      roster,
      // The first person added turns the cast on; removing the last turns it
      // off, so there is never an "enabled" environment with nobody in it.
      enabled: roster.length > 0,
      count: value.count && value.count < roster.length ? roster.length : value.count,
    });
  }

  function uniqueId(base: string, roster: CastPersona[]): string {
    if (!roster.some((p) => p.id === base)) return base;
    let suffix = 2;
    while (roster.some((p) => p.id === `${base}_${suffix}`)) suffix += 1;
    return `${base}_${suffix}`;
  }

  function addFrom(option: ArchetypeOption) {
    const copy = structuredClone(option);
    copy.id = uniqueId(option.id, value.roster);
    withRoster([...value.roster, copy]);
  }

  function addBlank() {
    withRoster([
      ...value.roster,
      {
        id: uniqueId("person", value.roster),
        name: "",
        role: "",
        backstory: "",
        goals: [],
        traits: {
          responsiveness: 50,
          initiative: 30,
          verbosity: 50,
          diligence: 70,
          formality: 50,
          patience: 50,
        },
        behavior: {
          allowed_actions: [],
          wake_on: [],
          activity: 25,
          latency_steps: 0,
          cooldown_steps: 1,
          max_actions_per_episode: null,
        },
      },
    ]);
  }

  function editPerson(index: number, patch: Partial<CastPersona>) {
    const roster = [...value.roster];
    roster[index] = { ...roster[index], ...patch };
    set({ roster });
  }

  const castSize = Math.max(value.count ?? value.roster.length, value.roster.length);

  return (
    <section className="cast-picker" aria-label="Simulated people">
      <div className="cast-picker__heading">
        <span>05</span>
        <div>
          <h2>Simulated people</h2>
          <p>
            Put other people in the environment so the agent has to work with
            someone rather than alone. Start from a disposition below and make it
            yours — the templates describe how someone behaves, not who they are.
            Optional: leave it empty for a world with nobody in it.
          </p>
        </div>
      </div>

      {loadError ? (
        <p className="cast-picker__note">
          Could not load the template library — is the backend running? You can add
          people after the build from the environment&apos;s Simulated People page.
        </p>
      ) : (
        <>
          <div className="cast-picker__grid" role="group" aria-label="Add a person">
            {archetypes.map((option) => (
              <button
                key={option.id}
                type="button"
                disabled={disabled}
                onClick={() => addFrom(option)}
                className="cast-card"
              >
                <strong>{option.name}</strong>
                <small>{option.backstory}</small>
                <em>+ Add</em>
              </button>
            ))}
            <button
              type="button"
              disabled={disabled}
              onClick={addBlank}
              className="cast-card cast-card--blank"
            >
              <strong>Someone else</strong>
              <small>
                Start from nothing and describe the person yourself. Fine-tune their
                disposition later on the Simulated People page.
              </small>
              <em>+ Add</em>
            </button>
          </div>

          {value.roster.length > 0 && (
            <div className="cast-roster">
              {value.roster.map((person, index) => (
                <div key={`${person.id}-${index}`} className="cast-person">
                  <div className="cast-person__head">
                    <span>{index + 1}</span>
                    <strong>{person.name || "Unnamed"}</strong>
                    <button
                      type="button"
                      disabled={disabled}
                      onClick={() =>
                        withRoster(value.roster.filter((_, i) => i !== index))
                      }
                    >
                      Remove
                    </button>
                  </div>
                  <div className="cast-person__fields">
                    <label>
                      <span>Name</span>
                      <input
                        value={person.name}
                        disabled={disabled}
                        placeholder="e.g. Alan Whitmore"
                        onChange={(e) => editPerson(index, { name: e.target.value })}
                      />
                    </label>
                    <label>
                      <span>Role</span>
                      <input
                        value={person.role ?? ""}
                        disabled={disabled}
                        placeholder="e.g. shift supervisor"
                        onChange={(e) => editPerson(index, { role: e.target.value })}
                      />
                    </label>
                  </div>
                  <label className="cast-person__about">
                    <span>Who they are</span>
                    <textarea
                      rows={2}
                      value={person.backstory ?? ""}
                      disabled={disabled}
                      placeholder="What their situation is, and why they behave the way they do."
                      onChange={(e) => editPerson(index, { backstory: e.target.value })}
                    />
                  </label>
                </div>
              ))}
            </div>
          )}

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
                  Above {value.roster.length}, the people above are cloned with
                  names of their own to fill the room.
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
