"use client";

import { useState } from "react";

export interface Traits {
  responsiveness: number;
  initiative: number;
  verbosity: number;
  diligence: number;
  formality: number;
  patience: number;
}

export interface CastPersona {
  id: string;
  name: string;
  role?: string;
  backstory?: string;
  goals?: string[];
  style?: string;
  traits: Traits;
  behavior: Record<string, unknown>;
}

export interface CastConfig {
  enabled: boolean;
  driver: string;
  max_actions_per_step: number;
  count?: number | null;
  roster: CastPersona[];
}

export const EMPTY_CAST: CastConfig = {
  enabled: false,
  driver: "scripted",
  max_actions_per_step: 1,
  roster: [],
};

const DEFAULT_TRAITS: Traits = {
  responsiveness: 50,
  initiative: 30,
  verbosity: 50,
  diligence: 70,
  formality: 50,
  patience: 50,
};

// Each dial is labelled by what it changes in the episode, not by its name.
// "Responsiveness" alone does not tell an author that it shortens reply
// latency; the ends of the scale do.
const TRAIT_COPY: Array<{ key: keyof Traits; label: string; low: string; high: string }> = [
  { key: "responsiveness", label: "Responsiveness", low: "Replies late", high: "Replies at once" },
  { key: "initiative", label: "Initiative", low: "Waits to be asked", high: "Speaks up unprompted" },
  { key: "verbosity", label: "Verbosity", low: "A sentence", high: "Full explanation" },
  { key: "diligence", label: "Diligence", low: "Misses details", high: "Checks everything" },
  { key: "formality", label: "Formality", low: "Shorthand", high: "Formal prose" },
  { key: "patience", label: "Patience", low: "Escalates fast", high: "Never chases" },
];

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

function blankPerson(id: string): CastPersona {
  return {
    id,
    name: "",
    role: "",
    backstory: "",
    goals: [],
    style: "",
    traits: { ...DEFAULT_TRAITS },
    behavior: {
      allowed_actions: [],
      wake_on: [],
      activity: 25,
      latency_steps: 0,
      cooldown_steps: 1,
      max_actions_per_episode: null,
    },
  };
}

function TraitDial({
  label,
  low,
  high,
  value,
  disabled,
  onChange,
}: {
  label: string;
  low: string;
  high: string;
  value: number;
  disabled?: boolean;
  onChange: (next: number) => void;
}) {
  return (
    <div className="cast-dial">
      <div className="cast-dial__head">
        <span>{label}</span>
        <b>{value}</b>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={value}
        disabled={disabled}
        aria-label={label}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="cast-dial__scale">
        <small>{low}</small>
        <small>{high}</small>
      </div>
    </div>
  );
}

/**
 * Writing the people who will share the environment with the agent.
 *
 * Everything about a person is authored here — who they are and how they
 * behave. There is deliberately no template library to pick from: a fixed set
 * of premade characters answers a question the author has already answered
 * better, and whatever shipped in it would quietly become the people every
 * environment has.
 *
 * Also deliberately no action picker: the environment's actions do not exist
 * until it is generated. This step settles who is in the world; what each of
 * them may do is chosen afterwards against the real generated surface. The copy
 * says so, rather than letting the omission read as a missing feature.
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
  const [openIndex, setOpenIndex] = useState<number | null>(null);

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

  function addPerson() {
    const taken = new Set(value.roster.map((p) => p.id));
    let index = value.roster.length + 1;
    while (taken.has(`person_${index}`)) index += 1;
    withRoster([...value.roster, blankPerson(`person_${index}`)]);
    setOpenIndex(value.roster.length);
  }

  function editPerson(index: number, patch: Partial<CastPersona>) {
    const roster = [...value.roster];
    roster[index] = { ...roster[index], ...patch };
    set({ roster });
  }

  function editTrait(index: number, key: keyof Traits, next: number) {
    const roster = [...value.roster];
    roster[index] = {
      ...roster[index],
      traits: { ...roster[index].traits, [key]: next },
    };
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
            someone rather than alone — a colleague who is slow to answer, a
            customer who has asked twice, someone whose approval is required.
            Optional: leave it empty for a world with nobody in it.
          </p>
        </div>
      </div>

      <div className="cast-roster">
        {value.roster.length === 0 && (
          <p className="cast-roster__empty">
            Nobody here yet. Describe the first person and set how they behave —
            what they can actually do is chosen after the build, once the
            environment has actions.
          </p>
        )}

        {value.roster.map((person, index) => {
          const open = openIndex === index;
          return (
            <div key={`${person.id}-${index}`} className="cast-person">
              <div className="cast-person__head">
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{person.name || "Unnamed person"}</strong>
                <em>{person.role}</em>
                <button
                  type="button"
                  disabled={disabled}
                  aria-expanded={open}
                  onClick={() => setOpenIndex(open ? null : index)}
                >
                  {open ? "Collapse" : "Edit"}
                </button>
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => {
                    withRoster(value.roster.filter((_, i) => i !== index));
                    setOpenIndex(null);
                  }}
                >
                  Remove
                </button>
              </div>

              {open && (
                <div className="cast-person__body">
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
                      rows={3}
                      value={person.backstory ?? ""}
                      disabled={disabled}
                      placeholder="Their situation, and why they behave the way they do. e.g. Answerable for anything that goes wrong on the shift, so unwilling to approve something undocumented."
                      onChange={(e) => editPerson(index, { backstory: e.target.value })}
                    />
                  </label>

                  <div className="cast-person__fields">
                    <label>
                      <span>What they want</span>
                      <textarea
                        rows={3}
                        value={(person.goals ?? []).join("\n")}
                        disabled={disabled}
                        placeholder={"Keep the process auditable\nNot be paged for anything routine"}
                        onChange={(e) =>
                          editPerson(index, {
                            goals: e.target.value
                              .split("\n")
                              .map((g) => g.trim())
                              .filter(Boolean),
                          })
                        }
                      />
                      <small>One per line.</small>
                    </label>
                    <label>
                      <span>Voice</span>
                      <textarea
                        rows={3}
                        value={person.style ?? ""}
                        disabled={disabled}
                        placeholder="e.g. Replies in clipped fragments. Skips greetings."
                        onChange={(e) => editPerson(index, { style: e.target.value })}
                      />
                      <small>How they write, in their own register.</small>
                    </label>
                  </div>

                  <fieldset className="cast-person__dials">
                    <legend>How they behave</legend>
                    <p>
                      Fixed for the whole episode and identical on every replay of a
                      seed.
                    </p>
                    <div className="cast-dials">
                      {TRAIT_COPY.map((t) => (
                        <TraitDial
                          key={t.key}
                          label={t.label}
                          low={t.low}
                          high={t.high}
                          value={person.traits[t.key]}
                          disabled={disabled}
                          onChange={(next) => editTrait(index, t.key, next)}
                        />
                      ))}
                    </div>
                  </fieldset>
                </div>
              )}
            </div>
          );
        })}

        <button
          type="button"
          disabled={disabled}
          onClick={addPerson}
          className="cast-roster__add"
        >
          + Add a person
        </button>
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
              Above {value.roster.length}, the people above are cloned with names of
              their own to fill the room.
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
          <strong>One step after the build:</strong> nobody can act until you choose
          what they&apos;re allowed to do. The environment&apos;s actions don&apos;t
          exist until it&apos;s generated, so you&apos;ll pick them from the real list
          on the Simulated People page — Forge will link you straight there.
        </p>
      )}
    </section>
  );
}
