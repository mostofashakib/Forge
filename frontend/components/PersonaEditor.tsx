"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { API_BASE } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

// ---------------------------------------------------------------------------
// Shape of the `personas:` block, mirroring forge/personas/config.py
// ---------------------------------------------------------------------------

export interface Traits {
  responsiveness: number;
  initiative: number;
  verbosity: number;
  diligence: number;
  formality: number;
  patience: number;
}

export interface Behavior {
  allowed_actions: string[];
  wake_on: string[];
  activity: number;
  latency_steps: number;
  cooldown_steps: number;
  max_actions_per_episode: number | null;
}

export interface PersonaEntry {
  id: string;
  name: string;
  role?: string;
  backstory?: string;
  goals?: string[];
  style?: string;
  knowledge?: Record<string, unknown>;
  traits: Traits;
  behavior: Behavior;
}

export interface Population {
  enabled: boolean;
  driver: string;
  max_actions_per_step: number;
  count?: number | null;
  seed?: number | null;
  roster: PersonaEntry[];
  archetypes?: PersonaEntry[];
}

interface ArchetypeEntry extends PersonaEntry {
  archetypeId: string;
}

interface Payload {
  personas: Population;
  environment_actions: string[];
  archetypes: Array<PersonaEntry & { id: string }>;
}

// Each dial is labelled by what it changes in the episode, not by its name.
// "Responsiveness" alone does not tell an author that it shortens reply
// latency; the caption does.
const TRAIT_COPY: Array<{ key: keyof Traits; label: string; low: string; high: string }> = [
  { key: "responsiveness", label: "Responsiveness", low: "Replies late", high: "Replies at once" },
  { key: "initiative", label: "Initiative", low: "Waits to be asked", high: "Speaks up unprompted" },
  { key: "verbosity", label: "Verbosity", low: "A sentence", high: "Full explanation" },
  { key: "diligence", label: "Diligence", low: "Misses details", high: "Checks everything" },
  { key: "formality", label: "Formality", low: "Shorthand", high: "Formal prose" },
  { key: "patience", label: "Patience", low: "Escalates fast", high: "Never chases" },
];

const DRIVERS = [
  { id: "scripted", label: "Scripted", hint: "Free, offline, byte-reproducible. Personas act at human-like times but say nothing new." },
  { id: "anthropic:claude-sonnet-5", label: "Claude Sonnet 5", hint: "A model decides each turn, inside the action space you set below." },
  { id: "anthropic:claude-opus-5", label: "Claude Opus 5", hint: "A model decides each turn, inside the action space you set below." },
  { id: "openai:gpt-4.1", label: "GPT-4.1", hint: "A model decides each turn, inside the action space you set below." },
];

const DEFAULT_TRAITS: Traits = {
  responsiveness: 50,
  initiative: 30,
  verbosity: 50,
  diligence: 70,
  formality: 50,
  patience: 50,
};

const DEFAULT_BEHAVIOR: Behavior = {
  allowed_actions: [],
  wake_on: [],
  activity: 25,
  latency_steps: 0,
  cooldown_steps: 1,
  max_actions_per_episode: null,
};

function blankPersona(index: number): PersonaEntry {
  return {
    id: `persona_${index + 1}`,
    name: `Persona ${index + 1}`,
    role: "",
    backstory: "",
    goals: [],
    style: "",
    traits: { ...DEFAULT_TRAITS },
    behavior: { ...DEFAULT_BEHAVIOR, allowed_actions: [], wake_on: [] },
  };
}

// ---------------------------------------------------------------------------

function TraitSlider({
  label,
  low,
  high,
  value,
  onChange,
}: {
  label: string;
  low: string;
  high: string;
  value: number;
  onChange: (next: number) => void;
}) {
  return (
    <div className="persona-trait">
      <div className="persona-trait__head">
        <Label className="persona-trait__label">{label}</Label>
        <span className="persona-trait__value">{value}</span>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="persona-trait__range"
        aria-label={label}
      />
      <div className="persona-trait__scale">
        <span>{low}</span>
        <span>{high}</span>
      </div>
    </div>
  );
}

function ActionPicker({
  legend,
  hint,
  available,
  selected,
  onToggle,
}: {
  legend: string;
  hint: string;
  available: string[];
  selected: string[];
  onToggle: (action: string) => void;
}) {
  if (available.length === 0) {
    return (
      <div className="persona-actions">
        <Label>{legend}</Label>
        <p className="persona-hint">
          This environment&apos;s action list could not be read. Edit the roster in the
          YAML config editor instead.
        </p>
      </div>
    );
  }
  return (
    <fieldset className="persona-actions">
      <legend className="persona-actions__legend">{legend}</legend>
      <p className="persona-hint">{hint}</p>
      <div className="persona-actions__grid">
        {available.map((action) => {
          const on = selected.includes(action);
          return (
            <button
              key={action}
              type="button"
              onClick={() => onToggle(action)}
              aria-pressed={on}
              className={`persona-chip ${on ? "persona-chip--on" : ""}`}
            >
              {action}
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}

function PersonaCard({
  persona,
  index,
  actions,
  onChange,
  onRemove,
}: {
  persona: PersonaEntry;
  index: number;
  actions: string[];
  onChange: (next: PersonaEntry) => void;
  onRemove: () => void;
}) {
  const [open, setOpen] = useState(false);
  const inert = persona.behavior.allowed_actions.length === 0;

  const set = <K extends keyof PersonaEntry>(key: K, value: PersonaEntry[K]) =>
    onChange({ ...persona, [key]: value });

  const setTrait = (key: keyof Traits, value: number) =>
    onChange({ ...persona, traits: { ...persona.traits, [key]: value } });

  const setBehavior = <K extends keyof Behavior>(key: K, value: Behavior[K]) =>
    onChange({ ...persona, behavior: { ...persona.behavior, [key]: value } });

  const toggle = (key: "allowed_actions" | "wake_on", action: string) => {
    const current = persona.behavior[key];
    setBehavior(
      key,
      current.includes(action)
        ? current.filter((a) => a !== action)
        : [...current, action],
    );
  };

  return (
    <Card className="persona-card">
      <CardHeader className="persona-card__head">
        <div className="persona-card__identity">
          <CardTitle className="persona-card__name">
            {persona.name || persona.id}
          </CardTitle>
          <p className="persona-card__role">{persona.role || "no role set"}</p>
        </div>
        <div className="persona-card__meta">
          {inert ? (
            <Badge variant="outline" className="persona-badge persona-badge--inert">
              Never acts
            </Badge>
          ) : (
            <Badge variant="secondary" className="persona-badge">
              {persona.behavior.allowed_actions.length} action
              {persona.behavior.allowed_actions.length === 1 ? "" : "s"}
            </Badge>
          )}
          <button
            type="button"
            className="persona-card__toggle"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
          >
            {open ? "Collapse" : "Edit"}
          </button>
          <button type="button" className="persona-card__remove" onClick={onRemove}>
            Remove
          </button>
        </div>
      </CardHeader>

      {open && (
        <CardContent className="persona-card__body">
          <div className="persona-grid">
            <div>
              <Label htmlFor={`id-${index}`}>Identifier</Label>
              <Input
                id={`id-${index}`}
                value={persona.id}
                onChange={(e) => set("id", e.target.value)}
              />
            </div>
            <div>
              <Label htmlFor={`name-${index}`}>Name</Label>
              <Input
                id={`name-${index}`}
                value={persona.name}
                onChange={(e) => set("name", e.target.value)}
              />
            </div>
            <div>
              <Label htmlFor={`role-${index}`}>Role</Label>
              <Input
                id={`role-${index}`}
                value={persona.role ?? ""}
                placeholder="charge nurse"
                onChange={(e) => set("role", e.target.value)}
              />
            </div>
          </div>

          <div>
            <Label htmlFor={`backstory-${index}`}>Background</Label>
            <Textarea
              id={`backstory-${index}`}
              rows={2}
              value={persona.backstory ?? ""}
              placeholder="Running a full clinic list. Answers between patients."
              onChange={(e) => set("backstory", e.target.value)}
            />
          </div>

          <div className="persona-grid persona-grid--two">
            <div>
              <Label htmlFor={`goals-${index}`}>What they want</Label>
              <Textarea
                id={`goals-${index}`}
                rows={2}
                value={(persona.goals ?? []).join("\n")}
                placeholder={"Keep patients safe\nNot be paged for anything routine"}
                onChange={(e) =>
                  set(
                    "goals",
                    e.target.value.split("\n").map((g) => g.trim()).filter(Boolean),
                  )
                }
              />
              <p className="persona-hint">One goal per line.</p>
            </div>
            <div>
              <Label htmlFor={`style-${index}`}>Voice</Label>
              <Textarea
                id={`style-${index}`}
                rows={2}
                value={persona.style ?? ""}
                placeholder="Replies in clipped fragments. Skips greetings."
                onChange={(e) => set("style", e.target.value)}
              />
            </div>
          </div>

          <section className="persona-section">
            <h4>Disposition</h4>
            <p className="persona-hint">
              Fixed for the whole episode and identical on every replay of a seed.
              These shape how a persona behaves, never what they are allowed to do.
            </p>
            <div className="persona-traits">
              {TRAIT_COPY.map((t) => (
                <TraitSlider
                  key={t.key}
                  label={t.label}
                  low={t.low}
                  high={t.high}
                  value={persona.traits[t.key]}
                  onChange={(v) => setTrait(t.key, v)}
                />
              ))}
            </div>
          </section>

          <section className="persona-section">
            <h4>Engagement</h4>
            <ActionPicker
              legend="Can do"
              hint="The complete set of actions this persona may ever take. A persona with none never acts."
              available={actions}
              selected={persona.behavior.allowed_actions}
              onToggle={(a) => toggle("allowed_actions", a)}
            />
            <ActionPicker
              legend="Wakes on"
              hint="Agent actions that make this persona due. Leave empty and anything the agent does concerns them."
              available={actions}
              selected={persona.behavior.wake_on}
              onToggle={(a) => toggle("wake_on", a)}
            />

            <div className="persona-grid">
              <div>
                <Label htmlFor={`activity-${index}`}>Unprompted rate</Label>
                <Input
                  id={`activity-${index}`}
                  type="number"
                  min={0}
                  max={100}
                  value={persona.behavior.activity}
                  onChange={(e) => setBehavior("activity", Number(e.target.value))}
                />
                <p className="persona-hint">Chance of acting with nobody waiting.</p>
              </div>
              <div>
                <Label htmlFor={`latency-${index}`}>Reply delay</Label>
                <Input
                  id={`latency-${index}`}
                  type="number"
                  min={0}
                  value={persona.behavior.latency_steps}
                  onChange={(e) => setBehavior("latency_steps", Number(e.target.value))}
                />
                <p className="persona-hint">Steps before answering. Shortened by responsiveness.</p>
              </div>
              <div>
                <Label htmlFor={`cooldown-${index}`}>Cooldown</Label>
                <Input
                  id={`cooldown-${index}`}
                  type="number"
                  min={0}
                  value={persona.behavior.cooldown_steps}
                  onChange={(e) => setBehavior("cooldown_steps", Number(e.target.value))}
                />
                <p className="persona-hint">Steps of quiet after acting.</p>
              </div>
              <div>
                <Label htmlFor={`budget-${index}`}>Turn budget</Label>
                <Input
                  id={`budget-${index}`}
                  type="number"
                  min={0}
                  placeholder="unlimited"
                  value={persona.behavior.max_actions_per_episode ?? ""}
                  onChange={(e) =>
                    setBehavior(
                      "max_actions_per_episode",
                      e.target.value === "" ? null : Number(e.target.value),
                    )
                  }
                />
                <p className="persona-hint">Hard cap per episode.</p>
              </div>
            </div>
          </section>
        </CardContent>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------

export default function PersonaEditor({
  envName,
  initial,
}: {
  envName: string;
  initial: Payload;
}) {
  const [population, setPopulation] = useState<Population>(initial.personas);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewSeed, setPreviewSeed] = useState(0);
  const [preview, setPreview] = useState<{ roster: PersonaEntry[]; warnings: string[] } | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  const actions = initial.environment_actions;
  const archetypes = initial.archetypes as ArchetypeEntry[];

  const mutate = useCallback((next: Partial<Population>) => {
    setPopulation((current) => ({ ...current, ...next }));
    setSaved(false);
  }, []);

  const setRoster = (roster: PersonaEntry[]) => mutate({ roster });

  // The cast an episode actually contains is not the roster: `count` above the
  // roster size clones archetypes. Showing the resolved list is the difference
  // between configuring four personas and discovering four personas.
  const refreshPreview = useCallback(async () => {
    setPreviewError(null);
    try {
      const res = await fetch(`${API_BASE}/api/envs/${envName}/personas/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ personas: population, seed: previewSeed }),
      });
      const body = await res.json();
      if (!res.ok) {
        setPreview(null);
        setPreviewError(body.detail ?? "Could not resolve this cast.");
        return;
      }
      setPreview(body);
    } catch {
      setPreviewError("Could not reach the API.");
    }
  }, [envName, population, previewSeed]);

  useEffect(() => {
    const timer = setTimeout(refreshPreview, 250);
    return () => clearTimeout(timer);
  }, [refreshPreview]);

  async function save() {
    setSaving(true);
    setError(null);
    setSaved(false);
    try {
      const res = await fetch(`${API_BASE}/api/envs/${envName}/personas`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ personas: population }),
      });
      const body = await res.json();
      if (!res.ok) {
        setError(typeof body.detail === "string" ? body.detail : "Could not save this cast.");
        return;
      }
      setPopulation(body.personas);
      setSaved(true);
    } catch {
      setError("Could not reach the API.");
    } finally {
      setSaving(false);
    }
  }

  function addBlank() {
    setRoster([...population.roster, blankPersona(population.roster.length)]);
  }

  function addArchetype(archetypeId: string) {
    const source = archetypes.find((a) => a.archetypeId === archetypeId || a.id === archetypeId);
    if (!source) return;
    const taken = new Set(population.roster.map((p) => p.id));
    let id = source.id;
    let suffix = 2;
    while (taken.has(id)) id = `${source.id}_${suffix++}`;
    setRoster([
      ...population.roster,
      { ...structuredClone(source), id, behavior: { ...structuredClone(source.behavior) } },
    ]);
  }

  // Every person unbound is the specific state a builder-created cast lands in,
  // and it reads very differently from one persona someone forgot: the whole
  // cast is inert and the author has not started binding yet.
  const unbound =
    population.roster.length > 0 &&
    population.roster.every((p) => p.behavior.allowed_actions.length === 0);

  const inertCount = useMemo(
    () => population.roster.filter((p) => p.behavior.allowed_actions.length === 0).length,
    [population.roster],
  );

  const driverHint = DRIVERS.find((d) => d.id === population.driver)?.hint;

  return (
    <div className="persona-editor">
      <header className="persona-editor__head">
        <div>
          <h1>Simulated people</h1>
          <p>
            The colleagues, patients, and customers who share{" "}
            <span className="persona-editor__env">{envName}</span> with the agent.
            Who they are and when they act is fixed by the episode seed; what they
            do each turn is decided by the driver, inside the actions you allow.
          </p>
        </div>
        <div className="persona-editor__actions">
          {saved && <Badge variant="secondary">Saved</Badge>}
          <Button onClick={save} disabled={saving}>
            {saving ? "Saving…" : "Save cast"}
          </Button>
        </div>
      </header>

      {error && <p className="persona-error">{error}</p>}

      {unbound && (
        <div className="persona-callout">
          <strong>Nobody can act yet.</strong>
          <p>
            You picked this cast while the environment was being built, before its
            actions existed. Open each person below and tick what they&apos;re
            allowed to do — until then they&apos;re in the world but silent.
          </p>
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="persona-panel__title">Population</CardTitle>
        </CardHeader>
        <CardContent className="persona-panel">
          <label className="persona-switch">
            <input
              type="checkbox"
              checked={population.enabled}
              onChange={(e) => mutate({ enabled: e.target.checked })}
            />
            <span>
              <strong>Put people in this environment</strong>
              <em>Off, the environment runs exactly as it does today.</em>
            </span>
          </label>

          <div className="persona-grid">
            <div>
              <Label htmlFor="count">Cast size</Label>
              <Input
                id="count"
                type="number"
                min={0}
                placeholder={String(population.roster.length)}
                value={population.count ?? ""}
                onChange={(e) =>
                  mutate({ count: e.target.value === "" ? null : Number(e.target.value) })
                }
              />
              <p className="persona-hint">
                Above the roster, archetypes are cloned to fill it. Below, the roster
                is trimmed.
              </p>
            </div>
            <div>
              <Label htmlFor="per-step">Speakers per step</Label>
              <Input
                id="per-step"
                type="number"
                min={1}
                value={population.max_actions_per_step}
                onChange={(e) => mutate({ max_actions_per_step: Number(e.target.value) })}
              />
              <p className="persona-hint">How many people may act on one agent step.</p>
            </div>
            <div>
              <Label htmlFor="seed">Cast seed</Label>
              <Input
                id="seed"
                type="number"
                placeholder="from episode seed"
                value={population.seed ?? ""}
                onChange={(e) =>
                  mutate({ seed: e.target.value === "" ? null : Number(e.target.value) })
                }
              />
              <p className="persona-hint">Pin to hold the cast fixed as the episode seed varies.</p>
            </div>
          </div>

          <div>
            <Label htmlFor="driver">Who decides each turn</Label>
            <select
              id="driver"
              className="persona-select"
              value={population.driver}
              onChange={(e) => mutate({ driver: e.target.value })}
            >
              {DRIVERS.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
              {!DRIVERS.some((d) => d.id === population.driver) && (
                <option value={population.driver}>{population.driver}</option>
              )}
            </select>
            {driverHint && <p className="persona-hint">{driverHint}</p>}
          </div>
        </CardContent>
      </Card>

      <section className="persona-roster">
        <div className="persona-roster__head">
          <h2>Roster</h2>
          <div className="persona-roster__add">
            <select
              className="persona-select"
              defaultValue=""
              onChange={(e) => {
                if (e.target.value) addArchetype(e.target.value);
                e.target.value = "";
              }}
              aria-label="Add from the archetype library"
            >
              <option value="">Add from library…</option>
              {archetypes.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name} — {a.role}
                </option>
              ))}
            </select>
            <Button variant="outline" onClick={addBlank}>
              Add blank
            </Button>
          </div>
        </div>

        {inertCount > 0 && !unbound && (
          <p className="persona-warning">
            {inertCount} {inertCount === 1 ? "person has" : "people have"} no allowed
            actions and will never act. Give them at least one action below.
          </p>
        )}

        {population.roster.length === 0 ? (
          <p className="persona-empty">
            No one here yet. Add someone from the library to start with a written
            disposition, or add a blank persona to write your own.
          </p>
        ) : (
          population.roster.map((persona, index) => (
            <PersonaCard
              key={`${persona.id}-${index}`}
              persona={persona}
              index={index}
              actions={actions}
              onChange={(next) => {
                const roster = [...population.roster];
                roster[index] = next;
                setRoster(roster);
              }}
              onRemove={() => setRoster(population.roster.filter((_, i) => i !== index))}
            />
          ))
        )}
      </section>

      <Card>
        <CardHeader className="persona-preview__head">
          <CardTitle className="persona-panel__title">Who a seed produces</CardTitle>
          <div className="persona-preview__seed">
            <Label htmlFor="preview-seed">Seed</Label>
            <Input
              id="preview-seed"
              type="number"
              value={previewSeed}
              onChange={(e) => setPreviewSeed(Number(e.target.value))}
            />
          </div>
        </CardHeader>
        <CardContent>
          {previewError && <p className="persona-error">{previewError}</p>}
          {preview?.warnings.map((w) => (
            <p key={w} className="persona-warning">
              {w}
            </p>
          ))}
          {preview && preview.roster.length > 0 ? (
            <table className="persona-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Role</th>
                  <th>Can do</th>
                  <th>Wakes on</th>
                </tr>
              </thead>
              <tbody>
                {preview.roster.map((p) => (
                  <tr key={p.id}>
                    <td>{p.name}</td>
                    <td>{p.role || "—"}</td>
                    <td>{p.behavior.allowed_actions.join(", ") || "nothing"}</td>
                    <td>{p.behavior.wake_on.join(", ") || "anything"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            !previewError && <p className="persona-empty">This seed produces nobody.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
