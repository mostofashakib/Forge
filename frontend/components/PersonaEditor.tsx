"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { API_BASE } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { PersonaCard } from "@/components/persona-editor/PersonaCard";
import { DEFAULT_TRAITS } from "@/lib/personaTraits";
import type { Behavior, PersonaEntry, Population, Payload } from "@/components/persona-editor/types";

export type { Behavior, PersonaEntry, Population } from "@/components/persona-editor/types";

const DRIVERS = [
  { id: "scripted", label: "Scripted", hint: "Free, offline, byte-reproducible. Personas act at human-like times but say nothing new." },
  { id: "anthropic:claude-sonnet-5", label: "Claude Sonnet 5", hint: "A model decides each turn, inside the action space you set below." },
  { id: "anthropic:claude-opus-5", label: "Claude Opus 5", hint: "A model decides each turn, inside the action space you set below." },
  { id: "openai:gpt-4.1", label: "GPT-4.1", hint: "A model decides each turn, inside the action space you set below." },
];

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
          <Button variant="outline" onClick={addBlank}>
            Add a person
          </Button>
        </div>

        {inertCount > 0 && !unbound && (
          <p className="persona-warning">
            {inertCount} {inertCount === 1 ? "person has" : "people have"} no allowed
            actions and will never act. Give them at least one action below.
          </p>
        )}

        {population.roster.length === 0 ? (
          <p className="persona-empty">
            No one here yet. Add a person and describe who they are, how they
            behave, and what they&apos;re allowed to do.
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
