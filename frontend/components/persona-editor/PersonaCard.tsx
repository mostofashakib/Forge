"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { TRAIT_COPY, type Traits } from "@/lib/personaTraits";
import { TraitSlider } from "./TraitSlider";
import { ActionPicker } from "./ActionPicker";

import type { Behavior, PersonaEntry } from "./types";

export function PersonaCard({
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
