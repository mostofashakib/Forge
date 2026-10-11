"use client";

import { useState } from "react";
import { API_BASE } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import type { PersonaEntry } from "./types";

const ROLE_PRESETS = [
  "Senior Auditor",
  "Triage Nurse",
  "Impatient Customer",
  "Security Architect",
  "Compliance Officer",
  "Support Specialist",
];

interface PersonaGeneratorCardProps {
  envName: string;
  environmentActions: string[];
  onAddPersonas: (newPersonas: PersonaEntry[]) => void;
  onReplaceRoster: (newPersonas: PersonaEntry[]) => void;
}

export function PersonaGeneratorCard({
  envName,
  environmentActions,
  onAddPersonas,
  onReplaceRoster,
}: PersonaGeneratorCardProps) {
  const [role, setRole] = useState("Senior Auditor");
  const [context, setContext] = useState(`Production ${envName} operations`);
  const [count, setCount] = useState(1);
  const [selectedActions, setSelectedActions] = useState<string[]>(environmentActions);
  const [seed, setSeed] = useState<string>("");
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastSuccess, setLastSuccess] = useState<string | null>(null);

  const toggleAction = (act: string) => {
    setSelectedActions((prev) =>
      prev.includes(act) ? prev.filter((a) => a !== act) : [...prev, act]
    );
  };

  const selectAllActions = () => setSelectedActions(environmentActions);
  const clearActions = () => setSelectedActions([]);

  async function handleGenerate(mode: "append" | "replace") {
    setGenerating(true);
    setError(null);
    setLastSuccess(null);

    try {
      const res = await fetch(`${API_BASE}/api/envs/${envName}/personas/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          role: role.trim() || "Colleague",
          context: context.trim(),
          count: Math.max(1, count),
          allowed_actions: selectedActions,
          seed: seed !== "" ? Number(seed) : undefined,
        }),
      });

      const body = await res.json();
      if (!res.ok) {
        setError(body.detail ?? "Failed to synthesize personas.");
        return;
      }

      const generated: PersonaEntry[] = body.personas;
      if (mode === "append") {
        onAddPersonas(generated);
        setLastSuccess(`Added ${generated.length} synthesized persona${generated.length > 1 ? "s" : ""} to roster.`);
      } else {
        onReplaceRoster(generated);
        setLastSuccess(`Replaced roster with ${generated.length} newly synthesized persona${generated.length > 1 ? "s" : ""}.`);
      }
    } catch {
      setError("Network error: Could not reach the persona generator service.");
    } finally {
      setGenerating(false);
    }
  }

  return (
    <Card className="border-primary/20 shadow-xs">
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <CardTitle className="text-base font-semibold flex items-center gap-2">
              <span>Persona Generator</span>
              <Badge variant="outline" className="text-xs bg-primary/5 text-primary border-primary/30">
                Generator
              </Badge>
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-0.5">
              Generate participants with workplace context, private knowledge, and behavior traits.
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <div className="rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
            {error}
          </div>
        )}

        {lastSuccess && (
          <div className="rounded-md border border-primary/40 bg-primary/5 px-3 py-2 text-xs text-primary font-medium">
            {lastSuccess}
          </div>
        )}

        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <Label htmlFor="persona-role" className="text-xs font-medium">
              Role or Team Description
            </Label>
            <Input
              id="persona-role"
              className="mt-1 h-8 text-xs"
              placeholder="e.g. Senior Auditor, Triage Nurse, Impatient Customer"
              value={role}
              onChange={(e) => setRole(e.target.value)}
            />
            <div className="flex flex-wrap gap-1 mt-1.5">
              {ROLE_PRESETS.map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => setRole(preset)}
                  className="rounded-xs border border-border px-1.5 py-0.5 text-[0.65rem] text-muted-foreground hover:border-primary hover:text-foreground transition-colors"
                >
                  {preset}
                </button>
              ))}
            </div>
          </div>

          <div>
            <Label htmlFor="persona-context" className="text-xs font-medium">
              Scenario / Environment Context
            </Label>
            <Input
              id="persona-context"
              className="mt-1 h-8 text-xs"
              placeholder="e.g. Critical quarterly review, midnight ICU shift"
              value={context}
              onChange={(e) => setContext(e.target.value)}
            />
            <p className="text-[0.65rem] text-muted-foreground mt-1">
              Sets background context, private knowledge, and communication style.
            </p>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          <div>
            <Label htmlFor="persona-count" className="text-xs font-medium">
              Number of Personas
            </Label>
            <Input
              id="persona-count"
              type="number"
              min={1}
              max={10}
              className="mt-1 h-8 text-xs"
              value={count}
              onChange={(e) => setCount(Math.max(1, parseInt(e.target.value) || 1))}
            />
            <p className="text-[0.65rem] text-muted-foreground mt-1">
              {count === 1 ? "Generates 1 detailed persona." : `Generates a team of ${count} distinct personas.`}
            </p>
          </div>

          <div>
            <Label htmlFor="persona-seed" className="text-xs font-medium">
              Generation Seed (Optional)
            </Label>
            <Input
              id="persona-seed"
              type="number"
              className="mt-1 h-8 text-xs"
              placeholder="Random seed"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
            />
            <p className="text-[0.65rem] text-muted-foreground mt-1">
              Pin seed for deterministic reproducibility.
            </p>
          </div>

          <div>
            <div className="flex items-center justify-between">
              <Label className="text-xs font-medium">Allowed Actions</Label>
              <div className="space-x-1">
                <button
                  type="button"
                  onClick={selectAllActions}
                  className="text-[0.65rem] text-primary hover:underline"
                >
                  All
                </button>
                <span className="text-[0.65rem] text-muted-foreground">·</span>
                <button
                  type="button"
                  onClick={clearActions}
                  className="text-[0.65rem] text-muted-foreground hover:underline"
                >
                  None
                </button>
              </div>
            </div>
            <div className="flex flex-wrap gap-1 mt-1 max-h-20 overflow-y-auto">
              {environmentActions.length === 0 ? (
                <span className="text-[0.65rem] text-muted-foreground">No actions declared in environment</span>
              ) : (
                environmentActions.map((act) => {
                  const active = selectedActions.includes(act);
                  return (
                    <button
                      key={act}
                      type="button"
                      onClick={() => toggleAction(act)}
                      className={`rounded-xs border px-1.5 py-0.5 text-[0.65rem] font-mono transition-colors ${
                        active
                          ? "border-primary bg-primary/10 text-primary font-medium"
                          : "border-border text-muted-foreground hover:border-foreground/30"
                      }`}
                    >
                      {act}
                    </button>
                  );
                })
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2 pt-2 border-t border-border/50">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={generating}
            onClick={() => handleGenerate("replace")}
          >
            {generating ? "Synthesizing…" : "Generate & Replace Roster"}
          </Button>
          <Button
            type="button"
            size="sm"
            disabled={generating}
            onClick={() => handleGenerate("append")}
          >
            {generating ? "Synthesizing…" : `Generate & Add to Roster (${count})`}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
