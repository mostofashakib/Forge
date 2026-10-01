/** Who decides each simulated person's turn. The backend accepts any `provider:model` id. */
export interface PersonaDriver {
  id: string;
  label: string;
}

export const PERSONA_DRIVERS: readonly PersonaDriver[] = [
  { id: "scripted", label: "Scripted" },
  { id: "anthropic:claude-sonnet-5", label: "Claude Sonnet 5" },
  { id: "anthropic:claude-opus-5", label: "Claude Opus 5" },
  { id: "openai:gpt-4.1", label: "GPT-4.1" },
];

/** The known drivers, plus the current one when a saved config names a model not listed. */
export function driverChoices(current: string): PersonaDriver[] {
  const known = PERSONA_DRIVERS.some((driver) => driver.id === current);
  return known || !current ? [...PERSONA_DRIVERS] : [...PERSONA_DRIVERS, { id: current, label: current }];
}

/** Mirrors the backend: these ids run the free, offline scripted driver. */
export function isScriptedDriver(id: string): boolean {
  return id === "" || id === "scripted" || id === "none";
}
