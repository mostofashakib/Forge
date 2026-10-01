import type { Traits } from "@/lib/personaTraits";

// ---------------------------------------------------------------------------
// Shape of the `personas:` block, mirroring forge/personas/config.py
// ---------------------------------------------------------------------------

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

export interface Payload {
  personas: Population;
  environment_actions: string[];
}
