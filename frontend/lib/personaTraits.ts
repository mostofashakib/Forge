// Persona traits as the `personas:` config defines them (forge/personas/config.py),
// shared by the cast picker on the builder and the editor on the personas page.

export interface Traits {
  responsiveness: number;
  initiative: number;
  verbosity: number;
  diligence: number;
  formality: number;
  patience: number;
}

export const DEFAULT_TRAITS: Traits = {
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
export const TRAIT_COPY: Array<{ key: keyof Traits; label: string; low: string; high: string }> = [
  { key: "responsiveness", label: "Responsiveness", low: "Replies late", high: "Replies at once" },
  { key: "initiative", label: "Initiative", low: "Waits to be asked", high: "Speaks up unprompted" },
  { key: "verbosity", label: "Verbosity", low: "A sentence", high: "Full explanation" },
  { key: "diligence", label: "Diligence", low: "Misses details", high: "Checks everything" },
  { key: "formality", label: "Formality", low: "Shorthand", high: "Formal prose" },
  { key: "patience", label: "Patience", low: "Escalates fast", high: "Never chases" },
];
