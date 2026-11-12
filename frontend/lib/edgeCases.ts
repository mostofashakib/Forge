/** Edge cases a task factory batch can plant. The backend says which ones each environment offers. */

export const EDGE_CASE_LABELS: Record<string, { label: string; desc: string }> = {
  boundary_conditions: { label: "Boundary values", desc: "Empty, zero, or at-the-limit inputs" },
  missing_deps: { label: "Missing pieces", desc: "Something the work needs must be created first" },
  conflicting_state: { label: "Conflicting state", desc: "Work left half done or in the way" },
  recovery: { label: "Recovery", desc: "The obvious approach fails visibly" },
};

/** The picked edge cases an environment still offers, in the order they were picked. */
export function keepOffered(picked: string[], offered: string[] | undefined): string[] {
  return picked.filter((name) => offered?.includes(name));
}

export function toggleEdgeCase(picked: string[], name: string): string[] {
  return picked.includes(name) ? picked.filter((item) => item !== name) : [...picked, name];
}
