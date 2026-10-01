"use client";

import { Label } from "@/components/ui/label";

export function ActionPicker({
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
