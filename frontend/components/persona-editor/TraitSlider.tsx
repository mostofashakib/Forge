"use client";

import { Label } from "@/components/ui/label";

export function TraitSlider({
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
