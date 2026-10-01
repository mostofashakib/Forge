"use client";

import type { ReactNode } from "react";
import { Cloud, Cpu, Zap } from "lucide-react";

export type InferenceMode = "auto" | "local_gpu" | "api_gateway";

interface Option {
  mode: InferenceMode;
  icon: ReactNode;
  label: string;
  hint: string;
}

/**
 * The Auto / Local GPU / API Gateway radio group used wherever a run picks
 * where inference happens. `gatewayHint` lets a page describe the gateway the
 * way it is used there.
 */
export function InferenceModePicker({
  name,
  value,
  onChange,
  disabled,
  gatewayHint = "Hosted vLLM / Ollama",
}: {
  name: string;
  value: InferenceMode;
  onChange: (mode: InferenceMode) => void;
  disabled: boolean;
  gatewayHint?: string;
}) {
  const options: Option[] = [
    { mode: "auto", icon: <Zap size={13} className="text-amber-500" />, label: "Auto", hint: "Detect local GPU or fallback" },
    { mode: "local_gpu", icon: <Cpu size={13} className="text-emerald-500" />, label: "Local GPU", hint: "CUDA / Apple Silicon MPS" },
    { mode: "api_gateway", icon: <Cloud size={13} className="text-blue-500" />, label: "API Gateway", hint: gatewayHint },
  ];

  return (
    <div className="benchmark-field">
      <div className="benchmark-field__label">
        <span>Inference</span>
        <small>Local GPU vs Cloud Gateway</small>
      </div>
      <div className="benchmark-domain-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
        {options.map((option) => (
          <label key={option.mode} className="benchmark-domain">
            <input
              type="radio"
              name={name}
              checked={value === option.mode}
              onChange={() => onChange(option.mode)}
              disabled={disabled}
            />
            <span className="benchmark-domain__check">✓</span>
            <span>
              <div className="flex items-center gap-1">
                {option.icon}
                <strong>{option.label}</strong>
              </div>
              <small>{option.hint}</small>
            </span>
          </label>
        ))}
      </div>
    </div>
  );
}
