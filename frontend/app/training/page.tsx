"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { Cpu, Cloud, Zap, CheckCircle2, AlertCircle, ArrowUpRight, Play, RefreshCw } from "lucide-react";

interface TrainingRun {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  objective: "grpo" | "dpo" | "sft" | "ppo";
  training_mode?: "online" | "offline";
  base_model: string;
  data_dir: string;
  output_dir: string;
  checkpoint_path: string | null;
  inference_mode: string;
  max_steps: number;
  num_examples: number | null;
  mean_reward: number | null;
  error: string | null;
  created_at: string | null;
  completed_at: string | null;
}

interface Checkpoint {
  directory: string;
  objective: string;
  training_mode?: "online" | "offline";
  base_model: string;
  num_examples: number;
  mean_reward: number;
  created_at: string;
  run_id: string;
}

interface DataSource {
  id: string;
  batch_id?: string;
  label: string;
  source_type: "generator" | "environment" | "directory";
  data_type: string;
  env_name: string;
  path: string;
  suggested_objective?: "grpo" | "dpo" | "sft" | "ppo";
}

interface HardwareInfo {
  hardware: {
    cuda_available: boolean;
    mps_available: boolean;
    device_count: number;
    devices: { id: number; name: string; total_memory_mb?: number }[];
  };
  default_mode: string;
  api_gateway: {
    endpoint_url: string;
    has_api_key: boolean;
  };
}

export default function TrainingPage() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [selectedSource, setSelectedSource] = useState<string>("exports");
  const [loading, setLoading] = useState(true);

  // Form State
  const [objective, setObjective] = useState<"grpo" | "dpo" | "sft" | "ppo">("grpo");
  const [trainingMode, setTrainingMode] = useState<"online" | "offline">("online");
  const [baseModel, setBaseModel] = useState("Qwen/Qwen2.5-Coder-7B-Instruct");
  const [dataDir, setDataDir] = useState("exports");
  const [outputDir, setOutputDir] = useState("forge_policy");
  const [maxSteps, setMaxSteps] = useState(500);
  const [inferenceMode, setInferenceMode] = useState<"auto" | "local_gpu" | "api_gateway">("auto");
  const [apiGatewayUrl, setApiGatewayUrl] = useState("https://api.openai.com/v1");
  const [gpuPrecision, setGpuPrecision] = useState<"bf16" | "fp16" | "fp32" | "fp8">("bf16");
  const [submitting, setSubmitting] = useState(false);
  const [activeTab, setActiveTab] = useState<"runs" | "checkpoints">("runs");
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const logRef = useRef<HTMLDivElement>(null);

  const appendLog = (line: string) => {
    setLogs((prev) => [...prev.slice(-199), line]);
    requestAnimationFrame(() => {
      if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    });
  };

  const loadData = useCallback(async () => {
    try {
      const [runsRes, cpRes, hwRes, dsRes] = await Promise.all([
        fetch(`${API_BASE}/api/training/runs`, { cache: "no-store" }),
        fetch(`${API_BASE}/api/training/checkpoints`, { cache: "no-store" }),
        fetch(`${API_BASE}/api/training/hardware`, { cache: "no-store" }),
        fetch(`${API_BASE}/api/training/data-sources`, { cache: "no-store" }),
      ]);

      if (runsRes.ok) setRuns(await runsRes.json());
      if (cpRes.ok) setCheckpoints(await cpRes.json());
      if (hwRes.ok) setHardware(await hwRes.json());
      if (dsRes.ok) setDataSources(await dsRes.json());
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, [loadData]);

  async function handleLaunch() {
    setSubmitting(true);
    setError(null);
    setSuccessMsg(null);
    appendLog(`[training] initiating ${objective.toUpperCase()} (${trainingMode.toUpperCase()} mode) training with base model ${baseModel}...`);

    try {
      const res = await fetch(`${API_BASE}/api/training/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          base_model: baseModel,
          data_dir: dataDir,
          output_dir: outputDir,
          objective,
          training_mode: trainingMode,
          max_steps: Number(maxSteps),
          inference_mode: inferenceMode,
          api_gateway_url: inferenceMode === "api_gateway" ? apiGatewayUrl : undefined,
          gpu_precision: gpuPrecision,
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Training launch failed with HTTP ${res.status}`);
      }

      const run = await res.json();
      setSuccessMsg(`Training run ${run.run_id} launched successfully!`);
      appendLog(`[training] run ${run.run_id} enqueued in background`);
      loadData();
    } catch (err) {
      const msg = (err as Error).message;
      setError(msg);
      appendLog(`[error] ${msg}`);
    } finally {
      setSubmitting(false);
    }
  }

  const activeRunsCount = runs.filter((r) => r.status === "running" || r.status === "queued").length;
  const systemState = activeRunsCount > 0 ? "running" : "idle";

  return (
    <div className="benchmark-run tasks-page">
      <header className="benchmark-run__hero">
        <div className="benchmark-run__hero-copy">
          <span className="benchmark-run__eyebrow">Policy Training & Alignment</span>
          <h1>
            FORGE THE<br />
            <em>POLICY.</em>
          </h1>
          <p>
            Train agents from graded rollouts and synthetic preference pairs. Optimize policies with
            Group Relative Policy Optimization (GRPO) or Direct Preference Optimization (DPO) in
            either Online (same model family) or Offline (cross-model distillation) mode.
          </p>
        </div>
        <div className="benchmark-run__readout" aria-label="Training system overview">
          <div>
            <span>Active Runs</span>
            <strong>{String(activeRunsCount).padStart(2, "0")}</strong>
          </div>
          <div>
            <span>Objective</span>
            <strong>{objective.toUpperCase()}</strong>
          </div>
          <div>
            <span>Mode</span>
            <strong className="uppercase">{trainingMode}</strong>
          </div>
          <div>
            <span>Checkpoints</span>
            <strong>{String(checkpoints.length).padStart(2, "0")}</strong>
          </div>
          <div className={`benchmark-run__state benchmark-run__state--${systemState}`}>
            <span>System State</span>
            <strong>
              <i />
              {systemState}
            </strong>
          </div>
        </div>
      </header>

      {/* Main Workbench Layout */}
      <div className="benchmark-workbench">
        {/* Left Column: Launch Configuration */}
        <section className="benchmark-config">
          <div className="benchmark-panel__heading">
            <div>
              <span>01</span>
              <h2>Launch Run</h2>
            </div>
          </div>

          {/* Objective Selection */}
          <div className="benchmark-field">
            <div className="benchmark-field__label">
              <span>Training Strategy / Algorithm</span>
              <small>Objective & Loss Formulation</small>
            </div>
            <div className="benchmark-domain-grid" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
              <label className="benchmark-domain">
                <input
                  type="radio"
                  name="objective"
                  checked={objective === "grpo"}
                  onChange={() => setObjective("grpo")}
                  disabled={submitting}
                />
                <span className="benchmark-domain__check">✓</span>
                <span>
                  <strong>GRPO</strong>
                  <small>Group Relative Advantage (from rollout groups)</small>
                </span>
              </label>

              <label className="benchmark-domain">
                <input
                  type="radio"
                  name="objective"
                  checked={objective === "dpo"}
                  onChange={() => setObjective("dpo")}
                  disabled={submitting}
                />
                <span className="benchmark-domain__check">✓</span>
                <span>
                  <strong>DPO</strong>
                  <small>Direct Preference Optimization (from pairs)</small>
                </span>
              </label>

              <label className="benchmark-domain">
                <input
                  type="radio"
                  name="objective"
                  checked={objective === "ppo"}
                  onChange={() => setObjective("ppo")}
                  disabled={submitting}
                />
                <span className="benchmark-domain__check">✓</span>
                <span>
                  <strong>PPO</strong>
                  <small>Proximal Policy Optimization (clipped surrogate)</small>
                </span>
              </label>

              <label className="benchmark-domain">
                <input
                  type="radio"
                  name="objective"
                  checked={objective === "sft"}
                  onChange={() => setObjective("sft")}
                  disabled={submitting}
                />
                <span className="benchmark-domain__check">✓</span>
                <span>
                  <strong>SFT</strong>
                  <small>Supervised Fine-Tuning (from demonstrations)</small>
                </span>
              </label>
            </div>
          </div>

          {/* Training Mode: Online vs Offline */}
          <div className="benchmark-field">
            <div className="benchmark-field__label">
              <span>Training Mode</span>
              <small>Model & Data Parity</small>
            </div>
            <div className="benchmark-domain-grid" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
              <label className="benchmark-domain">
                <input
                  type="radio"
                  name="training_mode"
                  checked={trainingMode === "online"}
                  onChange={() => setTrainingMode("online")}
                  disabled={submitting}
                />
                <span className="benchmark-domain__check">✓</span>
                <span>
                  <strong>Online Mode</strong>
                  <small>Same model / family used to generate data</small>
                </span>
              </label>

              <label className="benchmark-domain">
                <input
                  type="radio"
                  name="training_mode"
                  checked={trainingMode === "offline"}
                  onChange={() => setTrainingMode("offline")}
                  disabled={submitting}
                />
                <span className="benchmark-domain__check">✓</span>
                <span>
                  <strong>Offline Mode</strong>
                  <small>Cross-model / different family distillation</small>
                </span>
              </label>
            </div>
          </div>

          {/* Inference Target (GPU Contract) */}
          <div className="benchmark-field">
            <div className="benchmark-field__label">
              <span>Inference</span>
              <small>Local GPU vs Cloud Gateway</small>
            </div>
            <div className="benchmark-domain-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
              <label className="benchmark-domain">
                <input
                  type="radio"
                  name="inference_mode"
                  checked={inferenceMode === "auto"}
                  onChange={() => setInferenceMode("auto")}
                  disabled={submitting}
                />
                <span className="benchmark-domain__check">✓</span>
                <span>
                  <div className="flex items-center gap-1">
                    <Zap size={13} className="text-amber-500" />
                    <strong>Auto</strong>
                  </div>
                  <small>Detect local GPU or fallback</small>
                </span>
              </label>

              <label className="benchmark-domain">
                <input
                  type="radio"
                  name="inference_mode"
                  checked={inferenceMode === "local_gpu"}
                  onChange={() => setInferenceMode("local_gpu")}
                  disabled={submitting}
                />
                <span className="benchmark-domain__check">✓</span>
                <span>
                  <div className="flex items-center gap-1">
                    <Cpu size={13} className="text-emerald-500" />
                    <strong>Local GPU</strong>
                  </div>
                  <small>CUDA / Apple Silicon MPS</small>
                </span>
              </label>

              <label className="benchmark-domain">
                <input
                  type="radio"
                  name="inference_mode"
                  checked={inferenceMode === "api_gateway"}
                  onChange={() => setInferenceMode("api_gateway")}
                  disabled={submitting}
                />
                <span className="benchmark-domain__check">✓</span>
                <span>
                  <div className="flex items-center gap-1">
                    <Cloud size={13} className="text-blue-500" />
                    <strong>API Gateway</strong>
                  </div>
                  <small>Cloud OpenAI/REST gateway</small>
                </span>
              </label>
            </div>
          </div>

          {/* Base Model */}
          <label className="benchmark-field">
            <span className="benchmark-field__label">
              <span>Base Policy Model</span>
              <small>HuggingFace ID or Local Model</small>
            </span>
            <input
              type="text"
              value={baseModel}
              onChange={(e) => setBaseModel(e.target.value)}
              disabled={submitting}
              className="benchmark-input benchmark-input--mono"
              placeholder="e.g. Qwen/Qwen2.5-Coder-7B-Instruct"
            />
          </label>

          {/* Data Source Selector from Generator or Environments */}
          <div className="benchmark-field">
            <div className="benchmark-field__label">
              <span>Dataset / Data Source</span>
              <small>Select from Data Generator or Environments</small>
            </div>
            <select
              value={selectedSource}
              onChange={(e) => {
                const val = e.target.value;
                setSelectedSource(val);
                if (val !== "custom") {
                  const src = dataSources.find((s) => s.id === val);
                  if (src) {
                    setDataDir(src.path);
                    if (src.suggested_objective) {
                      setObjective(src.suggested_objective);
                    }
                  }
                }
              }}
              disabled={submitting}
              className="benchmark-input text-xs font-mono mb-2"
            >
              <optgroup label="Synthetic Data Generator Batches">
                {dataSources.filter((s) => s.source_type === "generator").length === 0 ? (
                  <option disabled value="">No generator batches yet (generate in Generator tab)</option>
                ) : (
                  dataSources
                    .filter((s) => s.source_type === "generator")
                    .map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.label}
                      </option>
                    ))
                )}
              </optgroup>
              <optgroup label="Environment Rollouts">
                {dataSources
                  .filter((s) => s.source_type === "environment")
                  .map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.label}
                    </option>
                  ))}
              </optgroup>
              <optgroup label="Direct File System">
                <option value="exports">Default Exports Directory (exports/)</option>
                <option value="custom">Custom Directory Path...</option>
              </optgroup>
            </select>
          </div>

          {/* Data and Output Directory */}
          <div className="grid grid-cols-2 gap-3">
            <label className="benchmark-field">
              <span className="benchmark-field__label">
                <span>Data Directory Path</span>
                <small>Source rollouts / pairs / SFT</small>
              </span>
              <input
                type="text"
                value={dataDir}
                onChange={(e) => {
                  setDataDir(e.target.value);
                  setSelectedSource("custom");
                }}
                disabled={submitting}
                className="benchmark-input benchmark-input--mono text-xs"
                placeholder="exports"
              />
            </label>

            <label className="benchmark-field">
              <span className="benchmark-field__label">
                <span>Output Directory</span>
                <small>Checkpoint destination</small>
              </span>
              <input
                type="text"
                value={outputDir}
                onChange={(e) => setOutputDir(e.target.value)}
                disabled={submitting}
                className="benchmark-input benchmark-input--mono text-xs"
              />
            </label>
          </div>

          {/* API Gateway details if selected */}
          {inferenceMode === "api_gateway" && (
            <label className="benchmark-field">
              <span className="benchmark-field__label">
                <span>API Gateway URL</span>
                <small>OpenAI-compatible endpoint</small>
              </span>
              <input
                type="url"
                value={apiGatewayUrl}
                onChange={(e) => setApiGatewayUrl(e.target.value)}
                disabled={submitting}
                className="benchmark-input benchmark-input--mono text-xs"
              />
            </label>
          )}

          {/* Precision for Local GPU */}
          {inferenceMode !== "api_gateway" && (
            <div className="benchmark-field">
              <div className="benchmark-field__label">
                <span>Precision</span>
                <small>Hardware acceleration</small>
              </div>
              <div className="flex gap-2">
                {(["bf16", "fp16", "fp8", "fp32"] as const).map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => setGpuPrecision(p)}
                    className={`px-3 py-1 rounded border text-xs font-mono transition-colors ${
                      gpuPrecision === p
                        ? "bg-primary text-primary-foreground border-primary"
                        : "border-border text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {p.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Max Steps Slider */}
          <div className="benchmark-field">
            <div className="benchmark-field__label">
              <span>Max Training Steps</span>
              <strong>{maxSteps}</strong>
            </div>
            <input
              type="range"
              min={50}
              max={2000}
              step={50}
              value={maxSteps}
              onChange={(e) => setMaxSteps(Number(e.target.value))}
              disabled={submitting}
              className="benchmark-range"
            />
            <div className="benchmark-range__legend">
              <span>50 steps</span>
              <span>2,000 steps</span>
            </div>
          </div>

          <button
            type="button"
            onClick={handleLaunch}
            disabled={submitting || !baseModel.trim() || !dataDir.trim()}
            className="tasks-button flex items-center justify-center gap-2 w-full py-3"
          >
            {submitting ? (
              <>
                <RefreshCw size={15} className="animate-spin" />
                Launching Training...
              </>
            ) : (
              <>
                <Play size={15} />
                Launch Policy Training →
              </>
            )}
          </button>

          {successMsg && (
            <div className="benchmark-notice benchmark-notice--success">
              <div>
                <p>Training Enqueued</p>
                <span>{successMsg}</span>
              </div>
            </div>
          )}

          {error && (
            <div className="benchmark-notice benchmark-notice--error">
              <div>
                <p>Launch Failed</p>
                <span>{error}</span>
              </div>
            </div>
          )}
        </section>

        {/* Right Column: Hardware Telemetry & Live Terminal Console */}
        <section className="benchmark-progress">
          <div className="benchmark-panel__heading">
            <div>
              <span>02</span>
              <h2>Compute Status & Live Log</h2>
            </div>
          </div>

          {/* Hardware Diagnostic Card */}
          {hardware && (
            <div className="border rounded-xl p-4 bg-muted/20 space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-muted-foreground uppercase tracking-wider">
                  Hardware Probing
                </span>
                <span className="font-mono text-emerald-500 flex items-center gap-1">
                  <CheckCircle2 size={13} />
                  Ready
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="border rounded-lg p-2.5 bg-card">
                  <div className="text-muted-foreground">Local Acceleration</div>
                  <strong className="text-sm font-mono block mt-1">
                    {hardware.hardware.cuda_available
                      ? `CUDA (${hardware.hardware.device_count} GPUs)`
                      : hardware.hardware.mps_available
                      ? "Apple Silicon (MPS)"
                      : "CPU Core"}
                  </strong>
                </div>

                <div className="border rounded-lg p-2.5 bg-card">
                  <div className="text-muted-foreground">Cloud API Gateway</div>
                  <strong className="text-sm font-mono block mt-1 truncate" title={hardware.api_gateway.endpoint_url}>
                    {hardware.api_gateway.endpoint_url.replace("https://", "")}
                  </strong>
                </div>
              </div>

              {hardware.hardware.devices.length > 0 && (
                <div className="text-[11px] font-mono text-muted-foreground space-y-1">
                  {hardware.hardware.devices.map((d) => (
                    <div key={d.id} className="flex justify-between border-t pt-1 border-border/40">
                      <span>Device {d.id}: {d.name}</span>
                      {d.total_memory_mb && <span>{d.total_memory_mb.toLocaleString()} MB</span>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Live Log Console */}
          <div className="tasks-log-terminal" ref={logRef}>
            {logs.length === 0 ? (
              <div className="tasks-log-empty">
                <span>&gt;_</span>
                <p>
                  Training runner idle.
                  <br />
                  Launch a training run to stream execution logs
                  <span className="animate-pulse">_</span>
                </p>
              </div>
            ) : (
              logs.map((line, i) => (
                <p key={i}>
                  <span>{String(i + 1).padStart(3, "0")}</span>
                  {line}
                </p>
              ))
            )}
          </div>
        </section>
      </div>

      {/* Historical Runs & Saved Checkpoints Section */}
      <section className="benchmark-config">
        <div className="flex items-center justify-between border-b pb-3">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setActiveTab("runs")}
              className={`text-lg font-semibold tracking-tight transition-colors ${
                activeTab === "runs" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Training Runs ({runs.length})
            </button>
            <span className="text-muted-foreground/40">•</span>
            <button
              onClick={() => setActiveTab("checkpoints")}
              className={`text-lg font-semibold tracking-tight transition-colors ${
                activeTab === "checkpoints" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Saved Checkpoints ({checkpoints.length})
            </button>
          </div>
          <button
            onClick={loadData}
            className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1.5"
          >
            <RefreshCw size={12} />
            Refresh
          </button>
        </div>

        {activeTab === "runs" ? (
          runs.length === 0 ? (
            <p className="tasks-empty">No training runs recorded yet. Configure and launch a run above.</p>
          ) : (
            <div className="tasks-table-wrap">
              <table className="tasks-table">
                <thead>
                  <tr>
                    <th>Run ID</th>
                    <th>Objective</th>
                    <th>Mode</th>
                    <th>Base Model</th>
                    <th>Compute</th>
                    <th>Status</th>
                    <th>Examples</th>
                    <th>Mean Reward</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.id}>
                      <td className="tasks-table__version font-mono text-xs">{r.id}</td>
                      <td>
                        <span className="tasks-difficulty tasks-difficulty--1 uppercase font-semibold">
                          {r.objective}
                        </span>
                      </td>
                      <td>
                        <span
                          className={`text-[11px] font-mono uppercase px-2 py-0.5 rounded border ${
                            (r.training_mode || "online") === "online"
                              ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                              : "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20"
                          }`}
                        >
                          {r.training_mode || "online"}
                        </span>
                      </td>
                      <td className="font-mono text-xs">{r.base_model}</td>
                      <td>
                        <span className="text-xs text-muted-foreground font-mono">
                          {r.inference_mode === "api_gateway" ? "Cloud Gateway" : "Local GPU"}
                        </span>
                      </td>
                      <td>
                        <span className={`tasks-status tasks-status--${r.status === "completed" ? "complete" : r.status}`}>
                          {r.status}
                        </span>
                      </td>
                      <td className="tabular-nums">{r.num_examples ?? "—"}</td>
                      <td className="font-mono text-xs">
                        {r.mean_reward !== null ? (
                          <span className={r.mean_reward > 0 ? "text-emerald-500" : "text-muted-foreground"}>
                            {r.mean_reward > 0 ? "+" : ""}
                            {r.mean_reward.toFixed(3)}
                          </span>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className="text-xs text-muted-foreground">
                        {r.created_at ? new Date(r.created_at).toLocaleString() : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        ) : checkpoints.length === 0 ? (
          <p className="tasks-empty">No policy checkpoints found in output directories.</p>
        ) : (
          <div className="tasks-table-wrap">
            <table className="tasks-table">
              <thead>
                <tr>
                  <th>Directory</th>
                  <th>Objective</th>
                  <th>Mode</th>
                  <th>Base Model</th>
                  <th>Trained Examples</th>
                  <th>Mean Reward</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {checkpoints.map((cp, idx) => (
                  <tr key={idx}>
                    <td className="font-mono text-xs">{cp.directory}</td>
                    <td>
                      <span className="tasks-difficulty tasks-difficulty--2 uppercase font-semibold">
                        {cp.objective}
                      </span>
                    </td>
                    <td>
                      <span
                        className={`text-[11px] font-mono uppercase px-2 py-0.5 rounded border ${
                          (cp.training_mode || "online") === "online"
                            ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                            : "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20"
                        }`}
                      >
                        {cp.training_mode || "online"}
                      </span>
                    </td>
                    <td className="font-mono text-xs">{cp.base_model}</td>
                    <td className="tabular-nums">{cp.num_examples}</td>
                    <td className="font-mono text-xs text-emerald-500">
                      +{cp.mean_reward.toFixed(3)}
                    </td>
                    <td className="text-xs text-muted-foreground">
                      {cp.created_at ? new Date(cp.created_at).toLocaleString() : "—"}
                    </td>
                    <td>
                      <Link
                        href="/benchmark/eval"
                        className="text-xs text-primary hover:underline flex items-center gap-1"
                      >
                        Evaluate
                        <ArrowUpRight size={13} />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
