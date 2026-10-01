"use client";

import { useRef, useState } from "react";
import { InferenceModePicker, type InferenceMode } from "@/components/InferenceModePicker";
import { API_BASE, wsBase } from "@/lib/api";

type Phase = "idle" | "running" | "done" | "error";

interface TransferResult {
  model_path?: string;
  eval_suite?: string;
  task_completion_rate?: number;
  pass_at_1?: number;
  pass_at_3?: number;
  num_eval_tasks?: number;
  inference_mode?: string;
  device?: string;
  [key: string]: string | number | boolean | null | undefined;
}

const PRESET_MODELS = [
  "meta-llama/Llama-3.1-8B",
  "Qwen/Qwen2.5-7B-Instruct",
  "mistralai/Mistral-7B-v0.3",
];

export default function BenchmarkTransferPage() {
  const [baseModel, setBaseModel] = useState("meta-llama/Llama-3.1-8B");
  const [dataDir, setDataDir] = useState("benchmark_results/data");
  const [evalSuite, setEvalSuite] = useState("held-out-transfer");
  const [outputDir, setOutputDir] = useState("benchmark_results/transfer");
  const [maxSteps, setMaxSteps] = useState(500);
  const [seeds, setSeeds] = useState(3);
  const [inferenceMode, setInferenceMode] = useState<InferenceMode>("auto");

  const [phase, setPhase] = useState<Phase>("idle");
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TransferResult | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  function appendLog(line: string) {
    setLogs((current) => [...current.slice(-998), line]);
    requestAnimationFrame(() => {
      if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    });
  }

  async function loadResult(runId: string) {
    try {
      const response = await fetch(`${API_BASE}/api/benchmark/transfer/${runId}`);
      if (!response.ok) return;
      const payload = await response.json();
      if (payload.result) setResult(payload.result);
    } catch {
      // Ignored if run is still persisting
    }
  }

  async function runTransfer() {
    setPhase("running");
    setLogs([]);
    setResult(null);
    setError(null);

    const payload = {
      base_model: baseModel,
      data_dir: dataDir,
      output_dir: outputDir,
      eval_suite: evalSuite,
      max_steps: maxSteps,
      seeds,
      inference_mode: inferenceMode,
    };

    let runId: string;
    try {
      const response = await fetch(`${API_BASE}/api/benchmark/transfer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await response.text());
      ({ run_id: runId } = await response.json());
    } catch (requestError) {
      setPhase("error");
      setError(requestError instanceof Error ? requestError.message : "Could not start transfer benchmark");
      return;
    }

    try {
      const socket = new WebSocket(`${wsBase()}/api/benchmark/ws/progress/${runId}`);
      socket.onmessage = async (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.log) appendLog(message.log);
          if (message.result) setResult(message.result);
          if (message.done) {
            await loadResult(runId);
            setPhase("done");
            socket.close();
          }
          if (message.error) {
            setError(message.error);
            setPhase("error");
            socket.close();
          }
        } catch {
          // Keep receiving messages
        }
      };
      socket.onerror = () => {
        // Fallback polling if websocket disconnects
        setTimeout(() => loadResult(runId).then(() => setPhase("done")), 2000);
      };
    } catch {
      setTimeout(() => loadResult(runId).then(() => setPhase("done")), 2000);
    }
  }

  const isRunning = phase === "running";

  return (
    <div className="benchmark-run">
      <header className="benchmark-run__hero !grid-cols-1">
        <div className="benchmark-run__hero-copy flex flex-col items-center text-center">
          <span className="benchmark-run__eyebrow">Transfer benchmark / 03</span>
          <h1>EVALUATE THE <em>TRANSFER.</em></h1>
          <p className="mx-auto">
            Measure cross-distribution generalization on held-out tasks and external benchmark environments.
          </p>
        </div>
      </header>

      <div className="benchmark-workbench">
        <section className="benchmark-config">
          <div className="benchmark-panel__heading">
            <div><span>01</span><h2>Transfer setup</h2></div>
            <p>Define base model and evaluation suite</p>
          </div>

          <InferenceModePicker
            name="transfer_inference_mode"
            value={inferenceMode}
            onChange={setInferenceMode}
            disabled={isRunning}
          />

          {/* Base Model */}
          <div className="benchmark-field">
            <div className="benchmark-field__label">
              <span>Base Model / Policy Checkpoint</span>
              <small>HuggingFace ID or local path</small>
            </div>
            <input
              type="text"
              value={baseModel}
              onChange={(e) => setBaseModel(e.target.value)}
              disabled={isRunning}
              className="benchmark-input benchmark-input--mono"
              placeholder="e.g. meta-llama/Llama-3.1-8B"
            />
            <div className="mt-1 flex flex-wrap gap-1">
              {PRESET_MODELS.map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => setBaseModel(preset)}
                  disabled={isRunning}
                  className="border border-foreground/15 bg-card px-2 py-0.5 font-mono text-[0.6rem] text-muted-foreground hover:border-foreground/40 hover:text-foreground transition-colors"
                >
                  {preset.split("/").pop()}
                </button>
              ))}
            </div>
          </div>

          {/* Transfer Data Directory & Suite */}
          <div className="benchmark-field-row">
            <label className="benchmark-field">
              <span className="benchmark-field__label">
                <span>Dataset path</span>
                <small>source tasks</small>
              </span>
              <input
                type="text"
                value={dataDir}
                onChange={(e) => setDataDir(e.target.value)}
                disabled={isRunning}
                className="benchmark-input benchmark-input--mono"
              />
            </label>

            <label className="benchmark-field">
              <span className="benchmark-field__label">
                <span>Evaluation suite</span>
                <small>target benchmark</small>
              </span>
              <input
                type="text"
                value={evalSuite}
                onChange={(e) => setEvalSuite(e.target.value)}
                disabled={isRunning}
                className="benchmark-input benchmark-input--mono"
              />
            </label>
          </div>

          {/* Steps & Seeds */}
          <div className="benchmark-field-row">
            <label className="benchmark-field">
              <span className="benchmark-field__label">
                <span>Max eval steps</span>
              </span>
              <input
                type="number"
                min={50}
                max={5000}
                value={maxSteps}
                onChange={(e) => setMaxSteps(Number(e.target.value))}
                disabled={isRunning}
                className="benchmark-input"
              />
            </label>

            <label className="benchmark-field">
              <span className="benchmark-field__label">
                <span>Seeds per task</span>
              </span>
              <input
                type="number"
                min={1}
                max={20}
                value={seeds}
                onChange={(e) => setSeeds(Number(e.target.value))}
                disabled={isRunning}
                className="benchmark-input"
              />
            </label>
          </div>

          {!isRunning ? (
            <button
              className="benchmark-launch"
              onClick={runTransfer}
            >
              <span>{phase === "idle" ? "Launch transfer benchmark" : "Run another benchmark"}</span>
              <span aria-hidden="true">↗</span>
            </button>
          ) : (
            <div className="benchmark-active-state">
              <span className="benchmark-active-state__pulse" />
              <span>
                <strong>Transfer benchmark in progress</strong>
                <small>Evaluating policy across target tasks</small>
              </span>
            </div>
          )}
        </section>

        {/* Console & Live Metrics */}
        <section className="benchmark-console">
          <div className="benchmark-console__bar">
            <div><i /><i /><i /></div>
            <span>worker://transfer/output</span>
            {isRunning && <span className="benchmark-console__live"><i /> live</span>}
          </div>

          <div ref={logRef} className="benchmark-console__output scrollbar-thin">
            {logs.length === 0 ? (
              <div className="benchmark-console__idle">
                <span>&gt;_</span>
                <p>Configure model and launch transfer benchmark<span className="animate-pulse">_</span></p>
              </div>
            ) : (
              logs.map((line, index) => (
                <p key={index}>
                  <span>{String(index + 1).padStart(3, "0")}</span>
                  {line}
                </p>
              ))
            )}
          </div>

          {result && (
            <div className="benchmark-eval-result">
              <div>
                <span>Task completion</span>
                <strong>{result.task_completion_rate ? `${(result.task_completion_rate * 100).toFixed(1)}%` : "—"}</strong>
              </div>
              <div>
                <span>Pass@1</span>
                <strong>{result.pass_at_1 ? `${(result.pass_at_1 * 100).toFixed(1)}%` : "—"}</strong>
              </div>
              <div>
                <span>Pass@3</span>
                <strong>{result.pass_at_3 ? `${(result.pass_at_3 * 100).toFixed(1)}%` : "—"}</strong>
              </div>
              <div>
                <span>Evaluated tasks</span>
                <strong>{result.num_eval_tasks ?? "—"}</strong>
              </div>
            </div>
          )}
        </section>
      </div>

      {phase === "error" && error && (
        <div className="benchmark-notice benchmark-notice--error">
          <div>
            <p>Transfer benchmark failed</p>
            <span>{error}</span>
          </div>
        </div>
      )}

      {phase === "done" && (
        <div className="benchmark-notice benchmark-notice--success">
          <div>
            <p>Transfer benchmark complete</p>
            <span>Evaluation metrics recorded successfully across target distribution.</span>
          </div>
        </div>
      )}
    </div>
  );
}
