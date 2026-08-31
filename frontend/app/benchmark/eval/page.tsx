"use client";

import { useEffect, useRef, useState } from "react";
import { API_BASE, wsBase } from "@/lib/api";

type Engine = "forge" | "harbor";
type Phase = "idle" | "running" | "done" | "error";
type EvalResult = Record<string, string | number | boolean | null>;

export default function BenchmarkEvalPage() {
  const [engine, setEngine] = useState<Engine>("forge");
  const [checkpoint, setCheckpoint] = useState("policy_checkpoint");
  const [experiment, setExperiment] = useState("experiments/internal_heldout.yaml");
  const [runsDir, setRunsDir] = useState("runs");
  const [seed, setSeed] = useState("");
  const [harborTaskPath, setHarborTaskPath] = useState("example_tasks/slack_task_1");
  const [harborAgent, setHarborAgent] = useState("fleet.agents.rl_agent:SlackExternalAgent");
  const [harborModel, setHarborModel] = useState("gemma4:26b");
  const [harborAvailable, setHarborAvailable] = useState<boolean | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EvalResult | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/benchmark/evals/capabilities`)
      .then((response) => response.ok ? response.json() : Promise.reject())
      .then((data) => setHarborAvailable(Boolean(data.engines?.harbor?.available)))
      .catch(() => setHarborAvailable(null));
  }, []);

  function appendLog(line: string) {
    setLogs((current) => [...current.slice(-998), line]);
    requestAnimationFrame(() => {
      if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    });
  }

  async function loadResult(runId: string) {
    const response = await fetch(`${API_BASE}/api/benchmark/evals/${runId}`);
    if (!response.ok) return;
    const payload = await response.json();
    if (payload.result) setResult(payload.result);
  }

  async function runEvaluation() {
    setPhase("running");
    setLogs([]);
    setResult(null);
    setError(null);
    const payload = engine === "forge"
      ? { engine, checkpoint, experiment, runs_dir: runsDir, seed: seed === "" ? null : Number(seed) }
      : { engine, harbor_task_path: harborTaskPath, harbor_agent: harborAgent, harbor_model: harborModel };

    let runId: string;
    try {
      const response = await fetch(`${API_BASE}/api/benchmark/evals`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await response.text());
      ({ run_id: runId } = await response.json());
    } catch (requestError) {
      setPhase("error");
      setError(requestError instanceof Error ? requestError.message : "Could not start evaluation");
      return;
    }

    const socket = new WebSocket(`${wsBase()}/api/benchmark/ws/progress/${runId}`);
    socket.onmessage = async (event) => {
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
    };
    socket.onerror = () => {
      setError("The evaluation progress stream disconnected.");
      setPhase("error");
    };
  }

  const isRunning = phase === "running";

  return (
    <div className="benchmark-run">
      <header className="benchmark-run__hero">
        <div className="benchmark-run__hero-copy">
          <span className="benchmark-run__eyebrow">Evaluation protocol / 04</span>
          <h1>VERIFY THE<br /><em>TRANSFER.</em></h1>
          <p>Run held-out policy evaluation from the workbench, with Forge-native controls or an optional Harbor task harness.</p>
        </div>
        <div className="benchmark-run__readout" aria-label="Evaluation status">
          <div><span>Engine</span><strong className="text-sm uppercase">{engine}</strong></div>
          <div><span>Split</span><strong className="text-sm uppercase">held-out</strong></div>
          <div><span>Harbor</span><strong className="text-sm uppercase">{harborAvailable === null ? "unknown" : harborAvailable ? "ready" : "offline"}</strong></div>
          <div className={`benchmark-run__state benchmark-run__state--${phase}`}><span>System state</span><strong><i />{phase}</strong></div>
        </div>
      </header>

      <div className="benchmark-workbench">
        <section className="benchmark-config">
          <div className="benchmark-panel__heading">
            <div><span>01</span><h2>Evaluation setup</h2></div>
            <p>Choose the execution harness</p>
          </div>
          <div className="benchmark-field">
            <div className="benchmark-field__label"><span>Evaluation engine</span><small>Forge remains the default</small></div>
            <div className="benchmark-domain-grid">
              {(["forge", "harbor"] as Engine[]).map((option) => (
                <label key={option} className="benchmark-domain">
                  <input type="radio" name="engine" checked={engine === option} disabled={isRunning} onChange={() => setEngine(option)} />
                  <span className="benchmark-domain__check" aria-hidden="true">✓</span>
                  <span><strong>{option === "forge" ? "Forge native" : "Harbor"}</strong><small>{option === "forge" ? "Checkpoint + experiment" : "Local Harbor task"}</small></span>
                </label>
              ))}
            </div>
          </div>

          {engine === "forge" ? (
            <>
              <EvalField label="Policy checkpoint" value={checkpoint} disabled={isRunning} onChange={setCheckpoint} />
              <EvalField label="Experiment YAML" hint="train / held-out split" value={experiment} disabled={isRunning} onChange={setExperiment} />
              <div className="benchmark-field-row">
                <EvalField label="Runs directory" value={runsDir} disabled={isRunning} onChange={setRunsDir} />
                <label className="benchmark-field"><span className="benchmark-field__label"><span>Seed override</span><small>optional</small></span><input type="number" className="benchmark-input benchmark-input--mono" value={seed} disabled={isRunning} placeholder="checkpoint seed" onChange={(event) => setSeed(event.target.value)} /></label>
              </div>
            </>
          ) : (
            <>
              <EvalField label="Harbor task path" value={harborTaskPath} disabled={isRunning} onChange={setHarborTaskPath} />
              <EvalField label="Agent" hint="built-in or import path" value={harborAgent} disabled={isRunning} onChange={setHarborAgent} />
              <EvalField label="Model" value={harborModel} disabled={isRunning} onChange={setHarborModel} />
              {harborAvailable === false && <p className="benchmark-domain-empty m-5">Harbor is optional and currently unavailable. Run <code>./example_tasks/run.sh setup</code> when you want to enable it.</p>}
            </>
          )}

          {!isRunning ? (
            <button className="benchmark-launch" onClick={runEvaluation}><span>{phase === "idle" ? "Run held-out evaluation" : "Run another evaluation"}</span><span aria-hidden="true">↗</span></button>
          ) : (
            <div className="benchmark-active-state"><span className="benchmark-active-state__pulse" /><span><strong>Evaluation in progress</strong><small>{engine} worker stream is connected</small></span></div>
          )}
        </section>

        <section className="benchmark-console">
          <div className="benchmark-console__bar"><div><i /><i /><i /></div><span>worker://evaluation/output</span>{isRunning && <span className="benchmark-console__live"><i /> live</span>}</div>
          <div ref={logRef} className="benchmark-console__output scrollbar-thin">
            {logs.length === 0 ? <div className="benchmark-console__idle"><span>&gt;_</span><p>Configure an engine and start the held-out run<span className="animate-pulse">_</span></p></div> : logs.map((line, index) => <p key={index}><span>{String(index + 1).padStart(3, "0")}</span>{line}</p>)}
          </div>
          {result && <div className="benchmark-eval-result">{Object.entries(result).filter(([, value]) => typeof value !== "object").map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><strong>{typeof value === "number" ? value.toFixed(3) : String(value)}</strong></div>)}</div>}
        </section>
      </div>

      {phase === "error" && error && <div className="benchmark-notice benchmark-notice--error"><div><p>Evaluation failed</p><span>{error}</span></div></div>}
      {phase === "done" && <div className="benchmark-notice benchmark-notice--success"><div><p>Evaluation complete</p><span>Results were written by the selected evaluation engine.</span></div></div>}
    </div>
  );
}

function EvalField({ label, hint, value, disabled, onChange }: { label: string; hint?: string; value: string; disabled: boolean; onChange: (value: string) => void }) {
  return <label className="benchmark-field"><span className="benchmark-field__label"><span>{label}</span>{hint && <small>{hint}</small>}</span><input className="benchmark-input benchmark-input--mono" value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)} /></label>;
}
