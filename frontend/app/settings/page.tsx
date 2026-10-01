"use client";
import { useCallback, useEffect, useState } from "react";
import { apiJson, type Settings } from "@/lib/taskFactory";

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    let cancelled = false;
    apiJson<Settings>("/api/settings")
      .then((current) => {
        if (cancelled) return;
        setSettings(current);
        setProvider(current.task_validator.provider ?? current.providers.find((p) => p.name !== current.writer.provider)?.name ?? "");
        setModel(current.task_validator.model ?? "");
      })
      .catch((err: Error) => !cancelled && setError(err.message));
    return () => {
      cancelled = true;
    };
  }, []);

  // Pulls the Ollama model list again, for models pulled since the page opened.
  const refreshOllama = useCallback(async () => {
    setRefreshing(true);
    try {
      setSettings(await apiJson<Settings>("/api/settings"));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRefreshing(false);
    }
  }, []);

  async function handleSave(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    setSaved(false);
    try {
      setSettings(await apiJson<Settings>("/api/settings/task-validator", {
        method: "PUT",
        body: JSON.stringify({ provider, model }),
      }));
      setSaved(true);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  }

  const validator = settings?.task_validator;
  const ollamaModels = settings?.ollama.models ?? [];
  const writerFamily = settings?.writer.family;
  const providerKey = settings?.providers.find((p) => p.name === provider);

  return (
    <div className="benchmark-run settings-page">
      <header className="benchmark-run__hero">
        <div className="benchmark-run__hero-copy">
          <span className="benchmark-run__eyebrow">System / settings</span>
          <h1>SET THE<br /><em>JUDGE.</em></h1>
          <p>
            The task factory writes tasks with your generation model and validates them with a model from a
            different family. Saving here rewrites only the two validator lines in backend/.env. The next batch
            picks it up with no restart.
          </p>
        </div>
        <div className="benchmark-run__readout">
          <div><span>Writer</span><strong className="settings-model">{settings?.writer.model ?? "—"}</strong></div>
          <div><span>Writer family</span><strong>{settings?.writer.family ?? "—"}</strong></div>
          <div><span>Validator</span><strong className="settings-model">{validator?.model ?? "Not set"}</strong></div>
          <div className={`benchmark-run__state benchmark-run__state--${validator?.configured ? "done" : "error"}`}>
            <span>Task factory</span><strong><i />{validator?.configured ? "ready" : "blocked"}</strong>
          </div>
        </div>
      </header>

      <div className="benchmark-workbench">
        <form className="benchmark-config" onSubmit={handleSave}>
          <div className="benchmark-panel__heading">
            <div><span>01</span><h2>Task validator</h2></div>
            <p>FORGE_TASK_VALIDATOR_*</p>
          </div>
          <label className="benchmark-field">
            <span className="benchmark-field__label"><span>Provider</span><small>FORGE_TASK_VALIDATOR_PROVIDER</small></span>
            <select className="benchmark-input" value={provider} onChange={(e) => setProvider(e.target.value)} required>
              <option value="" disabled>Pick a provider</option>
              {settings?.providers.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
            </select>
            {providerKey?.needs_key && !providerKey.key_set && (
              <p className="tasks-hint tasks-hint--warn">
                No API key for {provider} in backend/.env. Add it there, then restart the worker.
              </p>
            )}
          </label>
          <label className="benchmark-field">
            <span className="benchmark-field__label"><span>Model</span><small>FORGE_TASK_VALIDATOR_MODEL</small></span>
            {provider === "ollama" && ollamaModels.length > 0 ? (
              <div className="settings-model-row">
                <select className="benchmark-input benchmark-input--mono" value={model} onChange={(e) => setModel(e.target.value)} required>
                  <option value="" disabled>Pick a pulled model</option>
                  {ollamaModels.map((m) => (
                    <option key={m.name} value={m.name} disabled={m.family === writerFamily}>
                      {m.name}{m.parameters ? ` · ${m.parameters}` : ""} · {m.family}{m.cloud ? " · Ollama cloud" : ""}
                      {m.family === writerFamily ? " (writer's family)" : ""}
                    </option>
                  ))}
                </select>
                <button type="button" className="tasks-button tasks-button--ghost" onClick={refreshOllama} disabled={refreshing}>
                  {refreshing ? "…" : "Refresh"}
                </button>
              </div>
            ) : (
              <input
                className="benchmark-input benchmark-input--mono"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder={provider === "ollama" ? "qwen3:32b" : "gpt-5"}
                required
              />
            )}
            {provider === "ollama" && (
              <p className="tasks-hint">
                {settings?.ollama.error
                  ? <>Could not reach Ollama: {settings.ollama.error}. Type a model name, or start Ollama and <button type="button" className="settings-link" onClick={refreshOllama}>refresh</button>.</>
                  : ollamaModels.length === 0
                    ? <>Ollama has no models pulled. Run <code>ollama pull &lt;model&gt;</code>, then <button type="button" className="settings-link" onClick={refreshOllama}>refresh</button>.</>
                    : `${ollamaModels.filter((m) => !m.cloud).length} local and ${ollamaModels.filter((m) => m.cloud).length} Ollama cloud models available.`}
              </p>
            )}
            <p className="tasks-hint">
              Must come from a different family than the writer ({settings?.writer.family ?? "…"}). A cheaper tier of
              the same vendor counts as the same family.
            </p>
          </label>
          {error && <p className="tasks-hint tasks-hint--warn settings-message">{error}</p>}
          {saved && !error && <p className="tasks-hint tasks-hint--ok settings-message">Saved to backend/.env.</p>}
          <button className="benchmark-launch" disabled={saving || !provider || !model.trim()}>
            <span>{saving ? "Saving…" : "Save validator"}</span><span aria-hidden="true">↗</span>
          </button>
        </form>

        <section className="benchmark-config">
          <div className="benchmark-panel__heading">
            <div><span>02</span><h2>Provider keys</h2></div>
            <p>Read from backend/.env, never shown</p>
          </div>
          <ul className="settings-keys">
            {settings?.providers.map((p) => (
              <li key={p.name}>
                <strong>{p.name}</strong>
                <span className={`tasks-status tasks-status--${!p.needs_key || p.key_set ? "complete" : "failed"}`}>
                  {!p.needs_key ? "no key needed" : p.key_set ? "key set" : "no key"}
                </span>
              </li>
            ))}
          </ul>
          <p className="tasks-hint settings-note">
            Keys stay in backend/.env. This page reports whether each one exists and never reads its value.
          </p>
        </section>
      </div>
    </div>
  );
}
