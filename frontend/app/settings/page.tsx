"use client";
import { useEffect, useState } from "react";
import { apiJson, type Settings } from "@/lib/taskFactory";

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

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
            <input
              className="benchmark-input benchmark-input--mono"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="gpt-5"
              required
            />
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
