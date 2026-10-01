"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown } from "lucide-react";
import { apiJson, type SettingGroup, type SettingRow, type Settings } from "@/lib/taskFactory";

const GROUPS: { id: SettingGroup; title: string; note: string }[] = [
  { id: "models", title: "Models", note: "Generator, judge, quorum, task validator" },
  { id: "runtime", title: "Runtime", note: "Determinism and sandboxes" },
  { id: "containers", title: "Containers", note: "Resource limits and sandbox isolation" },
  { id: "budgets", title: "Generation budgets", note: "Token and context limits" },
  { id: "reliability", title: "Reliability & Retries", note: "Episode retries, snapshots, and replay" },
];

const REDUNDANT_CONTAINER_SETTINGS = new Set([
  "FORGE_PYTHON_BASE_IMAGE",
  "FORGE_CLI_IMAGE",
  "FORGE_BROWSER_IMAGE",
]);

// Each model field and the provider field that decides where it runs.
const MODEL_PROVIDER: Record<string, string> = {
  FORGE_LLM_MODEL: "FORGE_LLM_PROVIDER",
  FORGE_LLM_MODEL_CAPABLE: "FORGE_LLM_PROVIDER",
  FORGE_JUDGE_MODEL: "FORGE_JUDGE_PROVIDER",
  FORGE_TASK_VALIDATOR_MODEL: "FORGE_TASK_VALIDATOR_PROVIDER",
};

// These must come from another family than the generator.
const INDEPENDENT_MODELS = new Set(["FORGE_JUDGE_MODEL", "FORGE_TASK_VALIDATOR_MODEL"]);

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  // Initially, all sections are in the collapsed state
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(() => new Set());

  const toggleGroup = useCallback((groupId: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupId)) {
        next.delete(groupId);
      } else {
        next.add(groupId);
      }
      return next;
    });
  }, []);

  const expandAll = useCallback(() => {
    setExpandedGroups(new Set(GROUPS.map((g) => g.id)));
  }, []);

  const collapseAll = useCallback(() => {
    setExpandedGroups(new Set());
  }, []);

  const load = useCallback((next: Settings) => {
    setSettings(next);
    setDraft(Object.fromEntries(next.settings.map((row) => [row.key, row.value])));
  }, []);

  useEffect(() => {
    let cancelled = false;
    apiJson<Settings>("/api/settings")
      .then((current) => !cancelled && load(current))
      .catch((err: Error) => !cancelled && setError(err.message));
    return () => {
      cancelled = true;
    };
  }, [load]);

  // Pulls the Ollama model list again, for models pulled since the page opened.
  const refreshOllama = useCallback(async () => {
    setRefreshing(true);
    try {
      // Keeps the draft, so unsaved edits survive a refresh.
      setSettings(await apiJson<Settings>("/api/settings"));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRefreshing(false);
    }
  }, []);

  const rows = useMemo(() => settings?.settings ?? [], [settings]);
  const changed = rows.filter((row) => draft[row.key] !== undefined && draft[row.key] !== row.value);
  const pending = rows.filter((row) => row.pending_restart);

  async function handleSave(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    setSaved(false);
    try {
      load(await apiJson<Settings>("/api/settings", {
        method: "PUT",
        body: JSON.stringify({ values: Object.fromEntries(changed.map((row) => [row.key, draft[row.key]])) }),
      }));
      setSaved(true);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  }

  const validator = settings?.task_validator;
  const judge = rows.find((row) => row.key === "FORGE_JUDGE_MODEL")?.value;

  return (
    <div className="benchmark-run settings-page">
      <header className="benchmark-run__hero">
        <div className="benchmark-run__hero-copy">
          <span className="benchmark-run__eyebrow">System / settings</span>
          <h1>TUNE THE<br /><em>FORGE.</em></h1>
          <p>
            Models, runtime, containers and generation budgets for the whole platform. Saving rewrites only the
            changed lines in backend/.env. API keys stay in that file and never appear here.
          </p>
        </div>
        <div className="benchmark-run__readout">
          <div><span>Generator</span><strong className="settings-model">{settings?.writer.model ?? "—"}</strong></div>
          <div><span>Judge</span><strong className="settings-model">{judge || "Not set"}</strong></div>
          <div><span>Task validator</span><strong className="settings-model">{validator?.model ?? "Not set"}</strong></div>
          <div className={`benchmark-run__state benchmark-run__state--${pending.length ? "error" : "done"}`}>
            <span>Restart</span><strong><i />{pending.length ? `${pending.length} pending` : "up to date"}</strong>
          </div>
        </div>
      </header>

      {pending.length > 0 && (
        <p className="tasks-hint tasks-hint--warn settings-banner">
          {pending.length === 1 ? "1 saved change waits" : `${pending.length} saved changes wait`} for a restart of
          the API and Celery workers: {pending.map((row) => row.key).join(", ")}.
        </p>
      )}

      <form className="settings-form" onSubmit={handleSave}>
        <div className="flex items-center justify-between pb-1 text-xs">
          <span className="font-mono text-muted-foreground uppercase tracking-wider text-[11px]">
            Configuration Sections ({GROUPS.length})
          </span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={expandAll}
              className="text-xs text-muted-foreground hover:text-foreground underline underline-offset-4 decoration-dotted transition-colors"
            >
              Expand all
            </button>
            <span className="text-muted-foreground/30">•</span>
            <button
              type="button"
              onClick={collapseAll}
              className="text-xs text-muted-foreground hover:text-foreground underline underline-offset-4 decoration-dotted transition-colors"
            >
              Collapse all
            </button>
          </div>
        </div>

        {GROUPS.map((group, index) => {
          const isExpanded = expandedGroups.has(group.id);
          const groupRows = rows.filter(
            (row) => row.group === group.id && !REDUNDANT_CONTAINER_SETTINGS.has(row.key)
          );
          const groupChanged = groupRows.filter((row) => draft[row.key] !== undefined && draft[row.key] !== row.value);
          const groupPending = groupRows.filter((row) => row.pending_restart);

          return (
            <section key={group.id} className="benchmark-config overflow-hidden transition-all duration-200">
              <button
                type="button"
                onClick={() => toggleGroup(group.id)}
                aria-expanded={isExpanded}
                aria-controls={`group-content-${group.id}`}
                className={`benchmark-panel__heading w-full cursor-pointer text-left transition-colors hover:bg-muted/15 flex items-center justify-between px-5 py-4 ${
                  isExpanded ? "border-b border-foreground/25" : "border-b-0"
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="font-mono text-[0.58rem] text-primary">0{index + 1}</span>
                  <h2 className="text-xs font-semibold uppercase tracking-[0.16em]">{group.title}</h2>
                </div>
                <div className="flex items-center gap-3">
                  <p className="hidden font-mono text-[0.5rem] uppercase tracking-wider text-muted-foreground md:block">
                    {group.note}
                  </p>
                  {groupChanged.length > 0 && (
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-primary/10 text-primary border border-primary/20 font-semibold">
                      {groupChanged.length} changed
                    </span>
                  )}
                  {groupPending.length > 0 && (
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-500 border border-amber-500/20 font-semibold">
                      {groupPending.length} restart pending
                    </span>
                  )}
                  <ChevronDown
                    size={16}
                    className={`text-muted-foreground transition-transform duration-200 shrink-0 ${
                      isExpanded ? "rotate-180" : ""
                    }`}
                    aria-hidden="true"
                  />
                </div>
              </button>

              {isExpanded && (
                <div id={`group-content-${group.id}`} className="transition-all duration-200">
                  <div className="settings-grid">
                    {groupRows.map((row) => (
                      <SettingField
                        key={row.key}
                        row={row}
                        value={draft[row.key] ?? row.value}
                        draft={draft}
                        settings={settings!}
                        refreshing={refreshing}
                        onRefresh={refreshOllama}
                        onChange={(value) => setDraft((current) => ({ ...current, [row.key]: value }))}
                      />
                    ))}
                  </div>
                  {group.id === "models" && validator && !validator.configured && validator.error && (
                    <p className="tasks-hint tasks-hint--warn settings-message">Task validator: {validator.error}</p>
                  )}
                </div>
              )}
            </section>
          );
        })}

        <div className="settings-savebar">
          <span>
            {error
              ? <span className="settings-savebar__error">{error}</span>
              : saved && changed.length === 0
                ? "Saved to backend/.env."
                : changed.length === 0
                  ? "No unsaved changes."
                  : `${changed.length} unsaved ${changed.length === 1 ? "change" : "changes"}: ${changed.map((row) => row.key).join(", ")}`}
          </span>
          <div>
            <button
              type="button"
              className="tasks-button tasks-button--ghost"
              disabled={changed.length === 0 || saving}
              onClick={() => setDraft(Object.fromEntries(rows.map((row) => [row.key, row.value])))}
            >
              Discard
            </button>
            <button className="tasks-button" disabled={changed.length === 0 || saving}>
              {saving ? "Saving…" : "Save changes"}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}

function SettingField({
  row, value, draft, settings, refreshing, onRefresh, onChange,
}: {
  row: SettingRow;
  value: string;
  draft: Record<string, string>;
  settings: Settings;
  refreshing: boolean;
  onRefresh: () => void;
  onChange: (value: string) => void;
}) {
  const providerKey = MODEL_PROVIDER[row.key];
  // An unset judge provider means the judge runs on the generator's provider.
  const provider = providerKey ? draft[providerKey] || draft.FORGE_LLM_PROVIDER : undefined;
  const ollamaModels = settings.ollama.models;
  const useOllamaList = provider === "ollama" && ollamaModels.length > 0;
  const blockedFamily = INDEPENDENT_MODELS.has(row.key) ? settings.writer.family : null;
  const missingKey = row.kind === "choice" && settings.providers.find((p) => p.name === value && p.needs_key && !p.key_set);

  return (
    <label className={`benchmark-field settings-field ${value !== row.value ? "settings-field--changed" : ""}`}>
      <span className="benchmark-field__label">
        <span>{row.label}</span>
        <span className="settings-tags">
          {row.pending_restart
            ? <em className="settings-tag settings-tag--pending">restart pending</em>
            : <em className={`settings-tag ${row.live ? "settings-tag--live" : ""}`}>{row.live ? "next job" : "on restart"}</em>}
        </span>
      </span>
      <small className="settings-key">{row.key}</small>

      {row.kind === "choice" ? (
        <select className="benchmark-input" value={value} onChange={(e) => onChange(e.target.value)}>
          {row.optional && <option value="">Not set</option>}
          {row.choices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}
        </select>
      ) : useOllamaList ? (
        <div className="settings-model-row">
          <select className="benchmark-input benchmark-input--mono" value={value} onChange={(e) => onChange(e.target.value)}>
            <option value="" disabled={!row.optional}>{row.optional ? "Not set" : "Pick a pulled model"}</option>
            {value && !ollamaModels.some((m) => m.name === value) && <option value={value}>{value} (not pulled)</option>}
            {ollamaModels.map((m) => (
              <option key={m.name} value={m.name} disabled={m.family === blockedFamily}>
                {m.name}{m.parameters ? ` · ${m.parameters}` : ""} · {m.family}{m.cloud ? " · Ollama cloud" : ""}
                {m.family === blockedFamily ? " (generator's family)" : ""}
              </option>
            ))}
          </select>
          <button type="button" className="tasks-button tasks-button--ghost" onClick={onRefresh} disabled={refreshing}>
            {refreshing ? "…" : "Refresh"}
          </button>
        </div>
      ) : (
        <input
          className={`benchmark-input ${row.kind === "text" || row.kind === "quorum" || row.kind === "url" ? "benchmark-input--mono" : ""}`}
          type={row.kind === "integer" || row.kind === "number" ? "number" : "text"}
          step={row.kind === "number" ? "any" : undefined}
          min={row.minimum ?? undefined}
          value={value}
          placeholder={row.default || (row.optional ? "Not set" : "")}
          onChange={(e) => onChange(e.target.value)}
        />
      )}

      {row.help && <p className="tasks-hint">{row.help}</p>}
      {provider === "ollama" && row.key in MODEL_PROVIDER && settings.ollama.error && (
        <p className="tasks-hint">Could not reach Ollama, so type a model name. {settings.ollama.error}</p>
      )}
      {missingKey && (
        <p className="tasks-hint tasks-hint--warn">No API key for {value} in backend/.env. Add it there, then restart.</p>
      )}
      {value !== row.default && row.default && (
        <button type="button" className="settings-link settings-reset" onClick={() => onChange(row.default)}>
          Reset to default
        </button>
      )}
    </label>
  );
}
