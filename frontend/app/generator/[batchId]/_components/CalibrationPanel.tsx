"use client";
import { useCallback, useEffect, useState } from "react";
import {
  BAND_LABELS,
  MAX_ATTEMPTS,
  exportNote,
  histogramRows,
  isActive,
  needsJudgment,
  targetModelProblem,
  type AttemptView,
  type Band,
  type CalibrationDetail,
  type CalibrationRun,
  type Judgment,
  type TaskResult,
} from "@/lib/calibration";
import { apiJson, formatDate } from "@/lib/taskFactory";

const POLL_MS = 5000;
const BANDS: Band[] = ["too_easy", "frontier", "unsolved", "suspect"];

function Attempts({ base, task }: { base: string; task: TaskResult }) {
  const [attempts, setAttempts] = useState<AttemptView[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiJson<AttemptView[]>(`${base}/tasks/${task.task_id}`).then(setAttempts).catch((err: Error) => setError(err.message));
  }, [base, task.task_id]);

  if (error) return <p className="tasks-hint tasks-hint--warn">{error}</p>;
  if (!attempts) return <p className="tasks-empty">Loading attempts…</p>;
  return (
    <ol className="calib-attempts">
      {attempts.map((a) => (
        <li key={a.attempt_index}>
          <span className={`tasks-status tasks-status--${a.passed ? "complete" : "failed"}`}>
            {a.passed ? "pass" : "fail"}
          </span>{" "}
          <code>{a.termination}</code> · {a.steps.length} steps
          {a.labels.step_errors > 0 && ` · ${a.labels.step_errors} step errors`}
          {a.labels.failed_checks.length > 0 && ` · failed: ${a.labels.failed_checks.join("; ")}`}
          {a.labels.off_golden_tools.length > 0 && ` · off the golden path: ${a.labels.off_golden_tools.join(", ")}`}
          {a.error && <span className="tasks-table__reason"> · {a.error}</span>}
          <ol className="tasks-golden">
            {a.steps.map((s, i) => (
              <li key={i}>
                <code>{s.step.tool}</code>
                {Object.keys(s.step.args).length > 0 && <span>{JSON.stringify(s.step.args)}</span>}
                {!s.ok && <em>{s.error}</em>}
              </li>
            ))}
          </ol>
        </li>
      ))}
    </ol>
  );
}

function JudgmentButtons({ batchId, task, onJudged }: {
  batchId: string;
  task: TaskResult;
  onJudged: (taskId: string, judgment: Judgment) => void;
}) {
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function judge(verdict: Judgment) {
    setSaving(true);
    setError(null);
    try {
      await apiJson(`/api/task-factory/batches/${batchId}/tasks/${task.task_id}/judgment`, {
        method: "PUT",
        body: JSON.stringify({ verdict }),
      });
      onJudged(task.task_id, verdict);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="calib-judgment">
      {(["hard", "unfair"] as const).map((verdict) => (
        <button
          key={verdict} disabled={saving} onClick={() => judge(verdict)}
          className={`tasks-button ${task.judgment === verdict ? "" : "tasks-button--ghost"}`}
          aria-pressed={task.judgment === verdict}
        >
          {verdict}
        </button>
      ))}
      <small>{exportNote(task)}</small>
      {error && <p className="tasks-hint tasks-hint--warn">{error}</p>}
    </div>
  );
}

function RunDetail({ base, batchId, detail, onJudged }: {
  base: string;
  batchId: string;
  detail: CalibrationDetail;
  onJudged: (taskId: string, judgment: Judgment) => void;
}) {
  const [openTask, setOpenTask] = useState<string | null>(null);
  const rows = histogramRows(detail.summary.histogram);
  return (
    <>
      <div className="benchmark-run__readout calib-readout">
        {BANDS.map((band) => (
          <div key={band} title={BAND_LABELS[band].desc}>
            <span>{BAND_LABELS[band].label}</span>
            <strong>{detail.summary.bands[band]}</strong>
          </div>
        ))}
        <div><span>Tasks done</span><strong>{detail.summary.tasks}</strong></div>
      </div>

      {rows.length > 0 && (
        <div className="calib-histogram" aria-label="Solve-rate histogram">
          {rows.map((row) => (
            <div key={row.label}>
              <span>{row.label}</span>
              <div><i style={{ width: `${row.share * 100}%` }} /></div>
              <strong>{row.count}</strong>
            </div>
          ))}
        </div>
      )}

      {Object.keys(detail.summary.by_difficulty).length > 0 && (
        <div className="tasks-table-wrap">
          <table className="tasks-table">
            <thead><tr><th>Difficulty</th>{BANDS.map((b) => <th key={b}>{BAND_LABELS[b].label}</th>)}</tr></thead>
            <tbody>
              {Object.entries(detail.summary.by_difficulty).map(([level, counts]) => (
                <tr key={level}>
                  <td className="tasks-table__mono">D{level}</td>
                  {BANDS.map((b) => <td key={b}>{counts[b]}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="tasks-table-wrap">
        <table className="tasks-table">
          <thead><tr><th>Task</th><th>Solved</th><th>Band</th><th>Why it failed</th><th>Your call</th></tr></thead>
          <tbody>
            {detail.tasks.map((task) => (
              <tr key={task.task_id}>
                <td>
                  <button className="calib-task" onClick={() => setOpenTask(openTask === task.task_id ? null : task.task_id)}>
                    <span className="tasks-table__mono">{task.task_id} · D{task.difficulty}</span> {task.title}
                  </button>
                  {openTask === task.task_id && <Attempts base={base} task={task} />}
                </td>
                <td className="tasks-table__mono">{task.passed}/{task.attempts}</td>
                <td>{BAND_LABELS[task.band].label}</td>
                <td className="tasks-table__reason">{task.failure_summary?.text ?? "—"}</td>
                <td>{needsJudgment(task) ? <JudgmentButtons batchId={batchId} task={task} onJudged={onJudged} /> : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export default function CalibrationPanel({ batchId }: { batchId: string }) {
  const base = `/api/task-factory/batches/${batchId}/calibrations`;
  const [runs, setRuns] = useState<CalibrationRun[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<CalibrationDetail | null>(null);
  const [target, setTarget] = useState("");
  const [attempts, setAttempts] = useState(MAX_ATTEMPTS);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const loadRuns = useCallback(
    () =>
      apiJson<CalibrationRun[]>(base).then((list) => {
        setRuns(list);
        setSelected((current) => current ?? list[0]?.id ?? null);
      }),
    [base],
  );

  useEffect(() => {
    apiJson<CalibrationRun[]>(base)
      .then((list) => {
        setRuns(list);
        setSelected((current) => current ?? list[0]?.id ?? null);
      })
      .catch((err: Error) => setError(err.message));
  }, [base]);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const load = () =>
      apiJson<CalibrationDetail>(`${base}/${selected}`)
        .then((d) => {
          if (cancelled) return;
          setDetail(d);
          if (isActive(d.status)) timer = setTimeout(load, POLL_MS);
          else loadRuns().catch(() => undefined);
        })
        .catch((err: Error) => !cancelled && setError(err.message));
    load();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [base, selected, loadRuns]);

  const targetError = target ? targetModelProblem(target) : null;

  const onJudged = useCallback((taskId: string, judgment: Judgment) => {
    setDetail((d) => d && {
      ...d,
      tasks: d.tasks.map((t) => (t.task_id === taskId ? { ...t, judgment } : t)),
    });
  }, []);

  async function start() {
    setStarting(true);
    setError(null);
    try {
      const { run_id } = await apiJson<{ run_id: string }>(base, {
        method: "POST",
        body: JSON.stringify({ target_model: target.trim(), attempts }),
      });
      setSelected(run_id);
      await loadRuns();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setStarting(false);
    }
  }

  return (
    <section className="benchmark-config">
      <div className="benchmark-panel__heading">
        <div><span>04</span><h2>Calibration</h2></div>
        <p>How often a target model solves each task</p>
      </div>

      <div className="calib-form">
        <label className="benchmark-field">
          <span className="benchmark-field__label"><span>Target model</span><small>provider:model</small></span>
          <input
            className="benchmark-input benchmark-input--mono" placeholder="ollama:qwen3.8:27b"
            value={target} onChange={(e) => setTarget(e.target.value)}
          />
          {targetError && <p className="tasks-hint tasks-hint--warn">{targetError}</p>}
        </label>
        <div className="benchmark-field">
          <div className="benchmark-field__label"><span>Attempts per task</span><strong>{attempts}</strong></div>
          <input
            type="range" min={1} max={MAX_ATTEMPTS} value={attempts}
            onChange={(e) => setAttempts(Number(e.target.value))} className="benchmark-range"
          />
        </div>
        <button className="tasks-button" onClick={start} disabled={starting || !target || Boolean(targetError)}>
          {starting ? "Starting…" : "Start calibration"}
        </button>
      </div>
      {error && <p className="tasks-hint tasks-hint--warn">{error}</p>}

      {runs.length > 0 && (
        <div className="calib-runs">
          {runs.map((run) => (
            <button
              key={run.id} onClick={() => setSelected(run.id)}
              className={`calib-run ${run.id === selected ? "calib-run--active" : ""}`}
            >
              <span className={`tasks-status tasks-status--${run.status}`}>{run.status}</span>
              <strong>{run.target_model}</strong>
              <small>{run.attempts_per_task} attempts · {formatDate(run.created_at)}</small>
            </button>
          ))}
        </div>
      )}

      {detail && detail.id === selected && (
        <>
          {detail.status === "failed" && (
            <div className="benchmark-notice benchmark-notice--error">
              <div><p>Calibration failed</p><span>{detail.error}</span></div>
            </div>
          )}
          <RunDetail base={`${base}/${detail.id}`} batchId={batchId} detail={detail} onJudged={onJudged} />
        </>
      )}
    </section>
  );
}
