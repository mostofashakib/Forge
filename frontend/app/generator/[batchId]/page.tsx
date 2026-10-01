"use client";
import { use, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { API_BASE } from "@/lib/api";
import {
  apiJson,
  formatDate,
  STAGE_LABELS,
  type BatchDetail,
  type Check,
  type SyntheticTask,
} from "@/lib/taskFactory";

function describeCheck(check: Check): string {
  const json = (v: unknown) => JSON.stringify(v);
  switch (check.kind) {
    case "record_exists":
      return `${check.collection} has a row ${json(check.match)}${check.expect ? ` with ${json(check.expect)}` : ""}`;
    case "record_absent":
      return `${check.collection} has no row ${json(check.match)}`;
    case "record_count":
      return `rows in ${check.collection} matching ${json(check.match ?? {})} ${check.op} ${check.value}`;
    case "value":
      return `${check.path} ${check.op} ${json(check.value)}`;
    case "actions_called":
      return `called ${check.tools?.join(" → ")}${check.ordered ? " in order" : ""}`;
    case "actions_not_called":
      return `never called ${check.tools?.join(", ")}`;
    case "shell":
      return `$ ${check.command}`;
    case "dom":
      return `${check.selector} ${check.op === "exists" || check.op === "absent" ? check.op : `${check.prop === "attr" ? `[${check.attr}]` : check.prop ?? "text"} ${check.op} ${json(check.value)}`}`;
    case "url":
      return `page path ${check.op} ${json(check.value)}`;
    default:
      return check.kind;
  }
}

function seedSummary(task: SyntheticTask): string {
  const { records, setup, pages } = task.seed;
  const rows = Object.values(records).reduce((n, list) => n + list.length, 0);
  if (rows) return `${rows} seeded rows in ${Object.keys(records).join(", ")}`;
  if (setup.length) return `${setup.length} setup commands`;
  if (pages.length) return `${pages.length} pages, opens ${task.seed.start_path}`;
  return "No seed";
}

function TaskCard({ task }: { task: SyntheticTask }) {
  const [open, setOpen] = useState(false);
  const reflections = new Map(task.reflection_points.map((r) => [r.step, r]));
  return (
    <article className={`tasks-card ${open ? "tasks-card--open" : ""}`}>
      <button className="tasks-card__head" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className="tasks-card__id">{task.id}</span>
        <span className={`tasks-difficulty tasks-difficulty--${task.difficulty}`}>D{task.difficulty}</span>
        <span className="tasks-card__title">
          <strong>{task.title}</strong>
          <small>{task.category} · {task.golden.length} golden steps · budget {task.step_budget}</small>
        </span>
        <span aria-hidden="true">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="tasks-card__body">
          <p className="tasks-card__objective">{task.objective}</p>
          <div className="tasks-card__grid">
            <section>
              <h4>Golden solution</h4>
              <ol className="tasks-golden">
                {task.golden.map((step, i) => (
                  <li key={i}>
                    <code>{step.tool}</code>
                    {Object.keys(step.args).length > 0 && <span>{JSON.stringify(step.args)}</span>}
                    {reflections.has(i) && (
                      <em>⟲ {reflections.get(i)!.kind}: {reflections.get(i)!.note}</em>
                    )}
                  </li>
                ))}
              </ol>
            </section>
            <section>
              <h4>Checks</h4>
              <ul className="tasks-checks">
                {task.checks.map((check, i) => (
                  <li key={i}><code>{check.kind}</code> {check.description || describeCheck(check)}</li>
                ))}
              </ul>
              <h4>Seed</h4>
              <p className="tasks-muted">{seedSummary(task)}</p>
              <h4>Validator</h4>
              <ul className="tasks-checks">
                <li><code>realistic</code> {task.review.realistic_reason}</li>
                <li><code>fair</code> {task.review.fair_reason}</li>
                <li><code>sensible</code> {task.review.sensible_reason}</li>
              </ul>
              <p className="tasks-muted tasks-card__fingerprint">
                Round {task.round} · fingerprint {task.fingerprint.slice(0, 12)}
              </p>
            </section>
          </div>
        </div>
      )}
    </article>
  );
}

export default function BatchPage({ params }: { params: Promise<{ batchId: string }> }) {
  const { batchId } = use(params);
  const router = useRouter();
  const [batch, setBatch] = useState<BatchDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    let cancelled = false;
    apiJson<BatchDetail>(`/api/task-factory/batches/${batchId}`)
      .then((detail) => !cancelled && setBatch(detail))
      .catch((err: Error) => !cancelled && setError(err.message));
    return () => {
      cancelled = true;
    };
  }, [batchId]);

  async function handleDelete() {
    if (!batch || !window.confirm(`Delete version ${batch.version} of ${batch.env_name}? Its number is never reused.`)) return;
    setDeleting(true);
    try {
      await apiJson(`/api/task-factory/batches/${batchId}`, { method: "DELETE" });
      router.push("/generator");
    } catch (err) {
      setError((err as Error).message);
      setDeleting(false);
    }
  }

  if (error) {
    return (
      <div className="benchmark-notice benchmark-notice--error">
        <div><p>Could not load the batch</p><span>{error}</span></div>
        <Link href="/generator">Back to generator →</Link>
      </div>
    );
  }
  if (!batch) return <p className="tasks-empty">Loading batch…</p>;

  const finished = ["complete", "short", "failed"].includes(batch.status);
  const byDifficulty = [1, 2, 3, 4, 5].map((d) => batch.tasks.filter((t) => t.difficulty === d).length);

  return (
    <div className="benchmark-run tasks-page">
      <header className="benchmark-run__hero">
        <div className="benchmark-run__hero-copy">
          <span className="benchmark-run__eyebrow">
            <Link href="/generator">Generator</Link> / {batch.env_name}
          </span>
          <h1>{batch.version ? <>VERSION<br /><em>{batch.version}.</em></> : <>BATCH<br /><em>{batch.status}.</em></>}</h1>
          <p>
            Created {formatDate(batch.created_at)}. Writer {batch.writer_model}, validator {batch.validator_model}.
            Every task passed its golden solution {batch.pass_k} times in a row.
          </p>
          <div className="tasks-actions">
            {batch.version && (
              <a className="tasks-button" href={`${API_BASE}/api/task-factory/batches/${batch.id}/export`}>
                Export JSON ↓
              </a>
            )}
            {finished && (
              <button className="tasks-button tasks-button--ghost" onClick={handleDelete} disabled={deleting}>
                {deleting ? "Deleting…" : "Delete version"}
              </button>
            )}
          </div>
        </div>
        <div className="benchmark-run__readout">
          <div><span>Delivered</span><strong>{batch.delivered}/{batch.requested}</strong></div>
          <div><span>Data Type</span><strong>{batch.data_type === "preference_pairs" ? "Pair" : batch.data_type === "sft" ? "SFT" : "RL"}</strong></div>
          <div><span>Rejected drafts</span><strong>{String(batch.rejections.length).padStart(2, "0")}</strong></div>
          <div><span>pass^k</span><strong>{String(batch.pass_k).padStart(2, "0")}</strong></div>
          <div className={`benchmark-run__state benchmark-run__state--${batch.status === "failed" ? "error" : finished ? "done" : "running"}`}>
            <span>Status</span><strong><i />{batch.status}</strong>
          </div>
        </div>
      </header>

      {batch.status === "short" && (
        <div className="benchmark-notice benchmark-notice--error">
          <div>
            <p>Short by {batch.shortfall}</p>
            <span>Some slots failed validation in all three rounds. Their reasons are listed under rejections.</span>
          </div>
        </div>
      )}
      {batch.status === "failed" && (
        <div className="benchmark-notice benchmark-notice--error"><div><p>Batch failed</p><span>{batch.error}</span></div></div>
      )}

      {batch.taxonomy && (
        <section className="benchmark-config">
          <div className="benchmark-panel__heading">
            <div><span>01</span><h2>Taxonomy</h2></div>
            <p>{batch.taxonomy.categories.length} categories</p>
          </div>
          <div className="tasks-taxonomy">
            {batch.taxonomy.categories.map((c) => (
              <div key={c.name} className="tasks-taxonomy__item">
                <strong>{c.name}</strong>
                <p>{c.description}</p>
                <div>
                  {[1, 2, 3, 4, 5].map((d) => (
                    <span key={d} className={c.difficulties.includes(d) ? `tasks-difficulty tasks-difficulty--${d}` : "tasks-difficulty tasks-difficulty--off"}>
                      D{d}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
          {Object.keys(batch.taxonomy.difficulty_rubric).length > 0 && (
            <dl className="tasks-rubric">
              {Object.entries(batch.taxonomy.difficulty_rubric).map(([level, text]) => (
                <div key={level}><dt>D{level}</dt><dd>{text}</dd></div>
              ))}
            </dl>
          )}
        </section>
      )}

      <section className="benchmark-config">
        <div className="benchmark-panel__heading">
          <div><span>02</span><h2>Tasks</h2></div>
          <p>{byDifficulty.map((n, i) => `D${i + 1}·${n}`).join("  ")}</p>
        </div>
        {batch.tasks.length === 0 ? (
          <p className="tasks-empty">No accepted tasks in this batch.</p>
        ) : (
          <div className="tasks-list">{batch.tasks.map((t) => <TaskCard key={t.id} task={t} />)}</div>
        )}
      </section>

      {batch.rejections.length > 0 && (
        <section className="benchmark-config">
          <div className="benchmark-panel__heading">
            <div><span>03</span><h2>Rejections</h2></div>
            <p>Every draft that did not make it, with the reason</p>
          </div>
          <div className="tasks-table-wrap">
            <table className="tasks-table">
              <thead><tr><th>Round</th><th>Slot</th><th>Stage</th><th>Draft</th><th>Reason</th></tr></thead>
              <tbody>
                {batch.rejections.map((r, i) => (
                  <tr key={i}>
                    <td>{r.round}</td>
                    <td className="tasks-table__mono">{r.category} · D{r.difficulty}</td>
                    <td><span className="tasks-status tasks-status--failed">{STAGE_LABELS[r.stage]}</span></td>
                    <td>{r.draft?.title ?? "—"}</td>
                    <td className="tasks-table__reason">{r.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
