from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, Integer, Boolean, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.database import Base


class CompileJob(Base):
    __tablename__ = "compile_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_name: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="pending")
    prompt: Mapped[str] = mapped_column(Text)
    compiler_input_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_path: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    env_name: Mapped[str] = mapped_column(String, index=True)
    task_name: Mapped[str] = mapped_column(String)
    seed: Mapped[int] = mapped_column(Integer)
    agent_id: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="running")
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    total_reward: Mapped[float] = mapped_column(Float, default=0.0)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    termination_reason: Mapped[str] = mapped_column(String, default="unknown")
    failure_type: Mapped[str | None] = mapped_column(String, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    environment_version: Mapped[str | None] = mapped_column(String, nullable=True)
    attempts_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    jsonl_path: Mapped[str | None] = mapped_column(String, nullable=True)


class EpisodeStep(Base):
    __tablename__ = "episode_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_id: Mapped[str] = mapped_column(String, ForeignKey("episodes.id"), index=True)
    step_index: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(Text)
    reward: Mapped[float] = mapped_column(Float)
    verifier_results: Mapped[str] = mapped_column(Text)
    diff: Mapped[str] = mapped_column(Text)
    events: Mapped[str] = mapped_column(Text)
    state_hash_before: Mapped[str] = mapped_column(String)
    state_hash_after: Mapped[str] = mapped_column(String)
    terminated: Mapped[bool] = mapped_column(Boolean)
    truncated: Mapped[bool] = mapped_column(Boolean)


class RolloutJob(Base):
    __tablename__ = "rollout_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    env_name: Mapped[str] = mapped_column(String, index=True)
    task_name: Mapped[str] = mapped_column(String)
    agent_id: Mapped[str] = mapped_column(String)
    num_episodes: Mapped[int] = mapped_column(Integer)
    seed_start: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String, default="pending")
    episodes_completed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class ExportJob(Base):
    __tablename__ = "export_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    env_name: Mapped[str] = mapped_column(String, index=True)
    formats: Mapped[str] = mapped_column(Text)
    output_path: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_id: Mapped[str] = mapped_column(String, index=True)
    step_index: Mapped[int] = mapped_column(Integer)
    actor: Mapped[str] = mapped_column(String, default="agent")
    action_type: Mapped[str] = mapped_column(String)
    rule_id: Mapped[str] = mapped_column(String)
    violation: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    env_name: Mapped[str] = mapped_column(String, index=True)
    agent_id: Mapped[str] = mapped_column(String)
    objective: Mapped[str] = mapped_column(Text)
    num_episodes: Mapped[int] = mapped_column(Integer)
    max_steps: Mapped[int] = mapped_column(Integer, default=50)
    # No runner stops on a score threshold any more. These two columns stay so
    # existing databases keep loading.
    divergence_threshold: Mapped[float] = mapped_column(Float, default=0.2)
    consecutive_below_threshold: Mapped[int] = mapped_column(Integer, default=3)
    dead_end_patience: Mapped[int] = mapped_column(Integer, default=5)
    success_threshold: Mapped[float] = mapped_column(Float, default=0.9)
    seed_start: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, default="pending")
    episodes_completed: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AgentEpisode(Base):
    __tablename__ = "agent_episodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("agent_runs.id"), index=True)
    episode_index: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String, default="running")
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    total_reward: Mapped[float] = mapped_column(Float, default=0.0)
    final_objective_score: Mapped[float] = mapped_column(Float, default=0.0)
    termination_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    failure_type: Mapped[str | None] = mapped_column(String, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    environment_version: Mapped[str | None] = mapped_column(String, nullable=True)
    attempts_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    jsonl_path: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SandboxEnvironment(Base):
    __tablename__ = "sandbox_environments"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, default="building")
    container_id: Mapped[str | None] = mapped_column(String, nullable=True)
    container_port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_tag: Mapped[str | None] = mapped_column(String, nullable=True)
    ttl_days: Mapped[int] = mapped_column(Integer, default=30)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    policy_requirements: Mapped[str | None] = mapped_column(Text, nullable=True)
    reward_requirements: Mapped[str | None] = mapped_column(Text, nullable=True)
    env_type: Mapped[str] = mapped_column(String, default="general")
    # Whether the generated app ships a browsable UI. Drives the sandbox App tab.
    has_ui: Mapped[bool] = mapped_column(Boolean, default=True)
    state_schema: Mapped[str | None] = mapped_column(Text, nullable=True)
    validation_missing_fields: Mapped[str | None] = mapped_column(Text, nullable=True)


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, default="queued")
    domains: Mapped[str] = mapped_column(String)
    depth: Mapped[int] = mapped_column(Integer)
    seeds: Mapped[int] = mapped_column(Integer)
    output_dir: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    kind: Mapped[str] = mapped_column(String, default="benchmark")
    engine: Mapped[str] = mapped_column(String, default="forge")
    config_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class TaskBatch(Base):
    """One versioned batch of synthetic tasks for an environment.

    Never changes after it is saved. A failed batch keeps its error and gets
    no version. Deleting is a soft delete, so a version number is never reused.
    """

    __tablename__ = "task_batches"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    env_name: Mapped[str] = mapped_column(String, index=True)
    version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String, default="queued")
    requested: Mapped[int] = mapped_column(Integer)
    delivered: Mapped[int] = mapped_column(Integer, default=0)
    pass_k: Mapped[int] = mapped_column(Integer)
    data_type: Mapped[str] = mapped_column(String, default="rl_tasks")
    writer_model: Mapped[str] = mapped_column(String)
    validator_model: Mapped[str] = mapped_column(String)
    taxonomy_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class GeneratedTask(Base):
    __tablename__ = "generated_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String, ForeignKey("task_batches.id"), index=True)
    task_id: Mapped[str] = mapped_column(String)
    task_json: Mapped[str] = mapped_column(Text)


class TaskRejectionRecord(Base):
    __tablename__ = "task_rejections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String, ForeignKey("task_batches.id"), index=True)
    rejection_json: Mapped[str] = mapped_column(Text)


class QuarantinedTask(Base):
    __tablename__ = "quarantined_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String, index=True)
    env_name: Mapped[str] = mapped_column(String, index=True)
    environment_version: Mapped[str] = mapped_column(String)
    reason: Mapped[str] = mapped_column(Text)
    released: Mapped[bool] = mapped_column(Boolean, default=False)
    quarantined_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    released_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class FlaggedEnvironmentVersion(Base):
    __tablename__ = "flagged_environment_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    env_name: Mapped[str] = mapped_column(String, index=True)
    version: Mapped[str] = mapped_column(String, index=True)
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, default="investigating")  # "investigating", "resolved"
    flagged_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, default="queued")  # queued | running | completed | failed
    objective: Mapped[str] = mapped_column(String, default="grpo")  # grpo | dpo
    base_model: Mapped[str] = mapped_column(String)
    data_dir: Mapped[str] = mapped_column(String)
    output_dir: Mapped[str] = mapped_column(String)
    checkpoint_path: Mapped[str | None] = mapped_column(String, nullable=True)
    inference_mode: Mapped[str] = mapped_column(String, default="auto")
    max_steps: Mapped[int] = mapped_column(Integer, default=500)
    num_examples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mean_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
