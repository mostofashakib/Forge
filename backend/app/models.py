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
