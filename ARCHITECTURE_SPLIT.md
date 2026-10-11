# ARCHITECTURE_SPLIT: Forge & Autotune

This document defines the architectural split of the system into two independent, focused projects. It serves as the single source of truth across IDE sessions and repository workspaces.

---

## Executive Summary

| Project | Location | Focus | Core Value Proposition |
|---|---|---|---|
| **Forge** | `/Users/adibshakib/Coding/Forge` | Synthetic Data & Agent Verification Engine | Generates grounded, execution-verified agent trajectories and multi-turn datasets at scale. |
| **Autotune** | `/Users/adibshakib/Coding/Autotune` | Autonomous AI Training & Benchmarking Engine | Zero-friction, plug-and-play fine-tuning (LoRA + 4-bit QLoRA) and rigorous model evaluation. |

---

## 1. Project 1: Forge (Synthetic Data Engine)

### 1.1 Scope & Responsibilities
- **Environment Generation:** Docker sandboxes, mock web apps, Linux CLI shells, and browser sessions (`forge/envgen/`, `forge/compiler/`).
- **Task Synthesis:** Automated task generation and golden solution creation (`forge/taskfactory/`).
- **Dynamic Persona Generator:** (`forge/personas/generator.py`) Synthesizes diverse, context-specific simulated humans (personality traits, delay tolerance, communication styles) for multi-turn collaboration datasets.
- **Agent Execution:** Headless agent runs inside sandboxes collecting verifiable trajectories.
- **Grading & Verifiers:** Multi-tier verification, reward-hacking prevention, and rubric scoring (`forge/validation/`).
- **Dataset Export:** Serializes verified runs into standard chat, preference, and RL dataset files (`backend/app/services/export_writers/`).

### 1.2 Refactoring Tasks for Forge
1. Remove `forge/training` and `forge/benchmark` from Forge (moved to Autotune).
2. Add dynamic LLM-backed `PersonaGenerator` in `forge/personas/generator.py`.
3. Standardize export outputs with a unified `dataset_manifest.json`.

---

## 2. Project 2: Autotune (AI Training & Testing Engine)

### 2.1 Scope & Responsibilities
- **Zero-Config Data Ingestion:** Accepts CSV, JSONL, Parquet, or Hugging Face datasets. Automatically detects schemas (chat messages, completion pairs, reasoning traces).
- **AutoSanitizer:** Cleans corrupt records, fixes malformed prompt formatting, and builds balanced train/eval splits.
- **AutoRecipeSelector:** Analyzes available GPU VRAM and dataset size to automatically select:
  - Base model (e.g. Qwen 2.5 7B, Llama 3.1 8B).
  - Training method: **LoRA (16-bit)** for speed when VRAM allows, or **QLoRA (4-bit NF4)** via `bitsandbytes` to train 8B-14B models on 16GB-24GB GPUs without OOM errors.
  - Learning rate, batch size, gradient accumulation steps, and rank/alpha.
- **Training Orchestration:** Pluggable backends (TRL, PEFT, PyTorch) implementing clean `Protocol` interfaces.
- **Evaluation & Benchmarking:**
  - Evaluates trained checkpoints side-by-side against base models on held-out tasks.
  - Generates plain-English performance comparisons (pass@1, pass@3, task completion rate).
- **One-Click Export:** Exports directly to Ollama, vLLM, GGUF, or Hugging Face Hub.

### 2.2 Clean Architecture Layout for Autotune
```
Autotune/
├── pyproject.toml               # uv-managed project file
├── README.md
├── src/
│   └── autotune/
│       ├── core/
│       │   ├── domain/          # Pure entities: Job, Dataset, Recipe, EvalResult
│       │   ├── ports/           # Dependency Injection Protocols
│       │   │   ├── trainer.py   # TrainingBackend protocol
│       │   │   ├── sanitizer.py # DataSanitizer protocol
│       │   │   ├── evaluator.py # BenchmarkEvaluator protocol
│       │   │   └── storage.py   # CheckpointStorage protocol
│       │   └── services/        # Pure logic: AutoRecipeSelector, MetricsAggregator
│       ├── infrastructure/      # Adapter implementations
│       │   ├── backends/        # TrlPeftBackend (LoRA & 4-bit QLoRA)
│       │   ├── sanitizers/      # FormatNormalizer & SchemaDetector
│       │   ├── evaluators/      # TransferBenchmark & SideBySideEvaluator
│       │   └── storage/         # LocalDiskStorage & HuggingFaceStorage
│       └── presentation/
│           ├── cli/             # autotune train, autotune eval
│           └── api/             # FastAPI backend for web dashboard
└── tests/
    ├── unit/                    # Pure unit tests (no GPU required)
    └── integration/             # End-to-end pipeline tests with mocked trainers
```

---

## 3. The Interchange Data Contract

Forge and Autotune communicate exclusively via standard dataset formats. Neither imports the other's code.

### 3.1 Standard Record Format (`messages` JSONL)
```json
{
  "messages": [
    {"role": "user", "content": "Task description and context"},
    {"role": "assistant", "content": "Model thought and action/response"}
  ],
  "metadata": {
    "task_id": "task_104",
    "environment": "crm_app",
    "verified": true,
    "reward": 1.0
  }
}
```

### 3.2 Dataset Manifest (`dataset_manifest.json`)
```json
{
  "version": "1.0.0",
  "task_type": "sft",
  "num_records": 5000,
  "features": ["messages", "metadata"],
  "source": "forge-synthetic-engine",
  "created_at": "2026-10-11T15:30:00Z"
}
```

---

## 4. Senior Engineering Standards

Both codebases adhere to these principles:
1. **Single Responsibility:** Every module and class has one clear responsibility.
2. **Dependency Injection:** External libraries (PyTorch, TRL, Docker) are injected behind Python `typing.Protocol` interfaces so core logic can be tested in milliseconds without GPUs or containers.
3. **Graceful Error Handling:** Explicit domain exceptions with actionable remediation messages (no bare `except Exception:`).
4. **Dependency Management:** Managed strictly via `uv` in a virtual environment (`.venv`).
5. **No Code Smells:** Pure functions for transformations, immutable data structures where possible, and zero circular dependencies.

---

## 5. Next Steps When Autotune Workspace Opens

When the IDE opens `/Users/adibshakib/Coding/Autotune`:
1. Run `uv init` and set up `pyproject.toml` with dependencies (`torch`, `transformers`, `peft`, `trl`, `bitsandbytes`, `pydantic`, `typer`).
2. Scaffold `src/autotune/core/ports/` with the protocols defined in this document.
3. Implement `AutoRecipeSelector` supporting both LoRA and 4-bit QLoRA.
4. Implement the evaluation suite ported from `forge/benchmark/`.
