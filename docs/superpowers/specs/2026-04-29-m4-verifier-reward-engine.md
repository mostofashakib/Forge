# M4: Verifier & Reward Engine Design

**Goal:** Replace generated verifier and reward stubs with six concrete, composable verifier types, a five-component decomposed reward, SemanticVerifier with on-disk caching, and a per-task adversarial test suite generated at compile time.

**Scope:** Pure Python backend. No web UI changes.

---

## 1. Architecture Overview

M4 adds `forge/runtime/verifiers/` — one file per verifier type. These are building blocks consumed by generated `verify_<task>.py` files; the existing `VerifierEngine`, `ForgeEnv`, and `VerificationResult`/`CheckResult` schemas are unchanged except for two targeted additions:

- `SuccessCondition` in `forge/extraction/schemas.py` gets a `rubric: str = ""` field (consumed only by `SemanticVerifier`)
- `ForgeEnv.step()` tracks `_invalid_action_count` per episode and passes it in the `task` dict to the reward engine
- `RewardConfig` in `forge/customization/config.py` gets two new fields: `semantic_weight: float = 0.0` and `invalid_action_penalty: float = 0.5`

The compiler templates `verifier.py.j2` and `reward.py.j2` are updated to use the new types. A new generator `AdversarialTestGenerator` writes `tests/<task_name>_adversarial.py` per task into each generated package.

---

## 2. File Map

```
forge/
  runtime/
    verifiers/
      __init__.py                   re-exports all six types
      exact_state.py                ExactStateVerifier
      event.py                      EventVerifier
      temporal.py                   TemporalVerifier
      policy.py                     PolicyVerifier
      semantic.py                   SemanticVerifier (live/cached/mock + SQLite cache)
      negative.py                   NegativeVerifier
  extraction/
    schemas.py                      SuccessCondition adds rubric: str = ""
  customization/
    config.py                       RewardConfig adds semantic_weight, invalid_action_penalty
  runtime/
    env.py                          ForgeEnv tracks _invalid_action_count
  compiler/
    generators/
      adversarial_test.py           AdversarialTestGenerator
  templates/
    verifier.py.j2                  Updated: dispatches to the 6 verifier types
    reward.py.j2                    Updated: 5-component decomposed reward
    test_adversarial.py.j2          New: adversarial test template
tests/
  runtime/
    verifiers/
      __init__.py
      test_exact_state.py
      test_event.py
      test_temporal.py
      test_policy.py
      test_semantic.py
      test_negative.py
  compiler/
    test_adversarial_generator.py   Tests AdversarialTestGenerator output
```

---

## 3. Six Verifier Types

All live in `forge/runtime/verifiers/`. Each has one public method:

```python
def check(self, state: dict, trajectory, task: dict) -> CheckResult
```

### 3.1 ExactStateVerifier

```python
ExactStateVerifier(expression: str)
```

Evaluates `expression` with `eval(expression, {"__builtins__": {}}, state)`. Passes if the result is truthy. `evidence` on failure is the string representation of the evaluated result (or the exception message on `eval` error).

### 3.2 EventVerifier

```python
EventVerifier(event_type: str)
```

Checks that `event_type` appears as the `"type"` key in at least one event in `trajectory.events`. Evidence on failure: `"Event '<event_type>' not found in trajectory"`.

### 3.3 TemporalVerifier

```python
TemporalVerifier(expression: str)
```

`expression` is `"A before B"` where A and B are event type strings separated by the literal word ` before ` (e.g. `"ask_for_order_id before offer_refund"`). Passes if the index of the first occurrence of A in `trajectory.events` is strictly less than the index of the first occurrence of B. Fails with evidence if either event is absent or ordering is wrong.

### 3.4 PolicyVerifier

```python
PolicyVerifier(forbidden_action: str)
```

Passes if `forbidden_action` does not appear as `action["type"]` in any step of the trajectory. Evidence on failure: `"Forbidden action '<name>' found at step <N>"`.

### 3.5 SemanticVerifier

```python
SemanticVerifier(
    rubric: str,
    state_field: str,
    mode: str = "mock",          # "mock" | "cached" | "live"
    cache_path: Path | None = None,
    llm_client=None,
)
```

Extracts `state[state_field]` as the text to evaluate. Scores 0.0–1.0.

**Modes:**

- `mock` — always returns score 1.0. Default when `FORGE_ENV=test` or when the constructor receives no `llm_client`.
- `cached` — looks up `(sha256(rubric), sha256(text))` in a SQLite file at `cache_path` (default: `custom/.semantic_cache.db` relative to the package). On miss, calls LLM and stores the result.
- `live` — calls LLM judge on every invocation. No caching.

**LLM judge prompt:** asks the judge to return a single float 0.0–1.0 scoring how well `text` satisfies `rubric`. Response is parsed with a regex `r"\b([01](?:\.\d+)?|\d*\.\d+)\b"`. On parse failure → score 0.0, evidence `"LLM judge parse error: <raw response>"`.

**Cache schema** (SQLite table `semantic_cache`):
```sql
CREATE TABLE IF NOT EXISTS semantic_cache (
    rubric_hash TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    score       REAL NOT NULL,
    PRIMARY KEY (rubric_hash, text_hash)
)
```

Passes if `score >= 0.5`.

### 3.6 NegativeVerifier

```python
NegativeVerifier(prohibited_action: str)
```

Passes if `prohibited_action` does NOT appear as `action["type"]` in any trajectory step. Semantically identical to `PolicyVerifier` but represents a task-level negative constraint (the distinction is meaningful to the compiler template). Evidence on failure: `"Prohibited action '<name>' found at step <N>"`.

---

## 4. Decomposed Reward

`reward.py.j2` generates a `compute_<task>_reward` function with five named components:

```python
reward = (
    task_success_reward        # avg(vr.score for vr in verifier_results) × cfg.base_success
  + policy_compliance_reward   # -cfg.policy_violation_penalty × policy_violation_count
  + semantic_quality_reward    # semantic_score × cfg.semantic_weight  (0.0 if no SemanticVerifier)
  - action_cost                # cfg.step_penalty × trajectory.step_count
  - invalid_action_penalty     # cfg.invalid_action_penalty × task.get("invalid_action_count", 0)
)
total = max(cfg.min_reward, min(cfg.max_reward, reward))
```

Config is loaded at call time from `load_config(Path(__file__).parent)` so no recompile is needed when weights change.

**`RewardConfig` additions** (in `forge/customization/config.py`):

| Field | Default | Meaning |
|---|---|---|
| `semantic_weight` | `0.0` | Multiplier for `SemanticVerifier` score in reward |
| `invalid_action_penalty` | `0.5` | Per invalid action deduction |

**`ForgeEnv` change:** Add `self._invalid_action_count: int = 0`, reset to 0 in `reset()`, increment in the invalid-action branch of `step()`. Pass as `task["invalid_action_count"]` when calling the reward engine.

---

## 5. Updated Compiler Templates

### 5.1 `verifier.py.j2`

Maps each `SuccessCondition.type` to the appropriate verifier class:

| `type` | Class instantiated | Constructor arg |
|---|---|---|
| `state_check` | `ExactStateVerifier` | `expression` |
| `event_check` | `EventVerifier` | `expression` |
| `temporal_check` | `TemporalVerifier` | `expression` |
| `policy_check` | `PolicyVerifier` | `expression` |
| `semantic_check` | `SemanticVerifier` | `rubric=rubric`, `state_field=expression`, `mode="mock"` |
| `negative_check` | `NegativeVerifier` | `expression` |

Generated function calls `verifier.check(state, trajectory, task)` for each condition and collects `CheckResult` objects into `VerificationResult.from_checks(...)`.

### 5.2 `reward.py.j2`

Generates the 5-component decomposed reward function described in Section 4. Reads config via `load_config`. Includes a `semantic_score` variable set to the score of the first `SemanticVerifier` result if present, else `0.0`.

---

## 6. Adversarial Test Suite

### 6.1 Generator

`AdversarialTestGenerator` in `forge/compiler/generators/adversarial_test.py` takes a `CompilerInput` and returns `dict[str, str]` mapping `<task_name>_adversarial` → Python test code.

Plugged into `PackageBuilder.build()` after `TestSuiteGenerator`. Output goes to `tests/<task_name>_adversarial.py`.

### 6.2 Rule-Based Patterns (deterministic verifiers)

For each `SuccessCondition` in a task, one adversarial test is generated:

| Verifier type | Adversarial trajectory constructed |
|---|---|
| `state_check` | Empty state dict (expression evaluates false) |
| `event_check` | Trajectory with no events |
| `temporal_check` | Trajectory where B event precedes A event |
| `policy_check` | Trajectory containing one step with the forbidden action type |
| `negative_check` | Trajectory containing one step with the prohibited action type |

Each test constructs a minimal `Trajectory` / `StepSnapshot` inline, calls the generated `verify_<task>` function, and asserts `result.passed is False`.

### 6.3 LLM-Based Pattern (SemanticVerifier)

When a task contains a `semantic_check` condition, `AdversarialTestGenerator` calls the LLM once at compile time with:

> "Given rubric: `<rubric>`. Generate one short text sample (1-2 sentences) that clearly FAILS this rubric."

The returned text is embedded as a string literal in the generated test. `SemanticVerifier` is instantiated with `mode="live"` and a `MockLLMClient` that always returns `"0.0"`, so the verifier calls the mock, receives a failing score, and reports `passed=False`. If no LLM client is available at compile time, the test is emitted with `pytest.skip("No LLM client available for semantic adversarial test")`.

---

## 7. Testing Strategy

- Each of the six verifier types has its own test file in `tests/runtime/verifiers/` with pass and fail cases
- `SemanticVerifier` tests cover all three modes: mock returns 1.0, cached reads/writes SQLite, live calls the LLM client
- Updated `test_verifier_generator.py` verifies that `verifier.py.j2` emits the correct class imports and instantiation for each `SuccessCondition.type`
- Updated `test_reward_generator.py` verifies that `reward.py.j2` emits all five component names
- `test_adversarial_generator.py` verifies that adversarial tests are generated for each verifier type and that they contain the expected assertions
- All generated adversarial tests must pass (i.e., they successfully catch the adversarial pattern) when run with `forge validate`

---

## 8. What Is Not in M4

- `observation_transform` / `policy_rule` wiring in `ForgeEnv` (M7)
- Frontend verifier result visualisation (M5)
- Parallel rollout execution (M6)
- RBAC observation filtering (M7)
