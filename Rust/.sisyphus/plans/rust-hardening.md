# Rust Server Hardening — Production Readiness Pass

## TL;DR

> **Quick Summary**: Hardening pass on the Rust Axum NN monitor server addressing 11 issues from a code health analysis: dead config fields (no CORS, no body limit), missing WebSocket heartbeat, silent error swallowing, untested ingest pipeline, unbounded persist buffer, plus minor code smells (code dedup, logging consistency, dead code, stale README).
>
> **Deliverables**:
> - `DefaultBodyLimit` middleware applied (using `config.max_request_size`)
> - `CorsLayer` middleware applied (using `config.cors_origins`)
> - WS protocol-level heartbeat (30s ping interval, 10s pong timeout)
> - Persist buffer cap at `max_steps_per_run`
> - Error logging replacing silent `.let_()` in store + ws_route
> - Ingest pipeline tests (queue-full 503, queue-closed 503, stats counters)
> - Dead code cleanup (7 compiler warnings → 0)
> - `now_iso()` deduplicated, `println!()` → `tracing`, README updated
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES — 3 waves, Wave 1 (5 parallel) + Wave 2 (7 parallel)
> **Critical Path**: T2 → T7 → T13 → T14 → FINAL

---

## Context

### Original Request
User asked for a plan to address all issues identified in the Rust subproject code health analysis (11 items across P0/P1/Minor priority tiers).

### Interview Summary
**Key Discussions**:
- Strategy: TDD — write failing test first, then fix, then verify all tests pass
- Scope: Everything — all 11 issues from critical to minor code smells
- Test infrastructure: 167 existing tests all passing (`cargo test`)

**Metis Review Findings** (incorporated):
- `tower-http` "cors" feature flag missing from Cargo.toml — must add before implementing CORS
- `add_metrics()` in store.rs is used by 3 integration tests — MUST NOT delete, silence with comment
- `FiniteF64::value()`, `FiniteF64::new()` used in unit tests — MUST NOT delete
- WebSocket heartbeat must use protocol-level `Message::Ping` (NOT application-level JSON) to avoid breaking test helpers
- Both `build_test_app()` (common/mod.rs) and `build_ws_test_app()` (test_websocket.rs) must be updated when adding middleware
- `AppState.config` will remain unused after middleware addition (middleware applied at router construction, not request time) — remove from AppState
- Heartbeat interval: 30s ping + 10s pong timeout (long enough to not break tests)
- Persist buffer cap: `max_steps_per_run` from config (same limit as in-memory store)
- CORS `"*"` interpretation: use `CorsLayer::permissive()` when list contains `"*"`
- DefaultBodyLimit is from `axum` core, not `tower-http`

---

## Work Objectives

### Core Objective
Harden the Rust server for production use by fixing all 11 issues from the code health analysis, with zero regressions to the 167 passing tests.

### Concrete Deliverables
- Modified `Cargo.toml` — add `"cors"` feature to tower-http
- Modified `src/main.rs` — router with `DefaultBodyLimit` + `CorsLayer`, `tracing::info!` replacing `println!`
- Modified `src/ws_route.rs` — heartbeat (30s ping, 10s pong timeout), error logging on sends
- Modified `src/store.rs` — error logging on persist failure, dead code cleanup
- Modified `src/persist.rs` — buffer cap, `now_iso()` import from shared util
- Modified `src/config.rs` — dead field cleanup
- Modified `src/models.rs` — dead code cleanup, `now_iso()` moved here or deleted
- Modified `src/ingest.rs` — add TDD-written tests
- Modified `tests/common/mod.rs` — update `build_test_app()` with middleware
- Modified `tests/test_websocket.rs` — update `build_ws_test_app()` with middleware
- Modified `Rust/README.md` — remove Python references
- New shared utility (or model addition) for `now_iso()`

### Definition of Done
- [ ] `cargo check` — zero warnings
- [ ] `cargo test` — all 167+ existing tests pass, new tests pass
- [ ] `cargo build --release` — succeeds
- [ ] `POST` with body >2MB → 413 Payload Too Large
- [ ] Request from external origin → `Access-Control-Allow-Origin` header present
- [ ] WS client idle >30s → receives Ping frame (protocol-level)
- [ ] WS client idle >40s (no Pong) → connection closed
- [ ] Persist buffer exceeds `max_steps_per_run` → oldest steps dropped with `warn!`
- [ ] Ingest queue full → POST returns 503 Service Unavailable

### Must Have
- `DefaultBodyLimit` middleware on router using `config.max_request_size`
- `CorsLayer` middleware on router using `config.cors_origins` (permissive when `"*"`)
- WS protocol-level `Message::Ping` heartbeat: 30s interval, 10s pong timeout
- `tracing::error!()` (not silent `.let_()`) on persist buffer_step failure
- `tracing::error!()` (not silent `.let_()`) on WS sender.send failure
- Ingest tests: queue-full 503, queue-closed 503, stats counters
- Persist `buffer_step()` cap at `config.max_steps_per_run`
- Single `now_iso()` function (not duplicated)
- `tracing::info!()` in main.rs (not `println!()`)
- Zero `cargo check` warnings
- README.md free of Python references (main.py, pip, pydantic, etc.)

### Must NOT Have (Guardrails)
- No new crate dependencies (feature flags only)
- No application-level JSON heartbeat messages
- Do NOT delete `add_metrics()` from store.rs (used by 3 integration tests)
- Do NOT delete `FiniteF64::value()`, `FiniteF64::new()`, `NonNegativeF64::value()` (used in unit tests)
- No changes to API contract (endpoints, status codes except intentional additions)
- No changes to WebSocket message format (protocol-level ping is invisible to clients)
- No touching `API.md`, `SCHEMA.md`, `CONFIGURATION.md`, `docs/`
- Heartbeat interval NOT configurable (use constants, not ServerConfig fields)
- No refactoring of unrelated code
- No stress/performance/benchmark tests

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (cargo test with tokio::test, axum-test in dev-deps)
- **Automated tests**: YES (TDD — write failing test first, then minimal fix)
- **Framework**: cargo test + tokio::test + axum-test
- **TDD workflow**: RED (failing test) → GREEN (minimal impl) → REFACTOR → `cargo test` all

### QA Policy
Every task includes agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — ALL INDEPENDENT, 5 tasks):
├── T1: Dead code cleanup (7 warnings → 0)
├── T2: Add "cors" feature + update test routers
├── T3: Deduplicate now_iso() into shared location
├── T4: Convert println!() → tracing::info!() in main.rs
└── T5: Update Rust/README.md to remove Python references

Wave 2 (Core hardening — MAX PARALLEL, 7 tasks):
├── T6: Wire DefaultBodyLimit (TDD) [quick]
├── T7: Wire CorsLayer (TDD) [quick, depends on T2]
├── T8: Add WS heartbeat (TDD) [deep]
├── T9: Add persist buffer cap (TDD) [quick]
├── T10: Log persist errors (no swallow) [quick]
├── T11: Log WS send errors (no swallow) [quick]
└── T12: Add ingest pipeline tests (TDD) [unspecified-high]

Wave 3 (Integration + cleanup, 2 tasks):
├── T13: Verify & update test infrastructure for middleware
└── T14: Final integration verification [unspecified-high]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)
→ Present results → Get explicit user okay

Critical Path: T2 → T7 → T13 → T14 → FINAL
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 7 (Wave 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | - | - | 1 |
| 2 | - | 7, 13 | 1 |
| 3 | - | - | 1 |
| 4 | - | - | 1 |
| 5 | - | - | 1 |
| 6 | - | 13 | 2 |
| 7 | 2 | 13 | 2 |
| 8 | - | - | 2 |
| 9 | - | - | 2 |
| 10 | - | - | 2 |
| 11 | - | - | 2 |
| 12 | - | - | 2 |
| 13 | 2, 6, 7 | 14 | 3 |
| 14 | 13 | F1-F4 | 3 |

### Agent Dispatch Summary

- **Wave 1**: 5 tasks — T1→`deep`, T2→`quick`, T3→`quick`, T4→`quick`, T5→`quick`
- **Wave 2**: 7 tasks — T6→`quick`, T7→`quick`, T8→`deep`, T9→`quick`, T10→`quick`, T11→`quick`, T12→`unspecified-high`
- **Wave 3**: 2 tasks — T13→`quick`, T14→`unspecified-high`
- **FINAL**: 4 tasks — F1→`oracle`, F2→`unspecified-high`, F3→`unspecified-high`, F4→`deep`

---

## TODOs

- [x] 1. **Dead Code Cleanup** — 7 warnings → 0

  **What to do** (TDD: verify warnings exist first, fix, verify zero):
  1. Run `cargo check 2>&1 | grep "warning:" | wc -l` → expect 7 before fix, verify 0 after
  2. **`config.rs:50,60`** — `max_request_size` and `cors_origins` fields: will become "read" after T6/T7. Remove from dead code list — these belong to T6/T7 fixes.
  3. **`models.rs:11,14`** — `FiniteF64::value()` and `FiniteF64::new()`: used in unit tests (models.rs:553-568, persist.rs:361). Add `#[allow(dead_code)]` above each with comment `// Used in tests`.
  4. **`models.rs:59`** — `NonNegativeF64::value()`: check if used in unit tests. If yes, `#[allow(dead_code)]`. If no, delete.
  5. **`models.rs:250-254`** — `ErrorDetail` struct: confirmed dead (Metis verified). SaFE TO DELETE.
  6. **`routes/mod.rs:17`** — `AppState.config` field: confirmed will remain unused even after T6/T7 (middleware applied at router construction, not request time). REMOVE the `config` field from `AppState`. Update all constructors (main.rs, both test builders) to not pass config into AppState.
  7. **`store.rs:177`** — `add_metrics()`: used by 3 integration tests (test_storage.rs:150,157,162). DO NOT DELETE. Add `// Note: called from integration tests (tests/test_storage.rs)` comment above function to explain why the compiler sees it as dead.
  8. **`store.rs:323`** — `get_all_runs()`: confirmed truly dead (Metis verified). DELETE.
  9. **`store.rs:394`** — `get_latest_step()`: confirmed truly dead (Metis verified). DELETE.
  10. **`ws.rs:82`** — `build_initial_runs_message()` (standalone fn): the store has its own method `store.rs:417` that is actually called. The standalone was kept only for ws.rs inline test at ws.rs:327. MOVE the test assertion into the store's test module, then DELETE the standalone fn.

  **Must NOT do**:
  - Do NOT delete `add_metrics()` — it breaks 3 integration tests
  - Do NOT delete `FiniteF64::value()` / `FiniteF64::new()` — they break unit tests
  - Do NOT delete `config.rs` fields for max_request_size/cors_origins — they're used by T6/T7
  - Do NOT refactor any production logic — this is purely dead code suppression/removal

  **Recommended Agent Profile**:
  > `deep` — requires careful cross-referencing of test vs production usage. Must verify each symbol individually before acting.
  - **Category**: `deep`
    - Reason: Symbol-level dead code analysis requires methodical verification of every warning against test usage. Risk of breaking tests if done carelessly.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2-T5)
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/models.rs:11-14` — `FiniteF64::value()` and `FiniteF64::new()` (used in unit tests at lines 553-568)
  - `src/models.rs:59` — `NonNegativeF64::value()` (check test usage)
  - `src/models.rs:250-254` — `ErrorDetail` struct (dead, safe to delete)
  - `src/routes/mod.rs:14-24` — `AppState` struct with `config` field
  - `src/store.rs:177` — `add_metrics()` (used by tests/test_storage.rs:150,157,162)
  - `src/store.rs:323` — `get_all_runs()` (truly dead)
  - `src/store.rs:394` — `get_latest_step()` (truly dead)
  - `src/ws.rs:82` — `build_initial_runs_message()` standalone fn
  - `src/ws.rs:327` — inline test using the standalone fn
  - `src/store.rs:417` — the ACTUAL `build_initial_runs_message` method on MetricsStore
  - `tests/test_storage.rs:150,157,162` — integration tests calling `add_metrics()`

  **Acceptance Criteria** (TDD):
  - [ ] RED: `cargo check 2>&1 | grep "warning:" | wc -l` → 7
  - [ ] GREEN: `cargo check 2>&1 | grep "warning:" | wc -l` → 0 (or ≤ real remaining)
  - [ ] `cargo test` — all 167 existing tests pass (zero regressions)

  **QA Scenarios**:
  ```
  Scenario: Zero compiler warnings after cleanup
    Tool: Bash
    Preconditions: All deletions/suppressions applied
    Steps:
      1. Run `cargo check 2>&1`
      2. Count warnings: `cargo check 2>&1 | grep "warning:" | wc -l`
    Expected Result: Output is "0"
    Failure Indicators: Non-zero warning count, or any "error:" in output
    Evidence: .sisyphus/evidence/task-1-warnings.txt

  Scenario: All existing tests still pass
    Tool: Bash
    Steps:
      1. Run `cargo test 2>&1`
      2. Count failures: `cargo test 2>&1 | grep "FAILED" | wc -l`
    Expected Result: "0" failures, all test suites show "test result: ok"
    Failure Indicators: Any "FAILED" in output, any "test result: FAILED"
    Evidence: .sisyphus/evidence/task-1-tests.txt

  Scenario: add_metrics() still callable from integration tests
    Tool: Bash
    Steps:
      1. Run `cargo test test_step_dedup_replaces_data test_run_eviction_after_max test_step_eviction_after_max`
    Expected Result: All 3 tests pass (these use add_metrics directly)
    Evidence: .sisyphus/evidence/task-1-add-metrics.txt
  ```

  **Commit**: YES (groups with Wave 1)
  - Message: `chore: cleanup dead code warnings (7→0)`
  - Files: `src/models.rs`, `src/store.rs`, `src/ws.rs`, `src/routes/mod.rs`

- [x] 2. **Add `"cors"` Feature + Update Test Routers**

  **What to do**:
  1. Add `"cors"` to tower-http features in `Cargo.toml:10`: change `features = ["fs", "trace", "timeout"]` to `features = ["fs", "trace", "timeout", "cors"]`
  2. Update `tests/common/mod.rs:build_test_app()` — after T1 removes `AppState.config`, this function must be updated to not pass `config: config.clone()`. Also, add the `/ws` route to `build_test_app()` so that `test_websocket.rs` can use `common/mod.rs`'s builder instead of its own duplicated one.
  3. Update `tests/test_websocket.rs:build_ws_test_app()` — DELETE the duplicated function. Import and use `common::build_test_app()` instead. The only difference was the `/ws` route — now it's in common.
  4. Verify after the change that `test_websocket.rs` tests still pass with the common builder.
  5. Run `cargo check` to verify the `"cors"` feature resolves correctly.

  **Must NOT do**:
  - No other feature flag changes
  - No new crate dependencies
  - Don't implement CORS logic yet (that's T7)

  **Recommended Agent Profile**:
  > `quick` — simple Cargo.toml edit + test refactor. Low risk, well-understood.
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T1, T3, T4, T5)
  - **Parallel Group**: Wave 1
  - **Blocks**: T7, T13
  - **Blocked By**: None

  **References**:
  - `Cargo.toml:10` — tower-http feature list
  - `tests/common/mod.rs:9-68` — `build_test_app()` without /ws route
  - `tests/test_websocket.rs:18-69` — duplicated `build_ws_test_app()` with only /ws route added
  - `src/routes/ws_route.rs:1-7` — `ws_handler` function to add to common builder
  - `tests/test_websocket.rs:163,218,264,330,377,422` — tests importing `build_ws_test_app`

  **Acceptance Criteria**:
  - [ ] `cargo check` — no errors, `"cors"` feature resolved
  - [ ] `tests/common/mod.rs` includes `/ws` route in `build_test_app()`
  - [ ] `tests/test_websocket.rs` imports `build_test_app` from `common`, no `build_ws_test_app`
  - [ ] `cargo test —test test_websocket` — all 13 WS tests pass with common builder

  **QA Scenarios**:
  ```
  Scenario: Cors feature resolution
    Tool: Bash
    Steps:
      1. Run `cargo check 2>&1`
    Expected Result: No errors, no "failed to resolve" for cors feature
    Evidence: .sisyphus/evidence/task-2-cors-feature.txt

  Scenario: WebSocket tests still pass with common builder
    Tool: Bash
    Steps:
      1. Run `cargo test --test test_websocket 2>&1`
    Expected Result: All 13 tests pass, "test result: ok"
    Evidence: .sisyphus/evidence/task-2-ws-tests.txt

  Scenario: No duplicated build function
    Tool: Bash
    Steps:
      1. Run `grep -rn "build_ws_test_app" tests/`
    Expected Result: No results (function deleted)
    Evidence: .sisyphus/evidence/task-2-dedup.txt
  ```

  **Commit**: YES
  - Message: `chore(deps): add cors feature to tower-http, dedup test builders`
  - Files: `Cargo.toml`, `tests/common/mod.rs`, `tests/test_websocket.rs`

- [x] 3. **Deduplicate `now_iso()` into Shared Location**

  **What to do**:
  1. Identify the two identical copies: `store.rs:24-26` and `persist.rs:76-78`
  2. Move a single `pub(crate) fn now_iso() -> String` to `src/models.rs` (both store.rs and persist.rs already import models). Place it near the top of models.rs in a utility section, with a comment.
  3. In `store.rs`: delete the local `fn now_iso()`, add `use crate::models::now_iso;`
  4. In `persist.rs`: delete the local `fn now_iso()`, add `use crate::models::now_iso;`
  5. Run `cargo check` → verify zero new warnings
  6. Run `cargo test` → all pass

  **Must NOT do**:
  - Don't create a new file just for this (over-engineering)
  - Don't change the function signature or behavior

  **Recommended Agent Profile**:
  > `quick` — simple code deduplication with clear references.
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T1, T2, T4, T5)
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/store.rs:24-26` — local `fn now_iso()` copy 1
  - `src/persist.rs:76-78` — local `fn now_iso()` copy 2
  - `src/models.rs:1-5` — module head where `now_iso()` will be added
  - Both files already `use crate::models::...` — confirms models import exists

  **Acceptance Criteria**:
  - [ ] `grep -rn "fn now_iso" src/` returns exactly 1 result (in models.rs)
  - [ ] `cargo check` — zero errors, zero new warnings
  - [ ] `cargo test` — all tests pass

  **QA Scenarios**:
  ```
  Scenario: Single now_iso definition
    Tool: Bash
    Steps:
      1. Run `grep -rn "fn now_iso" src/`
    Expected Result: Exactly 1 match in src/models.rs
    Evidence: .sisyphus/evidence/task-3-dedup.txt

  Scenario: All tests pass after refactor
    Tool: Bash
    Steps:
      1. Run `cargo test 2>&1 | grep "test result:"`
    Expected Result: All test suites show "ok" with 0 failures
    Evidence: .sisyphus/evidence/task-3-tests.txt
  ```

  **Commit**: YES
  - Message: `refactor: deduplicate now_iso() into models.rs`
  - Files: `src/models.rs`, `src/store.rs`, `src/persist.rs`

- [x] 4. **Convert `println!()` → `tracing::info!()` in main.rs**

  **What to do**:
  1. In `src/main.rs`, replace ALL `println!()` calls with `tracing::info!()`:
     - Line ~32: `tracing::info!("Received Ctrl+C, shutting down gracefully...")`
     - Line ~33: `tracing::info!("Flushing persistence buffers...")`
     - Line ~37: `tracing::info!("Shutdown complete. Goodbye!")`
     - Lines ~104-120: The startup banner — each `println!()` line becomes `tracing::info!(...)`
  2. The startup banner currently uses `println!("{}", "─".repeat(60))` — keep this visual separator but use `tracing::info!("{}", "─".repeat(60))`
  3. Run `cargo check` → verify zero warnings from main.rs
  4. Run `cargo run` briefly → verify banner and shutdown messages appear in tracing output
  5. Verify `tracing-subscriber` is initialized before the first `tracing::info!()` call (it is — `main.rs` initializes subscriber before the banner)

  **Must NOT do**:
  - Don't change message content — only the macro
  - Don't add new dependencies

  **Recommended Agent Profile**:
  > `quick` — straightforward macro replacement with grep-verify.
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T1, T2, T3, T5)
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/main.rs:28-38` — shutdown_signal with println!
  - `src/main.rs:90-120` — startup banner with println! blocks
  - `src/main.rs:6` — `use tracing;` already imported
  - `src/main.rs:147` — `tracing_subscriber::fmt::init()` — subscriber initialized before banner

  **Acceptance Criteria**:
  - [ ] `grep -n "println!" src/main.rs` returns zero results
  - [ ] `grep -n "tracing::info!" src/main.rs` returns all startup/shutdown messages
  - [ ] `cargo check` — zero new warnings
  - [ ] `cargo test` — all tests pass

  **QA Scenarios**:
  ```
  Scenario: Zero println! in main.rs
    Tool: Bash
    Steps:
      1. Run `grep -c "println!" src/main.rs`
    Expected Result: "0"
    Evidence: .sisyphus/evidence/task-4-println.txt

  Scenario: Server starts and logs via tracing
    Tool: Bash
    Preconditions: Build complete
    Steps:
      1. Start server: `timeout 3 cargo run 2>&1 || true`
      2. Check output contains startup messages
    Expected Result: Banner appears via tracing, no raw println output
    Failure Indicators: Server panics, or "println" still in output
    Evidence: .sisyphus/evidence/task-4-banner.txt
  ```

  **Commit**: YES
  - Message: `refactor: replace println!() with tracing::info!() in main.rs`
  - Files: `src/main.rs`

- [x] 5. **Update README.md — Remove Python References**

  **What to do**:
  1. Read `Rust/README.md` (68 lines). Identify all Python-specific content:
     - Line 3: "FastAPI backend" → "Axum backend"
     - Line 39: `ServerConfig` reference tagged "(main.py:21-38)" → "(config.rs:35-60)"
     - Line 40: `MetricsStore` reference tagged "(main.py:250-326)" → "(store.rs:161-346)"
     - Line 41: `ConnectionManager` reference tagged "(main.py:334-366)" → "(ws.rs:17-66)"
     - Line 43: "Pydantic models (main.py:47-237)" → "Serde models (models.rs:96-237)"
     - Line 63: "async with self._lock for thread safety" → "tokio::sync::RwLock for async safety"
  2. Add Rust-specific sections:
     - Mention `cargo build --release` and `cargo run` as primary run commands
     - Mention `cargo test` for testing
     - Note JSONL persistence feature: configure via `NN_MONITOR_DATA_DIR`
  3. Verify after edits: `grep -ci "python\|main.py\|pip install\|pydantic\|FastAPI" Rust/README.md` returns 0

  **Must NOT do**:
  - Don't touch `API.md`, `SCHEMA.md`, `CONFIGURATION.md`, or `docs/`
  - Don't rewrite the entire README — targeted fixes only
  - Don't remove the Python test client references (test_client.py files are actual test tools in the Rust project)

  **Recommended Agent Profile**:
  > `writing` — documentation update with grep-verify. Targeted, not creative.
  - **Category**: `writing`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T1, T2, T3, T4)
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `Rust/README.md:1-68` — full file with Python references to fix
  - `src/config.rs:35-60` — ServerConfig (correct reference)
  - `src/store.rs:161-346` — MetricsStore (correct reference)
  - `src/ws.rs:17-66` — WsManager (correct reference)
  - `src/models.rs:96-237` — models (correct reference)

  **Acceptance Criteria**:
  - [ ] `grep -ci "python\|main.py\|pip install\|pydantic\|FastAPI" Rust/README.md` returns 0
  - [ ] README.md mentions `cargo build`, `cargo run`, `cargo test`
  - [ ] Line number references point to Rust source files (config.rs, store.rs, ws.rs, models.rs)

  **QA Scenarios**:
  ```
  Scenario: Zero Python references in README
    Tool: Bash
    Steps:
      1. Run `grep -ci "python\|main.py\|pip install\|pydantic\|FastAPI" Rust/README.md`
    Expected Result: "0"
    Evidence: .sisyphus/evidence/task-5-readme-py.txt

  Scenario: Rust build instructions present
    Tool: Bash
    Steps:
      1. Run `grep -c "cargo build\|cargo run\|cargo test" Rust/README.md`
    Expected Result: ≥ 3 (at least one mention of each)
    Evidence: .sisyphus/evidence/task-5-readme-rust.txt
  ```

  **Commit**: YES
  - Message: `docs: update README.md to reflect Rust implementation`
  - Files: `Rust/README.md`

- [x] 6. **Wire `DefaultBodyLimit` Middleware** ⬡ TDD

  **What to do** (RED → GREEN → REFACTOR):
  1. **RED**: Write a test in `tests/test_edge_cases.rs`:
     - Add test `test_body_too_large_rejected_before_parsing_413`: POST a body > 2MB with valid JSON shape, assert 413 Payload Too Large
     - Note: `build_test_app()` currently has NO body limit, so this test will FAIL first (the server accepts the oversized payload as 422 or 202 instead of 413)
  2. **GREEN**: In `src/main.rs`, add `.layer(DefaultBodyLimit::max(config.max_request_size))` to the router chain (before other layers)
  3. Also add the layer to `tests/common/mod.rs`'s `build_test_app()` so the integration test environment matches
  4. Run `cargo test test_body_too_large_rejected_before_parsing_413` → PASS
  5. Run `cargo test` → all tests pass
  6. **REFACTOR**: Verify `cargo check` — `max_request_size` should no longer be flagged as dead code

  **Must NOT do**:
  - Don't use `tower_http::limit::RequestBodyLimit` — use `axum::extract::DefaultBodyLimit` (built into axum 0.8)
  - Don't change the default `max_request_size` value (2MB)
  - Don't remove the field from config.rs

  **Recommended Agent Profile**:
  > `quick` — simple middleware addition with well-known axum API.
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T7-T12 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: T13
  - **Blocked By**: None

  **References**:
  - `src/main.rs:78-96` — router construction where `.layer()` must be added
  - `src/config.rs:49-50` — `max_request_size: usize` (default 2MB)
  - `tests/common/mod.rs:9-68` — `build_test_app()` router construction
  - `tests/test_edge_cases.rs:1-10` — where new test will be added (near other boundary tests)
  - Axum docs: `DefaultBodyLimit` is at `axum::extract::DefaultBodyLimit`

  **Acceptance Criteria** (TDD):
  - [ ] RED: `cargo test test_body_too_large_rejected_before_parsing_413` → FAIL (no 413 returned)
  - [ ] GREEN: Same test → PASS (413 returned for >2MB body)
  - [ ] `cargo test` — all existing tests pass
  - [ ] `cargo check` — `max_request_size` NOT in warnings

  **QA Scenarios**:
  ```
  Scenario: Oversized body returns 413
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_body_too_large_rejected_before_parsing_413`
    Expected Result: Test passes — 413 status code returned for >2MB POST body
    Evidence: .sisyphus/evidence/task-6-413.txt

  Scenario: Normal body still accepted
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_valid_metrics_accepted`
    Expected Result: Test passes — normal payload still returns 202
    Evidence: .sisyphus/evidence/task-6-normal.txt
  ```

  **Commit**: YES
  - Message: `feat(security): apply DefaultBodyLimit middleware`
  - Files: `src/main.rs`, `tests/common/mod.rs`, `tests/test_edge_cases.rs`

- [x] 7. **Wire `CorsLayer` Middleware** ⬡ TDD

  **What to do** (RED → GREEN → REFACTOR):
  1. **RED**: Write a test in `tests/test_runs_endpoint.rs`:
     - Add test `test_cors_headers_present_on_get_runs`: GET `/api/v1/runs` with `Origin: http://localhost:3000` header, assert `Access-Control-Allow-Origin` header present in response
     - Currently no CORS middleware → test FAILS (no CORS headers)
  2. **GREEN**: In `src/main.rs`, add CorsLayer to router:
     - If `config.cors_origins` contains `"*"`: use `CorsLayer::permissive()` (allow all origins, methods, headers)
     - Otherwise: use `CorsLayer::new().allow_origin(config.cors_origins.iter().map(|o| o.parse::<HeaderValue>().unwrap()).collect::<Vec<_>>())`
  3. Add same CorsLayer to `tests/common/mod.rs`'s `build_test_app()`
  4. Run `cargo test test_cors_headers_present_on_get_runs` → PASS
  5. Run `cargo test` → all tests pass
  6. **REFACTOR**: Verify `cargo check` — `cors_origins` should no longer be flagged as dead code

  **Must NOT do**:
  - Don't use `.allow_any_origin()` with `.allow_credentials(true)` simultaneously (browser security error)
  - Don't change CORS for WebSocket upgrade path (CorsLayer doesn't affect upgrades)
  - Don't forget to add `use axum::http::HeaderValue;` import

  **Recommended Agent Profile**:
  > `quick` — straightforward middleware addition. Dependency already resolved by T2.
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6, T8-T12 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: T13
  - **Blocked By**: T2 (cors feature must be enabled)

  **References**:
  - `src/main.rs:78-96` — router construction where `.layer()` must be added
  - `src/config.rs:59-60` — `cors_origins: Vec<String>` (default `["*"]`)
  - `tests/common/mod.rs:9-68` — `build_test_app()` router
  - `tests/test_runs_endpoint.rs:1-10` — where new test will be added
  - `Cargo.toml:10` — tower-http now with "cors" feature (added by T2)
  - Tower-http docs: `CorsLayer::permissive()` for `"*"` → allow all

  **Acceptance Criteria** (TDD):
  - [ ] RED: `cargo test test_cors_headers_present_on_get_runs` → FAIL (no CORS headers)
  - [ ] GREEN: Same test → PASS (Access-Control-Allow-Origin header present)
  - [ ] `cargo test` — all existing tests pass
  - [ ] `cargo check` — `cors_origins` NOT in warnings

  **QA Scenarios**:
  ```
  Scenario: CORS headers present for allowed origin
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_cors_headers_present_on_get_runs`
    Expected Result: Test passes — Access-Control-Allow-Origin header present
    Evidence: .sisyphus/evidence/task-7-cors.txt

  Scenario: Normal request still works
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_single_run test_health_endpoint`
    Expected Result: Both tests pass — endpoints work with CORS middleware
    Evidence: .sisyphus/evidence/task-7-normal.txt
  ```

  **Commit**: YES
  - Message: `feat(security): apply CorsLayer middleware`
  - Files: `src/main.rs`, `tests/common/mod.rs`, `tests/test_runs_endpoint.rs`

- [x] 8. **Add WebSocket Protocol-Level Heartbeat** ⬡ TDD

  **What to do** (RED → GREEN → REFACTOR):
  1. **RED**: Write test `test_ws_heartbeat_ping_received` in `tests/test_websocket.rs`:
     - Connect WebSocket client
     - Read initial `initial_runs` message
     - Wait 32 seconds (slightly >30s ping interval + safety margin)
     - Read next message — assert it is a `Message::Ping` frame
     - Test currently FAILS (no ping sent after 32s)
  2. **GREEN**: Modify `ws_route.rs:43-69` select! loop:
     - Add `let mut heartbeat = tokio::time::interval(Duration::from_secs(30));` before the loop
     - Add new arm: `tick = heartbeat.tick() => { let _ = sender.send(Message::Ping(vec![])).await; }`
     - Add pong tracking: `let mut last_pong = Instant::now();` before loop
     - In the `receiver.next()` arm, handle `Message::Pong(_)` → `last_pong = Instant::now();`
     - Before sending ping, check: if `last_pong.elapsed() > Duration::from_secs(40)` → `break;` (close connection)
  3. Run `cargo test test_ws_heartbeat_ping_received` → PASS
  4. Run ALL WebSocket tests → all pass (ping is protocol-level, invisible to test helpers that filter for `Message::Text`)
  5. Run `cargo test` → all tests pass

  **Must NOT do**:
  - Don't use application-level JSON `{"type":"ping"}` messages — must be `Message::Ping(vec![])` (protocol-level frame)
  - Don't make heartbeat interval configurable — use constants
  - Don't change existing WS message types or formats
  - Don't skip heartbeat logic for tests (the 30s interval is long enough that tests finish before first ping)

  **Recommended Agent Profile**:
  > `deep` — modifies the hot-path WebSocket select! loop. Must handle pong tracking, timed closure, and ensure no test breakage. Requires careful understanding of axum WS and tokio::select! semantics.
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6, T7, T9-T12 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/ws_route.rs:43-69` — select! loop to modify (currently 2 arms: broadcast_rx + receiver)
  - `tests/test_websocket.rs:91-109` — `read_ws_message()` helper (filters for `Message::Text`, Ping invisible)
  - `tests/test_websocket.rs:163` — test_ws_connect_and_initial_runs (pattern for new heartbeat test)
  - Tokio docs: `tokio::time::interval` for periodic heartbeat ticks
  - Axum WS docs: `Message::Ping(vec![])` and `Message::Pong(vec![])` are protocol-level frames

  **Acceptance Criteria** (TDD):
  - [ ] RED: `cargo test test_ws_heartbeat_ping_received` → FAIL (no ping after 32s)
  - [ ] GREEN: Same test → PASS (Ping frame received after 32s)
  - [ ] `cargo test --test test_websocket` — all 13 tests pass
  - [ ] `cargo test` — all 167+ tests pass
  - [ ] Manual: connection idle >30s receives Ping, idle >40s closed

  **QA Scenarios**:
  ```
  Scenario: Heartbeat sends Ping after 30s idle
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_ws_heartbeat_ping_received`
    Expected Result: Test passes — Ping frame received within 32s
    Evidence: .sisyphus/evidence/task-8-ping.txt

  Scenario: All existing WS tests unaffected
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test --test test_websocket 2>&1 | grep "test result:"`
    Expected Result: "test result: ok" with 0 failures
    Evidence: .sisyphus/evidence/task-8-regression.txt
  ```

  **Commit**: YES
  - Message: `feat(ws): add protocol-level heartbeat (30s ping, 10s pong timeout)`
  - Files: `src/ws_route.rs`, `tests/test_websocket.rs`

- [x] 9. **Add Persist Buffer Size Cap** ⬡ TDD

  **What to do** (RED → GREEN → REFACTOR):
  1. **RED**: Write inline test `test_buffer_cap_limits_size` in `src/persist.rs` `#[cfg(test)] mod tests`:
     - Create JsonlStore with `max_buffer_size = 5`
     - Buffer 7 steps to the same run_id
     - Assert buffer only contains 5 steps (oldest 2 dropped)
     - Assert `flush_run` writes only 5 steps to JSONL
     - Test currently FAILS (buffer grows unbounded)
  2. **GREEN**: Modify `persist.rs`:
     - Add `max_buffer_size: usize` field to `JsonlStore` struct
     - Update `JsonlStore::new()` to accept `max_buffer_size` parameter
     - In `buffer_step()`: after appending, if `run_buffer.steps.len() > self.max_buffer_size`, drain oldest steps with `tracing::warn!("Buffer cap reached for run {}, dropping {} oldest steps", run_id, excess)`
     - Drain from the front: `let excess = run_buffer.steps.len() - self.max_buffer_size; run_buffer.steps.drain(0..excess);`
  3. Update `main.rs` to pass `config.max_steps_per_run` as the buffer cap when creating JsonlStore
  4. Run `cargo test test_buffer_cap_limits_size` → PASS
  5. Run `cargo test` → all tests pass
  6. **REFACTOR**: Verify the cap matches in-memory limit (`max_steps_per_run`)

  **Must NOT do**:
  - Don't add a separate config field for buffer cap — reuse `max_steps_per_run`
  - Don't change the `buffer_step()` return type

  **Recommended Agent Profile**:
  > `quick` — simple capacity check with well-defined behavior.
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6-T8, T10-T12 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/persist.rs:108-119` — `buffer_step()` where cap check must be added
  - `src/persist.rs:71-101` — `JsonlStore::new()` signature to add parameter
  - `src/persist.rs:350-394` — existing inline test module (add new test here)
  - `src/main.rs:44-55` — where JsonlStore is constructed (pass max_steps_per_run)
  - `src/config.rs:47` — `max_steps_per_run: usize` (default 10000) used as buffer cap

  **Acceptance Criteria** (TDD):
  - [ ] RED: `cargo test test_buffer_cap_limits_size` → FAIL
  - [ ] GREEN: Same test → PASS (buffer capped at 5)
  - [ ] `cargo test` — all tests pass
  - [ ] `grep -n "tracing::warn!" src/persist.rs` confirms warning log on overflow

  **QA Scenarios**:
  ```
  Scenario: Buffer cap limits size
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_buffer_cap_limits_size`
    Expected Result: Buffer holds exactly 5 steps, oldest dropped, flush writes 5
    Evidence: .sisyphus/evidence/task-9-cap.txt

  Scenario: Normal buffer under cap unaffected
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_jsonl_write_and_read_roundtrip`
    Expected Result: Roundtrip test passes — 3 steps under cap of 10000
    Evidence: .sisyphus/evidence/task-9-roundtrip.txt
  ```

  **Commit**: YES
  - Message: `fix(persist): add buffer size cap at max_steps_per_run`
  - Files: `src/persist.rs`, `src/main.rs`

- [x] 10. **Log Persist Errors Instead of Silently Swallowing** ⬡ TDD

  **What to do** (RED → GREEN → REFACTOR):
  1. **RED**: Write inline test `test_persist_error_is_logged` in `src/store.rs` test module:
     - Create MetricsStore with a persist layer that will fail (e.g., data_dir set to `/dev/null/invalid_path` so buffer_step fails)
     - Call `insert_step_data` with valid payload
     - Verify the function does NOT panic (returns Ok) — error is logged, not surfaced
     - Test currently verifies behavior exists but can't easily capture tracing output. Instead: verify no panic and store data still inserted correctly (error resilience)
  2. **GREEN**: Modify `src/store.rs:260`:
     - Change `let _ = jsonl_store.buffer_step(&run_id, &step_data_for_persist).await;` to:
       ```rust
       if let Err(e) = jsonl_store.buffer_step(&run_id, &step_data_for_persist).await {
           tracing::error!(
               run_id = %run_id,
               error = %e,
               "Failed to buffer step for persistence"
           );
       }
       ```
  3. Run `cargo test test_persist_error_is_logged` → PASS
  4. Run `cargo test` → all tests pass

  **Must NOT do**:
  - Don't propagate the error to the caller — error must be logged, not returned
  - Don't change the function signature
  - Don't panic on persist failure

  **Recommended Agent Profile**:
  > `quick` — single line change with well-defined behavior.
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6-T9, T11, T12 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/store.rs:260` — `let _ = jsonl_store.buffer_step(...)` to replace
  - `src/store.rs:249-262` — `insert_step_data` function context
  - `src/persist.rs:20-35` — `PersistError` type (implements Display)
  - `src/store.rs:551-720` — inline test module (add new test here)

  **Acceptance Criteria** (TDD):
  - [ ] RED: `cargo test test_persist_error_is_logged` → FAIL (no error logging)
  - [ ] GREEN: Same test → PASS (store still works after persist failure, no panic)
  - [ ] `grep "tracing::error!" src/store.rs` returns the new log call
  - [ ] `cargo test` — all tests pass

  **QA Scenarios**:
  ```
  Scenario: Store survives persist failure
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_persist_error_is_logged`
    Expected Result: Store inserts data successfully despite persist layer failure
    Evidence: .sisyphus/evidence/task-10-resilient.txt

  Scenario: Normal persist still works
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_jsonl_write_and_read_roundtrip test_lazy_load_from_jsonl`
    Expected Result: Both persist tests pass
    Evidence: .sisyphus/evidence/task-10-normal.txt
  ```

  **Commit**: YES
  - Message: `fix(store): log persist errors instead of silently swallowing`
  - Files: `src/store.rs`

- [x] 11. **Log WebSocket Send Errors Instead of Silently Swallowing** ⬡ TDD

  **What to do** (RED → GREEN → REFACTOR):
  1. **RED**: Write inline test `test_ws_send_error_logged_on_disconnect` in `src/ws.rs` test module:
     - Create WsManager, connect client, get sender
     - Drop the receiver (simulating client disconnect)
     - Send a message via the sender
     - Assert the send returns an error (receiver dropped)
     - Test currently verifies send failure behavior
  2. **GREEN**: Modify `src/ws_route.rs` — replace all 4 `let _ = sender.send(...)` with proper error logging:
     - Line ~100 (broadcast new_metrics): replace `let _ = sender.send(...)` with:
       ```rust
       if let Err(e) = sender.send(msg).await {
           tracing::error!(error = %e, "Failed to send broadcast message to client");
       }
       ```
     - Line ~107 (subscribe_run response): same pattern
     - Line ~113 (ping response): same pattern
     - Line ~120 (subscribe_run with lite mode): same pattern
  3. Run `cargo test test_ws_send_error_logged_on_disconnect` → PASS
  4. Run `cargo test --test test_websocket` → all pass

  **Must NOT do**:
  - Don't break the connection on send error — the outer loop already handles this
  - Don't change send timing or ordering

  **Recommended Agent Profile**:
  > `quick` — 4 identical replacements with clear pattern.
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6-T10, T12 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/ws_route.rs:96-122` — all 4 `let _ = sender.send(...)` locations
  - `src/ws_route.rs:100` — broadcast new_metrics send
  - `src/ws_route.rs:107` — subscribe_run response send
  - `src/ws_route.rs:113` — ping response send
  - `src/ws_route.rs:120` — subscribe_run lite mode response send
  - `src/ws.rs:327-450` — inline test module (add new test here)

  **Acceptance Criteria** (TDD):
  - [ ] RED: `cargo test test_ws_send_error_logged_on_disconnect` → FAIL
  - [ ] GREEN: Same test → PASS (send error detected)
  - [ ] `grep -c "let _ = sender.send" src/ws_route.rs` returns 0
  - [ ] `grep -c "tracing::error!" src/ws_route.rs` returns ≥ 4
  - [ ] `cargo test --test test_websocket` — all 13 tests pass

  **QA Scenarios**:
  ```
  Scenario: No silent send failures
    Tool: Bash
    Steps:
      1. Run `grep -c "let _ = sender.send" src/ws_route.rs`
    Expected Result: "0"
    Evidence: .sisyphus/evidence/task-11-no-silent.txt

  Scenario: All WS tests pass after logging change
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test --test test_websocket 2>&1 | grep "test result:"`
    Expected Result: "test result: ok" with 0 failures
    Evidence: .sisyphus/evidence/task-11-regression.txt
  ```

  **Commit**: YES
  - Message: `fix(ws): log send errors instead of silently swallowing`
  - Files: `src/ws_route.rs`, `src/ws.rs` (test only)

- [x] 12. **Add Ingest Pipeline Tests** ⬡ TDD

  **What to do** (RED → GREEN → REFACTOR):
  1. **RED**: Write 3 inline tests in `src/ingest.rs` `#[cfg(test)]` module (create test module):
     - `test_queue_full_returns_full_error`: Create channel with capacity=1. Fill it (don't consume). Try to send again → assert `TrySendError::Full`. Verify the full scenario returns the correct error type.
     - `test_queue_closed_returns_closed_error`: Create channel, send one item, drop the receiver, try to send again → assert `TrySendError::Closed`. Verify the closed scenario returns the correct error type.
     - `test_ingest_stats_counters`: Create IngestStats, call `mark_accepted()` 3 times, `mark_dropped()` once. Assert `accepted_count()=3`, `dropped_count()=1`, `processed_count()=0`. Then `mark_processed()` 3 times, assert `processed_count()=3`. Verify `wait_for_accepted_items()` returns when `processed >= accepted`.
     These tests FAIL initially (no test module exists).
  2. **GREEN**: The tests exercise existing code — the ingest.rs functions are already implemented. Just add the `#[cfg(test)] mod tests { ... }` module with imports and test functions. The tests should pass immediately since they test existing behavior.
  3. **REFACTOR**: Extract test helper `make_test_channel()` if multiple tests need channel setup.
  4. Run `cargo test test_queue_full test_queue_closed test_ingest_stats` → all PASS

  **Must NOT do**:
  - Don't modify production code in ingest.rs — tests only exercise existing behavior
  - Don't add integration tests in tests/ directory — these are unit tests for the ingest module
  - Don't create a full server for these tests

  **Recommended Agent Profile**:
  > `unspecified-high` — requires understanding the mpsc channel + IngestStats interaction. Moderate complexity but well-bounded scope.
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6-T11 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/ingest.rs:1-96` — full module: IngestItem, IngestStats, spawn_worker, channel()
  - `src/ingest.rs:14-18` — `IngestStats` with AtomicU64 counters
  - `src/ingest.rs:29-37` — `mark_accepted()`, `mark_dropped()`, `mark_processed()`
  - `src/ingest.rs:56-63` — `wait_for_accepted_items()` using Notify
  - `src/ingest.rs:72-75` — `channel()` creating bounded mpsc
  - `src/ingest.rs:83-95` — `spawn_worker()` tokio task
  - `src/store.rs:551-720` — inline test module pattern to follow
  - `src/persist.rs:350-394` — another inline test module pattern

  **Acceptance Criteria** (TDD):
  - [ ] RED: `cargo test test_queue_full_returns_full_error` → FAIL (no test exists)
  - [ ] GREEN: Same test → PASS (queue full detected)
  - [ ] `cargo test test_queue_closed_returns_closed_error` → PASS
  - [ ] `cargo test test_ingest_stats_counters` → PASS
  - [ ] `cargo test` — all tests pass

  **QA Scenarios**:
  ```
  Scenario: Queue full behavior
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_queue_full_returns_full_error`
    Expected Result: Channel with capacity=1 correctly returns TrySendError::Full on second send
    Evidence: .sisyphus/evidence/task-12-full.txt

  Scenario: Ingest stats accuracy
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_ingest_stats_counters`
    Expected Result: accepted=3, dropped=1, processed=3 after all operations
    Evidence: .sisyphus/evidence/task-12-stats.txt

  Scenario: Wait for accepted items synchronization
    Tool: Bash (cargo test)
    Preconditions: test_ingest_stats_counters includes wait_for_accepted_items
    Steps:
      1. Run the test
    Expected Result: wait_for_accepted_items().await returns after process count catches up
    Evidence: .sisyphus/evidence/task-12-wait.txt
  ```

  **Commit**: YES
  - Message: `test(ingest): add TDD tests for queue-full, queue-closed, stats counters`
  - Files: `src/ingest.rs`

- [x] 13. **Verify & Update Test Infrastructure for Middleware Compatibility**

  **What to do**:
  1. Verify `tests/common/mod.rs:build_test_app()` already has `DefaultBodyLimit` and `CorsLayer` (added by T6/T7 when implementing middleware). If not, add them now matching main.rs.
  2. Verify `tests/test_websocket.rs` now uses `common::build_test_app()` instead of its own duplicated function (refactored by T2). Confirm all test imports updated.
  3. Verify `AppState` constructors in both test builders no longer pass `config` field (removed by T1).
  4. Run the complete test suite: `cargo test 2>&1` → all tests pass
  5. If any test fails due to middleware interactions (e.g., body limit rejecting tests that send intentionally malformed large payloads), adjust test payload sizes to fit within the limit.
  6. Specifically verify: WS tests still work with CorsLayer on router (CORS doesn't affect WebSocket upgrades).

  **Must NOT do**:
  - Don't disable middleware for tests — tests should exercise the same code paths as production
  - Don't modify test logic (only adjust payload sizes if needed)

  **Recommended Agent Profile**:
  > `quick` — verification and minor fixup of work already done by earlier tasks.
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T2, T6, T7 completion)
  - **Parallel Group**: Wave 3 (sequential, before T14)
  - **Blocks**: T14
  - **Blocked By**: T2, T6, T7

  **References**:
  - `tests/common/mod.rs:9-68` — `build_test_app()` (should have middleware by now)
  - `tests/test_websocket.rs:18-69` — WS test builder (should use common by now)
  - `src/main.rs:78-96` — reference for correct middleware chain
  - `src/routes/mod.rs:14-24` — AppState (no more `config` field)

  **Acceptance Criteria**:
  - [ ] `cargo test 2>&1 | grep "test result: ok"` — ALL test suites pass
  - [ ] `cargo test 2>&1 | grep "FAILED" | wc -l` → 0
  - [ ] Both test builders include `DefaultBodyLimit` and `CorsLayer`
  - [ ] No `build_ws_test_app` function exists (using common::build_test_app)

  **QA Scenarios**:
  ```
  Scenario: Full test suite passes
    Tool: Bash
    Steps:
      1. Run `cargo test 2>&1`
      2. Count failures: `cargo test 2>&1 | grep -c "FAILED"`
    Expected Result: "0" failures, all test result lines show "ok"
    Failure Indicators: Any "FAILED" or "test result: FAILED"
    Evidence: .sisyphus/evidence/task-13-full-suite.txt

  Scenario: WebSocket tests work with CORS middleware
    Tool: Bash
    Steps:
      1. Run `cargo test --test test_websocket 2>&1 | grep "test result:"`
    Expected Result: "test result: ok" with 0 failures
    Evidence: .sisyphus/evidence/task-13-ws-cors.txt
  ```

  **Commit**: NO (this is verification — no new changes unless fixup needed)

- [x] 14. **Final Integration Verification**

  **What to do**:
  1. Run the complete hardening verification suite:
     ```bash
     cargo clean && cargo check 2>&1 | grep -c "warning:"  # Expected: 0
     cargo test 2>&1 | grep "test result:"                  # Expected: all "ok"
     cargo build --release                                  # Expected: success
     grep -c "println!" src/main.rs                         # Expected: 0
     grep -rn "fn now_iso" src/                             # Expected: 1 match in models.rs
     grep -ci "python\|main.py\|pip install\|pydantic" Rust/README.md  # Expected: 0
     ```
  2. Quick smoke test: start server with `timeout 3 cargo run 2>&1`, verify startup banner shows via tracing, verify health endpoint responds
  3. Check `cargo check` warnings one final time — target is ZERO

  **Must NOT do**:
  - No new code changes (only verification)
  - Don't start the server in a way that leaves orphan processes

  **Recommended Agent Profile**:
  > `unspecified-high` — comprehensive verification across multiple dimensions. Must interpret test output and flag any anomalies.
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T13, blocks FINAL)
  - **Parallel Group**: Wave 3 (final pre-review task)
  - **Blocks**: F1-F4
  - **Blocked By**: T13

  **Acceptance Criteria**:
  - [ ] `cargo check` — ZERO warnings
  - [ ] `cargo test` — ALL test suites pass, ZERO failures
  - [ ] `cargo build --release` — success
  - [ ] `grep -c "println!" src/main.rs` → 0
  - [ ] `grep -rn "fn now_iso" src/` → 1 match
  - [ ] `grep -ci "python\|main.py\|pip install\|pydantic" Rust/README.md` → 0

  **QA Scenarios**:
  ```
  Scenario: Zero warnings on cargo check
    Tool: Bash
    Steps:
      1. Run `cargo check 2>&1`
      2. Count warnings: `cargo check 2>&1 | grep -c "warning:"`
    Expected Result: "0"
    Evidence: .sisyphus/evidence/task-14-zero-warnings.txt

  Scenario: Full test suite passes
    Tool: Bash
    Steps:
      1. Run `cargo test 2>&1 | grep -E "test result:|FAILED"`
    Expected Result: All "test result: ok", zero "FAILED" lines
    Evidence: .sisyphus/evidence/task-14-full-tests.txt

  Scenario: Release build succeeds
    Tool: Bash
    Steps:
      1. Run `cargo build --release 2>&1 | tail -5`
    Expected Result: "Finished release [optimized]" with no errors
    Evidence: .sisyphus/evidence/task-14-release.txt

  Scenario: All hardening targets verified
    Tool: Bash
    Steps:
      1. Run verification commands: println check, now_iso check, README check
    Expected Result: All return expected values (0, 1 match, 0 Python refs)
    Evidence: .sisyphus/evidence/task-14-targets.txt
  ```

  **Commit**: NO (verification only — if fixup needed, commit separately)

---
## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
>
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback → fix → re-run → present again → wait for okay.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, grep, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in `.sisyphus/evidence/`. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `cargo check` (zero warnings), `cargo test` (all pass), `cargo build --release`. Review all changed files for: `unwrap()`/`expect()` in non-test code, empty catches, console.log patterns, AI slop. Check `cargo check 2>&1 | grep -c "warning:"` returns 0. Verify no new dependencies in Cargo.toml (feature flags only).
  Output: `Build [PASS/FAIL] | Check [N warnings] | Tests [N pass/N fail] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill if UI)
  Start from clean state. Test body limit: POST oversized payload → 413. Test CORS: curl with Origin header → verify Access-Control-Allow-Origin. Test WS heartbeat: connect WS, wait 35s, verify connection alive. Test ingest backpressure: create tiny queue, flood POSTs, verify 503. Test persist buffer: post max_steps_per_run+10 steps, verify cap. Save evidence to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **T1**: `chore: cleanup dead code warnings` — models.rs, store.rs, ws.rs, routes/mod.rs
- **T2**: `chore(deps): add cors feature to tower-http` — Cargo.toml
- **T3**: `refactor: deduplicate now_iso() into shared location` — store.rs, persist.rs
- **T4**: `refactor: replace println!() with tracing::info!()` — main.rs
- **T5**: `docs: update README.md to reflect Rust implementation` — README.md
- **T6**: `feat(security): apply DefaultBodyLimit middleware` — main.rs, tests/common/mod.rs
- **T7**: `feat(security): apply CorsLayer middleware` — main.rs, tests/common/mod.rs
- **T8**: `feat(ws): add protocol-level heartbeat (30s ping, 10s pong timeout)` — ws_route.rs
- **T9**: `fix(persist): add buffer size cap at max_steps_per_run` — persist.rs
- **T10**: `fix(store): log persist errors instead of silently swallowing` — store.rs
- **T11**: `fix(ws): log send errors instead of silently swallowing` — ws_route.rs
- **T12**: `test(ingest): add TDD tests for queue-full, queue-closed, stats` — ingest.rs
- **T13**: `test: update test infrastructure for middleware compatibility` — tests/common/mod.rs, tests/test_websocket.rs
- **T14**: `test: final integration verification` — (no new files)

---

## Success Criteria

### Verification Commands
```bash
cargo check 2>&1 | grep -c "warning:"     # Expected: 0
cargo test 2>&1 | grep "test result:"      # Expected: all "ok" with 0 failures
cargo build --release                      # Expected: success
grep -c "Python\|main.py\|pip install\|pydantic" Rust/README.md  # Expected: 0
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All 167+ tests pass (existing + new)
- [ ] Zero `cargo check` warnings
- [ ] 413 on oversized POST body
- [ ] CORS headers on cross-origin requests
- [ ] WS heartbeat sends Ping after 30s idle
- [ ] Ingest queue-full returns 503
- [ ] Persist buffer capped at max_steps_per_run
- [ ] README.md free of Python references
