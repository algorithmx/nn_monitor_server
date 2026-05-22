# JSONL Persistence for Training History

## TL;DR

> **Quick Summary**: Add a JSONL-based persistence layer to the training monitor server. Each experiment (run) gets one `.jsonl` file. Steps are buffered in memory and flushed to disk when the experiment ends (detected by POST gap). On startup, metadata is scanned from files; full history loads lazily when a user selects a run.
>
> **Deliverables**:
> - New `src/persist.rs` module with `JsonlStore` (write, read, scan metadata, flush)
> - Config additions: `NN_MONITOR_DATA_DIR`, `NN_MONITOR_FLUSH_TIMEOUT_SECS`
> - `Deserialize` derives on `StepData` and `RunData` in models.rs
> - Integration with `MetricsStore` for flush triggers and lazy loading
> - Graceful shutdown flush
> - Tests for roundtrip, metadata scan, lazy load, corruption, sanitization, gap detection
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 (models) → Task 2 (config) → Task 3 (persist core) → Task 4 (store integration) → Task 5 (main wiring) → Task 6 (tests) → F1-F4

---

## Context

### Original Request
User asked to add history persistence using JSONL format, one file per experiment.

### Interview Summary
**Key Discussions**:
- Storage path: Configurable via `NN_MONITOR_DATA_DIR` (default: `./data`)
- Startup: Lazy load — scan metadata only, load history on demand
- Eviction: Keep JSONL files on disk when runs are evicted from memory
- Write strategy: Buffered — flush when experiment ends (POST gap detection)
- Experiment end detection: Configurable timeout (default 300s)
- Tests: Write after implementation

**Research Findings**:
- `StepData` and `RunData` have `Serialize` but NOT `Deserialize` — must add
- `run_id` is user input with no filesystem safety validation — must sanitize
- `tokio = { features = ["full"] }` already includes `tokio::fs` — no new deps needed
- Ingest worker is single-tasked (one `spawn_worker`) — single-writer guarantee

### Metis Review
**Identified Gaps** (addressed):
- Path traversal via run_id: Sanitize filenames (replace special chars)
- Missing `Deserialize` on StepData/RunData: Add derives
- Graceful shutdown flush: Wire into shutdown_signal
- max_steps_per_run vs persistence: Lazy load caps at max_steps_per_run
- Gap detection timeout: Configurable, default 300s
- Corrupt JSONL: Skip malformed lines with tracing::warn
- Concurrent lazy load: Guard with loading set
- Flush in background: Don't block ingest pipeline

---

## Work Objectives

### Core Objective
Persist training metrics to JSONL files on disk, one file per run, with lazy loading on server restart.

### Concrete Deliverables
- `src/persist.rs` — JsonlStore with write, read, scan_metadata, flush operations
- Modified `src/config.rs` — data_dir, flush_timeout_secs fields
- Modified `src/models.rs` — Deserialize on StepData, RunData
- Modified `src/store.rs` — flush trigger hooks, lazy load integration
- Modified `src/main.rs` — initialize persist, shutdown flush
- `src/tests/persist_tests.rs` or inline tests — comprehensive test coverage

### Definition of Done
- [ ] `cargo build` succeeds with zero errors
- [ ] `cargo test` passes (all existing + new tests)
- [ ] POST metrics → file appears in `./data/{run_id}.jsonl`
- [ ] Kill + restart server → run list populated from disk, selecting a run loads full history
- [ ] Ctrl+C gracefully flushes buffered data

### Must Have
- One JSONL file per run in configurable data directory
- Each line is a valid `StepData` JSON object
- Buffer steps in memory, flush on experiment-end detection (gap timeout)
- Lazy load: scan metadata on startup, load full history on demand
- Graceful shutdown flush of all in-memory buffers
- Sanitize run_id for filesystem safety
- Skip malformed JSONL lines during load
- `max_steps_per_run` cap applied to lazy-loaded data (load last N)
- Concurrent lazy load dedup (don't load same run twice)
- All existing tests continue to pass unchanged

### Must NOT Have (Guardrails)
- No new crate dependencies (use tokio::fs, serde_json, std::io)
- No changes to REST API contract (endpoints, status codes, response shapes)
- No changes to WebSocket message format
- No compression, retention policies, indexing, or migration tools
- No new API endpoints for persistence management
- No changes to ingest pipeline architecture (mpsc channel + worker stays)
- No file I/O in the ingest worker hot path (flush in background task)

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (cargo test with tokio::test)
- **Automated tests**: YES (tests after)
- **Framework**: cargo test + tokio::test
- **Test files**: Inline `#[cfg(test)] mod tests` in persist.rs + store.rs additions

### QA Policy
Every task includes agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Library/Module**: Use Bash (`cargo test`) — Run tests, assert pass/fail
- **Integration**: Use Bash (`cargo build` + `curl`) — Build, start server, test end-to-end

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — T1 ∥ T2, no dependencies):
├── Task 1: Add Deserialize to StepData/RunData + sanitize run_id [quick]
└── Task 2: Add persistence config fields [quick]

Wave 2 (Core persistence — T3 after T1+T2, then T4 after T3):
├── Task 3: Create persist.rs — JsonlStore core (write, read, scan) [deep]
└── Task 4: Integrate persist into MetricsStore (flush trigger + lazy load) [deep]

Wave 3 (Wiring + tests — T5 after T3+T4, then T6 after T5):
├── Task 5: Wire persist into main.rs (init, config, shutdown flush) [unspecified-high]
└── Task 6: Write all persistence tests [unspecified-high]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: T1 → T3 → T4 → T5 → T6 → F1-F4 (T2 overlaps with T1 only)
Parallel Speedup: ~15% (only T1 ∥ T2 in Wave 1; rest is sequential chain)
Max Concurrent: 2 (Wave 1) / 4 (FINAL only)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | - | 3, 4 | 1 |
| 2 | - | 3, 4, 5 | 1 |
| 3 | 1, 2 | 4, 5 | 2 |
| 4 | 1, 3 | 5, 6 | 2 |
| 5 | 2, 3, 4 | 6 | 3 |
| 6 | 4, 5 | F1-F4 | 3 |
| F1-F4 | ALL | user okay | FINAL |

### Agent Dispatch Summary

- **Wave 1**: 2 tasks — T1 → `quick`, T2 → `quick`
- **Wave 2**: 2 tasks — T3 → `deep`, T4 → `deep`
- **Wave 3**: 2 tasks — T5 → `unspecified-high`, T6 → `unspecified-high`
- **FINAL**: 4 tasks — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Add Deserialize to StepData/RunData + Sanitize Run ID

  **What to do**:
  - Add `Deserialize` to the `#[derive(...)]` on `StepData` (models.rs:222) and `RunData` (models.rs:232)
  - Create a `sanitize_filename(run_id: &str) -> String` function in models.rs that:
    - Replaces `/`, `\`, `..`, `\0`, and other filesystem-unsafe characters with `_`
    - Truncates to a reasonable max length (e.g., 255 chars)
  - Add `sanitize_filename` call in `MetricsPayload::validate()` (models.rs:164) OR use it only when constructing filenames in persist.rs (preferred — don't modify validate)
  - Add unit tests for `sanitize_filename` covering: path traversal (`../../etc/passwd`), slashes (`run/with/slashes`), null bytes, empty string, normal names, very long names

  **Must NOT do**:
  - Don't change any fields on StepData, RunData, or any other struct
  - Don't remove or modify existing derives
  - Don't change `MetricsPayload::validate()` — sanitization happens at persistence boundary

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 2)
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Tasks 3, 4
  - **Blocked By**: None

  **References**:
  - `src/models.rs:222-230` — `StepData` struct with `#[derive(Debug, Clone, Serialize)]`
  - `src/models.rs:232-237` — `RunData` struct with `#[derive(Debug, Clone, Serialize)]`
  - `src/models.rs:47-78` — `IntermediateFeatures`, `GradientFlow` etc. already have `Serialize, Deserialize`
  - `src/models.rs:140-148` — `LayerStatistic` already has `Serialize, Deserialize`

  **Acceptance Criteria**:
  - [ ] `cargo test` — existing tests still pass
  - [ ] New test `test_sanitize_filename_rejects_path_traversal` passes
  - [ ] New test `test_sanitize_filename_replaces_slashes` passes
  - [ ] New test `stepdata_deserialize_roundtrip` passes

  **QA Scenarios**:
  ```
  Scenario: StepData JSON roundtrip
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_stepdata_deserialize_roundtrip`
    Expected Result: Test passes — StepData serializes to JSON and deserializes back to equal value
    Evidence: .sisyphus/evidence/task-1-roundtrip.txt

  Scenario: Path traversal sanitized
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_sanitize_filename`
    Expected Result: All sanitize tests pass — no `../`, `/`, `\` in output
    Evidence: .sisyphus/evidence/task-1-sanitize.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add Deserialize derives to StepData/RunData + filename sanitizer`
  - Files: `src/models.rs`

- [x] 2. Add Persistence Config Fields

  **What to do**:
  - Add to `ServerConfig` in config.rs:
    - `data_dir: String` with default `"./data"` and env `NN_MONITOR_DATA_DIR`
    - `flush_timeout_secs: u64` with default `300` and env `NN_MONITOR_FLUSH_TIMEOUT_SECS`
  - Add corresponding `default_*()` functions and unit tests following existing pattern (config.rs:3-29)
  - Add `data_dir` to the startup banner output in main.rs (println block)

  **Must NOT do**:
  - Don't remove or modify existing config fields
  - Don't add new dependencies

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 1)
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Tasks 3, 4, 5
  - **Blocked By**: None

  **References**:
  - `src/config.rs:3-29` — Pattern for default_* functions
  - `src/config.rs:35-53` — `ServerConfig` struct with `#[serde(default)]` annotations
  - `src/config.rs:55-60` — `ServerConfig::load()` using envy
  - `src/main.rs:90-104` — Startup banner where data_dir should be printed

  **Acceptance Criteria**:
  - [ ] `cargo test test_default_data_dir` passes
  - [ ] `cargo test test_default_flush_timeout_secs` passes
  - [ ] Server startup banner shows data directory path

  **QA Scenarios**:
  ```
  Scenario: Config defaults applied
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_default_data_dir test_default_flush_timeout_secs`
    Expected Result: Both tests pass, data_dir="./data", flush_timeout_secs=300
    Evidence: .sisyphus/evidence/task-2-config.txt
  ```

  **Commit**: YES
  - Message: `feat(config): add data_dir and flush_timeout_secs persistence config`
  - Files: `src/config.rs`, `src/main.rs`

---

- [x] 3. Create persist.rs — JsonlStore Core (Write, Read, Scan)

  **What to do**:
  - Create new file `src/persist.rs` with `pub struct JsonlStore`
  - `JsonlStore` holds: `data_dir: PathBuf`, a buffered writer map (run_id → pending steps), flush timeout
  - Implement:
    - `new(data_dir: PathBuf, flush_timeout: Duration) -> Self` — creates data dir if missing
    - `buffer_step(run_id: &str, step: &StepData)` — adds step to in-memory buffer for this run
    - `flush_run(run_id: &str) -> Result<(), PersistError>` — writes buffered steps to `{sanitized_run_id}.jsonl` as append
    - `scan_metadata() -> Vec<(String, RunMeta)>` — reads data dir, for each .jsonl file extracts: run_id (from filename), created_at (first line timestamp), last_update (last line timestamp), step_count (line count), latest_step (last line step number). Only reads first and last lines per file for efficiency.
    - `load_run(run_id: &str, max_steps: usize) -> Result<Option<RunData>, PersistError>` — reads full JSONL file, parses each line as `StepData`, applies max_steps cap (keep last N), builds `RunData`
    - `flush_all()` — flushes all buffered runs (for shutdown)
  - Use `tokio::fs` for all file operations
  - Use `tokio::sync::Mutex` for buffer protection
  - Skip malformed JSONL lines during load with `tracing::warn!`
  - Start a background flush task that checks for stale runs periodically
  - Use `sanitize_filename` from models.rs for all file operations
  - Define `RunMeta` struct: `{ run_id: String, created_at: String, last_update: String, step_count: u32, latest_step: Option<u64> }`
  - Define `PersistError` enum: `Io(io::Error), Parse(serde_json::Error), ...`

  **Must NOT do**:
  - Don't block the ingest worker on file I/O
  - Don't use `std::fs` — use `tokio::fs` for async
  - Don't add new dependencies
  - Don't modify store.rs yet (that's Task 4)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 1, 2)
  - **Parallel Group**: Wave 2 (sequential after Wave 1)
  - **Blocks**: Tasks 4, 5
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `src/models.rs:222-230` — `StepData` struct (now with Deserialize from Task 1)
  - `src/models.rs:232-237` — `RunData` struct (now with Deserialize from Task 1)
  - `src/models.rs:150-154` — `CrossLayerAnalysis` with `Serialize, Deserialize`
  - `src/store.rs:27-31` — Pattern: struct with internal state + async methods
  - `src/store.rs:43-92` — Pattern: `StoreState` internal struct with caches
  - `src/ingest.rs:83-95` — `spawn_worker` pattern for background tasks
  - `Cargo.toml:8` — `tokio = { version = "1", features = ["full"] }` — includes tokio::fs

  **Acceptance Criteria**:
  - [ ] `cargo build` — no errors
  - [ ] `JsonlStore::new()` creates data directory
  - [ ] `buffer_step` + `flush_run` writes valid JSONL to disk
  - [ ] `scan_metadata` extracts correct metadata from JSONL files
  - [ ] `load_run` parses all lines, skips malformed, caps at max_steps
  - [ ] `flush_all` writes all buffered data

  **QA Scenarios**:
  ```
  Scenario: Write and read roundtrip
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_jsonl_write_and_read_roundtrip`
    Expected Result: 3 steps written, 3 steps read back with identical data
    Evidence: .sisyphus/evidence/task-3-roundtrip.txt

  Scenario: Malformed line handling
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_jsonl_skip_malformed_lines`
    Expected Result: File with 1 good + 1 corrupt + 1 good line loads 2 valid steps
    Evidence: .sisyphus/evidence/task-3-corrupt.txt

  Scenario: Metadata scan efficiency
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_jsonl_scan_metadata`
    Expected Result: Metadata extracted from first/last lines without full file read
    Evidence: .sisyphus/evidence/task-3-metadata.txt
  ```

  **Commit**: YES
  - Message: `feat(persist): add JsonlStore with write, read, and metadata scan`
  - Files: `src/persist.rs` (new file)

- [x] 4. Integrate Persist into MetricsStore (Flush Trigger + Lazy Load)

  **What to do**:
  - Add `Arc<JsonlStore>` field to `MetricsStore`
  - In `insert_step_data` (store.rs:179): after inserting the step, call `persist.buffer_step(run_id, &step_data)` to buffer it
  - Add a flush hook: when the background flush task detects a stale run (no new steps for `flush_timeout_secs`), call `persist.flush_run(run_id)`
  - Modify `get_run` / `build_run_history_message`: if run not in memory AND JSONL file exists on disk, lazy load from disk
  - Add a `loading: Arc<Mutex<HashSet<String>>>` guard to prevent concurrent lazy loads of the same run
  - Modify `build_initial_runs_message` / startup: include runs from `scan_metadata()` that aren't already in memory
  - Apply `max_steps_per_run` cap to lazy-loaded data (load last N steps from JSONL)
  - Ensure evicted runs' files remain on disk (don't delete on eviction)

  **Must NOT do**:
  - Don't change the ingest worker (ingest.rs)
  - Don't change API response formats
  - Don't block on file I/O in the hot path
  - Don't delete JSONL files on run eviction

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 1, 3)
  - **Parallel Group**: Wave 2 (after Task 3)
  - **Blocks**: Tasks 5, 6
  - **Blocked By**: Tasks 1, 3

  **References**:
  - `src/store.rs:143-227` — `MetricsStore` methods: `add_validated_metrics_and_message`, `insert_step_data`, `get_run`
  - `src/store.rs:299-343` — `build_run_history_message` with lazy cache pattern (read lock → miss → write lock → build → cache)
  - `src/store.rs:184-194` — Run eviction logic (oldest by last_update)
  - `src/store.rs:216-219` — Step eviction logic (drain oldest when over max_steps_per_run)
  - `src/store.rs:43-92` — `StoreState` internal struct with caches

  **Acceptance Criteria**:
  - [ ] `insert_step_data` buffers steps to persist layer
  - [ ] `get_run` lazy-loads from JSONL when run not in memory
  - [ ] `build_initial_runs_message` includes runs from disk
  - [ ] Concurrent lazy loads don't duplicate disk reads
  - [ ] Run eviction doesn't delete JSONL files

  **QA Scenarios**:
  ```
  Scenario: Lazy load from disk
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_lazy_load_from_jsonl`
    Expected Result: Run written to JSONL, evicted from memory, get_run loads from disk with correct data
    Evidence: .sisyphus/evidence/task-4-lazyload.txt

  Scenario: Startup includes disk runs
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_startup_includes_disk_runs`
    Expected Result: build_initial_runs_message includes runs from JSONL files on disk
    Evidence: .sisyphus/evidence/task-4-startup.txt
  ```

  **Commit**: YES
  - Message: `feat(store): integrate persistence with flush triggers and lazy load`
  - Files: `src/store.rs`

---

- [x] 5. Wire Persist into main.rs (Init, Config, Shutdown Flush)

  **What to do**:
  - In `main()`: create `JsonlStore` from config values (data_dir, flush_timeout_secs)
  - Pass `JsonlStore` Arc to `MetricsStore::new()` or add a builder method
  - Start the background flush task (from persist.rs) — store its `JoinHandle`
  - Modify `shutdown_signal()`: before returning, call `persist.flush_all()` to flush all buffered data
  - Add `mod persist;` declaration
  - Update the startup banner to show data_dir and flush timeout

  **Must NOT do**:
  - Don't change the route definitions or AppState structure unnecessarily
  - Don't add persistence to AppState if it's not needed by routes (keep it internal to MetricsStore)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 2, 3, 4)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 6
  - **Blocked By**: Tasks 2, 3, 4

  **References**:
  - `src/main.rs:1-7` — Module declarations
  - `src/main.rs:34-113` — `main()` function: config load, store creation, app setup, server start
  - `src/main.rs:27-32` — `shutdown_signal()` function
  - `src/main.rs:44-55` — Store + ws_manager + ingest creation pattern

  **Acceptance Criteria**:
  - [ ] Server starts with `--data-dir` or `NN_MONITOR_DATA_DIR` config
  - [ ] `data/` directory created on startup if missing
  - [ ] Ctrl+C flushes all buffered data before exit
  - [ ] Startup banner shows persistence config

  **QA Scenarios**:
  ```
  Scenario: Server starts with persistence enabled
    Tool: Bash
    Steps:
      1. Run `cargo build`
      2. Start server with `NN_MONITOR_DATA_DIR=/tmp/test_data cargo run`
      3. Check stdout for "Data directory: /tmp/test_data"
      4. `curl -s http://localhost:8000/health`
    Expected Result: Server healthy, data directory created
    Evidence: .sisyphus/evidence/task-5-startup.txt

  Scenario: Graceful shutdown flush
    Tool: Bash
    Steps:
      1. Start server
      2. Post 3 metrics via curl
      3. Send SIGINT (Ctrl+C)
      4. Check JSONL file exists and has 3 lines
    Expected Result: All 3 steps persisted before shutdown
    Evidence: .sisyphus/evidence/task-5-shutdown.txt
  ```

  **Commit**: YES
  - Message: `feat(main): wire persistence layer into server lifecycle`
  - Files: `src/main.rs`

- [x] 6. Write All Persistence Tests

  **What to do**:
  - Add comprehensive tests in `persist.rs` (`#[cfg(test)] mod tests`) and `store.rs` additions:
    1. `test_jsonl_write_and_read_roundtrip` — write 3 steps, read back, assert identical
    2. `test_jsonl_scan_metadata` — create file, scan metadata, assert correct fields
    3. `test_lazy_load_from_jsonl` — write to JSONL, evict from memory, load, assert data
    4. `test_jsonl_skip_malformed_lines` — write 1 good + 1 corrupt + 1 good, load, assert 2 valid
    5. `test_run_id_sanitization_for_filename` — assert path traversal blocked
    6. `test_flush_on_timeout` — buffer steps, wait for timeout, assert file written
    7. `test_concurrent_lazy_load_dedup` — two tasks request same evicted run, one disk read
    8. `test_max_steps_cap_on_lazy_load` — JSONL with 20 steps, max_steps=10, assert last 10 loaded
    9. `test_flush_all_on_shutdown` — buffer 3 runs, flush_all, assert all files written
    10. `test_empty_jsonl_file_handling` — empty file doesn't crash scan/load
  - Use `tempfile` pattern: create temp dir for each test, clean up after

  **Must NOT do**:
  - Don't break existing tests
  - Don't add new test dependencies (use std::env::temp_dir or manual temp dirs)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 4, 5)
  - **Parallel Group**: Wave 3 (after Task 5)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 4, 5

  **References**:
  - `src/store.rs:384-600` — Existing test pattern with `make_payload` helper
  - `src/models.rs:258-728` — Existing test patterns

  **Acceptance Criteria**:
  - [ ] All 10 new test functions pass
  - [ ] All existing tests still pass
  - [ ] `cargo test` shows 0 failures

  **QA Scenarios**:
  ```
  Scenario: Full test suite passes
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test 2>&1`
    Expected Result: All tests pass (existing + new), 0 failures
    Evidence: .sisyphus/evidence/task-6-tests.txt

  Scenario: Test coverage for edge cases
    Tool: Bash (cargo test)
    Steps:
      1. Run `cargo test test_jsonl_skip_malformed test_empty_jsonl test_run_id_sanitization test_max_steps_cap`
    Expected Result: All edge case tests pass
    Evidence: .sisyphus/evidence/task-6-edge.txt
  ```

  **Commit**: YES
  - Message: `test(persist): add comprehensive persistence tests`
  - Files: `src/persist.rs`, `src/store.rs`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `cargo build` + `cargo test` + `cargo clippy`. Review all changed files for: `as any`/`#[allow(unused)]`, empty catches, `unwrap()` in non-test code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names. Verify no new dependencies added to Cargo.toml.
  Output: `Build [PASS/FAIL] | Clippy [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start server from clean state. Post 3 metrics via curl. Verify JSONL file created in ./data/. Kill server. Restart server. Verify run appears in /api/v1/runs. Select run via WebSocket. Verify history loaded. Test edge cases: empty data dir, corrupt JSONL file, run_id with special chars. Save evidence.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Task 1**: `feat(models): add Deserialize derives to StepData and RunData` — models.rs
- **Task 2**: `feat(config): add data_dir and flush_timeout_secs persistence config` — config.rs
- **Task 3**: `feat(persist): add JsonlStore with write, read, and metadata scan` — persist.rs (new file)
- **Task 4**: `feat(store): integrate persistence with flush triggers and lazy load` — store.rs
- **Task 5**: `feat(main): wire persistence layer into server lifecycle` — main.rs
- **Task 6**: `test(persist): add comprehensive persistence tests` — persist.rs, store.rs

---

## Success Criteria

### Verification Commands
```bash
cargo build                            # Expected: success, 0 errors
cargo test                             # Expected: all pass, 0 failures
cargo clippy                           # Expected: no new warnings
ls ./data/*.jsonl                      # Expected: files present after training
curl -s http://localhost:8000/api/v1/runs  # Expected: runs loaded from disk
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass (existing + new)
- [ ] JSONL files created in data directory during training
- [ ] Server restart recovers run history from disk
- [ ] Graceful shutdown flushes buffered data
