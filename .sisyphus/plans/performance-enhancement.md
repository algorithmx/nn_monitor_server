# Performance Enhancement — nn_monitor_server (Rust)

## TL;DR

> **Quick Summary**: Implement Phases 1+2 of PERFORMANCE.md — 10 performance optimizations that eliminate double JSON parsing, reduce allocations, and improve data structure efficiency. The P0 change (build from typed structs) is the largest refactor and naturally subsumes the lock optimization.
> 
> **Deliverables**:
> - Single-parse POST handler (eliminate raw.clone() + double deserialization)
> - build_step_data rewritten to construct Value from typed structs
> - hashbrown::HashMap for internal storage
> - binary_search for step dedup (O(n) → O(log n))
> - Inner Arc removal from MetricsStore
> - Cow<str> for sanitize_layer_id
> - Reduced now_iso() allocations
> - Fixed broadcast signature
> - Dead dependency cleanup + release profile
> 
> **Estimated Effort**: Medium (1 week)
> **Parallel Execution**: YES — 3 waves + 1 verification wave
> **Critical Path**: Task 1 → Task 3 → Task 4 → Final Verification

---

## Context

### Original Request
Plan performance enhancements based on findings in `Rust/PERFORMANCE.md`. Emphasis on JSON schema conformity and functional equivalence after fixes.

### Interview Summary
**Key Discussions**:
- Schema conformity: User accepted dropping extra fields beyond schema — server defines the schema, clients should only send schema-defined fields
- Scope: Phases 1+2 only (items #1, #3, #4, #5, #6, #7, #8, #9, #10, #11 from PERFORMANCE.md)
- Test strategy: Tests-after with existing 244+ test suite (98 Rust + 146 Python)

**Research Findings**:
- Dual-path data design: POST handler parses bytes → raw Value, then raw.clone() → MetricsPayload. Both passed to store.
- build_step_data mixes typed fields (step, timestamp, batch_size from payload) with raw opaque fields (layers from raw JSON, cross_layer from raw JSON)
- 244+ tests provide comprehensive coverage of validation, storage, dedup, eviction, WebSocket, concurrency, error formats
- hashbrown::HashMap is a drop-in replacement with identical serde behavior
- Steps are kept sorted (binary_search is safe)
- MetricsStore is already behind Arc<MetricsStore> in AppState (inner Arc removal is safe)

### Metis Review
**Identified Gaps** (addressed):
- Blast radius: 38 call sites (not 27) — make_payload_and_raw helper rewrite handles 30 mechanically
- binary_search ownership: must clone step_data before match to avoid move-after-use
- #5 (lock hold time) is subsumed by P0 — build_step_data naturally moves before lock
- hashbrown scope: only store.rs, NOT models.rs (keep std::collections::HashMap in public API types)
- ws.rs broadcast tests need .to_string() for new String parameter

---

## Work Objectives

### Core Objective
Reduce per-request CPU by ~60% and memory allocations by ~70% through 10 targeted optimizations while preserving exact JSON schema output and functional equivalence.

### Concrete Deliverables
- `Cargo.toml`: hashbrown added, dead deps removed, release profile added
- `src/routes/metrics.rs`: Single-parse POST handler, broadcast call updated
- `src/store.rs`: Rewritten build_step_data, new add_metrics signature, hashbrown, binary_search, no inner Arc, Cow<str>, reduced now_iso
- `src/ws.rs`: broadcast(String) signature, test updates
- All 38 add_metrics call sites updated across store.rs unit tests + test_storage.rs

### Definition of Done
- [ ] `cargo test` passes with 0 failures (all 98+ Rust tests)
- [ ] `cargo clippy` passes with 0 warnings
- [ ] `cargo build --release` succeeds
- [ ] JSON output from GET endpoints matches pre-change schema exactly

### Must Have
- JSON schema output identical to current (same field names, same structure, same types)
- All 98+ Rust tests pass without modification to test assertions (only test helper signature changes)
- Error response formats unchanged (422 array format, 404 dict format)
- WebSocket message formats unchanged (new_metrics, run_history, initial_runs, pong, error)
- Layer ID sanitization (dots → slashes) preserved in both layers and layer_groups
- Step dedup behavior unchanged (replace by global_step, keep sorted ascending)
- Run/step eviction behavior unchanged (oldest by last_update / oldest step first)

### Must NOT Have (Guardrails)
- NO changes to `src/models.rs` struct definitions (StepData, RunData, RunInfo, MetricsPayload, etc.)
- NO changes to `src/routes/runs.rs` (unaffected by optimizations)
- NO changes to `src/routes/health.rs` (unaffected)
- NO changes to `src/routes/ws_route.rs` (unaffected — compact_layer/compact_step work by field name, which stays the same)
- NO new test files or benchmark infrastructure
- NO changes to HTTP-level integration test assertions (test_metrics_endpoint.rs, test_runs_endpoint.rs, test_edge_cases.rs, test_error_responses.rs, test_websocket.rs — these use HTTP interface, unaffected)
- NO changes to Python test suite (tests external HTTP API, unaffected)
- NO changes to error handling patterns (keep expect() where MetricsPayload.validate() guarantees safety)
- NO touching compact_layer/compact_step in ws.rs (field names unchanged, works as-is)
- NO adding sonic-rs, dashmap, or bumpalo (Phase 3 items, excluded)

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (cargo test, 98+ tests)
- **Automated tests**: Tests-after (implement, then verify with existing suite)
- **Framework**: cargo test (Rust built-in)
- **No TDD**: Existing 244+ tests provide comprehensive regression coverage

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Rust backend**: Use Bash (cargo test, cargo clippy, cargo build)
- **JSON output**: Use Bash (curl against running server) for schema regression
- **Build verification**: Use Bash (cargo build --release)

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — 2 independent tasks):
├── Task 1: Cargo.toml cleanup + release profile [quick]
└── Task 2: Fix broadcast signature [quick]

Wave 2 (After Wave 1 — THE critical refactoring):
└── Task 3: P0 Eliminate double parse + typed build_step_data + lock optimization [deep]
    (Also subsumes PERFORMANCE.md #5: move build_step_data outside lock)

Wave 3 (After Wave 2 — remaining store.rs optimizations):
└── Task 4: P1/P2 hashbrown + binary_search + Arc removal + Cow + now_iso [unspecified-high]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)
→ Present results → Get explicit user okay

Critical Path: Task 1 → Task 3 → Task 4 → F1-F4 → user okay
Parallel Speedup: Wave 1 (2 tasks) → Wave 2 (1 task) → Wave 3 (1 task) → Final (4 tasks)
Max Concurrent: 4 (Wave FINAL)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | — | 3 (needs hashbrown dep) | 1 |
| 2 | — | 3 (routes/metrics.rs shared) | 1 |
| 3 | 1, 2 | 4 (store.rs shared) | 2 |
| 4 | 3 | F1-F4 | 3 |
| F1-F4 | 4 | user okay | FINAL |

### Agent Dispatch Summary

- **Wave 1**: **2** — T1 → `quick`, T2 → `quick`
- **Wave 2**: **1** — T3 → `deep`
- **Wave 3**: **1** — T4 → `unspecified-high`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Cargo.toml Cleanup + Release Profile

  **What to do**:
  - Remove `validator` dependency (line with `validator = { version = "0.18", features = ["derive"] }`)
  - Remove `tokio-util` dependency (line with `tokio-util = { version = "0.7", features = ["rt"] }`)
  - Add `hashbrown = "0.15"` to `[dependencies]`
  - Add `[profile.release]` section at the end of the file:
    ```toml
    [profile.release]
    lto = "thin"
    codegen-units = 1
    opt-level = 3
    strip = true
    ```
  - Verify no other files import `validator` or `tokio-util` (grep confirms zero imports)

  **Must NOT do**:
  - Do NOT modify any .rs source files
  - Do NOT remove any other dependencies
  - Do NOT change dependency versions of remaining deps

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file edit, mechanical changes, no logic involved
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Task 3 (needs hashbrown in Cargo.toml)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `Rust/Cargo.toml` — Full file. Identify the validator and tokio-util lines to remove, the position to add hashbrown, and the end of file to append [profile.release]

  **API/Type References**:
  - `Rust/Cargo.toml` — Current dependency list. hashbrown 0.15 uses foldhash by default for 2-4× faster map ops vs SipHash

  **WHY Each Reference Matters**:
  - Cargo.toml: This is the ONLY file to modify. Read it first to identify exact lines.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Cargo.toml is valid after edits
    Tool: Bash (cargo check)
    Preconditions: Cargo.toml has been edited
    Steps:
      1. cd Rust && cargo check 2>&1
      2. Verify exit code 0
    Expected Result: cargo check succeeds with no errors
    Failure Indicators: "error: failed to select a version", "no matching package", exit code != 0
    Evidence: .sisyphus/evidence/task-1-cargo-check.txt

  Scenario: Dead dependencies are removed
    Tool: Bash (grep)
    Preconditions: Cargo.toml edited
    Steps:
      1. grep -c "validator" Rust/Cargo.toml — expect 0
      2. grep -c "tokio-util" Rust/Cargo.toml — expect 0
    Expected Result: Both grep commands return 0 matches
    Failure Indicators: Any match found
    Evidence: .sisyphus/evidence/task-1-dead-deps-removed.txt

  Scenario: hashbrown is added and release profile exists
    Tool: Bash (grep)
    Preconditions: Cargo.toml edited
    Steps:
      1. grep "hashbrown" Rust/Cargo.toml — expect match with version "0.15"
      2. grep -A4 "\[profile.release\]" Rust/Cargo.toml — expect lto, codegen-units, opt-level, strip
    Expected Result: hashbrown present with correct version, release profile has all 4 fields
    Failure Indicators: Missing hashbrown or incomplete release profile
    Evidence: .sisyphus/evidence/task-1-hashbrown-release.txt
  ```

  **Commit**: YES
  - Message: `perf(cargo): remove dead deps, add hashbrown, add release profile`
  - Files: `Rust/Cargo.toml`
  - Pre-commit: `cd Rust && cargo check`

- [x] 2. Fix Broadcast Signature

  **What to do**:
  - In `src/ws.rs`, change `broadcast(&self, message: &str)` to `broadcast(&self, message: String)` — remove the `.to_string()` inside the method body since the caller already owns the String
  - In `src/routes/metrics.rs`, change `state.ws_manager.broadcast(&msg)` to `state.ws_manager.broadcast(msg)` — pass ownership of the String instead of borrowing
  - In `src/ws.rs` unit tests (around line 240-243), update any test calls that pass `&str` literals to pass `String`:
    - `mgr.broadcast("hello")` → `mgr.broadcast("hello".to_string())`
    - Or any similar literal string broadcasts in ws.rs tests

  **Must NOT do**:
  - Do NOT change the broadcast channel capacity
  - Do NOT change any WebSocket message format
  - Do NOT modify compact_layer, compact_step, or any message builder functions

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Signature change in 2 functions + test updates, trivial mechanical change
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 3 (routes/metrics.rs shared)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `Rust/src/ws.rs:44-45` — Current `broadcast` method: `pub fn broadcast(&self, message: &str) { self.tx.send(message.to_string()); }`
  - `Rust/src/routes/metrics.rs:44-45` — Current call site: `let msg = build_new_metrics_message(&run_id, &step_data); state.ws_manager.broadcast(&msg);`
  - `Rust/src/ws.rs` — Unit tests for broadcast (search for `broadcast("` to find literal string test calls)

  **API/Type References**:
  - `tokio::sync::broadcast::Sender<String>::send(String)` — The send method already takes String by value. Current code converts &str → String unnecessarily.

  **WHY Each Reference Matters**:
  - ws.rs:44-45: This is the function signature to change
  - routes/metrics.rs:44-45: This is the production call site to update
  - ws.rs tests: Test calls with string literals need `.to_string()` appended

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Code compiles after broadcast signature change
    Tool: Bash (cargo check)
    Preconditions: Both files edited
    Steps:
      1. cd Rust && cargo check 2>&1
      2. Verify exit code 0
    Expected Result: Compiles successfully, no type errors
    Failure Indicators: "mismatched types", "expected String found &str", exit code != 0
    Evidence: .sisyphus/evidence/task-2-cargo-check.txt

  Scenario: ws.rs tests pass with new signature
    Tool: Bash (cargo test)
    Preconditions: Code compiles
    Steps:
      1. cd Rust && cargo test ws:: 2>&1
      2. Verify all ws module tests pass
    Expected Result: All ws module tests pass, 0 failures
    Failure Indicators: Any test failure in ws module
    Evidence: .sisyphus/evidence/task-2-ws-tests.txt
  ```

  **Commit**: YES
  - Message: `perf(ws): broadcast accepts String to avoid re-allocation`
  - Files: `Rust/src/ws.rs`, `Rust/src/routes/metrics.rs`
  - Pre-commit: `cd Rust && cargo test ws::`

- [x] 3. P0 — Eliminate Double Parse + Typed build_step_data + Lock Optimization

  **What to do**:
  This is THE critical refactoring. It combines PERFORMANCE.md items #1 (eliminate double parse) and #5 (move build_step_data outside lock) into one atomic change. The lock optimization is automatically achieved because build_step_data no longer needs the `raw` parameter.

  **Step-by-step:**

  **A. Rewrite `routes/metrics.rs` POST handler (lines 9-51):**
  - Remove the `raw: serde_json::Value = serde_json::from_slice(&body)?` line
  - Change `serde_json::from_value(raw.clone())` to `serde_json::from_slice(&body)` — single parse directly to MetricsPayload
  - Remove the `raw` variable entirely
  - Change `state.store.add_metrics(payload, raw).await` to `state.store.add_metrics(payload).await`

  **B. Rewrite `build_step_data` in `store.rs` (lines 117-163):**
  - Change signature from `fn build_step_data(payload: &MetricsPayload, raw: &serde_json::Value) -> StepData` to `fn build_step_data(payload: &MetricsPayload) -> StepData`
  - Rewrite body to construct `serde_json::Value` from typed struct fields:
    ```rust
    fn build_step_data(payload: &MetricsPayload) -> StepData {
        let sanitized_layers: Vec<serde_json::Value> = payload.layer_statistics
            .iter()
            .map(|ls| {
                let sanitized_id = sanitize_layer_id(&ls.layer_id);
                serde_json::json!({
                    "layer_id": sanitized_id,
                    "layer_type": ls.layer_type,
                    "depth_index": ls.depth_index,
                    "intermediate_features": {
                        "activation_std": ls.intermediate_features.activation_std,
                        "activation_mean": ls.intermediate_features.activation_mean,
                        "activation_shape": ls.intermediate_features.activation_shape,
                        "cross_layer_std_ratio": ls.intermediate_features.cross_layer_std_ratio,
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": ls.gradient_flow.gradient_l2_norm,
                        "gradient_std": ls.gradient_flow.gradient_std,
                        "gradient_max_abs": ls.gradient_flow.gradient_max_abs,
                    },
                    "parameter_statistics": {
                        "weight": {
                            "std": ls.parameter_statistics.weight.std,
                            "mean": ls.parameter_statistics.weight.mean,
                            "spectral_norm": ls.parameter_statistics.weight.spectral_norm,
                            "frobenius_norm": ls.parameter_statistics.weight.frobenius_norm,
                        },
                        "bias": ls.parameter_statistics.bias,
                    },
                })
            })
            .collect();

        let sanitized_layer_groups = payload.metadata.layer_groups.as_ref().map(|groups| {
            groups.iter().map(|(key, layer_ids)| {
                let sanitized_ids: Vec<String> = layer_ids.iter()
                    .map(|id| sanitize_layer_id(id).into())
                    .collect();
                (key.clone(), sanitized_ids)
            }).collect()
        });

        let cross_layer = serde_json::to_value(&payload.cross_layer_analysis)
            .expect("cross_layer_analysis serialization should never fail");

        StepData {
            step: payload.metadata.global_step,
            timestamp: *payload.metadata.timestamp,
            batch_size: payload.metadata.batch_size,
            layers: sanitized_layers,
            cross_layer,
            layer_groups: sanitized_layer_groups,
        }
    }
    ```
  - NOTE: The `sanitize_layer_id` function still returns `String` at this point. Task 4 will change it to `Cow<str>`. The `.into()` call on the layer_groups sanitization anticipates this (Cow→String via .into() works for both String and Cow).

  **C. Rewrite `add_metrics` in `store.rs` (lines 35-86):**
  - Change signature from `pub async fn add_metrics(&self, payload: MetricsPayload, raw: serde_json::Value) -> StepData` to `pub async fn add_metrics(&self, payload: MetricsPayload) -> StepData`
  - Move `build_step_data` call BEFORE write lock acquisition:
    ```rust
    pub async fn add_metrics(&self, payload: MetricsPayload) -> StepData {
        payload.validate().expect("validation failed");

        let step_data = build_step_data(&payload);  // BEFORE lock
        let run_id = payload.metadata.run_id.clone();

        let mut runs = self.runs.write().await;  // Lock acquired AFTER build_step_data
        // ... rest of logic unchanged (eviction, insertion, sorting)
    ```
  - This automatically achieves PERFORMANCE.md #5 (reduce lock hold time by ~30%)

  **D. Update `make_payload_and_raw` helper in `store.rs` unit tests (around line 171-220):**
  - Rename to `make_payload` and return just `MetricsPayload` (remove the `serde_json::Value` from the return tuple)
  - Update all ~30 call sites that destructure `(payload, raw)` to just `payload`
  - Remove all `raw` variables from the test helper and call sites

  **E. Update direct `add_metrics` calls in store.rs unit tests (~4 additional call sites beyond the helper):**
  - Search for `store.add_metrics(` in store.rs — each call passing `(payload, raw)` or `(p, r)` must change to `(payload)` or `(p)` only
  - The payload construction stays the same (serde_json::from_str/json! → MetricsPayload)
  - The raw construction is simply deleted

  **F. Update `tests/test_storage.rs` (3 direct store calls at lines ~153, 160, 165):**
  - These tests call `store.add_metrics(payload, json)` directly
  - Remove the `json` parameter, keep only `payload`
  - Remove the `serde_json::Value` construction for the `json` variable (keep the `MetricsPayload` construction)

  **G. Update `tests/common/mod.rs` helper:**
  - If `valid_payload()` or similar helpers return `(MetricsPayload, serde_json::Value)`, change to return just `MetricsPayload`
  - Check if any test helper constructs `raw: serde_json::Value` alongside the payload

  **Must NOT do**:
  - Do NOT change `StepData`, `RunData`, or any struct in `models.rs`
  - Do NOT change the validation logic in `MetricsPayload::validate()`
  - Do NOT change error response formats
  - Do NOT change WebSocket message formats (build_new_metrics_message, etc.)
  - Do NOT change the compact_layer/compact_step functions in ws.rs
  - Do NOT touch routes/runs.rs, routes/health.rs, routes/ws_route.rs
  - Do NOT modify HTTP-level integration test assertions (test_metrics_endpoint.rs, test_runs_endpoint.rs, test_edge_cases.rs, test_error_responses.rs, test_websocket.rs)
  - Do NOT add any new dependencies beyond what Task 1 added to Cargo.toml

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: This is a fundamental data flow refactoring touching 2 core files + 3 test files. Requires understanding the dual-path design, serde behavior, and careful test updates across 38 call sites. High risk if done wrong, but well-defined scope.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Wave 1)
  - **Blocks**: Task 4
  - **Blocked By**: Task 1 (hashbrown in Cargo.toml), Task 2 (routes/metrics.rs shared)

  **References**:

  **Pattern References**:
  - `Rust/src/routes/metrics.rs:9-51` — Full POST handler. The dual-parse pattern to eliminate. Lines 13 (from_slice→Value), 23 (from_value→MetricsPayload with raw.clone()), 42 (add_metrics with both args)
  - `Rust/src/store.rs:117-163` — Current `build_step_data` that reads from raw Value. The function to rewrite completely. Note how it reads `raw["layer_statistics"].as_array()` and clones each layer Value, plus `raw["cross_layer_analysis"].cloned()`
  - `Rust/src/store.rs:35-86` — Current `add_metrics` with write lock at line 40, build_step_data at line 69 (inside lock). After P0, build_step_data moves to before line 40
  - `Rust/src/store.rs:171-220` — `make_payload_and_raw` test helper. Returns `(MetricsPayload, serde_json::Value)`. Must become `make_payload` returning just `MetricsPayload`

  **API/Type References**:
  - `Rust/src/models.rs:142-150` — `LayerStatistic` struct with all fields (layer_id, layer_type, depth_index, intermediate_features, gradient_flow, parameter_statistics). These are the fields to use in json! macro.
  - `Rust/src/models.rs:152-156` — `CrossLayerAnalysis` struct (feature_std_gradient, gradient_norm_ratio). Use `serde_json::to_value()` for this.
  - `Rust/src/models.rs:107-113` — `IntermediateFeatures` struct fields
  - `Rust/src/models.rs:115-120` — `GradientFlow` struct fields
  - `Rust/src/models.rs:122-140` — `WeightStats`, `BiasStats`, `ParameterStatistics` struct fields
  - `Rust/src/models.rs:158-206` — `MetricsPayload` with `metadata`, `layer_statistics`, `cross_layer_analysis`
  - `Rust/src/models.rs:98-105` — `Metadata` with `run_id`, `timestamp`, `global_step`, `batch_size`, `layer_groups`
  - `Rust/src/models.rs:224-232` — `StepData` struct (unchanged, but this is what build_step_data must produce)

  **Test References**:
  - `Rust/tests/test_storage.rs:146-177` — Direct `store.add_metrics(payload, json)` calls that need the `json` parameter removed
  - `Rust/tests/common/mod.rs` — Test helpers that may need updating

  **External References**:
  - `serde_json::json!` macro: Constructs Value from typed expressions. Handles FiniteF64/NonNegativeF64 via their Serialize implementations (which serialize as plain f64 numbers).
  - `serde_json::to_value`: Converts any Serialize type to Value. Use for CrossLayerAnalysis.

  **WHY Each Reference Matters**:
  - routes/metrics.rs: The entry point of the dual-path design. Must become single-parse.
  - store.rs:117-163: The function to completely rewrite. Understanding what it reads from `raw` vs `payload` is critical.
  - store.rs:171-220: The test helper that generates (payload, raw) tuples for 30 test sites. Fixing this one function fixes 30 call sites mechanically.
  - models.rs structs: Every field in every struct must appear in the json! macro output. Missing a field = schema break.
  - The FiniteF64 and NonNegativeF64 types serialize as plain f64 via their custom Serialize impl — json! handles this transparently.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All Rust tests pass after P0 refactoring
    Tool: Bash (cargo test)
    Preconditions: All code changes applied
    Steps:
      1. cd Rust && cargo test 2>&1
      2. Count total tests and failures
    Expected Result: All 98+ tests pass, 0 failures. Test count same or higher than before.
    Failure Indicators: Any test failure, compilation error
    Evidence: .sisyphus/evidence/task-3-cargo-test.txt

  Scenario: JSON output schema matches pre-change
    Tool: Bash (cargo test + jq)
    Preconditions: Tests pass
    Steps:
      1. cd Rust && cargo test test_add_metrics_preserves_cross_layer 2>&1 — verify cross_layer_analysis is preserved through typed struct round-trip
      2. cd Rust && cargo test test_layer_id_sanitization 2>&1 — verify dots→slashes still works via typed path
      3. cd Rust && cargo test test_cross_layer 2>&1 — verify cross_layer field structure
      4. cd Rust && cargo test test_edge_cases -- layer_id 2>&1 — verify sanitization in integration tests
    Expected Result: All schema-related tests pass. Layer objects have all 6 fields. Cross-layer has feature_std_gradient and gradient_norm_ratio.
    Failure Indicators: Missing fields, wrong types, assertion failures
    Evidence: .sisyphus/evidence/task-3-schema-tests.txt

  Scenario: Error handling still works (422 for invalid input)
    Tool: Bash (cargo test)
    Preconditions: Tests pass
    Steps:
      1. cd Rust && cargo test test_error 2>&1
      2. cd Rust && cargo test test_metrics_endpoint 2>&1
    Expected Result: All error handling tests pass. 422 format unchanged.
    Failure Indicators: Any validation test failure
    Evidence: .sisyphus/evidence/task-3-error-tests.txt

  Scenario: Write lock is held for less time (build_step_data outside lock)
    Tool: Bash (cargo test)
    Preconditions: Code compiles
    Steps:
      1. Verify in store.rs that build_step_data(&payload) is called BEFORE self.runs.write().await
      2. cd Rust && cargo test test_concurrent 2>&1 — verify concurrent access tests pass
    Expected Result: build_step_data call is above the write lock line. Concurrent tests pass.
    Failure Indicators: build_step_data still inside lock, concurrent test failures
    Evidence: .sisyphus/evidence/task-3-lock-order.txt
  ```

  **Commit**: YES
  - Message: `perf(core): eliminate double JSON parse, build StepData from typed structs`
  - Files: `Rust/src/routes/metrics.rs`, `Rust/src/store.rs`, `Rust/tests/common/mod.rs`, `Rust/tests/test_storage.rs`
  - Pre-commit: `cd Rust && cargo test`

- [x] 4. P1/P2 — hashbrown + binary_search + Arc Removal + Cow + now_iso Reduction

  **What to do**:
  Apply 5 remaining store.rs optimizations as one coordinated pass. All changes are internal to store.rs only.

  **Step-by-step:**

  **A. Replace `std::collections::HashMap` with `hashbrown::HashMap` in store.rs:**
  - Add `use hashbrown::HashMap;` at the top of store.rs (replacing or supplementing the existing `use std::collections::HashMap;`)
  - The `runs: Arc<RwLock<HashMap<String, RunData>>>` (now `runs: RwLock<HashMap<String, RunData>>` after step C below) uses hashbrown automatically
  - Do NOT change models.rs — keep `std::collections::HashMap` there (public API types with serde derives)
  - Do NOT change any HashMap in ws.rs, routes/*, etc.

  **B. Binary search for step dedup (lines ~71-76 area, after Task 3 refactoring):**
  - Replace the current linear scan with binary search:
    ```rust
    // Clone step_data before the match — both arms need ownership
    let existing = run.steps.binary_search_by_key(&step_data.step, |s| s.step);
    match existing {
        Ok(idx) => run.steps[idx] = step_data,
        Err(idx) => run.steps.insert(idx, step_data),
    }
    ```
  - IMPORTANT: Do NOT use `step_data` directly in both match arms — it would be moved. The pattern above works because `step_data` is moved into the vector in both arms (only one arm executes). If the compiler complains, clone before the match.
  - Remove the `run.steps.sort_by_key(|s| s.step)` line that was previously needed after push — binary_search inserts at the correct position, so sorting is unnecessary

  **C. Remove inner `Arc` from MetricsStore struct:**
  - Change `runs: Arc<RwLock<HashMap<String, RunData>>>` to `runs: RwLock<HashMap<String, RunData>>`
  - In `new()`, change `Arc::new(RwLock::new(HashMap::new()))` to `RwLock::new(HashMap::new())`
  - All `self.runs.read().await` and `self.runs.write().await` calls work identically — RwLock provides the methods directly

  **D. `Cow<str>` for `sanitize_layer_id`:**
  - Change function signature from `fn sanitize_layer_id(layer_id: &str) -> String` to `fn sanitize_layer_id(layer_id: &str) -> Cow<'_, str>`
  - Change body to:
    ```rust
    fn sanitize_layer_id(layer_id: &str) -> Cow<'_, str> {
        if layer_id.contains('.') {
            Cow::Owned(layer_id.replace('.', "/"))
        } else {
            Cow::Borrowed(layer_id)
        }
    }
    ```
  - Add `use std::borrow::Cow;` at the top of store.rs
  - The call sites in build_step_data (from Task 3) should work because `Cow<str>` implements `Serialize` and is usable in `serde_json::json!`. If the json! macro needs help, wrap with `.as_ref()` or `.into()`.

  **E. Reduce `now_iso()` calls:**
  - Restructure the run creation/update path to avoid redundant timestamp generation:
    ```rust
    // For NEW runs:
    let now = now_iso();
    // Create RunData with now.clone() for both created_at and last_update
    // Then DON'T call now_iso() again at the end — the timestamp is already set

    // For EXISTING runs:
    // Only generate fresh timestamp at the end (run.last_update = now_iso())
    ```
  - Current code calls now_iso() once at line ~54, then again at line ~83. For new runs, the second call is redundant since no time-sensitive work happened between them.

  **Must NOT do**:
  - Do NOT modify models.rs (keep std::collections::HashMap there)
  - Do NOT modify routes/metrics.rs (already done in Task 3)
  - Do NOT modify ws.rs or any routes
  - Do NOT change test assertions (only test helper signatures if needed)
  - Do NOT change the sort stability or dedup behavior (must remain semantically identical)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multiple coordinated optimizations in a single file. Each is individually simple but requires care to avoid introducing bugs. High effort but well-defined scope.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential after Wave 2)
  - **Blocks**: F1-F4 (final verification)
  - **Blocked By**: Task 3 (store.rs shared file)

  **References**:

  **Pattern References**:
  - `Rust/src/store.rs:20-24` — Current MetricsStore struct with inner Arc<RwLock<HashMap>>
  - `Rust/src/store.rs:27-33` — `new()` constructor
  - `Rust/src/store.rs:12-14` — Current `sanitize_layer_id` returning String
  - `Rust/src/store.rs:35-86` — `add_metrics` with now_iso() calls at lines ~54 and ~83, and step dedup at lines ~71-76
  - `Rust/src/store.rs:88-91` — `get_run` with RwLock read (Arc removal doesn't change this pattern)

  **API/Type References**:
  - `hashbrown::HashMap` — Drop-in replacement for std::collections::HashMap. Same API, different hasher (foldhash vs SipHash). Implements Serialize/Deserialize via serde feature.
  - `std::borrow::Cow<'_, str>` — Clone-on-write smart pointer. Implements Serialize (as str) and Into<String>.
  - `Vec::binary_search_by_key` — Returns Result<usize, usize>. Ok = found at index. Err = insertion point.

  **Test References**:
  - `Rust/src/store.rs` unit tests — All store unit tests validate the optimizations:
    - `test_step_dedup_replaces` — validates binary_search Ok path
    - `test_step_dedup_in_middle` — validates insertion ordering
    - `test_steps_sorted` — validates binary_search Err path maintains sort
    - `test_run_eviction_oldest` — validates HashMap operations
    - `test_layer_id_sanitization` — validates Cow::Owned path (dots → slashes)
    - Tests using `"layer1"` (no dots) — validates Cow::Borrowed path
    - `test_last_update_on_dedup` — validates now_iso() behavior

  **External References**:
  - hashbrown 0.15 docs: Uses foldhash by default, ~2-4× faster than SipHash for lookups

  **WHY Each Reference Matters**:
  - store.rs struct definition: The Arc removal changes the struct layout
  - add_metrics body: All 5 optimizations apply to different parts of this one function + sanitize_layer_id
  - Unit tests: These are the primary verification — if they pass, the optimizations are correct

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All Rust tests pass with all optimizations applied
    Tool: Bash (cargo test)
    Preconditions: All changes applied
    Steps:
      1. cd Rust && cargo test 2>&1
      2. Count total tests and failures
    Expected Result: All 98+ tests pass, 0 failures
    Failure Indicators: Any test failure
    Evidence: .sisyphus/evidence/task-4-cargo-test.txt

  Scenario: Clippy passes with no warnings
    Tool: Bash (cargo clippy)
    Preconditions: Code compiles
    Steps:
      1. cd Rust && cargo clippy --all-targets --all-features 2>&1
      2. Check for warnings about Cow usage, unnecessary allocations, etc.
    Expected Result: 0 warnings from clippy
    Failure Indicators: Any clippy warning
    Evidence: .sisyphus/evidence/task-4-clippy.txt

  Scenario: Release build succeeds
    Tool: Bash (cargo build --release)
    Preconditions: All tests pass
    Steps:
      1. cd Rust && cargo build --release 2>&1
      2. Verify exit code 0
    Expected Result: Successful release build with LTO, single codegen unit, opt-level 3
    Failure Indicators: Compilation error, linking error
    Evidence: .sisyphus/evidence/task-4-release-build.txt

  Scenario: hashbrown is used in store.rs but not models.rs
    Tool: Bash (grep)
    Preconditions: Changes applied
    Steps:
      1. grep "hashbrown" Rust/src/store.rs — expect import present
      2. grep "hashbrown" Rust/src/models.rs — expect NOT present
      3. grep "std::collections::HashMap" Rust/src/store.rs — expect NOT present (replaced)
      4. grep "std::collections::HashMap" Rust/src/models.rs — expect present (kept)
    Expected Result: hashbrown only in store.rs, std::collections::HashMap only in models.rs
    Failure Indicators: hashbrown in models.rs, or std::collections::HashMap in store.rs
    Evidence: .sisyphus/evidence/task-4-hashbrown-scope.txt

  Scenario: Binary search correctness — step dedup and insertion
    Tool: Bash (cargo test)
    Preconditions: Changes applied
    Steps:
      1. cd Rust && cargo test test_step_dedup 2>&1 — validates replacement (Ok path)
      2. cd Rust && cargo test test_steps_sorted 2>&1 — validates insertion (Err path)
      3. cd Rust && cargo test test_step_dedup_in_middle 2>&1 — validates ordering
    Expected Result: All dedup and sorting tests pass
    Failure Indicators: Any test failure in dedup/sort tests
    Evidence: .sisyphus/evidence/task-4-binary-search.txt
  ```

  **Commit**: YES
  - Message: `perf(store): hashbrown, binary_search, remove Arc, Cow, reduce now_iso`
  - Files: `Rust/src/store.rs`
  - Pre-commit: `cd Rust && cargo test && cargo clippy --all-targets --all-features`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
>
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `cargo clippy --all-targets --all-features` + `cargo test` + `cargo build --release`. Review all changed files for: `as any`/unnecessary `unsafe`, empty catches, console.log-equivalents, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Clippy [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Build and start the server. Execute curl-based JSON regression tests: POST a known payload, GET the run, compare JSON output structure. Verify each endpoint returns correct shape. Test edge cases: duplicate step, layer_id with dots, cross_layer_analysis preservation, empty runs.
  Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Schema [MATCH/MISMATCH] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination: Task N touching Task M's files. Flag unaccounted changes.
  Verify these files were NOT modified: `src/models.rs`, `src/routes/runs.rs`, `src/routes/health.rs`, `src/routes/ws_route.rs`, any Python test files.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Task 1**: `perf(cargo): remove dead deps, add hashbrown, add release profile` — Cargo.toml
- **Task 2**: `perf(ws): broadcast accepts String to avoid re-allocation` — src/ws.rs, src/routes/metrics.rs
- **Task 3**: `perf(core): eliminate double JSON parse, build StepData from typed structs` — src/routes/metrics.rs, src/store.rs, tests/common/mod.rs, tests/test_storage.rs
- **Task 4**: `perf(store): hashbrown, binary_search, remove Arc, Cow, reduce now_iso` — src/store.rs

---

## Success Criteria

### Verification Commands
```bash
cd Rust && cargo test                    # Expected: 98+ tests, 0 failures
cd Rust && cargo clippy --all-targets     # Expected: 0 warnings
cd Rust && cargo build --release          # Expected: successful compilation
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] JSON output schema matches pre-change exactly
- [ ] No changes to models.rs struct definitions
- [ ] No changes to runs.rs, health.rs, ws_route.rs
