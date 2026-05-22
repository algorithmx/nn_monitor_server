# Rust Migration: NN Monitor Server

## TL;DR

> **Quick Summary**: Migrate the Python/FastAPI NN Training Monitor Server to Rust/Axum in `./Rust/`, preserving the exact REST API + WebSocket protocol + JSON schema. Copy all language-agnostic files (frontend, docs, test clients) as-is.
>
> **Deliverables**:
> - Complete Rust/Axum server in `./Rust/` with identical service contract
> - Copied static frontend, docs, and Python test clients
> - Rust unit/integration tests + Python contract test verification
> - Cargo.toml with all dependencies
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: Task 1 → Task 5 → Task 8 → Task 11 → Task 15 → Task 16 → F1-F4

---

## Context

### Original Request
Migrate the Python/FastAPI NN Training Monitor Server backend to Rust for lower memory cost and lower CPU occupation, keeping the exact service contract to clients. Copy as many useful files as possible from the Python project into `./Rust/`.

### Interview Summary
**Key Discussions**:
- Framework: User chose **Axum** (over Actix-web) for best tokio ecosystem alignment
- Test strategy: User chose **Rust tests + Python contract tests** — write Rust unit/integration tests AND copy Python test clients for contract verification
- Target: `./Rust/` directory (already exists, empty)

**Research Findings**:
- Current server: 613-line single-file Python FastAPI app
- 5 REST endpoints + 1 WebSocket endpoint
- In-memory storage with async locks, step dedup, run/step eviction
- Pydantic models for validation (complex nested, with custom validators)
- WebSocket protocol: initial_runs, subscribe_run (with `lite` flag), ping/pong, new_metrics broadcast, error messages
- Layer ID sanitization: dots → slashes (in both layer_statistics and layer_groups)
- Compact payload functions for WebSocket `lite` mode (frontend uses this)

### Metis Review
**Identified Gaps** (addressed):
- WebSocket `lite` mode was not discussed but frontend depends on it → Added as mandatory requirement
- Dual error response formats (422 list vs 400/404/500 dict) → Must replicate exactly
- `serde_json` NaN/Inf trap (silently converts to `null`) → Must use custom `FiniteF64` newtype
- Timestamp format mismatch: Python `isoformat()` vs Rust `chrono` → Must match Python format exactly
- CORS middleware missing in Python despite config field → Match Python behavior (no CORS)
- `broadcast::channel` capacity for slow WebSocket clients → Set capacity 1024+, handle `Lagged` explicitly
- `StepData.layers` stored as `Vec<serde_json::Value>` (untyped dicts), not typed structs → Must match
- `check_layer_ordering` error message format must be replicated exactly

---

## Work Objectives

### Core Objective
Create a functionally equivalent Rust/Axum server in `./Rust/` that produces identical HTTP responses, WebSocket messages, and error formats as the Python/FastAPI server.

### Concrete Deliverables
- `./Rust/Cargo.toml` — Project manifest with all dependencies
- `./Rust/src/` — Rust source code (main.rs, models.rs, store.rs, ws.rs, config.rs, routes/)
- `./Rust/static/` — Copied frontend (index.html, app.js, styles.css)
- `./Rust/tests/` — Rust integration tests
- `./Rust/API.md`, `./Rust/SCHEMA.md`, `./Rust/CONFIGURATION.md` — Copied docs
- `./Rust/test_client.py`, `./Rust/test_client_brutal.py`, `./Rust/test_client_comprehensive.py` — Python contract tests
- `./Rust/example_client/` — Copied PyTorch client examples

### Definition of Done
- [ ] `cd Rust && cargo build --release` succeeds with 0 errors
- [ ] `cd Rust && cargo test` — all tests pass
- [ ] Python `test_client.py` runs successfully against Rust server
- [ ] Frontend at `http://localhost:8000/` renders and connects via WebSocket
- [ ] All REST endpoints return identical JSON responses as Python server
- [ ] WebSocket protocol messages are identical to Python server

### Must Have
- Identical REST API: same routes, same status codes, same response bodies
- Identical WebSocket protocol: same message types, same data shapes
- Identical validation: same constraints (NaN/Inf rejection, depth_index sorting, field bounds)
- Identical storage semantics: step dedup, run eviction, step eviction
- WebSocket `lite` mode (compact payloads) — frontend depends on it
- Dual error response formats: 422 = `{"detail": [list]}`, 400/404/500 = `{"detail": {"error": "...", "message": "..."}}`
- Layer ID sanitization in both `layer_statistics[].layer_id` AND `layer_groups` values
- Environment variable config with `NN_MONITOR_` prefix
- Static file serving for frontend
- Graceful WebSocket connection management with broadcast
- `FiniteF64` custom type rejecting NaN/Inf at deserialization time
- Timestamp format matching Python's `datetime.now().isoformat()` (6 decimal places, no timezone)

### Must NOT Have (Guardrails)
- NO CORS middleware (Python doesn't use it despite having the config field)
- NO authentication/authorization
- NO database/persistence
- NO new API endpoints or WebSocket message types
- NO changes to default port (8000) or env var prefix (`NN_MONITOR_`)
- NO changes to WebSocket message field names or types
- NO `serde_json` default NaN handling (silently converts to `null` — use `FiniteF64`)
- NO `tokio::sync::Mutex` for MetricsStore (use `RwLock` — read-heavy workload)
- NO `sort_unstable_by_key` for step ordering (must use stable sort)
- NO optimization of storage data structures (match Python semantics first)

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO (new Rust project)
- **Automated tests**: YES (tests-after — Rust tests written alongside implementation)
- **Framework**: Rust built-in test framework + `axum-test` for HTTP integration tests + `tokio-tungstenite` for WS tests
- **Contract verification**: Python test clients (`test_client.py`, `test_client_brutal.py`) run against Rust server

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **REST API**: Use Bash (curl) — Send requests, assert status + response fields
- **WebSocket**: Use Bash (websocat or Python script) — Connect, send messages, assert responses
- **Storage**: Use Bash (cargo test) — Run unit tests for dedup/eviction
- **Frontend**: Use Playwright — Navigate to dashboard, verify WebSocket connection, verify rendering

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - scaffolding + file copying):
├── Task 1: Cargo init + dependency setup [quick]
├── Task 2: Copy language-agnostic files to ./Rust/ [quick]
├── Task 3: Config module (env vars + ServerConfig) [quick]
├── Task 4: Serde models + FiniteF64 newtype + validation [deep]
└── Task 5: MetricsStore (in-memory storage engine) [deep]

Wave 2 (After Wave 1 - core server):
├── Task 6: WebSocket manager (broadcast channel) [deep]
├── Task 7: REST routes — POST metrics + GET runs [unspecified-high]
├── Task 8: REST routes — GET run by ID + GET latest + GET health [unspecified-high]
├── Task 9: WebSocket endpoint (full protocol) [deep]
└── Task 10: main.rs (server entry + static files + graceful shutdown) [unspecified-high]

Wave 3 (After Wave 2 - testing):
├── Task 11: Rust integration tests — REST endpoints [unspecified-high]
├── Task 12: Rust integration tests — WebSocket protocol [unspecified-high]
├── Task 13: Rust unit tests — Storage (dedup + eviction) [unspecified-high]
├── Task 14: Python contract test verification [unspecified-high]
└── Task 15: Frontend integration verification (Playwright) [unspecified-high]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: Task 1 → Task 4 → Task 5 → Task 7 → Task 10 → Task 11 → Task 15 → F1-F4
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 5 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1 | - | 3, 4, 5 |
| 2 | 1 | 10 |
| 3 | 1 | 7, 8, 10 |
| 4 | 1 | 5, 7, 8, 9 |
| 5 | 1, 4 | 7, 8, 9, 13 |
| 6 | 1 | 9, 10 |
| 7 | 3, 4, 5, 6 | 10, 11 |
| 8 | 3, 4, 5 | 10, 11 |
| 9 | 4, 5, 6 | 10, 12 |
| 10 | 2, 3, 7, 8, 9 | 14, 15 |
| 11 | 7, 8 | F1-F4 |
| 12 | 9 | F1-F4 |
| 13 | 5 | F1-F4 |
| 14 | 10 | F1-F4 |
| 15 | 10 | F1-F4 |

### Agent Dispatch Summary

- **Wave 1**: 5 tasks — T1 → `quick`, T2 → `quick`, T3 → `quick`, T4 → `deep`, T5 → `deep`
- **Wave 2**: 5 tasks — T6 → `deep`, T7 → `unspecified-high`, T8 → `unspecified-high`, T9 → `deep`, T10 → `unspecified-high`
- **Wave 3**: 5 tasks — T11 → `unspecified-high`, T12 → `unspecified-high`, T13 → `unspecified-high`, T14 → `unspecified-high`, T15 → `unspecified-high`
- **FINAL**: 4 tasks — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Cargo Init + Dependency Setup

  **What to do**:
  - Run `cargo init ./Rust --name nn_monitor_server` to initialize the Rust project
  - Write `./Rust/Cargo.toml` with all required dependencies:
    ```toml
    [dependencies]
    axum = { version = "0.8", features = ["ws"] }
    tokio = { version = "1", features = ["full"] }
    tower = "0.5"
    tower-http = { version = "0.6", features = ["fs", "cors", "trace", "timeout"] }
    serde = { version = "1", features = ["derive"] }
    serde_json = "1"
    validator = "0.18"
    dotenvy = "0.15"
    envy = "0.4"
    futures-util = "0.3"
    tokio-util = { version = "0.7", features = ["rt"] }
    tracing = "0.1"
    tracing-subscriber = { version = "0.3", features = ["env-filter"] }
    chrono = "0.4"

    [dev-dependencies]
    axum-test = "16"
    ```
  - Create `./Rust/src/main.rs` with a minimal `fn main() { println!("hello"); }` placeholder
  - Verify `cd Rust && cargo build` succeeds

  **Must NOT do**:
  - Do NOT add `rusdantic` or other unstable crates — use battle-tested `validator`
  - Do NOT add `cors` feature to tower-http if Python doesn't use CORS middleware

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single config file creation, no logic
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `playwright`: No UI involved

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Tasks 3, 4, 5
  - **Blocked By**: None

  **References**:
  **Pattern References**:
  - `requirements.txt` — Python dependency list to translate to Cargo.toml equivalents

  **External References**:
  - Axum docs: https://docs.rs/axum/0.8 — version compatibility
  - tower-http features: https://docs.rs/tower-http/0.6 — `fs` for ServeDir, `cors` for CorsLayer

  **WHY Each Reference Matters**:
  - `requirements.txt` shows exact Python dependencies and their Rust equivalents (fastapi→axum, uvicorn→tokio, pydantic→serde+validator, pydantic-settings→envy+dotenvy, websockets→axum ws feature)

  **Acceptance Criteria**:
  - [ ] `./Rust/Cargo.toml` exists with all dependencies listed
  - [ ] `./Rust/src/main.rs` exists with minimal main function
  - [ ] `cd Rust && cargo build` succeeds with 0 errors

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Cargo build succeeds
    Tool: Bash
    Preconditions: ./Rust/Cargo.toml exists
    Steps:
      1. Run: cd Rust && cargo build 2>&1
      2. Check exit code is 0
      3. Check output contains "Compiling nn_monitor_server"
      4. Check output contains "Finished" with no "error" mentions
    Expected Result: Build succeeds with 0 errors
    Failure Indicators: "error[E", "could not compile", exit code != 0
    Evidence: .sisyphus/evidence/task-1-cargo-build.txt

  Scenario: All required crate features present
    Tool: Bash
    Preconditions: ./Rust/Cargo.toml exists
    Steps:
      1. Run: grep -c "axum" Rust/Cargo.toml
      2. Run: grep -c "tokio" Rust/Cargo.toml
      3. Run: grep -c "serde" Rust/Cargo.toml
      4. Run: grep -c "validator" Rust/Cargo.toml
    Expected Result: Each grep returns count >= 1
    Failure Indicators: Any grep returns 0
    Evidence: .sisyphus/evidence/task-1-deps-check.txt
  ```

  **Commit**: YES
  - Message: `feat(rust): initialize project with Cargo.toml and dependencies`
  - Files: `Rust/Cargo.toml`, `Rust/src/main.rs`

- [x] 2. Copy Language-Agnostic Files to ./Rust/

  **What to do**:
  - Create directory structure in `./Rust/`:
    ```
    Rust/static/         ← from root static/
    Rust/docs/           ← from root docs/
    Rust/example_client/ ← from root example_client/
    ```
  - Copy files VERBATIM (no modifications):
    - `static/index.html` → `Rust/static/index.html`
    - `static/app.js` → `Rust/static/app.js`
    - `static/styles.css` → `Rust/static/styles.css`
    - `API.md` → `Rust/API.md`
    - `SCHEMA.md` → `Rust/SCHEMA.md`
    - `CONFIGURATION.md` → `Rust/CONFIGURATION.md`
    - `docs/FEATURES.md` → `Rust/docs/FEATURES.md`
    - `dev-notes/spec.md` → `Rust/dev-notes/spec.md` (create dev-notes/ dir)
    - `example_client/nn_monitor.py` → `Rust/example_client/nn_monitor.py`
    - `example_client/train_monitored.py` → `Rust/example_client/train_monitored.py`
    - `example_client/dataset_loader.py` → `Rust/example_client/dataset_loader.py`
    - `test_client.py` → `Rust/test_client.py`
    - `test_client_brutal.py` → `Rust/test_client_brutal.py`
    - `test_client_comprehensive.py` → `Rust/test_client_comprehensive.py`
  - Copy and EDIT:
    - `README.md` → `Rust/README.md` — Update commands: `pip install` → `cargo build`, `python main.py` → `cargo run`
  - Create `Rust/.gitignore` adapted for Rust:
    ```
    /target
    Cargo.lock
    __pycache__/
    *.pyc
    .venv/
    .claude/
    .vscode/
    ```

  **Must NOT do**:
  - Do NOT modify any frontend files (app.js, styles.css, index.html)
  - Do NOT modify API.md, SCHEMA.md, CONFIGURATION.md
  - Do NOT copy Python-specific files (main.py, requirements.txt, pytest.ini, tests/)
  - Do NOT copy .claude/ or .vscode/ config

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Pure file copying, minimal editing
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Task 10
  - **Blocked By**: Task 1

  **References**:
  **Pattern References**:
  - `static/index.html` (122 lines) — HTML shell, verify paths remain `/static/app.js` etc.
  - `static/app.js` (1202 lines) — Uses `location.host` for WebSocket URL, no hardcoded paths
  - `README.md` (71 lines) — Commands to update: pip→cargo, python→cargo run

  **WHY Each Reference Matters**:
  - `app.js` uses `location.host` for WS connection — no change needed since Rust serves on same port
  - `README.md` is the only file needing content changes (Python→Rust commands)

  **Acceptance Criteria**:
  - [ ] `Rust/static/index.html` exists and `diff static/index.html Rust/static/index.html` shows no differences
  - [ ] `Rust/static/app.js` exists and is identical to source
  - [ ] `Rust/static/styles.css` exists and is identical to source
  - [ ] `Rust/API.md` exists and is identical to source
  - [ ] `Rust/SCHEMA.md` exists and is identical to source
  - [ ] `Rust/CONFIGURATION.md` exists and is identical to source
  - [ ] `Rust/test_client.py` exists and is identical to source
  - [ ] `Rust/test_client_brutal.py` exists and is identical to source
  - [ ] `Rust/test_client_comprehensive.py` exists and is identical to source
  - [ ] `Rust/example_client/nn_monitor.py` exists
  - [ ] `Rust/README.md` exists and contains `cargo build` (not `pip install`)
  - [ ] `Rust/.gitignore` exists and contains `/target`

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All static files copied correctly
    Tool: Bash
    Preconditions: Copy operations completed
    Steps:
      1. diff static/index.html Rust/static/index.html
      2. diff static/app.js Rust/static/app.js
      3. diff static/styles.css Rust/static/styles.css
      4. diff API.md Rust/API.md
      5. diff SCHEMA.md Rust/SCHEMA.md
    Expected Result: All diffs produce no output (files identical)
    Failure Indicators: Any diff outputs text (files differ)
    Evidence: .sisyphus/evidence/task-2-file-comparison.txt

  Scenario: README updated for Rust
    Tool: Bash
    Preconditions: Rust/README.md exists
    Steps:
      1. grep -c "cargo build" Rust/README.md
      2. grep -c "pip install" Rust/README.md
    Expected Result: cargo build count >= 1, pip install count == 0
    Failure Indicators: pip install found, or cargo build not found
    Evidence: .sisyphus/evidence/task-2-readme-check.txt
  ```

  **Commit**: YES (groups with Task 1)
  - Message: `feat(rust): initialize project scaffolding and copied assets`
  - Files: All copied files

- [x] 3. Config Module (Environment Variables + ServerConfig)

  **What to do**:
  - Create `./Rust/src/config.rs` with:
    ```rust
    use serde::Deserialize;

    #[derive(Deserialize, Debug, Clone)]
    pub struct ServerConfig {
        #[serde(default = "default_max_runs")]
        pub max_runs: usize,
        #[serde(default = "default_max_steps_per_run")]
        pub max_steps_per_run: usize,
        #[serde(default = "default_max_request_size")]
        pub max_request_size: usize,
        #[serde(default = "default_host")]
        pub host: String,
        #[serde(default = "default_port")]
        pub port: u16,
        #[serde(default = "default_log_level")]
        pub log_level: String,
        #[serde(default = "default_cors_origins")]
        pub cors_origins: Vec<String>,
    }
    // Default functions matching Python's defaults
    // Load from env vars with prefix NN_MONITOR_ using envy::prefixed("NN_MONITOR_")
    // Load .env file using dotenvy::dotenv().ok()
    ```
  - Defaults must match Python exactly: max_runs=10, max_steps_per_run=1000, max_request_size=2000000, host="0.0.0.0", port=8000, log_level="warning", cors_origins=["*"]
  - Add `mod config;` to `src/main.rs`

  **Must NOT do**:
  - Do NOT add CORS middleware (even though config has cors_origins)
  - Do NOT change default values from Python's defaults

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small file, well-defined structure, no complex logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Tasks 7, 8, 10
  - **Blocked By**: Task 1

  **References**:
  **Pattern References**:
  - `main.py:72-92` — Python ServerConfig class with exact default values and validation ranges (ge=1, le=100 for max_runs etc.)
  - `main.py:449-597` — How config is used in route handlers and server startup

  **API/Type References**:
  - `CONFIGURATION.md` — Documented env vars with types, defaults, and ranges

  **External References**:
  - envy crate: https://docs.rs/envy/0.4 — `envy::prefixed("NN_MONITOR_").from_env::<ServerConfig>()`
  - dotenvy crate: https://docs.rs/dotenvy/0.15 — `dotenvy::dotenv().ok()`

  **WHY Each Reference Matters**:
  - `main.py:72-92` defines exact field names, types, defaults, and validation ranges — must replicate
  - `CONFIGURATION.md` is the user-facing contract for env vars — must match

  **Acceptance Criteria**:
  - [ ] `Rust/src/config.rs` exists and compiles
  - [ ] `ServerConfig` struct has all 7 fields with correct types
  - [ ] Loading with no env vars produces defaults matching Python: max_runs=10, port=8000, host="0.0.0.0"
  - [ ] `NN_MONITOR_PORT=9000` env var correctly overrides default

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Default config matches Python defaults
    Tool: Bash
    Preconditions: src/config.rs compiles
    Steps:
      1. Write a small Rust test in config.rs that creates ServerConfig with no env vars
      2. Assert max_runs == 10, max_steps_per_run == 1000, port == 8000, host == "0.0.0.0"
      3. Run: cd Rust && cargo test test_default_config
    Expected Result: Test passes with all defaults matching
    Failure Indicators: Any assertion fails
    Evidence: .sisyphus/evidence/task-3-default-config.txt

  Scenario: Env var override works
    Tool: Bash
    Preconditions: src/config.rs compiles
    Steps:
      1. Write a test that sets env vars NN_MONITOR_PORT=9000, NN_MONITOR_MAX_RUNS=50
      2. Load config and assert port == 9000, max_runs == 50
      3. Run: cd Rust && cargo test test_env_override
    Expected Result: Env vars correctly override defaults
    Failure Indicators: Values don't match set env vars
    Evidence: .sisyphus/evidence/task-3-env-override.txt
  ```

  **Commit**: YES
  - Message: `feat(rust): add environment variable configuration module`
  - Files: `Rust/src/config.rs`, `Rust/src/main.rs`

- [x] 4. Serde Models + FiniteF64 Newtype + Validation

  **What to do**:
  - Create `./Rust/src/models.rs` with ALL data models matching the Python Pydantic models EXACTLY:

  **Critical: FiniteF64 newtype** — This is the #1 migration pitfall. `serde_json` silently converts `f64::NAN` to `null` and `f64::INFINITY` to `null` during serialization. You MUST create a custom type:

    ```rust
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct FiniteF64(f64);

    impl FiniteF64 {
        pub fn value(&self) -> f64 { self.0 }
        pub fn new(v: f64) -> Result<Self, String> {
            if v.is_finite() { Ok(Self(v)) } else { Err("Value must be finite (not NaN or Infinity)".into()) }
        }
    }
    // Custom serde deserializer that REJECTS NaN/Inf at parse time
    // Custom validator that ensures ge=0, gt=0 constraints
    ```

  **Models to implement** (field order MUST match Python declaration order for consistent JSON serialization):
    - `MetricsPayload { metadata, layer_statistics, cross_layer_analysis }`
    - `Metadata { run_id: String, timestamp: FiniteF64(>0), global_step: u64(>=0), batch_size: u32(>0), layer_groups: Option<HashMap<String, Vec<String>>> }`
    - `LayerStatistic { layer_id: String(!empty), layer_type: String(!empty), depth_index: u32(>=0), intermediate_features, gradient_flow, parameter_statistics }`
    - `IntermediateFeatures { activation_std: FiniteF64(>=0), activation_mean: FiniteF64, activation_shape: Vec<u64>(len>=2), cross_layer_std_ratio: Option<FiniteF64(>=0)> }`
    - `GradientFlow { gradient_l2_norm: FiniteF64(>=0), gradient_std: FiniteF64(>=0), gradient_max_abs: FiniteF64(>=0) }`
    - `ParameterStatistics { weight: WeightStats, bias: Option<BiasStats> }`
    - `WeightStats { std: FiniteF64(>=0), mean: FiniteF64, spectral_norm: FiniteF64(>=0), frobenius_norm: FiniteF64(>=0) }`
    - `BiasStats { std: FiniteF64(>=0), mean_abs: FiniteF64(>=0) }`
    - `CrossLayerAnalysis { feature_std_gradient: FiniteF64, gradient_norm_ratio: HashMap<String, f64> }`

  **Custom validators**:
    - `check_layer_ordering`: Verify `depth_index` is monotonically non-decreasing. Error message must EXACTLY match: `"Layers must be sorted by depth_index: {layer_id} has depth_index {d1} but {next_layer_id} has depth_index {d2}"`
    - All `FiniteF64` fields with `>=0` constraint: custom validator that checks `value >= 0.0`
    - `run_id` non-empty string validation
    - `layer_statistics` non-empty array validation
    - `activation_shape` minimum length 2 validation

  **Response models**:
    - `MetricsAcceptedResponse { status: "accepted", run_id: String }`
    - `RunInfo { created_at: String, last_update: String, step_count: u32, latest_step: Option<u64> }`
    - `StepData { step: u64, timestamp: f64, batch_size: u32, layers: Vec<serde_json::Value>, cross_layer: serde_json::Value, layer_groups: Option<HashMap<String, Vec<String>>> }`
    - `RunData { created_at: String, last_update: String, steps: Vec<StepData> }`
    - `HealthResponse { status: "healthy", active_connections: u32 }`
    - `ErrorDetail { error: String, message: String }`

  **Important**: `StepData.layers` is `Vec<serde_json::Value>` (untyped dicts), NOT typed `Vec<LayerStatistic>`. This matches Python where sanitized layers are stored as plain dicts.

  **Important**: `status` fields use `#[serde(rename_all = "snake_case")]` and literal string defaults via `#[serde(default = "default_accepted")]`.

  **Add `mod models;` to `src/main.rs`.**

  **Must NOT do**:
  - Do NOT use raw `f64` for any field that must reject NaN/Inf — always use `FiniteF64`
  - Do NOT change field declaration order from Python's Pydantic model order
  - Do NOT use `serde_json` default NaN handling

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex type system design, custom serde impl, many validation rules, critical correctness requirement
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (depends only on Cargo.toml)
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Tasks 5, 7, 8, 9
  - **Blocked By**: Task 1

  **References**:
  **Pattern References**:
  - `main.py:98-299` — ALL Pydantic models with exact field names, types, constraints, validators. THIS IS THE PRIMARY REFERENCE. Every struct must match field-for-field.
  - `main.py:111-116` — `validate_finite` validator: rejects NaN and Infinity with exact error message `"Value must be finite (not NaN or Infinity)"`
  - `main.py:189-202` — `check_layer_ordering` validator with exact error message format
  - `main.py:21-58` — Compact payload helpers (`_compact_layer`, `_compact_step`, `_compact_run`) — these define which fields are included in WebSocket lite mode

  **API/Type References**:
  - `SCHEMA.md` — Complete field-by-field documentation with types, required/optional, validation rules
  - `API.md` — Response format examples for all endpoints and WebSocket messages

  **Test References**:
  - `test_pydantic_models.py` (738 lines) — Boundary test cases: NaN rejection, Inf rejection, zero values, negative values, optional bias, run_id patterns. Each test defines what must ACCEPT vs REJECT.
  - `tests/fixtures/metric_fixtures.py` (1220 lines) — Valid payload factories. The JSON output of each factory is a valid input for Rust.
  - `tests/fixtures/error_fixtures.py` (1064 lines) — Invalid payload factories. Each one defines what Rust must REJECT.

  **External References**:
  - `validator` crate: https://docs.rs/validator/0.18 — `#[derive(Validate)]`, `#[validate(range(min = 0))]`, custom validators
  - `serde` custom deserialize: https://serde.rs/custom-date-format.html — Pattern for FiniteF64

  **WHY Each Reference Matters**:
  - `main.py:98-299` is the single source of truth for all data shapes. Every Rust struct is a 1:1 translation.
  - `main.py:189-202` contains the EXACT error message format that contract tests check for.
  - `test_pydantic_models.py` defines the validation boundary: which values are accepted vs rejected. This is the behavioral spec for Rust validation.
  - `FiniteF64` is critical because `serde_json` silently converts NaN→null, which would break contract tests that expect 422 errors for NaN input.

  **Acceptance Criteria**:
  - [ ] `Rust/src/models.rs` exists and compiles
  - [ ] All 9 input models and 6 response models defined
  - [ ] `FiniteF64` rejects NaN and Infinity during deserialization (not just validation)
  - [ ] `check_layer_ordering` validator produces exact Python error message format
  - [ ] `cargo test` includes model validation tests that pass

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: NaN rejected at deserialization
    Tool: Bash
    Preconditions: models.rs compiles
    Steps:
      1. Write test: deserialize JSON with "activation_std": NaN
      2. Assert result is Err (deserialization fails)
      3. Run: cd Rust && cargo test test_nan_rejected
    Expected Result: Test passes — NaN is rejected at parse time
    Failure Indicators: NaN is accepted (deserialization succeeds)
    Evidence: .sisyphus/evidence/task-4-nan-reject.txt

  Scenario: Infinity rejected at deserialization
    Tool: Bash
    Preconditions: models.rs compiles
    Steps:
      1. Write test: deserialize JSON with "gradient_l2_norm": Infinity
      2. Assert result is Err
      3. Run: cd Rust && cargo test test_inf_rejected
    Expected Result: Test passes — Infinity is rejected
    Failure Indicators: Infinity is accepted
    Evidence: .sisyphus/evidence/task-4-inf-reject.txt

  Scenario: Unsorted depth_index rejected with exact error message
    Tool: Bash
    Preconditions: models.rs compiles
    Steps:
      1. Write test: deserialize JSON with layers having depth_index [1, 0]
      2. Assert error message contains "Layers must be sorted by depth_index"
      3. Run: cd Rust && cargo test test_depth_ordering
    Expected Result: Rejected with exact error format matching Python
    Failure Indicators: Accepted, or error message doesn't match Python format
    Evidence: .sisyphus/evidence/task-4-depth-order.txt

  Scenario: Zero values accepted where ge=0 constraint applies
    Tool: Bash
    Preconditions: models.rs compiles
    Steps:
      1. Write test: deserialize JSON with activation_std: 0.0, gradient_l2_norm: 0.0
      2. Assert deserialization succeeds
      3. Run: cd Rust && cargo test test_zero_values_accepted
    Expected Result: Zero values accepted for fields with ge=0 constraint
    Failure Indicators: Zero values rejected
    Evidence: .sisyphus/evidence/task-4-zero-accepted.txt

  Scenario: Optional bias accepted as null
    Tool: Bash
    Steps:
      1. Write test: deserialize ParameterStatistics with bias: null
      2. Assert deserialization succeeds and bias is None
      3. Run: cd Rust && cargo test test_optional_bias
    Expected Result: bias field correctly deserialized as None
    Evidence: .sisyphus/evidence/task-4-optional-bias.txt
  ```

  **Commit**: YES
  - Message: `feat(rust): add serde models with FiniteF64 and validation`
  - Files: `Rust/src/models.rs`, `Rust/src/main.rs`

- [x] 5. MetricsStore (In-Memory Storage Engine)

  **What to do**:
  - Create `./Rust/src/store.rs` implementing the storage engine:
    ```rust
    pub struct MetricsStore {
        runs: Arc<RwLock<HashMap<String, RunData>>>,
        max_runs: usize,
        max_steps_per_run: usize,
    }
    ```
  - Implement `add_metrics(&self, payload: MetricsPayload) -> StepData`:
    1. Acquire write lock
    2. If run_id not in runs AND runs.len() >= max_runs: evict oldest run (by `last_update` timestamp string comparison)
    3. If run_id not in runs: create new RunData with `created_at = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.6f")`, empty steps
    4. **Sanitize layer IDs**: Replace `.` with `/` in every `layer.layer_id` AND every layer ID inside `layer_groups` values (NOT keys)
    5. Create StepData from payload (layers stored as `Vec<serde_json::Value>`)
    6. **Step dedup**: If step with same number exists, REPLACE it (don't append)
    7. Otherwise append and **stable sort** by step number
    8. **Step eviction**: If `steps.len() > max_steps_per_run`, keep only the last `max_steps_per_run` steps
    9. Update `last_update` to current ISO timestamp (even for duplicate steps!)
    10. Return the StepData

  - Implement `get_run(&self, run_id: &str) -> Option<RunData>`
  - Implement `get_all_runs(&self) -> HashMap<String, RunInfo>` (maps run_id → RunInfo with step_count, latest_step)
  - Implement `get_latest_step(&self, run_id: &str) -> Option<StepData>`

  **CRITICAL SEMANTICS** (must match Python exactly):
    - `last_update` is updated on EVERY `add_metrics` call, even for duplicate steps
    - Run eviction picks the run with the OLDEST `last_update` string (lexicographic comparison of ISO timestamps)
    - Step dedup: same `global_step` → REPLACE entire step data, don't increment count
    - Steps sorted ascending by step number (use stable sort)
    - `latest_step` is the LAST element in the sorted steps Vec (highest step number)

  **Important**: Use `tokio::sync::RwLock` (not `std::sync::RwLock`) because we're in an async context. The Python uses `asyncio.Lock` which is equivalent.

  **Important**: Timestamp format must be `chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.6f")` to match Python's `datetime.now().isoformat()` which produces `2024-02-10T12:34:56.789000` (6 decimal places, no timezone suffix).

  **Add `mod store;` to `src/main.rs`.**

  **Must NOT do**:
  - Do NOT use BTreeMap or any optimized data structure — match Python's HashMap + sorted Vec approach
  - Do NOT use unstable sort (`sort_unstable_by_key`)
  - Do NOT update `created_at` on duplicate step submission (only `last_update`)
  - Do NOT skip layer ID sanitization for `layer_groups` values

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex state management, subtle semantics (dedup, eviction, sorting, sanitization), must match Python exactly
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (depends only on Task 1 + Task 4 for types)
  - **Parallel Group**: Wave 1 (with Tasks 3, 4)
  - **Blocks**: Tasks 7, 8, 9, 13
  - **Blocked By**: Task 1, Task 4

  **References**:
  **Pattern References**:
  - `main.py:301-403` — MetricsStore class with EXACT semantics: add_metrics (line 321-380), get_run (382-384), get_all_runs (386-396), get_latest_step (398-403)
  - `main.py:303-309` — `_sanitize_layer_id` function: replaces `.` with `/`
  - `main.py:340-353` — Layer ID sanitization applied to BOTH layer_statistics AND layer_groups values
  - `main.py:366-372` — Step dedup logic: same step → replace; new step → append + sort
  - `main.py:375-376` — Step eviction: keep last max_steps_per_run steps
  - `main.py:378` — last_update always set to now(), even for duplicate steps

  **Test References**:
  - `tests/test_storage/test_step_deduplication.py` (247 lines) — 14 tests verifying: replace behavior, count tracking, sorting
  - `tests/test_storage/test_run_eviction.py` (331 lines) — 16 tests verifying: oldest eviction, step limit, per-run enforcement
  - `tests/fixtures/metric_fixtures.py` — `populate_run` helper showing how to populate store for testing

  **WHY Each Reference Matters**:
  - `main.py:321-380` is the complete add_metrics logic — every line must have a Rust equivalent
  - `main.py:340-353` shows sanitization applied to layer_groups VALUES (not keys) — easy to miss
  - `main.py:366-372` shows dedup is find-by-step-number and replace — not append-and-dedup
  - The eviction/step tests (30 assertions total) define exact behavioral expectations

  **Acceptance Criteria**:
  - [ ] `Rust/src/store.rs` exists and compiles
  - [ ] `MetricsStore::new(max_runs, max_steps_per_run)` creates empty store
  - [ ] `add_metrics` stores step data, sanitizes layer IDs, deduplicates steps
  - [ ] Run eviction works: oldest by last_update evicted when max_runs exceeded
  - [ ] Step eviction works: oldest steps removed when max_steps_per_run exceeded
  - [ ] `cargo test` passes for store unit tests

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Step deduplication replaces existing step
    Tool: Bash
    Preconditions: store.rs compiles
    Steps:
      1. Create store with max_runs=10, max_steps=100
      2. Add metrics for run_id="r1", global_step=100
      3. Add metrics for run_id="r1", global_step=100 (duplicate, different values)
      4. Assert step_count == 1
      5. Assert the stored step has the UPDATED values
      6. Run: cd Rust && cargo test test_step_dedup
    Expected Result: 1 step with updated values
    Failure Indicators: step_count != 1, or values not updated
    Evidence: .sisyphus/evidence/task-5-step-dedup.txt

  Scenario: Run eviction removes oldest by last_update
    Tool: Bash
    Steps:
      1. Create store with max_runs=3
      2. Add metrics for run_id="r1", step=1
      3. Sleep 10ms
      4. Add metrics for run_id="r2", step=1
      5. Sleep 10ms
      6. Add metrics for run_id="r3", step=1
      7. Add metrics for run_id="r4", step=1 (should evict r1)
      8. Assert get_run("r1") returns None
      9. Assert get_run("r2"), get_run("r3"), get_run("r4") return Some
      10. Run: cd Rust && cargo test test_run_eviction
    Expected Result: r1 evicted, r2/r3/r4 present
    Failure Indicators: Wrong run evicted, or no eviction
    Evidence: .sisyphus/evidence/task-5-run-eviction.txt

  Scenario: Layer ID sanitization (dots to slashes)
    Tool: Bash
    Steps:
      1. Add metrics with layer_id "encoder.linear1" and layer_groups {"enc": ["encoder.linear1"]}
      2. Retrieve stored step
      3. Assert layer_id is "encoder/linear1" (slash, not dot)
      4. Assert layer_groups values contain "encoder/linear1"
      5. Run: cd Rust && cargo test test_layer_sanitization
    Expected Result: All dots replaced with slashes in layer IDs and layer_groups values
    Evidence: .sisyphus/evidence/task-5-layer-sanitize.txt
  ```

  **Commit**: YES
  - Message: `feat(rust): add in-memory MetricsStore with dedup and eviction`
  - Files: `Rust/src/store.rs`, `Rust/src/main.rs`

- [x] 6. WebSocket Manager (Broadcast Channel)

  **What to do**:
  - Create `./Rust/src/ws.rs` implementing WebSocket connection management:
    ```rust
    pub struct WsManager {
        tx: broadcast::Sender<String>,  // Fan-out channel
        connections: Arc<AtomicU32>,     // Active connection count
    }
    ```
  - Use `tokio::sync::broadcast::channel(1024)` — capacity 1024 for slow client tolerance
  - Implement:
    - `new() -> Self` — Create broadcast channel
    - `subscribe(&self) -> broadcast::Receiver<String>` — Get a receiver for new connections
    - `broadcast(&self, message: &str)` — Send to all subscribers. Log warning on `RecvError::Lagged` but don't crash.
    - `connect(&self)` — Increment connection counter
    - `disconnect(&self)` — Decrement connection counter
    - `active_count(&self) -> u32` — Read counter (no lock needed, atomic read)

  - Implement per-connection WebSocket handler:
    1. Accept WebSocket upgrade via `WebSocketUpgrade`
    2. On connect: send `initial_runs` message with current runs from MetricsStore
    3. Spawn two tasks per connection (Axum chat pattern):
       - **Send task**: Listen to broadcast receiver, forward to client WebSocket
       - **Receive task**: Read client messages, handle actions:
         - `{"action": "subscribe_run", "run_id": "..."}` → Look up run in store, send `run_history` (or `error` if not found). Support `"lite": true` flag for compact payloads.
         - `{"action": "ping"}` → Send `{"type": "pong"}`
         - Invalid JSON → Send `{"type": "error", "message": "Invalid JSON format"}`
         - Unknown action → Ignore gracefully (no crash, no hang)
    4. Use `tokio::select!` to abort the other task when one finishes
    5. On disconnect: decrement connection counter

  **WebSocket `lite` mode (CRITICAL — frontend depends on this)**:
    - When `subscribe_run` message includes `"lite": true`, return compact run data:
    - Compact layer: only `layer_id`, `layer_type`, `depth_index`, `intermediate_features.activation_std`, `intermediate_features.cross_layer_std_ratio`, `gradient_flow.gradient_l2_norm`, `gradient_flow.gradient_max_abs`
    - Compact step: only `step`, `timestamp`, `batch_size`, `layers` (compact), `cross_layer`, `layer_groups`
    - Compact run: only `created_at`, `last_update`, `steps` (compact)
    - Reference: `main.py:21-58` for exact field lists

  **WebSocket message formats (must match Python exactly)**:
    - `initial_runs`: `{"type": "initial_runs", "data": {run_id: {created_at, last_update, step_count, latest_step}}}`
    - `run_history`: `{"type": "run_history", "run_id": "...", "data": {created_at, last_update, steps: [...]}}`
    - `new_metrics`: `{"type": "new_metrics", "run_id": "...", "data": {step, timestamp, batch_size, layers, cross_layer, layer_groups}}`
    - `pong`: `{"type": "pong"}`
    - `error`: `{"type": "error", "message": "..."}`

  **Add `mod ws;` to `src/main.rs`.**

  **Must NOT do**:
  - Do NOT use `tokio::sync::Mutex` for the connection set — use `AtomicU32` for count
  - Do NOT block on broadcast send — use `try_send` or handle `Lagged` errors
  - Do NOT implement `lite` mode by removing fields from full response — build compact objects separately

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex async WebSocket lifecycle, broadcast pattern, per-connection task spawning, lite mode compact payloads
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (depends only on Cargo.toml)
  - **Parallel Group**: Wave 2 (with Tasks 7, 8, 9, 10)
  - **Blocks**: Tasks 9, 10
  - **Blocked By**: Task 1

  **References**:
  **Pattern References**:
  - `main.py:410-444` — ConnectionManager class: connect, disconnect, broadcast (sequential send under lock, silent failure cleanup)
  - `main.py:514-571` — WebSocket endpoint: full protocol handling (initial_runs, subscribe_run with lite flag, ping/pong, error handling)
  - `main.py:21-58` — Compact payload functions: exact fields to include/exclude in lite mode

  **Test References**:
  - `tests/test_websocket/test_connection_lifecycle.py` (295 lines) — 18 tests for WS lifecycle, message protocol, broadcast

  **External References**:
  - Axum chat example: https://github.com/tokio-rs/axum/blob/main/examples/chat/src/main.rs — Canonical broadcast pattern
  - `tokio::sync::broadcast`: https://docs.rs/tokio/1/#broadcast-channel — Capacity, lagged handling

  **WHY Each Reference Matters**:
  - `main.py:514-571` shows EXACT WebSocket message handling logic, including the `lite` flag check and compact payload generation
  - `main.py:21-58` defines EXACTLY which fields are included in compact/lite mode — the frontend uses `lite: true` by default
  - Axum chat example shows the canonical `socket.split()` + `tokio::select!` pattern for per-connection tasks

  **Acceptance Criteria**:
  - [ ] `Rust/src/ws.rs` exists and compiles
  - [ ] WebSocket connections accepted at `/ws`
  - [ ] `initial_runs` message sent on connect
  - [ ] `subscribe_run` returns `run_history` or `error`
  - [ ] `lite` mode returns compact payloads with correct field subset
  - [ ] `ping` returns `pong`
  - [ ] Broadcast sends `new_metrics` to all connected clients
  - [ ] Disconnected clients don't crash broadcasts

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: WebSocket connection and initial_runs
    Tool: Bash (websocat or python websocket client)
    Preconditions: Server running
    Steps:
      1. Connect to ws://localhost:8000/ws
      2. Read first message
      3. Parse JSON and assert type == "initial_runs"
      4. Assert data is a dict (empty if no runs)
    Expected Result: initial_runs message received immediately on connect
    Evidence: .sisyphus/evidence/task-6-ws-initial.txt

  Scenario: subscribe_run with lite=true returns compact data
    Tool: Bash (python script)
    Steps:
      1. POST metrics to create a run
      2. Connect WebSocket
      3. Send: {"action": "subscribe_run", "run_id": "test", "lite": true}
      4. Read response, assert type == "run_history"
      5. Assert layers do NOT contain "activation_mean" or "parameter_statistics"
    Expected Result: Compact layer data without omitted fields
    Evidence: .sisyphus/evidence/task-6-ws-lite.txt
  ```

  **Commit**: YES
  - Message: `feat(rust): add WebSocket manager with broadcast and lite mode`
  - Files: `Rust/src/ws.rs`, `Rust/src/main.rs`

- [x] 7. REST Routes — POST Metrics + GET Runs

  **What to do**:
  - Create `./Rust/src/routes/` directory with `mod.rs`
  - Create `./Rust/src/routes/metrics.rs`:
    - `POST /api/v1/metrics/layerwise` handler:
      1. Deserialize body as `MetricsPayload` (serde handles validation)
      2. On validation error: return 422 with Pydantic-format `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` — this is a LIST format
      3. On success: call `store.add_metrics(payload)` → get StepData
      4. Broadcast `new_metrics` via WsManager: `{"type": "new_metrics", "run_id": "...", "data": step_data_json}`
      5. Return 202 with `{"status": "accepted", "run_id": "..."}`
      6. On internal error: return 500 with `{"detail": {"error": "internal_error", "message": "Failed to process metrics"}}`

  - Create `./Rust/src/routes/runs.rs`:
    - `GET /api/v1/runs` handler: Return `store.get_all_runs()` as JSON
    - `GET /api/v1/runs/{run_id}` handler: Return `store.get_run(run_id)` or 404 with `{"detail": {"error": "not_found", "message": "Run '{run_id}' not found"}}`

  **DUAL ERROR FORMAT (CRITICAL)**:
    - **422 validation errors**: `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` — a **list** of error objects. This matches Pydantic's format.
    - **400/404/500 business errors**: `{"detail": {"error": "not_found", "message": "..."}}` — a **dict** with `error` and `message` keys.

  **Custom IntoResponse for validation errors**:
    - Write a custom `IntoResponse` impl that transforms `validator::ValidationErrors` (or serde deserialization errors) into the Pydantic-style 422 format with `loc`, `msg`, `type` fields.
    - This is critical for Python contract tests that check error response structure.

  **Add `mod routes;` to `src/main.rs`.**

  **Must NOT do**:
  - Do NOT use Axum's default JSON rejection (produces different error format)
  - Do NOT change error response format from Python's dual format
  - Do NOT add request body size limit checking yet (Task 10 handles this)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multiple route handlers, custom error format matching, tight integration with store and WS manager
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 8, 9, 10)
  - **Blocks**: Tasks 10, 11
  - **Blocked By**: Tasks 3, 4, 5, 6

  **References**:
  **Pattern References**:
  - `main.py:449-479` — POST metrics endpoint: validation, store, broadcast, error handling
  - `main.py:482-497` — GET runs and GET run by ID endpoints: response format, 404 handling
  - `main.py:293-299` — ErrorDetail model for business errors

  **Test References**:
  - `tests/test_rest_api/test_metrics_endpoint.py` (218 lines) — 22 tests for POST endpoint
  - `tests/test_rest_api/test_runs_endpoint.py` (277 lines) — 23 tests for GET endpoints
  - `tests/test_rest_api/test_error_responses.py` (182 lines) — 16 tests for error format (DUAL FORMAT)

  **WHY Each Reference Matters**:
  - `main.py:449-479` shows exact error handling: ValueError → 400, Exception → 500. Rust must replicate status codes and error bodies.
  - `test_error_responses.py` verifies BOTH error formats — 422 list and 404 dict. Contract tests will fail if formats don't match.

  **Acceptance Criteria**:
  - [ ] `POST /api/v1/metrics/layerwise` returns 202 with `{"status": "accepted", "run_id": "..."}` for valid payload
  - [ ] `POST /api/v1/metrics/layerwise` returns 422 with `{"detail": [list]}` for invalid payload
  - [ ] `GET /api/v1/runs` returns 200 with `{run_id: RunInfo}` dict
  - [ ] `GET /api/v1/runs/{id}` returns 200 or 404 with correct error format
  - [ ] After POST, WebSocket receives `new_metrics` broadcast

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Valid metrics accepted
    Tool: Bash (curl)
    Steps:
      1. curl -X POST http://localhost:8000/api/v1/metrics/layerwise -H "Content-Type: application/json" -d '<valid payload from SCHEMA.md example>'
      2. Assert HTTP 202
      3. Assert response body contains {"status": "accepted", "run_id": "experiment_2024_0210_v2"}
    Expected Result: 202 Accepted with correct response body
    Evidence: .sisyphus/evidence/task-7-post-valid.txt

  Scenario: Invalid payload returns Pydantic-format 422
    Tool: Bash (curl)
    Steps:
      1. curl -X POST http://localhost:8000/api/v1/metrics/layerwise -H "Content-Type: application/json" -d '{"metadata":{}}'
      2. Assert HTTP 422
      3. Assert response body has "detail" key containing a LIST
      4. Assert first element has "loc" and "msg" keys
    Expected Result: 422 with Pydantic error list format
    Failure Indicators: detail is a dict (wrong format), or missing loc field
    Evidence: .sisyphus/evidence/task-7-post-invalid.txt

  Scenario: GET runs returns correct structure
    Tool: Bash (curl)
    Steps:
      1. POST valid metrics for run_id="test_run"
      2. curl http://localhost:8000/api/v1/runs
      3. Assert 200
      4. Assert response is {"test_run": {"created_at": "...", "last_update": "...", "step_count": 1, "latest_step": 1500}}
    Expected Result: 200 with run info dict
    Evidence: .sisyphus/evidence/task-7-get-runs.txt

  Scenario: Nonexistent run returns 404 with dict error
    Tool: Bash (curl)
    Steps:
      1. curl http://localhost:8000/api/v1/runs/nonexistent
      2. Assert HTTP 404
      3. Assert response body: {"detail": {"error": "not_found", "message": "Run 'nonexistent' not found"}}
    Expected Result: 404 with dict-format error (NOT list format)
    Evidence: .sisyphus/evidence/task-7-not-found.txt
  ```

  **Commit**: YES
  - Message: `feat(rust): add REST routes for POST metrics and GET runs`
  - Files: `Rust/src/routes/mod.rs`, `Rust/src/routes/metrics.rs`, `Rust/src/routes/runs.rs`

- [x] 8. REST Routes — GET Latest + GET Health

  **What to do**:
  - Create `./Rust/src/routes/health.rs`:
    - `GET /health` handler: Return `{"status": "healthy", "active_connections": N}` where N is `ws_manager.active_count()`
  - Add to `routes/runs.rs`:
    - `GET /api/v1/runs/{run_id}/latest` handler: Return `store.get_latest_step(run_id)` or 404 with same dict-format error

  **Must NOT do**:
  - Do NOT acquire a lock to read `active_connections` — use atomic read (matches Python's lockless read)
  - Do NOT change health response format from `{"status": "healthy", "active_connections": N}`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multiple route handlers, integration with store and WS manager
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 9, 10)
  - **Blocks**: Tasks 10, 11
  - **Blocked By**: Tasks 3, 4, 5

  **References**:
  **Pattern References**:
  - `main.py:500-509` — GET latest endpoint: 200 or 404
  - `main.py:593-595` — GET health: returns active_connections count

  **Test References**:
  - `tests/test_rest_api/test_runs_endpoint.py` — Tests for GET latest endpoint

  **Acceptance Criteria**:
  - [ ] `GET /health` returns 200 with `{"status": "healthy", "active_connections": N}`
  - [ ] `GET /api/v1/runs/{id}/latest` returns 200 with latest StepData
  - [ ] `GET /api/v1/runs/{id}/latest` returns 404 for nonexistent run

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Health endpoint returns correct structure
    Tool: Bash (curl)
    Steps:
      1. curl http://localhost:8000/health
      2. Assert 200
      3. Assert response has "status" == "healthy" and "active_connections" is integer >= 0
    Expected Result: {"status": "healthy", "active_connections": 0}
    Evidence: .sisyphus/evidence/task-8-health.txt

  Scenario: Latest step returns most recent
    Tool: Bash (curl)
    Steps:
      1. POST metrics for run_id="r1", step=100
      2. POST metrics for run_id="r1", step=200
      3. curl http://localhost:8000/api/v1/runs/r1/latest
      4. Assert step == 200
    Expected Result: Returns step 200 (the latest)
    Evidence: .sisyphus/evidence/task-8-latest.txt
  ```

  **Commit**: YES (groups with Task 7)
  - Message: `feat(rust): add REST routes for GET latest, GET health`
  - Files: `Rust/src/routes/health.rs`, `Rust/src/routes/runs.rs`

- [x] 9. WebSocket Endpoint (Full Protocol)

  **What to do**:
  - Create `./Rust/src/routes/ws_route.rs`:
    - Wire up the WebSocket handler from `src/ws.rs` to Axum's `WebSocketUpgrade` extractor
    - `GET /ws` handler: Accept upgrade, call `ws::handle_socket()`
  - Ensure all WebSocket message types work end-to-end:
    - On connect → `initial_runs` (from store.get_all_runs())
    - `subscribe_run` → `run_history` (full or compact based on `lite` flag)
    - `ping` → `pong`
    - Invalid JSON → `error`
    - Unknown action → ignore (no crash, no hang under 2 seconds)
  - Ensure `new_metrics` broadcasts work when POST endpoint receives metrics

  **Must NOT do**:
  - Do NOT change message type strings from Python's values
  - Do NOT add new WebSocket message types

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Integrating WS manager with Axum routing, testing full protocol, handling edge cases
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 8, 10)
  - **Blocks**: Tasks 10, 12
  - **Blocked By**: Tasks 4, 5, 6

  **References**:
  **Pattern References**:
  - `main.py:514-571` — Complete WebSocket endpoint with all message handling

  **Test References**:
  - `tests/test_websocket/test_connection_lifecycle.py` (295 lines) — 18 protocol tests

  **Acceptance Criteria**:
  - [ ] WebSocket connects at `/ws` and receives `initial_runs`
  - [ ] `subscribe_run` returns `run_history` for existing run
  - [ ] `subscribe_run` returns `error` for nonexistent run
  - [ ] `ping` returns `pong`
  - [ ] Invalid JSON returns `error` message
  - [ ] `new_metrics` broadcast received by all connected clients

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Full WebSocket protocol flow
    Tool: Bash (python websocket client script)
    Steps:
      1. POST metrics for run_id="flow_test", step=100
      2. Connect to ws://localhost:8000/ws
      3. Read initial message, assert type == "initial_runs"
      4. Send: {"action": "subscribe_run", "run_id": "flow_test"}
      5. Read response, assert type == "run_history" and run_id == "flow_test"
      6. Send: {"action": "ping"}
      7. Read response, assert type == "pong"
      8. Send: "not json"
      9. Read response, assert type == "error" and message contains "JSON"
    Expected Result: All message types work correctly
    Evidence: .sisyphus/evidence/task-9-ws-protocol.txt

  Scenario: Broadcast received by multiple clients
    Tool: Bash (python script)
    Steps:
      1. Connect 2 WebSocket clients
      2. POST metrics via HTTP
      3. Assert BOTH clients receive new_metrics within 2 seconds
    Expected Result: Both clients get broadcast
    Evidence: .sisyphus/evidence/task-9-ws-broadcast.txt
  ```

  **Commit**: YES
  - Message: `feat(rust): add WebSocket route with full protocol`
  - Files: `Rust/src/routes/ws_route.rs`

- [x] 10. main.rs — Server Entry + Static Files + Graceful Shutdown

  **What to do**:
  - Implement `./Rust/src/main.rs` to wire everything together:
    ```rust
    #[tokio::main]
    async fn main() {
        // 1. Load config from env vars
        // 2. Initialize MetricsStore
        // 3. Initialize WsManager
        // 4. Build Axum Router:
        //    - POST /api/v1/metrics/layerwise
        //    - GET /api/v1/runs
        //    - GET /api/v1/runs/{run_id}
        //    - GET /api/v1/runs/{run_id}/latest
        //    - GET /health
        //    - GET /ws (WebSocket)
        //    - GET / → serve static/index.html
        //    - Static files: /static/* → serve Rust/static/
        //    - Fallback: serve index.html for SPA routing
        // 5. Print startup banner (matching Python's output format)
        // 6. Bind to config.host:config.port
        // 7. Run with graceful shutdown (Ctrl+C)
    }
    ```
  - Use `tower_http::services::ServeDir::new("static")` for static files
  - Use `tower_http::services::ServeFile::new("static/index.html")` for root route
  - Print startup banner matching Python format:
    ```
    ==================================================
    NN Training Monitor Server (Rust)
    ==================================================
    Server started successfully!
    Host: {host}
    Port: {port}
    ...
    ```
  - Implement graceful shutdown with `tokio::signal::ctrl_c()`

  **Important**: Static file paths must be relative to the working directory where the server is started. The user runs `cd Rust && cargo run` so `static/` is relative to `Rust/`.

  **Must NOT do**:
  - Do NOT add CORS middleware
  - Do NOT change default port from 8000
  - Do NOT add request body size limiting yet (can be a follow-up)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Ties all modules together, static file serving config, startup/shutdown logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO — depends on all other Wave 2 tasks
  - **Parallel Group**: Sequential (end of Wave 2)
  - **Blocks**: Tasks 14, 15
  - **Blocked By**: Tasks 2, 3, 7, 8, 9

  **References**:
  **Pattern References**:
  - `main.py:574-612` — Static file serving, root route, startup banner
  - `main.py:598-612` — Startup print statements with exact format

  **External References**:
  - tower-http ServeDir: https://docs.rs/tower-http/0.6 — Static file serving
  - Axum graceful shutdown: https://docs.rs/axum/0.8 — `axum::serve().with_graceful_shutdown()`

  **Acceptance Criteria**:
  - [ ] Server starts and prints banner
  - [ ] `GET /` returns `static/index.html`
  - [ ] `GET /static/app.js` returns the JS file
  - [ ] All REST and WebSocket routes accessible
  - [ ] `Ctrl+C` triggers graceful shutdown

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Server starts and serves frontend
    Tool: Bash
    Steps:
      1. cd Rust && cargo run --release &
      2. Sleep 3 seconds
      3. curl -sf http://localhost:8000/ | head -5
      4. Assert HTML content returned (contains "<!DOCTYPE html>" or "<html>")
      5. curl -sf http://localhost:8000/static/app.js | head -3
      6. Assert JavaScript content returned
      7. kill %1
    Expected Result: Frontend HTML and JS served correctly
    Evidence: .sisyphus/evidence/task-10-frontend.txt

  Scenario: All endpoints accessible
    Tool: Bash
    Steps:
      1. Start server
      2. curl -sf http://localhost:8000/health | jq '.status'
      3. Assert "healthy"
      4. curl -sf http://localhost:8000/api/v1/runs
      5. Assert returns {}
      6. kill %1
    Expected Result: Health and runs endpoints respond
    Evidence: .sisyphus/evidence/task-10-endpoints.txt
  ```

  **Commit**: YES
  - Message: `feat(rust): wire up main.rs with static files and graceful shutdown`
  - Files: `Rust/src/main.rs`

- [x] 11. Rust Integration Tests — REST Endpoints

  **What to do**:
  - Create `./Rust/tests/test_metrics_endpoint.rs`:
    - Test POST with valid payloads (minimal, 3-layer, vanishing, exploding, healthy patterns)
    - Test POST with invalid payloads: missing fields, wrong types, NaN, Infinity, negative values, empty arrays, depth_index ordering
    - Verify 202 status + `{"status": "accepted", "run_id": "..."}` for valid
    - Verify 422 status + `{"detail": [list]}` for validation errors
    - Use `axum-test` or spawn test server with `tokio::net::TcpListener`
  - Create `./Rust/tests/test_runs_endpoint.rs`:
    - Test GET /api/v1/runs (empty, populated, multiple runs)
    - Test GET /api/v1/runs/{run_id} (existing, nonexistent)
    - Test GET /api/v1/runs/{run_id}/latest (existing, nonexistent, single step, after duplicate)
    - Test GET /health (response structure, active_connections type)
  - Create `./Rust/tests/test_error_responses.rs`:
    - Verify DUAL error format: 422 returns `detail` as list, 404 returns `detail` as dict
    - Verify error field names: `loc`, `msg`, `type` for 422; `error`, `message` for 404
    - Verify not_found message includes run_id
    - Verify content-type is `application/json`

  **Reference for test data**: Extract JSON payloads from Python fixtures (`tests/fixtures/metric_fixtures.py`, `tests/fixtures/error_fixtures.py`) as string constants in Rust test files. These define the exact valid/invalid inputs.

  **Must NOT do**:
  - Do NOT use Python test framework — these are pure Rust tests
  - Do NOT skip the dual error format verification

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Many test cases, careful verification of response formats, JSON payload construction
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 12, 13, 14, 15)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 7, 8

  **References**:
  **Test References**:
  - `tests/test_rest_api/test_metrics_endpoint.py` (218 lines) — 22 test cases to replicate
  - `tests/test_rest_api/test_runs_endpoint.py` (277 lines) — 23 test cases to replicate
  - `tests/test_rest_api/test_error_responses.py` (182 lines) — 16 test cases to replicate
  - `tests/fixtures/metric_fixtures.py` (1220 lines) — Valid JSON payloads as test input data
  - `tests/fixtures/error_fixtures.py` (1064 lines) — Invalid JSON payloads as test input data

  **WHY Each Reference Matters**:
  - Each Python test file defines exact assertions that must have Rust equivalents
  - The fixture files contain deterministic JSON that can be used as string constants

  **Acceptance Criteria**:
  - [ ] `cd Rust && cargo test --test test_metrics_endpoint` passes with >= 20 test cases
  - [ ] `cd Rust && cargo test --test test_runs_endpoint` passes with >= 20 test cases
  - [ ] `cd Rust && cargo test --test test_error_responses` passes with >= 10 test cases
  - [ ] Dual error format verified: 422=list, 404=dict

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All REST tests pass
    Tool: Bash
    Steps:
      1. cd Rust && cargo test --test test_metrics_endpoint --test test_runs_endpoint --test test_error_responses 2>&1
      2. Assert "test result: ok" in output
      3. Count passed tests >= 50
    Expected Result: All tests pass
    Failure Indicators: Any test failure
    Evidence: .sisyphus/evidence/task-11-rest-tests.txt
  ```

  **Commit**: YES
  - Message: `test(rust): add REST endpoint integration tests`
  - Files: `Rust/tests/test_metrics_endpoint.rs`, `Rust/tests/test_runs_endpoint.rs`, `Rust/tests/test_error_responses.rs`

- [x] 12. Rust Integration Tests — WebSocket Protocol

  **What to do**:
  - Create `./Rust/tests/test_websocket.rs`:
    - Test connection lifecycle: connect → initial_runs, disconnect
    - Test ping/pong: send ping → receive pong
    - Test subscribe_run: existing run → run_history, nonexistent → error
    - Test lite mode: subscribe with `"lite": true` → compact payload (verify missing fields)
    - Test invalid JSON → error message
    - Test new_metrics broadcast: POST metrics → WS receives new_metrics
    - Test multiple clients: both receive broadcast
    - Test disconnected client: remaining clients still receive broadcast

  **Important**: WebSocket tests need a running test server. Use `axum::Router` with test state, bind to `TcpListener::bind("127.0.0.1:0")` for random port.

  **Must NOT do**:
  - Do NOT skip lite mode tests — frontend depends on it

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Async WebSocket testing, message timing, multi-client coordination
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 13, 14, 15)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 9

  **References**:
  **Test References**:
  - `tests/test_websocket/test_connection_lifecycle.py` (295 lines) — 18 test cases
  - `tests/test_integration/test_concurrent_clients.py` (363 lines) — WebSocket-specific tests
  - `tests/test_integration/test_time_travel.py` (395 lines) — History and subscribe tests

  **Acceptance Criteria**:
  - [ ] `cd Rust && cargo test --test test_websocket` passes with >= 15 test cases
  - [ ] Lite mode verified: compact payloads lack activation_mean, parameter_statistics
  - [ ] Broadcast verified: multiple clients receive same message

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: WebSocket tests pass
    Tool: Bash
    Steps:
      1. cd Rust && cargo test --test test_websocket 2>&1
      2. Assert "test result: ok"
      3. Count >= 15 tests passed
    Expected Result: All WS tests pass
    Evidence: .sisyphus/evidence/task-12-ws-tests.txt
  ```

  **Commit**: YES
  - Message: `test(rust): add WebSocket protocol integration tests`
  - Files: `Rust/tests/test_websocket.rs`

- [x] 13. Rust Unit Tests — Storage (Dedup + Eviction)

  **What to do**:
  - Create `./Rust/tests/test_storage.rs`:
    - **Step deduplication tests** (matching Python's 14 tests):
      - Duplicate step replaces existing data
      - Duplicate doesn't increase count
      - Duplicate updates latest_step
      - Duplicate among other steps
      - Multiple duplicates (last wins)
      - Steps sorted by step number
      - Steps sorted after duplicate in middle
      - Step count increments with new steps
      - Step count unchanged by duplicate
      - Count across multiple runs
    - **Run eviction tests** (matching Python's 16 tests):
      - Oldest run evicted when max_runs exceeded
      - Eviction based on last_update (not created_at)
      - Eviction via HTTP (end-to-end)
      - Eviction preserves newest runs
      - Eviction doesn't corrupt remaining data
      - Simultaneous timestamps handled
      - Oldest step evicted when max_steps exceeded
      - Per-run enforcement
      - Via HTTP
      - Preserves latest steps
      - Duplicate doesn't count toward limit
    - **Layer sanitization tests**:
      - Dots replaced with slashes in layer_id
      - Dots replaced with slashes in layer_groups values
      - Layer_groups keys NOT sanitized

  **Must NOT do**:
  - Do NOT test implementation details — test behavioral outcomes only

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Many test cases with precise behavioral expectations, timing-sensitive eviction tests
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 12, 14, 15)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 5

  **References**:
  **Test References**:
  - `tests/test_storage/test_step_deduplication.py` (247 lines) — 14 dedup tests with exact assertions
  - `tests/test_storage/test_run_eviction.py` (331 lines) — 16 eviction tests with exact assertions

  **Acceptance Criteria**:
  - [ ] `cd Rust && cargo test --test test_storage` passes with >= 25 test cases
  - [ ] Dedup: same step → replace, count unchanged
  - [ ] Eviction: oldest by last_update removed

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Storage tests pass
    Tool: Bash
    Steps:
      1. cd Rust && cargo test --test test_storage 2>&1
      2. Assert "test result: ok"
    Expected Result: All storage tests pass
    Evidence: .sisyphus/evidence/task-13-storage-tests.txt
  ```

  **Commit**: YES
  - Message: `test(rust): add storage unit tests for dedup and eviction`
  - Files: `Rust/tests/test_storage.rs`

- [x] 14. Python Contract Test Verification

  **What to do**:
  - Start the Rust server: `cd Rust && cargo run --release`
  - Run Python test clients against it:
    ```bash
    # Basic test client
    python3 Rust/test_client.py --run-id contract_rust --steps 50 --interval 0.5 --endpoint http://localhost:8000/api/v1/metrics/layerwise

    # Comprehensive test client (various scenarios)
    python3 Rust/test_client_comprehensive.py --endpoint http://localhost:8000/api/v1/metrics/layerwise
    ```
  - Verify each test client completes without errors
  - Capture output as evidence
  - If `test_client_brutal.py` has hardcoded endpoint (no `--endpoint` flag), patch it to point to localhost:8000 or note it as a known limitation

  **Must NOT do**:
  - Do NOT modify the Python test client logic — only the endpoint URL
  - Do NOT skip this step — it's the primary contract verification

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Running external processes, capturing output, diagnosing failures
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 12, 13, 15)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 10

  **References**:
  **Pattern References**:
  - `test_client.py` (189 lines) — Has `--endpoint` CLI arg, works with any server
  - `test_client_brutal.py` (375 lines) — Check for `--endpoint` arg; if missing, note in evidence
  - `test_client_comprehensive.py` (312 lines) — Has `--endpoint` CLI arg

  **Acceptance Criteria**:
  - [ ] `test_client.py` runs successfully against Rust server (all steps show "Metrics sent successfully")
  - [ ] `test_client_comprehensive.py` runs without errors
  - [ ] Output captured as evidence

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Python test_client.py succeeds against Rust server
    Tool: Bash
    Preconditions: Rust server running on port 8000
    Steps:
      1. cargo run --release (in Rust/) &
      2. sleep 3
      3. python3 Rust/test_client.py --run-id contract_test --steps 30 --interval 0.3 --endpoint http://localhost:8000/api/v1/metrics/layerwise
      4. Check output contains "Training simulation complete!"
      5. Check no "Error" or "Failed" messages
    Expected Result: All 3 steps sent successfully
    Failure Indicators: Any "Error" or "Failed" messages in output
    Evidence: .sisyphus/evidence/task-14-python-contract.txt

  Scenario: Python test_client_comprehensive.py succeeds
    Tool: Bash
    Steps:
      1. python3 Rust/test_client_comprehensive.py --endpoint http://localhost:8000/api/v1/metrics/layerwise
      2. Check output for successful completions
    Expected Result: All scenarios complete without HTTP errors
    Evidence: .sisyphus/evidence/task-14-comprehensive.txt
  ```

  **Commit**: YES
  - Message: `test(rust): verify Python contract tests pass against Rust server`
  - Files: No new files (verification only, evidence in .sisyphus/evidence/)

- [x] 15. Frontend Integration Verification (Playwright)

  **What to do**:
  - Start the Rust server: `cd Rust && cargo run --release`
  - Use Playwright to:
    1. Navigate to `http://localhost:8000/`
    2. Verify page loads (contains "NN Training Monitor" or similar title)
    3. Verify WebSocket connects (check for connection indicator or absence of error)
    4. POST metrics via curl while browser is open
    5. Verify dashboard updates (canvas element renders, or no error visible)
    6. Take screenshot as evidence

  **Must NOT do**:
  - Do NOT modify any frontend code
  - Do NOT write new frontend tests — just verify existing frontend works with Rust backend

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Playwright automation, visual verification, debugging UI issues
  - **Skills**: [`playwright`]
    - `playwright`: Browser automation for frontend verification

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 12, 13, 14)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 10

  **References**:
  **Pattern References**:
  - `static/index.html` (122 lines) — Dashboard HTML structure
  - `static/app.js` (1202 lines) — WebSocket connection at line ~`new WebSocket("ws://"+location.host+"/ws")`
  - `docs/FEATURES.md` — Time travel feature, UI elements

  **Acceptance Criteria**:
  - [ ] Dashboard loads at `http://localhost:8000/`
  - [ ] WebSocket connects (no error in browser console)
  - [ ] Metrics appear in dashboard after POST
  - [ ] Screenshot captured as evidence

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Frontend loads and WebSocket connects
    Tool: Playwright
    Preconditions: Rust server running on port 8000
    Steps:
      1. Navigate to http://localhost:8000/
      2. Wait for page to load (timeout: 5s)
      3. Assert page title contains "NN" or "Monitor"
      4. Wait 2 seconds for WebSocket connection
      5. Assert no visible error messages
      6. Take screenshot
    Expected Result: Dashboard renders without errors
    Failure Indicators: Page shows error, blank page, WebSocket error
    Evidence: .sisyphus/evidence/task-15-frontend-load.png

  Scenario: Metrics appear after POST
    Tool: Playwright + Bash (curl)
    Steps:
      1. Navigate to http://localhost:8000/
      2. POST metrics via curl (valid payload for run_id="visual_test")
      3. Wait 2 seconds
      4. Assert canvas or chart element is visible
      5. Take screenshot
    Expected Result: Visualization renders with metric data
    Evidence: .sisyphus/evidence/task-15-frontend-metrics.png
  ```

  **Commit**: YES
  - Message: `test(rust): verify frontend integration with Playwright`
  - Files: No new files (verification only)

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `cd Rust && cargo build --release` + `cargo clippy -- -D warnings` + `cargo test`. Review all Rust files for: `unwrap()` in production paths (should use `?` or `map_err`), `todo!()`/`unimplemented!()`, commented-out code, unused imports. Check for proper error handling throughout. Verify no `as f64` casts that could introduce NaN.
  Output: `Build [PASS/FAIL] | Clippy [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill if UI)
  Start Rust server from clean state (`cd Rust && cargo run --release`). Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (POST metrics → WebSocket broadcast → frontend renders). Test edge cases: empty state, invalid input, rapid submissions. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect unaccounted changes. Specifically verify: no CORS middleware added, no auth added, no persistence added, no new endpoints.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `feat(rust): initialize project scaffolding and copied assets` — Cargo.toml, static/, docs, test clients
- **Wave 1**: `feat(rust): add config, models, and storage engine` — src/config.rs, src/models.rs, src/store.rs
- **Wave 2**: `feat(rust): implement REST routes and WebSocket handler` — src/routes/, src/ws.rs, src/main.rs
- **Wave 3**: `test(rust): add integration and contract tests` — tests/, verification
- **FINAL**: `chore(rust): final review and cleanup` — any fixes from review

---

## Success Criteria

### Verification Commands
```bash
cd Rust && cargo build --release 2>&1 | tail -1  # Expected: "Finished release"
cd Rust && cargo test 2>&1 | tail -3  # Expected: "test result: ok. N passed; 0 failed"
cd Rust && cargo run --release &
sleep 2
curl -sf http://localhost:8000/health | jq '.status'  # Expected: "healthy"
curl -sf -X POST http://localhost:8000/api/v1/metrics/layerwise \
  -H "Content-Type: application/json" \
  -d '{"metadata":{"run_id":"contract_test","timestamp":1.0,"global_step":0,"batch_size":32},"layer_statistics":[{"layer_id":"l1","layer_type":"Linear","depth_index":0,"intermediate_features":{"activation_std":0.5,"activation_mean":0.1,"activation_shape":[32,64]},"gradient_flow":{"gradient_l2_norm":0.1,"gradient_std":0.05,"gradient_max_abs":0.2},"parameter_statistics":{"weight":{"std":0.1,"mean":0.0,"spectral_norm":1.0,"frobenius_norm":0.5}}}],"cross_layer_analysis":{"feature_std_gradient":0.01}}' \
  | jq '.status'  # Expected: "accepted"
python3 Rust/test_client.py --endpoint http://localhost:8000/api/v1/metrics/layerwise  # Expected: success messages
kill %1
```

### Final Checklist
- [ ] All "Must Have" present (15 items)
- [ ] All "Must NOT Have" absent (10 items)
- [ ] All Rust tests pass (`cargo test`)
- [ ] Python contract tests pass against Rust server
- [ ] Frontend loads and connects WebSocket
- [ ] Memory usage lower than Python version
