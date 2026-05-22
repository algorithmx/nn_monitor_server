# Learnings - jsonl-persistence

## 2026-05-20 Initial Analysis

### Codebase Patterns
- **Config pattern**: `default_*()` functions + `#[serde(default = "...")]` on fields in `ServerConfig` (config.rs:3-33)
- **Config loading**: `envy::prefixed("NN_MONITOR_").from_env()` (config.rs:58)
- **Store pattern**: `MetricsStore` wraps `RwLock<StoreState>` with read-then-write cache pattern (store.rs:27-31)
- **Test pattern**: `make_payload()` helper in `#[cfg(test)] mod tests` (store.rs:389)
- **Async pattern**: All store methods are async, use `tokio::test` for tests
- **Ingest**: Single `spawn_worker` with mpsc channel (ingest.rs:83-95)

### Key Observations
- `StepData` and `RunData` have `#[derive(Debug, Clone, Serialize)]` — NO Deserialize
- All nested types (LayerStatistic, CrossLayerAnalysis, IntermediateFeatures, etc.) already have Deserialize
- `run_id` is user input validated only for non-empty (models.rs:166)
- `FiniteF64` and `NonNegativeF64` have custom Serialize/Deserialize impls
- Store uses `hashbrown::HashMap` internally, `StdHashMap` for public API
- `Cargo.toml` has `tokio = { version = "1", features = ["full"] }` — includes tokio::fs

### Constraints
- No new crate dependencies
- No API contract changes
- No WebSocket format changes
- No file I/O in ingest worker hot path

## F4 Scope Fidelity Check (2026-05-20)

### Key Findings
- All 6 tasks are 1:1 compliant with their specs
- JSON format preserved: `OwnedRunInfo` produces identical shape to old `RunInfoRef` (`step_count`/`latest_step` fields)
- REST endpoint bugfix (get_run_json/get_all_runs_json using lazy-load) is within Task 4 scope
- Missing 1 test out of 10: `test_flush_on_timeout` (background flush timing test)
- No cross-task contamination detected
- No unaccounted changes (static files are pre-existing)
- Cargo.toml has zero diff — no new dependencies added
- sanitize_filename in models.rs correctly used only at persistence boundary, not in validate()

### Architecture Pattern
- Builder pattern for MetricsStore: `MetricsStore::new(...).with_persist(Arc<clone>)` — clean separation
- `Option<Arc<JsonlStore>>` allows MetricsStore to work with or without persistence
- All persist operations are non-blocking on ingest hot path (buffer_step = memory insert only)
