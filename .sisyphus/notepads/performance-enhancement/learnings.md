# Performance Enhancement - Learnings

## 2026-05-14 Session Start
- Dual-path data design: raw Value + typed MetricsPayload both passed to add_metrics
- build_step_data reads typed fields (step, timestamp, batch_size) from payload AND raw opaque fields (layers, cross_layer) from raw Value
- StepData.layers is Vec<serde_json::Value> — opaque passthrough with only layer_id mutated (dots→slashes)
- StepData.cross_layer is serde_json::Value — raw cloned from input
- MetricsPayload is Deserialize-only (no Serialize)
- FiniteF64/NonNegativeF64 serialize as plain f64 — json! macro handles transparently
- hashbrown::HashMap serde behavior is identical to std::collections::HashMap
- Task 1 complete: removed validator & tokio-util (dead deps), added hashbrown = "0.15", added [profile.release] (lto=thin, codegen-units=1, opt-level=3, strip=true)
- Steps are always kept sorted ascending by step number (binary_search safe)
- MetricsStore is behind Arc<MetricsStore> in AppState — inner Arc removal safe
- make_payload_and_raw helper returns (MetricsPayload, serde_json::Value) — 30 call sites
- 38 total add_metrics call sites to update (34 unit + 3 integration + 1 production)

## 2026-05-14 Task 4: 5 Optimizations Applied to store.rs
- hashbrown::HashMap used for internal storage; get_all_runs() converts to std::collections::HashMap via .collect() for API boundary compatibility with ws.rs
- Added `use std::collections::HashMap as StdHashMap;` for the public return type only
- binary_search_by_key replaces linear iter_mut().find() + sort — O(log n) insert, no post-insert sort needed
- binary_search moves step_data into exactly one match arm (Ok or Err), so clone before the match to preserve return value
- Inner Arc removed: MetricsStore already wrapped in Arc<AppState>, so `runs: RwLock<HashMap>` instead of `Arc<RwLock<HashMap>>`
- After removing Arc from outer scope, tests that use Arc<MetricsStore> need `use std::sync::Arc;` inside test module
- Cow<'_, str> for sanitize_layer_id avoids allocation when no dots present; json! macro handles Cow<str> transparently
- layer_groups map needs `.into_owned()` to convert Cow to String for Vec<String>
- now_iso() reduction: track `is_new_run` flag, only call now_iso() for existing runs; new runs reuse the same timestamp for created_at and last_update
