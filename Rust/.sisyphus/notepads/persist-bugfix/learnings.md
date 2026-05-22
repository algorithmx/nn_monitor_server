
## Persist Bugfix Round 2 (2026-05-20)

### Pattern: Lazy-load delegation via `get_run()`
Methods that access run data must delegate to `get_run()` for lazy-loading from disk, not check `state.runs.contains_key()` directly. This pattern was already applied to `get_run_json()` and `get_all_runs_json()` but missed in `build_run_history_message()` and `get_latest_step_json()`.

### Pattern: `get_run()` populates caches
`get_run()` calls `refresh_run_cache()` which populates `latest_step_json`, `run_json`, and history message caches. After a `get_run()` call, the caches are ready — no need to rebuild them manually.

### Dedup strategy for JSONL load
After sorting steps by step number, `dedup_by` keeps the FIRST occurrence. For JSONL replay, we want the LAST (most recent write). Solution: iterate and replace when adjacent duplicates found.

### `get_all_runs()` vs `get_all_runs_json()` parity
Both methods need disk scanning when persist is enabled. Use `combined.entry(meta.run_id).or_insert(...)` to prefer in-memory data over disk metadata.
