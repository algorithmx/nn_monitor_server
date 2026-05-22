# Decisions - jsonl-persistence

## 2026-05-20 Planning Phase

- **Format**: One `{run_id}.jsonl` file per experiment
- **Write strategy**: Buffer in memory, flush on experiment-end (gap detection) or shutdown
- **Gap detection**: Configurable timeout, default 300s
- **Startup**: Lazy load — scan metadata (first/last lines), load full history on demand
- **Eviction**: Keep JSONL files on disk when runs evicted from memory
- **Corrupt lines**: Skip with `tracing::warn!`
- **max_steps_per_run**: Applied to lazy-loaded data too (load last N)
- **Sanitization**: At persistence boundary only, NOT in validate()
- **No new dependencies**: Use tokio::fs, serde_json, std::io
