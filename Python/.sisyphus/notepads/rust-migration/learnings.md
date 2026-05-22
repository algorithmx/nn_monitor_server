# Learnings

## 2026-05-14 Session Start
- Rust project target: ./Rust/ (empty, confirmed)
- Framework: Axum 0.8 + tokio
- Validation: serde + validator crate + custom FiniteF64 newtype
- WebSocket: tokio::sync::broadcast(1024) for fan-out
- Config: envy + dotenvy with NN_MONITOR_ prefix
- Key traps: serde NaN→null, dual error formats, WS lite mode, timestamp format
