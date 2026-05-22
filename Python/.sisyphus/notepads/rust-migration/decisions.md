# Decisions

## 2026-05-14
- Framework: Axum (user chose over Actix-web)
- Test strategy: Rust tests + Python contract tests (user chose)
- No CORS middleware (Python doesn't use it)
- No auth, no persistence, no new features
- FiniteF64 custom type for NaN/Inf rejection
- Timestamp: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.6f") to match Python
- Active connections: AtomicU32 (matches Python's lockless read)
- broadcast::channel(1024) for WS fan-out
