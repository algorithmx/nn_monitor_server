# F3: Real Manual QA Results

**Date:** 2026-05-21  
**Executor:** F3 agent (QA)

## Scenario Results

| # | Scenario | Test | Result |
|---|----------|------|--------|
| 1 | Body limit 413 | `test_body_too_large_rejected_before_parsing_413` | ✓ PASS |
| 2 | CORS headers | `test_cors_headers_present_on_get_runs` | ✓ PASS |
| 3 | WS heartbeat | `test_ws_heartbeat_ping_received` | ✓ PASS (30s) |
| 4 | Persist buffer cap | `test_buffer_cap_limits_size` | ✓ PASS |
| 5a | Ingest queue full | `test_queue_full_returns_full_error` | ✓ PASS |
| 5b | Ingest queue closed | `test_queue_closed_returns_closed_error` | ✓ PASS |
| 5c | Ingest stats | `test_ingest_stats_counters` | ✓ PASS |
| 6 | Full WS suite | `cargo test --test test_websocket` | ✓ 13/13 PASS |
| 7 | Full suite | `cargo test` | ✓ 259/259 PASS |
| 8 | Server startup | `timeout 4 cargo run` with `RUST_LOG=info` | ✓ Clean startup, full banner |

## Startup Output (with RUST_LOG=info)

```
NN Training Monitor Server (Rust)
Server started successfully!
Host: 0.0.0.0
Port: 8000
Access URL: http://localhost:8000
WebSocket endpoint: ws://localhost:8000/ws
Max concurrent runs: 10
Max steps per run: 1000
Data directory: ./data
Flush timeout: 300s
Ingest queue size: 4096
```

## Notes
- Default log level is "warning" — needs `RUST_LOG=info` to see INFO-level banner
- All 259 tests pass across 7 test suites
- Zero warnings from `cargo check`
