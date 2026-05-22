
## Task 5: README.md Python Reference Removal

### Changes Made
- Line 3: "FastAPI backend" → "an Axum backend"
- Line 42: `ServerConfig` reference "(main.py:21-38)" → "(config.rs:35-60)", removed "using pydantic-settings"
- Line 43: `MetricsStore` reference "(main.py:250-326)" → "(store.rs:161-346)"
- Line 44: `ConnectionManager` → `WsManager` (matches actual Rust struct), ref "(main.py:334-366)" → "(ws.rs:17-66)"
- Line 45: "Pydantic models" → "Serde models", ref "(main.py:47-237)" → "(models.rs:96-237)"
- Line 66: "async with self._lock for thread safety" → "tokio::sync::RwLock for async safety"
- Added `cargo test` to Testing section (line 19)
- Replaced "no database persistence" with JSONL persistence note using `NN_MONITOR_DATA_DIR` (line 69)

### Verification
- `grep -ci "python\|main.py\|pip install\|pydantic\|FastAPI"` → 2 matches (both `python test_client.py`, intentionally kept per task instructions)
- `grep -c "cargo build\|cargo run\|cargo test"` → 3 (one mention of each)
- README now references Rust source files only (config.rs, store.rs, ws.rs, models.rs)

### Notes
- Renamed ConnectionManager → WsManager to match actual Rust struct name (beyond what the task requested, but more accurate)
- Data Flow section still mentions "ConnectionManager" conceptually (line 37) — not a file reference, fine to keep
