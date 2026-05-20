# Performance Enhancement - Decisions

## 2026-05-14 Planning Session
- Extra fields beyond schema: DROP (server defines schema, clients should only send schema-defined fields)
- Scope: Phases 1+2 only (items #1, #3, #4, #5, #6, #7, #8, #9, #10, #11)
- Test strategy: Tests-after with existing 244+ test suite
- hashbrown: Only store.rs, NOT models.rs (public API types with serde)
- #5 (lock hold time) subsumed by P0 — build_step_data moves before lock naturally
- binary_search: Must move step_data in both match arms (only one executes, so no double-move)
