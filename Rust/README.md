## Project Overview

This is a **Neural Network Training Monitor Server** - an Axum backend that provides real-time monitoring of neural network training metrics. It serves as a lightweight alternative to TensorBoard/WandB, focusing on layer-wise granularity (activations, gradients, parameters per layer) with async HTTP POST ingestion and WebSocket real-time visualization.

## Running the Server

```bash
# Build the project
cargo build

# Start the server (default port 8000)
cargo run
```

## Testing

```bash
# Run unit and integration tests
cargo test

# Run the test client (simulates training with fake metrics)
python test_client.py

# Customize test parameters
python test_client.py --run-id my_experiment --steps 200 --interval 0.5

# Health check
curl http://localhost:8000/health
```

## Architecture

### Data Flow

1. **Training scripts** POST metrics to `/api/v1/metrics/layerwise` (async, returns 202 immediately)
2. **MetricsStore** stores data in-memory with automatic cleanup
3. **ConnectionManager** broadcasts new metrics to WebSocket subscribers
4. **Frontend** at `/static/index.html` visualizes data via WebSocket at `/ws`

### Key Components

- **`ServerConfig`** (config.rs:35-60): Environment-based configuration
- **`MetricsStore`** (store.rs:161-346): Thread-safe in-memory storage with async locks, automatic eviction of oldest runs, step deduplication, and sorted step indexing
- **`WsManager`** (ws.rs:17-66): WebSocket connection lifecycle with automatic cleanup of failed connections
- **Serde models** (models.rs:96-237): Schema validation for hierarchical metrics structure

### JSON Schema

The server accepts metrics via `POST /api/v1/metrics/layerwise`. See **`SCHEMA.md`** for complete schema documentation including:

- Field descriptions and types
- Required vs optional fields
- Complete JSON examples
- Validation rules

### API

See **`API.md`** for complete REST and WebSocket endpoint documentation including request/response formats, message types, and error handling.

### Configuration

See **`CONFIGURATION.md`** for environment variable configuration options including storage limits, server settings, and CORS configuration.

## Development Notes

- All storage operations use `tokio::sync::RwLock` for async safety
- WebSocket broadcasts silently drop failed connections (automatic cleanup)
- Metrics are accepted synchronously but processed asynchronously to avoid blocking training
- Optional JSONL persistence to disk: configure `NN_MONITOR_DATA_DIR` to persist metrics between restarts
- Storage limits prevent unbounded memory growth: configure via `NN_MONITOR_MAX_RUNS` and `NN_MONITOR_MAX_STEPS_PER_RUN` environment variables
- The frontend supports **Time Travel History Navigation** - see `docs/FEATURES.md` for details
