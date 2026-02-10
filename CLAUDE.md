# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Neural Network Training Monitor Server** - a FastAPI backend that provides real-time monitoring of neural network training metrics. It serves as a lightweight alternative to TensorBoard/WandB, focusing on layer-wise granularity (activations, gradients, parameters per layer) with async HTTP POST ingestion and WebSocket real-time visualization.

## Running the Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (default port 8000)
python main.py

# Or with uvicorn directly for more control
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing

```bash
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
2. **MetricsStore** stores data in-memory with automatic cleanup (max 10 runs, 1000 steps each)
3. **ConnectionManager** broadcasts new metrics to WebSocket subscribers
4. **Frontend** at `/static/index.html` visualizes data via WebSocket at `/ws`

### Key Components

- **`MetricsStore`** (main.py:71-147): Thread-safe in-memory storage with async locks, automatic eviction of oldest runs, step deduplication, and sorted step indexing.
- **`ConnectionManager`** (main.py:155-185): WebSocket connection lifecycle with automatic cleanup of failed connections.
- **Pydantic models** (main.py:18-66): Schema validation for hierarchical metrics structure.

### JSON Schema

The server accepts metrics via `POST /api/v1/metrics/layerwise`. See **`SCHEMA.md`** for complete schema documentation including:

- Field descriptions and types
- Required vs optional fields
- Complete JSON examples
- Validation rules

The Pydantic models defining this schema are in **main.py:18-66** (`MetricsPayload`, `LayerStatistic`, `IntermediateFeatures`, `GradientFlow`, `ParameterStatistics`).

### API

See **`API.md`** for complete REST and WebSocket endpoint documentation including request/response formats, message types, and error handling.

## Development Notes

- All storage operations use `async with self._lock` for thread safety
- WebSocket broadcasts silently drop failed connections (automatic cleanup)
- Metrics are accepted synchronously but processed asynchronously to avoid blocking training
- The server has no database persistence - all data is in-memory and lost on restart
- Storage limits prevent unbounded memory growth: configure `max_runs` and `max_steps_per_run` in `MetricsStore.__init__`
