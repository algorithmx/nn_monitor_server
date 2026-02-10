# API Documentation

## Features

The frontend supports **Time Travel History Navigation** - see [docs/FEATURES.md](docs/FEATURES.md) for details on browsing historical training data.

## Base URL

```
http://localhost:8000
```

---

## Server Configuration

The server can be configured via environment variables with the `NN_MONITOR_` prefix:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `NN_MONITOR_MAX_RUNS` | `10` | Maximum number of concurrent runs to store (1-100) |
| `NN_MONITOR_MAX_STEPS_PER_RUN` | `1000` | Maximum steps to keep per run (10-10000) |
| `NN_MONITOR_MAX_REQUEST_SIZE` | `2000000` | Maximum request size in bytes (min 100000) |
| `NN_MONITOR_HOST` | `0.0.0.0` | Server host address |
| `NN_MONITOR_PORT` | `8000` | Server port (1-65535) |
| `NN_MONITOR_LOG_LEVEL` | `info` | Log level (e.g., `debug`, `info`, `warning`, `error`) |
| `NN_MONITOR_CORS_ORIGINS` | `["*"]` | CORS allowed origins (JSON array) |

---

## REST Endpoints

### POST /api/v1/metrics/layerwise

Submit layer-wise training metrics.

**Request Body:** See `SCHEMA.md` for complete JSON schema.

**Response:** `202 Accepted`

```json
{
  "status": "accepted",
  "run_id": "experiment_2024_0210_v2"
}
```

**Behavior:**
- Non-blocking: returns immediately after validating payload
- Stores metrics in memory and broadcasts to WebSocket subscribers
- If storage limits exceeded, oldest run is evicted

---

### GET /api/v1/runs

List all training runs (metadata summary only).

**Response:** `200 OK`

```json
{
  "experiment_2024_0210_v2": {
    "created_at": "2024-02-10T12:34:56.789",
    "last_update": "2024-02-10T12:45:23.456",
    "step_count": 15,
    "latest_step": 1500
  },
  "test_run_1707589200": {
    "created_at": "2024-02-10T11:30:00.123",
    "last_update": "2024-02-10T11:35:00.456",
    "step_count": 10,
    "latest_step": 90
  }
}
```

**Notes:**
- `latest_step` will be `null` if the run has no steps recorded yet

---

### GET /api/v1/runs/{run_id}

Get complete data for a specific run including all step history.

**Response:** `200 OK` or `404 Not Found`

```json
{
  "created_at": "2024-02-10T12:34:56.789",
  "last_update": "2024-02-10T12:45:23.456",
  "steps": [
    {
      "step": 1400,
      "timestamp": 1707589000.123,
      "batch_size": 64,
      "layers": [...],
      "cross_layer": {...}
    },
    {
      "step": 1500,
      "timestamp": 1707589200.123,
      "batch_size": 64,
      "layers": [...],
      "cross_layer": {...}
    }
  ]
}
```

---

### GET /api/v1/runs/{run_id}/latest

Get only the most recent step data for a run.

**Response:** `200 OK` or `404 Not Found`

```json
{
  "step": 1500,
  "timestamp": 1707589200.123,
  "batch_size": 64,
  "layers": [...],
  "cross_layer": {...}
}
```

---

### GET /health

Health check endpoint.

**Response:** `200 OK`

```json
{
  "status": "healthy",
  "active_connections": 2
}
```

---

## WebSocket

### Endpoint

```
ws://localhost:8000/ws
```

### Connection Flow

1. Connect to `/ws`
2. Server sends `initial_runs` message with current runs list
3. Send JSON messages to interact (see below)
4. Server broadcasts `new_metrics` for each incoming metric POST

### Client → Server Messages

**Subscribe to specific run history:**

```json
{
  "action": "subscribe_run",
  "run_id": "experiment_2024_0210_v2"
}
```

**Ping/pong for keep-alive:**

```json
{
  "action": "ping"
}
```

### Server → Client Messages

**Initial runs list (sent on connect):**

```json
{
  "type": "initial_runs",
  "data": {
    "experiment_2024_0210_v2": {
      "created_at": "2024-02-10T12:34:56.789",
      "last_update": "2024-02-10T12:45:23.456",
      "step_count": 15,
      "latest_step": 1500
    }
  }
}
```

**Run history (response to subscribe_run):**

```json
{
  "type": "run_history",
  "run_id": "experiment_2024_0210_v2",
  "data": {
    "created_at": "2024-02-10T12:34:56.789",
    "steps": [...]
  }
}
```

**New metrics (broadcasted on each POST):**

```json
{
  "type": "new_metrics",
  "run_id": "experiment_2024_0210_v2",
  "data": {
    "step": 1500,
    "timestamp": 1707589200.123,
    "batch_size": 64,
    "layers": [...],
    "cross_layer": {...}
  }
}
```

**Pong response:**

```json
{
  "type": "pong"
}
```

**Error response:**

```json
{
  "type": "error",
  "message": "Run 'unknown_run' not found"
}
```

The server sends error messages in response to:
- Invalid JSON format
- Subscribe requests for non-existent runs

---

## Error Responses

All REST endpoints may return:

| Status | Description |
|--------|-------------|
| `400` | Validation error - invalid metrics payload |
| `404` | Run not found |
| `500` | Internal server error |

Error responses use the following format:

```json
{
  "detail": {
    "error": "validation_error",
    "message": "Layers must be sorted by depth_index: ..."
  }
}
```

For validation errors, the `error` field can be:
- `validation_error` - Invalid request payload
- `internal_error` - Server processing failure
- `not_found` - Requested resource does not exist
