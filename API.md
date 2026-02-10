# API Documentation

## Base URL

```
http://localhost:8000
```

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

---

## Error Responses

All endpoints may return:

| Status | Description |
|--------|-------------|
| `404` | Run not found |
| `500` | Internal server error (includes error message in `detail` field) |
