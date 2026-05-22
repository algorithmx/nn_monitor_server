# Server Configuration

The Neural Network Training Monitor can be configured via environment variables with the `NN_MONITOR_` prefix.

## Environment Variables

| Environment Variable | Type | Default | Range | Description |
|---------------------|------|---------|-------|-------------|
| `NN_MONITOR_MAX_RUNS` | integer | `10` | 1-100 | Maximum number of concurrent runs to store in memory |
| `NN_MONITOR_MAX_STEPS_PER_RUN` | integer | `1000` | 10-10000 | Maximum steps to keep per run (oldest steps are evicted) |
| `NN_MONITOR_MAX_REQUEST_SIZE` | integer | `2000000` | 100000+ | Maximum request size in bytes (2MB default) |
| `NN_MONITOR_HOST` | string | `0.0.0.0` | - | Server host address |
| `NN_MONITOR_PORT` | integer | `8000` | 1-65535 | Server port |
| `NN_MONITOR_LOG_LEVEL` | string | `info` | - | Log level (`debug`, `info`, `warning`, `error`) |
| `NN_MONITOR_CORS_ORIGINS` | JSON array | `["*"]` | - | CORS allowed origins |

## Configuration File (.env)

You can also create a `.env` file in the server directory:

```bash
NN_MONITOR_HOST=127.0.0.1
NN_MONITOR_PORT=8080
NN_MONITOR_MAX_RUNS=20
NN_MONITOR_MAX_STEPS_PER_RUN=5000
NN_MONITOR_LOG_LEVEL=debug
```

## Storage Limits

The server uses in-memory storage with automatic eviction:

- **Run limit**: When `max_runs` is reached, the oldest run (by `last_update` time) is evicted
- **Step limit**: When `max_steps_per_run` is reached for a run, the oldest steps are evicted
- **Memory consideration**: Adjust these limits based on your available RAM and training metrics size

## Example Configurations

### Development (Low Resource)
```bash
NN_MONITOR_MAX_RUNS=5
NN_MONITOR_MAX_STEPS_PER_RUN=500
NN_MONITOR_PORT=8000
```

### Production (High Volume)
```bash
NN_MONITOR_MAX_RUNS=50
NN_MONITOR_MAX_STEPS_PER_RUN=10000
NN_MONITOR_MAX_REQUEST_SIZE=10000000
NN_MONITOR_LOG_LEVEL=warning
```

### Behind Reverse Proxy
```bash
NN_MONITOR_HOST=127.0.0.1
NN_MONITOR_PORT=8000
NN_MONITOR_CORS_ORIGINS=["https://monitor.example.com"]
```
