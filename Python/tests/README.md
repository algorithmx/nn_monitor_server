# Test Suite for Neural Network Training Monitor Server

## Overview

This is a comprehensive pytest-based test suite for the Neural Network Training Monitor Server. The suite provides full-stack testing including REST API endpoints, WebSocket protocol, storage behavior, and integration scenarios.

## Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run specific test directory
pytest tests/test_rest_api/
pytest tests/test_websocket/
pytest tests/test_storage/
pytest tests/test_integration/

# Run specific test file
pytest tests/test_rest_api/test_metrics_endpoint.py

# Run specific test
pytest tests/test_rest_api/test_metrics_endpoint.py::TestMetricsEndpointValidRequests::test_valid_minimal_payload_returns_202

# Run with coverage
pytest --cov=. --cov-report=html

# Run only fast tests (exclude slow ones)
pytest -m "not slow"

# Run with verbose output
pytest -v
```

## Test Organization

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── fixtures/
│   ├── metric_fixtures.py      # Valid metric payloads (deterministic scenarios)
│   └── error_fixtures.py       # Invalid payloads for validation testing
├── test_rest_api/
│   ├── test_metrics_endpoint.py       # POST /api/v1/metrics/layerwise
│   ├── test_runs_endpoint.py          # GET /api/v1/runs endpoints
│   └── test_error_responses.py        # Error response formats
├── test_websocket/
│   └── test_connection_lifecycle.py   # WebSocket connection & messaging
├── test_storage/
│   ├── test_step_deduplication.py     # Step deduplication behavior
│   └── test_run_eviction.py           # Storage limits & eviction
└── test_integration/
    ├── test_concurrent_clients.py     # Concurrent HTTP/WebSocket clients
    └── test_time_travel.py            # Time travel history feature
```

## Key Fixtures

### Test Client (`client`)
Provides a FastAPI `TestClient` with isolated dependencies:
```python
def test_example(client: TestClient):
    response = client.post("/api/v1/metrics/layerwise", json=payload)
    assert response.status_code == 202
```

### Mock Store (`mock_store`)
Provides a fresh `MetricsStore` instance with test limits:
- `max_runs=5` (vs default 10)
- `max_steps_per_run=20` (vs default 1000)

```python
def test_with_store(mock_store):
    async def add_data():
        await mock_store.add_metrics(payload)
    asyncio.run(add_data())
```

### Payload Factories
```python
def test_with_factory(create_sample_payload):
    payload = create_sample_payload(
        run_id="my_test",
        step=500,
        batch_size=128,
        layer_count=3
    )
    response = client.post("/api/v1/metrics/layerwise", json=payload)
```

### Deterministic Fixtures
Pre-defined test data scenarios covering:
- Valid payloads (minimal, three-layer, vanishing/exploding gradients, healthy)
- Error scenarios (missing fields, invalid values, NaN/Infinity, depth ordering)
- Storage scenarios (eviction, step limits, deduplication)

```python
from tests.fixtures.metric_fixtures import (
    valid_minimal_payload,
    vanishing_gradient_pattern,
    storage_eviction_scenario
)

from tests.fixtures.error_fixtures import (
    unsorted_depth_index,
    nan_activation_std
)
```

## Test Coverage Areas

### REST API Testing
- ✅ POST /api/v1/metrics/layerwise (all validation rules)
- ✅ GET /api/v1/runs (metadata summary)
- ✅ GET /api/v1/runs/{run_id} (complete run history)
- ✅ GET /api/v1/runs/{run_id}/latest (most recent step)
- ✅ GET /health (health check)
- ✅ All error response codes and formats

### WebSocket Protocol Testing
- ✅ Connection lifecycle and initial_runs message
- ✅ Subscribe/unsubscribe to specific runs
- ✅ Ping/pong keepalive
- ✅ Error message handling
- ✅ Broadcast behavior on metrics POST

### Storage Behavior Testing
- ✅ Step deduplication (same step submitted twice)
- ✅ Oldest-run eviction when limit exceeded
- ✅ Depth ordering validation
- ✅ Step limit enforcement per run

### Integration Testing
- ✅ Concurrent HTTP clients submitting metrics
- ✅ Multiple WebSocket subscribers
- ✅ Time travel WebSocket message flow (run_history, new_metrics)
- ✅ Storage limit scenarios

## Architecture Notes

### Test Isolation
- Each test function gets fresh `mock_store` and `mock_manager` instances
- No shared state between tests
- Custom storage limits ensure quick eviction testing

### Deterministic Data
- All fixtures use hard-coded values (no random data)
- Timestamps are controlled for predictable ordering
- Edge cases are explicitly crafted

### Async Testing
- Uses pytest's async support
- Fixtures provide both sync and async interfaces
- WebSocket tests use TestClient's async context manager

## Design Principles

1. **YAGNI**: Only tests what exists in the codebase
2. **One assertion per test** (mostly) for clear failure messages
3. **Descriptive test names** that explain what is being tested
4. **Parametrized tests** for similar test cases
5. **Fixtures over setup** for reusable test components

## Contributing

When adding new tests:

1. Place tests in the appropriate directory based on what they test
2. Use existing fixtures when possible
3. Create new fixtures in `fixtures/` if needed
4. Follow the naming convention: `test_<what>_<scenario>_expected`
5. Add docstrings explaining complex test scenarios
6. Use parametrize for multiple similar cases

## Debugging Failed Tests

```bash
# Run with output on failure
pytest -v --tb=long

# Run with pdb on failure
pytest --pdb

# Run only failing tests from last run
pytest --lf

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l
```

## Known Limitations

- Tests don't cover frontend JavaScript (only backend API)
- No performance/load testing (yet)
- WebSocket timeout handling uses implicit timing
- Some edge cases in error handling may be untested
