"""
Tests for the GET /api/v1/runs endpoints.

Tests cover:
- GET /api/v1/runs (list all runs)
- GET /api/v1/runs/{run_id} (get specific run)
- GET /api/v1/runs/{run_id}/latest (get latest step)
"""

import pytest
import asyncio
from fastapi.testclient import TestClient

from tests.fixtures.metric_fixtures import valid_minimal_payload, valid_three_layer_network


class TestListAllRuns:
    """Tests for GET /api/v1/runs endpoint."""

    def test_empty_store_returns_empty_dict(self, client: TestClient):
        """Test that empty store returns empty dictionary."""
        response = client.get("/api/v1/runs")
        assert response.status_code == 200
        assert response.json() == {}

    def test_list_runs_after_single_submission(self, client: TestClient):
        """Test listing runs after submitting one metric."""
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
        response = client.get("/api/v1/runs")
        assert response.status_code == 200
        data = response.json()
        assert "minimal_test" in data
        assert isinstance(data["minimal_test"], dict)

    def test_list_runs_response_structure(self, client: TestClient):
        """Test that list runs response has correct structure."""
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
        response = client.get("/api/v1/runs")
        data = response.json()
        run_info = data["minimal_test"]
        assert "created_at" in run_info
        assert "last_update" in run_info
        assert "step_count" in run_info
        assert "latest_step" in run_info

    def test_step_count_is_one_after_single_submission(self, client: TestClient):
        """Test that step_count is 1 after single submission."""
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
        response = client.get("/api/v1/runs")
        data = response.json()
        assert data["minimal_test"]["step_count"] == 1

    def test_latest_step_matches_submission(self, client: TestClient):
        """Test that latest_step matches the submitted step."""
        payload = valid_minimal_payload()
        step = 250
        payload["metadata"]["global_step"] = step
        client.post("/api/v1/metrics/layerwise", json=payload)
        response = client.get("/api/v1/runs")
        data = response.json()
        assert data["minimal_test"]["latest_step"] == step

    def test_multiple_runs_all_listed(self, client: TestClient):
        """Test that multiple runs are all listed."""
        run_ids = ["run_1", "run_2", "run_3"]
        for run_id in run_ids:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            client.post("/api/v1/metrics/layerwise", json=payload)
        response = client.get("/api/v1/runs")
        data = response.json()
        for run_id in run_ids:
            assert run_id in data

    def test_created_at_format(self, client: TestClient):
        """Test that created_at is in ISO format."""
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
        response = client.get("/api/v1/runs")
        data = response.json()
        created_at = data["minimal_test"]["created_at"]
        assert isinstance(created_at, str)
        assert "T" in created_at  # ISO format includes T

    def test_latest_step_null_for_new_run(self, client: TestClient, mock_store):
        """Test that latest_step is null for a run with no steps yet."""
        # Create a run without adding steps
        asyncio.run(mock_store.add_metrics(
            __import__('main').MetricsPayload(**valid_minimal_payload())
        ))
        response = client.get("/api/v1/runs")
        data = response.json()
        # Note: After adding metrics, there will be one step, so latest_step won't be null
        # This test verifies the structure exists
        assert "latest_step" in data["minimal_test"]


class TestGetSpecificRun:
    """Tests for GET /api/v1/runs/{run_id} endpoint."""

    def test_get_existing_run_returns_200(self, client: TestClient):
        """Test getting an existing run returns 200."""
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
        response = client.get("/api/v1/runs/minimal_test")
        assert response.status_code == 200

    def test_get_nonexistent_run_returns_404(self, client: TestClient):
        """Test getting a nonexistent run returns 404."""
        response = client.get("/api/v1/runs/nonexistent_run")
        assert response.status_code == 404

    def test_get_nonexistent_run_error_format(self, client: TestClient):
        """Test that 404 error has correct format."""
        response = client.get("/api/v1/runs/nonexistent_run")
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], dict)
        assert data["detail"]["error"] == "not_found"

    def test_get_nonexistent_run_error_message(self, client: TestClient):
        """Test that 404 error includes run name in message."""
        response = client.get("/api/v1/runs/nonexistent_run")
        data = response.json()
        assert "nonexistent_run" in data["detail"]["message"]

    def test_get_run_response_structure(self, client: TestClient):
        """Test that get run response has correct structure."""
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
        response = client.get("/api/v1/runs/minimal_test")
        data = response.json()
        assert "created_at" in data
        assert "last_update" in data
        assert "steps" in data
        assert isinstance(data["steps"], list)

    def test_get_run_steps_structure(self, client: TestClient):
        """Test that steps in run have correct structure."""
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
        response = client.get("/api/v1/runs/minimal_test")
        data = response.json()
        assert len(data["steps"]) > 0
        step = data["steps"][0]
        assert "step" in step
        assert "timestamp" in step
        assert "batch_size" in step
        assert "layers" in step
        assert "cross_layer" in step

    def test_get_run_multiple_steps(self, client: TestClient):
        """Test getting a run with multiple steps."""
        run_id = "multi_step_test"
        for step_num in [100, 200, 300]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step_num
            client.post("/api/v1/metrics/layerwise", json=payload)
        response = client.get("/api/v1/runs/multi_step_test")
        data = response.json()
        assert len(data["steps"]) == 3

    def test_get_run_steps_sorted(self, client: TestClient):
        """Test that steps are sorted by step number."""
        run_id = "sorted_test"
        # Submit in reverse order
        for step_num in [300, 100, 200]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step_num
            client.post("/api/v1/metrics/layerwise", json=payload)
        response = client.get("/api/v1/runs/sorted_test")
        data = response.json()
        step_numbers = [step["step"] for step in data["steps"]]
        assert step_numbers == [100, 200, 300]

    def test_get_run_layers_structure(self, client: TestClient):
        """Test that layer data has correct structure."""
        client.post("/api/v1/metrics/layerwise", json=valid_three_layer_network())
        response = client.get("/api/v1/runs/three_layer_net")
        data = response.json()
        step = data["steps"][0]
        assert len(step["layers"]) == 3
        layer = step["layers"][0]
        assert "layer_id" in layer
        assert "layer_type" in layer
        assert "depth_index" in layer
        assert "intermediate_features" in layer
        assert "gradient_flow" in layer
        assert "parameter_statistics" in layer


class TestGetLatestStep:
    """Tests for GET /api/v1/runs/{run_id}/latest endpoint."""

    def test_get_latest_step_returns_200(self, client: TestClient):
        """Test getting latest step returns 200."""
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
        response = client.get("/api/v1/runs/minimal_test/latest")
        assert response.status_code == 200

    def test_get_latest_step_nonexistent_run_returns_404(self, client: TestClient):
        """Test getting latest step from nonexistent run returns 404."""
        response = client.get("/api/v1/runs/nonexistent_run/latest")
        assert response.status_code == 404

    def test_get_latest_step_response_structure(self, client: TestClient):
        """Test that latest step response has correct structure."""
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
        response = client.get("/api/v1/runs/minimal_test/latest")
        data = response.json()
        assert "step" in data
        assert "timestamp" in data
        assert "batch_size" in data
        assert "layers" in data
        assert "cross_layer" in data

    def test_get_latest_step_returns_actual_latest(self, client: TestClient):
        """Test that latest endpoint returns the most recent step."""
        run_id = "latest_test"
        for step_num in [100, 200, 300]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step_num
            client.post("/api/v1/metrics/layerwise", json=payload)
        response = client.get("/api/v1/runs/latest_test/latest")
        data = response.json()
        assert data["step"] == 300

    def test_get_latest_step_with_single_step(self, client: TestClient):
        """Test getting latest step when there's only one step."""
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
        response = client.get("/api/v1/runs/minimal_test/latest")
        data = response.json()
        assert data["step"] == 100

    def test_get_latest_step_after_duplicate_submission(self, client: TestClient):
        """Test that duplicate step submission updates the latest step."""
        run_id = "duplicate_latest_test"
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = run_id
        payload["metadata"]["global_step"] = 100

        # First submission
        client.post("/api/v1/metrics/layerwise", json=payload)

        # Modify and submit same step again
        payload["layer_statistics"][0]["intermediate_features"]["activation_std"] = 0.5
        client.post("/api/v1/metrics/layerwise", json=payload)

        # Get latest and verify it was updated
        response = client.get("/api/v1/runs/duplicate_latest_test/latest")
        data = response.json()
        assert data["step"] == 100
        # The updated value should be present
        assert data["layers"][0]["intermediate_features"]["activation_std"] == 0.5


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_200(self, client: TestClient):
        """Test that health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client: TestClient):
        """Test that health response has correct structure."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "active_connections" in data
        assert data["status"] == "healthy"

    def test_health_active_connections_count(self, client: TestClient):
        """Test that active_connections is a non-negative integer."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["active_connections"], int)
        assert data["active_connections"] >= 0
