"""
Tests for step deduplication behavior.

Tests cover:
- Duplicate step submission handling
- Step replacement (same step, different data)
- Steps sorted by step number
- Step count tracking
"""

import pytest
import asyncio
from fastapi.testclient import TestClient

from tests.fixtures.metric_fixtures import valid_minimal_payload, duplicate_step_scenario
from main import MetricsPayload


class TestStepDeduplication:
    """Tests for step deduplication in runs."""

    def test_duplicate_step_replaces_existing(self, client: TestClient):
        """Test that submitting the same step twice replaces the first."""
        run_id = "duplicate_test"
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = run_id
        payload["metadata"]["global_step"] = 100
        original_std = payload["layer_statistics"][0]["intermediate_features"]["activation_std"]

        # First submission
        client.post("/api/v1/metrics/layerwise", json=payload)

        # Modify and submit same step again
        payload["layer_statistics"][0]["intermediate_features"]["activation_std"] = 0.5
        client.post("/api/v1/metrics/layerwise", json=payload)

        # Get run data
        response = client.get(f"/api/v1/runs/{run_id}")
        data = response.json()

        # Should still have only 1 step
        assert len(data["steps"]) == 1
        # The value should be updated
        assert data["steps"][0]["layers"][0]["intermediate_features"]["activation_std"] == 0.5

    def test_duplicate_step_doesnt_increase_count(self, client: TestClient):
        """Test that duplicate step doesn't increase step count."""
        run_id = "count_test"
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = run_id
        payload["metadata"]["global_step"] = 100

        # First submission
        client.post("/api/v1/metrics/layerwise", json=payload)

        # Check runs list
        response = client.get("/api/v1/runs")
        data = response.json()
        assert data[run_id]["step_count"] == 1

        # Submit same step again
        client.post("/api/v1/metrics/layerwise", json=payload)

        # Count should still be 1
        response = client.get("/api/v1/runs")
        data = response.json()
        assert data[run_id]["step_count"] == 1

    def test_duplicate_step_updates_latest_step(self, client: TestClient):
        """Test that duplicate step updates latest_step if it's the latest."""
        run_id = "latest_test"
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = run_id
        payload["metadata"]["global_step"] = 100

        client.post("/api/v1/metrics/layerwise", json=payload)

        # Submit same step with updated data
        payload["layer_statistics"][0]["intermediate_features"]["activation_std"] = 0.5
        client.post("/api/v1/metrics/layerwise", json=payload)

        # Get latest step
        response = client.get(f"/api/v1/runs/{run_id}/latest")
        data = response.json()

        assert data["step"] == 100
        assert data["layers"][0]["intermediate_features"]["activation_std"] == 0.5

    def test_duplicate_step_among_others(self, client: TestClient):
        """Test duplicate step when there are other steps in the run."""
        run_id = "mixed_test"

        # Submit steps 100, 200, 300
        for step in [100, 200, 300]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        # Now resubmit step 200 with different data
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = run_id
        payload["metadata"]["global_step"] = 200
        payload["layer_statistics"][0]["intermediate_features"]["activation_std"] = 0.999
        client.post("/api/v1/metrics/layerwise", json=payload)

        # Check that we still have 3 steps
        response = client.get(f"/api/v1/runs/{run_id}")
        data = response.json()
        assert len(data["steps"]) == 3

        # Check that step 200 was updated
        step_200 = next(s for s in data["steps"] if s["step"] == 200)
        assert step_200["layers"][0]["intermediate_features"]["activation_std"] == 0.999

    def test_multiple_duplicate_submissions(self, client: TestClient):
        """Test multiple duplicate submissions of the same step."""
        run_id = "multi_duplicate_test"
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = run_id
        payload["metadata"]["global_step"] = 100

        # Submit the same step 5 times with different values
        expected_values = []
        for i in range(5):
            payload["layer_statistics"][0]["intermediate_features"]["activation_std"] = 0.1 * (i + 1)
            expected_values.append(0.1 * (i + 1))
            client.post("/api/v1/metrics/layerwise", json=payload)

        # Should still have only 1 step
        response = client.get(f"/api/v1/runs/{run_id}")
        data = response.json()
        assert len(data["steps"]) == 1

        # The value should be the last one submitted
        actual_value = data["steps"][0]["layers"][0]["intermediate_features"]["activation_std"]
        assert actual_value == expected_values[-1]


class TestStepSorting:
    """Tests for step sorting behavior."""

    def test_steps_sorted_by_step_number(self, client: TestClient):
        """Test that steps are sorted by step number, not submission order."""
        run_id = "sort_test"

        # Submit in reverse order: 300, 100, 200
        for step in [300, 100, 200]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        # Get run data
        response = client.get(f"/api/v1/runs/{run_id}")
        data = response.json()

        # Verify they're sorted
        step_numbers = [s["step"] for s in data["steps"]]
        assert step_numbers == [100, 200, 300]

    def test_steps_sorted_after_duplicate_in_middle(self, client: TestClient):
        """Test that sorting is maintained after duplicate submission."""
        run_id = "sort_duplicate_test"

        # Submit steps 100, 200, 300
        for step in [100, 200, 300]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        # Add step 50 (should go to beginning)
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = run_id
        payload["metadata"]["global_step"] = 50
        client.post("/api/v1/metrics/layerwise", json=payload)

        # Verify sorted order
        response = client.get(f"/api/v1/runs/{run_id}")
        data = response.json()
        step_numbers = [s["step"] for s in data["steps"]]
        assert step_numbers == [50, 100, 200, 300]


class TestStepCountTracking:
    """Tests for step count tracking."""

    def test_step_count_increments_with_new_steps(self, client: TestClient):
        """Test that step_count increments with each new step."""
        run_id = "count_increment_test"

        for i, step in enumerate([100, 200, 300], 1):
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

            response = client.get("/api/v1/runs")
            data = response.json()
            assert data[run_id]["step_count"] == i

    def test_step_count_unchanged_by_duplicate(self, client: TestClient):
        """Test that step_count doesn't change with duplicate submission."""
        run_id = "count_unchanged_test"

        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = run_id
        payload["metadata"]["global_step"] = 100

        client.post("/api/v1/metrics/layerwise", json=payload)

        response = client.get("/api/v1/runs")
        data = response.json()
        count_after_first = data[run_id]["step_count"]

        # Submit duplicate
        client.post("/api/v1/metrics/layerwise", json=payload)

        response = client.get("/api/v1/runs")
        data = response.json()
        assert data[run_id]["step_count"] == count_after_first

    def test_step_count_across_multiple_runs(self, client: TestClient):
        """Test step_count tracking for multiple independent runs."""
        run_1 = "run_1"
        run_2 = "run_2"

        # Add 2 steps to run_1
        for step in [100, 200]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_1
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        # Add 3 steps to run_2
        for step in [100, 200, 300]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_2
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        response = client.get("/api/v1/runs")
        data = response.json()

        assert data[run_1]["step_count"] == 2
        assert data[run_2]["step_count"] == 3
