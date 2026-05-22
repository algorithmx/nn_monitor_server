"""
Tests for run eviction when storage limits are exceeded.

Tests cover:
- Oldest run eviction when max_runs exceeded
- last_update-based eviction ordering
- Eviction notification behavior
- Storage limit configuration
"""

import pytest
import asyncio
from fastapi.testclient import TestClient

from tests.fixtures.metric_fixtures import valid_minimal_payload, storage_eviction_scenario
from main import MetricsPayload


class TestRunEviction:
    """Tests for automatic run eviction when storage limit is reached."""

    def test_oldest_run_evicted_when_limit_exceeded(self, mock_store):
        """Test that the oldest run is evicted when max_runs is exceeded."""
        # Mock store has max_runs=5 from conftest
        max_runs = 5

        # Add max_runs + 1 runs
        for i in range(max_runs + 1):
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = f"run_{i}"
            payload["metadata"]["timestamp"] = 1000 + i  # Different timestamps
            asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        # Check that first run was evicted
        run_0 = asyncio.run(mock_store.get_run("run_0"))
        assert run_0 is None, "First run should be evicted"

        # Check that other runs still exist
        for i in range(1, max_runs + 1):
            run = asyncio.run(mock_store.get_run(f"run_{i}"))
            assert run is not None, f"Run {i} should still exist"

    def test_eviction_based_on_last_update(self, mock_store):
        """Test that eviction is based on last_update timestamp."""
        # Create runs with specific timestamps to control eviction order
        # The store uses datetime.now() for last_update, so we need to add in order

        # Run 0: oldest
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = "oldest_run"
        payload["metadata"]["timestamp"] = 1000
        asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        # Runs 1-3: middle
        for i in range(1, 4):
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = f"run_{i}"
            payload["metadata"]["timestamp"] = 2000 + i
            asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        # Run 4: newest before eviction
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = "newest_before_eviction"
        payload["metadata"]["timestamp"] = 3000
        asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        # Add one more run to exceed limit (max_runs=5)
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = "trigger_eviction"
        payload["metadata"]["timestamp"] = 4000
        asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        # The oldest run should be evicted
        oldest_run = asyncio.run(mock_store.get_run("oldest_run"))
        assert oldest_run is None, "Oldest run should be evicted"

        # The newest runs should still exist
        newest_run = asyncio.run(mock_store.get_run("trigger_eviction"))
        assert newest_run is not None, "Newest run should not be evicted"

    def test_eviction_with_client(self, client: TestClient):
        """Test eviction through the HTTP API."""
        max_runs = 5

        # Add max_runs runs
        for i in range(max_runs):
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = f"api_run_{i}"
            payload["metadata"]["timestamp"] = 1000 + i
            client.post("/api/v1/metrics/layerwise", json=payload)

        # Verify all exist
        response = client.get("/api/v1/runs")
        data = response.json()
        assert len(data) == max_runs
        assert "api_run_0" in data

        # Add one more run
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = "api_run_new"
        payload["metadata"]["timestamp"] = 2000
        client.post("/api/v1/metrics/layerwise", json=payload)

        # Verify first run was evicted
        response = client.get("/api/v1/runs")
        data = response.json()
        assert "api_run_0" not in data, "First run should be evicted"
        assert "api_run_new" in data
        assert len(data) == max_runs

    def test_eviction_preserves_newest_runs(self, mock_store):
        """Test that eviction preserves the most recently updated runs."""
        max_runs = 5

        # Add runs with increasing timestamps
        for i in range(max_runs + 1):
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = f"run_{i}"
            payload["metadata"]["timestamp"] = (i + 1) * 100  # Increasing timestamps, must be > 0
            asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        # The newest runs should be preserved
        for i in range(1, max_runs + 1):
            run = asyncio.run(mock_store.get_run(f"run_{i}"))
            assert run is not None, f"Run {i} should be preserved"

        # The oldest run should be evicted
        run_0 = asyncio.run(mock_store.get_run("run_0"))
        assert run_0 is None, "Oldest run should be evicted"

    def test_eviction_does_not_affect_other_runs_data(self, mock_store):
        """Test that eviction doesn't corrupt remaining runs' data."""
        max_runs = 5

        # Add runs with multiple steps
        for i in range(max_runs + 1):
            run_id = f"multi_step_run_{i}"
            for step in [100, 200, 300]:
                payload = valid_minimal_payload()
                payload["metadata"]["run_id"] = run_id
                payload["metadata"]["global_step"] = step
                payload["metadata"]["timestamp"] = i * 1000 + step  # Unique timestamps
                asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        # Check that remaining runs have all their steps
        for i in range(1, max_runs + 1):
            run_id = f"multi_step_run_{i}"
            run = asyncio.run(mock_store.get_run(run_id))
            assert run is not None
            assert len(run.steps) == 3, f"Run {i} should have all 3 steps"

    def test_eviction_with_simultaneous_timestamps(self, mock_store):
        """Test eviction behavior when runs have identical timestamps."""
        # Add runs with the same timestamp
        base_time = 1000
        for i in range(6):  # max_runs=5, so 6 runs
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = f"simultaneous_{i}"
            payload["metadata"]["timestamp"] = base_time
            asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        # One run should be evicted (implementation may vary by dictionary ordering)
        response = asyncio.run(mock_store.get_all_runs())
        assert len(response) == 5, "Should have exactly max_runs after eviction"


class TestStepLimitEnforcement:
    """Tests for per-run step limit enforcement."""

    def test_oldest_step_evicted_when_limit_exceeded(self, mock_store):
        """Test that oldest steps are evicted when max_steps_per_run is exceeded."""
        max_steps = 20  # From conftest

        # Add more steps than the limit
        run_id = "step_limit_test"
        for i in range(max_steps + 5):
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = i * 10
            payload["metadata"]["timestamp"] = i + 1  # Must be > 0
            asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        # Get the run
        run = asyncio.run(mock_store.get_run(run_id))
        assert run is not None

        # Should have exactly max_steps
        assert len(run.steps) == max_steps

        # The oldest steps should be evicted
        step_numbers = [s.step for s in run.steps]
        # Should not have the earliest steps
        assert 0 not in step_numbers
        assert 10 not in step_numbers

        # Should have the most recent steps
        assert max_steps * 10 in step_numbers  # Last step

    def test_step_limit_enforced_per_run(self, mock_store):
        """Test that step limits are enforced independently per run."""
        max_steps = 20

        # Add multiple runs, each exceeding the limit
        for run_idx in range(3):
            run_id = f"step_limit_run_{run_idx}"
            for i in range(max_steps + 5):
                payload = valid_minimal_payload()
                payload["metadata"]["run_id"] = run_id
                payload["metadata"]["global_step"] = i * 10
                payload["metadata"]["timestamp"] = i + 1  # Must be > 0
                asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        # Each run should have exactly max_steps
        for run_idx in range(3):
            run_id = f"step_limit_run_{run_idx}"
            run = asyncio.run(mock_store.get_run(run_id))
            assert run is not None
            assert len(run.steps) == max_steps

    def test_step_limit_with_client(self, client: TestClient):
        """Test step limit enforcement through HTTP API."""
        max_steps = 20

        # Add more steps than the limit
        run_id = "api_step_limit_test"
        for i in range(max_steps + 5):
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = i * 10
            payload["metadata"]["timestamp"] = i
            client.post("/api/v1/metrics/layerwise", json=payload)

        # Get the run
        response = client.get(f"/api/v1/runs/{run_id}")
        data = response.json()

        # Should have exactly max_steps
        assert len(data["steps"]) == max_steps

    def test_step_limit_preserves_latest_steps(self, mock_store):
        """Test that step limit keeps the most recent steps."""
        max_steps = 20

        run_id = "preserve_latest_test"
        # Add steps with specific numbers
        step_numbers = list(range(0, (max_steps + 5) * 10, 10))

        for step_num in step_numbers:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step_num
            payload["metadata"]["timestamp"] = step_num + 1  # Must be > 0
            asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        run = asyncio.run(mock_store.get_run(run_id))
        actual_steps = [s.step for s in run.steps]

        # Should have the last max_steps steps
        expected_steps = step_numbers[-max_steps:]
        assert actual_steps == expected_steps

    def test_duplicate_step_does_not_count_toward_limit(self, mock_store):
        """Test that duplicate steps don't count toward the limit."""
        max_steps = 20

        run_id = "duplicate_limit_test"

        # Add max_steps unique steps
        for i in range(max_steps):
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = i * 10
            payload["metadata"]["timestamp"] = i + 1  # Must be > 0
            asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        # Verify count
        run = asyncio.run(mock_store.get_run(run_id))
        assert len(run.steps) == max_steps

        # Submit a duplicate of an existing step
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = run_id
        payload["metadata"]["global_step"] = 50  # Existing step
        payload["metadata"]["timestamp"] = 100
        asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        # Count should still be max_steps
        run = asyncio.run(mock_store.get_run(run_id))
        assert len(run.steps) == max_steps

        # Add one more new step - should trigger eviction
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = run_id
        payload["metadata"]["global_step"] = 1000  # New step
        payload["metadata"]["timestamp"] = 101
        asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        # Should still have max_steps
        run = asyncio.run(mock_store.get_run(run_id))
        assert len(run.steps) == max_steps


class TestStorageLimitsConfiguration:
    """Tests for storage limit configuration and behavior."""

    def test_custom_storage_limits_in_client(self, client: TestClient):
        """Test that client uses custom limits from mock store (conftest)."""
        # From conftest: max_runs=5, max_steps_per_run=20
        for i in range(5):
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = f"custom_test_{i}"
            client.post("/api/v1/metrics/layerwise", json=payload)

        response = client.get("/api/v1/runs")
        data = response.json()
        assert len(data) == 5  # Mock store has max_runs=5

    def test_custom_storage_limits_in_mock(self, mock_store):
        """Test that mock store uses custom limits from conftest."""
        # From conftest: max_runs=5, max_steps_per_run=20
        assert mock_store.max_runs == 5
        assert mock_store.max_steps_per_run == 20

        # Test that limits are enforced
        for i in range(6):
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = f"custom_limit_{i}"
            asyncio.run(mock_store.add_metrics(MetricsPayload(**payload)))

        runs = asyncio.run(mock_store.get_all_runs())
        assert len(runs) == 5, "Should enforce custom max_runs limit"
