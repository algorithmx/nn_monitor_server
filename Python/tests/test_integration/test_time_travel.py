"""
Integration tests for time travel history navigation feature.

Tests cover:
- WebSocket run_history message
- Historical step data retrieval
- Step ordering and completeness
- History preservation across submissions
"""

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.metric_fixtures import valid_minimal_payload, valid_three_layer_network


class TestWebSocketRunHistory:
    """Tests for WebSocket run_history message used by time travel feature."""

    def test_subscribe_run_sends_history(self, client: TestClient):
        """Test that subscribing to a run sends its complete history."""
        run_id = "history_test"

        # Create a run with multiple steps
        for step in [100, 200, 300]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        # Subscribe via WebSocket
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            websocket.send_json({"action": "subscribe_run", "run_id": run_id})

            # Should receive run_history
            data = websocket.receive_json()
            assert data["type"] == "run_history"
            assert data["run_id"] == run_id

    def test_run_history_contains_all_steps(self, client: TestClient):
        """Test that run_history contains all submitted steps."""
        run_id = "full_history_test"
        expected_steps = [100, 200, 300, 400, 500]

        for step in expected_steps:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs
            websocket.send_json({"action": "subscribe_run", "run_id": run_id})

            data = websocket.receive_json()
            steps = data["data"]["steps"]
            step_numbers = [s["step"] for s in steps]

            assert step_numbers == expected_steps

    def test_run_history_structure(self, client: TestClient):
        """Test that run_history has correct structure for time travel."""
        client.post("/api/v1/metrics/layerwise", json=valid_three_layer_network())

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs
            websocket.send_json({"action": "subscribe_run", "run_id": "three_layer_net"})

            data = websocket.receive_json()

            # Top-level structure
            assert "created_at" in data["data"]
            assert "steps" in data["data"]
            assert isinstance(data["data"]["steps"], list)

            # Step structure
            step = data["data"]["steps"][0]
            assert "step" in step
            assert "timestamp" in step
            assert "batch_size" in step
            assert "layers" in step
            assert "cross_layer" in step

            # Layer structure (for visualization)
            layer = step["layers"][0]
            assert "layer_id" in layer
            assert "layer_type" in layer
            assert "depth_index" in layer
            assert "intermediate_features" in layer
            assert "gradient_flow" in layer
            assert "parameter_statistics" in layer

    def test_run_history_steps_sorted(self, client: TestClient):
        """Test that history steps are sorted by step number."""
        run_id = "sorted_history_test"

        # Submit in random order
        for step in [300, 100, 500, 200, 400]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs
            websocket.send_json({"action": "subscribe_run", "run_id": run_id})

            data = websocket.receive_json()
            step_numbers = [s["step"] for s in data["data"]["steps"]]

            assert step_numbers == [100, 200, 300, 400, 500]

    def test_run_history_with_single_step(self, client: TestClient):
        """Test run_history with only one step."""
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs
            websocket.send_json({"action": "subscribe_run", "run_id": "minimal_test"})

            data = websocket.receive_json()
            assert len(data["data"]["steps"]) == 1
            assert data["data"]["steps"][0]["step"] == 100


class TestHistoricalDataRetrieval:
    """Tests for retrieving historical data via HTTP API."""

    def test_get_run_returns_complete_history(self, client: TestClient):
        """Test that GET /api/v1/runs/{run_id} returns complete history."""
        run_id = "http_history_test"

        for step in [100, 200, 300]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        response = client.get(f"/api/v1/runs/{run_id}")
        data = response.json()

        assert "steps" in data
        assert len(data["steps"]) == 3
        step_numbers = [s["step"] for s in data["steps"]]
        assert step_numbers == [100, 200, 300]

    def test_get_specific_step_from_history(self, client: TestClient):
        """Test retrieving a specific step from history."""
        run_id = "specific_step_test"

        for step in [100, 200, 300]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        response = client.get(f"/api/v1/runs/{run_id}")
        data = response.json()

        # Find step 200
        step_200 = next(s for s in data["steps"] if s["step"] == 200)
        assert step_200 is not None
        assert step_200["step"] == 200

    def test_history_preserved_across_submissions(self, client: TestClient):
        """Test that history is preserved when adding new steps."""
        run_id = "preserve_history_test"

        # Initial steps
        for step in [100, 200, 300]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        # Get history
        response1 = client.get(f"/api/v1/runs/{run_id}")
        data1 = response1.json()
        original_steps = [s["step"] for s in data1["steps"]]

        # Add more steps
        for step in [400, 500]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        # Get history again
        response2 = client.get(f"/api/v1/runs/{run_id}")
        data2 = response2.json()
        all_steps = [s["step"] for s in data2["steps"]]

        # Original steps should still be there
        for step in original_steps:
            assert step in all_steps

        assert all_steps == [100, 200, 300, 400, 500]

    def test_duplicate_step_updates_history(self, client: TestClient):
        """Test that duplicate step submission updates the history."""
        run_id = "update_history_test"

        # Initial submission
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = run_id
        payload["metadata"]["global_step"] = 100
        original_std = payload["layer_statistics"][0]["intermediate_features"]["activation_std"]
        client.post("/api/v1/metrics/layerwise", json=payload)

        # Get history
        response1 = client.get(f"/api/v1/runs/{run_id}")
        data1 = response1.json()
        assert data1["steps"][0]["layers"][0]["intermediate_features"]["activation_std"] == original_std

        # Update same step
        payload["layer_statistics"][0]["intermediate_features"]["activation_std"] = 0.5
        client.post("/api/v1/metrics/layerwise", json=payload)

        # Get updated history
        response2 = client.get(f"/api/v1/runs/{run_id}")
        data2 = response2.json()
        assert data2["steps"][0]["layers"][0]["intermediate_features"]["activation_std"] == 0.5

        # Should still have only one step
        assert len(data2["steps"]) == 1


class TestTimeTravelWebSocketFlow:
    """Tests for complete WebSocket flow used by time travel feature."""

    def test_initial_runs_then_subscribe_flow(self, client: TestClient):
        """Test the initial_runs -> subscribe_run -> run_history flow."""
        # Create a run
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())

        with client.websocket_connect("/ws") as websocket:
            # 1. Receive initial_runs
            data1 = websocket.receive_json()
            assert data1["type"] == "initial_runs"
            assert "minimal_test" in data1["data"]

            # 2. Subscribe to run
            websocket.send_json({"action": "subscribe_run", "run_id": "minimal_test"})

            # 3. Receive run_history
            data2 = websocket.receive_json()
            assert data2["type"] == "run_history"
            assert data2["run_id"] == "minimal_test"

    def test_new_metrics_received_after_subscription(self, client: TestClient):
        """Test that new_metrics broadcasts are received after subscription."""
        run_id = "new_metrics_test"

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            # Subscribe
            websocket.send_json({"action": "subscribe_run", "run_id": run_id})
            websocket.receive_json()  # run_history (empty)

            # Submit new metric
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = 100
            client.post("/api/v1/metrics/layerwise", json=payload)

            # Should receive new_metrics
            data = websocket.receive_json()
            assert data["type"] == "new_metrics"
            assert data["run_id"] == run_id
            assert data["data"]["step"] == 100

    def test_multiple_subscriptions_same_connection(self, client: TestClient):
        """Test subscribing to multiple runs on same connection."""
        run_ids = ["multi_sub_1", "multi_sub_2"]

        # Create runs
        for run_id in run_ids:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            client.post("/api/v1/metrics/layerwise", json=payload)

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            # Subscribe to first run
            websocket.send_json({"action": "subscribe_run", "run_id": run_ids[0]})
            data1 = websocket.receive_json()
            assert data1["type"] == "run_history"
            assert data1["run_id"] == run_ids[0]

            # Subscribe to second run
            websocket.send_json({"action": "subscribe_run", "run_id": run_ids[1]})
            data2 = websocket.receive_json()
            assert data2["type"] == "run_history"
            assert data2["run_id"] == run_ids[1]

    def test_history_pagination_via_api(self, client: TestClient):
        """Test that complete history can be retrieved via API."""
        run_id = "pagination_test"

        # Create a run with many steps
        for step in range(100, 300, 10):
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        # Get all history via API
        response = client.get(f"/api/v1/runs/{run_id}")
        data = response.json()

        # Should have all steps
        assert len(data["steps"]) == 20
        step_numbers = [s["step"] for s in data["steps"]]
        expected = list(range(100, 300, 10))
        assert step_numbers == expected


class TestTimeTravelEdgeCases:
    """Tests for edge cases in time travel functionality."""

    def test_subscribe_to_nonexistent_run(self, client: TestClient):
        """Test subscribing to a run that doesn't exist."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            websocket.send_json({"action": "subscribe_run", "run_id": "nonexistent"})

            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "not found" in data["message"].lower()

    def test_empty_run_history(self, client: TestClient):
        """Test getting history for a run with no steps (edge case)."""
        # This is hard to test via HTTP since we can't create a run without steps
        # But we can verify the structure handles empty steps
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs (empty)

            # Try to subscribe to nonexistent run
            websocket.send_json({"action": "subscribe_run", "run_id": "empty_test"})

            data = websocket.receive_json()
            assert data["type"] == "error"

    def test_history_with_step_limit(self, client: TestClient):
        """Test history when step limit has been reached."""
        # This tests the time travel feature with storage limits
        # The mock store has max_steps_per_run=20
        run_id = "step_limit_history"

        # Add more than the limit
        for i in range(25):
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = i * 10
            client.post("/api/v1/metrics/layerwise", json=payload)

        # Get history - should have most recent 20 steps
        response = client.get(f"/api/v1/runs/{run_id}")
        data = response.json()

        assert len(data["steps"]) == 20
        # Should have the latest steps
        step_numbers = [s["step"] for s in data["steps"]]
        assert step_numbers == list(range(50, 250, 10))  # Last 20 steps

    def test_history_maintains_timestamp_order(self, client: TestClient):
        """Test that history maintains correct timestamp ordering."""
        run_id = "timestamp_order_test"

        # Submit with specific timestamps
        import time
        timestamps = []
        for i, step in enumerate([100, 200, 300]):
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            payload["metadata"]["timestamp"] = time.time() + i
            timestamps.append(payload["metadata"]["timestamp"])
            client.post("/api/v1/metrics/layerwise", json=payload)

        response = client.get(f"/api/v1/runs/{run_id}")
        data = response.json()

        # Steps should be sorted by step number, not timestamp
        step_numbers = [s["step"] for s in data["steps"]]
        assert step_numbers == [100, 200, 300]

        # But timestamps should be preserved
        actual_timestamps = [s["timestamp"] for s in data["steps"]]
        assert actual_timestamps == timestamps
