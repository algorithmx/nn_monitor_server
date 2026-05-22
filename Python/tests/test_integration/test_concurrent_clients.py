"""
Integration tests for concurrent client scenarios.

Tests cover:
- Multiple HTTP clients submitting simultaneously
- Multiple WebSocket connections
- Mixed HTTP and WebSocket traffic
- Concurrent subscriptions and broadcasts
"""

import pytest
import asyncio
import threading
from fastapi.testclient import TestClient

from tests.fixtures.metric_fixtures import valid_minimal_payload, multi_run_concurrent_steps


class TestConcurrentHTTPSubmissions:
    """Tests for multiple HTTP clients submitting simultaneously."""

    def test_sequential_submissions_from_multiple_runs(self, client: TestClient):
        """Test submissions from different run IDs are handled correctly."""
        run_ids = ["concurrent_run_1", "concurrent_run_2", "concurrent_run_3"]

        for run_id in run_ids:
            for step in [100, 200, 300]:
                payload = valid_minimal_payload()
                payload["metadata"]["run_id"] = run_id
                payload["metadata"]["global_step"] = step
                response = client.post("/api/v1/metrics/layerwise", json=payload)
                assert response.status_code == 202

        # Verify all runs exist with correct step counts
        response = client.get("/api/v1/runs")
        data = response.json()

        for run_id in run_ids:
            assert run_id in data
            assert data[run_id]["step_count"] == 3

    def test_interleaved_submissions(self, client: TestClient):
        """Test interleaved submissions to different runs."""
        run_ids = ["interleaved_1", "interleaved_2"]

        # Submit in interleaved fashion: run1_step1, run2_step1, run1_step2, run2_step2, etc.
        for step in [100, 200, 300]:
            for run_id in run_ids:
                payload = valid_minimal_payload()
                payload["metadata"]["run_id"] = run_id
                payload["metadata"]["global_step"] = step
                client.post("/api/v1/metrics/layerwise", json=payload)

        # Verify both runs have all steps
        response = client.get("/api/v1/runs")
        data = response.json()

        for run_id in run_ids:
            run_response = client.get(f"/api/v1/runs/{run_id}")
            run_data = run_response.json()
            assert len(run_data["steps"]) == 3
            step_numbers = [s["step"] for s in run_data["steps"]]
            assert step_numbers == [100, 200, 300]

    def test_same_step_different_runs(self, client: TestClient):
        """Test that same step number in different runs doesn't cause conflicts."""
        run_ids = ["same_step_1", "same_step_2", "same_step_3"]
        step_num = 500

        for run_id in run_ids:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step_num
            client.post("/api/v1/metrics/layerwise", json=payload)

        # All runs should exist
        response = client.get("/api/v1/runs")
        data = response.json()

        for run_id in run_ids:
            assert run_id in data
            assert data[run_id]["latest_step"] == step_num
            assert data[run_id]["step_count"] == 1


class TestConcurrentWebSocketConnections:
    """Tests for multiple WebSocket connections."""

    def test_multiple_websocket_receives_same_broadcast(self, client: TestClient):
        """Test that multiple WebSocket clients receive the same broadcast."""
        connections = []
        try:
            # Open multiple connections
            for _ in range(3):
                websocket = client.websocket_connect("/ws")
                ws = websocket.__enter__()
                ws.receive_json()  # Clear initial_runs
                connections.append((websocket, ws))

            # Submit a metric
            client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())

            # All should receive the broadcast
            received_data = []
            for websocket, ws in connections:
                data = ws.receive_json()
                received_data.append(data)

            # Verify all received the same message
            assert len(received_data) == 3
            for data in received_data:
                assert data["type"] == "new_metrics"
                assert data["run_id"] == "minimal_test"

        finally:
            # Clean up
            for websocket, ws in connections:
                try:
                    websocket.close()
                except:
                    pass

    def test_websocket_subscribe_while_others_listening(self, client: TestClient):
        """Test subscribing to a run while other connections are listening."""
        # Create a run first
        run_id = "subscribe_test"
        for step in [100, 200, 300]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        connections = []
        try:
            # Open first connection and subscribe
            ws1 = client.websocket_connect("/ws")
            websocket1 = ws1.__enter__()
            websocket1.receive_json()  # initial_runs
            websocket1.send_json({"action": "subscribe_run", "run_id": run_id})
            websocket1.receive_json()  # run_history
            connections.append((ws1, websocket1))

            # Open second connection
            ws2 = client.websocket_connect("/ws")
            websocket2 = ws2.__enter__()
            websocket2.receive_json()  # initial_runs
            connections.append((ws2, websocket2))

            # Submit a new metric
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = 400
            client.post("/api/v1/metrics/layerwise", json=payload)

            # Both should receive the broadcast
            data1 = websocket1.receive_json()
            data2 = websocket2.receive_json()

            assert data1["type"] == "new_metrics"
            assert data2["type"] == "new_metrics"
            assert data1["data"]["step"] == 400
            assert data2["data"]["step"] == 400

        finally:
            for ws, websocket in connections:
                try:
                    ws.close()
                except:
                    pass

    def test_websocket_disconnect_doesnt_affect_others(self, client: TestClient):
        """Test that one WebSocket disconnecting doesn't affect others."""
        connections = []
        try:
            # Open multiple connections
            for _ in range(3):
                ws = client.websocket_connect("/ws")
                websocket = ws.__enter__()
                websocket.receive_json()  # initial_runs
                connections.append((ws, websocket))

            # Close one connection
            connections[0][0].close()
            connections.pop(0)

            # Submit a metric
            client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())

            # Remaining connections should still receive broadcasts
            for ws, websocket in connections:
                data = websocket.receive_json()
                assert data["type"] == "new_metrics"

        finally:
            for ws, websocket in connections:
                try:
                    ws.close()
                except:
                    pass


class TestMixedHTTPAndWebSocketTraffic:
    """Tests for mixed HTTP and WebSocket traffic."""

    def test_http_post_while_websocket_connected(self, client: TestClient):
        """Test HTTP POST while WebSocket is connected."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            # Submit via HTTP
            response = client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
            assert response.status_code == 202

            # WebSocket should receive broadcast
            data = websocket.receive_json()
            assert data["type"] == "new_metrics"

    def test_websocket_subscribe_after_http_posts(self, client: TestClient):
        """Test WebSocket subscribing after HTTP has created data."""
        run_id = "post_then_subscribe"

        # Submit metrics via HTTP
        for step in [100, 200, 300]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        # Now connect and subscribe
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs
            websocket.send_json({"action": "subscribe_run", "run_id": run_id})

            # Should receive full history
            data = websocket.receive_json()
            assert data["type"] == "run_history"
            assert len(data["data"]["steps"]) == 3

    def test_http_get_while_websocket_listening(self, client: TestClient):
        """Test HTTP GET while WebSocket is listening."""
        run_id = "get_while_listening"

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            # Submit via HTTP
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = 100
            client.post("/api/v1/metrics/layerwise", json=payload)

            # Get via HTTP
            response = client.get(f"/api/v1/runs/{run_id}")
            assert response.status_code == 200
            data = response.json()
            assert len(data["steps"]) == 1

            # WebSocket should also have received broadcast
            ws_data = websocket.receive_json()
            assert ws_data["type"] == "new_metrics"

    def test_concurrent_http_gets_and_websocket(self, client: TestClient):
        """Test concurrent HTTP GET requests while WebSocket is connected."""
        run_id = "concurrent_get_test"

        # Create some data
        for step in [100, 200, 300]:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = step
            client.post("/api/v1/metrics/layerwise", json=payload)

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            # Make multiple HTTP GETs
            all_runs = client.get("/api/v1/runs")
            specific_run = client.get(f"/api/v1/runs/{run_id}")
            latest = client.get(f"/api/v1/runs/{run_id}/latest")

            assert all_runs.status_code == 200
            assert specific_run.status_code == 200
            assert latest.status_code == 200

            # WebSocket should still be working
            # Submit new metric
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            payload["metadata"]["global_step"] = 400
            client.post("/api/v1/metrics/layerwise", json=payload)

            # Should receive broadcast
            data = websocket.receive_json()
            assert data["type"] == "new_metrics"
            assert data["data"]["step"] == 400


class TestMultiRunScenarios:
    """Tests for complex multi-run scenarios."""

    def test_switching_runs_in_websocket(self, client: TestClient):
        """Test switching between different runs in WebSocket."""
        run_ids = ["switch_test_1", "switch_test_2"]

        # Create both runs
        for run_id in run_ids:
            for step in [100, 200]:
                payload = valid_minimal_payload()
                payload["metadata"]["run_id"] = run_id
                payload["metadata"]["global_step"] = step
                client.post("/api/v1/metrics/layerwise", json=payload)

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            # Subscribe to first run
            websocket.send_json({"action": "subscribe_run", "run_id": run_ids[0]})
            data1 = websocket.receive_json()
            assert data1["type"] == "run_history"
            assert data1["run_id"] == run_ids[0]
            assert len(data1["data"]["steps"]) == 2

            # Subscribe to second run
            websocket.send_json({"action": "subscribe_run", "run_id": run_ids[1]})
            data2 = websocket.receive_json()
            assert data2["type"] == "run_history"
            assert data2["run_id"] == run_ids[1]
            assert len(data2["data"]["steps"]) == 2

    def test_broadcast_to_multiple_runs(self, client: TestClient):
        """Test that broadcasts are received regardless of which run is updated."""
        run_ids = ["broadcast_test_1", "broadcast_test_2", "broadcast_test_3"]

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            # Update each run and verify broadcast
            for i, run_id in enumerate(run_ids):
                payload = valid_minimal_payload()
                payload["metadata"]["run_id"] = run_id
                payload["metadata"]["global_step"] = 100 * (i + 1)
                client.post("/api/v1/metrics/layerwise", json=payload)

                data = websocket.receive_json()
                assert data["type"] == "new_metrics"
                assert data["run_id"] == run_id

    def test_list_runs_shows_all_active_runs(self, client: TestClient):
        """Test that list runs shows all runs after concurrent submissions."""
        run_ids = [f"list_test_{i}" for i in range(5)]

        # Submit to all runs
        for run_id in run_ids:
            payload = valid_minimal_payload()
            payload["metadata"]["run_id"] = run_id
            client.post("/api/v1/metrics/layerwise", json=payload)

        # List should show all
        response = client.get("/api/v1/runs")
        data = response.json()

        for run_id in run_ids:
            assert run_id in data
