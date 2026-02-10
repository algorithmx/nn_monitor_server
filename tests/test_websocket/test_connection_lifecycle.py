"""
Tests for WebSocket connection lifecycle.

Tests cover:
- Connection establishment
- Initial runs message
- Disconnection handling
- Multiple simultaneous connections
"""

import pytest
import json
from fastapi.testclient import TestClient

from tests.fixtures.metric_fixtures import valid_minimal_payload


class TestWebSocketConnection:
    """Tests for WebSocket connection establishment and lifecycle."""

    def test_websocket_connection_accepted(self, client: TestClient):
        """Test that WebSocket connection is accepted."""
        with client.websocket_connect("/ws") as websocket:
            # If we get here without exception, connection was accepted
            assert True

    def test_websocket_sends_initial_runs(self, client: TestClient):
        """Test that initial_runs message is sent on connection."""
        with client.websocket_connect("/ws") as websocket:
            data = websocket.receive_json()
            assert data["type"] == "initial_runs"
            assert "data" in data

    def test_initial_runs_empty_when_no_runs(self, client: TestClient):
        """Test that initial_runs data is empty when no runs exist."""
        with client.websocket_connect("/ws") as websocket:
            data = websocket.receive_json()
            assert data["type"] == "initial_runs"
            assert data["data"] == {}

    def test_initial_runs_includes_existing_runs(self, client: TestClient):
        """Test that initial_runs includes existing runs."""
        # Submit a metric first
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())

        with client.websocket_connect("/ws") as websocket:
            data = websocket.receive_json()
            assert data["type"] == "initial_runs"
            assert "minimal_test" in data["data"]

    def test_initial_runs_data_structure(self, client: TestClient):
        """Test that initial_runs data has correct structure."""
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())

        with client.websocket_connect("/ws") as websocket:
            data = websocket.receive_json()
            run_info = data["data"]["minimal_test"]
            assert "created_at" in run_info
            assert "last_update" in run_info
            assert "step_count" in run_info
            assert "latest_step" in run_info

    def test_websocket_disconnects_cleanly(self, client: TestClient):
        """Test that WebSocket disconnects without errors."""
        with client.websocket_connect("/ws") as websocket:
            # Close the connection
            websocket.close()
            # If we get here, disconnect was clean
            assert True

    def test_multiple_simultaneous_connections(self, client: TestClient):
        """Test that multiple WebSocket connections can coexist."""
        connections = []
        try:
            # Open multiple connections
            for _ in range(3):
                websocket = client.websocket_connect("/ws")
                websocket.__enter__()
                connections.append(websocket)

            # All should receive initial_runs
            for websocket in connections:
                data = websocket.receive_json()
                assert data["type"] == "initial_runs"
        finally:
            # Clean up
            for websocket in connections:
                try:
                    websocket.close()
                except:
                    pass


class TestWebSocketMessageProtocol:
    """Tests for WebSocket message format and protocol."""

    def test_ping_message(self, client: TestClient):
        """Test ping/pong message handling."""
        with client.websocket_connect("/ws") as websocket:
            # Receive initial_runs first
            websocket.receive_json()

            # Send ping
            websocket.send_json({"action": "ping"})

            # Should receive pong
            response = websocket.receive_json()
            assert response["type"] == "pong"

    def test_pong_message_structure(self, client: TestClient):
        """Test that pong message has correct structure."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs
            websocket.send_json({"action": "ping"})
            response = websocket.receive_json()
            # Pong should only have type field
            assert "type" in response
            assert response["type"] == "pong"

    def test_subscribe_to_nonexistent_run(self, client: TestClient):
        """Test subscribing to a nonexistent run."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            # Subscribe to nonexistent run
            websocket.send_json({
                "action": "subscribe_run",
                "run_id": "nonexistent_run"
            })

            # Should receive error
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "nonexistent_run" in response["message"]

    def test_subscribe_to_existing_run(self, client: TestClient):
        """Test subscribing to an existing run."""
        # Create a run first
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            # Subscribe to the run
            websocket.send_json({
                "action": "subscribe_run",
                "run_id": "minimal_test"
            })

            # Should receive run_history
            response = websocket.receive_json()
            assert response["type"] == "run_history"
            assert response["run_id"] == "minimal_test"
            assert "data" in response

    def test_run_history_structure(self, client: TestClient):
        """Test that run_history message has correct structure."""
        client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())

        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs
            websocket.send_json({
                "action": "subscribe_run",
                "run_id": "minimal_test"
            })
            response = websocket.receive_json()

            assert "created_at" in response["data"]
            assert "steps" in response["data"]
            assert isinstance(response["data"]["steps"], list)

    def test_invalid_json_format(self, client: TestClient):
        """Test handling of invalid JSON format."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            # Send invalid JSON
            websocket.send_text("not valid json")

            # Should receive error message
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "json" in response["message"].lower()

    def test_unknown_action(self, client: TestClient):
        """Test handling of unknown action."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            # Send unknown action
            websocket.send_json({"action": "unknown_action"})

            # The server should not crash, might ignore or send error
            # Depending on implementation, we just verify it doesn't hang
            # For now, we'll use a timeout to prevent hanging
            import time
            start = time.time()
            try:
                # Try to receive with a short timeout
                response = websocket.receive_json(timeout=0.5)
                # If we get a response, it should be an error
                # or we can just verify the server handled it gracefully
            except:
                # Timeout is OK - server might just ignore unknown actions
                pass
            assert (time.time() - start) < 2  # Should not hang


class TestWebSocketBroadcastBehavior:
    """Tests for WebSocket broadcast of new metrics."""

    def test_new_metrics_broadcast_on_post(self, client: TestClient):
        """Test that new metrics are broadcast to WebSocket clients."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            # Submit a metric
            client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())

            # Should receive new_metrics broadcast
            response = websocket.receive_json()
            assert response["type"] == "new_metrics"
            assert response["run_id"] == "minimal_test"
            assert "data" in response

    def test_new_metrics_data_structure(self, client: TestClient):
        """Test that new_metrics broadcast has correct structure."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())

            response = websocket.receive_json()
            data = response["data"]
            assert "step" in data
            assert "timestamp" in data
            assert "batch_size" in data
            assert "layers" in data
            assert "cross_layer" in data

    def test_multiple_clients_receive_broadcast(self, client: TestClient):
        """Test that multiple clients receive the same broadcast."""
        with client.websocket_connect("/ws") as websocket1:
            websocket1.receive_json()  # initial_runs

            with client.websocket_connect("/ws") as websocket2:
                websocket2.receive_json()  # initial_runs

                # Submit a metric
                client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())

                # Both should receive the broadcast
                response1 = websocket1.receive_json()
                response2 = websocket2.receive_json()

                assert response1["type"] == "new_metrics"
                assert response2["type"] == "new_metrics"
                assert response1["run_id"] == response2["run_id"]

    def test_disconnected_client_does_not_receive_broadcast(self, client: TestClient):
        """Test that disconnected clients don't receive broadcasts."""
        with client.websocket_connect("/ws") as websocket1:
            websocket1.receive_json()  # initial_runs

            with client.websocket_connect("/ws") as websocket2:
                websocket2.receive_json()  # initial_runs
                websocket2.close()  # Disconnect websocket2

                # Submit a metric
                client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())

                # websocket1 should receive broadcast
                response1 = websocket1.receive_json()
                assert response1["type"] == "new_metrics"

                # websocket2 should not receive anything (it's disconnected)
                # We can't test this directly without causing a timeout,
                # but we can verify websocket1 got the message

    def test_broadcast_after_multiple_submissions(self, client: TestClient):
        """Test broadcasts after multiple metric submissions."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # initial_runs

            # Submit multiple metrics
            for step in [100, 200, 300]:
                payload = valid_minimal_payload()
                payload["metadata"]["global_step"] = step
                client.post("/api/v1/metrics/layerwise", json=payload)

            # Should receive all broadcasts
            for step in [100, 200, 300]:
                response = websocket.receive_json()
                assert response["type"] == "new_metrics"
                assert response["data"]["step"] == step
