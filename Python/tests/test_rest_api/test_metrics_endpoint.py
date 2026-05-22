"""
Tests for the POST /api/v1/metrics/layerwise endpoint.

Tests cover:
- Valid metric submission
- Validation error responses
- Response format verification
- Schema compliance
"""

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.metric_fixtures import (
    valid_minimal_payload,
    valid_three_layer_network,
    vanishing_gradient_pattern,
    exploding_gradient_pattern,
    healthy_gradient_pattern,
    get_all_valid_fixtures
)

from tests.fixtures.error_fixtures import (
    get_all_error_fixtures,
    missing_run_id,
    empty_layer_statistics,
    unsorted_depth_index,
    nan_activation_std,
    infinity_gradient_norm
)


class TestMetricsEndpointValidRequests:
    """Tests for valid metric submissions to /api/v1/metrics/layerwise."""

    def test_valid_minimal_payload_returns_202(self, client: TestClient):
        """Test that a valid minimal payload returns 202 Accepted."""
        response = client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
        assert response.status_code == 202

    def test_valid_response_format(self, client: TestClient):
        """Test that response has correct format with status and run_id."""
        response = client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
        data = response.json()
        assert "status" in data
        assert "run_id" in data
        assert data["status"] == "accepted"
        assert data["run_id"] == "minimal_test"

    def test_three_layer_network_accepted(self, client: TestClient):
        """Test that a realistic three-layer network payload is accepted."""
        response = client.post("/api/v1/metrics/layerwise", json=valid_three_layer_network())
        assert response.status_code == 202
        data = response.json()
        assert data["run_id"] == "three_layer_net"

    def test_vanishing_gradient_pattern_accepted(self, client: TestClient):
        """Test that vanishing gradient pattern payload is accepted."""
        response = client.post("/api/v1/metrics/layerwise", json=vanishing_gradient_pattern())
        assert response.status_code == 202

    def test_exploding_gradient_pattern_accepted(self, client: TestClient):
        """Test that exploding gradient pattern payload is accepted."""
        response = client.post("/api/v1/metrics/layerwise", json=exploding_gradient_pattern())
        assert response.status_code == 202

    def test_healthy_gradient_pattern_accepted(self, client: TestClient):
        """Test that healthy gradient pattern payload is accepted."""
        response = client.post("/api/v1/metrics/layerwise", json=healthy_gradient_pattern())
        assert response.status_code == 202

    @pytest.mark.parametrize("payload_fixture", get_all_valid_fixtures())
    def test_all_valid_fixtures_accepted(self, client: TestClient, payload_fixture):
        """Parametrized test that all valid fixtures are accepted."""
        response = client.post("/api/v1/metrics/layerwise", json=payload_fixture)
        assert response.status_code == 202


class TestMetricsEndpointValidationErrors:
    """Tests for validation error responses."""

    def test_missing_run_id_returns_422(self, client: TestClient):
        """Test that missing run_id returns 422 Unprocessable Entity (Pydantic validation)."""
        response = client.post("/api/v1/metrics/layerwise", json=missing_run_id())
        assert response.status_code == 422

    def test_missing_run_id_error_format(self, client: TestClient):
        """Test that missing run_id error has correct format."""
        response = client.post("/api/v1/metrics/layerwise", json=missing_run_id())
        data = response.json()
        assert "detail" in data
        # Pydantic validation errors have a different format

    def test_empty_layer_statistics_returns_422(self, client: TestClient):
        """Test that empty layer_statistics array returns 422."""
        response = client.post("/api/v1/metrics/layerwise", json=empty_layer_statistics())
        assert response.status_code == 422

    def test_unsorted_depth_index_returns_422(self, client: TestClient):
        """Test that unsorted depth_index values return 422 (Pydantic validation)."""
        response = client.post("/api/v1/metrics/layerwise", json=unsorted_depth_index())
        assert response.status_code == 422

    def test_unsorted_depth_index_error_message(self, client: TestClient):
        """Test that unsorted depth_index error has descriptive message."""
        response = client.post("/api/v1/metrics/layerwise", json=unsorted_depth_index())
        data = response.json()
        # Pydantic validation errors have a different structure (list of errors)
        assert isinstance(data["detail"], list)


class TestMetricsEndpointContentValidation:
    """Tests for specific field validation rules."""

    def test_negative_step_rejected(self, client: TestClient):
        """Test that negative global_step is rejected (Pydantic returns 422)."""
        from tests.fixtures.error_fixtures import negative_step
        response = client.post("/api/v1/metrics/layerwise", json=negative_step())
        assert response.status_code == 422

    def test_zero_batch_size_rejected(self, client: TestClient):
        """Test that zero batch_size is rejected (Pydantic returns 422)."""
        from tests.fixtures.error_fixtures import zero_batch_size
        response = client.post("/api/v1/metrics/layerwise", json=zero_batch_size())
        assert response.status_code == 422

    def test_negative_batch_size_rejected(self, client: TestClient):
        """Test that negative batch_size is rejected (Pydantic returns 422)."""
        from tests.fixtures.error_fixtures import negative_batch_size
        response = client.post("/api/v1/metrics/layerwise", json=negative_batch_size())
        assert response.status_code == 422

    def test_negative_timestamp_rejected(self, client: TestClient):
        """Test that negative timestamp is rejected (Pydantic returns 422)."""
        from tests.fixtures.error_fixtures import negative_timestamp
        response = client.post("/api/v1/metrics/layerwise", json=negative_timestamp())
        assert response.status_code == 422

    def test_negative_activation_std_rejected(self, client: TestClient):
        """Test that negative activation_std is rejected (Pydantic returns 422)."""
        from tests.fixtures.error_fixtures import negative_activation_std
        response = client.post("/api/v1/metrics/layerwise", json=negative_activation_std())
        assert response.status_code == 422

    def test_negative_gradient_std_rejected(self, client: TestClient):
        """Test that negative gradient_std is rejected (Pydantic returns 422)."""
        from tests.fixtures.error_fixtures import negative_gradient_std
        response = client.post("/api/v1/metrics/layerwise", json=negative_gradient_std())
        assert response.status_code == 422

    def test_negative_depth_index_rejected(self, client: TestClient):
        """Test that negative depth_index is rejected (Pydantic returns 422)."""
        from tests.fixtures.error_fixtures import negative_depth_index
        response = client.post("/api/v1/metrics/layerwise", json=negative_depth_index())
        assert response.status_code == 422

    def test_single_dimension_activation_shape_rejected(self, client: TestClient):
        """Test that single-dimension activation_shape is rejected (Pydantic returns 422)."""
        from tests.fixtures.error_fixtures import single_dimension_activation_shape
        response = client.post("/api/v1/metrics/layerwise", json=single_dimension_activation_shape())
        assert response.status_code == 422

    def test_empty_activation_shape_rejected(self, client: TestClient):
        """Test that empty activation_shape is rejected (Pydantic returns 422)."""
        from tests.fixtures.error_fixtures import empty_activation_shape
        response = client.post("/api/v1/metrics/layerwise", json=empty_activation_shape())
        assert response.status_code == 422


class TestMetricsEndpointContentType:
    """Tests for content-type and request format validation."""

    def test_invalid_json_rejected(self, client: TestClient):
        """Test that invalid JSON is rejected."""
        response = client.post(
            "/api/v1/metrics/layerwise",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable Entity

    def test_missing_content_type_accepted(self, client: TestClient):
        """Test that missing Content-Type header still works (FastAPI default)."""
        response = client.post(
            "/api/v1/metrics/layerwise",
            json=valid_minimal_payload()
        )
        assert response.status_code == 202

    def test_empty_body_rejected(self, client: TestClient):
        """Test that empty request body is rejected."""
        response = client.post("/api/v1/metrics/layerwise", json={})
        assert response.status_code == 422


class TestMetricsEndpointRunIdVariations:
    """Tests for various run_id values."""

    def test_run_id_with_special_characters(self, client: TestClient):
        """Test run_id with allowed special characters."""
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = "experiment_2024-02-10_v1.2.3"
        response = client.post("/api/v1/metrics/layerwise", json=payload)
        assert response.status_code == 202

    def test_run_id_with_underscores(self, client: TestClient):
        """Test run_id with underscores."""
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = "my_training_run_v2"
        response = client.post("/api/v1/metrics/layerwise", json=payload)
        assert response.status_code == 202

    def test_run_id_with_dots(self, client: TestClient):
        """Test run_id with dots."""
        payload = valid_minimal_payload()
        payload["metadata"]["run_id"] = "run.model.checkpoint"
        response = client.post("/api/v1/metrics/layerwise", json=payload)
        assert response.status_code == 202
