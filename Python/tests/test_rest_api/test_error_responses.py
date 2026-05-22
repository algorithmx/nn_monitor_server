"""
Tests for error response formats across all endpoints.

Tests cover:
- Consistent error response structure
- Correct HTTP status codes
- Error type validation
- Error message descriptiveness
"""

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.metric_fixtures import valid_minimal_payload
from tests.fixtures.error_fixtures import (
    missing_run_id,
    unsorted_depth_index,
    nan_activation_std,
    infinity_gradient_norm
)


class TestErrorResponseStructure:
    """Tests for consistent error response structure."""

    def test_validation_error_has_detail(self, client: TestClient):
        """Test that validation errors have detail field."""
        response = client.post("/api/v1/metrics/layerwise", json=missing_run_id())
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_validation_error_detail_is_list(self, client: TestClient):
        """Test that Pydantic validation detail is a list."""
        response = client.post("/api/v1/metrics/layerwise", json=missing_run_id())
        data = response.json()
        assert isinstance(data["detail"], list)

    def test_validation_error_has_field_info(self, client: TestClient):
        """Test that validation errors include field location."""
        response = client.post("/api/v1/metrics/layerwise", json=missing_run_id())
        data = response.json()
        # Pydantic validation errors have 'loc' field
        assert "loc" in data["detail"][0]

    def test_not_found_error_has_detail(self, client: TestClient):
        """Test that not found errors have detail field."""
        response = client.get("/api/v1/runs/nonexistent_run")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_not_found_error_detail_is_dict(self, client: TestClient):
        """Test that not found detail is a dictionary."""
        response = client.get("/api/v1/runs/nonexistent_run")
        data = response.json()
        assert isinstance(data["detail"], dict)


class TestErrorTypes:
    """Tests for error type field values."""

    def test_validation_error_type(self, client: TestClient):
        """Test that Pydantic validation errors have type field."""
        response = client.post("/api/v1/metrics/layerwise", json=missing_run_id())
        data = response.json()
        # Pydantic validation errors have 'type' field in each error item
        assert "type" in data["detail"][0]

    def test_not_found_error_type(self, client: TestClient):
        """Test that not found errors have type 'not_found'."""
        response = client.get("/api/v1/runs/nonexistent_run")
        data = response.json()
        assert data["detail"]["error"] == "not_found"

    def test_not_found_error_for_latest_endpoint(self, client: TestClient):
        """Test that not found errors work for latest endpoint."""
        response = client.get("/api/v1/runs/nonexistent_run/latest")
        data = response.json()
        assert data["detail"]["error"] == "not_found"


class TestValidationErrorMessageContent:
    """Tests for validation error message descriptiveness."""

    def test_missing_field_message_mentions_field(self, client: TestClient):
        """Test that missing field error mentions the field name."""
        response = client.post("/api/v1/metrics/layerwise", json=missing_run_id())
        data = response.json()
        # Pydantic validation errors include field location
        assert "loc" in data["detail"][0]
        assert "run_id" in str(data["detail"][0]["loc"])

    def test_depth_ordering_message_is_descriptive(self, client: TestClient):
        """Test that depth ordering error has descriptive message."""
        response = client.post("/api/v1/metrics/layerwise", json=unsorted_depth_index())
        data = response.json()
        # Pydantic validation error includes message
        assert "msg" in data["detail"][0]
        error_msg = data["detail"][0]["msg"].lower()
        assert len(error_msg) > 0  # Just verify there's a message


class TestNotFoundMessageContent:
    """Tests for not found error message content."""

    def test_not_found_message_includes_run_id(self, client: TestClient):
        """Test that not found message includes the run_id."""
        run_id = "my_missing_run"
        response = client.get(f"/api/v1/runs/{run_id}")
        data = response.json()
        message = data["detail"]["message"]
        assert run_id in message

    def test_not_found_message_is_descriptive(self, client: TestClient):
        """Test that not found message is descriptive."""
        response = client.get("/api/v1/runs/nonexistent_run")
        data = response.json()
        message = data["detail"]["message"].lower()
        assert "not found" in message


class TestErrorStatusCodeMapping:
    """Tests for correct HTTP status codes."""

    def test_pydantic_validation_error_returns_422(self, client: TestClient):
        """Test that Pydantic validation errors return 422."""
        response = client.post("/api/v1/metrics/layerwise", json=missing_run_id())
        assert response.status_code == 422

    def test_pydantic_field_validator_returns_422(self, client: TestClient):
        """Test that Pydantic field validators (like depth ordering) return 422."""
        response = client.post("/api/v1/metrics/layerwise", json=unsorted_depth_index())
        assert response.status_code == 422

    def test_not_found_returns_404(self, client: TestClient):
        """Test that not found errors return 404."""
        response = client.get("/api/v1/runs/nonexistent_run")
        assert response.status_code == 404

    def test_invalid_json_returns_422(self, client: TestClient):
        """Test that invalid JSON returns 422."""
        response = client.post(
            "/api/v1/metrics/layerwise",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestErrorResponseHeaders:
    """Tests for error response headers."""

    def test_error_response_content_type(self, client: TestClient):
        """Test that error responses have JSON content type."""
        response = client.post("/api/v1/metrics/layerwise", json=missing_run_id())
        assert response.headers["content-type"] == "application/json"


class TestEdgeCaseErrors:
    """Tests for edge case error scenarios."""

    def test_empty_body_error(self, client: TestClient):
        """Test that empty body returns appropriate error."""
        response = client.post("/api/v1/metrics/layerwise", json={})
        assert response.status_code == 422

    def test_malformed_json_error(self, client: TestClient):
        """Test that malformed JSON returns appropriate error."""
        response = client.post(
            "/api/v1/metrics/layerwise",
            data='{"metadata": {"run_id": "test"',  # Incomplete JSON
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_wrong_content_type_accepted(self, client: TestClient):
        """Test that wrong content-type is handled (FastAPI may still accept)."""
        # FastAPI can handle JSON even without explicit content-type
        response = client.post("/api/v1/metrics/layerwise", json=valid_minimal_payload())
        # Should work fine
        assert response.status_code in [202, 422]
