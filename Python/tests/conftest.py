"""
Shared fixtures and configuration for NN Monitor Server tests.

This module provides pytest fixtures for testing the Neural Network Training
Monitor Server with deterministic test data and isolated test environments.
"""

import sys
import os
import asyncio
import pytest
import time
from typing import AsyncGenerator, Generator, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app, store, manager, MetricsStore, ConnectionManager


# ==================== Test Configuration ====================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def mock_store():
    """
    Provide a fresh MetricsStore instance for each test function.

    This fixture ensures complete isolation between tests by creating
    a new in-memory store with minimal storage limits for testing.
    """
    # Create a new store with small limits for testing storage eviction
    test_store = MetricsStore(max_runs=5, max_steps_per_run=20)
    yield test_store
    # Cleanup happens automatically when the store is garbage collected


@pytest.fixture(scope="function")
def mock_manager():
    """Provide a fresh ConnectionManager instance for each test function."""
    test_manager = ConnectionManager()
    yield test_manager


@pytest.fixture(scope="function")
def test_config():
    """
    Provide test configuration values.

    Returns a dictionary with test-specific configuration that matches
    the ServerConfig defaults but can be overridden in specific tests.
    """
    return {
        "max_runs": 5,
        "max_steps_per_run": 20,
        "max_request_size": 2_000_000,
        "test_host": "127.0.0.1",
        "test_port": 8765
    }


@pytest.fixture(scope="function")
def client(mock_store, mock_manager):
    """
    Provide a TestClient with injected mock dependencies.

    This fixture creates a test client that uses isolated store and manager
    instances to prevent test interference.
    """
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock, patch

    # Create a fresh app instance for testing
    from fastapi import FastAPI
    test_app = FastAPI(title="NN Training Monitor Test")

    # Mock the dependencies
    with patch('main.store', mock_store), \
         patch('main.manager', mock_manager):

        # Import and setup routes after patching
        import main

        # Override the store and manager in the main module
        original_store = main.store
        original_manager = main.manager
        main.store = mock_store
        main.manager = mock_manager

        # Create test client with the real app but mocked dependencies
        test_client = TestClient(app)

        yield test_client

        # Restore original dependencies
        main.store = original_store
        main.manager = original_manager


# ==================== WebSocket Fixtures ====================

@pytest.fixture
async def websocket_connection(mock_store, mock_manager):
    """
    Provide a WebSocket connection for testing.

    This fixture creates a test WebSocket connection that can be used
    to test WebSocket message handling and broadcast behavior.
    """
    from fastapi.testclient import TestClient
    from fastapi import WebSocket

    # Create a test client with WebSocket support
    test_client = TestClient(app)

    with test_client.websocket_connect("/ws") as websocket:
        yield websocket


# ==================== Helper Functions ====================

@pytest.fixture
def create_sample_payload():
    """
    Factory function for creating sample metrics payloads.

    Returns a function that generates valid MetricsPayload dictionaries
    with customizable parameters.
    """
    def _create(
        run_id: str = "test_run",
        step: int = 100,
        timestamp: float = None,
        batch_size: int = 64,
        layer_count: int = 3
    ) -> Dict[str, Any]:
        if timestamp is None:
            timestamp = time.time()

        # Generate layer statistics
        layers = []
        for i in range(layer_count):
            layer = {
                "layer_id": f"layer_{i}",
                "layer_type": "Linear" if i % 2 == 0 else "ReLU",
                "depth_index": i,
                "intermediate_features": {
                    "activation_std": 0.8 - (i * 0.1),
                    "activation_mean": -0.02 + (i * 0.01),
                    "activation_shape": [batch_size, 256 // (2 ** i)],
                    "cross_layer_std_ratio": None if i == 0 else 0.9 - (i * 0.05)
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.15 - (i * 0.02),
                    "gradient_std": 0.003 - (i * 0.0005),
                    "gradient_max_abs": 0.09 - (i * 0.01)
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.04 - (i * 0.005),
                        "mean": -0.001 + (i * 0.0005),
                        "spectral_norm": 1.4 - (i * 0.1),
                        "frobenius_norm": 2.2 - (i * 0.2)
                    },
                    "bias": {
                        "std": 0.01 + (i * 0.001),
                        "mean_abs": 0.008 - (i * 0.001)
                    }
                }
            }
            layers.append(layer)

        # Generate cross-layer analysis
        gradient_ratios = {}
        for i in range(1, layer_count):
            gradient_ratios[f"layer_{i}_to_prev"] = 0.9 - (i * 0.1)

        payload = {
            "metadata": {
                "run_id": run_id,
                "timestamp": timestamp,
                "global_step": step,
                "batch_size": batch_size
            },
            "layer_statistics": layers,
            "cross_layer_analysis": {
                "feature_std_gradient": -0.05 - (layer_count * 0.01),
                "gradient_norm_ratio": gradient_ratios
            }
        }

        return payload

    return _create


@pytest.fixture
def populate_run(mock_store):
    """
    Helper to populate a run with multiple steps.

    Returns an async function that creates a run with the specified
    number of steps using the store's add_metrics method.
    """
    async def _populate(
        run_id: str = "populated_run",
        step_count: int = 10,
        start_step: int = 0,
        interval: float = 1.0
    ) -> None:
        """Populate a run with the given number of steps."""
        from main import MetricsPayload, Metadata, LayerStatistic, \
            IntermediateFeatures, GradientFlow, ParameterStatistics, \
            WeightStats, BiasStats, CrossLayerAnalysis

        for i in range(step_count):
            step_num = start_step + i
            timestamp = time.time() - ((step_count - i) * interval)

            # Create a simple valid payload
            payload = MetricsPayload(
                metadata=Metadata(
                    run_id=run_id,
                    timestamp=timestamp,
                    global_step=step_num,
                    batch_size=64
                ),
                layer_statistics=[
                    LayerStatistic(
                        layer_id=f"layer_0",
                        layer_type="Linear",
                        depth_index=0,
                        intermediate_features=IntermediateFeatures(
                            activation_std=0.8,
                            activation_mean=-0.02,
                            activation_shape=[64, 256],
                            cross_layer_std_ratio=None
                        ),
                        gradient_flow=GradientFlow(
                            gradient_l2_norm=0.15,
                            gradient_std=0.003,
                            gradient_max_abs=0.09
                        ),
                        parameter_statistics=ParameterStatistics(
                            weight=WeightStats(
                                std=0.04,
                                mean=-0.001,
                                spectral_norm=1.4,
                                frobenius_norm=2.2
                            ),
                            bias=BiasStats(
                                std=0.01,
                                mean_abs=0.008
                            )
                        )
                    )
                ],
                cross_layer_analysis=CrossLayerAnalysis(
                    feature_std_gradient=-0.05,
                    gradient_norm_ratio={}
                )
            )

            await mock_store.add_metrics(payload)

    return _populate


@pytest.fixture
def wait_for_broadcast():
    """
    Helper to wait for WebSocket broadcast to complete.

    Returns an async function that waits a small amount of time
    to ensure broadcasts have been processed.
    """
    async def _wait(delay: float = 0.1):
        """Wait for the specified delay."""
        await asyncio.sleep(delay)

    return _wait
