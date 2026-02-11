"""
Deterministic metric fixtures for testing edge cases and validation rules.

This module provides pre-defined realistic test data scenarios that cover:
- Valid metric payloads with various characteristics
- Edge cases for storage behavior (deduplication, eviction, limits)
- Gradient patterns (vanishing, exploding, healthy)
- Cross-layer analysis scenarios
"""

import math
import time
from typing import Dict, Any, List


# ==================== Valid Metric Fixtures ====================

def valid_minimal_payload() -> Dict[str, Any]:
    """
    Minimal valid payload with single layer.

    Tests the basic required fields without optional data.
    """
    return {
        "metadata": {
            "run_id": "minimal_test",
            "timestamp": 1707589200.123,
            "global_step": 100,
            "batch_size": 32
        },
        "layer_statistics": [
            {
                "layer_id": "single_layer",
                "layer_type": "Linear",
                "depth_index": 0,
                "intermediate_features": {
                    "activation_std": 0.5,
                    "activation_mean": 0.0,
                    "activation_shape": [32, 128],
                    "cross_layer_std_ratio": None
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.1,
                    "gradient_std": 0.01,
                    "gradient_max_abs": 0.05
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.1,
                        "mean": 0.0,
                        "spectral_norm": 1.0,
                        "frobenius_norm": 1.5
                    },
                    "bias": {
                        "std": 0.01,
                        "mean_abs": 0.005
                    }
                }
            }
        ],
        "cross_layer_analysis": {
            "feature_std_gradient": -0.02,
            "gradient_norm_ratio": {}
        }
    }


def valid_three_layer_network() -> Dict[str, Any]:
    """
    Realistic three-layer network (Linear-ReLU-Linear).

    Tests proper depth ordering and cross-layer ratios.
    Includes layer_groups to test grouping functionality.
    """
    return {
        "metadata": {
            "run_id": "three_layer_net",
            "timestamp": 1707589200.123,
            "global_step": 500,
            "batch_size": 64,
            "layer_groups": {
                "encoder": ["encoder/linear1", "encoder/relu1"],
                "decoder": ["encoder/linear2"]
            }
        },
        "layer_statistics": [
            {
                "layer_id": "encoder/linear1",
                "layer_type": "Linear",
                "depth_index": 0,
                "intermediate_features": {
                    "activation_std": 0.847,
                    "activation_mean": -0.023,
                    "activation_shape": [64, 256],
                    "cross_layer_std_ratio": None
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.152,
                    "gradient_std": 0.0034,
                    "gradient_max_abs": 0.089
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.037,
                        "mean": -0.001,
                        "spectral_norm": 1.42,
                        "frobenius_norm": 2.18
                    },
                    "bias": {
                        "std": 0.012,
                        "mean_abs": 0.008
                    }
                }
            },
            {
                "layer_id": "encoder/relu1",
                "layer_type": "ReLU",
                "depth_index": 1,
                "intermediate_features": {
                    "activation_std": 0.795,
                    "activation_mean": 0.415,
                    "activation_shape": [64, 256],
                    "cross_layer_std_ratio": 0.938
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.089,
                    "gradient_std": 0.0021,
                    "gradient_max_abs": 0.042
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.0,
                        "mean": 0.0,
                        "spectral_norm": 0.0,
                        "frobenius_norm": 0.0
                    },
                    "bias": {
                        "std": 0.0,
                        "mean_abs": 0.0
                    }
                }
            },
            {
                "layer_id": "encoder/linear2",
                "layer_type": "Linear",
                "depth_index": 2,
                "intermediate_features": {
                    "activation_std": 0.247,
                    "activation_mean": 0.012,
                    "activation_shape": [64, 128],
                    "cross_layer_std_ratio": 0.31
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.047,
                    "gradient_std": 0.0015,
                    "gradient_max_abs": 0.023
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.029,
                        "mean": 0.003,
                        "spectral_norm": 1.15,
                        "frobenius_norm": 1.67
                    },
                    "bias": {
                        "std": 0.009,
                        "mean_abs": 0.005
                    }
                }
            }
        ],
        "cross_layer_analysis": {
            "feature_std_gradient": -0.2,
            "gradient_norm_ratio": {
                "encoder/relu1_to_prev": 0.586,
                "encoder/linear2_to_prev": 0.528
            }
        }
    }


def vanishing_gradient_pattern() -> Dict[str, Any]:
    """
    Network exhibiting vanishing gradient pattern.

    Characterized by:
    - Strongly negative feature_std_gradient (< -0.1)
    - Decreasing gradient norms with depth
    - Low activation std in deeper layers
    """
    return {
        "metadata": {
            "run_id": "vanishing_gradient_test",
            "timestamp": 1707589200.123,
            "global_step": 1000,
            "batch_size": 128
        },
        "layer_statistics": [
            {
                "layer_id": "layer_0",
                "layer_type": "Linear",
                "depth_index": 0,
                "intermediate_features": {
                    "activation_std": 1.2,
                    "activation_mean": 0.1,
                    "activation_shape": [128, 512],
                    "cross_layer_std_ratio": None
                },
                "gradient_flow": {
                    "gradient_l2_norm": 1.5,
                    "gradient_std": 0.1,
                    "gradient_max_abs": 0.8
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.15,
                        "mean": 0.0,
                        "spectral_norm": 2.5,
                        "frobenius_norm": 5.0
                    },
                    "bias": {
                        "std": 0.05,
                        "mean_abs": 0.02
                    }
                }
            },
            {
                "layer_id": "layer_1",
                "layer_type": "ReLU",
                "depth_index": 1,
                "intermediate_features": {
                    "activation_std": 0.6,
                    "activation_mean": 0.3,
                    "activation_shape": [128, 512],
                    "cross_layer_std_ratio": 0.5
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.3,
                    "gradient_std": 0.03,
                    "gradient_max_abs": 0.2
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.0,
                        "mean": 0.0,
                        "spectral_norm": 0.0,
                        "frobenius_norm": 0.0
                    },
                    "bias": {
                        "std": 0.0,
                        "mean_abs": 0.0
                    }
                }
            },
            {
                "layer_id": "layer_2",
                "layer_type": "Linear",
                "depth_index": 2,
                "intermediate_features": {
                    "activation_std": 0.15,
                    "activation_mean": 0.01,
                    "activation_shape": [128, 256],
                    "cross_layer_std_ratio": 0.25
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.03,
                    "gradient_std": 0.003,
                    "gradient_max_abs": 0.015
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.08,
                        "mean": 0.0,
                        "spectral_norm": 0.8,
                        "frobenius_norm": 1.2
                    },
                    "bias": {
                        "std": 0.01,
                        "mean_abs": 0.005
                    }
                }
            },
            {
                "layer_id": "layer_3",
                "layer_type": "ReLU",
                "depth_index": 3,
                "intermediate_features": {
                    "activation_std": 0.02,
                    "activation_mean": 0.01,
                    "activation_shape": [128, 256],
                    "cross_layer_std_ratio": 0.13
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.003,
                    "gradient_std": 0.0003,
                    "gradient_max_abs": 0.001
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.0,
                        "mean": 0.0,
                        "spectral_norm": 0.0,
                        "frobenius_norm": 0.0
                    },
                    "bias": {
                        "std": 0.0,
                        "mean_abs": 0.0
                    }
                }
            }
        ],
        "cross_layer_analysis": {
            "feature_std_gradient": -0.3,
            "gradient_norm_ratio": {
                "layer_1_to_prev": 0.2,
                "layer_2_to_prev": 0.1,
                "layer_3_to_prev": 0.1
            }
        }
    }


def exploding_gradient_pattern() -> Dict[str, Any]:
    """
    Network exhibiting exploding gradient pattern.

    Characterized by:
    - Strongly positive feature_std_gradient (> 0.1)
    - Increasing gradient norms with depth
    - High activation std in deeper layers
    """
    return {
        "metadata": {
            "run_id": "exploding_gradient_test",
            "timestamp": 1707589200.123,
            "global_step": 500,
            "batch_size": 64
        },
        "layer_statistics": [
            {
                "layer_id": "layer_0",
                "layer_type": "Linear",
                "depth_index": 0,
                "intermediate_features": {
                    "activation_std": 0.3,
                    "activation_mean": 0.0,
                    "activation_shape": [64, 128],
                    "cross_layer_std_ratio": None
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.05,
                    "gradient_std": 0.005,
                    "gradient_max_abs": 0.02
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.02,
                        "mean": 0.0,
                        "spectral_norm": 0.5,
                        "frobenius_norm": 0.8
                    },
                    "bias": {
                        "std": 0.005,
                        "mean_abs": 0.002
                    }
                }
            },
            {
                "layer_id": "layer_1",
                "layer_type": "Linear",
                "depth_index": 1,
                "intermediate_features": {
                    "activation_std": 0.8,
                    "activation_mean": 0.1,
                    "activation_shape": [64, 256],
                    "cross_layer_std_ratio": 2.67
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.3,
                    "gradient_std": 0.02,
                    "gradient_max_abs": 0.1
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.05,
                        "mean": 0.0,
                        "spectral_norm": 1.5,
                        "frobenius_norm": 2.5
                    },
                    "bias": {
                        "std": 0.01,
                        "mean_abs": 0.005
                    }
                }
            },
            {
                "layer_id": "layer_2",
                "layer_type": "Linear",
                "depth_index": 2,
                "intermediate_features": {
                    "activation_std": 2.5,
                    "activation_mean": 0.5,
                    "activation_shape": [64, 512],
                    "cross_layer_std_ratio": 3.125
                },
                "gradient_flow": {
                    "gradient_l2_norm": 1.8,
                    "gradient_std": 0.1,
                    "gradient_max_abs": 0.8
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.15,
                        "mean": 0.0,
                        "spectral_norm": 4.5,
                        "frobenius_norm": 8.0
                    },
                    "bias": {
                        "std": 0.03,
                        "mean_abs": 0.015
                    }
                }
            }
        ],
        "cross_layer_analysis": {
            "feature_std_gradient": 0.8,
            "gradient_norm_ratio": {
                "layer_1_to_prev": 6.0,
                "layer_2_to_prev": 6.0
            }
        }
    }


def healthy_gradient_pattern() -> Dict[str, Any]:
    """
    Network with healthy gradient flow.

    Characterized by:
    - Stable feature_std_gradient near 0
    - Balanced gradient norms across layers
    - Consistent activation std across layers
    """
    return {
        "metadata": {
            "run_id": "healthy_gradient_test",
            "timestamp": 1707589200.123,
            "global_step": 2000,
            "batch_size": 256
        },
        "layer_statistics": [
            {
                "layer_id": "layer_0",
                "layer_type": "Linear",
                "depth_index": 0,
                "intermediate_features": {
                    "activation_std": 0.7,
                    "activation_mean": 0.0,
                    "activation_shape": [256, 1024],
                    "cross_layer_std_ratio": None
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.5,
                    "gradient_std": 0.02,
                    "gradient_max_abs": 0.15
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.04,
                        "mean": 0.0,
                        "spectral_norm": 1.2,
                        "frobenius_norm": 2.0
                    },
                    "bias": {
                        "std": 0.01,
                        "mean_abs": 0.005
                    }
                }
            },
            {
                "layer_id": "layer_1",
                "layer_type": "ReLU",
                "depth_index": 1,
                "intermediate_features": {
                    "activation_std": 0.65,
                    "activation_mean": 0.32,
                    "activation_shape": [256, 1024],
                    "cross_layer_std_ratio": 0.93
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.45,
                    "gradient_std": 0.018,
                    "gradient_max_abs": 0.13
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.0,
                        "mean": 0.0,
                        "spectral_norm": 0.0,
                        "frobenius_norm": 0.0
                    },
                    "bias": {
                        "std": 0.0,
                        "mean_abs": 0.0
                    }
                }
            },
            {
                "layer_id": "layer_2",
                "layer_type": "Linear",
                "depth_index": 2,
                "intermediate_features": {
                    "activation_std": 0.6,
                    "activation_mean": 0.0,
                    "activation_shape": [256, 512],
                    "cross_layer_std_ratio": 0.92
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.4,
                    "gradient_std": 0.015,
                    "gradient_max_abs": 0.12
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.035,
                        "mean": 0.0,
                        "spectral_norm": 1.1,
                        "frobenius_norm": 1.8
                    },
                    "bias": {
                        "std": 0.008,
                        "mean_abs": 0.004
                    }
                }
            },
            {
                "layer_id": "layer_3",
                "layer_type": "ReLU",
                "depth_index": 3,
                "intermediate_features": {
                    "activation_std": 0.58,
                    "activation_mean": 0.29,
                    "activation_shape": [256, 512],
                    "cross_layer_std_ratio": 0.97
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.38,
                    "gradient_std": 0.014,
                    "gradient_max_abs": 0.11
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.0,
                        "mean": 0.0,
                        "spectral_norm": 0.0,
                        "frobenius_norm": 0.0
                    },
                    "bias": {
                        "std": 0.0,
                        "mean_abs": 0.0
                    }
                }
            },
            {
                "layer_id": "layer_4",
                "layer_type": "Linear",
                "depth_index": 4,
                "intermediate_features": {
                    "activation_std": 0.55,
                    "activation_mean": 0.0,
                    "activation_shape": [256, 256],
                    "cross_layer_std_ratio": 0.95
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.35,
                    "gradient_std": 0.013,
                    "gradient_max_abs": 0.1
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.03,
                        "mean": 0.0,
                        "spectral_norm": 1.0,
                        "frobenius_norm": 1.5
                    },
                    "bias": {
                        "std": 0.007,
                        "mean_abs": 0.003
                    }
                }
            }
        ],
        "cross_layer_analysis": {
            "feature_std_gradient": -0.03,
            "gradient_norm_ratio": {
                "layer_1_to_prev": 0.9,
                "layer_2_to_prev": 0.89,
                "layer_3_to_prev": 0.95,
                "layer_4_to_prev": 0.92
            }
        }
    }


# ==================== Multi-Run Scenarios ====================

def multi_run_concurrent_steps() -> List[Dict[str, Any]]:
    """
    Multiple runs with interleaved/concurrent step timestamps.

    Simulates three training runs happening concurrently with different
    step rates. Useful for testing storage behavior with multiple runs.
    """
    base_time = 1707589200.123

    runs = []

    # Run 1: Fast training, steps every 0.5 seconds
    for step in range(0, 100, 10):
        runs.append({
            "metadata": {
                "run_id": "fast_run",
                "timestamp": base_time + (step * 0.5),
                "global_step": step,
                "batch_size": 32
            },
            "layer_statistics": [
                {
                    "layer_id": "layer_0",
                    "layer_type": "Linear",
                    "depth_index": 0,
                    "intermediate_features": {
                        "activation_std": 0.7,
                        "activation_mean": 0.0,
                        "activation_shape": [32, 128],
                        "cross_layer_std_ratio": None
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": 0.5,
                        "gradient_std": 0.02,
                        "gradient_max_abs": 0.15
                    },
                    "parameter_statistics": {
                        "weight": {
                            "std": 0.04,
                            "mean": 0.0,
                            "spectral_norm": 1.2,
                            "frobenius_norm": 2.0
                        },
                        "bias": {
                            "std": 0.01,
                            "mean_abs": 0.005
                        }
                    }
                }
            ],
            "cross_layer_analysis": {
                "feature_std_gradient": -0.02,
                "gradient_norm_ratio": {}
            }
        })

    # Run 2: Medium speed, steps every 1 second
    for step in range(0, 50, 5):
        runs.append({
            "metadata": {
                "run_id": "medium_run",
                "timestamp": base_time + (step * 1.0),
                "global_step": step,
                "batch_size": 64
            },
            "layer_statistics": [
                {
                    "layer_id": "layer_0",
                    "layer_type": "Linear",
                    "depth_index": 0,
                    "intermediate_features": {
                        "activation_std": 0.7,
                        "activation_mean": 0.0,
                        "activation_shape": [64, 128],
                        "cross_layer_std_ratio": None
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": 0.5,
                        "gradient_std": 0.02,
                        "gradient_max_abs": 0.15
                    },
                    "parameter_statistics": {
                        "weight": {
                            "std": 0.04,
                            "mean": 0.0,
                            "spectral_norm": 1.2,
                            "frobenius_norm": 2.0
                        },
                        "bias": {
                            "std": 0.01,
                            "mean_abs": 0.005
                        }
                    }
                }
            ],
            "cross_layer_analysis": {
                "feature_std_gradient": -0.02,
                "gradient_norm_ratio": {}
            }
        })

    # Run 3: Slow training, steps every 2 seconds
    for step in range(0, 25, 5):
        runs.append({
            "metadata": {
                "run_id": "slow_run",
                "timestamp": base_time + (step * 2.0),
                "global_step": step,
                "batch_size": 128
            },
            "layer_statistics": [
                {
                    "layer_id": "layer_0",
                    "layer_type": "Linear",
                    "depth_index": 0,
                    "intermediate_features": {
                        "activation_std": 0.7,
                        "activation_mean": 0.0,
                        "activation_shape": [128, 128],
                        "cross_layer_std_ratio": None
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": 0.5,
                        "gradient_std": 0.02,
                        "gradient_max_abs": 0.15
                    },
                    "parameter_statistics": {
                        "weight": {
                            "std": 0.04,
                            "mean": 0.0,
                            "spectral_norm": 1.2,
                            "frobenius_norm": 2.0
                        },
                        "bias": {
                            "std": 0.01,
                            "mean_abs": 0.005
                        }
                    }
                }
            ],
            "cross_layer_analysis": {
                "feature_std_gradient": -0.02,
                "gradient_norm_ratio": {}
            }
        })

    return runs


def storage_eviction_scenario(max_runs: int = 5) -> List[Dict[str, Any]]:
    """
    Generate exactly max_runs + 1 runs to trigger eviction.

    The first run should be evicted when the (max_runs + 1)th run is added.
    Each run is timestamped to ensure deterministic eviction order.
    """
    base_time = 1707589200.123
    runs = []

    for i in range(max_runs + 1):
        run_id = f"eviction_test_run_{i}"
        timestamp = base_time + (i * 100)  # Ensure clear ordering

        runs.append({
            "metadata": {
                "run_id": run_id,
                "timestamp": timestamp,
                "global_step": 100,
                "batch_size": 64
            },
            "layer_statistics": [
                {
                    "layer_id": "layer_0",
                    "layer_type": "Linear",
                    "depth_index": 0,
                    "intermediate_features": {
                        "activation_std": 0.7,
                        "activation_mean": 0.0,
                        "activation_shape": [64, 128],
                        "cross_layer_std_ratio": None
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": 0.5,
                        "gradient_std": 0.02,
                        "gradient_max_abs": 0.15
                    },
                    "parameter_statistics": {
                        "weight": {
                            "std": 0.04,
                            "mean": 0.0,
                            "spectral_norm": 1.2,
                            "frobenius_norm": 2.0
                        },
                        "bias": {
                            "std": 0.01,
                            "mean_abs": 0.005
                        }
                    }
                }
            ],
            "cross_layer_analysis": {
                "feature_std_gradient": -0.02,
                "gradient_norm_ratio": {}
            }
        })

    return runs


def step_limit_reached_scenario(max_steps: int = 20) -> List[Dict[str, Any]]:
    """
    Generate exactly max_steps + 1 steps to trigger step limit enforcement.

    When these steps are added to a run, the oldest step should be evicted.
    """
    base_time = 1707589200.123
    steps = []

    for i in range(max_steps + 1):
        step_num = i * 10
        timestamp = base_time + (i * 1.0)

        steps.append({
            "metadata": {
                "run_id": "step_limit_test",
                "timestamp": timestamp,
                "global_step": step_num,
                "batch_size": 64
            },
            "layer_statistics": [
                {
                    "layer_id": "layer_0",
                    "layer_type": "Linear",
                    "depth_index": 0,
                    "intermediate_features": {
                        "activation_std": 0.7,
                        "activation_mean": 0.0,
                        "activation_shape": [64, 128],
                        "cross_layer_std_ratio": None
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": 0.5,
                        "gradient_std": 0.02,
                        "gradient_max_abs": 0.15
                    },
                    "parameter_statistics": {
                        "weight": {
                            "std": 0.04,
                            "mean": 0.0,
                            "spectral_norm": 1.2,
                            "frobenius_norm": 2.0
                        },
                        "bias": {
                            "std": 0.01,
                            "mean_abs": 0.005
                        }
                    }
                }
            ],
            "cross_layer_analysis": {
                "feature_std_gradient": -0.02,
                "gradient_norm_ratio": {}
            }
        })

    return steps


def duplicate_step_scenario() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate two payloads with the same step number but different data.

    Returns:
        tuple: (first_submission, second_submission)

    The second submission should replace the first due to step deduplication.
    """
    base_time = 1707589200.123

    first_submission = {
        "metadata": {
            "run_id": "duplicate_step_test",
            "timestamp": base_time,
            "global_step": 100,
            "batch_size": 64
        },
        "layer_statistics": [
            {
                "layer_id": "layer_0",
                "layer_type": "Linear",
                "depth_index": 0,
                "intermediate_features": {
                    "activation_std": 0.7,
                    "activation_mean": 0.0,
                    "activation_shape": [64, 128],
                    "cross_layer_std_ratio": None
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.5,
                    "gradient_std": 0.02,
                    "gradient_max_abs": 0.15
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.04,
                        "mean": 0.0,
                        "spectral_norm": 1.2,
                        "frobenius_norm": 2.0
                    },
                    "bias": {
                        "std": 0.01,
                        "mean_abs": 0.005
                    }
                }
            }
        ],
        "cross_layer_analysis": {
            "feature_std_gradient": -0.02,
            "gradient_norm_ratio": {}
        }
    }

    second_submission = {
        "metadata": {
            "run_id": "duplicate_step_test",
            "timestamp": base_time + 1.0,  # Later timestamp
            "global_step": 100,  # Same step number!
            "batch_size": 64
        },
        "layer_statistics": [
            {
                "layer_id": "layer_0",
                "layer_type": "Linear",
                "depth_index": 0,
                "intermediate_features": {
                    "activation_std": 0.65,  # Different value
                    "activation_mean": 0.01,
                    "activation_shape": [64, 128],
                    "cross_layer_std_ratio": None
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.45,  # Different value
                    "gradient_std": 0.018,
                    "gradient_max_abs": 0.14
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.038,  # Different value
                        "mean": 0.001,
                        "spectral_norm": 1.18,
                        "frobenius_norm": 1.95
                    },
                    "bias": {
                        "std": 0.009,
                        "mean_abs": 0.004
                    }
                }
            }
        ],
        "cross_layer_analysis": {
            "feature_std_gradient": -0.025,
            "gradient_norm_ratio": {}
        }
    }

    return first_submission, second_submission


# ==================== Edge Case Fixtures ====================

def zero_activation_std() -> Dict[str, Any]:
    """
    Layer with zero activation standard deviation (dead neurons).

    This is valid but indicates potential issues with the network.
    """
    base = valid_minimal_payload()
    base["layer_statistics"][0]["intermediate_features"]["activation_std"] = 0.0
    base["metadata"]["run_id"] = "zero_activation_std_test"
    return base


def zero_gradient_norm() -> Dict[str, Any]:
    """
    Layer with zero gradient norm (possible frozen layer).

    This is valid but may indicate a frozen or disconnected layer.
    """
    base = valid_minimal_payload()
    base["layer_statistics"][0]["gradient_flow"]["gradient_l2_norm"] = 0.0
    base["layer_statistics"][0]["gradient_flow"]["gradient_std"] = 0.0
    base["layer_statistics"][0]["gradient_flow"]["gradient_max_abs"] = 0.0
    base["metadata"]["run_id"] = "zero_gradient_norm_test"
    return base


def very_large_batch_size() -> Dict[str, Any]:
    """
    Payload with a very large batch size.

    Tests handling of large batch sizes (e.g., 4096).
    """
    base = valid_minimal_payload()
    base["metadata"]["batch_size"] = 4096
    base["metadata"]["run_id"] = "large_batch_test"
    base["layer_statistics"][0]["intermediate_features"]["activation_shape"] = [4096, 128]
    return base


def very_high_step_number() -> Dict[str, Any]:
    """
    Payload with a very high step number.

    Tests handling of networks that have trained for many steps.
    """
    base = valid_minimal_payload()
    base["metadata"]["global_step"] = 1_000_000
    base["metadata"]["run_id"] = "high_step_test"
    return base


# ==================== Convenience Functions ====================

def get_all_valid_fixtures() -> List[Dict[str, Any]]:
    """
    Return all valid metric fixtures for batch testing.

    Useful for parametrized tests that want to check multiple scenarios.
    """
    return [
        valid_minimal_payload(),
        valid_three_layer_network(),
        vanishing_gradient_pattern(),
        exploding_gradient_pattern(),
        healthy_gradient_pattern(),
        zero_activation_std(),
        zero_gradient_norm(),
        very_large_batch_size(),
        very_high_step_number()
    ]


def get_fixture_by_name(name: str) -> Dict[str, Any]:
    """
    Get a specific fixture by name.

    Args:
        name: Name of the fixture function

    Returns:
        The fixture data dictionary

    Raises:
        ValueError: If fixture name is not found
    """
    fixtures = {
        "valid_minimal": valid_minimal_payload,
        "three_layer": valid_three_layer_network,
        "vanishing": vanishing_gradient_pattern,
        "exploding": exploding_gradient_pattern,
        "healthy": healthy_gradient_pattern,
        "zero_activation": zero_activation_std,
        "zero_gradient": zero_gradient_norm,
        "large_batch": very_large_batch_size,
        "high_step": very_high_step_number
    }

    if name not in fixtures:
        raise ValueError(f"Unknown fixture: {name}. Available: {list(fixtures.keys())}")

    return fixtures[name]()
