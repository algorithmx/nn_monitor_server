"""
Error and invalid payload fixtures for testing validation rules.

This module provides pre-defined invalid payloads that should trigger
specific validation errors according to the JSON schema and API specs.
"""

import math
from typing import Dict, Any, List


# ==================== Required Field Missing ====================

def missing_run_id() -> Dict[str, Any]:
    """Payload missing the run_id field."""
    return {
        "metadata": {
            # "run_id" is missing
            "timestamp": 1707589200.123,
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


def missing_layer_statistics() -> Dict[str, Any]:
    """Payload missing the layer_statistics field."""
    return {
        "metadata": {
            "run_id": "missing_layers_test",
            "timestamp": 1707589200.123,
            "global_step": 100,
            "batch_size": 64
        },
        # "layer_statistics" is missing
        "cross_layer_analysis": {
            "feature_std_gradient": -0.02,
            "gradient_norm_ratio": {}
        }
    }


def empty_layer_statistics() -> Dict[str, Any]:
    """Payload with empty layer_statistics array."""
    return {
        "metadata": {
            "run_id": "empty_layers_test",
            "timestamp": 1707589200.123,
            "global_step": 100,
            "batch_size": 64
        },
        "layer_statistics": [],  # Empty array - should fail validation
        "cross_layer_analysis": {
            "feature_std_gradient": -0.02,
            "gradient_norm_ratio": {}
        }
    }


def missing_cross_layer_analysis() -> Dict[str, Any]:
    """Payload missing the cross_layer_analysis field."""
    return {
        "metadata": {
            "run_id": "missing_cross_layer_test",
            "timestamp": 1707589200.123,
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
        ]
        # "cross_layer_analysis" is missing
    }


# ==================== Invalid Field Types ====================

def invalid_run_id_type() -> Dict[str, Any]:
    """Payload with non-string run_id."""
    return {
        "metadata": {
            "run_id": 123456,  # Should be string
            "timestamp": 1707589200.123,
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


def invalid_step_type() -> Dict[str, Any]:
    """Payload with non-integer global_step."""
    return {
        "metadata": {
            "run_id": "invalid_step_type_test",
            "timestamp": 1707589200.123,
            "global_step": "100",  # Should be integer
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


def invalid_batch_size_type() -> Dict[str, Any]:
    """Payload with non-integer batch_size."""
    return {
        "metadata": {
            "run_id": "invalid_batch_size_type_test",
            "timestamp": 1707589200.123,
            "global_step": 100,
            "batch_size": "64"  # Should be integer
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


# ==================== Invalid Field Values ====================

def empty_string_run_id() -> Dict[str, Any]:
    """Payload with empty string run_id."""
    return {
        "metadata": {
            "run_id": "",  # Empty string - should fail validation
            "timestamp": 1707589200.123,
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


def negative_step() -> Dict[str, Any]:
    """Payload with negative global_step."""
    return {
        "metadata": {
            "run_id": "negative_step_test",
            "timestamp": 1707589200.123,
            "global_step": -100,  # Negative step - should fail validation
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


def zero_batch_size() -> Dict[str, Any]:
    """Payload with zero batch_size."""
    return {
        "metadata": {
            "run_id": "zero_batch_size_test",
            "timestamp": 1707589200.123,
            "global_step": 100,
            "batch_size": 0  # Zero batch size - should fail validation
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


def negative_batch_size() -> Dict[str, Any]:
    """Payload with negative batch_size."""
    return {
        "metadata": {
            "run_id": "negative_batch_size_test",
            "timestamp": 1707589200.123,
            "global_step": 100,
            "batch_size": -64  # Negative batch size - should fail validation
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


def zero_timestamp() -> Dict[str, Any]:
    """Payload with zero timestamp."""
    return {
        "metadata": {
            "run_id": "zero_timestamp_test",
            "timestamp": 0.0,  # Zero timestamp - should fail validation
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


def negative_timestamp() -> Dict[str, Any]:
    """Payload with negative timestamp."""
    return {
        "metadata": {
            "run_id": "negative_timestamp_test",
            "timestamp": -100.0,  # Negative timestamp - should fail validation
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


# ==================== NaN and Infinity Values ====================

def nan_activation_std() -> Dict[str, Any]:
    """Payload with NaN in activation_std."""
    return {
        "metadata": {
            "run_id": "nan_activation_std_test",
            "timestamp": 1707589200.123,
            "global_step": 100,
            "batch_size": 64
        },
        "layer_statistics": [
            {
                "layer_id": "layer_0",
                "layer_type": "Linear",
                "depth_index": 0,
                "intermediate_features": {
                    "activation_std": float('nan'),  # NaN - should fail validation
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


def infinity_gradient_norm() -> Dict[str, Any]:
    """Payload with infinity in gradient_l2_norm."""
    return {
        "metadata": {
            "run_id": "infinity_gradient_test",
            "timestamp": 1707589200.123,
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
                    "gradient_l2_norm": float('inf'),  # Infinity - should fail validation
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


def negative_activation_std() -> Dict[str, Any]:
    """Payload with negative activation_std (standard deviation can't be negative)."""
    return {
        "metadata": {
            "run_id": "negative_activation_std_test",
            "timestamp": 1707589200.123,
            "global_step": 100,
            "batch_size": 64
        },
        "layer_statistics": [
            {
                "layer_id": "layer_0",
                "layer_type": "Linear",
                "depth_index": 0,
                "intermediate_features": {
                    "activation_std": -0.7,  # Negative std - should fail validation
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


def negative_gradient_std() -> Dict[str, Any]:
    """Payload with negative gradient_std (standard deviation can't be negative)."""
    return {
        "metadata": {
            "run_id": "negative_gradient_std_test",
            "timestamp": 1707589200.123,
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
                    "gradient_std": -0.02,  # Negative std - should fail validation
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


# ==================== Depth Ordering Violations ====================

def unsorted_depth_index() -> Dict[str, Any]:
    """
    Payload with layers not sorted by depth_index.

    Layer 0 has depth_index=0, layer 1 has depth_index=2,
    layer 2 has depth_index=1 - this is out of order.
    """
    return {
        "metadata": {
            "run_id": "unsorted_depth_test",
            "timestamp": 1707589200.123,
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
            },
            {
                "layer_id": "layer_1",
                "layer_type": "Linear",
                "depth_index": 2,  # This comes after depth_index=1, so it's out of order
                "intermediate_features": {
                    "activation_std": 0.6,
                    "activation_mean": 0.0,
                    "activation_shape": [64, 128],
                    "cross_layer_std_ratio": None
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
                "layer_id": "layer_2",
                "layer_type": "Linear",
                "depth_index": 1,  # Should come before depth_index=2
                "intermediate_features": {
                    "activation_std": 0.65,
                    "activation_mean": 0.0,
                    "activation_shape": [64, 128],
                    "cross_layer_std_ratio": None
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.45,
                    "gradient_std": 0.018,
                    "gradient_max_abs": 0.14
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.038,
                        "mean": 0.0,
                        "spectral_norm": 1.15,
                        "frobenius_norm": 1.9
                    },
                    "bias": {
                        "std": 0.009,
                        "mean_abs": 0.0045
                    }
                }
            }
        ],
        "cross_layer_analysis": {
            "feature_std_gradient": -0.02,
            "gradient_norm_ratio": {}
        }
    }


def negative_depth_index() -> Dict[str, Any]:
    """Payload with negative depth_index."""
    return {
        "metadata": {
            "run_id": "negative_depth_test",
            "timestamp": 1707589200.123,
            "global_step": 100,
            "batch_size": 64
        },
        "layer_statistics": [
            {
                "layer_id": "layer_0",
                "layer_type": "Linear",
                "depth_index": -1,  # Negative depth index - should fail validation
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


# ==================== Activation Shape Issues ====================

def single_dimension_activation_shape() -> Dict[str, Any]:
    """Payload with activation_shape having only 1 dimension (needs at least 2)."""
    return {
        "metadata": {
            "run_id": "single_dim_shape_test",
            "timestamp": 1707589200.123,
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
                    "activation_shape": [64],  # Only 1 dimension - should fail validation
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


def empty_activation_shape() -> Dict[str, Any]:
    """Payload with empty activation_shape array."""
    return {
        "metadata": {
            "run_id": "empty_shape_test",
            "timestamp": 1707589200.123,
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
                    "activation_shape": [],  # Empty array - should fail validation
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


# ==================== Convenience Functions ====================

def get_all_error_fixtures() -> List[tuple[str, Dict[str, Any]]]:
    """
    Return all error fixtures with their expected error descriptions.

    Returns a list of tuples: (fixture_name, fixture_data)
    Useful for parametrized tests.
    """
    return [
        ("missing_run_id", missing_run_id()),
        ("missing_layer_statistics", missing_layer_statistics()),
        ("empty_layer_statistics", empty_layer_statistics()),
        ("missing_cross_layer_analysis", missing_cross_layer_analysis()),
        ("invalid_run_id_type", invalid_run_id_type()),
        ("invalid_step_type", invalid_step_type()),
        ("invalid_batch_size_type", invalid_batch_size_type()),
        ("empty_string_run_id", empty_string_run_id()),
        ("negative_step", negative_step()),
        ("zero_batch_size", zero_batch_size()),
        ("negative_batch_size", negative_batch_size()),
        ("zero_timestamp", zero_timestamp()),
        ("negative_timestamp", negative_timestamp()),
        ("nan_activation_std", nan_activation_std()),
        ("infinity_gradient_norm", infinity_gradient_norm()),
        ("negative_activation_std", negative_activation_std()),
        ("negative_gradient_std", negative_gradient_std()),
        ("unsorted_depth_index", unsorted_depth_index()),
        ("negative_depth_index", negative_depth_index()),
        ("single_dimension_activation_shape", single_dimension_activation_shape()),
        ("empty_activation_shape", empty_activation_shape())
    ]


def get_fixture_by_name(name: str) -> Dict[str, Any]:
    """
    Get a specific error fixture by name.

    Args:
        name: Name of the fixture function

    Returns:
        The fixture data dictionary

    Raises:
        ValueError: If fixture name is not found
    """
    fixtures = {
        "missing_run_id": missing_run_id,
        "missing_layer_statistics": missing_layer_statistics,
        "empty_layer_statistics": empty_layer_statistics,
        "missing_cross_layer_analysis": missing_cross_layer_analysis,
        "invalid_run_id_type": invalid_run_id_type,
        "invalid_step_type": invalid_step_type,
        "invalid_batch_size_type": invalid_batch_size_type,
        "empty_string_run_id": empty_string_run_id,
        "negative_step": negative_step,
        "zero_batch_size": zero_batch_size,
        "negative_batch_size": negative_batch_size,
        "zero_timestamp": zero_timestamp,
        "negative_timestamp": negative_timestamp,
        "nan_activation_std": nan_activation_std,
        "infinity_gradient_norm": infinity_gradient_norm,
        "negative_activation_std": negative_activation_std,
        "negative_gradient_std": negative_gradient_std,
        "unsorted_depth_index": unsorted_depth_index,
        "negative_depth_index": negative_depth_index,
        "single_dimension_activation_shape": single_dimension_activation_shape,
        "empty_activation_shape": empty_activation_shape
    }

    if name not in fixtures:
        raise ValueError(f"Unknown fixture: {name}. Available: {list(fixtures.keys())}")

    return fixtures[name]()
