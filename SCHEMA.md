# JSON Schema Documentation

This document describes the JSON schema for submitting metrics to the Neural Network Training Monitor. For API endpoint details and WebSocket protocol, see [API.md](API.md).

## API Endpoint

```
POST /api/v1/metrics/layerwise
Content-Type: application/json

Response: 202 Accepted
```

## Root Schema

The root object contains three top-level fields:

```json
{
  "metadata": Metadata,
  "layer_statistics": LayerStatistic[],
  "cross_layer_analysis": CrossLayerAnalysis
}
```

---

## Field Definitions

### 1. `metadata` (Metadata)

Top-level information about the training step.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `run_id` | string | Yes | Unique identifier for the training run (e.g., "experiment_42") |
| `timestamp` | number | Yes | Unix epoch timestamp (seconds since 1970-01-01 UTC) |
| `global_step` | integer | Yes | Current training step number |
| `batch_size` | integer | Yes | Batch size used for this step |
| `layer_groups` | object | No | Grouping specification for layers; maps group names to arrays of layer_id strings |

**Note:** Layer IDs use forward slash (`/`) as separator (e.g., `encoder/linear1`) for JSON compatibility. Dots (`.`) are automatically converted to slashes by the monitoring client.

```json
"metadata": {
  "run_id": "experiment_2024_0210_v2",
  "timestamp": 1707589200.123,
  "global_step": 1500,
  "batch_size": 64,
  "layer_groups": {
    "encoder": ["encoder/linear1", "encoder/relu1", "encoder/linear2"],
    "decoder": ["decoder/linear1", "decoder/relu1", "decoder/linear2"]
  }
}
```

**Note:** If `layer_groups` is provided, the frontend will display layers organized by these groups. Layers not included in any group will be displayed in an "Ungrouped" section.

---

### 2. `layer_statistics` (LayerStatistic[])

Array of layer-specific metrics. One element per monitored layer.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `layer_id` | string | Yes | Unique layer identifier (e.g., "encoder/linear1") |
| `layer_type` | string | Yes | Layer class name (e.g., "Linear", "Conv2d", "ReLU") |
| `depth_index` | integer | Yes | 0-based position of layer in the network |
| `intermediate_features` | IntermediateFeatures | Yes | Activation statistics |
| `gradient_flow` | GradientFlow | Yes | Gradient statistics |
| `parameter_statistics` | ParameterStatistics | Yes | Weight/bias statistics |

#### 2.1 `intermediate_features` (IntermediateFeatures)

Statistics about the output activations of this layer.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `activation_std` | float | Yes | Standard deviation of output activations |
| `activation_mean` | float | Yes | Mean of output activations |
| `activation_shape` | int[] | Yes | Tensor shape (e.g., [64, 256]) |
| `cross_layer_std_ratio` | float \| null | No | Ratio of this layer's std to previous layer's std; null for first layer |

```json
"intermediate_features": {
  "activation_std": 0.847,
  "activation_mean": -0.023,
  "activation_shape": [64, 256],
  "cross_layer_std_ratio": 0.94
}
```

#### 2.2 `gradient_flow` (GradientFlow)

Statistics about gradients flowing through this layer.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `gradient_l2_norm` | float | Yes | L2 norm of the gradient tensor (||∇L||₂) |
| `gradient_std` | float | Yes | Standard deviation of gradient values |
| `gradient_max_abs` | float | Yes | Maximum absolute gradient value (for clipping detection) |

```json
"gradient_flow": {
  "gradient_l2_norm": 0.152,
  "gradient_std": 0.0034,
  "gradient_max_abs": 0.089
}
```

#### 2.3 `parameter_statistics` (ParameterStatistics)

Statistics about the layer's learnable parameters (weights and biases).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `weight` | WeightStats | Yes | Weight statistics |
| `bias` | BiasStats | No | Bias statistics (optional - some layers like Conv2d with bias=False don't have bias) |

##### `weight` (WeightStats)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `std` | float | Yes | Standard deviation of weight values (σ(W)) |
| `mean` | float | Yes | Mean of weight values (μ(W)) |
| `spectral_norm` | float | Yes | Approximated spectral norm (||W||₂, via power iteration) |
| `frobenius_norm` | float | Yes | Frobenius norm (||W||_F) |

```json
"weight": {
  "std": 0.037,
  "mean": -0.001,
  "spectral_norm": 1.42,
  "frobenius_norm": 2.18
}
```

##### `bias` (BiasStats)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `std` | float | Yes | Standard deviation of bias values (σ(b)) |
| `mean_abs` | float | Yes | Mean of absolute bias values (<|b|>) |

```json
"bias": {
  "std": 0.012,
  "mean_abs": 0.008
}
```

---

### 3. `cross_layer_analysis` (CrossLayerAnalysis)

Aggregated metrics computed across multiple layers to detect depth-wise patterns.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `feature_std_gradient` | float | Yes | Linear slope of activation std vs depth (negative = vanishing) |
| `gradient_norm_ratio` | object | Yes | Map of layer pairs to gradient norm ratios (||∇_i|| / ||∇_{i-1}||) |

```json
"cross_layer_analysis": {
  "feature_std_gradient": -0.052,
  "gradient_norm_ratio": {
    "encoder.linear2_to_prev": 0.586,
    "encoder.relu1_to_prev": 0.923
  }
}
```

---

## Complete Example

```json
{
  "metadata": {
    "run_id": "experiment_2024_0210_v2",
    "timestamp": 1707589200.123,
    "global_step": 1500,
    "batch_size": 64,
    "layer_groups": {
      "encoder": ["encoder/linear1", "encoder/relu1", "encoder/linear2"]
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
        "cross_layer_std_ratio": null
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
        "activation_mean": 0.015,
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
```

---

## Validation Rules

1. **All numeric fields must be finite** - no NaN or Infinity values
2. **`depth_index` must be sequential** - starting from 0 for the first layer
3. **`layer_statistics` array must not be empty**
4. **`activation_shape` must have at least 2 dimensions** (batch_size + feature dimensions)
5. **`gradient_norm_ratio` keys should use `{layer_id}_to_prev` format**
6. **Layers without parameters** (e.g., ReLU, Dropout) should still include `parameter_statistics` with zero values
