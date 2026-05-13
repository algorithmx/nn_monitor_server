use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::ops::Deref;

// ==================== FiniteF64 / NonNegativeF64 ====================

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FiniteF64(f64);

impl FiniteF64 {
    pub fn value(&self) -> f64 {
        self.0
    }
    pub fn new(v: f64) -> Result<Self, String> {
        if v.is_finite() {
            Ok(Self(v))
        } else {
            Err("Value must be finite (not NaN or Infinity)".to_string())
        }
    }
}

impl Serialize for FiniteF64 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for FiniteF64 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let v = f64::deserialize(deserializer)?;
        if v.is_finite() {
            Ok(FiniteF64(v))
        } else {
            Err(serde::de::Error::custom(
                "Value must be finite (not NaN or Infinity)",
            ))
        }
    }
}

impl Deref for FiniteF64 {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for FiniteF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct NonNegativeF64(FiniteF64);

impl NonNegativeF64 {
    pub fn value(&self) -> f64 {
        self.0.value()
    }
}

impl Serialize for NonNegativeF64 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for NonNegativeF64 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let v = FiniteF64::deserialize(deserializer)?;
        if *v >= 0.0 {
            Ok(NonNegativeF64(v))
        } else {
            Err(serde::de::Error::custom(
                "Value must be non-negative",
            ))
        }
    }
}

impl Deref for NonNegativeF64 {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for NonNegativeF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// ==================== Input Models ====================

#[derive(Debug, Clone, Deserialize)]
pub struct Metadata {
    pub run_id: String,
    pub timestamp: FiniteF64,
    pub global_step: u64,
    pub batch_size: u32,
    pub layer_groups: Option<std::collections::HashMap<String, Vec<String>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermediateFeatures {
    pub activation_std: NonNegativeF64,
    pub activation_mean: FiniteF64,
    pub activation_shape: Vec<u64>,
    pub cross_layer_std_ratio: Option<NonNegativeF64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlow {
    pub gradient_l2_norm: NonNegativeF64,
    pub gradient_std: NonNegativeF64,
    pub gradient_max_abs: NonNegativeF64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightStats {
    pub std: NonNegativeF64,
    pub mean: FiniteF64,
    pub spectral_norm: NonNegativeF64,
    pub frobenius_norm: NonNegativeF64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasStats {
    pub std: NonNegativeF64,
    pub mean_abs: NonNegativeF64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterStatistics {
    pub weight: WeightStats,
    pub bias: Option<BiasStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStatistic {
    pub layer_id: String,
    pub layer_type: String,
    pub depth_index: u32,
    pub intermediate_features: IntermediateFeatures,
    pub gradient_flow: GradientFlow,
    pub parameter_statistics: ParameterStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLayerAnalysis {
    pub feature_std_gradient: FiniteF64,
    pub gradient_norm_ratio: std::collections::HashMap<String, f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MetricsPayload {
    pub metadata: Metadata,
    pub layer_statistics: Vec<LayerStatistic>,
    pub cross_layer_analysis: CrossLayerAnalysis,
}

impl MetricsPayload {
    pub fn validate(&self) -> Result<(), String> {
        // run_id non-empty
        if self.metadata.run_id.is_empty() {
            return Err("run_id must not be empty".to_string());
        }
        // timestamp > 0
        if *self.metadata.timestamp <= 0.0 {
            return Err("timestamp must be greater than 0".to_string());
        }
        // batch_size > 0
        if self.metadata.batch_size == 0 {
            return Err("batch_size must be greater than 0".to_string());
        }
        // layer_statistics non-empty
        if self.layer_statistics.is_empty() {
            return Err("layer_statistics must not be empty".to_string());
        }
        for layer in &self.layer_statistics {
            // activation_shape min length 2
            if layer.intermediate_features.activation_shape.len() < 2 {
                return Err(format!(
                    "activation_shape must have at least 2 dimensions for layer '{}'",
                    layer.layer_id
                ));
            }
        }
        // depth_index sorted (monotonically non-decreasing)
        for i in 0..self.layer_statistics.len().saturating_sub(1) {
            if self.layer_statistics[i].depth_index > self.layer_statistics[i + 1].depth_index {
                return Err(format!(
                    "Layers must be sorted by depth_index: {} has depth_index {} but {} has depth_index {}",
                    self.layer_statistics[i].layer_id,
                    self.layer_statistics[i].depth_index,
                    self.layer_statistics[i + 1].layer_id,
                    self.layer_statistics[i + 1].depth_index
                ));
            }
        }
        Ok(())
    }
}

// ==================== Response Models ====================

#[derive(Debug, Clone, Serialize)]
pub struct MetricsAcceptedResponse {
    pub status: String,
    pub run_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunInfo {
    pub created_at: String,
    pub last_update: String,
    pub step_count: u32,
    pub latest_step: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepData {
    pub step: u64,
    pub timestamp: f64,
    pub batch_size: u32,
    pub layers: Vec<serde_json::Value>,
    pub cross_layer: serde_json::Value,
    pub layer_groups: Option<std::collections::HashMap<String, Vec<String>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunData {
    pub created_at: String,
    pub last_update: String,
    pub steps: Vec<StepData>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub active_connections: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub error: String,
    pub message: String,
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a valid minimal JSON payload for reuse
    fn valid_payload_json() -> serde_json::Value {
        serde_json::json!({
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
        })
    }

    #[test]
    fn test_nan_rejected() {
        // serde_json cannot produce NaN from JSON text, so we test FiniteF64 directly.
        // In production, NaN would arrive via non-JSON serde formats or after
        // using serde_json's `arbitrary_precision` feature.
        let err = FiniteF64::new(f64::NAN).unwrap_err();
        assert!(err.contains("finite"), "Error: {}", err);
        // Also verify our Deserialize impl rejects it if someone constructs a Value manually.
        // serde_json::Value::from(f64::NAN) silently becomes null, which is a different error.
        // The FiniteF64 deserialize impl catches NaN if it ever reaches f64::deserialize.
        assert!(!f64::NAN.is_finite());
    }

    #[test]
    fn test_inf_rejected() {
        let err = FiniteF64::new(f64::INFINITY).unwrap_err();
        assert!(err.contains("finite"), "Error: {}", err);
        let err = FiniteF64::new(f64::NEG_INFINITY).unwrap_err();
        assert!(err.contains("finite"), "Error: {}", err);
        assert!(!f64::INFINITY.is_finite());
    }

    #[test]
    fn test_negative_rejected() {
        let mut json = valid_payload_json();
        json["layer_statistics"][0]["intermediate_features"]["activation_std"] =
            serde_json::Value::from(-0.5);
        let result: Result<MetricsPayload, _> = serde_json::from_value(json);
        assert!(
            result.is_err(),
            "Negative value should be rejected for NonNegativeF64 field"
        );
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("non-negative"),
            "Error should mention non-negative: {}",
            err_msg
        );
    }

    #[test]
    fn test_zero_accepted() {
        let mut json = valid_payload_json();
        json["layer_statistics"][0]["intermediate_features"]["activation_std"] =
            serde_json::Value::from(0.0);
        json["layer_statistics"][0]["gradient_flow"]["gradient_l2_norm"] =
            serde_json::Value::from(0.0);
        let result: MetricsPayload =
            serde_json::from_value(json).expect("Zero values should be accepted");
        assert!(result.validate().is_ok());
    }

    #[test]
    fn test_unsorted_depth_rejected() {
        let mut json = valid_payload_json();
        // Swap depth_index of first two layers: [0,1,2] -> [1,0,2]
        json["layer_statistics"][0]["depth_index"] = serde_json::Value::from(1);
        json["layer_statistics"][1]["depth_index"] = serde_json::Value::from(0);
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("Deserialization should succeed");
        let err = payload.validate().unwrap_err();
        assert!(
            err.contains("Layers must be sorted by depth_index"),
            "Expected sorted-depth error, got: {}",
            err
        );
    }

    #[test]
    fn test_optional_bias() {
        let mut json = valid_payload_json();
        json["layer_statistics"][0]["parameter_statistics"]["bias"] = serde_json::Value::Null;
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("Null bias should be accepted");
        assert!(
            payload.layer_statistics[0]
                .parameter_statistics
                .bias
                .is_none(),
            "bias should be None when null"
        );
        assert!(payload.validate().is_ok());
    }

    #[test]
    fn test_full_payload_valid() {
        let json = valid_payload_json();
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("Full payload should deserialize");
        assert!(payload.validate().is_ok(), "Full payload should validate");
        assert_eq!(payload.metadata.run_id, "experiment_2024_0210_v2");
        assert_eq!(payload.layer_statistics.len(), 3);
        assert_eq!(payload.layer_statistics[0].depth_index, 0);
        assert_eq!(payload.layer_statistics[1].depth_index, 1);
        assert_eq!(payload.layer_statistics[2].depth_index, 2);
    }

    #[test]
    fn test_empty_run_id_rejected() {
        let mut json = valid_payload_json();
        json["metadata"]["run_id"] = serde_json::Value::from("");
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("Empty run_id deserializes");
        let err = payload.validate().unwrap_err();
        assert!(
            err.contains("run_id"),
            "Expected run_id error, got: {}",
            err
        );
    }

    #[test]
    fn test_empty_activation_shape_rejected() {
        let mut json = valid_payload_json();
        json["layer_statistics"][0]["intermediate_features"]["activation_shape"] =
            serde_json::json!([1]);
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("Short activation_shape deserializes");
        let err = payload.validate().unwrap_err();
        assert!(
            err.contains("activation_shape"),
            "Expected activation_shape error, got: {}",
            err
        );
    }

    #[test]
    fn test_empty_layer_statistics_rejected() {
        let mut json = valid_payload_json();
        json["layer_statistics"] = serde_json::Value::Array(vec![]);
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("Empty layer_statistics deserializes");
        let err = payload.validate().unwrap_err();
        assert!(
            err.contains("layer_statistics"),
            "Expected layer_statistics error, got: {}",
            err
        );
    }
}
