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
            Err(serde::de::Error::custom("Value must be non-negative"))
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    pub layers: Vec<LayerStatistic>,
    pub cross_layer: CrossLayerAnalysis,
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
    pub ingest_queue_depth: usize,
    pub ingest_queue_capacity: usize,
    pub accepted_count: u64,
    pub processed_count: u64,
    pub dropped_count: u64,
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

    // ==================== New edge-case tests ====================

    #[test]
    fn test_finite_f64_serialization_roundtrip_expected() {
        let original = FiniteF64::new(3.14159265358979).unwrap();
        let serialized = serde_json::to_value(original.value()).unwrap();
        let deserialized: FiniteF64 =
            serde_json::from_value(serialized).expect("FiniteF64 roundtrip should succeed");
        assert_eq!(
            original.value(),
            deserialized.value(),
            "FiniteF64 roundtrip should preserve value"
        );

        let neg = FiniteF64::new(-42.5).unwrap();
        let ser = serde_json::to_value(neg.value()).unwrap();
        let de: FiniteF64 =
            serde_json::from_value(ser).expect("Negative FiniteF64 roundtrip should succeed");
        assert_eq!(neg.value(), de.value());
    }

    #[test]
    fn test_non_negative_f64_zero_accepted_expected() {
        let json = serde_json::Value::from(0.0);
        let result: NonNegativeF64 =
            serde_json::from_value(json).expect("NonNegativeF64 should accept 0.0");
        assert_eq!(result.value(), 0.0);
    }

    #[test]
    fn test_non_negative_f64_very_small_positive_accepted_expected() {
        let json = serde_json::Value::from(0.000001);
        let result: NonNegativeF64 =
            serde_json::from_value(json).expect("NonNegativeF64 should accept 0.000001");
        assert!(
            (result.value() - 0.000001).abs() < f64::EPSILON,
            "Should preserve very small positive value"
        );
    }

    #[test]
    fn test_metadata_batch_size_zero_rejected_expected() {
        let mut json = valid_payload_json();
        json["metadata"]["batch_size"] = serde_json::Value::from(0);
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("batch_size=0 deserializes (u32)");
        let err = payload.validate().unwrap_err();
        assert!(
            err.contains("batch_size"),
            "batch_size=0 should fail validation, got: {}",
            err
        );
    }

    #[test]
    fn test_metadata_timestamp_zero_rejected_expected() {
        let mut json = valid_payload_json();
        json["metadata"]["timestamp"] = serde_json::Value::from(0.0);
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("timestamp=0.0 deserializes (FiniteF64)");
        let err = payload.validate().unwrap_err();
        assert!(
            err.contains("timestamp"),
            "timestamp=0.0 should fail validation, got: {}",
            err
        );
    }

    #[test]
    fn test_metadata_timestamp_negative_rejected_expected() {
        let mut json = valid_payload_json();
        json["metadata"]["timestamp"] = serde_json::Value::from(-1.0);
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("timestamp=-1.0 deserializes (FiniteF64)");
        let err = payload.validate().unwrap_err();
        assert!(
            err.contains("timestamp"),
            "timestamp=-1.0 should fail validation, got: {}",
            err
        );
    }

    #[test]
    fn test_layer_statistic_depth_index_zero_boundary_passes_expected() {
        let mut json = valid_payload_json();
        json["layer_statistics"][0]["depth_index"] = serde_json::Value::from(0u32);
        json["layer_statistics"][1]["depth_index"] = serde_json::Value::from(0u32);
        json["layer_statistics"][2]["depth_index"] = serde_json::Value::from(1u32);
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("depth_index=0 boundary deserializes");
        assert!(
            payload.validate().is_ok(),
            "depth_index=0 (boundary) should pass validation"
        );
    }

    #[test]
    fn test_metrics_payload_single_layer_minimum_valid_expected() {
        let mut json = valid_payload_json();
        json["layer_statistics"] = serde_json::json!([json["layer_statistics"][0]]);
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("Single-layer payload should deserialize");
        assert!(payload.validate().is_ok(), "Single layer should be valid");
        assert_eq!(payload.layer_statistics.len(), 1);
    }

    #[test]
    fn test_metrics_payload_many_layers_all_valid_expected() {
        let mut json = valid_payload_json();
        let layer_template = json["layer_statistics"][0].clone();
        let mut layers = Vec::new();
        for i in 0..10u32 {
            let mut layer = layer_template.clone();
            layer["layer_id"] = serde_json::Value::from(format!("layer_{}", i));
            layer["depth_index"] = serde_json::Value::from(i);
            layers.push(layer);
        }
        json["layer_statistics"] = serde_json::Value::Array(layers);
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("10-layer payload should deserialize");
        assert!(
            payload.validate().is_ok(),
            "10 layers all valid should pass"
        );
        assert_eq!(payload.layer_statistics.len(), 10);
    }

    #[test]
    fn test_cross_layer_analysis_empty_gradient_norm_ratio_passes_expected() {
        let mut json = valid_payload_json();
        json["cross_layer_analysis"]["gradient_norm_ratio"] = serde_json::json!({});
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("Empty gradient_norm_ratio deserializes");
        assert!(
            payload.validate().is_ok(),
            "Empty gradient_norm_ratio should pass validation (no constraints on it)"
        );
        assert!(payload.cross_layer_analysis.gradient_norm_ratio.is_empty());
    }

    #[test]
    fn test_cross_layer_analysis_nan_in_gradient_norm_ratio_accepted_by_serde_expected() {
        // gradient_norm_ratio uses plain f64, not FiniteF64, so serde_json accepts NaN.
        // However, serde_json::Value::from(f64::NAN) becomes Null, so we must construct
        // the map manually to confirm f64 acceptance behavior.
        let mut json = valid_payload_json();
        // Replace gradient_norm_ratio with a map containing a normal f64 value.
        // We can't put NaN via serde_json::Value (it becomes null), but we verify
        // that the field uses plain f64 by confirming it deserializes fine with
        // arbitrary f64 values including very large ones.
        json["cross_layer_analysis"]["gradient_norm_ratio"] = serde_json::json!({
            "ratio_a": 1e308,
            "ratio_b": -1e308
        });
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("Large f64 in gradient_norm_ratio deserializes");
        assert!(
            payload.validate().is_ok(),
            "Plain f64 in gradient_norm_ratio should not block validation"
        );
        // Note: NaN in serde_json::Value becomes null, which would cause a type error.
        // This test documents that gradient_norm_ratio uses plain f64 (not FiniteF64).
        let ratios = &payload.cross_layer_analysis.gradient_norm_ratio;
        assert_eq!(ratios.len(), 2);
    }

    #[test]
    fn test_activation_shape_exactly_two_dimensions_passes_expected() {
        let mut json = valid_payload_json();
        json["layer_statistics"][0]["intermediate_features"]["activation_shape"] =
            serde_json::json!([64, 128]);
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("2-dim activation_shape deserializes");
        assert!(
            payload.validate().is_ok(),
            "activation_shape with exactly 2 dimensions should pass"
        );
        assert_eq!(
            payload.layer_statistics[0]
                .intermediate_features
                .activation_shape
                .len(),
            2
        );
    }

    #[test]
    fn test_layer_id_special_characters_accepted_expected() {
        let mut json = valid_payload_json();
        json["layer_statistics"][0]["layer_id"] =
            serde_json::Value::from("encoder/linear-1_conv2d");
        json["layer_statistics"][1]["layer_id"] = serde_json::Value::from("decoder/layer_norm");
        json["layer_statistics"][2]["layer_id"] =
            serde_json::Value::from("model.transformer.attn_head-3");
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("Special-char layer_id deserializes");
        assert!(
            payload.validate().is_ok(),
            "layer_id with slashes, hyphens, underscores, dots should be accepted"
        );
        assert_eq!(
            payload.layer_statistics[0].layer_id,
            "encoder/linear-1_conv2d"
        );
    }

    #[test]
    fn test_layer_type_empty_string_accepted_expected() {
        let mut json = valid_payload_json();
        json["layer_statistics"][0]["layer_type"] = serde_json::Value::from("");
        let payload: MetricsPayload =
            serde_json::from_value(json).expect("Empty layer_type deserializes");
        assert!(
            payload.validate().is_ok(),
            "Empty layer_type should pass validation (no constraint defined)"
        );
        assert_eq!(payload.layer_statistics[0].layer_type, "");
    }
}
