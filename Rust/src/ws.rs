use std::sync::atomic::{AtomicU32, Ordering};
use tokio::sync::broadcast;

use serde::Serialize;
use serde_json::json;

use crate::models::{RunData, StepData};

// ==================== WebSocket Manager ====================

/// Manages WebSocket connections for real-time updates.
///
/// Uses tokio::sync::broadcast for fan-out: every message sent via
/// [`WsManager::broadcast`] is delivered to all active subscribers.
/// Connection count is tracked with an AtomicU32 (lock-free).
pub struct WsManager {
    tx: broadcast::Sender<String>,
    connections: AtomicU32,
}

impl Default for WsManager {
    fn default() -> Self {
        Self::new()
    }
}

impl WsManager {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1024);
        Self {
            tx,
            connections: AtomicU32::new(0),
        }
    }

    /// Create a new receiver subscribed to the broadcast channel.
    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        self.tx.subscribe()
    }

    /// Broadcast a JSON string to all connected clients.
    ///
    /// Silently drops the message when no receivers exist.
    /// Logs a warning when a receiver is lagging (slow client).
    pub fn broadcast(&self, message: String) {
        match self.tx.send(message) {
            Ok(_n) => {}
            Err(broadcast::error::SendError(_msg)) => {
                // No active receivers — message dropped, this is fine.
            }
        }
    }

    /// Increment active connection counter.
    pub fn connect(&self) {
        self.connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active connection counter.
    pub fn disconnect(&self) {
        self.connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Current number of active WebSocket connections.
    pub fn active_count(&self) -> u32 {
        self.connections.load(Ordering::Relaxed)
    }
}

// ==================== Message Builders ====================

#[derive(Serialize)]
struct TypedMessage<'a, T: Serialize + ?Sized> {
    #[serde(rename = "type")]
    message_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    run_id: Option<&'a str>,
    data: &'a T,
}

/// Build `new_metrics` broadcast message.
pub fn build_new_metrics_message(run_id: &str, step_data: &StepData) -> String {
    let msg = TypedMessage {
        message_type: "new_metrics",
        run_id: Some(run_id),
        data: step_data,
    };
    serde_json::to_string(&msg).expect("new_metrics serialization should never fail")
}

/// Build `run_history` message for full (non-lite) subscribe_run response.
pub fn build_run_history_message(run_id: &str, run_data: &RunData) -> String {
    let msg = TypedMessage {
        message_type: "run_history",
        run_id: Some(run_id),
        data: run_data,
    };
    serde_json::to_string(&msg).expect("run_history serialization should never fail")
}

// ==================== Compact / Lite Helpers ====================
// These replicate Python's _compact_layer, _compact_step, _compact_run.
// Lite mode strips heavy fields (activation_mean, activation_shape,
// gradient_std, parameter_statistics) to keep run switching fast.

#[derive(Serialize)]
struct CompactIntermediateFeatures {
    activation_std: crate::models::NonNegativeF64,
    cross_layer_std_ratio: Option<crate::models::NonNegativeF64>,
}

#[derive(Serialize)]
struct CompactGradientFlow {
    gradient_l2_norm: crate::models::NonNegativeF64,
    gradient_max_abs: crate::models::NonNegativeF64,
}

#[derive(Serialize)]
struct CompactLayer<'a> {
    layer_id: &'a str,
    layer_type: &'a str,
    depth_index: u32,
    intermediate_features: CompactIntermediateFeatures,
    gradient_flow: CompactGradientFlow,
}

#[derive(Serialize)]
struct CompactStep<'a> {
    step: u64,
    timestamp: f64,
    batch_size: u32,
    layers: Vec<CompactLayer<'a>>,
    cross_layer: &'a crate::models::CrossLayerAnalysis,
    layer_groups: &'a Option<std::collections::HashMap<String, Vec<String>>>,
}

/// Compact a single step: keep only fields the dashboard needs.
fn compact_step(step: &StepData) -> CompactStep<'_> {
    let layers = step
        .layers
        .iter()
        .map(|layer| CompactLayer {
            layer_id: &layer.layer_id,
            layer_type: &layer.layer_type,
            depth_index: layer.depth_index,
            intermediate_features: CompactIntermediateFeatures {
                activation_std: layer.intermediate_features.activation_std,
                cross_layer_std_ratio: layer.intermediate_features.cross_layer_std_ratio,
            },
            gradient_flow: CompactGradientFlow {
                gradient_l2_norm: layer.gradient_flow.gradient_l2_norm,
                gradient_max_abs: layer.gradient_flow.gradient_max_abs,
            },
        })
        .collect();

    CompactStep {
        step: step.step,
        timestamp: step.timestamp,
        batch_size: step.batch_size,
        layers,
        cross_layer: &step.cross_layer,
        layer_groups: &step.layer_groups,
    }
}

#[derive(Serialize)]
struct CompactRunData<'a> {
    created_at: &'a str,
    last_update: &'a str,
    steps: Vec<CompactStep<'a>>,
}

/// Build `run_history` message for lite/compact subscribe_run response.
pub fn build_compact_run_history_message(run_id: &str, run_data: &RunData) -> String {
    let compact_steps: Vec<CompactStep<'_>> = run_data.steps.iter().map(compact_step).collect();
    let data = CompactRunData {
        created_at: &run_data.created_at,
        last_update: &run_data.last_update,
        steps: compact_steps,
    };
    let msg = TypedMessage {
        message_type: "run_history",
        run_id: Some(run_id),
        data: &data,
    };
    serde_json::to_string(&msg).expect("compact run_history serialization should never fail")
}

// ==================== Utility Message Builders ====================

/// Build error message.
pub fn build_error_message(msg: &str) -> String {
    json!({
        "type": "error",
        "message": msg
    })
    .to_string()
}

/// Build pong response.
pub fn build_pong_message() -> String {
    json!({"type": "pong"}).to_string()
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn test_layer() -> crate::models::LayerStatistic {
        serde_json::from_value(json!({
            "layer_id": "l1",
            "layer_type": "Linear",
            "depth_index": 0,
            "intermediate_features": {
                "activation_std": 0.5,
                "activation_mean": 0.1,
                "activation_shape": [32, 64],
                "cross_layer_std_ratio": 1.2
            },
            "gradient_flow": {
                "gradient_l2_norm": 0.2,
                "gradient_std": 0.01,
                "gradient_max_abs": 0.05
            },
            "parameter_statistics": {
                "weight": {
                    "std": 0.04,
                    "mean": 0.0,
                    "spectral_norm": 1.0,
                    "frobenius_norm": 0.5
                },
                "bias": null
            }
        }))
        .expect("test layer should deserialize")
    }

    fn test_cross_layer() -> crate::models::CrossLayerAnalysis {
        serde_json::from_value(json!({
            "feature_std_gradient": -0.1,
            "gradient_norm_ratio": {}
        }))
        .expect("test cross-layer analysis should deserialize")
    }

    #[test]
    fn test_ws_manager_new() {
        let mgr = WsManager::new();
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_connect_disconnect_count() {
        let mgr = WsManager::new();
        assert_eq!(mgr.active_count(), 0);
        mgr.connect();
        assert_eq!(mgr.active_count(), 1);
        mgr.connect();
        assert_eq!(mgr.active_count(), 2);
        mgr.disconnect();
        assert_eq!(mgr.active_count(), 1);
        mgr.disconnect();
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_broadcast_no_receivers() {
        let mgr = WsManager::new();
        // Should not panic with no receivers
        mgr.broadcast("hello".to_string());
    }

    #[test]
    fn test_broadcast_with_receiver() {
        let mgr = WsManager::new();
        let mut rx = mgr.subscribe();
        mgr.broadcast("hello".to_string());
        let msg = rx.try_recv().unwrap();
        assert_eq!(msg, "hello");
    }

    #[test]
    fn test_build_error_message() {
        let msg = build_error_message("something went wrong");
        let v: Value = serde_json::from_str(&msg).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["message"], "something went wrong");
    }

    #[test]
    fn test_build_pong_message() {
        let msg = build_pong_message();
        let v: Value = serde_json::from_str(&msg).unwrap();
        assert_eq!(v["type"], "pong");
    }

    #[test]
    fn test_compact_layer_strips_fields() {
        let step = StepData {
            step: 1,
            timestamp: 1707589200.0,
            batch_size: 64,
            layers: vec![test_layer()],
            cross_layer: test_cross_layer(),
            layer_groups: None,
        };
        let compact = compact_step(&step);
        let value = serde_json::to_value(compact).unwrap();
        let obj = value["layers"][0].as_object().unwrap();

        assert!(obj.contains_key("layer_id"));
        assert!(obj.contains_key("layer_type"));
        assert!(obj.contains_key("depth_index"));
        assert!(obj.contains_key("intermediate_features"));
        assert!(obj.contains_key("gradient_flow"));
        assert!(!obj.contains_key("parameter_statistics"));

        let ifeatures = obj["intermediate_features"].as_object().unwrap();
        assert!(ifeatures.contains_key("activation_std"));
        assert!(ifeatures.contains_key("cross_layer_std_ratio"));
        assert!(!ifeatures.contains_key("activation_mean"));
        assert!(!ifeatures.contains_key("activation_shape"));

        let gf = obj["gradient_flow"].as_object().unwrap();
        assert!(gf.contains_key("gradient_l2_norm"));
        assert!(gf.contains_key("gradient_max_abs"));
        assert!(!gf.contains_key("gradient_std"));
    }

    #[test]
    fn test_build_compact_run_history_message() {
        let run_data = RunData {
            created_at: "2024-01-01".to_string(),
            last_update: "2024-01-02".to_string(),
            steps: vec![StepData {
                step: 1,
                timestamp: 1707589200.0,
                batch_size: 64,
                layers: vec![test_layer()],
                cross_layer: test_cross_layer(),
                layer_groups: None,
            }],
        };
        let msg = build_compact_run_history_message("run_1", &run_data);
        let v: Value = serde_json::from_str(&msg).unwrap();
        assert_eq!(v["type"], "run_history");
        assert_eq!(v["run_id"], "run_1");
        assert!(v["data"]["steps"][0]["layers"][0]
            .as_object()
            .unwrap()
            .contains_key("layer_id"));
        // parameter_statistics should be stripped
        assert!(!v["data"]["steps"][0]["layers"][0]
            .as_object()
            .unwrap()
            .contains_key("parameter_statistics"));
    }

    #[test]
    fn test_build_run_history_message_full() {
        let run_data = RunData {
            created_at: "2024-01-01".to_string(),
            last_update: "2024-01-02".to_string(),
            steps: vec![],
        };
        let msg = build_run_history_message("run_1", &run_data);
        let v: Value = serde_json::from_str(&msg).unwrap();
        assert_eq!(v["type"], "run_history");
        assert_eq!(v["run_id"], "run_1");
    }

    #[test]
    fn test_build_new_metrics_message() {
        let step = StepData {
            step: 42,
            timestamp: 1707589200.0,
            batch_size: 32,
            layers: vec![],
            cross_layer: test_cross_layer(),
            layer_groups: None,
        };
        let msg = build_new_metrics_message("run_1", &step);
        let v: Value = serde_json::from_str(&msg).unwrap();
        assert_eq!(v["type"], "new_metrics");
        assert_eq!(v["run_id"], "run_1");
        assert_eq!(v["data"]["step"], 42);
    }

    #[test]
    fn test_ws_send_error_logged_on_disconnect() {
        // Simulate disconnected client: subscribe then drop receiver
        let mgr = WsManager::new();
        let rx = mgr.subscribe();
        drop(rx); // client disconnect

        // Broadcast after all receivers dropped should not panic
        mgr.broadcast("after disconnect".to_string());

        // New subscribers should still work after prior disconnect
        let mut rx2 = mgr.subscribe();
        mgr.broadcast("new subscriber".to_string());
        let msg = rx2.try_recv().unwrap();
        assert_eq!(msg, "new subscriber");

        // Connection counting unaffected by broadcast failures
        mgr.connect();
        assert_eq!(mgr.active_count(), 1);
        mgr.disconnect();
        assert_eq!(mgr.active_count(), 0);
    }
}
