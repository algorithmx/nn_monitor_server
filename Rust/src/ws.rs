use std::sync::atomic::{AtomicU32, Ordering};
use tokio::sync::broadcast;

use serde_json::{json, Value};

use crate::models::{RunData, RunInfo, StepData};

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
    pub fn broadcast(&self, message: &str) {
        match self.tx.send(message.to_string()) {
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

/// Build `initial_runs` message sent on WebSocket connect.
pub fn build_initial_runs_message(runs: &std::collections::HashMap<String, RunInfo>) -> String {
    let runs_map: serde_json::Map<String, Value> = runs
        .iter()
        .map(|(id, info)| (id.clone(), serde_json::to_value(info).unwrap()))
        .collect();

    json!({
        "type": "initial_runs",
        "data": Value::Object(runs_map)
    })
    .to_string()
}

/// Build `new_metrics` broadcast message.
pub fn build_new_metrics_message(run_id: &str, step_data: &StepData) -> String {
    json!({
        "type": "new_metrics",
        "run_id": run_id,
        "data": step_data
    })
    .to_string()
}

/// Build `run_history` message for full (non-lite) subscribe_run response.
pub fn build_run_history_message(run_id: &str, run_data: &RunData) -> String {
    json!({
        "type": "run_history",
        "run_id": run_id,
        "data": run_data
    })
    .to_string()
}

// ==================== Compact / Lite Helpers ====================
// These replicate Python's _compact_layer, _compact_step, _compact_run.
// Lite mode strips heavy fields (activation_mean, activation_shape,
// gradient_std, parameter_statistics) to keep run switching fast.

/// Compact a single layer: keep only fields the dashboard needs.
fn compact_layer(layer: &Value) -> Value {
    let mut compact = serde_json::Map::new();
    if let Some(obj) = layer.as_object() {
        compact.insert(
            "layer_id".into(),
            obj.get("layer_id").cloned().unwrap_or(Value::Null),
        );
        compact.insert(
            "layer_type".into(),
            obj.get("layer_type").cloned().unwrap_or(Value::Null),
        );
        compact.insert(
            "depth_index".into(),
            obj.get("depth_index").cloned().unwrap_or(Value::Null),
        );

        // Compact intermediate_features
        if let Some(ifeatures) = obj.get("intermediate_features").and_then(|v| v.as_object()) {
            let mut cif = serde_json::Map::new();
            cif.insert(
                "activation_std".into(),
                ifeatures
                    .get("activation_std")
                    .cloned()
                    .unwrap_or(Value::from(0)),
            );
            cif.insert(
                "cross_layer_std_ratio".into(),
                ifeatures
                    .get("cross_layer_std_ratio")
                    .cloned()
                    .unwrap_or(Value::Null),
            );
            compact.insert("intermediate_features".into(), Value::Object(cif));
        }

        // Compact gradient_flow
        if let Some(gf) = obj.get("gradient_flow").and_then(|v| v.as_object()) {
            let mut cgf = serde_json::Map::new();
            cgf.insert(
                "gradient_l2_norm".into(),
                gf.get("gradient_l2_norm")
                    .cloned()
                    .unwrap_or(Value::from(0)),
            );
            cgf.insert(
                "gradient_max_abs".into(),
                gf.get("gradient_max_abs")
                    .cloned()
                    .unwrap_or(Value::from(0)),
            );
            compact.insert("gradient_flow".into(), Value::Object(cgf));
        }
    }
    Value::Object(compact)
}

/// Compact a single step: compact each layer, keep step metadata.
fn compact_step(step: &StepData) -> Value {
    let compact_layers: Vec<Value> = step.layers.iter().map(compact_layer).collect();
    json!({
        "step": step.step,
        "timestamp": step.timestamp,
        "batch_size": step.batch_size,
        "layers": compact_layers,
        "cross_layer": step.cross_layer,
        "layer_groups": step.layer_groups,
    })
}

/// Build `run_history` message for lite/compact subscribe_run response.
pub fn build_compact_run_history_message(run_id: &str, run_data: &RunData) -> String {
    let compact_steps: Vec<Value> = run_data.steps.iter().map(compact_step).collect();
    json!({
        "type": "run_history",
        "run_id": run_id,
        "data": {
            "created_at": run_data.created_at,
            "last_update": run_data.last_update,
            "steps": compact_steps,
        }
    })
    .to_string()
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
    use std::collections::HashMap;

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
        mgr.broadcast("hello");
    }

    #[test]
    fn test_broadcast_with_receiver() {
        let mgr = WsManager::new();
        let mut rx = mgr.subscribe();
        mgr.broadcast("hello");
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
    fn test_build_initial_runs_message() {
        let mut runs = HashMap::new();
        runs.insert(
            "run_1".to_string(),
            RunInfo {
                created_at: "2024-01-01".to_string(),
                last_update: "2024-01-02".to_string(),
                step_count: 10,
                latest_step: Some(9),
            },
        );
        let msg = build_initial_runs_message(&runs);
        let v: Value = serde_json::from_str(&msg).unwrap();
        assert_eq!(v["type"], "initial_runs");
        assert!(v["data"]["run_1"].is_object());
    }

    #[test]
    fn test_compact_layer_strips_fields() {
        let full_layer = json!({
            "layer_id": "linear1",
            "layer_type": "Linear",
            "depth_index": 0,
            "intermediate_features": {
                "activation_std": 0.847,
                "activation_mean": -0.023,
                "activation_shape": [64, 256],
                "cross_layer_std_ratio": 1.2
            },
            "gradient_flow": {
                "gradient_l2_norm": 0.152,
                "gradient_std": 0.0034,
                "gradient_max_abs": 0.089
            },
            "parameter_statistics": {
                "weight": {"std": 0.037}
            }
        });
        let compact = compact_layer(&full_layer);
        let obj = compact.as_object().unwrap();

        // Should keep these
        assert!(obj.contains_key("layer_id"));
        assert!(obj.contains_key("layer_type"));
        assert!(obj.contains_key("depth_index"));
        assert!(obj.contains_key("intermediate_features"));
        assert!(obj.contains_key("gradient_flow"));

        // Should NOT keep parameter_statistics
        assert!(!obj.contains_key("parameter_statistics"));

        // intermediate_features should only have activation_std + cross_layer_std_ratio
        let ifeatures = obj["intermediate_features"].as_object().unwrap();
        assert!(ifeatures.contains_key("activation_std"));
        assert!(ifeatures.contains_key("cross_layer_std_ratio"));
        assert!(!ifeatures.contains_key("activation_mean"));
        assert!(!ifeatures.contains_key("activation_shape"));

        // gradient_flow should only have gradient_l2_norm + gradient_max_abs
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
                layers: vec![json!({
                    "layer_id": "l1",
                    "layer_type": "Linear",
                    "depth_index": 0,
                    "intermediate_features": {
                        "activation_std": 0.5,
                        "activation_mean": 0.1,
                        "activation_shape": [32, 64],
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": 0.2,
                        "gradient_std": 0.01,
                        "gradient_max_abs": 0.05,
                    },
                    "parameter_statistics": { "weight": { "std": 0.04 } }
                })],
                cross_layer: json!({"feature_std_gradient": -0.1}),
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
            cross_layer: json!(null),
            layer_groups: None,
        };
        let msg = build_new_metrics_message("run_1", &step);
        let v: Value = serde_json::from_str(&msg).unwrap();
        assert_eq!(v["type"], "new_metrics");
        assert_eq!(v["run_id"], "run_1");
        assert_eq!(v["data"]["step"], 42);
    }
}
