use std::collections::HashMap;
use std::sync::Arc;

use chrono::Local;
use tokio::sync::RwLock;

use crate::models::{MetricsPayload, RunData, RunInfo, StepData};

// ==================== In-Memory Storage =====================

/// Sanitize layer ID: convert dots to slashes for consistent format.
fn sanitize_layer_id(layer_id: &str) -> String {
    layer_id.replace('.', "/")
}

fn now_iso() -> String {
    Local::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
}

pub struct MetricsStore {
    runs: Arc<RwLock<HashMap<String, RunData>>>,
    max_runs: usize,
    max_steps_per_run: usize,
}

impl MetricsStore {
    pub fn new(max_runs: usize, max_steps_per_run: usize) -> Self {
        Self {
            runs: Arc::new(RwLock::new(HashMap::new())),
            max_runs,
            max_steps_per_run,
        }
    }

    pub async fn add_metrics(&self, payload: MetricsPayload, raw: serde_json::Value) -> StepData {
        payload
            .validate()
            .expect("add_metrics: payload validation failed");

        let mut runs = self.runs.write().await;
        let run_id = &payload.metadata.run_id;

        if !runs.contains_key(run_id) && runs.len() >= self.max_runs {
            if let Some(oldest_key) = runs
                .iter()
                .min_by_key(|(_, run)| run.last_update.as_str())
                .map(|(k, _)| k.clone())
            {
                runs.remove(&oldest_key);
            }
        }

        if !runs.contains_key(run_id) {
            let now = now_iso();
            runs.insert(
                run_id.clone(),
                RunData {
                    created_at: now.clone(),
                    last_update: now,
                    steps: Vec::new(),
                },
            );
        }

        let run = runs
            .get_mut(run_id)
            .expect("run was just inserted");

        let step_data = build_step_data(&payload, &raw);

        if let Some(existing) = run.steps.iter_mut().find(|s| s.step == step_data.step) {
            *existing = step_data.clone();
        } else {
            run.steps.push(step_data.clone());
            run.steps.sort_by_key(|s| s.step);
        }

        if run.steps.len() > self.max_steps_per_run {
            let excess = run.steps.len() - self.max_steps_per_run;
            run.steps.drain(0..excess);
        }

        run.last_update = now_iso();

        step_data
    }

    pub async fn get_run(&self, run_id: &str) -> Option<RunData> {
        let runs = self.runs.read().await;
        runs.get(run_id).cloned()
    }

    pub async fn get_all_runs(&self) -> HashMap<String, RunInfo> {
        let runs = self.runs.read().await;
        runs.iter()
            .map(|(run_id, run)| {
                (
                    run_id.clone(),
                    RunInfo {
                        created_at: run.created_at.clone(),
                        last_update: run.last_update.clone(),
                        step_count: run.steps.len() as u32,
                        latest_step: run.steps.last().map(|s| s.step),
                    },
                )
            })
            .collect()
    }

    pub async fn get_latest_step(&self, run_id: &str) -> Option<StepData> {
        let runs = self.runs.read().await;
        runs.get(run_id).and_then(|run| run.steps.last().cloned())
    }
}

/// Build StepData from validated payload + raw JSON, applying layer ID sanitization.
fn build_step_data(payload: &MetricsPayload, raw: &serde_json::Value) -> StepData {
    let raw_layers = raw
        .get("layer_statistics")
        .and_then(|v| v.as_array())
        .expect("build_step_data: layer_statistics missing or not an array");

    let sanitized_layers: Vec<serde_json::Value> = raw_layers
        .iter()
        .map(|layer| {
            let mut layer_val = layer.clone();
            if let Some(obj) = layer_val.as_object_mut() {
                if let Some(layer_id) = obj.get("layer_id").and_then(|v| v.as_str()) {
                    obj.insert(
                        "layer_id".to_string(),
                        serde_json::Value::String(sanitize_layer_id(layer_id)),
                    );
                }
            }
            layer_val
        })
        .collect();

    let sanitized_layer_groups = payload.metadata.layer_groups.as_ref().map(|groups| {
        groups
            .iter()
            .map(|(key, layer_ids)| {
                let sanitized_ids: Vec<String> =
                    layer_ids.iter().map(|id| sanitize_layer_id(id)).collect();
                (key.clone(), sanitized_ids)
            })
            .collect()
    });

    let cross_layer = raw
        .get("cross_layer_analysis")
        .cloned()
        .expect("build_step_data: cross_layer_analysis missing");

    StepData {
        step: payload.metadata.global_step,
        timestamp: *payload.metadata.timestamp,
        batch_size: payload.metadata.batch_size,
        layers: sanitized_layers,
        cross_layer,
        layer_groups: sanitized_layer_groups,
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_payload_and_raw(run_id: &str, global_step: u64) -> (MetricsPayload, serde_json::Value) {
        let json = serde_json::json!({
            "metadata": {
                "run_id": run_id,
                "timestamp": 1707589200.123,
                "global_step": global_step,
                "batch_size": 32,
                "layer_groups": {
                    "enc": ["encoder.linear1"]
                }
            },
            "layer_statistics": [
                {
                    "layer_id": "encoder.linear1",
                    "layer_type": "Linear",
                    "depth_index": 0,
                    "intermediate_features": {
                        "activation_std": 0.5,
                        "activation_mean": 0.01,
                        "activation_shape": [32, 64],
                        "cross_layer_std_ratio": null
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": 0.1,
                        "gradient_std": 0.01,
                        "gradient_max_abs": 0.05
                    },
                    "parameter_statistics": {
                        "weight": {
                            "std": 0.02,
                            "mean": 0.001,
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
                "feature_std_gradient": -0.1,
                "gradient_norm_ratio": {}
            }
        });
        let payload: MetricsPayload =
            serde_json::from_value(json.clone()).expect("test payload should deserialize");
        (payload, json)
    }

    fn make_payload_with_dots(run_id: &str, global_step: u64) -> (MetricsPayload, serde_json::Value) {
        let json = serde_json::json!({
            "metadata": {
                "run_id": run_id,
                "timestamp": 1707589200.123,
                "global_step": global_step,
                "batch_size": 32,
                "layer_groups": {
                    "enc": ["encoder.linear1", "encoder.relu.1"]
                }
            },
            "layer_statistics": [
                {
                    "layer_id": "encoder.linear1",
                    "layer_type": "Linear",
                    "depth_index": 0,
                    "intermediate_features": {
                        "activation_std": 0.5,
                        "activation_mean": 0.01,
                        "activation_shape": [32, 64],
                        "cross_layer_std_ratio": null
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": 0.1,
                        "gradient_std": 0.01,
                        "gradient_max_abs": 0.05
                    },
                    "parameter_statistics": {
                        "weight": {
                            "std": 0.02,
                            "mean": 0.001,
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
                "feature_std_gradient": -0.1,
                "gradient_norm_ratio": {}
            }
        });
        let payload: MetricsPayload =
            serde_json::from_value(json.clone()).expect("test payload should deserialize");
        (payload, json)
    }

    #[tokio::test]
    async fn test_step_dedup_replaces() {
        let store = MetricsStore::new(10, 1000);
        let (p1, r1) = make_payload_and_raw("run1", 100);
        let (p2, r2) = make_payload_and_raw("run1", 100);

        let s1 = store.add_metrics(p1, r1).await;
        let s2 = store.add_metrics(p2, r2).await;

        let run = store.get_run("run1").await.expect("run should exist");
        assert_eq!(run.steps.len(), 1, "step count should be 1 after dedup");
        assert_eq!(s2.step, 100);
        assert_eq!(run.steps[0].step, 100);
        assert_eq!(s1.step, s2.step);
    }

    #[tokio::test]
    async fn test_run_eviction_oldest() {
        let store = MetricsStore::new(3, 1000);

        for i in 0..3u64 {
            let (p, r) = make_payload_and_raw(&format!("run{}", i), i);
            store.add_metrics(p, r).await;
            tokio::time::sleep(std::time::Duration::from_millis(15)).await;
        }

        assert!(store.get_run("run0").await.is_some());
        assert!(store.get_run("run1").await.is_some());
        assert!(store.get_run("run2").await.is_some());

        let (p, r) = make_payload_and_raw("run3", 0);
        store.add_metrics(p, r).await;

        assert!(
            store.get_run("run0").await.is_none(),
            "run0 should be evicted"
        );
        assert!(store.get_run("run1").await.is_some());
        assert!(store.get_run("run2").await.is_some());
        assert!(store.get_run("run3").await.is_some());
    }

    #[tokio::test]
    async fn test_step_eviction_oldest() {
        let store = MetricsStore::new(10, 5);

        for step in 1..=7u64 {
            let (p, r) = make_payload_and_raw("run1", step);
            store.add_metrics(p, r).await;
        }

        let run = store.get_run("run1").await.expect("run should exist");
        assert_eq!(run.steps.len(), 5, "should keep only last 5 steps");
        assert_eq!(run.steps[0].step, 3);
        assert_eq!(run.steps[4].step, 7);
    }

    #[tokio::test]
    async fn test_layer_id_sanitization() {
        let store = MetricsStore::new(10, 1000);
        let (p, r) = make_payload_with_dots("run1", 1);

        let step = store.add_metrics(p, r).await;

        let first_layer = &step.layers[0];
        let layer_id = first_layer
            .get("layer_id")
            .expect("layer_id should exist")
            .as_str()
            .expect("layer_id should be string");
        assert_eq!(
            layer_id, "encoder/linear1",
            "layer_id dots should become slashes"
        );

        let groups = step
            .layer_groups
            .as_ref()
            .expect("layer_groups should exist");
        let enc_ids = groups
            .get("enc")
            .expect("enc group should exist");
        assert_eq!(enc_ids[0], "encoder/linear1");
        assert_eq!(enc_ids[1], "encoder/relu/1");

        let run = store.get_run("run1").await.expect("run should exist");
        let stored_layer_id = run.steps[0].layers[0]
            .get("layer_id")
            .expect("layer_id should exist")
            .as_str()
            .expect("layer_id should be string");
        assert_eq!(stored_layer_id, "encoder/linear1");

        let stored_groups = run.steps[0]
            .layer_groups
            .as_ref()
            .expect("layer_groups should exist");
        let stored_enc = stored_groups.get("enc").expect("enc group should exist");
        assert_eq!(stored_enc[0], "encoder/linear1");
        assert_eq!(stored_enc[1], "encoder/relu/1");
    }

    #[tokio::test]
    async fn test_last_update_on_dedup() {
        let store = MetricsStore::new(10, 1000);

        let (p1, r1) = make_payload_and_raw("run1", 42);
        store.add_metrics(p1, r1).await;

        let first_update = store
            .get_run("run1")
            .await
            .expect("run should exist")
            .last_update
            .clone();

        tokio::time::sleep(std::time::Duration::from_millis(15)).await;

        let (p2, r2) = make_payload_and_raw("run1", 42);
        store.add_metrics(p2, r2).await;

        let second_update = store
            .get_run("run1")
            .await
            .expect("run should exist")
            .last_update
            .clone();

        assert_ne!(
            first_update, second_update,
            "last_update must change even for duplicate step submission"
        );
    }

    #[tokio::test]
    async fn test_steps_sorted() {
        let store = MetricsStore::new(10, 1000);

        // Submit steps out of order: 300, 100, 200 → should sort to [100, 200, 300]
        let (p300, r300) = make_payload_and_raw("run1", 300);
        let (p100, r100) = make_payload_and_raw("run1", 100);
        let (p200, r200) = make_payload_and_raw("run1", 200);

        store.add_metrics(p300, r300).await;
        store.add_metrics(p100, r100).await;
        store.add_metrics(p200, r200).await;

        let run = store.get_run("run1").await.expect("run should exist");
        assert_eq!(run.steps.len(), 3);
        assert_eq!(run.steps[0].step, 100, "steps should be sorted: [100, 200, 300]");
        assert_eq!(run.steps[1].step, 200);
        assert_eq!(run.steps[2].step, 300);
    }

    #[tokio::test]
    async fn test_latest_step() {
        let store = MetricsStore::new(10, 1000);

        let (p, r) = make_payload_and_raw("run1", 50);
        store.add_metrics(p, r).await;
        let (p, r) = make_payload_and_raw("run1", 150);
        store.add_metrics(p, r).await;
        let (p, r) = make_payload_and_raw("run1", 100);
        store.add_metrics(p, r).await;

        let latest = store
            .get_latest_step("run1")
            .await
            .expect("should have latest step");
        assert_eq!(
            latest.step, 150,
            "latest step should be highest step number"
        );
    }
}
