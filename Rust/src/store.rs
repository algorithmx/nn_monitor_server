use std::borrow::Cow;
use std::collections::HashMap as StdHashMap;

use chrono::Local;
use hashbrown::HashMap;
use tokio::sync::RwLock;

use crate::models::{MetricsPayload, RunData, RunInfo, StepData};

// ==================== In-Memory Storage =====================

/// Sanitize layer ID: convert dots to slashes for consistent format.
fn sanitize_layer_id(layer_id: &str) -> Cow<'_, str> {
    if layer_id.contains('.') {
        Cow::Owned(layer_id.replace('.', "/"))
    } else {
        Cow::Borrowed(layer_id)
    }
}

fn now_iso() -> String {
    Local::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
}

pub struct MetricsStore {
    runs: RwLock<HashMap<String, RunData>>,
    max_runs: usize,
    max_steps_per_run: usize,
}

impl MetricsStore {
    pub fn new(max_runs: usize, max_steps_per_run: usize) -> Self {
        Self {
            runs: RwLock::new(HashMap::new()),
            max_runs,
            max_steps_per_run,
        }
    }

    pub async fn add_metrics(&self, payload: MetricsPayload) -> StepData {
        payload
            .validate()
            .expect("add_metrics: payload validation failed");

        let step_data = build_step_data(&payload);
        let run_id = payload.metadata.run_id;

        let mut runs = self.runs.write().await;

        let is_new_run = !runs.contains_key(&run_id);

        if is_new_run && runs.len() >= self.max_runs {
            if let Some(oldest_key) = runs
                .iter()
                .min_by_key(|(_, run)| run.last_update.as_str())
                .map(|(k, _)| k.clone())
            {
                runs.remove(&oldest_key);
            }
        }

        if is_new_run {
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
            .get_mut(&run_id)
            .expect("run was just inserted");

        let result = step_data.clone();
        let step = step_data.step;
        match run.steps.binary_search_by_key(&step, |s| s.step) {
            Ok(idx) => run.steps[idx] = step_data,
            Err(idx) => run.steps.insert(idx, step_data),
        }

        if run.steps.len() > self.max_steps_per_run {
            let excess = run.steps.len() - self.max_steps_per_run;
            run.steps.drain(0..excess);
        }

        if !is_new_run {
            run.last_update = now_iso();
        }

        result
    }

    pub async fn get_run(&self, run_id: &str) -> Option<RunData> {
        let runs = self.runs.read().await;
        runs.get(run_id).cloned()
    }

    pub async fn get_all_runs(&self) -> StdHashMap<String, RunInfo> {
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

/// Build StepData from validated payload, applying layer ID sanitization.
fn build_step_data(payload: &MetricsPayload) -> StepData {
    let sanitized_layers: Vec<serde_json::Value> = payload.layer_statistics
        .iter()
        .map(|ls| {
            let sanitized_id = sanitize_layer_id(&ls.layer_id);
            serde_json::json!({
                "layer_id": sanitized_id,
                "layer_type": ls.layer_type,
                "depth_index": ls.depth_index,
                "intermediate_features": {
                    "activation_std": ls.intermediate_features.activation_std,
                    "activation_mean": ls.intermediate_features.activation_mean,
                    "activation_shape": ls.intermediate_features.activation_shape,
                    "cross_layer_std_ratio": ls.intermediate_features.cross_layer_std_ratio,
                },
                "gradient_flow": {
                    "gradient_l2_norm": ls.gradient_flow.gradient_l2_norm,
                    "gradient_std": ls.gradient_flow.gradient_std,
                    "gradient_max_abs": ls.gradient_flow.gradient_max_abs,
                },
                "parameter_statistics": {
                    "weight": {
                        "std": ls.parameter_statistics.weight.std,
                        "mean": ls.parameter_statistics.weight.mean,
                        "spectral_norm": ls.parameter_statistics.weight.spectral_norm,
                        "frobenius_norm": ls.parameter_statistics.weight.frobenius_norm,
                    },
                    "bias": ls.parameter_statistics.bias,
                },
            })
        })
        .collect();

    let sanitized_layer_groups = payload.metadata.layer_groups.as_ref().map(|groups| {
        groups
            .iter()
            .map(|(key, layer_ids)| {
                let sanitized_ids: Vec<String> =
                    layer_ids.iter().map(|id| sanitize_layer_id(id).into_owned()).collect();
                (key.clone(), sanitized_ids)
            })
            .collect()
    });

    let cross_layer = serde_json::to_value(&payload.cross_layer_analysis)
        .expect("cross_layer_analysis serialization should never fail");

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
    use std::sync::Arc;

    use super::*;

    fn make_payload(run_id: &str, global_step: u64) -> MetricsPayload {
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
        serde_json::from_value(json).expect("test payload should deserialize")
    }

    fn make_payload_with_dots(run_id: &str, global_step: u64) -> MetricsPayload {
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
        serde_json::from_value(json).expect("test payload should deserialize")
    }

    #[tokio::test]
    async fn test_step_dedup_replaces() {
        let store = MetricsStore::new(10, 1000);
        let p1 = make_payload("run1", 100);
        let p2 = make_payload("run1", 100);

        let s1 = store.add_metrics(p1).await;
        let s2 = store.add_metrics(p2).await;

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
            let p = make_payload(&format!("run{}", i), i);
            store.add_metrics(p).await;
            tokio::time::sleep(std::time::Duration::from_millis(15)).await;
        }

        assert!(store.get_run("run0").await.is_some());
        assert!(store.get_run("run1").await.is_some());
        assert!(store.get_run("run2").await.is_some());

        let p = make_payload("run3", 0);
        store.add_metrics(p).await;

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
            let p = make_payload("run1", step);
            store.add_metrics(p).await;
        }

        let run = store.get_run("run1").await.expect("run should exist");
        assert_eq!(run.steps.len(), 5, "should keep only last 5 steps");
        assert_eq!(run.steps[0].step, 3);
        assert_eq!(run.steps[4].step, 7);
    }

    #[tokio::test]
    async fn test_layer_id_sanitization() {
        let store = MetricsStore::new(10, 1000);
        let p = make_payload_with_dots("run1", 1);

        let step = store.add_metrics(p).await;

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

        let p1 = make_payload("run1", 42);
        store.add_metrics(p1).await;

        let first_update = store
            .get_run("run1")
            .await
            .expect("run should exist")
            .last_update
            .clone();

        tokio::time::sleep(std::time::Duration::from_millis(15)).await;

        let p2 = make_payload("run1", 42);
        store.add_metrics(p2).await;

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
        let p300 = make_payload("run1", 300);
        let p100 = make_payload("run1", 100);
        let p200 = make_payload("run1", 200);

        store.add_metrics(p300).await;
        store.add_metrics(p100).await;
        store.add_metrics(p200).await;

        let run = store.get_run("run1").await.expect("run should exist");
        assert_eq!(run.steps.len(), 3);
        assert_eq!(run.steps[0].step, 100, "steps should be sorted: [100, 200, 300]");
        assert_eq!(run.steps[1].step, 200);
        assert_eq!(run.steps[2].step, 300);
    }

    #[tokio::test]
    async fn test_latest_step() {
        let store = MetricsStore::new(10, 1000);

        let p = make_payload("run1", 50);
        store.add_metrics(p).await;
        let p = make_payload("run1", 150);
        store.add_metrics(p).await;
        let p = make_payload("run1", 100);
        store.add_metrics(p).await;

        let latest = store
            .get_latest_step("run1")
            .await
            .expect("should have latest step");
        assert_eq!(
            latest.step, 150,
            "latest step should be highest step number"
        );
    }

    // ---- New comprehensive tests ----

    #[tokio::test]
    async fn test_run_eviction_boundary_at_max() {
        let store = MetricsStore::new(3, 1000);

        for i in 0..3u64 {
            let p = make_payload(&format!("run{}", i), i);
            store.add_metrics(p).await;
            tokio::time::sleep(std::time::Duration::from_millis(15)).await;
        }

        assert!(store.get_run("run0").await.is_some(), "run0 should exist");
        assert!(store.get_run("run1").await.is_some(), "run1 should exist");
        assert!(store.get_run("run2").await.is_some(), "run2 should exist");

        let p = make_payload("run3", 0);
        store.add_metrics(p).await;

        assert!(
            store.get_run("run0").await.is_none(),
            "run0 should be evicted when max_runs boundary exceeded"
        );
        assert!(store.get_run("run1").await.is_some(), "run1 should remain");
        assert!(store.get_run("run2").await.is_some(), "run2 should remain");
        assert!(store.get_run("run3").await.is_some(), "run3 should exist");
    }

    #[tokio::test]
    async fn test_run_eviction_updates_last_update() {
        let store = MetricsStore::new(2, 1000);

        let p = make_payload("run0", 0);
        store.add_metrics(p).await;
        tokio::time::sleep(std::time::Duration::from_millis(15)).await;

        let p = make_payload("run1", 1);
        store.add_metrics(p).await;

        let run0_update = store.get_run("run0").await.unwrap().last_update;
        let run1_update = store.get_run("run1").await.unwrap().last_update;

        assert!(
            run0_update < run1_update,
            "run0 last_update ({}) should be older than run1 ({})",
            run0_update,
            run1_update
        );

        let p = make_payload("run2", 2);
        store.add_metrics(p).await;

        assert!(
            store.get_run("run0").await.is_none(),
            "evicted run should have had the oldest last_update"
        );
        assert!(store.get_run("run1").await.is_some());
        assert!(store.get_run("run2").await.is_some());
    }

    #[tokio::test]
    async fn test_step_eviction_boundary_at_max() {
        let store = MetricsStore::new(10, 5);

        for step in 1..=5u64 {
            let p = make_payload("run1", step);
            store.add_metrics(p).await;
        }

        // All 5 steps should exist
        let run = store.get_run("run1").await.expect("run should exist");
        assert_eq!(run.steps.len(), 5, "all 5 steps should exist at boundary");
        assert_eq!(run.steps[0].step, 1);
        assert_eq!(run.steps[4].step, 5);

        let p = make_payload("run1", 6);
        store.add_metrics(p).await;

        let run = store.get_run("run1").await.expect("run should exist");
        assert_eq!(run.steps.len(), 5, "should still have max_steps_per_run steps");
        assert_eq!(
            run.steps[0].step, 2,
            "oldest step should be evicted when boundary exceeded"
        );
        assert_eq!(run.steps[4].step, 6, "newest step should be step 6");
    }

    #[tokio::test]
    async fn test_multiple_layers_sanitization() {
        let store = MetricsStore::new(10, 1000);

        let json = serde_json::json!({
            "metadata": {
                "run_id": "run1",
                "timestamp": 1707589200.123,
                "global_step": 1,
                "batch_size": 32,
                "layer_groups": {
                    "all": ["layer.0.weight", "layer.1.bias", "layer.2.norm"]
                }
            },
            "layer_statistics": [
                {
                    "layer_id": "layer.0.weight",
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
                },
                {
                    "layer_id": "layer.1.bias",
                    "layer_type": "Linear",
                    "depth_index": 1,
                    "intermediate_features": {
                        "activation_std": 0.4,
                        "activation_mean": 0.02,
                        "activation_shape": [32, 64],
                        "cross_layer_std_ratio": null
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": 0.08,
                        "gradient_std": 0.008,
                        "gradient_max_abs": 0.04
                    },
                    "parameter_statistics": {
                        "weight": {
                            "std": 0.015,
                            "mean": 0.002,
                            "spectral_norm": 0.9,
                            "frobenius_norm": 1.2
                        },
                        "bias": {
                            "std": 0.008,
                            "mean_abs": 0.003
                        }
                    }
                },
                {
                    "layer_id": "layer.2.norm",
                    "layer_type": "BatchNorm",
                    "depth_index": 2,
                    "intermediate_features": {
                        "activation_std": 0.3,
                        "activation_mean": 0.015,
                        "activation_shape": [32, 64],
                        "cross_layer_std_ratio": null
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": 0.06,
                        "gradient_std": 0.005,
                        "gradient_max_abs": 0.03
                    },
                    "parameter_statistics": {
                        "weight": {
                            "std": 0.01,
                            "mean": 0.0,
                            "spectral_norm": 0.8,
                            "frobenius_norm": 1.0
                        },
                        "bias": {
                            "std": 0.005,
                            "mean_abs": 0.002
                        }
                    }
                }
            ],
            "cross_layer_analysis": {
                "feature_std_gradient": -0.15,
                "gradient_norm_ratio": {}
            }
        });

        let payload: MetricsPayload =
            serde_json::from_value(json).expect("payload should deserialize");
        let step = store.add_metrics(payload).await;

        assert_eq!(step.layers.len(), 3, "should have 3 layers");

        let expected_ids = vec!["layer/0/weight", "layer/1/bias", "layer/2/norm"];
        for (i, expected) in expected_ids.iter().enumerate() {
            let actual = step.layers[i]
                .get("layer_id")
                .expect("layer_id should exist")
                .as_str()
                .expect("layer_id should be string");
            assert_eq!(
                actual, *expected,
                "layer {} id should be sanitized: dots → slashes",
                i
            );
        }

        let groups = step.layer_groups.as_ref().expect("layer_groups should exist");
        let all_ids = groups.get("all").expect("'all' group should exist");
        assert_eq!(all_ids.len(), 3);
        assert_eq!(all_ids[0], "layer/0/weight");
        assert_eq!(all_ids[1], "layer/1/bias");
        assert_eq!(all_ids[2], "layer/2/norm");
    }

    #[tokio::test]
    async fn test_layer_groups_multiple_groups() {
        let store = MetricsStore::new(10, 1000);

        let json = serde_json::json!({
            "metadata": {
                "run_id": "run1",
                "timestamp": 1707589200.123,
                "global_step": 1,
                "batch_size": 32,
                "layer_groups": {
                    "encoder": ["enc.linear.1", "enc.relu.2"],
                    "decoder": ["dec.linear.1", "dec.softmax.1"]
                }
            },
            "layer_statistics": [
                {
                    "layer_id": "enc.linear.1",
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
                },
                {
                    "layer_id": "enc.relu.2",
                    "layer_type": "ReLU",
                    "depth_index": 1,
                    "intermediate_features": {
                        "activation_std": 0.4,
                        "activation_mean": 0.02,
                        "activation_shape": [32, 64],
                        "cross_layer_std_ratio": null
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": 0.08,
                        "gradient_std": 0.008,
                        "gradient_max_abs": 0.04
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
                    "layer_id": "dec.linear.1",
                    "layer_type": "Linear",
                    "depth_index": 2,
                    "intermediate_features": {
                        "activation_std": 0.3,
                        "activation_mean": 0.015,
                        "activation_shape": [32, 64],
                        "cross_layer_std_ratio": null
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": 0.06,
                        "gradient_std": 0.005,
                        "gradient_max_abs": 0.03
                    },
                    "parameter_statistics": {
                        "weight": {
                            "std": 0.01,
                            "mean": 0.0,
                            "spectral_norm": 0.8,
                            "frobenius_norm": 1.0
                        },
                        "bias": {
                            "std": 0.005,
                            "mean_abs": 0.002
                        }
                    }
                },
                {
                    "layer_id": "dec.softmax.1",
                    "layer_type": "Softmax",
                    "depth_index": 3,
                    "intermediate_features": {
                        "activation_std": 0.2,
                        "activation_mean": 0.25,
                        "activation_shape": [32, 10],
                        "cross_layer_std_ratio": null
                    },
                    "gradient_flow": {
                        "gradient_l2_norm": 0.04,
                        "gradient_std": 0.003,
                        "gradient_max_abs": 0.02
                    },
                    "parameter_statistics": {
                        "weight": {
                            "std": 0.008,
                            "mean": 0.001,
                            "spectral_norm": 0.5,
                            "frobenius_norm": 0.7
                        },
                        "bias": {
                            "std": 0.004,
                            "mean_abs": 0.001
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
            serde_json::from_value(json).expect("payload should deserialize");
        let step = store.add_metrics(payload).await;

        let groups = step.layer_groups.as_ref().expect("layer_groups should exist");

        let enc = groups.get("encoder").expect("encoder group should exist");
        assert_eq!(enc[0], "enc/linear/1", "encoder layer 1 sanitized");
        assert_eq!(enc[1], "enc/relu/2", "encoder layer 2 sanitized");

        let dec = groups.get("decoder").expect("decoder group should exist");
        assert_eq!(dec[0], "dec/linear/1", "decoder layer 1 sanitized");
        assert_eq!(dec[1], "dec/softmax/1", "decoder layer 2 sanitized");
    }

    #[tokio::test]
    async fn test_get_run_nonexistent() {
        let store = MetricsStore::new(10, 1000);

        let result = store.get_run("nonexistent_run").await;
        assert!(
            result.is_none(),
            "get_run should return None for non-existent run_id"
        );
    }

    #[tokio::test]
    async fn test_get_latest_step_nonexistent() {
        let store = MetricsStore::new(10, 1000);

        let result = store.get_latest_step("ghost_run").await;
        assert!(
            result.is_none(),
            "get_latest_step should return None for non-existent run_id"
        );

        let p = make_payload("existing_run", 1);
        store.add_metrics(p).await;

        let result = store.get_latest_step("still_nonexistent").await;
        assert!(
            result.is_none(),
            "get_latest_step should return None when run exists but queried run_id does not"
        );
    }

    #[tokio::test]
    async fn test_get_all_runs_empty() {
        let store = MetricsStore::new(10, 1000);

        let all_runs = store.get_all_runs().await;
        assert!(
            all_runs.is_empty(),
            "get_all_runs should return empty HashMap when no runs exist"
        );
    }

    #[tokio::test]
    async fn test_get_all_runs_multiple() {
        let store = MetricsStore::new(10, 1000);

        let p = make_payload("run_a", 1);
        store.add_metrics(p).await;
        tokio::time::sleep(std::time::Duration::from_millis(15)).await;

        let p = make_payload("run_b", 10);
        store.add_metrics(p).await;
        let p = make_payload("run_b", 20);
        store.add_metrics(p).await;
        tokio::time::sleep(std::time::Duration::from_millis(15)).await;

        let p = make_payload("run_c", 100);
        store.add_metrics(p).await;
        let p = make_payload("run_c", 200);
        store.add_metrics(p).await;
        let p = make_payload("run_c", 300);
        store.add_metrics(p).await;

        let all_runs = store.get_all_runs().await;
        assert_eq!(all_runs.len(), 3, "should have 3 runs");

        let info_a = all_runs.get("run_a").expect("run_a should exist");
        assert_eq!(info_a.step_count, 1);
        assert_eq!(info_a.latest_step, Some(1));

        let info_b = all_runs.get("run_b").expect("run_b should exist");
        assert_eq!(info_b.step_count, 2);
        assert_eq!(info_b.latest_step, Some(20));

        let info_c = all_runs.get("run_c").expect("run_c should exist");
        assert_eq!(info_c.step_count, 3);
        assert_eq!(info_c.latest_step, Some(300));

        for (name, info) in all_runs {
            assert!(
                !info.created_at.is_empty(),
                "{} created_at should not be empty",
                name
            );
            assert!(
                !info.last_update.is_empty(),
                "{} last_update should not be empty",
                name
            );
        }
    }

    #[tokio::test]
    async fn test_add_metrics_preserves_cross_layer() {
        let store = MetricsStore::new(10, 1000);

        let p = make_payload("run1", 42);
        let expected_cross = serde_json::to_value(&p.cross_layer_analysis)
            .expect("cross_layer_analysis serialization should never fail");

        let step = store.add_metrics(p).await;

        assert_eq!(
            step.cross_layer, expected_cross,
            "cross_layer_analysis should be preserved from typed struct serialization"
        );

        let run = store.get_run("run1").await.expect("run should exist");
        assert_eq!(
            run.steps[0].cross_layer, expected_cross,
            "cross_layer_analysis should be preserved in stored step"
        );
    }

    #[tokio::test]
    async fn test_concurrent_add_metrics() {
        let store = Arc::new(MetricsStore::new(10, 1000));
        let num_tasks = 10;

        let mut handles = Vec::new();
        for i in 0..num_tasks {
            let store_clone = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                let p = make_payload(&format!("concurrent_run_{}", i), i as u64);
                store_clone.add_metrics(p).await
            }));
        }

        for handle in handles {
            let result = handle.await.expect("task should not panic");
            assert_eq!(
                result.step, result.step,
                "each task should return a valid StepData"
            );
        }

        for i in 0..num_tasks {
            let run_id = format!("concurrent_run_{}", i);
            let run = store
                .get_run(&run_id)
                .await
                .unwrap_or_else(|| panic!("{} should exist", run_id));
            assert_eq!(
                run.steps.len(),
                1,
                "{} should have exactly 1 step",
                run_id
            );
            assert_eq!(
                run.steps[0].step, i as u64,
                "{} should have step {}",
                run_id, i
            );
        }

        let all_runs = store.get_all_runs().await;
        assert_eq!(
            all_runs.len(),
            num_tasks,
            "total run count should match number of concurrent tasks"
        );
    }
}
