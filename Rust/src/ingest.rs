use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::{mpsc, Notify};

use crate::models::MetricsPayload;
use crate::{store::MetricsStore, ws::WsManager};

pub struct IngestItem {
    pub payload: MetricsPayload,
}

#[derive(Debug)]
pub struct IngestStats {
    accepted: AtomicU64,
    processed: AtomicU64,
    dropped: AtomicU64,
    capacity: usize,
    notify: Notify,
}

impl IngestStats {
    pub fn new(capacity: usize) -> Self {
        Self {
            accepted: AtomicU64::new(0),
            processed: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            capacity,
            notify: Notify::new(),
        }
    }

    pub fn accepted_count(&self) -> u64 {
        self.accepted.load(Ordering::Relaxed)
    }

    pub fn processed_count(&self) -> u64 {
        self.processed.load(Ordering::Relaxed)
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn mark_accepted(&self) {
        self.accepted.fetch_add(1, Ordering::Relaxed);
    }

    pub fn mark_processed(&self) {
        self.processed.fetch_add(1, Ordering::Relaxed);
        self.notify.notify_waiters();
    }

    pub async fn wait_for_accepted_items(&self) {
        let target = self.accepted_count();
        while self.processed_count() < target {
            self.notify.notified().await;
        }
    }

    pub fn mark_dropped(&self) {
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn channel(
    capacity: usize,
) -> (
    mpsc::Sender<IngestItem>,
    mpsc::Receiver<IngestItem>,
    Arc<IngestStats>,
) {
    let capacity = capacity.max(1);
    let (tx, rx) = mpsc::channel(capacity);
    let stats = Arc::new(IngestStats::new(capacity));
    (tx, rx, stats)
}

pub fn spawn_worker(
    mut rx: mpsc::Receiver<IngestItem>,
    store: Arc<MetricsStore>,
    ws_manager: Arc<WsManager>,
    stats: Arc<IngestStats>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(item) = rx.recv().await {
            let (_run_id, msg) = store.add_validated_metrics_and_message(item.payload).await;
            ws_manager.broadcast(msg);
            stats.mark_processed();
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::mpsc::error::TrySendError;

    fn make_test_payload() -> MetricsPayload {
        serde_json::from_value(serde_json::json!({
            "metadata": {
                "run_id": "test-run",
                "timestamp": 1707589200.123,
                "global_step": 1,
                "batch_size": 32,
                "layer_groups": null
            },
            "layer_statistics": [{
                "layer_id": "layer.1",
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
                    "bias": null
                }
            }],
            "cross_layer_analysis": {
                "feature_std_gradient": -0.1,
                "gradient_norm_ratio": {}
            }
        }))
        .expect("test payload should parse")
    }

    #[tokio::test]
    async fn test_queue_full_returns_full_error() {
        let (tx, _rx, _stats) = channel(1);
        let item = IngestItem {
            payload: make_test_payload(),
        };
        tx.try_send(item).expect("first send should succeed");

        let item2 = IngestItem {
            payload: make_test_payload(),
        };
        let result = tx.try_send(item2);
        assert!(
            matches!(result, Err(TrySendError::Full(_))),
            "expected TrySendError::Full, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_queue_closed_returns_closed_error() {
        let (tx, rx, _stats) = channel(1);
        let item = IngestItem {
            payload: make_test_payload(),
        };
        tx.try_send(item).expect("first send should succeed");

        drop(rx);

        let item2 = IngestItem {
            payload: make_test_payload(),
        };
        let result = tx.try_send(item2);
        assert!(
            matches!(result, Err(TrySendError::Closed(_))),
            "expected TrySendError::Closed, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_ingest_stats_counters() {
        let stats = IngestStats::new(256);

        // Mark accepted and dropped
        stats.mark_accepted();
        stats.mark_accepted();
        stats.mark_accepted();
        stats.mark_dropped();

        assert_eq!(stats.accepted_count(), 3);
        assert_eq!(stats.dropped_count(), 1);
        assert_eq!(stats.processed_count(), 0);
        assert_eq!(stats.capacity(), 256);

        // Mark processed
        stats.mark_processed();
        stats.mark_processed();
        stats.mark_processed();
        assert_eq!(stats.processed_count(), 3);

        // wait_for_accepted_items should return immediately when processed >= accepted
        let stats = Arc::new(stats);
        let stats2 = stats.clone();
        let result = tokio::time::timeout(Duration::from_millis(500), async move {
            stats2.wait_for_accepted_items().await;
        })
        .await;
        assert!(
            result.is_ok(),
            "wait_for_accepted_items should return immediately when processed >= accepted"
        );

        // Test that wait_for_accepted_items blocks when processed < accepted
        let stats = IngestStats::new(256);
        stats.mark_accepted();
        let stats = Arc::new(stats);
        let stats2 = stats.clone();
        let result = tokio::time::timeout(Duration::from_millis(100), async move {
            stats2.wait_for_accepted_items().await;
        })
        .await;
        assert!(
            result.is_err(),
            "wait_for_accepted_items should block when processed < accepted"
        );
    }
}
