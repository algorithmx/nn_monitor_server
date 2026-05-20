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
