use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Local, Utc};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

use crate::models::{now_iso, sanitize_filename, RunData, StepData};

// ==================== Error ====================

#[derive(Debug)]
pub enum PersistError {
    Io(std::io::Error),
    Parse(String),
}

impl std::fmt::Display for PersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistError::Io(e) => write!(f, "IO error: {}", e),
            PersistError::Parse(s) => write!(f, "Parse error: {}", s),
        }
    }
}

impl std::error::Error for PersistError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PersistError::Io(e) => Some(e),
            PersistError::Parse(_) => None,
        }
    }
}

impl From<std::io::Error> for PersistError {
    fn from(e: std::io::Error) -> Self {
        PersistError::Io(e)
    }
}

// ==================== RunMeta ====================

#[derive(Debug, Clone, serde::Serialize)]
pub struct RunMeta {
    pub run_id: String,
    pub created_at: String,
    pub last_update: String,
    pub step_count: u32,
    pub latest_step: Option<u64>,
}

// ==================== Internal State ====================

struct RunBuffer {
    steps: Vec<StepData>,
    last_activity: Instant,
}

struct PersistState {
    buffers: HashMap<String, RunBuffer>,
}

// ==================== JsonlStore ====================

pub struct JsonlStore {
    data_dir: PathBuf,
    flush_timeout: Duration,
    max_buffer_size: usize,
    state: Mutex<PersistState>,
}

// ==================== Helpers ====================

fn timestamp_to_iso(ts: f64) -> String {
    let secs = ts as i64;
    let nanos = ((ts - secs as f64) * 1_000_000_000.0) as u32;
    DateTime::from_timestamp(secs, nanos)
        .map(|dt: DateTime<Utc>| {
            dt.with_timezone(&Local)
                .format("%Y-%m-%dT%H:%M:%S%.6f")
                .to_string()
        })
        .unwrap_or_else(now_iso)
}

// ==================== Implementation ====================

impl JsonlStore {
    pub async fn new(data_dir: PathBuf, flush_timeout: Duration, max_buffer_size: usize) -> Self {
        tokio::fs::create_dir_all(&data_dir)
            .await
            .expect("Failed to create data directory");
        Self {
            data_dir,
            flush_timeout,
            max_buffer_size,
            state: Mutex::new(PersistState {
                buffers: HashMap::new(),
            }),
        }
    }

    pub async fn buffer_step(&self, run_id: &str, step: &StepData) -> Result<(), PersistError> {
        let mut state = self.state.lock().await;
        let entry = state
            .buffers
            .entry(run_id.to_string())
            .or_insert(RunBuffer {
                steps: Vec::new(),
                last_activity: Instant::now(),
            });
        entry.steps.push(step.clone());
        if entry.steps.len() > self.max_buffer_size {
            let excess = entry.steps.len() - self.max_buffer_size;
            entry.steps.drain(0..excess);
            tracing::warn!("Buffer for run {} exceeded max size, drained {} oldest steps", run_id, excess);
        }
        entry.last_activity = Instant::now();
        Ok(())
    }

    pub async fn flush_run(&self, run_id: &str) -> Result<(), PersistError> {
        let steps = {
            let mut state = self.state.lock().await;
            match state.buffers.get_mut(run_id) {
                Some(buffer) if !buffer.steps.is_empty() => std::mem::take(&mut buffer.steps),
                _ => return Ok(()),
            }
        };

        let filename = sanitize_filename(run_id);
        let path = self.data_dir.join(format!("{}.jsonl", filename));

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;

        for step in &steps {
            let line =
                serde_json::to_string(step).map_err(|e| PersistError::Parse(e.to_string()))?;
            file.write_all(line.as_bytes()).await?;
            file.write_all(b"\n").await?;
        }

        file.flush().await?;
        Ok(())
    }

    pub async fn scan_metadata(&self) -> Vec<RunMeta> {
        let mut results = Vec::new();
        let mut entries = match tokio::fs::read_dir(&self.data_dir).await {
            Ok(e) => e,
            Err(_) => return results,
        };

        while let Some(entry) = match entries.next_entry().await {
            Ok(e) => e,
            Err(_) => None,
        } {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }

            let run_id = match path.file_stem().and_then(|s| s.to_str()) {
                Some(f) => f.to_string(),
                None => continue,
            };

            let file = match tokio::fs::File::open(&path).await {
                Ok(f) => f,
                Err(e) => {
                    tracing::warn!("Failed to open {}: {}", path.display(), e);
                    continue;
                }
            };

            let reader = BufReader::new(file);
            let mut lines = reader.lines();

            let first_line = match lines.next_line().await {
                Ok(Some(line)) if !line.trim().is_empty() => line,
                _ => continue,
            };

            let first_step: StepData = match serde_json::from_str(first_line.trim()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse first line of {}: {}",
                        path.display(),
                        e
                    );
                    continue;
                }
            };

            let mut step_count: u32 = 1;
            let mut last_line = first_line;

            while let Ok(Some(line)) = lines.next_line().await {
                if !line.trim().is_empty() {
                    step_count += 1;
                    last_line = line;
                }
            }

            let last_step: StepData = match serde_json::from_str(last_line.trim()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse last line of {}: {}",
                        path.display(),
                        e
                    );
                    continue;
                }
            };

            results.push(RunMeta {
                run_id,
                created_at: timestamp_to_iso(first_step.timestamp),
                last_update: timestamp_to_iso(last_step.timestamp),
                step_count,
                latest_step: Some(last_step.step),
            });
        }

        results
    }

    pub async fn load_run(
        &self,
        run_id: &str,
        max_steps: usize,
    ) -> Result<Option<RunData>, PersistError> {
        let filename = sanitize_filename(run_id);
        let path = self.data_dir.join(format!("{}.jsonl", filename));

        if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
            return Ok(None);
        }

        let file = tokio::fs::File::open(&path).await?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let mut steps = Vec::new();

        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<StepData>(&line) {
                Ok(step) => steps.push(step),
                Err(e) => {
                    tracing::warn!("Skipping malformed line in {}: {}", path.display(), e);
                }
            }
        }

        if steps.is_empty() {
            return Ok(None);
        }

        steps.sort_by_key(|s| s.step);

        // Deduplicate by step number, keeping the last (most recent) occurrence
        let mut deduped: Vec<StepData> = Vec::new();
        for step in steps {
            if deduped.last().map_or(false, |s| s.step == step.step) {
                deduped.pop();
            }
            deduped.push(step);
        }
        steps = deduped;

        if steps.len() > max_steps {
            let excess = steps.len() - max_steps;
            steps.drain(0..excess);
        }

        // NOTE: created_at is derived from first step's timestamp, not wall-clock time.
        // This differs from in-memory runs which use server receive time (now_iso).
        // The step timestamp is arguably more meaningful (actual training time vs server time).
        let created_at = timestamp_to_iso(steps.first().unwrap().timestamp);
        let last_update = timestamp_to_iso(steps.last().unwrap().timestamp);

        Ok(Some(RunData {
            created_at,
            last_update,
            steps,
        }))
    }

    pub async fn flush_all(&self) -> Result<(), PersistError> {
        let run_ids: Vec<String> = {
            let state = self.state.lock().await;
            state.buffers.keys().cloned().collect()
        };

        for run_id in run_ids {
            self.flush_run(&run_id).await?;
        }

        Ok(())
    }

    pub fn start_flush_task(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let this = Arc::clone(self);
        let interval = this.flush_timeout / 2;

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;

                let run_ids_to_flush: Vec<String> = {
                    let state = this.state.lock().await;
                    let now = Instant::now();
                    state
                        .buffers
                        .iter()
                        .filter(|(_, buffer)| {
                            !buffer.steps.is_empty()
                                && now.duration_since(buffer.last_activity) >= this.flush_timeout
                        })
                        .map(|(run_id, _)| run_id.clone())
                        .collect()
                };

                for run_id in run_ids_to_flush {
                    if let Err(e) = this.flush_run(&run_id).await {
                        tracing::warn!("Failed to flush run {}: {}", run_id, e);
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{CrossLayerAnalysis, FiniteF64};
    use std::collections::HashMap as StdHashMap;

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("{}_{}", prefix, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn make_step(step: u64) -> StepData {
        StepData {
            step,
            timestamp: 1707589200.0 + step as f64,
            batch_size: 32,
            layers: vec![],
            cross_layer: CrossLayerAnalysis {
                feature_std_gradient: FiniteF64::new(0.0).unwrap(),
                gradient_norm_ratio: StdHashMap::new(),
            },
            layer_groups: None,
        }
    }

    #[tokio::test]
    async fn test_jsonl_write_and_read_roundtrip() {
        let dir = temp_dir("nn_monitor_test_roundtrip");
        let store = JsonlStore::new(dir.clone(), Duration::from_secs(60), 100).await;

        let steps = vec![make_step(1), make_step(2), make_step(3)];
        for s in &steps {
            store.buffer_step("test_run", s).await.unwrap();
        }
        store.flush_run("test_run").await.unwrap();

        let loaded = store.load_run("test_run", 100).await.unwrap().expect("should load");
        assert_eq!(loaded.steps.len(), 3, "should have 3 steps");
        for (i, expected) in steps.iter().enumerate() {
            assert_eq!(loaded.steps[i].step, expected.step, "step number mismatch at index {}", i);
            assert!(
                (loaded.steps[i].timestamp - expected.timestamp).abs() < 0.001,
                "timestamp mismatch at index {}",
                i
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_jsonl_scan_metadata() {
        let dir = temp_dir("nn_monitor_test_scan");
        let store = JsonlStore::new(dir.clone(), Duration::from_secs(60), 100).await;

        let steps = vec![make_step(10), make_step(20), make_step(30)];
        for s in &steps {
            store.buffer_step("meta_run", s).await.unwrap();
        }
        store.flush_run("meta_run").await.unwrap();

        let metas = store.scan_metadata().await;
        assert_eq!(metas.len(), 1, "should find 1 run");
        let meta = &metas[0];
        assert_eq!(meta.run_id, "meta_run");
        assert_eq!(meta.step_count, 3);
        assert_eq!(meta.latest_step, Some(30));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_jsonl_skip_malformed_lines() {
        let dir = temp_dir("nn_monitor_test_malformed");
        let store = JsonlStore::new(dir.clone(), Duration::from_secs(60), 100).await;

        let file_path = dir.join("corrupt_run.jsonl");
        let valid1 = serde_json::to_string(&make_step(1)).unwrap();
        let valid2 = serde_json::to_string(&make_step(2)).unwrap();
        let content = format!("{}\nnot json\n{}\n", valid1, valid2);
        std::fs::write(&file_path, content).unwrap();

        let loaded = store.load_run("corrupt_run", 100).await.unwrap().expect("should load");
        assert_eq!(loaded.steps.len(), 2, "should return 2 valid steps, skipping corrupt line");
        assert_eq!(loaded.steps[0].step, 1);
        assert_eq!(loaded.steps[1].step, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_flush_all_on_shutdown() {
        let dir = temp_dir("nn_monitor_test_flushall");
        let store = JsonlStore::new(dir.clone(), Duration::from_secs(60), 100).await;

        for run_num in 1..=3u64 {
            let run_id = format!("run_{}", run_num);
            store.buffer_step(&run_id, &make_step(run_num)).await.unwrap();
        }

        store.flush_all().await.unwrap();

        for run_num in 1..=3u64 {
            let run_id = format!("run_{}", run_num);
            let filename = sanitize_filename(&run_id);
            let path = dir.join(format!("{}.jsonl", filename));
            assert!(path.exists(), "{}.jsonl should exist", filename);

            let loaded = store.load_run(&run_id, 100).await.unwrap().expect("should load");
            assert_eq!(loaded.steps.len(), 1);
            assert_eq!(loaded.steps[0].step, run_num);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_empty_jsonl_file_handling() {
        let dir = temp_dir("nn_monitor_test_empty");
        let store = JsonlStore::new(dir.clone(), Duration::from_secs(60), 100).await;

        let empty_path = dir.join("empty_run.jsonl");
        std::fs::write(&empty_path, "").unwrap();

        let metas = store.scan_metadata().await;
        assert!(metas.is_empty(), "empty file should produce no metadata");

        let loaded = store.load_run("empty_run", 100).await.unwrap();
        assert!(loaded.is_none(), "empty file should load as None");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_run_id_sanitization_for_filename() {
        let dir = temp_dir("nn_monitor_test_sanitize");
        let store = JsonlStore::new(dir.clone(), Duration::from_secs(60), 100).await;

        let malicious_id = "../../etc/passwd";
        store.buffer_step(malicious_id, &make_step(1)).await.unwrap();
        store.flush_run(malicious_id).await.unwrap();

        let sanitized = sanitize_filename(malicious_id);
        assert!(!sanitized.contains(".."), "sanitized name should not contain ..");
        assert!(!sanitized.contains('/'), "sanitized name should not contain /");

        let expected_path = dir.join(format!("{}.jsonl", sanitized));
        assert!(expected_path.exists(), "file should exist with sanitized name");

        let loaded = store.load_run(malicious_id, 100).await.unwrap().expect("should load");
        assert_eq!(loaded.steps.len(), 1);
        assert_eq!(loaded.steps[0].step, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_max_steps_cap_on_lazy_load() {
        let dir = temp_dir("nn_monitor_test_maxsteps");
        let store = JsonlStore::new(dir.clone(), Duration::from_secs(60), 100).await;

        for i in 1..=20u64 {
            store.buffer_step("capped_run", &make_step(i)).await.unwrap();
        }
        store.flush_run("capped_run").await.unwrap();

        let loaded = store.load_run("capped_run", 10).await.unwrap().expect("should load");
        assert_eq!(loaded.steps.len(), 10, "should return only last 10 steps");
        assert_eq!(loaded.steps[0].step, 11, "first returned step should be 11");
        assert_eq!(loaded.steps[9].step, 20, "last returned step should be 20");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_buffer_cap_limits_size() {
        let dir = temp_dir("nn_monitor_test_bufcap");
        let store = JsonlStore::new(dir.clone(), Duration::from_secs(60), 5).await;

        for i in 1..=7u64 {
            store.buffer_step("cap_run", &make_step(i)).await.unwrap();
        }

        store.flush_run("cap_run").await.unwrap();

        let loaded = store
            .load_run("cap_run", 100)
            .await
            .unwrap()
            .expect("should load");
        assert_eq!(loaded.steps.len(), 5, "should cap at 5 buffered steps");
        assert_eq!(loaded.steps[0].step, 3, "oldest step (1,2) drained");
        assert_eq!(loaded.steps[4].step, 7, "newest step preserved");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
