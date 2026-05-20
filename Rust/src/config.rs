use serde::Deserialize;

fn default_max_runs() -> usize {
    10
}

fn default_max_steps_per_run() -> usize {
    1000
}

fn default_max_request_size() -> usize {
    2_000_000
}

fn default_ingest_queue_size() -> usize {
    4096
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8000
}

fn default_log_level() -> String {
    "warning".to_string()
}

fn default_cors_origins() -> Vec<String> {
    vec!["*".to_string()]
}

fn default_data_dir() -> String {
    "./data".to_string()
}

fn default_flush_timeout_secs() -> u64 {
    300
}

#[derive(Deserialize, Debug, Clone)]
pub struct ServerConfig {
    #[serde(default = "default_max_runs")]
    pub max_runs: usize,
    #[serde(default = "default_max_steps_per_run")]
    pub max_steps_per_run: usize,
    #[serde(default = "default_max_request_size")]
    pub max_request_size: usize,
    #[serde(default = "default_ingest_queue_size")]
    pub ingest_queue_size: usize,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default = "default_cors_origins")]
    pub cors_origins: Vec<String>,
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    #[serde(default = "default_flush_timeout_secs")]
    pub flush_timeout_secs: u64,
}

impl ServerConfig {
    pub fn load() -> Result<Self, envy::Error> {
        dotenvy::dotenv().ok();
        envy::prefixed("NN_MONITOR_").from_env::<ServerConfig>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_max_runs() {
        assert_eq!(default_max_runs(), 10);
    }

    #[test]
    fn test_default_max_steps_per_run() {
        assert_eq!(default_max_steps_per_run(), 1000);
    }

    #[test]
    fn test_default_max_request_size() {
        assert_eq!(default_max_request_size(), 2_000_000);
    }

    #[test]
    fn test_default_ingest_queue_size() {
        assert_eq!(default_ingest_queue_size(), 4096);
    }

    #[test]
    fn test_default_host() {
        assert_eq!(default_host(), "0.0.0.0");
    }

    #[test]
    fn test_default_port() {
        assert_eq!(default_port(), 8000);
    }

    #[test]
    fn test_default_log_level() {
        assert_eq!(default_log_level(), "warning");
    }

    #[test]
    fn test_default_cors_origins() {
        assert_eq!(default_cors_origins(), vec!["*"]);
    }

    #[test]
    fn test_default_data_dir() {
        assert_eq!(default_data_dir(), "./data");
    }

    #[test]
    fn test_default_flush_timeout_secs() {
        assert_eq!(default_flush_timeout_secs(), 300);
    }

    #[test]
    fn test_config_deserialize_with_defaults() {
        let config: ServerConfig =
            serde_json::from_value(serde_json::json!({})).expect("empty JSON should deserialize");
        assert_eq!(config.max_runs, 10);
        assert_eq!(config.max_steps_per_run, 1000);
        assert_eq!(config.max_request_size, 2_000_000);
        assert_eq!(config.ingest_queue_size, 4096);
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8000);
        assert_eq!(config.log_level, "warning");
        assert_eq!(config.cors_origins, vec!["*"]);
        assert_eq!(config.data_dir, "./data");
        assert_eq!(config.flush_timeout_secs, 300);
    }

    #[test]
    fn test_config_deserialize_partial() {
        let config: ServerConfig = serde_json::from_value(serde_json::json!({ "max_runs": 5 }))
            .expect("partial JSON should deserialize");
        assert_eq!(config.max_runs, 5);
        assert_eq!(config.max_steps_per_run, 1000);
        assert_eq!(config.max_request_size, 2_000_000);
        assert_eq!(config.ingest_queue_size, 4096);
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8000);
        assert_eq!(config.log_level, "warning");
        assert_eq!(config.cors_origins, vec!["*"]);
        assert_eq!(config.data_dir, "./data");
        assert_eq!(config.flush_timeout_secs, 300);
    }

    #[test]
    fn test_config_load_does_not_panic_without_env() {
        // dotenvy::dotenv().ok() silently ignores missing .env,
        // and envy reads from the current process environment,
        // so load() should succeed with defaults regardless.
        let config = ServerConfig::load().expect("load without env vars should succeed");
        assert_eq!(config.max_runs, 10);
        assert_eq!(config.max_steps_per_run, 1000);
        assert_eq!(config.max_request_size, 2_000_000);
        assert_eq!(config.ingest_queue_size, 4096);
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8000);
        assert_eq!(config.log_level, "warning");
        assert_eq!(config.cors_origins, vec!["*"]);
        assert_eq!(config.data_dir, "./data");
        assert_eq!(config.flush_timeout_secs, 300);
    }
}
