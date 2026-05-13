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

#[derive(Deserialize, Debug, Clone)]
pub struct ServerConfig {
    #[serde(default = "default_max_runs")]
    pub max_runs: usize,
    #[serde(default = "default_max_steps_per_run")]
    pub max_steps_per_run: usize,
    #[serde(default = "default_max_request_size")]
    pub max_request_size: usize,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default = "default_cors_origins")]
    pub cors_origins: Vec<String>,
}

impl ServerConfig {
    pub fn load() -> Result<Self, envy::Error> {
        dotenvy::dotenv().ok();
        envy::prefixed("NN_MONITOR_").from_env::<ServerConfig>()
    }
}
