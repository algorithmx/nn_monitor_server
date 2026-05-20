pub mod health;
pub mod metrics;
pub mod runs;
pub mod ws_route;

use std::sync::Arc;

/// Shared application state injected into all route handlers via Axum's State extractor.
#[derive(Clone)]
pub struct AppState {
    pub store: Arc<crate::store::MetricsStore>,
    pub ws_manager: Arc<crate::ws::WsManager>,
    pub config: crate::config::ServerConfig,
}
