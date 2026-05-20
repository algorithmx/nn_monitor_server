use axum::extract::State;
use axum::response::IntoResponse;

use super::AppState;

pub async fn get_health(State(state): State<AppState>) -> impl IntoResponse {
    let queue_capacity = state.ingest_stats.capacity();
    let remaining_capacity = state.ingest_tx.capacity();
    let response = crate::models::HealthResponse {
        status: "healthy".to_string(),
        active_connections: state.ws_manager.active_count(),
        ingest_queue_depth: queue_capacity.saturating_sub(remaining_capacity),
        ingest_queue_capacity: queue_capacity,
        accepted_count: state.ingest_stats.accepted_count(),
        processed_count: state.ingest_stats.processed_count(),
        dropped_count: state.ingest_stats.dropped_count(),
    };
    axum::Json(response)
}
