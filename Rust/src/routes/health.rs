use axum::extract::State;
use axum::response::IntoResponse;

use super::AppState;

pub async fn get_health(State(state): State<AppState>) -> impl IntoResponse {
    let response = crate::models::HealthResponse {
        status: "healthy".to_string(),
        active_connections: state.ws_manager.active_count(),
    };
    axum::Json(response)
}
