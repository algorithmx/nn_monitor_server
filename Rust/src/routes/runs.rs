use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde_json::json;

use super::AppState;

pub async fn get_runs(State(state): State<AppState>) -> impl IntoResponse {
    let runs = state.store.get_all_runs().await;
    axum::Json(runs)
}

pub async fn get_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> impl IntoResponse {
    match state.store.get_run(&run_id).await {
        Some(run_data) => axum::Json(run_data).into_response(),
        None => {
            let error = json!({
                "detail": {"error": "not_found", "message": format!("Run '{}' not found", run_id)}
            });
            (StatusCode::NOT_FOUND, axum::Json(error)).into_response()
        }
    }
}

pub async fn get_latest_step(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> impl IntoResponse {
    match state.store.get_latest_step(&run_id).await {
        Some(step) => axum::Json(step).into_response(),
        None => {
            let error = json!({
                "detail": {"error": "not_found", "message": format!("Run '{}' not found", run_id)}
            });
            (StatusCode::NOT_FOUND, axum::Json(error)).into_response()
        }
    }
}
