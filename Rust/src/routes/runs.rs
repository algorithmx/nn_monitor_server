use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use serde_json::json;

use super::AppState;

fn json_body(body: String) -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/json")], body)
}

pub async fn get_runs(State(state): State<AppState>) -> impl IntoResponse {
    json_body(state.store.get_all_runs_json().await)
}

pub async fn get_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> impl IntoResponse {
    match state.store.get_run_json(&run_id).await {
        Some(run_data) => json_body(run_data).into_response(),
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
    match state.store.get_latest_step_json(&run_id).await {
        Some(step) => json_body(step).into_response(),
        None => {
            let error = json!({
                "detail": {"error": "not_found", "message": format!("Run '{}' not found", run_id)}
            });
            (StatusCode::NOT_FOUND, axum::Json(error)).into_response()
        }
    }
}
