use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde_json::json;
use tokio::sync::mpsc::error::TrySendError;

use super::AppState;

pub async fn post_metrics(State(state): State<AppState>, body: Bytes) -> impl IntoResponse {
    let payload: crate::models::MetricsPayload = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(e) => {
            let (msg, error_type) = match e.classify() {
                serde_json::error::Category::Data => (e.to_string(), "value_error"),
                _ => (format!("Invalid JSON: {}", e), "json_invalid"),
            };
            let error_detail = json!({
                "detail": [{"loc": ["body"], "msg": msg, "type": error_type}]
            });
            return (StatusCode::UNPROCESSABLE_ENTITY, axum::Json(error_detail)).into_response();
        }
    };

    if let Err(e) = payload.validate() {
        let error_detail = json!({
            "detail": [{"loc": ["body"], "msg": e, "type": "value_error"}]
        });
        return (StatusCode::UNPROCESSABLE_ENTITY, axum::Json(error_detail)).into_response();
    }

    let run_id = payload.metadata.run_id.clone();
    match state
        .ingest_tx
        .try_send(crate::ingest::IngestItem { payload })
    {
        Ok(()) => state.ingest_stats.mark_accepted(),
        Err(TrySendError::Full(_)) => {
            state.ingest_stats.mark_dropped();
            let error_detail = json!({
                "detail": {"error": "ingest_queue_full", "message": "Ingest queue is full"}
            });
            return (StatusCode::SERVICE_UNAVAILABLE, axum::Json(error_detail)).into_response();
        }
        Err(TrySendError::Closed(_)) => {
            state.ingest_stats.mark_dropped();
            let error_detail = json!({
                "detail": {"error": "ingest_closed", "message": "Ingest worker is not running"}
            });
            return (StatusCode::SERVICE_UNAVAILABLE, axum::Json(error_detail)).into_response();
        }
    }

    let response = crate::models::MetricsAcceptedResponse {
        status: "accepted".to_string(),
        run_id,
    };
    (StatusCode::ACCEPTED, axum::Json(response)).into_response()
}
