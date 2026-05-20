use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;

use crate::routes::AppState;

/// WebSocket upgrade handler.
pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Main WebSocket connection handler.
///
/// Protocol:
/// 1. On connect: send `initial_runs` with all known runs
/// 2. Forward broadcast messages (new_metrics) to client in real time
/// 3. Handle client actions: subscribe_run (with lite mode), ping/pong
/// 4. On disconnect: decrement connection counter
async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    state.ws_manager.connect();

    let initial_msg = state.store.build_initial_runs_message().await;
    if sender
        .send(Message::Text(initial_msg.into()))
        .await
        .is_err()
    {
        state.ws_manager.disconnect();
        return;
    }

    let mut broadcast_rx = state.ws_manager.subscribe();

    loop {
        tokio::select! {
            result = broadcast_rx.recv() => {
                match result {
                    Ok(msg) => {
                        if sender.send(Message::Text(msg.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("WebSocket client lagged, skipped {} messages", n);
                        continue;
                    }
                    Err(_) => break,
                }
            }
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        handle_client_message(&text, &state, &mut sender).await;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }
        }
    }

    state.ws_manager.disconnect();
}

/// Handle a single text message from the client.
///
/// Supports:
/// - `subscribe_run`: fetch full or compact run history
/// - `ping`: respond with pong
/// - Invalid JSON: respond with error
/// - Unknown action: ignore gracefully
async fn handle_client_message(
    text: &str,
    state: &AppState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
) {
    let data: Result<Value, _> = serde_json::from_str(text);

    match data {
        Ok(data) => {
            let action = data.get("action").and_then(|v| v.as_str());

            match action {
                Some("subscribe_run") => {
                    let run_id = data.get("run_id").and_then(|v| v.as_str()).unwrap_or("");
                    let lite = data.get("lite").and_then(|v| v.as_bool()).unwrap_or(false);

                    match state.store.build_run_history_message(run_id, lite).await {
                        Some(msg) => {
                            let _ = sender.send(Message::Text(msg.into())).await;
                        }
                        None => {
                            let msg = crate::ws::build_error_message(&format!(
                                "Run '{}' not found",
                                run_id
                            ));
                            let _ = sender.send(Message::Text(msg.into())).await;
                        }
                    }
                }
                Some("ping") => {
                    let msg = crate::ws::build_pong_message();
                    let _ = sender.send(Message::Text(msg.into())).await;
                }
                _ => {}
            }
        }
        Err(_) => {
            let msg = crate::ws::build_error_message("Invalid JSON format");
            let _ = sender.send(Message::Text(msg.into())).await;
        }
    }
}
