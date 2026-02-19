//! Servidor web Axum com HTMX e WebSocket para visualização do NER em tempo real

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use ner_core::{
    corpus::demo_texts,
    pipeline::{AlgorithmMode, NerPipeline, PipelineEvent},
    tokenizer::TokenizerMode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use tracing::info;

/// Estado compartilhado da aplicação
struct AppState {
    pipeline: NerPipeline,
}

// NerPipeline somente usa &self → é seguro compartilhar entre threads
unsafe impl Send for AppState {}
unsafe impl Sync for AppState {}

#[derive(Deserialize)]
struct AnalyzeRequest {
    text: String,
    #[serde(default)]
    mode: Option<AlgorithmMode>,
    #[serde(default)]
    tokenizer_mode: Option<TokenizerMode>,
}

/// Mensagem WebSocket recebida do cliente
#[derive(Deserialize)]
struct WsRequest {
    text: String,
    #[serde(default)]
    mode: Option<AlgorithmMode>,
    #[serde(default)]
    tokenizer_mode: Option<TokenizerMode>,
}

#[derive(Serialize)]
struct AnalyzeResponse {
    entities: Vec<ner_core::tagger::EntitySpan>,
    tagged_tokens: Vec<ner_core::tagger::TaggedToken>,
    processing_ms: u64,
    total_tokens: usize,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let pipeline = NerPipeline::new();
    let state = Arc::new(AppState { pipeline });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Caminho absoluto para a pasta docs/ (relativo ao workspace raiz)
    let docs_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("docs");

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/analyze", post(analyze_handler))
        .route("/ws", get(ws_handler))
        .route("/demo-texts", get(demo_texts_handler))
        .nest_service("/docs", ServeDir::new(docs_dir))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    info!("🚀 Servidor NER iniciado em http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}

/// Retorna a página principal HTML
async fn index_handler() -> impl IntoResponse {
    Html(include_str!("templates/index.html"))
}

/// Análise NER via HTTP POST (sem streaming)
async fn analyze_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnalyzeRequest>,
) -> impl IntoResponse {
    if req.text.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Texto vazio"})),
        )
            .into_response();
    }

    let mode = req.mode.unwrap_or_default();
    let tokenizer_mode = req.tokenizer_mode.unwrap_or(TokenizerMode::Standard);
    let (tagged, entities) = state.pipeline.analyze_with_mode(&req.text, mode, tokenizer_mode);
    let total_tokens = tagged.len();

    Json(AnalyzeResponse {
        processing_ms: 0,
        entities,
        tagged_tokens: tagged,
        total_tokens,
    })
    .into_response()
}

/// Retorna textos de demonstração
async fn demo_texts_handler() -> impl IntoResponse {
    let texts: Vec<serde_json::Value> = demo_texts()
        .iter()
        .map(|(domain, text)| {
            serde_json::json!({
                "domain": domain,
                "text": text
            })
        })
        .collect();
    Json(texts)
}

/// Upgrade HTTP → WebSocket
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_websocket(socket, state))
}

/// Lógica do WebSocket: recebe texto, executa pipeline e envia eventos em tempo real
async fn handle_websocket(mut socket: WebSocket, state: Arc<AppState>) {
    info!("WebSocket conectado");

    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Text(text) => {
                // Tenta parsear como JSON {text, mode, tokenizer_mode}; senão usa como texto puro
                let (text_str, mode, tokenizer_mode) = if let Ok(req) =
                    serde_json::from_str::<WsRequest>(&text)
                {
                    let m = req.mode.unwrap_or_default();
                    let t = req.tokenizer_mode.unwrap_or(TokenizerMode::Standard);
                    (req.text.trim().to_string(), m, t)
                } else {
                    (text.trim().to_string(), AlgorithmMode::Hybrid, TokenizerMode::Standard)
                };

                if text_str.is_empty() {
                    continue;
                }

                info!("Analisando via WebSocket [{:?} | {:?}]: {} chars", mode, tokenizer_mode, text_str.len());

                // Executa o pipeline em um tokio::task::spawn_blocking para não bloquear o runtime
                let (tx_std, rx_std) = std::sync::mpsc::channel::<PipelineEvent>();

                // Cria um Arc clone para o closure da thread
                let pipeline_arc = Arc::clone(&state);
                let text_for_thread = text_str.clone();

                // Roda pipeline em thread separada (é síncrono)
                let handle = tokio::task::spawn_blocking(move || {
                    pipeline_arc.pipeline.analyze_streaming(&text_for_thread, mode, tokenizer_mode, tx_std);
                });

                // Coleta todos os eventos da fila std::mpsc (após pipeline concluir)
                handle.await.ok();

                // Coleta todos os eventos numa Vec (isso descarta o rx_std, que não é Send)
                let events: Vec<PipelineEvent> = rx_std.try_iter().collect();

                loop {
                    // Send events with delay
                    // We need to iterate carefully since we can't await in a simple for loop over iterator
                    // Actually, `events` is a Vec, so we can iterate.
                    for event in &events {
                         if let Ok(json) = serde_json::to_string(event) {
                             if socket.send(Message::Text(json.into())).await.is_err() {
                                 return; // cliente desconectou
                             }
                             // Pequena pausa para animação visual (passo a passo)
                             tokio::time::sleep(tokio::time::Duration::from_millis(35)).await;
                         }
                    }
                    break;
                }
            }
            Message::Close(_) => {
                info!("WebSocket desconectado");
                return;
            }
            Message::Ping(payload) => {
                let _ = socket.send(Message::Pong(payload)).await;
            }
            _ => {}
        }
    }
}
