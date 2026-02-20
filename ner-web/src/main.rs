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
use askama::Template;
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
///
/// O Axum exige que o estado seja `Clone` e `Send` + `Sync` para ser compartilhado entre threads.
/// Envolvemos o `pipeline` em um `Arc` (Atomic Reference Counting) implicitamente ao colocar
/// no `AppState` que será envolto em `Arc` na main.
///
/// O `NerPipeline` é imutável após a criação (só leitura do modelo), então é thread-safe.
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

#[derive(Deserialize)]
struct TokenizeRequest {
    text: String,
    #[serde(default)]
    tokenizer_mode: Option<TokenizerMode>,
}

#[derive(Deserialize)]
struct SotaRequest {
    text: String,
    classes: String, // vírgula separadas
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
        .route("/tokenizer", get(tokenizer_page_handler))
        .route("/ned", get(ned_page_handler))
        .route("/nel", get(nel_page_handler))
        .route("/sota", get(sota_page_handler))
        .route("/gliner2", get(gliner2_page_handler))
        .route("/htmx/tokenize", post(htmx_tokenize_handler))
        .route("/htmx/ned", post(htmx_ned_handler))
        .route("/htmx/nel", post(htmx_nel_handler))
        .route("/htmx/sota", post(htmx_sota_handler))
        .nest_service("/docs", ServeDir::new(docs_dir))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    info!("🚀 Servidor NER iniciado em http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}

#[derive(Template)]
#[template(path = "ner.html")]
struct NerTemplate {}

#[derive(Template)]
#[template(path = "tokenizer.html")]
struct TokenizerTemplate {}

#[derive(Template)]
#[template(path = "ned.html")]
struct NedTemplate {}

#[derive(Template)]
#[template(path = "nel.html")]
struct NelTemplate {}

#[derive(Template)]
#[template(path = "sota.html")]
struct SotaTemplate {}

#[derive(Template)]
#[template(path = "gliner2.html")]
struct Gliner2Template {}

/// Retorna a página principal HTML
async fn index_handler() -> impl IntoResponse {
    Html(NerTemplate {}.render().unwrap())
}

async fn tokenizer_page_handler() -> impl IntoResponse {
    Html(TokenizerTemplate {}.render().unwrap())
}

async fn ned_page_handler() -> impl IntoResponse {
    Html(NedTemplate {}.render().unwrap())
}

async fn nel_page_handler() -> impl IntoResponse {
    Html(NelTemplate {}.render().unwrap())
}

async fn sota_page_handler() -> impl IntoResponse {
    Html(SotaTemplate {}.render().unwrap())
}

async fn gliner2_page_handler() -> impl IntoResponse {
    Html(Gliner2Template {}.render().unwrap())
}

#[derive(Template)]
#[template(path = "components/token_list.html")]
struct TokenListTemplate {
    tokens: Vec<ner_core::tokenizer::Token>,
}

use axum::Form;

async fn htmx_tokenize_handler(
    Form(req): Form<TokenizeRequest>,
) -> impl IntoResponse {
    let mode = req.tokenizer_mode.unwrap_or(TokenizerMode::Standard);
    let tokens = ner_core::tokenizer::tokenize_with_mode(&req.text, mode);
    Html(TokenListTemplate { tokens }.render().unwrap())
}

#[derive(Template)]
#[template(path = "components/ned_results.html")]
struct NedResultsTemplate {
    results: Vec<ner_core::ned::DisambiguatedEntity>,
}

async fn htmx_ned_handler(
    State(state): State<Arc<AppState>>,
    Form(req): Form<TokenizeRequest>,
) -> impl IntoResponse {
    let mode = AlgorithmMode::Hybrid; // NED usaremos o melhor modelo por default
    let tokenizer_mode = req.tokenizer_mode.unwrap_or(TokenizerMode::Standard);
    
    // 1. Roda a pipeline normal para extrair entidades e tokens
    let (tagged_tokens, entities) = state.pipeline.analyze_with_mode(&req.text, mode, tokenizer_mode);
    let tokens: Vec<_> = tagged_tokens.into_iter().map(|t| t.token).collect();
    
    // 2. Roda a desambiguação com base no contexto
    let results = ner_core::ned::disambiguate(&tokens, &entities);
    
    Html(NedResultsTemplate { results }.render().unwrap())
}

#[derive(Template)]
#[template(path = "components/nel_results.html")]
struct NelResultsTemplate {
    results: Vec<ner_core::nel::LinkedEntity>,
}

async fn htmx_nel_handler(
    State(state): State<Arc<AppState>>,
    Form(req): Form<TokenizeRequest>,
) -> impl IntoResponse {
    let mode = AlgorithmMode::Hybrid; // NEL usa o pipeline mais robusto
    let tokenizer_mode = req.tokenizer_mode.unwrap_or(TokenizerMode::Standard);
    
    // 1. NER
    let (tagged_tokens, entities) = state.pipeline.analyze_with_mode(&req.text, mode, tokenizer_mode);
    let tokens: Vec<_> = tagged_tokens.into_iter().map(|t| t.token).collect();
    
    // 2. Desambiguação (NED)
    let disambiguated = ner_core::ned::disambiguate(&tokens, &entities);
    
    // 3. Entity Linking em KB mokada
    let kb = ner_core::nel::KnowledgeBase::new();
    let results = kb.link(&disambiguated);
    
    Html(NelResultsTemplate { results }.render().unwrap())
}

#[derive(Template)]
#[template(path = "components/sota_results.html")]
struct SotaResultsTemplate {
    results: Vec<ner_core::sota_2024::SotaPrediction>,
}

async fn htmx_sota_handler(
    Form(req): Form<SotaRequest>,
) -> impl IntoResponse {
    let tokens = ner_core::tokenizer::tokenize_with_mode(&req.text, TokenizerMode::Standard);
    
    // Converte a string de classes (ex: "PER, LOC") para vetor ["PER", "LOC"]
    let user_classes: Vec<String> = req.classes
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
        
    // Chama a rede neural "simulada" q faz Span-based NER
    // Threshold fixo em 0.5 para simulação
    let results = ner_core::sota_2024::simulate_gliner(&tokens, &user_classes, 0.5, 4);

    Html(SotaResultsTemplate { results }.render().unwrap())
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
///
/// Rota que inicia o handshake WebSocket. Se bem sucedido, transfere o controle
/// para `handle_websocket`.
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_websocket(socket, state))
}

/// Lógica do WebSocket: recebe texto, executa pipeline e envia eventos em tempo real.
///
/// # Protocolo
/// 1. Cliente envia JSON: `{"text": "...", "mode": "hybrid", "tokenizer_mode": "standard"}`
/// 2. Servidor responde com fluxo de eventos JSON:
///    - `TokenizationDone`
///    - `FeaturesComputed`...
///    - `Done`
///
/// A análise roda em uma thread dedicada (`spawn_blocking`) para não travar o loop de eventos assíncrono do Tokio,
/// já que o pipeline é CPU-bound e síncrono.
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

                // Aguarda o término do processamento
                if handle.await.is_err() {
                    // Se a thread panicar
                    let _ = socket.send(Message::Text(serde_json::json!({
                        "type": "Error",
                        "data": { "message": "Erro interno no pipeline" }
                    }).to_string().into())).await;
                    continue;
                }

                // Coleta todos os eventos numa Vec (o rx_std não é Async, então consumimos tudo de uma vez após o término)
                // OBS: Numa implementação real de streaming, o canal deveria ser consumido enquanto a thread produz.
                // Mas como o mpsc std bloqueia, e queremos async await no socket send, essa abordagem de bufferizar
                // é um compromisso simples para este demo.
                let events: Vec<PipelineEvent> = rx_std.try_iter().collect();

                for event in events {
                     if let Ok(json) = serde_json::to_string(&event) {
                         if socket.send(Message::Text(json.into())).await.is_err() {
                             return; // cliente desconectou
                         }
                         // Pequena pausa para animação visual (passo a passo) no front-end ficar fluida
                         tokio::time::sleep(tokio::time::Duration::from_millis(35)).await;
                     }
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
