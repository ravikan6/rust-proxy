use std::{convert::Infallible, sync::Arc, time::Duration};

use arc_swap::ArcSwap;
use axum::{
    body::Body,
    extract::{rejection::JsonRejection, DefaultBodyLimit, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as AxumRouter,
};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::sync::Semaphore;
use tower_http::{catch_panic::CatchPanicLayer, trace::TraceLayer};
use uuid::Uuid;

use crate::{
    anthropic::{normalize, sse_event, StreamEncoder},
    auth::AuthRegistry,
    config::AppConfig,
    domain::AnthropicRequest,
    error::{ErrorKind, ProxyError, Result},
    routing::Router,
};

pub struct Runtime {
    pub config: AppConfig,
    pub auth: AuthRegistry,
    pub router: Router,
    global_concurrency: Arc<Semaphore>,
}

pub type SharedState = Arc<ArcSwap<Runtime>>;

impl Runtime {
    pub fn new(config: AppConfig) -> Result<Self> {
        config.validate()?;
        let auth = AuthRegistry::new(&config.clients)?;
        let router = Router::new(&config)?;
        let global_concurrency = Arc::new(Semaphore::new(config.limits.global_concurrency));
        Ok(Self {
            config,
            auth,
            router,
            global_concurrency,
        })
    }
}

pub fn build_app(state: SharedState) -> AxumRouter {
    let max_body = state.load().config.server.max_body_bytes;
    AxumRouter::new()
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/health/live", get(live))
        .route("/health/ready", get(ready))
        .route("/health", get(ready))
        .layer(DefaultBodyLimit::max(max_body))
        .layer(CatchPanicLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn messages(
    State(shared): State<SharedState>,
    headers: HeaderMap,
    payload: std::result::Result<Json<AnthropicRequest>, JsonRejection>,
) -> Response {
    let started = std::time::Instant::now();
    let request_id = request_id(&headers);
    let response = match messages_inner(shared, headers, payload, request_id.clone()).await {
        Ok(mut response) => {
            set_request_id(&mut response, &request_id);
            response
        }
        Err(error) => error.with_request_id(request_id).into_response(),
    };
    metrics::counter!("proxy_requests_total", "endpoint" => "messages", "status" => response.status().as_u16().to_string()).increment(1);
    metrics::histogram!("proxy_request_duration_seconds", "endpoint" => "messages")
        .record(started.elapsed().as_secs_f64());
    response
}

async fn messages_inner(
    shared: SharedState,
    headers: HeaderMap,
    payload: std::result::Result<Json<AnthropicRequest>, JsonRejection>,
    request_id: String,
) -> Result<Response> {
    let runtime = shared.load_full();
    validate_version(&runtime, &headers)?;
    let identity = runtime.auth.authenticate(&headers)?;
    let client_permit = identity.acquire()?;
    let global_permit = runtime
        .global_concurrency
        .clone()
        .try_acquire_owned()
        .map_err(|_| ProxyError::new(ErrorKind::Overloaded, "proxy concurrency limit exceeded"))?;
    let Json(request) = payload.map_err(|error| {
        ProxyError::invalid(format!("invalid JSON request: {}", error.body_text()))
    })?;
    let request = normalize(request, runtime.config.server.loose_input_validation)?;
    let routed = runtime.router.resolve(&request)?;
    if !identity.allows_route(&routed.route.id) {
        return Err(ProxyError::new(
            ErrorKind::Permission,
            "client is not allowed to use this route",
        ));
    }

    tracing::info!(
        request_id,
        client_id = identity.id,
        route = routed.route.id,
        stream = request.stream,
        "request accepted"
    );
    if !request.stream {
        let response = runtime.router.execute(routed).await?;
        return Ok(Json(response).into_response());
    }

    let (input_tokens, mut upstream) = runtime.router.stream(routed).await?;
    let message_id = format!("msg_{}", Uuid::new_v4().simple());
    let model = request.original_model.clone();
    let idle_timeout = Duration::from_secs(runtime.config.limits.stream_idle_timeout_seconds);
    let body_stream = async_stream::stream! {
        let _client_permit = client_permit;
        let _global_permit = global_permit;
        let _stream_metric = ActiveStreamMetric::new();
        let (mut encoder, start) = StreamEncoder::new(message_id, model, input_tokens);
        yield Ok::<Bytes, Infallible>(Bytes::from(start));
        loop {
            match tokio::time::timeout(idle_timeout, upstream.next()).await {
                Ok(Some(Ok(event))) => {
                    for encoded in encoder.encode(event) { yield Ok(Bytes::from(encoded)); }
                }
                Ok(Some(Err(error))) => {
                    yield Ok(Bytes::from(stream_error(&error, &request_id)));
                    break;
                }
                Ok(None) => break,
                Err(_) => {
                    let error = ProxyError::new(ErrorKind::Timeout, "upstream stream idle timeout");
                    yield Ok(Bytes::from(stream_error(&error, &request_id)));
                    break;
                }
            }
        }
    };
    let mut response = Response::new(Body::from_stream(body_stream));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    Ok(response)
}

struct ActiveStreamMetric;

impl ActiveStreamMetric {
    fn new() -> Self {
        metrics::gauge!("proxy_active_streams").increment(1.0);
        Self
    }
}

impl Drop for ActiveStreamMetric {
    fn drop(&mut self) {
        metrics::gauge!("proxy_active_streams").decrement(1.0);
    }
}

async fn count_tokens(
    State(shared): State<SharedState>,
    headers: HeaderMap,
    payload: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    let request_id = request_id(&headers);
    let outcome = (|| -> Result<Response> {
        let runtime = shared.load_full();
        validate_version(&runtime, &headers)?;
        let identity = runtime.auth.authenticate(&headers)?;
        let _permit = identity.acquire()?;
        let Json(mut value) = payload.map_err(|error| {
            ProxyError::invalid(format!("invalid JSON request: {}", error.body_text()))
        })?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| ProxyError::invalid("request body must be an object"))?;
        object.insert("max_tokens".into(), json!(1));
        object.insert("stream".into(), json!(false));
        let request: AnthropicRequest = serde_json::from_value(value).map_err(|error| {
            ProxyError::invalid(format!("invalid token count request: {error}"))
        })?;
        let request = normalize(request, runtime.config.server.loose_input_validation)?;
        let routed = runtime.router.resolve(&request)?;
        if !identity.allows_route(&routed.route.id) {
            return Err(ProxyError::new(
                ErrorKind::Permission,
                "client is not allowed to use this route",
            ));
        }
        let tokens = runtime.router.count_tokens(routed)?;
        Ok(Json(json!({"input_tokens":tokens})).into_response())
    })();
    match outcome {
        Ok(mut response) => {
            set_request_id(&mut response, &request_id);
            response
        }
        Err(error) => error.with_request_id(request_id).into_response(),
    }
}

async fn live() -> impl IntoResponse {
    Json(json!({"status":"ok"}))
}

async fn ready(State(shared): State<SharedState>) -> impl IntoResponse {
    let runtime = shared.load();
    (
        StatusCode::OK,
        Json(json!({
            "status":"ready",
            "routes": runtime.config.routes.len(),
            "providers": runtime.config.providers.len()
        })),
    )
}

fn validate_version(runtime: &Runtime, headers: &HeaderMap) -> Result<()> {
    if !runtime.config.server.require_anthropic_version {
        return Ok(());
    }
    match headers
        .get("anthropic-version")
        .and_then(|value| value.to_str().ok())
    {
        Some("2023-06-01") => Ok(()),
        Some(version) => Err(ProxyError::invalid(format!(
            "unsupported anthropic-version {version}"
        ))),
        None => Err(ProxyError::invalid("anthropic-version header is required")),
    }
}

fn request_id(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| {
            value.len() <= 128
                && value
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || "-_".contains(character))
        })
        .map(str::to_owned)
        .unwrap_or_else(|| format!("req_{}", Uuid::new_v4().simple()))
}

fn set_request_id(response: &mut Response, request_id: &str) {
    if let Ok(value) = HeaderValue::from_str(request_id) {
        response.headers_mut().insert("request-id", value.clone());
        response.headers_mut().insert("x-request-id", value);
    }
}

fn stream_error(error: &ProxyError, request_id: &str) -> String {
    sse_event(
        "error",
        json!({"type":"error","error":{"type":error.anthropic_type(),"message":error.message},"request_id":request_id}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malicious_request_ids() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-request-id",
            "bad value\n"
                .parse()
                .unwrap_or_else(|_| HeaderValue::from_static("bad value")),
        );
        assert!(request_id(&headers).starts_with("req_"));
    }
}
