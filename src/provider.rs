use std::{
    collections::HashMap,
    pin::Pin,
    sync::Mutex,
    time::{Duration, Instant},
};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use reqwest::{header::HeaderMap, StatusCode};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::{
    anthropic::{count_tokens, response_content},
    config::{CapabilityProfile, CredentialConfig, LimitsConfig, ProviderConfig, ProviderKind},
    domain::{
        AnthropicResponse, ImageData, InputBlock, MessageRole, NormalizedRequest, ProviderEvent,
        StopReason, ToolChoice, Usage,
    },
    error::{ErrorKind, ProxyError, Result},
};

pub type EventStream = Pin<Box<dyn Stream<Item = Result<ProviderEvent>> + Send>>;

pub struct Provider {
    pub id: String,
    kind: ProviderKind,
    endpoint: String,
    credential: CredentialConfig,
    pub capabilities: CapabilityProfile,
    client: reqwest::Client,
    circuit: Mutex<CircuitState>,
    failure_threshold: u32,
    cooldown: Duration,
}

#[derive(Debug)]
struct CircuitState {
    failures: u32,
    open_until: Option<Instant>,
}

impl Provider {
    pub fn new(config: &ProviderConfig, limits: &LimitsConfig) -> Result<Self> {
        let mut headers = HeaderMap::new();
        for (name, value) in &config.headers {
            let name = reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|_| {
                ProxyError::new(ErrorKind::Internal, format!("invalid header name {name}"))
            })?;
            let value = reqwest::header::HeaderValue::from_str(&value.resolve()?)
                .map_err(|_| ProxyError::new(ErrorKind::Internal, "invalid custom header value"))?;
            headers.insert(name, value);
        }
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(limits.connect_timeout_ms))
            .default_headers(headers.clone())
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(30))
            .build()
            .map_err(|error| {
                ProxyError::new(
                    ErrorKind::Internal,
                    format!("cannot build provider client: {error}"),
                )
            })?;

        Ok(Self {
            id: config.id.clone(),
            kind: config.kind,
            endpoint: config.endpoint.trim_end_matches('/').to_owned(),
            credential: config.credential.clone(),
            capabilities: config.capability_profile.clone(),
            client,
            circuit: Mutex::new(CircuitState {
                failures: 0,
                open_until: None,
            }),
            failure_threshold: limits.circuit_failure_threshold,
            cooldown: Duration::from_secs(limits.circuit_cooldown_seconds),
        })
    }

    pub fn available(&self) -> bool {
        let mut state = self.circuit.lock().expect("circuit poisoned");
        if let Some(until) = state.open_until {
            if Instant::now() < until {
                return false;
            }
            state.open_until = None;
        }
        true
    }

    pub fn record_success(&self) {
        let mut state = self.circuit.lock().expect("circuit poisoned");
        state.failures = 0;
        state.open_until = None;
    }

    pub fn record_failure(&self, error: &ProxyError) {
        if !matches!(
            error.kind,
            ErrorKind::Upstream | ErrorKind::Timeout | ErrorKind::Overloaded
        ) {
            return;
        }
        let mut state = self.circuit.lock().expect("circuit poisoned");
        state.failures += 1;
        if state.failures >= self.failure_threshold {
            state.open_until = Some(Instant::now() + self.cooldown);
        }
    }

    pub fn adapt_request(&self, request: &NormalizedRequest) -> Result<NormalizedRequest> {
        let mut adapted = request.clone();

        if adapted.max_tokens > self.capabilities.max_output_tokens {
            if self.capabilities.allow_max_tokens_clamping {
                metrics::counter!(
                    "proxy_request_adapted_total",
                    "provider" => self.id.clone(),
                    "hint" => "max_tokens_clamped"
                )
                .increment(1);
                adapted.max_tokens = self.capabilities.max_output_tokens;
            } else {
                return Err(ProxyError::invalid(format!(
                    "max_tokens exceeds target limit {}",
                    self.capabilities.max_output_tokens
                )));
            }
        }

        if adapted.temperature.is_some() && !self.capabilities.supports_temperature {
            if self.capabilities.allow_temperature_fallback {
                metrics::counter!(
                    "proxy_request_adapted_total",
                    "provider" => self.id.clone(),
                    "hint" => "temperature_dropped"
                )
                .increment(1);
                adapted.temperature = None;
            } else {
                return Err(ProxyError::invalid("target does not support temperature"));
            }
        }
        if adapted.top_p.is_some() && !self.capabilities.supports_top_p {
            if self.capabilities.allow_top_p_fallback {
                metrics::counter!(
                    "proxy_request_adapted_total",
                    "provider" => self.id.clone(),
                    "hint" => "top_p_dropped"
                )
                .increment(1);
                adapted.top_p = None;
            } else {
                return Err(ProxyError::invalid("target does not support top_p"));
            }
        }
        if !adapted.stop_sequences.is_empty() && !self.capabilities.supports_stop {
            if self.capabilities.allow_stop_fallback {
                metrics::counter!(
                    "proxy_request_adapted_total",
                    "provider" => self.id.clone(),
                    "hint" => "stop_sequences_cleared"
                )
                .increment(1);
                adapted.stop_sequences.clear();
            } else {
                return Err(ProxyError::invalid(
                    "target does not support stop sequences",
                ));
            }
        }
        if !adapted.tools.is_empty() && !self.capabilities.tools {
            return Err(ProxyError::invalid("target does not support tools"));
        }
        if matches!(
            adapted.tool_choice,
            Some(
                ToolChoice::Auto {
                    disable_parallel: false
                } | ToolChoice::Any {
                    disable_parallel: false
                } | ToolChoice::Tool {
                    disable_parallel: false,
                    ..
                }
            )
        ) && !self.capabilities.parallel_tools
        {
            if self.capabilities.allow_parallel_tool_fallback {
                metrics::counter!(
                    "proxy_request_adapted_total",
                    "provider" => self.id.clone(),
                    "hint" => "parallel_tools_disabled"
                )
                .increment(1);
                adapted.tool_choice = adapted.tool_choice.take().map(|choice| match choice {
                    ToolChoice::Auto { .. } => ToolChoice::Auto {
                        disable_parallel: true,
                    },
                    ToolChoice::Any { .. } => ToolChoice::Any {
                        disable_parallel: true,
                    },
                    ToolChoice::Tool { name, .. } => ToolChoice::Tool {
                        name,
                        disable_parallel: true,
                    },
                    ToolChoice::None => ToolChoice::None,
                });
            } else {
                return Err(ProxyError::invalid(
                    "target does not support parallel tools",
                ));
            }
        }
        for message in &mut adapted.messages {
            for block in &mut message.blocks {
                match block {
                    InputBlock::Image { .. } if !self.capabilities.vision => {
                        return Err(ProxyError::invalid("target does not support image input"))
                    }
                    InputBlock::ToolResult { content, .. } if !content.is_string() => {
                        if self.capabilities.allow_structured_tool_results_to_string {
                            let serialized = serde_json::to_string(content).map_err(|_| {
                                ProxyError::invalid(
                                    "could not serialize structured tool result to string",
                                )
                            })?;
                            *content = serde_json::Value::String(serialized);
                            metrics::counter!(
                                "proxy_request_adapted_total",
                                "provider" => self.id.clone(),
                                "hint" => "tool_result_json_stringified"
                            )
                            .increment(1);
                        } else {
                            return Err(ProxyError::invalid(
                                "structured or multimodal tool results are not supported by Chat Completions targets",
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(adapted)
    }

    pub fn validate_request(&self, request: &NormalizedRequest) -> Result<()> {
        self.adapt_request(request).map(|_| ())
    }

    pub fn count_tokens(&self, request: &NormalizedRequest) -> Result<u64> {
        let adapted = self.adapt_request(request)?;
        count_tokens(&adapted, &self.capabilities.tokenizer)
    }

    pub async fn execute(
        &self,
        request: &NormalizedRequest,
        model: &str,
    ) -> Result<AnthropicResponse> {
        let adapted = self.adapt_request(request)?;
        let body = self.build_request(&adapted, model, false)?;
        let response = self.send(body).await?;
        let parsed: OpenAiResponse = response.json().await.map_err(|error| {
            ProxyError::new(
                ErrorKind::Upstream,
                format!("provider returned invalid JSON: {error}"),
            )
        })?;
        convert_response(parsed, &request.original_model)
    }

    pub async fn stream(&self, request: &NormalizedRequest, model: &str) -> Result<EventStream> {
        let adapted = self.adapt_request(request)?;
        let body = self.build_request(&adapted, model, true)?;
        let response = self.send(body).await?;
        let byte_stream = response.bytes_stream();
        Ok(Box::pin(parse_sse(byte_stream)))
    }

    pub async fn probe(&self) -> Result<()> {
        let url = format!("{}/models", self.endpoint);
        let builder = self.apply_auth(self.client.get(url))?;
        let response = builder.send().await.map_err(classify_transport)?;
        if response.status().is_success() {
            return Ok(());
        }
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        Err(classify_status(status, &body, None))
    }

    async fn send(&self, body: Value) -> Result<reqwest::Response> {
        let started = Instant::now();
        let url = chat_completions_url(&self.endpoint);
        let builder = self.client.post(url).json(&body);
        let builder = self.apply_auth(builder)?;
        let response = builder.send().await.map_err(|error| {
            metrics::counter!("proxy_upstream_requests_total", "provider" => self.id.clone(), "outcome" => "transport_error").increment(1);
            classify_transport(error)
        })?;
        metrics::histogram!("proxy_upstream_duration_seconds", "provider" => self.id.clone())
            .record(started.elapsed().as_secs_f64());
        if response.status().is_success() {
            metrics::counter!("proxy_upstream_requests_total", "provider" => self.id.clone(), "outcome" => "success").increment(1);
            return Ok(response);
        }
        let status = response.status();
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse().ok());
        let body = response.text().await.unwrap_or_default();
        metrics::counter!("proxy_upstream_requests_total", "provider" => self.id.clone(), "outcome" => status.as_u16().to_string()).increment(1);
        Err(classify_status(status, &body, retry_after))
    }

    fn apply_auth(&self, mut builder: reqwest::RequestBuilder) -> Result<reqwest::RequestBuilder> {
        let secret = self.credential.secret().resolve()?;
        builder = match (&self.kind, &self.credential) {
            (_, CredentialConfig::Bearer { .. }) => builder.bearer_auth(secret),
            (ProviderKind::AzureChat, CredentialConfig::ApiKey { .. }) => {
                builder.header("api-key", secret)
            }
            (ProviderKind::AzureChat, CredentialConfig::AzureEntra { .. }) => {
                builder.bearer_auth(secret)
            }
            (_, CredentialConfig::ApiKey { .. }) => builder.bearer_auth(secret),
            (ProviderKind::OpenaiChat, CredentialConfig::AzureEntra { .. }) => {
                builder.bearer_auth(secret)
            }
        };
        Ok(builder)
    }

    fn build_request(
        &self,
        request: &NormalizedRequest,
        model: &str,
        stream: bool,
    ) -> Result<Value> {
        let mut body = Map::new();
        body.insert("model".into(), json!(model));
        body.insert("messages".into(), Value::Array(convert_messages(request)?));
        let token_field = if self.capabilities.use_max_completion_tokens {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        body.insert(token_field.into(), json!(request.max_tokens));
        if let Some(value) = request.temperature {
            body.insert("temperature".into(), json!(value));
        }
        if let Some(value) = request.top_p {
            body.insert("top_p".into(), json!(value));
        }
        if !request.stop_sequences.is_empty() {
            body.insert("stop".into(), json!(request.stop_sequences));
        }
        if !request.tools.is_empty() {
            body.insert("tools".into(), Value::Array(request.tools.iter().map(|tool| json!({
                "type":"function", "function":{"name":tool.name,"description":tool.description,"parameters":tool.input_schema}
            })).collect()));
        }
        if let Some(choice) = &request.tool_choice {
            let (value, parallel) = match choice {
                ToolChoice::Auto { disable_parallel } => (json!("auto"), Some(!disable_parallel)),
                ToolChoice::Any { disable_parallel } => {
                    (json!("required"), Some(!disable_parallel))
                }
                ToolChoice::None => (json!("none"), None),
                ToolChoice::Tool {
                    name,
                    disable_parallel,
                } => (
                    json!({"type":"function","function":{"name":name}}),
                    Some(!disable_parallel),
                ),
            };
            body.insert("tool_choice".into(), value);
            if let Some(parallel) = parallel {
                body.insert("parallel_tool_calls".into(), json!(parallel));
            }
        }
        body.insert("stream".into(), json!(stream));
        if stream {
            body.insert("stream_options".into(), json!({"include_usage":true}));
        }
        Ok(Value::Object(body))
    }
}

fn chat_completions_url(endpoint: &str) -> String {
    match endpoint.split_once('?') {
        Some((base, query)) => format!("{base}/chat/completions?{query}"),
        None => format!("{endpoint}/chat/completions"),
    }
}

fn convert_messages(request: &NormalizedRequest) -> Result<Vec<Value>> {
    let mut messages = Vec::new();
    for message in &request.messages {
        match message.role {
            MessageRole::System | MessageRole::User => {
                let role = if message.role == MessageRole::System {
                    "system"
                } else {
                    "user"
                };
                let mut content = Vec::new();
                let mut tool_results = Vec::new();
                for block in &message.blocks {
                    match block {
                        InputBlock::Text(text) => content.push(json!({"type":"text","text":text})),
                        InputBlock::Image { media_type, data } => {
                            let url = match data {
                                ImageData::Base64(data) => {
                                    format!("data:{media_type};base64,{data}")
                                }
                                ImageData::Url(url) => url.clone(),
                            };
                            content.push(json!({"type":"image_url","image_url":{"url":url}}));
                        }
                        InputBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            let value = content.as_str().ok_or_else(|| {
                                ProxyError::invalid("tool result content must be a string")
                            })?;
                            tool_results.push(json!({"role":"tool","tool_call_id":tool_use_id,"content": if *is_error { format!("Error: {value}") } else { value.to_owned() }}));
                        }
                        InputBlock::ToolUse { .. } => {
                            return Err(ProxyError::invalid(
                                "tool_use blocks are only valid in assistant messages",
                            ))
                        }
                    }
                }
                messages.extend(tool_results);
                if !content.is_empty() {
                    let content =
                        if content.len() == 1 && content[0].get("type") == Some(&json!("text")) {
                            content[0]["text"].clone()
                        } else {
                            Value::Array(content)
                        };
                    messages.push(json!({"role":role,"content":content}));
                }
            }
            MessageRole::Assistant => {
                let mut text = String::new();
                let mut calls = Vec::new();
                for block in &message.blocks {
                    match block {
                        InputBlock::Text(part) => text.push_str(part),
                        InputBlock::ToolUse { id, name, input } => calls.push(json!({"id":id,"type":"function","function":{"name":name,"arguments":serde_json::to_string(input).map_err(|error| ProxyError::invalid(error.to_string()))?}})),
                        _ => return Err(ProxyError::invalid("assistant messages may contain only text and tool_use blocks")),
                    }
                }
                let mut value = json!({"role":"assistant","content": if text.is_empty() { Value::Null } else { json!(text) }});
                if !calls.is_empty() {
                    value["tool_calls"] = Value::Array(calls);
                }
                messages.push(value);
            }
        }
    }
    Ok(messages)
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    id: String,
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAiToolCall>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCall {
    id: String,
    function: OpenAiFunction,
}
#[derive(Debug, Deserialize)]
struct OpenAiFunction {
    name: String,
    arguments: String,
}
#[derive(Debug, Clone, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

fn convert_response(response: OpenAiResponse, model: &str) -> Result<AnthropicResponse> {
    let choice = response.choices.into_iter().next().ok_or_else(|| {
        ProxyError::new(ErrorKind::Upstream, "provider response contains no choices")
    })?;
    let mut text = choice.message.content;
    if text.is_none() {
        text = choice.message.refusal;
    }
    let tools = choice
        .message
        .tool_calls
        .into_iter()
        .map(|call| {
            let input = serde_json::from_str(&call.function.arguments).map_err(|_| {
                ProxyError::new(
                    ErrorKind::Upstream,
                    "provider returned invalid tool-call JSON",
                )
            })?;
            Ok((call.id, call.function.name, input))
        })
        .collect::<Result<Vec<_>>>()?;
    let reason = map_stop_reason(choice.finish_reason.as_deref())?;
    let usage = response
        .usage
        .map(|usage| Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        })
        .unwrap_or(Usage {
            input_tokens: 0,
            output_tokens: 0,
        });
    Ok(AnthropicResponse {
        id: response.id,
        kind: "message",
        role: "assistant",
        model: model.to_owned(),
        content: response_content(text, tools),
        stop_reason: Some(reason.as_str().to_owned()),
        stop_sequence: None,
        usage,
    })
}

fn map_stop_reason(reason: Option<&str>) -> Result<StopReason> {
    match reason {
        Some("stop") | None => Ok(StopReason::EndTurn),
        Some("length") => Ok(StopReason::MaxTokens),
        Some("tool_calls") | Some("function_call") => Ok(StopReason::ToolUse),
        Some("content_filter") => Ok(StopReason::Refusal),
        Some(other) => Err(ProxyError::new(
            ErrorKind::Upstream,
            format!("unknown provider finish reason {other}"),
        )),
    }
}

fn parse_sse<S>(input: S) -> impl Stream<Item = Result<ProviderEvent>> + Send
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
{
    async_stream::try_stream! {
        futures::pin_mut!(input);
        let mut buffer = Vec::new();
        let mut tools: HashMap<usize, StreamedTool> = HashMap::new();
        let mut final_usage: Option<Usage> = None;
        let mut final_reason: Option<StopReason> = None;
        while let Some(chunk) = input.next().await {
            let chunk = chunk.map_err(classify_transport)?;
            buffer.extend_from_slice(&chunk);
            while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
                let mut line = buffer.drain(..=position).collect::<Vec<_>>();
                line.pop();
                if line.last() == Some(&b'\r') { line.pop(); }
                let line = std::str::from_utf8(&line).map_err(|_| ProxyError::new(ErrorKind::Upstream, "provider stream is not UTF-8"))?;
                let Some(data) = line.strip_prefix("data:").map(str::trim) else { continue };
                if data.is_empty() { continue; }
                if data == "[DONE]" {
                    validate_streamed_tools(&tools)?;
                    let reason = final_reason.ok_or_else(|| ProxyError::new(ErrorKind::Upstream, "provider stream ended without a finish reason"))?;
                    yield ProviderEvent::Finished { reason, usage: final_usage.clone() };
                    return;
                }
                let chunk: Value = serde_json::from_str(data).map_err(|error| ProxyError::new(ErrorKind::Upstream, format!("invalid provider SSE JSON: {error}")))?;
                if let Some(usage) = chunk.get("usage").filter(|value| !value.is_null()) {
                    if let (Some(input), Some(output)) = (usage.get("prompt_tokens").and_then(Value::as_u64), usage.get("completion_tokens").and_then(Value::as_u64)) {
                        let usage = Usage { input_tokens: input, output_tokens: output };
                        final_usage = Some(usage.clone());
                        yield ProviderEvent::Usage(usage);
                    }
                }
                for choice in chunk.get("choices").and_then(Value::as_array).into_iter().flatten() {
                    if let Some(content) = choice.pointer("/delta/content").and_then(Value::as_str) {
                        if !content.is_empty() { yield ProviderEvent::Text(content.to_owned()); }
                    }
                    if let Some(calls) = choice.pointer("/delta/tool_calls").and_then(Value::as_array) {
                        for call in calls {
                            let index = call.get("index").and_then(Value::as_u64).ok_or_else(|| ProxyError::new(ErrorKind::Upstream, "streamed tool call lacks index"))? as usize;
                            let tool = tools.entry(index).or_default();
                            if let Some(id) = call.get("id").and_then(Value::as_str) { tool.id.push_str(id); }
                            if let Some(name) = call.pointer("/function/name").and_then(Value::as_str) { tool.name.push_str(name); }
                            if !tool.started && !tool.id.is_empty() && !tool.name.is_empty() {
                                tool.started = true;
                                yield ProviderEvent::ToolStart { index, id: tool.id.clone(), name: tool.name.clone() };
                                if !tool.arguments.is_empty() {
                                    yield ProviderEvent::ToolDelta { index, arguments: tool.arguments.clone() };
                                }
                            }
                            if let Some(arguments) = call.pointer("/function/arguments").and_then(Value::as_str) {
                                tool.arguments.push_str(arguments);
                                if tool.started {
                                    yield ProviderEvent::ToolDelta { index, arguments: arguments.to_owned() };
                                }
                            }
                        }
                    }
                    if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                        final_reason = Some(map_stop_reason(Some(reason))?);
                    }
                }
            }
        }
        validate_streamed_tools(&tools)?;
        let reason = final_reason.ok_or_else(|| ProxyError::new(ErrorKind::Upstream, "provider stream terminated before completion"))?;
        yield ProviderEvent::Finished { reason, usage: final_usage };
    }
}

#[derive(Default)]
struct StreamedTool {
    id: String,
    name: String,
    arguments: String,
    started: bool,
}

fn validate_streamed_tools(tools: &HashMap<usize, StreamedTool>) -> Result<()> {
    for tool in tools.values() {
        if !tool.started {
            return Err(ProxyError::new(
                ErrorKind::Upstream,
                "provider returned an incomplete streamed tool call",
            ));
        }
        serde_json::from_str::<Value>(&tool.arguments).map_err(|_| {
            ProxyError::new(
                ErrorKind::Upstream,
                "provider returned invalid streamed tool-call JSON",
            )
        })?;
    }
    Ok(())
}

fn classify_transport(error: reqwest::Error) -> ProxyError {
    if error.is_connect() {
        ProxyError::new(
            ErrorKind::Upstream,
            format!("upstream connection failure: {error}"),
        )
        .with_retryable()
    } else if error.is_timeout() {
        ProxyError::new(ErrorKind::Timeout, "upstream request timed out")
    } else {
        ProxyError::new(
            ErrorKind::Upstream,
            format!("upstream transport failure: {error}"),
        )
    }
}

fn classify_status(status: StatusCode, body: &str, retry_after: Option<u64>) -> ProxyError {
    let kind = match status.as_u16() {
        400 | 404 | 409 | 422 => ErrorKind::InvalidRequest,
        401 => ErrorKind::Upstream,
        403 => ErrorKind::Upstream,
        408 => ErrorKind::Timeout,
        429 => ErrorKind::RateLimit,
        500 | 502 | 503 | 504 | 529 => ErrorKind::Overloaded,
        _ => ErrorKind::Upstream,
    };
    let provider_message = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| format!("HTTP {status}"));
    let mut error = ProxyError::new(
        kind,
        format!(
            "upstream rejected request: {}",
            truncate(&provider_message, 512)
        ),
    );
    if matches!(status.as_u16(), 408 | 429 | 500 | 502 | 503 | 504 | 529) {
        error.is_retryable = true;
    }
    error.retry_after_seconds = retry_after;
    error
}

fn truncate(value: &str, max: usize) -> String {
    value.chars().take(max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{anthropic::normalize, domain::AnthropicRequest};
    use futures::TryStreamExt;

    #[test]
    fn converts_modern_tools() {
        let request: AnthropicRequest = serde_json::from_value(json!({
            "model":"claude", "max_tokens":100, "messages":[
                {"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"shell","input":{"cmd":"pwd"}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"/tmp"}]}
            ], "tools":[{"name":"shell","input_schema":{"type":"object"}}]
        })).unwrap();
        let messages = convert_messages(&normalize(request, false).unwrap()).unwrap();
        assert_eq!(messages[0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[1]["role"], "tool");
    }

    #[tokio::test]
    async fn parses_sse_across_utf8_and_line_boundaries() {
        let payload = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"héllo\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        )
        .as_bytes();
        let chunks = payload
            .chunks(3)
            .map(|chunk| Ok::<Bytes, reqwest::Error>(Bytes::copy_from_slice(chunk)))
            .collect::<Vec<_>>();
        let events = parse_sse(futures::stream::iter(chunks))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert!(matches!(&events[0], ProviderEvent::Text(text) if text == "héllo"));
        assert!(matches!(
            events.last(),
            Some(ProviderEvent::Finished {
                reason: StopReason::EndTurn,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn rejects_truncated_stream() {
        let chunks = vec![Ok::<Bytes, reqwest::Error>(Bytes::from_static(
            b"data: {\"choices\":[]}\n\n",
        ))];
        let error = parse_sse(futures::stream::iter(chunks))
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(error.to_string().contains("before completion"));
    }
}
