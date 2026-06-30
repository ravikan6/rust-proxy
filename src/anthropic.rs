use base64::Engine;
use serde_json::{json, Value};

use crate::{
    domain::{
        AnthropicRequest, ImageData, InputBlock, MessageRole, NormalizedMessage, NormalizedRequest,
        OutputBlock, ProviderEvent, ToolChoice,
    },
    error::{ProxyError, Result},
};

pub fn normalize(request: AnthropicRequest, loose_validation: bool) -> Result<NormalizedRequest> {
    let strict = !loose_validation;
    if request.model.trim().is_empty() {
        return Err(ProxyError::invalid("model must not be empty"));
    }
    if request.messages.is_empty() {
        return Err(ProxyError::invalid("messages must not be empty"));
    }
    if request.messages.len() > 10_000 {
        return Err(ProxyError::invalid("messages exceeds the 10000 item limit"));
    }
    if request.tools.len() > 128 {
        return Err(ProxyError::invalid("tools exceeds the 128 item limit"));
    }
    if request
        .stop_sequences
        .as_ref()
        .is_some_and(|items| items.len() > 4 || items.iter().any(String::is_empty))
    {
        return Err(ProxyError::invalid(
            "stop_sequences must contain at most four non-empty strings",
        ));
    }
    if request.max_tokens == 0 {
        return Err(ProxyError::invalid(
            "max_tokens must be greater than zero for Chat Completions targets",
        ));
    }
    if strict && !request.extensions.keys().all(|key| key == "cache_control") {
        return Err(ProxyError::invalid(format!(
            "unsupported request fields: {}",
            request
                .extensions
                .keys()
                .filter(|key| key.as_str() != "cache_control")
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if request.extensions.contains_key("cache_control") {
        metrics::counter!("proxy_translation_hints_discarded_total", "hint" => "cache_control")
            .increment(1);
    }
    // if request.thinking.is_some() {
    //     return Err(ProxyError::invalid(
    //         "extended thinking cannot be represented by Chat Completions targets",
    //     ));
    // }
    if request.top_k.is_some() {
        return Err(ProxyError::invalid(
            "top_k is not supported by configured Chat Completions targets",
        ));
    }
    if !(0.0..=1.0).contains(&request.temperature.unwrap_or(0.0)) {
        return Err(ProxyError::invalid("temperature must be between 0 and 1"));
    }
    if !(0.0..=1.0).contains(&request.top_p.unwrap_or(1.0)) {
        return Err(ProxyError::invalid("top_p must be between 0 and 1"));
    }

    let mut messages = Vec::new();
    if let Some(system) = request.system.as_ref() {
        let blocks = parse_system(system, strict)?;
        if !blocks.is_empty() {
            messages.push(NormalizedMessage {
                role: MessageRole::System,
                blocks,
            });
        }
    }

    for (index, message) in request.messages.iter().enumerate() {
        let role = match message.role.as_str() {
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            "system" => MessageRole::System,
            _ => {
                return Err(ProxyError::invalid(format!(
                    "messages[{index}].role must be user, assistant, or system"
                )))
            }
        };
        let blocks = parse_content(&message.content, index, strict)?;
        messages.push(NormalizedMessage { role, blocks });
    }
    validate_tool_history(&messages)?;

    let mut names = std::collections::HashSet::new();
    for tool in &request.tools {
        if tool.name.is_empty()
            || tool.name.len() > 64
            || !tool.name.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '_' | '-')
            })
            || !names.insert(&tool.name)
        {
            return Err(ProxyError::invalid(
                "tool names must be unique and contain 1-64 letters, digits, underscores, or hyphens",
            ));
        }
        if strict && !tool.extensions.keys().all(|key| key == "cache_control") {
            return Err(ProxyError::invalid(format!(
                "tool {} contains unsupported fields",
                tool.name
            )));
        }
        if !tool.input_schema.is_object() {
            return Err(ProxyError::invalid(format!(
                "tool {} input_schema must be an object",
                tool.name
            )));
        }
    }

    let tool_choice = parse_tool_choice(request.tool_choice.as_ref(), strict)?;
    if let Some(ToolChoice::Tool { name, .. }) = &tool_choice {
        if !names.contains(name) {
            return Err(ProxyError::invalid(format!(
                "tool_choice references undefined tool {name}"
            )));
        }
    }
    if request.tools.is_empty()
        && matches!(
            tool_choice,
            Some(ToolChoice::Any { .. } | ToolChoice::Tool { .. })
        )
    {
        return Err(ProxyError::invalid(
            "tool_choice requires at least one tool definition",
        ));
    }

    Ok(NormalizedRequest {
        original_model: request.model,
        max_tokens: request.max_tokens,
        messages,
        temperature: request.temperature,
        top_p: request.top_p,
        stop_sequences: request.stop_sequences.unwrap_or_default(),
        tools: request.tools,
        tool_choice,
        stream: request.stream,
    })
}

fn parse_system(value: &Value, strict: bool) -> Result<Vec<InputBlock>> {
    match value {
        Value::String(text) => Ok(vec![InputBlock::Text(text.clone())]),
        Value::Array(blocks) => blocks
            .iter()
            .enumerate()
            .map(|(index, block)| {
                if block.get("type").and_then(Value::as_str) != Some("text") {
                    return Err(ProxyError::invalid(format!(
                        "system[{index}] must be a text block"
                    )));
                }
                let text = required_str(block, "text", &format!("system[{index}]"))?;
                ensure_only(
                    block,
                    &["type", "text", "cache_control"],
                    &format!("system[{index}]"),
                    strict,
                )?;
                Ok(InputBlock::Text(text.to_owned()))
            })
            .collect(),
        _ => Err(ProxyError::invalid(
            "system must be a string or an array of text blocks",
        )),
    }
}

fn parse_content(value: &Value, message_index: usize, strict: bool) -> Result<Vec<InputBlock>> {
    if let Some(text) = value.as_str() {
        return Ok(vec![InputBlock::Text(text.to_owned())]);
    }
    let blocks = value.as_array().ok_or_else(|| {
        ProxyError::invalid(format!(
            "messages[{message_index}].content must be a string or array"
        ))
    })?;
    if blocks.is_empty() {
        return Err(ProxyError::invalid(format!(
            "messages[{message_index}].content must not be empty"
        )));
    }
    let mut output = Vec::with_capacity(blocks.len());
    for (block_index, block) in blocks.iter().enumerate() {
        let path = format!("messages[{message_index}].content[{block_index}]");
        let kind = required_str(block, "type", &path)?;
        let parsed = match kind {
            "text" => {
                ensure_only(
                    block,
                    &["type", "text", "cache_control", "citations"],
                    &path,
                    strict,
                )?;
                if block.get("citations").is_some_and(|value| !value.is_null()) {
                    return Err(ProxyError::invalid(format!(
                        "{path}.citations is unsupported"
                    )));
                }
                InputBlock::Text(required_str(block, "text", &path)?.to_owned())
            }
            "image" => parse_image(block, &path, strict)?,
            "tool_use" => {
                ensure_only(
                    block,
                    &["type", "id", "name", "input", "cache_control"],
                    &path,
                    strict,
                )?;
                InputBlock::ToolUse {
                    id: required_str(block, "id", &path)?.to_owned(),
                    name: required_str(block, "name", &path)?.to_owned(),
                    input: block
                        .get("input")
                        .filter(|value| value.is_object())
                        .cloned()
                        .ok_or_else(|| {
                            ProxyError::invalid(format!("{path}.input must be an object"))
                        })?,
                }
            }
            "tool_result" => {
                ensure_only(
                    block,
                    &[
                        "type",
                        "tool_use_id",
                        "content",
                        "is_error",
                        "cache_control",
                    ],
                    &path,
                    strict,
                )?;
                InputBlock::ToolResult {
                    tool_use_id: required_str(block, "tool_use_id", &path)?.to_owned(),
                    content: block
                        .get("content")
                        .cloned()
                        .unwrap_or(Value::String(String::new())),
                    is_error: block
                        .get("is_error")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                }
            }
            other => {
                return Err(ProxyError::invalid(format!(
                    "{path} has unsupported content block type {other}"
                )))
            }
        };
        output.push(parsed);
    }
    Ok(output)
}

fn parse_image(block: &Value, path: &str, strict: bool) -> Result<InputBlock> {
    ensure_only(block, &["type", "source", "cache_control"], path, strict)?;
    let source = block
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| ProxyError::invalid(format!("{path}.source must be an object")))?;
    let source_type = source
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| ProxyError::invalid(format!("{path}.source.type is required")))?;
    match source_type {
        "base64" => {
            if let Some(key) = source
                .keys()
                .find(|key| !["type", "media_type", "data"].contains(&key.as_str()))
            {
                return Err(ProxyError::invalid(format!(
                    "{path}.source.{key} is unsupported"
                )));
            }
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ProxyError::invalid(format!("{path}.source.media_type is required"))
                })?;
            if !matches!(
                media_type,
                "image/jpeg" | "image/png" | "image/gif" | "image/webp"
            ) {
                return Err(ProxyError::invalid(format!(
                    "{path}.source.media_type is unsupported"
                )));
            }
            let data = source
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| ProxyError::invalid(format!("{path}.source.data is required")))?;
            base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|_| {
                    ProxyError::invalid(format!("{path}.source.data is not valid base64"))
                })?;
            Ok(InputBlock::Image {
                media_type: media_type.to_owned(),
                data: ImageData::Base64(data.to_owned()),
            })
        }
        "url" => {
            if let Some(key) = source
                .keys()
                .find(|key| !["type", "url"].contains(&key.as_str()))
            {
                return Err(ProxyError::invalid(format!(
                    "{path}.source.{key} is unsupported"
                )));
            }
            let url = source
                .get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| ProxyError::invalid(format!("{path}.source.url is required")))?;
            let parsed = reqwest::Url::parse(url)
                .map_err(|_| ProxyError::invalid(format!("{path}.source.url is invalid")))?;
            if !matches!(parsed.scheme(), "http" | "https") {
                return Err(ProxyError::invalid(format!(
                    "{path}.source.url must use HTTP or HTTPS"
                )));
            }
            Ok(InputBlock::Image {
                media_type: "application/octet-stream".into(),
                data: ImageData::Url(url.to_owned()),
            })
        }
        other => Err(ProxyError::invalid(format!(
            "{path}.source type {other} is unsupported"
        ))),
    }
}

fn validate_tool_history(messages: &[NormalizedMessage]) -> Result<()> {
    let mut calls = std::collections::HashSet::new();
    let mut results = std::collections::HashSet::new();
    for message in messages {
        for block in &message.blocks {
            match block {
                InputBlock::ToolUse { id, .. }
                    if message.role != MessageRole::Assistant
                        || id.is_empty()
                        || !calls.insert(id.clone()) =>
                {
                    return Err(ProxyError::invalid(
                        "tool_use blocks require unique non-empty IDs in assistant messages",
                    ));
                }
                InputBlock::ToolResult { tool_use_id, .. }
                    if message.role != MessageRole::User
                        || !calls.contains(tool_use_id)
                        || !results.insert(tool_use_id.clone()) =>
                {
                    return Err(ProxyError::invalid(
                        "tool_result must reference one prior unresolved assistant tool_use",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn parse_tool_choice(value: Option<&Value>, strict: bool) -> Result<Option<ToolChoice>> {
    let Some(value) = value else { return Ok(None) };
    let kind = required_str(value, "type", "tool_choice")?;
    let disable_parallel = value
        .get("disable_parallel_tool_use")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let choice = match kind {
        "auto" => {
            ensure_only(value, &["type", "disable_parallel_tool_use"], "tool_choice", strict)?;
            ToolChoice::Auto { disable_parallel }
        }
        "any" => {
            ensure_only(value, &["type", "disable_parallel_tool_use"], "tool_choice", strict)?;
            ToolChoice::Any { disable_parallel }
        }
        "none" => {
            ensure_only(value, &["type"], "tool_choice", strict)?;
            ToolChoice::None
        }
        "tool" => {
            ensure_only(
                value,
                &["type", "name", "disable_parallel_tool_use"],
                "tool_choice",
                strict,
            )?;
            ToolChoice::Tool {
                name: required_str(value, "name", "tool_choice")?.to_owned(),
                disable_parallel,
            }
        }
        _ => {
            return Err(ProxyError::invalid(format!(
                "unsupported tool_choice type {kind}"
            )))
        }
    };
    Ok(Some(choice))
}

fn required_str<'a>(value: &'a Value, key: &str, path: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ProxyError::invalid(format!("{path}.{key} must be a string")))
}

fn ensure_only(value: &Value, allowed: &[&str], path: &str, strict: bool) -> Result<()> {
    if !strict {
        return Ok(());
    }
    let object = value
        .as_object()
        .ok_or_else(|| ProxyError::invalid(format!("{path} must be an object")))?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(ProxyError::invalid(format!("{path}.{key} is unsupported")));
    }
    Ok(())
}

pub fn sse_event(event: &str, data: Value) -> String {
    format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(&data).expect("JSON event serialization cannot fail")
    )
}

pub struct StreamEncoder {
    next_block: usize,
    text_block: Option<usize>,
    tools: std::collections::BTreeMap<usize, BufferedTool>,
    input_tokens: u64,
    output_tokens: u64,
}

struct BufferedTool {
    id: String,
    name: String,
    arguments: String,
}

impl StreamEncoder {
    pub fn new(message_id: String, model: String, input_tokens: u64) -> (Self, String) {
        let start = sse_event(
            "message_start",
            json!({
                "type": "message_start",
                "message": {"id": message_id, "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": null, "stop_sequence": null,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
            }),
        );
        (
            Self {
                next_block: 0,
                text_block: None,
                tools: Default::default(),
                input_tokens,
                output_tokens: 0,
            },
            start,
        )
    }

    pub fn encode(&mut self, event: ProviderEvent) -> Vec<String> {
        match event {
            ProviderEvent::Text(text) => {
                let index = if let Some(index) = self.text_block {
                    index
                } else {
                    let index = self.next_block;
                    self.next_block += 1;
                    self.text_block = Some(index);
                    return vec![
                        sse_event(
                            "content_block_start",
                            json!({"type":"content_block_start","index":index,"content_block":{"type":"text","text":""}}),
                        ),
                        sse_event(
                            "content_block_delta",
                            json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":text}}),
                        ),
                    ];
                };
                vec![sse_event(
                    "content_block_delta",
                    json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":text}}),
                )]
            }
            ProviderEvent::ToolStart {
                index: provider_index,
                id,
                name,
            } => {
                self.tools.insert(
                    provider_index,
                    BufferedTool {
                        id,
                        name,
                        arguments: String::new(),
                    },
                );
                vec![]
            }
            ProviderEvent::ToolDelta {
                index: provider_index,
                arguments,
            } => {
                if let Some(tool) = self.tools.get_mut(&provider_index) {
                    tool.arguments.push_str(&arguments);
                }
                vec![]
            }
            ProviderEvent::Usage(usage) => {
                self.input_tokens = usage.input_tokens;
                self.output_tokens = usage.output_tokens;
                vec![]
            }
            ProviderEvent::Finished { reason, usage } => {
                if let Some(usage) = usage {
                    self.input_tokens = usage.input_tokens;
                    self.output_tokens = usage.output_tokens;
                }
                let mut events = Vec::new();
                if let Some(index) = self.text_block {
                    events.push(sse_event(
                        "content_block_stop",
                        json!({"type":"content_block_stop","index":index}),
                    ));
                }
                if self.text_block.is_none() && self.tools.is_empty() {
                    let index = self.next_block;
                    self.next_block += 1;
                    events.push(sse_event("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"text","text":""}})));
                    events.push(sse_event(
                        "content_block_stop",
                        json!({"type":"content_block_stop","index":index}),
                    ));
                }
                for tool in self.tools.values() {
                    let index = self.next_block;
                    self.next_block += 1;
                    events.push(sse_event("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"tool_use","id":tool.id,"name":tool.name,"input":{}}})));
                    events.push(sse_event("content_block_delta", json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":tool.arguments}})));
                    events.push(sse_event(
                        "content_block_stop",
                        json!({"type":"content_block_stop","index":index}),
                    ));
                }
                events.push(sse_event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":reason.as_str(),"stop_sequence":null},"usage":{"output_tokens":self.output_tokens}})));
                events.push(sse_event("message_stop", json!({"type":"message_stop"})));
                events
            }
        }
    }
}

pub fn count_tokens(request: &NormalizedRequest, tokenizer: &str) -> Result<u64> {
    let bpe = match tokenizer {
        "o200k_base" => tiktoken_rs::o200k_base(),
        "cl100k_base" => tiktoken_rs::cl100k_base(),
        other => {
            return Err(ProxyError::invalid(format!(
                "unsupported tokenizer profile {other}"
            )))
        }
    }
    .map_err(|error| ProxyError::invalid(format!("failed to initialize tokenizer: {error}")))?;
    let mut tokens = 3_u64;
    for message in &request.messages {
        tokens += 4;
        for block in &message.blocks {
            tokens += match block {
                InputBlock::Text(text) => bpe.encode_with_special_tokens(text).len() as u64,
                InputBlock::Image { .. } => 85,
                InputBlock::ToolUse { name, input, .. } => {
                    bpe.encode_with_special_tokens(&format!("{name}{input}"))
                        .len() as u64
                        + 8
                }
                InputBlock::ToolResult { content, .. } => {
                    bpe.encode_with_special_tokens(&content.to_string()).len() as u64 + 4
                }
            };
        }
    }
    for tool in &request.tools {
        tokens += bpe
            .encode_with_special_tokens(&format!(
                "{}{}{}",
                tool.name,
                tool.description.as_deref().unwrap_or(""),
                tool.input_schema
            ))
            .len() as u64
            + 8;
    }
    Ok(tokens.max(1))
}

pub fn response_content(
    text: Option<String>,
    tools: Vec<(String, String, Value)>,
) -> Vec<OutputBlock> {
    let mut output = Vec::new();
    if let Some(text) = text.filter(|text| !text.is_empty()) {
        output.push(OutputBlock::Text { text });
    }
    output.extend(
        tools
            .into_iter()
            .map(|(id, name, input)| OutputBlock::ToolUse { id, name, input }),
    );
    if output.is_empty() {
        output.push(OutputBlock::Text {
            text: String::new(),
        });
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::AnthropicRequest;

    #[test]
    fn normalizes_tools_and_images() {
        let request: AnthropicRequest = serde_json::from_value(json!({
            "model":"claude-sonnet-4", "max_tokens":100, "messages":[{"role":"user","content":[
                {"type":"text","text":"look"}, {"type":"image","source":{"type":"base64","media_type":"image/png","data":"YWJj"}}
            ]}], "tools":[{"name":"lookup","input_schema":{"type":"object"}}]
        })).unwrap();
        let normalized = normalize(request, false).unwrap();
        assert_eq!(normalized.tools.len(), 1);
        assert_eq!(normalized.messages.len(), 1);
    }

    #[test]
    fn rejects_silent_semantic_loss() {
        let request: AnthropicRequest = serde_json::from_value(json!({
            "model":"x", "max_tokens":1, "messages":[{"role":"user","content":"x"}], "thinking":{"type":"enabled","budget_tokens":10}
        })).unwrap();
        assert!(normalize(request, false)
            .unwrap_err()
            .to_string()
            .contains("thinking"));
    }

    #[test]
    fn allows_extra_claude_fields_when_loose_validation_enabled() {
        let request: AnthropicRequest = serde_json::from_value(json!({
            "model":"claude-sonnet-4",
            "max_tokens":100,
            "messages":[{"role":"user","content":"hello"}],
            "tools":[{"name":"lookup","input_schema":{"type":"object"}}],
            "thinking":{"type":"enabled","budget_tokens":10},
            "extra_field":"some-value"
        })).unwrap();
        let result = normalize(request, true);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("thinking"));
    }

    #[test]
    fn accepts_unknown_tool_extensions_when_loose_validation_enabled() {
        let request: AnthropicRequest = serde_json::from_value(json!({
            "model":"claude-sonnet-4",
            "max_tokens":100,
            "messages":[{"role":"user","content":"hello"}],
            "tools":[{"name":"lookup","input_schema":{"type":"object"}, "unknown_tool_meta":"x"}]
        })).unwrap();
        assert!(normalize(request, true).is_ok());
    }
}
