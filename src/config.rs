use std::{
    collections::{HashMap, HashSet},
    env, fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Duration,
};

use serde::Deserialize;

use crate::error::{ErrorKind, ProxyError, Result};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub limits: LimitsConfig,
    pub clients: Vec<ClientConfig>,
    pub providers: Vec<ProviderConfig>,
    pub routes: Vec<RouteConfig>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub bind: SocketAddr,
    pub metrics_bind: Option<SocketAddr>,
    pub max_body_bytes: usize,
    pub shutdown_grace_seconds: u64,
    pub require_anthropic_version: bool,
    pub loose_input_validation: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: ([0, 0, 0, 0], 8082).into(),
            metrics_bind: Some(([127, 0, 0, 1], 9090).into()),
            max_body_bytes: 16 * 1024 * 1024,
            shutdown_grace_seconds: 30,
            require_anthropic_version: true,
            loose_input_validation: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LimitsConfig {
    pub global_concurrency: usize,
    pub connect_timeout_ms: u64,
    pub request_timeout_seconds: u64,
    pub stream_idle_timeout_seconds: u64,
    pub max_attempts: usize,
    pub circuit_failure_threshold: u32,
    pub circuit_cooldown_seconds: u64,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            global_concurrency: 512,
            connect_timeout_ms: 5_000,
            request_timeout_seconds: 120,
            stream_idle_timeout_seconds: 60,
            max_attempts: 2,
            circuit_failure_threshold: 5,
            circuit_cooldown_seconds: 30,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientConfig {
    pub id: String,
    pub key: SecretRef,
    #[serde(default)]
    pub allowed_routes: Vec<String>,
    #[serde(default = "default_rpm")]
    pub requests_per_minute: u32,
    #[serde(default = "default_client_concurrency")]
    pub concurrent_requests: usize,
}

fn default_rpm() -> u32 {
    600
}
fn default_client_concurrency() -> usize {
    100
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    pub id: String,
    pub kind: ProviderKind,
    pub endpoint: String,
    pub credential: CredentialConfig,
    #[serde(default)]
    pub headers: HashMap<String, SecretOrLiteral>,
    #[serde(default)]
    pub capability_profile: CapabilityProfile,
    #[serde(default)]
    pub allow_insecure_http: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    OpenaiChat,
    AzureChat,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CredentialConfig {
    Bearer { secret: SecretRef },
    ApiKey { secret: SecretRef },
    AzureEntra { token: SecretRef },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SecretOrLiteral {
    Literal(String),
    Secret(SecretRef),
}

impl SecretOrLiteral {
    pub fn resolve(&self) -> Result<String> {
        match self {
            Self::Literal(value) => Ok(value.clone()),
            Self::Secret(secret) => secret.resolve(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SecretRef {
    Env { env: String },
    File { file: PathBuf },
}

impl SecretRef {
    pub fn resolve(&self) -> Result<String> {
        let value = match self {
            Self::Env { env: name } => env::var(name).map_err(|_| {
                ProxyError::new(
                    ErrorKind::Internal,
                    format!("required secret environment variable {name} is not set"),
                )
            })?,
            Self::File { file } => fs::read_to_string(file).map_err(|error| {
                ProxyError::new(
                    ErrorKind::Internal,
                    format!("cannot read secret file {}: {error}", file.display()),
                )
            })?,
        };
        let value = value.trim().to_owned();
        if value.is_empty() {
            return Err(ProxyError::new(
                ErrorKind::Internal,
                "resolved secret is empty",
            ));
        }
        Ok(value)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CapabilityProfile {
    pub tokenizer: String,
    pub vision: bool,
    pub tools: bool,
    pub parallel_tools: bool,
    pub supports_temperature: bool,
    pub supports_top_p: bool,
    pub supports_stop: bool,
    pub max_output_tokens: u32,
    pub use_max_completion_tokens: bool,
    pub allow_max_tokens_clamping: bool,
    pub allow_temperature_fallback: bool,
    pub allow_top_p_fallback: bool,
    pub allow_stop_fallback: bool,
    pub allow_parallel_tool_fallback: bool,
    pub allow_structured_tool_results_to_string: bool,
}

impl Default for CapabilityProfile {
    fn default() -> Self {
        Self {
            tokenizer: "o200k_base".into(),
            vision: true,
            tools: true,
            parallel_tools: true,
            supports_temperature: true,
            supports_top_p: true,
            supports_stop: true,
            max_output_tokens: 16_384,
            use_max_completion_tokens: false,
            allow_max_tokens_clamping: false,
            allow_temperature_fallback: false,
            allow_top_p_fallback: false,
            allow_stop_fallback: false,
            allow_parallel_tool_fallback: false,
            allow_structured_tool_results_to_string: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteConfig {
    pub id: String,
    pub models: Vec<String>,
    pub targets: Vec<TargetConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetConfig {
    pub provider: String,
    pub model: String,
    #[serde(default = "default_priority")]
    pub priority: u32,
    #[serde(default = "default_weight")]
    pub weight: u32,
}

fn default_priority() -> u32 {
    1
}
fn default_weight() -> u32 {
    100
}

impl AppConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path).map_err(|error| {
            ProxyError::new(
                ErrorKind::Internal,
                format!("cannot read config {}: {error}", path.display()),
            )
        })?;
        let config: Self = serde_yaml::from_str(&source).map_err(|error| {
            ProxyError::new(
                ErrorKind::Internal,
                format!("invalid config {}: {error}", path.display()),
            )
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.clients.is_empty() || self.providers.is_empty() || self.routes.is_empty() {
            return Err(ProxyError::new(
                ErrorKind::Internal,
                "clients, providers, and routes must not be empty",
            ));
        }
        if self.limits.global_concurrency == 0 || self.limits.max_attempts == 0 {
            return Err(ProxyError::new(
                ErrorKind::Internal,
                "concurrency and max_attempts must be greater than zero",
            ));
        }

        let mut ids = HashSet::new();
        for provider in &self.providers {
            unique_id(&mut ids, "provider", &provider.id)?;
            let parsed_endpoint = reqwest::Url::parse(&provider.endpoint).map_err(|error| {
                ProxyError::new(
                    ErrorKind::Internal,
                    format!("provider {} has invalid endpoint: {error}", provider.id),
                )
            })?;
            if !parsed_endpoint.username().is_empty() || parsed_endpoint.password().is_some() {
                return Err(ProxyError::new(
                    ErrorKind::Internal,
                    format!(
                        "provider {} endpoint must not contain credentials",
                        provider.id
                    ),
                ));
            }
            if !(provider.endpoint.starts_with("https://")
                || provider.allow_insecure_http && provider.endpoint.starts_with("http://"))
            {
                return Err(ProxyError::new(
                    ErrorKind::Internal,
                    format!(
                        "provider {} must use HTTPS or explicitly allow insecure HTTP",
                        provider.id
                    ),
                ));
            }
            provider.credential.secret().resolve()?;
            for name in provider.headers.keys() {
                validate_header_name(name)?;
            }
        }

        ids.clear();
        let provider_ids: HashSet<_> = self.providers.iter().map(|p| p.id.as_str()).collect();
        let mut route_ids = HashSet::new();
        for route in &self.routes {
            unique_id(&mut ids, "route", &route.id)?;
            route_ids.insert(route.id.as_str());
            if route.models.is_empty() || route.targets.is_empty() {
                return Err(ProxyError::new(
                    ErrorKind::Internal,
                    format!("route {} needs models and targets", route.id),
                ));
            }
            for target in &route.targets {
                if !provider_ids.contains(target.provider.as_str()) || target.weight == 0 {
                    return Err(ProxyError::new(
                        ErrorKind::Internal,
                        format!(
                            "route {} has an invalid target {}",
                            route.id, target.provider
                        ),
                    ));
                }
            }
        }

        ids.clear();
        for client in &self.clients {
            unique_id(&mut ids, "client", &client.id)?;
            if client.key.resolve()?.len() < 24 {
                return Err(ProxyError::new(
                    ErrorKind::Internal,
                    format!("client {} key must be at least 24 bytes", client.id),
                ));
            }
            if client.concurrent_requests == 0 || client.requests_per_minute == 0 {
                return Err(ProxyError::new(
                    ErrorKind::Internal,
                    format!("client {} limits must be non-zero", client.id),
                ));
            }
            for route in &client.allowed_routes {
                if !route_ids.contains(route.as_str()) {
                    return Err(ProxyError::new(
                        ErrorKind::Internal,
                        format!("client {} references unknown route {route}", client.id),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.limits.request_timeout_seconds)
    }
}

impl CredentialConfig {
    pub fn secret(&self) -> &SecretRef {
        match self {
            Self::Bearer { secret }
            | Self::ApiKey { secret }
            | Self::AzureEntra { token: secret } => secret,
        }
    }
}

fn unique_id(ids: &mut HashSet<String>, kind: &str, id: &str) -> Result<()> {
    if id.is_empty() || !ids.insert(id.to_owned()) {
        return Err(ProxyError::new(
            ErrorKind::Internal,
            format!("{kind} id is empty or duplicated: {id}"),
        ));
    }
    Ok(())
}

fn validate_header_name(name: &str) -> Result<()> {
    let forbidden = [
        "authorization",
        "api-key",
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
    ];
    if forbidden.iter().any(|item| name.eq_ignore_ascii_case(item)) {
        return Err(ProxyError::new(
            ErrorKind::Internal,
            format!("custom header {name} is forbidden"),
        ));
    }
    http::HeaderName::from_bytes(name.as_bytes()).map_err(|_| {
        ProxyError::new(
            ErrorKind::Internal,
            format!("invalid custom header name {name}"),
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_server_is_safe() {
        let config = ServerConfig::default();
        assert!(config.require_anthropic_version);
        assert_eq!(config.max_body_bytes, 16 * 1024 * 1024);
    }
}
