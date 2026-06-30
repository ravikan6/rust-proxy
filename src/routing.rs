use std::{collections::HashMap, sync::Arc};

use futures::StreamExt;
use globset::{Glob, GlobSet, GlobSetBuilder};
use rand::Rng;

use crate::{
    config::{AppConfig, RouteConfig, TargetConfig},
    domain::{AnthropicResponse, NormalizedRequest},
    error::{ErrorKind, ProxyError, Result},
    provider::{EventStream, Provider},
};

pub struct Router {
    routes: Vec<Route>,
    providers: HashMap<String, Arc<Provider>>,
    max_attempts: usize,
    timeout: std::time::Duration,
}

pub struct Route {
    pub id: String,
    patterns: GlobSet,
    targets: Vec<TargetConfig>,
}

pub struct RoutedRequest<'a> {
    pub route: &'a Route,
    pub request: &'a NormalizedRequest,
}

impl Router {
    pub fn new(config: &AppConfig) -> Result<Self> {
        let providers = config
            .providers
            .iter()
            .map(|provider| {
                Ok((
                    provider.id.clone(),
                    Arc::new(Provider::new(provider, &config.limits)?),
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let routes = config
            .routes
            .iter()
            .map(Route::compile)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            routes,
            providers,
            max_attempts: config.limits.max_attempts,
            timeout: config.request_timeout(),
        })
    }

    pub fn resolve<'a>(&'a self, request: &'a NormalizedRequest) -> Result<RoutedRequest<'a>> {
        let matches = self
            .routes
            .iter()
            .filter(|route| route.patterns.is_match(&request.original_model))
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => Err(ProxyError::invalid(format!(
                "no route matches model {}",
                request.original_model
            ))),
            [route] => Ok(RoutedRequest { route, request }),
            _ => Err(ProxyError::invalid(format!(
                "multiple routes match model {}; configuration is ambiguous",
                request.original_model
            ))),
        }
    }

    pub async fn execute(&self, routed: RoutedRequest<'_>) -> Result<AnthropicResponse> {
        let candidates = self.candidates(routed.route, routed.request)?;
        let mut errors = Vec::new();
        for (attempt, (provider, target, request)) in
            candidates.into_iter().take(self.max_attempts).enumerate()
        {
            if attempt > 0 {
                metrics::counter!("proxy_fallback_total", "route" => routed.route.id.clone())
                    .increment(1);
            }
            let outcome = tokio::time::timeout(
                self.timeout,
                provider.execute(&request, &target.model),
            )
            .await;
            match outcome {
                Ok(Ok(response)) => {
                    provider.record_success();
                    return Ok(response);
                }
                Ok(Err(error)) => {
                    provider.record_failure(&error);
                    if !error.retryable() {
                        return Err(error);
                    }
                    errors.push(error);
                }
                Err(_) => {
                    let error = ProxyError::new(ErrorKind::Timeout, "upstream request timed out");
                    provider.record_failure(&error);
                    return Err(error);
                }
            }
        }
        Err(errors.pop().unwrap_or_else(|| {
            ProxyError::new(ErrorKind::Overloaded, "no healthy target is available")
        }))
    }

    pub async fn stream(&self, routed: RoutedRequest<'_>) -> Result<(u64, EventStream)> {
        let candidates = self.candidates(routed.route, routed.request)?;
        let input_tokens = candidates
            .first()
            .map(|(provider, _, request)| provider.count_tokens(request))
            .transpose()?
            .unwrap_or(0);
        let mut errors = Vec::new();
        for (attempt, (provider, target, request)) in
            candidates.into_iter().take(self.max_attempts).enumerate()
        {
            if attempt > 0 {
                metrics::counter!("proxy_fallback_total", "route" => routed.route.id.clone())
                    .increment(1);
            }
            match tokio::time::timeout(self.timeout, provider.stream(&request, &target.model))
                .await
            {
                Ok(Ok(mut upstream)) => {
                    let provider_for_stream = provider.clone();
                    let output = async_stream::stream! {
                        while let Some(event) = upstream.next().await {
                            match event {
                                Ok(event) => yield Ok(event),
                                Err(error) => {
                                    provider_for_stream.record_failure(&error);
                                    yield Err(error);
                                    return;
                                }
                            }
                        }
                        provider_for_stream.record_success();
                    };
                    return Ok((input_tokens, Box::pin(output)));
                }
                Ok(Err(error)) => {
                    provider.record_failure(&error);
                    if !error.retryable() {
                        return Err(error);
                    }
                    errors.push(error);
                }
                Err(_) => {
                    let error = ProxyError::new(
                        ErrorKind::Timeout,
                        "upstream stream establishment timed out",
                    );
                    provider.record_failure(&error);
                    return Err(error);
                }
            }
        }
        Err(errors.pop().unwrap_or_else(|| {
            ProxyError::new(ErrorKind::Overloaded, "no healthy target is available")
        }))
    }

    pub fn count_tokens(&self, routed: RoutedRequest<'_>) -> Result<u64> {
        let candidates = self.candidates(routed.route, routed.request)?;
        let (provider, _, request) = candidates.first().ok_or_else(|| {
            ProxyError::new(ErrorKind::Overloaded, "no healthy target is available")
        })?;
        provider.count_tokens(request)
    }

    pub async fn probe(&self, provider_id: &str) -> Result<()> {
        let provider = self
            .providers
            .get(provider_id)
            .ok_or_else(|| ProxyError::invalid(format!("unknown provider {provider_id}")))?;
        provider.probe().await
    }

    fn candidates(
        &self,
        route: &Route,
        request: &NormalizedRequest,
    ) -> Result<Vec<(Arc<Provider>, TargetConfig, NormalizedRequest)>> {
        let mut compatible = Vec::new();
        let mut capability_error = None;
        for target in &route.targets {
            let provider = self
                .providers
                .get(&target.provider)
                .expect("validated provider reference");
            match provider.adapt_request(request) {
                Ok(adapted) if provider.available() => {
                    compatible.push((provider.clone(), target.clone(), adapted))
                }
                Ok(_) => {}
                Err(error) => {
                    capability_error.get_or_insert(error);
                }
            }
        }
        if compatible.is_empty() {
            return Err(capability_error.unwrap_or_else(|| {
                ProxyError::new(
                    ErrorKind::Overloaded,
                    "all route targets have open circuits",
                )
            }));
        }
        Ok(weighted_priority_order(compatible))
    }
}

impl Route {
    fn compile(config: &RouteConfig) -> Result<Self> {
        let mut builder = GlobSetBuilder::new();
        for pattern in &config.models {
            builder.add(Glob::new(pattern).map_err(|error| {
                ProxyError::new(
                    ErrorKind::Internal,
                    format!("invalid model glob {pattern}: {error}"),
                )
            })?);
        }
        let patterns = builder.build().map_err(|error| {
            ProxyError::new(
                ErrorKind::Internal,
                format!("cannot compile route {}: {error}", config.id),
            )
        })?;
        Ok(Self {
            id: config.id.clone(),
            patterns,
            targets: config.targets.clone(),
        })
    }
}

fn weighted_priority_order(
    mut candidates: Vec<(Arc<Provider>, TargetConfig, NormalizedRequest)>,
) -> Vec<(Arc<Provider>, TargetConfig, NormalizedRequest)> {
    candidates.sort_by_key(|(_, target, _)| target.priority);
    let mut output = Vec::with_capacity(candidates.len());
    while !candidates.is_empty() {
        let priority = candidates[0].1.priority;
        let tier_len = candidates
            .iter()
            .take_while(|(_, target, _)| target.priority == priority)
            .count();
        let total: u64 = candidates[..tier_len]
            .iter()
            .map(|(_, target, _)| target.weight as u64)
            .sum();
        let mut choice = rand::rng().random_range(0..total);
        let mut selected = 0;
        for (index, (_, target, _)) in candidates[..tier_len].iter().enumerate() {
            if choice < target.weight as u64 {
                selected = index;
                break;
            }
            choice -= target.weight as u64;
        }
        output.push(candidates.remove(selected));
    }
    output
}
