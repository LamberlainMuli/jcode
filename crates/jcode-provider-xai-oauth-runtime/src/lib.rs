//! SuperGrok (`xai-oauth`) Responses runtime.
//!
//! Paid `xai` and `grok-build` stay on their own paths. This crate only talks
//! to `https://api.x.ai/v1/responses` with an OAuth bearer.

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use jcode_message_types::{Message, StreamEvent, ToolDefinition};
use jcode_provider_core::{EventStream, Provider, shared_http_client};
use jcode_provider_openai::request::{
    build_responses_input, build_tools_for_xai_oauth, reconcile_xai_oauth_tool_choice,
};
use jcode_provider_openai::stream::{OpenAIResponsesStream, OpenAiResponseParseMode};
use reqwest::Client;
use serde_json::{Value, json};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

const DEFAULT_MODEL: &str = "grok-4.6";
const API_BASE: &str = "https://api.x.ai/v1";
const PROVIDER_NAME: &str = "xai-oauth";
const DISPLAY_NAME: &str = "xAI Grok OAuth";
const API_METHOD: &str = "xai-oauth-responses";
const SEED_MODELS: &[&str] = &[
    "grok-4.6",
    "grok-4.5",
    "grok-4.20-0309-reasoning",
    "grok-code-fast-1",
];
const INITIAL_RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(180);

type TokenResolver = Arc<dyn Fn() -> Option<String> + Send + Sync>;

pub struct XaiOauthProvider {
    client: Client,
    model: Arc<RwLock<String>>,
    token: Option<String>,
    token_resolver: Option<TokenResolver>,
    prompt_cache_key: Option<String>,
    fetched_models: Arc<RwLock<Vec<String>>>,
}

impl XaiOauthProvider {
    pub fn new() -> Self {
        Self::with_token(None)
    }

    pub fn with_token(token: Option<String>) -> Self {
        Self {
            client: shared_http_client(),
            model: Arc::new(RwLock::new(
                std::env::var("JCODE_XAI_OAUTH_MODEL")
                    .ok()
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            )),
            token: token
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
            token_resolver: None,
            prompt_cache_key: std::env::var("JCODE_XAI_OAUTH_PROMPT_CACHE_KEY")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
            fetched_models: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn with_token_resolver(
        mut self,
        resolver: impl Fn() -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        self.token_resolver = Some(Arc::new(resolver));
        self
    }

    async fn ensure_access_token(&self) -> Result<String> {
        if let Some(token) = self
            .token
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Ok(token.to_string());
        }
        if let Some(resolver) = &self.token_resolver
            && let Some(token) = resolver()
        {
            let trimmed = token.trim();
            if !trimmed.is_empty() {
                return Ok(trimmed.to_string());
            }
        }
        jcode_base::auth::xai_oauth::ensure_access_token(&self.client)
            .await
            .context("xai-oauth token missing: set XAI_OAUTH_TOKEN or complete SuperGrok login")
    }

    fn refreshable_file_token(&self) -> bool {
        self.token.is_none() && jcode_base::auth::xai_oauth::can_refresh_stored_credentials()
    }

    fn user_agent() -> String {
        std::env::var("JCODE_USER_AGENT")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| jcode_provider_core::JCODE_USER_AGENT.to_string())
    }

    fn current_model(&self) -> String {
        self.model
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn prompt_cache_key(&self, resume_session_id: Option<&str>) -> Option<String> {
        self.prompt_cache_key.clone().or_else(|| {
            resume_session_id
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        })
    }
}

impl Default for XaiOauthProvider {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn build_xai_oauth_response_request(
    model: &str,
    instructions: &str,
    input: &[Value],
    tools: &[Value],
    prompt_cache_key: Option<&str>,
) -> Value {
    let mut request = json!({
        "model": model,
        "input": input,
        "tools": tools,
        "stream": true,
        "store": false,
        "include": ["reasoning.encrypted_content"],
        "parallel_tool_calls": !tools.is_empty(),
    });
    if !instructions.trim().is_empty() {
        request["instructions"] = json!(instructions);
    }
    if tools.is_empty() {
        request
            .as_object_mut()
            .expect("object")
            .remove("tool_choice");
    } else {
        let mut tool_choice = json!("auto");
        reconcile_xai_oauth_tool_choice(&mut tool_choice, tools);
        if tool_choice.is_null() {
            request
                .as_object_mut()
                .expect("object")
                .remove("tool_choice");
        } else {
            request["tool_choice"] = tool_choice;
        }
    }
    if let Some(key) = prompt_cache_key.filter(|value| !value.is_empty()) {
        request["prompt_cache_key"] = json!(key);
    }
    request
}

fn merge_seed_models(dynamic: &[String], current: &str) -> Vec<String> {
    let mut models: Vec<String> = SEED_MODELS
        .iter()
        .map(|model| (*model).to_string())
        .collect();
    for model in dynamic {
        let trimmed = model.trim();
        if !trimmed.is_empty() && !models.iter().any(|existing| existing == trimmed) {
            models.push(trimmed.to_string());
        }
    }
    if !current.trim().is_empty() && !models.iter().any(|existing| existing == current) {
        models.insert(0, current.to_string());
    }
    models
}

fn parse_models_payload(value: &Value) -> Vec<String> {
    let ids = value
        .get("data")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| value.as_array().cloned())
        .unwrap_or_default();
    ids.into_iter()
        .filter_map(|entry| {
            entry
                .get("id")
                .and_then(Value::as_str)
                .or_else(|| entry.as_str())
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(str::to_string)
        })
        .collect()
}

fn responses_url() -> String {
    let base = std::env::var("JCODE_XAI_OAUTH_API_BASE")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| API_BASE.to_string());
    format!("{}/responses", base.trim_end_matches('/'))
}

async fn stream_xai_oauth_response(
    client: Client,
    mut token: String,
    request: Value,
    prompt_cache_key: Option<String>,
    tx: mpsc::Sender<Result<StreamEvent>>,
    refreshable: bool,
) -> Result<()> {
    let url = responses_url();
    let mut retried_unauthorized = false;
    let response = loop {
        let user_agent = XaiOauthProvider::user_agent();
        let mut builder = client
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .header("User-Agent", user_agent)
            .header("Accept", "text/event-stream");
        if let Some(key) = prompt_cache_key.as_deref() {
            builder = builder.header("x-grok-conv-id", key);
        }

        let response = jcode_provider_core::transport::send_with_initial_response_timeout(
            builder.json(&request),
            INITIAL_RESPONSE_TIMEOUT,
        )
        .await
        .context("Failed to send xai-oauth Responses request")?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED
            && refreshable
            && !retried_unauthorized
        {
            retried_unauthorized = true;
            match jcode_base::auth::xai_oauth::refresh_access_token(&client).await {
                Ok(credentials) => {
                    token = credentials.access_token;
                    continue;
                }
                Err(error) => {
                    let status = response.status();
                    let body = jcode_provider_core::http_error_body(response, "xai-oauth").await;
                    anyhow::bail!(
                        "xai-oauth API error {status}: {body} (refresh failed: {error})"
                    );
                }
            }
        }
        break response;
    };

    if !response.status().is_success() {
        let status = response.status();
        let body = jcode_provider_core::http_error_body(response, "xai-oauth").await;
        anyhow::bail!("xai-oauth API error {status}: {body}");
    }

    let _ = tx
        .send(Ok(StreamEvent::ConnectionType {
            connection: "https/sse".to_string(),
        }))
        .await;

    let mut stream = OpenAIResponsesStream::new_with_mode(
        response.bytes_stream(),
        OpenAiResponseParseMode { xai_oauth: true },
    );
    loop {
        let next = match tokio::time::timeout(STREAM_IDLE_TIMEOUT, stream.next()).await {
            Ok(item) => item,
            Err(_) => {
                anyhow::bail!(
                    "xai-oauth stream idle timeout after {}s",
                    STREAM_IDLE_TIMEOUT.as_secs()
                );
            }
        };
        match next {
            Some(Ok(event)) => {
                if tx.send(Ok(event)).await.is_err() {
                    break;
                }
            }
            Some(Err(error)) => {
                let _ = tx.send(Err(error)).await;
                break;
            }
            None => break,
        }
    }
    Ok(())
}

#[async_trait]
impl Provider for XaiOauthProvider {
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        system: &str,
        resume_session_id: Option<&str>,
    ) -> Result<EventStream> {
        let token = self.ensure_access_token().await?;
        let refreshable = self.refreshable_file_token();
        let model = self.current_model();
        let input = build_responses_input(messages);
        let api_tools = build_tools_for_xai_oauth(tools);
        let prompt_cache_key = self.prompt_cache_key(resume_session_id);
        let request = build_xai_oauth_response_request(
            &model,
            system,
            &input,
            &api_tools,
            prompt_cache_key.as_deref(),
        );

        let (tx, rx) = mpsc::channel::<Result<StreamEvent>>(100);
        let client = self.client.clone();
        tokio::spawn(async move {
            if let Err(error) = stream_xai_oauth_response(
                client,
                token,
                request,
                prompt_cache_key,
                tx.clone(),
                refreshable,
            )
            .await
            {
                let _ = tx.send(Err(error)).await;
            }
        });
        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn display_name(&self) -> String {
        DISPLAY_NAME.to_string()
    }

    fn model(&self) -> String {
        self.current_model()
    }

    fn set_model(&self, model: &str) -> Result<()> {
        let trimmed = jcode_provider_core::strip_own_model_prefix(model, "xai-oauth:");
        if trimmed.is_empty() {
            anyhow::bail!("xai-oauth model cannot be empty");
        }
        *self
            .model
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = trimmed.to_string();
        Ok(())
    }

    fn available_models(&self) -> Vec<&'static str> {
        SEED_MODELS.to_vec()
    }

    fn available_models_display(&self) -> Vec<String> {
        let dynamic = self
            .fetched_models
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        merge_seed_models(&dynamic, &self.current_model())
    }

    fn model_routes(&self) -> Vec<jcode_provider_core::ModelRoute> {
        self.available_models_display()
            .into_iter()
            .map(|model| jcode_provider_core::ModelRoute {
                model,
                provider: DISPLAY_NAME.to_string(),
                api_method: API_METHOD.to_string(),
                available: true,
                detail: String::new(),
                cheapness: None,
            })
            .collect()
    }

    async fn prefetch_models(&self) -> Result<()> {
        let Ok(token) = self.ensure_access_token().await else {
            return Ok(());
        };
        let url = format!("{API_BASE}/models");
        let response = self
            .client
            .get(url)
            .header("Authorization", format!("Bearer {token}"))
            .header("User-Agent", Self::user_agent())
            .send()
            .await;
        let Ok(response) = response else {
            return Ok(());
        };
        if !response.status().is_success() {
            return Ok(());
        }
        let Ok(payload) = response.json::<Value>().await else {
            return Ok(());
        };
        let models = parse_models_payload(&payload);
        if !models.is_empty() {
            *self
                .fetched_models
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = models;
        }
        Ok(())
    }

    fn handles_tools_internally(&self) -> bool {
        false
    }

    fn fork(&self) -> Arc<dyn Provider> {
        Arc::new(Self {
            client: self.client.clone(),
            model: Arc::new(RwLock::new(self.current_model())),
            token: self.token.clone(),
            token_resolver: self.token_resolver.clone(),
            prompt_cache_key: self.prompt_cache_key.clone(),
            fetched_models: self.fetched_models.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jcode_base::auth::xai_oauth::{
        ENV_TOKEN, XaiOauthCredentials, load_credentials, save_credentials,
    };
    use jcode_message_types::ToolDefinition;
    use std::ffi::{OsStr, OsString};
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    #[test]
    fn parallel_tool_calls_true_iff_tools_non_empty() {
        let with_tools = build_xai_oauth_response_request(
            "grok-4.6",
            "sys",
            &[],
            &[json!({"type":"function","name":"read"})],
            None,
        );
        assert_eq!(with_tools["parallel_tool_calls"], json!(true));
        assert_eq!(with_tools["tool_choice"], json!("auto"));
        assert_eq!(with_tools["stream"], json!(true));

        assert_eq!(with_tools["store"], json!(false));
        assert!(with_tools.get("reasoning").is_none());
        assert!(with_tools.get("presence_penalty").is_none());
        assert!(with_tools.get("frequency_penalty").is_none());

        let empty = build_xai_oauth_response_request("grok-4.6", "", &[], &[], Some("sess"));
        assert_eq!(empty["parallel_tool_calls"], json!(false));
        assert!(empty.get("tool_choice").is_none());
        assert_eq!(empty["prompt_cache_key"], json!("sess"));
    }
    #[test]
    fn seed_models_exclude_grok_build() {
        assert!(
            !SEED_MODELS.contains(&"grok-build"),
            "grok-build is a separate product and must not be offered via xai-oauth"
        );
    }


    #[test]
    fn identity_defaults() {
        let provider = XaiOauthProvider::new();
        assert_eq!(provider.name(), "xai-oauth");
        assert_eq!(provider.display_name(), "xAI Grok OAuth");
        assert!(!provider.handles_tools_internally());
        assert!(
            provider
                .model_routes()
                .iter()
                .all(|route| route.api_method == "xai-oauth-responses")
        );
    }

    #[test]
    fn build_tools_for_xai_oauth_is_used_by_request_path() {
        let defs = vec![ToolDefinition {
            name: "mcp__codebase_memory_check_index_coverage".to_string(),
            description: "coverage".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "project": { "type": "string" },
                    "paths": { "type": "array", "items": { "type": "string" } },
                    "scopes": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["project"],
                "anyOf": [{ "required": ["paths"] }, { "required": ["scopes"] }]
            }),
        }];
        let tools = build_tools_for_xai_oauth(&defs);
        assert!(tools[0]["parameters"].get("anyOf").is_none());
        let request = build_xai_oauth_response_request("grok-4.6", "", &[], &tools, None);
        assert_eq!(request["parallel_tool_calls"], json!(true));
    }

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: impl AsRef<OsStr>) -> Self {
            let previous = std::env::var_os(key);
            jcode_base::env::set_var(key, value);
            Self { key, previous }
        }

        fn unset(key: &'static str) -> Self {
            let previous = std::env::var_os(key);
            jcode_base::env::remove_var(key);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => jcode_base::env::set_var(self.key, value),
                None => jcode_base::env::remove_var(self.key),
            }
        }
    }

    async fn serve_http_sequence(replies: Vec<(u16, &'static str, &'static str)>) -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            for (status, content_type, body) in replies {
                let (stream, _) = listener.accept().await.unwrap();
                let (reader, mut writer) = stream.into_split();
                let mut reader = BufReader::new(reader);
                let mut request_line = String::new();
                reader.read_line(&mut request_line).await.unwrap();
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).await.unwrap();
                    if line.trim().is_empty() {
                        break;
                    }
                    if line.to_ascii_lowercase().starts_with("content-length:")
                        && let Some((_, value)) = line.split_once(':')
                    {
                        content_length = value.trim().parse().unwrap_or(0);
                    }
                }
                if content_length > 0 {
                    let mut buf = vec![0u8; content_length];
                    reader.read_exact(&mut buf).await.unwrap();
                }
                let response = format!(
                    "HTTP/1.1 {status} OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                writer.write_all(response.as_bytes()).await.unwrap();
            }
        });
        port
    }

    #[tokio::test]
    async fn retries_responses_once_after_401_refresh() {
        let _lock = jcode_base::storage::lock_test_env();
        let temp = tempfile::TempDir::new().unwrap();
        let _home = EnvVarGuard::set("JCODE_HOME", temp.path());
        let _oauth = EnvVarGuard::unset(ENV_TOKEN);
        save_credentials(&XaiOauthCredentials {
            access_token: "stale-access".to_string(),
            refresh_token: Some("stored-refresh".to_string()),
            expires_at: Some(1_800_000_000_000),
            account: None,
            email: None,
        })
        .unwrap();

        let token_body = r#"{"access_token":"new-access","refresh_token":"new-refresh","expires_in":3600}"#;
        let sse_body = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"output\":[]}}\n\n";
        let port = serve_http_sequence(vec![
            (401, "application/json", r#"{"error":"invalid_token"}"#),
            (200, "application/json", token_body),
            (200, "text/event-stream", sse_body),
        ])
        .await;
        let origin = format!("http://127.0.0.1:{port}");
        let _api = EnvVarGuard::set("JCODE_XAI_OAUTH_API_BASE", &origin);
        let _token_url = EnvVarGuard::set(
            "JCODE_XAI_OAUTH_TOKEN_URL",
            format!("{origin}/oauth2/token"),
        );

        let (tx, mut rx) = mpsc::channel(8);
        stream_xai_oauth_response(
            reqwest::Client::new(),
            "stale-access".to_string(),
            json!({"model": "grok-4.6", "input": [], "stream": true}),
            None,
            tx,
            true,
        )
        .await
        .expect("401 should refresh and retry");

        let first = rx.recv().await.expect("connection event");
        assert!(matches!(
            first,
            Ok(StreamEvent::ConnectionType { .. })
        ));
        assert_eq!(
            load_credentials().unwrap().access_token,
            "new-access"
        );
    }
}
