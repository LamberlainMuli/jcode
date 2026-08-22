//! SuperGrok (`xai-oauth`) RFC 8628 device-code identity.
//!
//! Distinct from paid `xai` (`XAI_API_KEY`) and from `grok-build` (ACP +
//! `~/.grok/auth.json`). Credentials live only in `~/.jcode/xai-oauth.json`
//! (via `jcode_dir()`) or `XAI_OAUTH_TOKEN`.

use anyhow::{Context, Result, bail};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const ENV_TOKEN: &str = "XAI_OAUTH_TOKEN";
const OAUTH_ISSUER: &str = "https://auth.x.ai";
const OAUTH_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const OAUTH_SCOPES: &str = "openid profile email offline_access grok-cli:access api:access";
const ACCESS_TOKEN_CLIENT_SKEW_MS: i64 = 5 * 60 * 1000;
const CREDENTIALS_FILE: &str = "xai-oauth.json";

#[derive(Clone, Debug, Deserialize)]
pub struct DeviceAuthorization {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub verification_uri_complete: Option<String>,
    pub expires_in: u64,
    #[serde(default = "default_poll_interval")]
    pub interval: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct XaiOauthCredentials {
    pub access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
}

impl XaiOauthCredentials {
    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(expires_at) => chrono::Utc::now().timestamp_millis() >= expires_at,
            None => false,
        }
    }
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct TokenError {
    error: String,
    error_description: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct JwtClaims {
    sub: Option<String>,
    email: Option<String>,
}

fn default_poll_interval() -> u64 {
    5
}

pub fn credentials_path() -> Result<PathBuf> {
    Ok(crate::storage::jcode_dir()?.join(CREDENTIALS_FILE))
}

fn env_oauth_token() -> Option<String> {
    std::env::var(ENV_TOKEN)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub fn env_token_present() -> bool {
    env_oauth_token().is_some()
}

/// Env `XAI_OAUTH_TOKEN` first, else the stored access token.
pub fn access_token() -> Option<String> {
    if let Some(token) = env_oauth_token() {
        return Some(token);
    }
    load_credentials()
        .ok()
        .map(|credentials| credentials.access_token)
        .filter(|token| !token.trim().is_empty())
}

pub fn has_cached_login() -> bool {
    if env_oauth_token().is_some() {
        return true;
    }
    load_credentials()
        .ok()
        .is_some_and(|credentials| !credentials.access_token.trim().is_empty())
}

pub fn load_credentials() -> Result<XaiOauthCredentials> {
    let path = credentials_path()?;
    if !path.exists() {
        bail!("No SuperGrok credentials found. Run `jcode login --provider xai-oauth`.");
    }
    crate::storage::harden_secret_file_permissions(&path);
    let credentials: XaiOauthCredentials = crate::storage::read_json(&path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    if credentials.access_token.trim().is_empty() {
        bail!("No SuperGrok credentials found. Run `jcode login --provider xai-oauth`.");
    }
    Ok(credentials)
}

pub fn save_credentials(credentials: &XaiOauthCredentials) -> Result<()> {
    let path = credentials_path()?;
    crate::storage::write_json_secret(&path, credentials)?;
    super::AuthStatus::invalidate_cache();
    Ok(())
}

pub fn clear_credentials() -> Result<()> {
    let path = credentials_path()?;
    if path.exists() {
        std::fs::remove_file(&path)
            .with_context(|| format!("Failed to remove {}", path.display()))?;
    }
    super::AuthStatus::invalidate_cache();
    Ok(())
}

pub async fn initiate_device_login(client: &reqwest::Client) -> Result<DeviceAuthorization> {
    client
        .post(format!("{OAUTH_ISSUER}/oauth2/device/code"))
        .form(&[("client_id", OAUTH_CLIENT_ID), ("scope", OAUTH_SCOPES)])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await
        .context("invalid xAI SuperGrok device authorization response")
}

pub async fn complete_device_login(
    client: &reqwest::Client,
    authorization: &DeviceAuthorization,
) -> Result<XaiOauthCredentials> {
    let tokens = poll_device_token(client, authorization).await?;
    persist_token_response(tokens, None)
}

pub async fn ensure_access_token(client: &reqwest::Client) -> Result<String> {
    if let Some(token) = env_oauth_token() {
        return Ok(token);
    }
    Ok(load_or_refresh_credentials(client).await?.access_token)
}

pub fn can_refresh_stored_credentials() -> bool {
    if env_oauth_token().is_some() {
        return false;
    }
    load_credentials().ok().is_some_and(|credentials| {
        credentials
            .refresh_token
            .as_deref()
            .is_some_and(|token| !token.trim().is_empty())
    })
}

pub async fn load_or_refresh_credentials(
    client: &reqwest::Client,
) -> Result<XaiOauthCredentials> {
    let credentials = load_credentials()?;
    if !credentials.is_expired() {
        return Ok(credentials);
    }
    refresh_access_token(client).await
}

/// Predicate for [`single_flight`]: the reloaded stored credentials count as
/// "already refreshed" only when they differ from what the caller observed
/// AND are still unexpired. A rotated-but-expired token must not short-circuit;
/// the caller needs another network refresh.
fn reloaded_credentials_fresh(
    observed: Option<&XaiOauthCredentials>,
    stored: &XaiOauthCredentials,
) -> bool {
    observed.is_some_and(|previous| previous.access_token != stored.access_token)
        && !stored.is_expired()
}

/// Refresh SuperGrok tokens. Serialized so concurrent 401s share one exchange.
pub async fn refresh_access_token(client: &reqwest::Client) -> Result<XaiOauthCredentials> {
    let observed = load_credentials().ok();
    crate::auth::refresh_coordinator::single_flight(
        "xai-oauth".to_string(),
        || load_credentials().ok(),
        |stored| reloaded_credentials_fresh(observed.as_ref(), stored),
        {
            let client = client.clone();
            move |stored| async move { refresh_access_token_uncoordinated(&client, stored).await }
        },
    )
    .await
}

fn token_url() -> String {
    std::env::var("JCODE_XAI_OAUTH_TOKEN_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| format!("{OAUTH_ISSUER}/oauth2/token"))
}

async fn refresh_access_token_uncoordinated(
    client: &reqwest::Client,
    stored: Option<XaiOauthCredentials>,
) -> Result<XaiOauthCredentials> {
    let current = match stored {
        Some(credentials) => credentials,
        None => load_credentials()?,
    };
    let refresh_token = current
        .refresh_token
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .context("SuperGrok refresh token is missing; run `jcode login --provider xai-oauth`")?
        .to_string();
    crate::auth::refresh_state::ensure_refresh_allowed(
        "xai-oauth",
        &refresh_token,
        "run `jcode login --provider xai-oauth`",
    )?;
    let result: Result<XaiOauthCredentials> = async {
        let response = client
            .post(token_url())
            .form(&[
                ("grant_type", "refresh_token"),
                ("client_id", OAUTH_CLIENT_ID),
                ("refresh_token", refresh_token.as_str()),
            ])
            .send()
            .await?;
        let status = response.status();
        let body = response.bytes().await?;
        if !status.is_success() {
            let error: TokenError = serde_json::from_slice(&body)
                .with_context(|| format!("xAI SuperGrok token refresh failed with {status}"))?;
            bail!(
                "xAI SuperGrok token refresh failed: {}",
                error.error_description.unwrap_or(error.error)
            );
        }
        let tokens: TokenResponse =
            serde_json::from_slice(&body).context("invalid xAI SuperGrok refresh response")?;
        persist_token_response(tokens, Some(&current))
    }
    .await;
    crate::auth::refresh_state::record_refresh_outcome("xai-oauth", &refresh_token, &result);
    result
}

async fn poll_device_token(
    client: &reqwest::Client,
    authorization: &DeviceAuthorization,
) -> Result<TokenResponse> {
    let deadline = tokio::time::Instant::now()
        + std::time::Duration::from_secs(authorization.expires_in.max(600));
    let mut interval = authorization.interval.max(1);
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
        if tokio::time::Instant::now() >= deadline {
            bail!("xAI SuperGrok device authorization expired");
        }
        let response = client
            .post(format!("{OAUTH_ISSUER}/oauth2/token"))
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("device_code", authorization.device_code.as_str()),
                ("client_id", OAUTH_CLIENT_ID),
            ])
            .send()
            .await?;
        let status = response.status();
        let body = response.bytes().await?;
        if status.is_success() {
            return serde_json::from_slice(&body).context("invalid xAI SuperGrok token response");
        }
        let error: TokenError = serde_json::from_slice(&body)
            .with_context(|| format!("xAI SuperGrok token request failed with {status}"))?;
        match error.error.as_str() {
            "authorization_pending" => continue,
            "slow_down" => {
                interval += 5;
                continue;
            }
            _ => bail!(
                "xAI SuperGrok login failed: {}",
                error.error_description.unwrap_or(error.error)
            ),
        }
    }
}

fn persist_token_response(
    tokens: TokenResponse,
    previous: Option<&XaiOauthCredentials>,
) -> Result<XaiOauthCredentials> {
    let claims = jwt_claims(&tokens.access_token);
    let expires_at = tokens.expires_in.map(|seconds| {
        chrono::Utc::now().timestamp_millis() + (seconds as i64 * 1000)
            - ACCESS_TOKEN_CLIENT_SKEW_MS
    });
    let refresh_token = tokens
        .refresh_token
        .filter(|value| !value.trim().is_empty())
        .or_else(|| previous.and_then(|creds| creds.refresh_token.clone()));
    let credentials = XaiOauthCredentials {
        access_token: tokens.access_token,
        refresh_token,
        expires_at,
        account: claims
            .sub
            .filter(|value| !value.trim().is_empty())
            .or_else(|| previous.and_then(|creds| creds.account.clone())),
        email: claims
            .email
            .filter(|value| !value.trim().is_empty())
            .or_else(|| previous.and_then(|creds| creds.email.clone())),
    };
    save_credentials(&credentials)?;
    Ok(credentials)
}

fn jwt_claims(access_token: &str) -> JwtClaims {
    access_token
        .split('.')
        .nth(1)
        .and_then(|part| {
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(part)
                .ok()
        })
        .and_then(|bytes| serde_json::from_slice::<JwtClaims>(&bytes).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{OsStr, OsString};

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: impl AsRef<OsStr>) -> Self {
            let previous = std::env::var_os(key);
            crate::env::set_var(key, value);
            Self { key, previous }
        }

        fn unset(key: &'static str) -> Self {
            let previous = std::env::var_os(key);
            crate::env::remove_var(key);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => crate::env::set_var(self.key, value),
                None => crate::env::remove_var(self.key),
            }
        }
    }

    fn isolated_home() -> (
        tempfile::TempDir,
        EnvVarGuard,
        EnvVarGuard,
        EnvVarGuard,
        EnvVarGuard,
    ) {
        let temp = tempfile::TempDir::new().unwrap();
        let home = EnvVarGuard::set("JCODE_HOME", temp.path());
        let oauth = EnvVarGuard::unset(ENV_TOKEN);
        let api_key = EnvVarGuard::unset("XAI_API_KEY");
        let token_url = EnvVarGuard::unset("JCODE_XAI_OAUTH_TOKEN_URL");
        (temp, home, oauth, api_key, token_url)
    }

    async fn mock_token_server(status: u16, response_body: &str) -> (u16, tokio::task::JoinHandle<String>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let resp_body = response_body.to_string();
        let handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
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
                if let Some((_, value)) = line.split_once(':')
                    && line.to_ascii_lowercase().starts_with("content-length:")
                {
                    content_length = value.trim().parse().unwrap_or(0);
                }
            }
            let mut body_bytes = vec![0u8; content_length];
            if content_length > 0 {
                reader.read_exact(&mut body_bytes).await.unwrap();
            }
            let body = String::from_utf8(body_bytes).unwrap_or_default();
            let response = format!(
                "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{resp_body}",
                resp_body.len()
            );
            writer.write_all(response.as_bytes()).await.unwrap();
            body
        });
        (port, handle)
    }

    fn expired_stored() -> XaiOauthCredentials {
        XaiOauthCredentials {
            access_token: "stale-access".to_string(),
            refresh_token: Some("stored-refresh".to_string()),
            expires_at: Some(1),
            account: None,
            email: None,
        }
    }

    #[test]
    fn xai_api_key_alone_is_not_supergrok_login() {
        let _lock = crate::storage::lock_test_env();
        let (_temp, _home, _oauth, _api, _token_url) = isolated_home();
        let _paid = EnvVarGuard::set("XAI_API_KEY", "xai-paid-key");
        assert!(!has_cached_login());
        assert_eq!(access_token(), None);
    }

    #[test]
    fn xai_oauth_token_env_marks_signed_in() {
        let _lock = crate::storage::lock_test_env();
        let (_temp, _home, _oauth, _api, _token_url) = isolated_home();
        let _token = EnvVarGuard::set(ENV_TOKEN, "env-oauth-bearer");
        assert!(has_cached_login());
        assert_eq!(access_token().as_deref(), Some("env-oauth-bearer"));
    }

    #[test]
    fn saved_file_marks_signed_in_and_roundtrips() {
        let _lock = crate::storage::lock_test_env();
        let (temp, _home, _oauth, _api, _token_url) = isolated_home();
        assert!(!has_cached_login());

        let stored = XaiOauthCredentials {
            access_token: "stored-access".to_string(),
            refresh_token: Some("stored-refresh".to_string()),
            expires_at: Some(1_800_000_000_000),
            account: Some("user-1".to_string()),
            email: Some("user@example.com".to_string()),
        };
        save_credentials(&stored).unwrap();

        let path = credentials_path().unwrap();
        assert_eq!(path, temp.path().join("xai-oauth.json"));
        assert!(
            !path
                .components()
                .any(|part| part.as_os_str() == OsStr::new(".grok"))
        );
        assert!(has_cached_login());

        let loaded = load_credentials().unwrap();
        assert_eq!(loaded, stored);
        assert_eq!(access_token().as_deref(), Some("stored-access"));

        let _token = EnvVarGuard::set(ENV_TOKEN, "env-wins");
        assert_eq!(access_token().as_deref(), Some("env-wins"));
    }

    #[test]
    fn persist_keeps_previous_refresh_when_response_omits_it() {
        let _lock = crate::storage::lock_test_env();
        let (_temp, _home, _oauth, _api, _token_url) = isolated_home();
        let previous = XaiOauthCredentials {
            access_token: "old-access".to_string(),
            refresh_token: Some("keep-me".to_string()),
            expires_at: Some(1),
            account: Some("acct".to_string()),
            email: Some("old@example.com".to_string()),
        };
        let next = persist_token_response(
            TokenResponse {
                access_token: "new-access".to_string(),
                refresh_token: None,
                expires_in: Some(3600),
            },
            Some(&previous),
        )
        .unwrap();
        assert_eq!(next.access_token, "new-access");
        assert_eq!(next.refresh_token.as_deref(), Some("keep-me"));
        assert_eq!(next.account.as_deref(), Some("acct"));
        assert!(next.expires_at.is_some());
        clear_credentials().unwrap();
    }

    #[tokio::test]
    async fn ensure_access_token_returns_env_without_refresh() {
        let _lock = crate::storage::lock_test_env();
        let (_temp, _home, _oauth, _api, _token_url) = isolated_home();
        save_credentials(&expired_stored()).unwrap();
        let _token = EnvVarGuard::set(ENV_TOKEN, "env-oauth-bearer");
        let _broken = EnvVarGuard::set(
            "JCODE_XAI_OAUTH_TOKEN_URL",
            "http://127.0.0.1:1/oauth2/token",
        );
        let client = reqwest::Client::new();
        let token = ensure_access_token(&client).await.expect("env token");
        assert_eq!(token, "env-oauth-bearer");
        assert!(!can_refresh_stored_credentials());
    }

    #[tokio::test]
    async fn ensure_access_token_keeps_fresh_file_token() {
        let _lock = crate::storage::lock_test_env();
        let (_temp, _home, _oauth, _api, _token_url) = isolated_home();
        save_credentials(&XaiOauthCredentials {
            access_token: "fresh-access".to_string(),
            refresh_token: Some("stored-refresh".to_string()),
            expires_at: Some(1_800_000_000_000),
            account: None,
            email: None,
        })
        .unwrap();
        let _broken = EnvVarGuard::set(
            "JCODE_XAI_OAUTH_TOKEN_URL",
            "http://127.0.0.1:1/oauth2/token",
        );
        let client = reqwest::Client::new();
        let token = ensure_access_token(&client).await.expect("fresh token");
        assert_eq!(token, "fresh-access");
        assert!(can_refresh_stored_credentials());
    }

    #[tokio::test]
    async fn ensure_access_token_refreshes_expired_file_token() {
        let _lock = crate::storage::lock_test_env();
        let (_temp, _home, _oauth, _api, _token_url) = isolated_home();
        save_credentials(&expired_stored()).unwrap();
        let body = serde_json::json!({
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600
        })
        .to_string();
        let (port, handle) = mock_token_server(200, &body).await;
        let _url = EnvVarGuard::set(
            "JCODE_XAI_OAUTH_TOKEN_URL",
            format!("http://127.0.0.1:{port}/oauth2/token"),
        );
        let client = reqwest::Client::new();
        let token = ensure_access_token(&client).await.expect("refresh");
        assert_eq!(token, "new-access");
        let posted = handle.await.expect("mock server");
        assert!(
            posted.contains("grant_type=refresh_token"),
            "refresh form missing: {posted}"
        );
        assert!(
            posted.contains("refresh_token=stored-refresh"),
            "refresh token missing: {posted}"
        );
        let stored = load_credentials().unwrap();
        assert_eq!(stored.access_token, "new-access");
        assert_eq!(stored.refresh_token.as_deref(), Some("new-refresh"));
        assert!(!stored.is_expired());
    }

    #[test]
    fn reloaded_credentials_fresh_requires_new_and_unexpired_token() {
        let observed = expired_stored();
        let rotated_unexpired = XaiOauthCredentials {
            access_token: "rotated-access".to_string(),
            refresh_token: Some("rotated-refresh".to_string()),
            expires_at: Some(1_800_000_000_000),
            account: None,
            email: None,
        };
        let rotated_expired = XaiOauthCredentials {
            expires_at: Some(1),
            ..rotated_unexpired.clone()
        };

        // Another task refreshed to a new, unexpired token: short-circuit.
        assert!(reloaded_credentials_fresh(Some(&observed), &rotated_unexpired));
        // Rotated but still expired: must NOT count as already refreshed.
        assert!(!reloaded_credentials_fresh(Some(&observed), &rotated_expired));
        // Same token we observed (no concurrent refresh): keep refreshing.
        assert!(!reloaded_credentials_fresh(
            Some(&observed),
            &observed
        ));
        // Nothing was stored when the caller looked: keep refreshing.
        assert!(!reloaded_credentials_fresh(None, &rotated_unexpired));
    }
}
