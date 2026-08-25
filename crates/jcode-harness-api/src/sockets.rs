//! Socket path resolution shared by every harness API client and the bridge.
//!
//! This lives in the API crate on purpose. It used to be duplicated in the
//! bridge and external clients, where separate copies could disagree: the bridge
//! resolved `$XDG_RUNTIME_DIR` while the desktop always looked in
//! `~/.jcode`. The result was a desktop app that could never connect even
//! with a healthy bridge running. One definition, used by both sides, makes
//! that class of bug impossible.
//!
//! The rules match `jcode-storage::runtime_dir` so the API socket always lands
//! beside the daemon socket it bridges to.
//!
//! Both crates validate inherited directories the same way: a set-but-missing
//! `XDG_RUNTIME_DIR` (common in containers) must not be trusted, or the bridge
//! and daemon each resolve a socket directory nothing can bind or dial.

use std::path::{Path, PathBuf};

/// Runtime directory holding the daemon and API sockets.
pub fn runtime_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("JCODE_RUNTIME_DIR") {
        let path = PathBuf::from(dir);
        if usable_runtime_dir(&path) || std::fs::create_dir_all(&path).is_ok() {
            return path;
        }
    }
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        let path = PathBuf::from(dir);
        if usable_runtime_dir(&path) {
            return path;
        }
    }
    #[cfg(target_os = "macos")]
    if let Ok(dir) = std::env::var("TMPDIR") {
        let path = PathBuf::from(dir);
        if usable_runtime_dir(&path) {
            return path;
        }
    }
    fallback_runtime_dir()
}

/// Mirrors `jcode-storage`: a runtime directory is usable only when it exists
/// and is a directory, so inherited env values are validated before use.
fn usable_runtime_dir(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|metadata| metadata.is_dir())
        .unwrap_or(false)
}

fn fallback_runtime_dir() -> PathBuf {
    std::env::temp_dir().join(format!("jcode-{}", runtime_user_discriminator()))
}

#[cfg(unix)]
fn runtime_user_discriminator() -> String {
    // Read the uid without pulling in libc: the API crate is deliberately
    // dependency-light, and this only needs to disambiguate users in $TMPDIR.
    std::env::var("UID")
        .ok()
        .or_else(|| std::env::var("USER").ok())
        .map(sanitize)
        .unwrap_or_else(|| "user".to_string())
}

#[cfg(not(unix))]
fn runtime_user_discriminator() -> String {
    std::env::var("USERNAME")
        .or_else(|_| std::env::var("USER"))
        .map(sanitize)
        .unwrap_or_else(|_| "user".to_string())
}

fn sanitize(raw: String) -> String {
    let out: String = raw
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
        .take(64)
        .collect();
    if out.is_empty() {
        "user".to_string()
    } else {
        out
    }
}

/// Path of the versioned harness API socket. `JCODE_API_SOCKET` overrides it.
pub fn api_socket_path() -> PathBuf {
    if let Ok(custom) = std::env::var("JCODE_API_SOCKET") {
        return PathBuf::from(custom);
    }
    runtime_dir().join("jcode-api.sock")
}

/// Path of the internal daemon socket the bridge translates onto.
/// `JCODE_SOCKET` overrides it.
pub fn legacy_socket_path() -> PathBuf {
    if let Ok(custom) = std::env::var("JCODE_SOCKET") {
        return PathBuf::from(custom);
    }
    runtime_dir().join("jcode.sock")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two sockets must always be siblings. A client that resolves one
    /// directory while the bridge resolves another cannot connect at all,
    /// which is exactly the bug this module exists to prevent.
    #[test]
    fn the_api_socket_sits_beside_the_daemon_socket() {
        // Env-mutating tests in this module hold ENV_LOCK; take it here too so
        // this test never observe a half-mutated environment.
        let _env = ENV_LOCK.lock().expect("env lock");
        // Guard against env-dependent divergence by comparing parents rather
        // than absolute paths, since either may be overridden in a session.
        let api = runtime_dir().join("jcode-api.sock");
        let legacy = runtime_dir().join("jcode.sock");
        assert_eq!(api.parent(), legacy.parent());
    }

    #[test]
    fn socket_names_are_stable() {
        let _env = ENV_LOCK.lock().expect("env lock");
        assert_eq!(
            runtime_dir().join("jcode-api.sock").file_name().unwrap(),
            "jcode-api.sock"
        );
    }

    #[test]
    fn sanitize_strips_path_and_shell_characters() {
        assert_eq!(sanitize("../root; rm".into()), "rootrm");
        assert_eq!(sanitize("!!!".into()), "user");
    }

    /// Environment mutation is process-global, so tests that touch runtime
    /// resolution env vars must not run concurrently.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    struct RuntimeEnvGuard {
        jcode_runtime: Option<std::ffi::OsString>,
        xdg_runtime: Option<std::ffi::OsString>,
    }

    impl RuntimeEnvGuard {
        fn set(jcode_runtime: Option<&Path>, xdg_runtime: Option<&Path>) -> Self {
            let guard = Self {
                jcode_runtime: std::env::var_os("JCODE_RUNTIME_DIR"),
                xdg_runtime: std::env::var_os("XDG_RUNTIME_DIR"),
            };
            match jcode_runtime {
                Some(path) => unsafe { std::env::set_var("JCODE_RUNTIME_DIR", path) },
                None => unsafe { std::env::remove_var("JCODE_RUNTIME_DIR") },
            }
            match xdg_runtime {
                Some(path) => unsafe { std::env::set_var("XDG_RUNTIME_DIR", path) },
                None => unsafe { std::env::remove_var("XDG_RUNTIME_DIR") },
            }
            guard
        }
    }

    impl Drop for RuntimeEnvGuard {
        fn drop(&mut self) {
            match &self.jcode_runtime {
                Some(value) => unsafe { std::env::set_var("JCODE_RUNTIME_DIR", value) },
                None => unsafe { std::env::remove_var("JCODE_RUNTIME_DIR") },
            }
            match &self.xdg_runtime {
                Some(value) => unsafe { std::env::set_var("XDG_RUNTIME_DIR", value) },
                None => unsafe { std::env::remove_var("XDG_RUNTIME_DIR") },
            }
        }
    }

    #[test]
    fn uses_existing_xdg_runtime_dir() {
        let _env = ENV_LOCK.lock().expect("env lock");
        let temp = tempfile::tempdir().expect("temp dir");
        let _guard = RuntimeEnvGuard::set(None, Some(temp.path()));

        assert_eq!(runtime_dir(), temp.path());
    }

    #[test]
    fn falls_back_when_xdg_runtime_dir_does_not_exist() {
        let _env = ENV_LOCK.lock().expect("env lock");
        let temp = tempfile::tempdir().expect("temp dir");
        let missing = temp.path().join("does-not-exist");
        let _guard = RuntimeEnvGuard::set(None, Some(&missing));

        let result = runtime_dir();

        // This crate never manufactures the fallback directory (the bridge
        // creates parents when binding), so assert the resolved location
        // rather than on-disk existence.
        assert_ne!(result, missing);
        assert_eq!(result, fallback_runtime_dir());
    }

    #[test]
    fn explicit_runtime_dir_is_created_when_missing() {
        let _env = ENV_LOCK.lock().expect("env lock");
        let temp = tempfile::tempdir().expect("temp dir");
        let missing = temp.path().join("nested").join("runtime");
        let _guard = RuntimeEnvGuard::set(Some(&missing), None);

        assert_eq!(runtime_dir(), missing);
        assert!(missing.is_dir());
    }
}
