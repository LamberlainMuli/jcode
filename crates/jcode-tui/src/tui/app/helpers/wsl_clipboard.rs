//! Windows-clipboard image fallback for WSL.
//!
//! WSLg syncs text between the Windows clipboard and the Wayland/X11
//! clipboards, but it does not sync images: after `Win+Shift+S`, `wl-paste
//! --list-types` only advertises text targets and `arboard` sees nothing.
//! The Windows clipboard itself does hold the image, so when jcode runs
//! inside WSL and every Linux clipboard backend came up empty, ask Windows
//! directly through PowerShell and stream the PNG back as base64 on stdout.
//!
//! Constraints this module is written against:
//!
//! * Only runs under WSL (see [`is_wsl`]); never on native Linux, macOS, or
//!   Windows, where the existing clipboard backends already work.
//! * `powershell.exe` is not necessarily on `$PATH` in WSL, so resolution
//!   falls back to the standard Windows PowerShell 5.1 location. Nothing is
//!   hardcoded per-user and `/mnt/c/Users` is never scanned.
//! * The clipboard image is streamed as base64 on stdout (Option A): no
//!   Windows temp files, no `wslpath` conversion, no filename collisions.
//! * Everything is a plain argument array; no shell string is ever built.
//! * Called only from the clipboard paste path (off the UI thread, inside
//!   `spawn_blocking_or_thread`), never per keystroke.
//! * Any failure returns `None` and the caller keeps its existing behavior.

use std::path::PathBuf;

/// Whether this process is running inside the Windows Subsystem for Linux.
///
/// Mirrors the signals used by `jcode_app_core::perf::detect_wsl`: the
/// WSL-specific environment variables first (fast, reliable when present),
/// then the `microsoft`/`WSL` kernel marker in `/proc/version` for distros
/// that scrub their environment.
pub(super) fn is_wsl() -> bool {
    is_wsl_from(
        std::env::var_os("WSL_DISTRO_NAME"),
        std::env::var_os("WSL_INTEROP"),
        std::env::var_os("WSLENV"),
        read_proc_version(),
    )
}

fn read_proc_version() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/version").ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Pure form of [`is_wsl`] so tests can exercise every signal combination
/// without depending on (or mutating) the host environment.
fn is_wsl_from(
    distro_name: Option<std::ffi::OsString>,
    interop: Option<std::ffi::OsString>,
    wslenv: Option<std::ffi::OsString>,
    proc_version: Option<String>,
) -> bool {
    if distro_name.is_some() || interop.is_some() || wslenv.is_some() {
        return true;
    }
    match proc_version {
        Some(v) => {
            let lower = v.to_ascii_lowercase();
            lower.contains("microsoft") || lower.contains("wsl")
        }
        None => false,
    }
}

/// PowerShell executables worth trying, most preferred first.
///
/// `powershell.exe` / `pwsh.exe` resolve through `$PATH` (WSL usually appends
/// `/mnt/c/Windows/System32` and friends via interop); the absolute Windows
/// PowerShell 5.1 path covers setups where interop PATH inheritance is
/// disabled (the reported environment: `which powershell.exe` finds nothing
/// but `/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe`
/// works). Nothing here depends on the Windows username.
const POWERSHELL_CANDIDATES: [&str; 3] = [
    "powershell.exe",
    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
    "/mnt/c/Program Files/PowerShell/7/pwsh.exe",
];

/// All PowerShell executables to attempt, most preferred first.
///
/// Several are returned rather than one because a bare `powershell.exe` can
/// pass resolution yet fail at spawn time when it is not on `$PATH` at all:
/// the caller then moves on to the absolute interop paths instead of giving
/// up (that combination is exactly the reported WSL setup). An empty list
/// means no candidate is plausible, which skips the fallback entirely
/// (plain Linux without interop, for example).
fn resolve_powershell() -> Vec<PathBuf> {
    resolve_powershell_with(
        POWERSHELL_CANDIDATES
            .iter()
            .map(|s| s.to_string())
            .collect(),
    )
}

fn resolve_powershell_with(candidates: Vec<String>) -> Vec<PathBuf> {
    candidates
        .into_iter()
        .map(PathBuf::from)
        .filter(is_runnable)
        .collect()
}

/// `true` when `path` names a file we can execute. A bare name like
/// `powershell.exe` is resolved through `$PATH` by the spawn itself, so
/// those are always worth trying (there is no file to stat, and `is_file`
/// on a relative name would only check the CWD); absolute interop paths
/// must exist and carry an executable bit, since a Windows drive mounted
/// with metadata can hold non-executable files.
fn is_runnable(path: &PathBuf) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if path.is_absolute() {
            return std::fs::metadata(path)
                .map(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
                .unwrap_or(false);
        }
    }
    // Bare program names go through PATH lookup at spawn time; there is no
    // file to stat here, so the candidate is always worth trying.
    true
}

/// Read an image from the Windows clipboard as `(media_type, base64_data)`.
///
/// `None` means "no image / fallback failed", which the caller treats
/// exactly like an empty Linux clipboard. Errors are logged, never surfaced:
/// a missing PowerShell, a text-only clipboard, or a malformed script must
/// not break pasting.
pub(super) fn windows_clipboard_image() -> Option<(String, String)> {
    windows_clipboard_image_with(is_wsl, resolve_powershell, run_powershell_clipboard_query)
}

/// Injectable composition so tests can pin the fallback-selection contract
/// without spawning real processes: WSL detection is consulted first and, when
/// it fails, not even PowerShell resolution runs (that is the "never invoke
/// PowerShell on plain Linux" guarantee).
fn windows_clipboard_image_with<IsWsl, Resolve, Run>(
    is_wsl: IsWsl,
    resolve: Resolve,
    run: Run,
) -> Option<(String, String)>
where
    IsWsl: Fn() -> bool,
    Resolve: Fn() -> Vec<PathBuf>,
    Run: Fn(&PathBuf) -> Option<String>,
{
    if !is_wsl() {
        return None;
    }
    resolve()
        .iter()
        .filter_map(|powershell| {
            run(powershell).and_then(|stdout| image_from_powershell_output(&stdout))
        })
        .map(|base64| ("image/png".to_string(), base64))
        .next()
}

/// The clipboard query, split out so tests can pin the exact argument array.
/// Every token is a literal; user data never reaches this command line.
fn powershell_args() -> Vec<String> {
    [
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        POWERSHELL_SCRIPT,
    ]
    .map(String::from)
    .to_vec()
}

/// Saves the clipboard image as PNG to a memory stream and prints it as
/// base64, or prints `NONE` when the clipboard has no image.
///
/// `[Console]::OutputEncoding` is forced to UTF-8: when PowerShell is
/// spawned from a non-console parent (as WSL interop does) its default
/// output encoding is ASCII-ish OEM codepage, which mangles base64 output.
/// Errors are caught so a failure prints `NONE` instead of stack traces on
/// stdout.
const POWERSHELL_SCRIPT: &str = concat!(
    "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; ",
    "try { ",
    "Add-Type -AssemblyName System.Windows.Forms; ",
    "Add-Type -AssemblyName System.Drawing; ",
    "$img=[Windows.Forms.Clipboard]::GetImage(); ",
    "if ($null -eq $img) { Write-Output 'NONE' } ",
    "else { ",
    "$ms=New-Object System.IO.MemoryStream; ",
    "$img.Save($ms,[System.Drawing.Imaging.ImageFormat]::Png); ",
    "Write-Output ([Convert]::ToBase64String($ms.ToArray())) ",
    "} ",
    "} catch { Write-Output 'NONE' }"
);

fn run_powershell_clipboard_query(powershell: &PathBuf) -> Option<String> {
    let output = std::process::Command::new(powershell)
        .args(powershell_args())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .inspect_err(|error| {
            crate::logging::info(&format!(
                "wsl clipboard: failed to run {}: {}",
                powershell.display(),
                error
            ));
        })
        .ok()?;

    if !output.status.success() {
        crate::logging::info(&format!(
            "wsl clipboard: powershell exited with {:?}",
            output.status.code()
        ));
        return None;
    }

    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Turn PowerShell stdout into base64 PNG data, rejecting every malformed
/// shape: the `NONE` sentinel, stray console noise, non-base64 characters
/// (including UTF-8 BOM or codepage-mangled letters), and empty output.
fn image_from_powershell_output(stdout: &str) -> Option<String> {
    let line = stdout
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    if line == "NONE" {
        return None;
    }
    if line.len() < 64 || !line.bytes().all(is_base64_byte) {
        crate::logging::info(&format!(
            "wsl clipboard: unrecognized powershell output ({} bytes)",
            stdout.len()
        ));
        return None;
    }
    Some(line.to_string())
}

fn is_base64_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'+' || byte == b'/' || byte == b'='
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(value: &str) -> Option<std::ffi::OsString> {
        Some(std::ffi::OsString::from(value))
    }

    #[test]
    fn wsl_detection_accepts_each_signal_alone() {
        assert!(is_wsl_from(
            env("Ubuntu"),
            None,
            None,
            Some("Linux version 5.15".into())
        ));
        assert!(is_wsl_from(
            None,
            env("/run/WSL/1"),
            None,
            Some("Linux version 5.15".into())
        ));
        assert!(is_wsl_from(
            None,
            None,
            env("WT_SESSION"),
            Some("Linux version 5.15".into())
        ));
        assert!(is_wsl_from(
            None,
            None,
            None,
            Some("Linux version 6.18.33.2-microsoft-standard-WSL2".into())
        ));
    }

    #[test]
    fn wsl_detection_rejects_native_linux() {
        assert!(!is_wsl_from(
            None,
            None,
            None,
            Some("Linux version 6.12 arch".into())
        ));
        assert!(!is_wsl_from(None, None, None, None));
    }

    #[test]
    fn wsl_detection_matches_case_insensitive_kernel_marker() {
        assert!(is_wsl_from(
            None,
            None,
            None,
            Some("Linux version 5.15.153.1-microsoft-standard-WSL2".into())
        ));
        assert!(is_wsl_from(
            None,
            None,
            None,
            Some("Linux version ...WSL2".into())
        ));
    }

    #[test]
    fn powershell_args_are_literal_and_hidden() {
        let args = powershell_args();
        assert_eq!(
            args[..4],
            [
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass"
            ]
        );
        assert_eq!(args[4], "-Command");
        assert!(args[5].contains("Clipboard]::GetImage()"));
    }

    #[test]
    fn base64_output_is_accepted() {
        let b64 = "A".repeat(100);
        assert_eq!(image_from_powershell_output(&b64), Some(b64.clone()));
        assert_eq!(
            image_from_powershell_output(&format!("{b64}\r\n")),
            Some(b64.clone())
        );
        assert_eq!(
            image_from_powershell_output(&format!("  {b64}  ")),
            Some(b64)
        );
    }

    #[test]
    fn none_sentinel_is_not_an_image() {
        assert_eq!(image_from_powershell_output("NONE"), None);
        assert_eq!(image_from_powershell_output("NONE\r\n"), None);
    }

    #[test]
    fn malformed_output_is_rejected() {
        assert_eq!(image_from_powershell_output(""), None);
        assert_eq!(image_from_powershell_output("\n\n"), None);
        assert_eq!(image_from_powershell_output("short"), None);
        // Codepage-mangled base64 must never become a corrupted image.
        assert_eq!(
            image_from_powershell_output(&format!("{}é", "A".repeat(100))),
            None
        );
        assert_eq!(
            image_from_powershell_output(&format!("Add-Type : error\r\n{}", "A".repeat(100))),
            None
        );
    }

    #[test]
    fn resolution_skips_missing_candidates() {
        let missing = "/nonexistent/jcode-powershell-test".to_string();
        assert_eq!(
            resolve_powershell_with(vec![missing]),
            Vec::<PathBuf>::new()
        );

        // A non-executable file (mode 0o644) under an absolute path must be
        // skipped: Windows drives mounted with metadata can carry those.
        let dir = tempfile::tempdir().unwrap();
        let inert = dir.path().join("not-executable.exe");
        std::fs::write(&inert, b"MZ fake").unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&inert, std::fs::Permissions::from_mode(0o644)).unwrap();
        let runnable = dir.path().join("executable.exe");
        std::fs::write(&runnable, b"MZ fake").unwrap();
        std::fs::set_permissions(&runnable, std::fs::Permissions::from_mode(0o755)).unwrap();

        assert_eq!(
            resolve_powershell_with(vec![
                inert.to_string_lossy().into_owned(),
                runnable.to_string_lossy().into_owned()
            ]),
            vec![runnable]
        );
    }

    #[test]
    fn bare_name_candidates_are_always_worth_trying() {
        // `powershell.exe` with no slash is a PATH-lookup candidate; there is
        // no file to stat, so it must survive resolution even when nothing
        // named that exists on disk next to the process.
        assert_eq!(
            resolve_powershell_with(vec!["powershell.exe".to_string()]),
            vec![PathBuf::from("powershell.exe")]
        );
    }

    #[test]
    fn resolution_keeps_every_runnable_candidate_in_order() {
        // CI is not WSL, so assert the ordering contract against candidates
        // that definitely exist there: every runnable one is kept, in order.
        let first = "/bin/sh".to_string();
        let second = "/bin/cat".to_string();
        let missing = "/nonexistent/jcode-powershell-test".to_string();
        assert_eq!(
            resolve_powershell_with(vec![missing, first, second]),
            vec![PathBuf::from("/bin/sh"), PathBuf::from("/bin/cat")]
        );
    }

    #[test]
    fn native_linux_never_touches_powershell() {
        use std::cell::Cell;
        let spawned = Cell::new(0usize);
        let result = windows_clipboard_image_with(
            || false,
            || {
                spawned.set(spawned.get() + 1);
                vec![PathBuf::from("/definitely/not/invoked")]
            },
            |_| {
                spawned.set(spawned.get() + 1);
                Some("A".repeat(100))
            },
        );
        assert_eq!(result, None);
        assert_eq!(
            spawned.get(),
            0,
            "non-WSL hosts must not resolve or run PowerShell"
        );
    }

    #[test]
    fn wsl_without_any_powershell_candidate_is_not_an_image() {
        let result = windows_clipboard_image_with(|| true, Vec::new, |_| panic!("must not run"));
        assert_eq!(result, None);
    }

    #[test]
    fn wsl_tries_the_next_candidate_when_one_fails_to_spawn() {
        // The reported WSL setup: `powershell.exe` is not on PATH (spawn
        // fails) but the absolute interop path works. The fallback must move
        // on to the second candidate instead of giving up after the first.
        use std::cell::Cell;
        let attempted = Cell::new(0usize);
        let b64 = "iVBORw0KGgoAAAANSUhEUg".to_string() + &"A".repeat(80);
        let result = windows_clipboard_image_with(
            || true,
            || {
                vec![
                    PathBuf::from("powershell.exe"),
                    PathBuf::from("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"),
                ]
            },
            |program| {
                attempted.set(attempted.get() + 1);
                if program.as_os_str() == "powershell.exe" {
                    None // spawn failure: not on PATH
                } else {
                    Some(format!("{b64}\r\n"))
                }
            },
        );
        assert_eq!(attempted.get(), 2, "second candidate must be attempted");
        assert_eq!(result, Some(("image/png".to_string(), b64)));
    }

    #[test]
    fn wsl_with_windows_image_streams_png() {
        let b64 = "iVBORw0KGgoAAAANSUhEUg".to_string() + &"A".repeat(80);
        let result = windows_clipboard_image_with(
            || true,
            || {
                vec![PathBuf::from(
                    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                )]
            },
            |_| Some(format!("{b64}\r\n")),
        );
        assert_eq!(result, Some(("image/png".to_string(), b64)));
    }

    #[test]
    fn wsl_with_failing_powershell_falls_through() {
        for stdout in [None, Some(String::new()), Some("NONE".to_string())] {
            let result = windows_clipboard_image_with(
                || true,
                || vec![PathBuf::from("pwsh")],
                |_| stdout.clone(),
            );
            assert_eq!(result, None, "stdout={stdout:?}");
        }
    }
}
