# Self-Dev Fork Runbook

How to run jcode from your own checkout (personal fork branch) and ship your
changes into the shared daemon.

## How the pieces fit

| Piece | Path | Role |
|---|---|---|
| Launcher | `~/.local/bin/jcode` | Symlink → `~/.jcode/builds/current/jcode` |
| `current` channel | `~/.jcode/builds/current/` | Points at the newest self-dev build |
| Immutable builds | `~/.jcode/builds/versions/<git-hash>-dirty-<hash>/jcode` | Never overwritten; one per build |
| Shared daemon | `jcode serve` | Serves all sessions; must be reloaded to pick up a new build |

Source-tree resolution for `jcode self-dev --build`
(`crates/jcode-build-support/src/paths.rs` -> `get_repo_dir()`), in order:

1. `$JCODE_REPO_DIR` env var
2. The directory **baked into the installed binary at compile time**
3. Upward search from the current working directory

The baked-in path is why builds can silently come from the wrong checkout
(e.g. `~/PycharmProjects/jcode-integration`). Always confirm from the compile
log lines: `Compiling ... /Users/<you>/PycharmProjects/<which>/...`.

## Standard rebuild loop

```bash
cd ~/PycharmProjects/jcode        # your fork checkout, on your branch

# 1. Commit first — committed hash = reproducible build label
git add -p && git commit

# 2. Build + publish + repoint current (first time: pin the tree explicitly)
JCODE_REPO_DIR=$HOME/PycharmProjects/jcode jcode self-dev --build

# 3. If sessions still behave like the old code:
jcode server reload            # usually enough
jcode server reload --force    # if reload false-negatives through symlinks
```

Verify what is actually running:

```bash
readlink ~/.local/bin/jcode                       # -> builds/current/jcode
cat ~/.jcode/builds/current-version               # published version label
pgrep -fl "jcode.*serve"                          # daemon pid
lsof -p <daemon-pid> | awk '$4=="txt" && /versions/ {print $NF}'   # real inode
```

If `lsof` shows an older `versions/<hash>` than `current-version`, the daemon
is running stale code — use `server reload --force`.

## Verify a behavior change end-to-end

`cargo build` proves nothing. Exercise the changed path:

```bash
# Quick smoke through the shared daemon:
jcode run --no-update 'Reply with exactly: OK'

# Or isolated (does not touch the shared daemon):
cargo build --profile selfdev
./target/selfdev/jcode run --no-update --socket "$TMPDIR/jcode-test.sock" '<prompt>'

# Runtime logs (look for "Applied default model ...", provider init lines):
tail ~/.jcode/logs/jcode-$(date +%F).log
```

Instrument with `eprintln!`, never `crate::logging::info` (it writes to the log
file, not stderr). Delete before committing.

## Gotchas

- **Wrong tree built:** if the compile log shows another checkout, stop and
  rerun with `JCODE_REPO_DIR=<your fork>`. After your first successful install,
  the baked-in path becomes your tree and plain `jcode self-dev --build` works.
- **Dirty-tree builds work but are unlabelled** — you cannot tell later what
  shipped. Commit before building.
- **`jcode update` pulls upstream over `current`.** Rebase your branch onto
  main, then rerun `self-dev --build`.
- **LuLu firewall:** each build lands at a new `versions/<hash>/jcode` path, so
  LuLu treats it as a brand-new app. When prompted, choose Allow *permanently*
  for `~/.jcode/builds/current/jcode`; expect one prompt per rebuild.
- **`server stop` kills live headless/swarm sessions.** Prefer `server reload`;
  use `stop --force` only for a wedged daemon.

## Reference session (2026-08-22, xai-oauth persistence fix)

Symptom: restart lost the SuperGrok route; model switch failed with
"GitHub Copilot credentials not available".

Root cause: the xai-oauth route identity (`xai-oauth:` prefix, api_method
`xai-oauth-responses`) was missing from every persistence/restore vocabulary,
so restore fell back to whichever provider was active.

Fix locations:

- `crates/jcode-base/src/provider/mod.rs` — subscription-runtime vocabulary
  helpers, `set_config_default_model` routing, bare-id ownership fallback,
  fork spec preservation.
- `crates/jcode-base/src/provider/selection.rs` — session provider-key parsing,
  `model_switch_request_for_session_model/_route`,
  `default_model_selection_from_route`.
- `crates/jcode-base/src/provider/startup.rs` — no spurious warning for known
  subscription keys.

First `self-dev --build` attempt compiled `~/PycharmProjects/jcode-integration`
(branch without the fix); stopped it, reran with `JCODE_REPO_DIR`. Daemon kept
the old inode until `server reload --force`. Verified with a one-shot turn plus
log lines `Initialized SuperGrok provider from cached login` /
`Applied default model 'grok-4.6' from config`.
