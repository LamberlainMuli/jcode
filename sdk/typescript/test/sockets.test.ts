import { test } from "node:test";
import assert from "node:assert";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { runtimeDir, apiSocketPath } from "../src/sockets.ts";

/**
 * Mirrors the Rust tests in jcode-storage and jcode-harness-api: containers can
 * inherit a set-but-missing XDG_RUNTIME_DIR, and it must not be trusted.
 */

test("runtimeDir uses an existing XDG_RUNTIME_DIR", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "jcode-xdg-"));
  const prev = process.env.XDG_RUNTIME_DIR;
  const prevJcode = process.env.JCODE_RUNTIME_DIR;
  delete process.env.JCODE_RUNTIME_DIR;
  process.env.XDG_RUNTIME_DIR = dir;
  try {
    assert.equal(runtimeDir(), dir);
    assert.match(apiSocketPath(), new RegExp(`^${dir}/jcode-api\\.sock$`));
  } finally {
    if (prev === undefined) delete process.env.XDG_RUNTIME_DIR;
    else process.env.XDG_RUNTIME_DIR = prev;
    if (prevJcode === undefined) delete process.env.JCODE_RUNTIME_DIR;
    else process.env.JCODE_RUNTIME_DIR = prevJcode;
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("runtimeDir falls back when XDG_RUNTIME_DIR does not exist", () => {
  const missing = path.join(os.tmpdir(), `jcode-missing-${process.pid}`);
  const prev = process.env.XDG_RUNTIME_DIR;
  const prevJcode = process.env.JCODE_RUNTIME_DIR;
  delete process.env.JCODE_RUNTIME_DIR;
  process.env.XDG_RUNTIME_DIR = missing;
  try {
    const result = runtimeDir();
    assert.notEqual(result, missing);
    // The fallback directory is not manufactured during resolution (the
    // bridge/server creates it when binding), so assert the resolved
    // location rather than on-disk existence.
    assert.match(result, /jcode-/);
  } finally {
    if (prev === undefined) delete process.env.XDG_RUNTIME_DIR;
    else process.env.XDG_RUNTIME_DIR = prev;
    if (prevJcode === undefined) delete process.env.JCODE_RUNTIME_DIR;
    else process.env.JCODE_RUNTIME_DIR = prevJcode;
  }
});

test("runtimeDir falls back when XDG_RUNTIME_DIR points at a file", () => {
  const file = path.join(
    fs.mkdtempSync(path.join(os.tmpdir(), "jcode-xdg-file-")),
    "not-a-dir",
  );
  fs.writeFileSync(file, "");
  const prev = process.env.XDG_RUNTIME_DIR;
  const prevJcode = process.env.JCODE_RUNTIME_DIR;
  delete process.env.JCODE_RUNTIME_DIR;
  process.env.XDG_RUNTIME_DIR = file;
  try {
    const result = runtimeDir();
    assert.notEqual(result, file);
    assert.notEqual(result, file);
    assert.match(result, /jcode-/);
  } finally {
    if (prev === undefined) delete process.env.XDG_RUNTIME_DIR;
    else process.env.XDG_RUNTIME_DIR = prev;
    if (prevJcode === undefined) delete process.env.JCODE_RUNTIME_DIR;
    else process.env.JCODE_RUNTIME_DIR = prevJcode;
  }
});

test("runtimeDir creates a missing JCODE_RUNTIME_DIR", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "jcode-explicit-"));
  const missing = path.join(root, "nested", "runtime");
  const prev = process.env.JCODE_RUNTIME_DIR;
  const prevXdg = process.env.XDG_RUNTIME_DIR;
  delete process.env.XDG_RUNTIME_DIR;
  process.env.JCODE_RUNTIME_DIR = missing;
  try {
    assert.equal(runtimeDir(), missing);
    assert.ok(fs.statSync(missing).isDirectory());
  } finally {
    if (prev === undefined) delete process.env.JCODE_RUNTIME_DIR;
    else process.env.JCODE_RUNTIME_DIR = prev;
    if (prevXdg === undefined) delete process.env.XDG_RUNTIME_DIR;
    else process.env.XDG_RUNTIME_DIR = prevXdg;
    fs.rmSync(root, { recursive: true, force: true });
  }
});
