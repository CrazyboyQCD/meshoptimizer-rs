# ADR-0002: Dual configuration — default `std`, `no_std` + `alloc` via `--no-default-features`

- Status: Accepted
- Date: 2026-10-10

## Context

meshoptimizer consumers include game consoles, embedded/mobile renderers, and (via wasm)
browsers, so `no_std` capability is a hard requirement. The algorithms need growable scratch
buffers and returned `Vec`s, so a pure `core` library is not achievable. On the other hand,
host development and consumers on `std` platforms want the standard library's math
implementations (platform-optimized, exactly the reference upstream was tested against).

## Decision

The crate compiles in two configurations, selected by the `std` feature (enabled by default):

- **`std` (default):** crate links the standard library; the `math` wrappers
  ([ADR-0004](0004-numeric-via-libm.md)) call `f32`/`f64` methods (platform libm).
- **`--no-default-features`:** crate root applies `#![no_std]` via
  `#![cfg_attr(not(feature = "std"), no_std)]` plus `extern crate alloc;`; the `math` wrappers
  call `libm`. Library code uses only `core` and `alloc` items (imported from `core`/`alloc`
  paths, which work identically under both configurations).

**`no_std` proof without extra targets:** building with `--no-default-features` on the host is
a complete proof that library code uses no `std` items — with the attribute active, any `std`
path is a compile error. The gate therefore runs clippy with `--no-default-features` on the
host; no embedded target installation is required or used.

Unit tests run on host targets in both configurations; `#[cfg(test)] extern crate std;` makes
`std` nameable in test code.

## Consequences

- Both configurations are first-class: the full gate covers `--all-features` *and*
  `--no-default-features`.
- `ToString`/`ToOwned`/`Vec`/`String` are imported from `alloc` unconditionally — a known
  `no_std` pitfall when modules are written under `std` only.
- Under `std`, float math may vary across platforms by ≤1 ULP on non-correctly-rounded
  functions (ADR-0004); under `no_std` it is bit-stable everywhere via pure-Rust `libm`.
- If a future need requires `std`-only conveniences (`std::error::Error` impls), they are
  cfg'd on the existing `std` feature — no new feature.
