# ADR-0003: Forbid `unsafe`; deny panic/unwrap/expect primitives

- Status: Accepted
- Date: 2026-10-10

## Context

Phase-1 charter: a fully scalar, 100% safe implementation. `meshoptimizer`'s C++ code relies on
raw pointers, `memcpy`, and unchecked arithmetic internally; the Rust port must achieve the same
behavior with none of those tools. Decoders additionally face hostile input: upstream decoders
are hardened against malformed streams and must return a failure indicator, never crash.

## Decision

Enforced mechanically via `Cargo.toml` `[lints]` (do not weaken without a new ADR):

- `[lints.rust] unsafe_code = "forbid"` — no `unsafe` anywhere, tests included.
- `[lints.clippy]` at `deny`: `panic`, `unwrap_used`, `expect_used`, `todo`,
  `unimplemented`, `unreachable`.

Related policy decisions:

- **Allocation failure aborts.** Rust's default alloc error handler aborts on OOM. This is a
  documented divergence from upstream, whose C API returns 0/NULL on allocation failure. GPU mesh
  processing workloads do not treat OOM as recoverable; recovering would require `try_reserve`
  plumbing through every function and a `Result`-based API (deferred; see ADR-0005).
- **Contract violations use `debug_assert!`.** Upstream guards preconditions (e.g.
  `index_count % 3 == 0`) with `assert()`, which release builds compile out (`NDEBUG`). The port
  mirrors this with `debug_assert!`: release builds skip the check, and a violated contract can
  at worst produce wrong results — never UB, thanks to memory safety. This is strictly safer
  than upstream release builds.
- **Decoders never panic on arbitrary input.** All parsing of encoded buffers is bounds-checked
  with explicit `if` guards returning `false`/`0`. This is a hard review criterion for every
  `decode*` function, and a target for fuzzing (TDD §7).
- The lints above cannot catch slice-indexing panics; that residual risk is handled by review
  discipline (every indexed access in decode paths must be preceded by a length guard) and is
  tracked as fuzz-coverage work, not left to hope.

## Consequences

- `unwrap`/`expect` may not appear even in tests within library files; tests use explicit
  `assert!(x.is_some())` / `match` or small local helpers.
- Generic bounds-checks inside hot loops are written to hoist the length check out of the loop
  (upstream does the same with raw pointers), keeping the safety without a measurable cost.
