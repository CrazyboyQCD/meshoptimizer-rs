# ADR-0008: Licensing — dual MIT OR Apache-2.0 with upstream attribution

- Status: Accepted
- Date: 2026-10-10

## Context

This port derives algorithmically from meshoptimizer, which is MIT-licensed. The code is
rewritten (not transliterated), but a port is a derivative work for copyright purposes; the
upstream MIT license therefore requires retaining its copyright and permission notice.
Rust ecosystem convention prefers `MIT OR Apache-2.0` dual licensing for maximally
uncomplicated consumption, and both are compatible with upstream MIT.

## Decision

- `Cargo.toml` declares `license = "MIT OR Apache-2.0"`.
- The repository carries `LICENSE-MIT` and `LICENSE-APACHE` for this crate's code, plus
  upstream's license text and copyright notice retained in `LICENSE-THIRD-PARTY` (added with the
  first ported module, never later — attribution must exist from the first published commit).
- Doc-level attribution: the crate root doc comment states the upstream project, pinned version
  (ADR-0001), and upstream license.
- The `crates.io` package name is verified at publish time; `meshoptimizer-rs` is the working
  name and first fallback. No code depends on the final name before then.

## Consequences

- Publishing to crates.io is unblocked without further legal review; contributors see the
  attribution chain from the repo root.
- If upstream's license ever changes on re-pin (ADR-0001), this ADR must be revisited before the
  next release.
