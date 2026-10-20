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

## Erratum 1 (2026-10-19): single MIT instead of dual licensing

Supersedes the dual `MIT OR Apache-2.0` decision above (user ruling): the crate is
distributed under the **MIT** license only. `Cargo.toml` declares `license = "MIT"`, the
repository carries `LICENSE-MIT` (this crate) and `LICENSE-THIRD-PARTY` (upstream
attribution), and `LICENSE-APACHE` from the original plan is not used.

The attribution obligations are unchanged. Upstream meshoptimizer is MIT (Copyright
(c) 2016-2026 Arseny Kapoulkine, verified at the pin), its license text explicitly grants
sublicensing rights, so a downstream MIT-only license is lawful as long as the copyright
and permission notice are retained — which `LICENSE-THIRD-PARTY` and the crate root doc
comment provide. The Rust-ecosystem dual-licensing convention and the Apache-2.0 patent
grant documented above are consciously given up in favor of a single maximally simple
license.

## Erratum 2 (2026-10-20): crates.io name is `rmeshopt`

Resolves the working-name assumption above: `meshoptimizer-rs` on crates.io was taken by
another project (first published 2026-10-04), so the package ships as **`rmeshopt`**
(repository renamed accordingly; lib name `rmeshopt`). The licensing decision is unaffected.

