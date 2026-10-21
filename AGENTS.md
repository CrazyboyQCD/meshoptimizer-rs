# AGENTS.md — rmeshopt

Safe, deterministic, `no_std` Rust port of [meshoptimizer](https://github.com/zeux/meshoptimizer)
**v1.3** (pin: `9e1f07b1`, see `docs/adr/0001`). Phase 1: full scalar implementation.

Documents: [docs/srs.md](docs/srs.md) (requirements) · [docs/design.md](docs/design.md) (technical
design) · [docs/tdd.md](docs/tdd.md) (test-driven process) · [docs/adr/](docs/adr/) (decisions).
**Read the SRS, design, and TDD docs before the first port of each module.**

## Hard rules (non-negotiable)

1. **No `unsafe`, no `unwrap`, no `expect`, no `panic!`/`todo!`/`unimplemented!`/`unreachable!`**
   in library code. Enforced by the `[lints]` table in `Cargo.toml` — never weaken it without a
   new ADR.
2. **`no_std`-capable, `core`/`alloc` only in library code.** Default build uses the `std`
   feature (math backend only); `--no-default-features` must compile clean as `no_std` + `alloc`
   — [ADR-0002](docs/adr/0002-no-std-alloc-baseline.md). Import items from `core`/`alloc`
   paths so both configurations work.
3. **English** for all code, comments, docs, and commit messages.
4. **C parity, byte-identical output**: functions mirror upstream v1.3 1:1 (names, contracts) and
   outputs match upstream exactly — byte-identical for everything that serializes to bytes
   ([ADR-0005](docs/adr/0005-c-parity-api.md), [TDD §1](docs/tdd.md)). Every public fn's doc
   comment names its upstream counterpart. Do not "improve" algorithms while porting.
5. **Numeric kernels go through `src/math.rs`** (wrappers over `libm`), never hand-rolled
   approximations, never `std` float methods — [ADR-0004](docs/adr/0004-numeric-via-libm.md).
   Do not reassociate float expressions or substitute `mul_add`.

## Style

- Modern APIs first: `as_chunks`/`as_chunks_mut`, array types `[u8; N]`, slice pattern
  destructuring — replace legacy index-juggling idioms when porting.
- Files stay under ~1000 lines; split large ports along algorithmic seams (design.md §4).
  Tests are **colocated** in `#[cfg(test)] mod tests` inside the implementation file — never in
  separate files, never counted toward split gains.
- No comments except where the *why* is non-obvious (upstream-divergence notes, invariants,
  endianness/stride traps). Ported code should read like upstream with names Rust-ified.
- Test code within library files obeys the same lint rules as library code (no `unwrap`).
- Do not modify files via CLI text tools (sed/awk/shell redirects); use editor tools only.
- CHANGELOG.md house format (mirrors the bullet_physics repo): Keep a Changelog skeleton with
  `[Unreleased]` pinned at the top; release entries are terse per-subsystem keyword lists
  (`Module: a, b, c.`) under `### Added`/`### Changed` — no prose descriptions, no link blocks.

## Testing

- Golden fixtures are **generated from upstream v1.3**, never hand-typed; long vectors are
  committed as generated segment literals or `include_bytes!` files with `const _LEN_CHECK`
  length pins — [ADR-0007](docs/adr/0007-golden-vectors.md).
- Fixtures are **repository-only**: the published package excludes `tests/`, so `cargo test`
  from a crates.io tarball is not supported (verification lives in the repo/CI). Regular
  `cargo build`/`cargo check` of the published crate is unaffected.
- If a decoder fails a fixture: re-verify the fixture against upstream first; only then suspect
  the decoder.
- Strictness: encoders byte-exact; integer outputs exact; float outputs exact except
  `acos`/`log2`/`sin`/`cos` paths (≤1 ULP, tolerance justified in a comment).
- Every decoder gets hostile-input tests (truncated prefixes, bit-flips) asserting `false`/`0`
  with no panic.

## Gate (run exactly as written; all four must pass)

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo clippy --no-default-features --all-targets -- -D warnings   # no_std configuration proof
cargo test
```

- The `--no-default-features` run *is* the `no_std` proof: with the crate attribute active, any
  `std` usage in library code is a compile error. No extra rustup targets are involved.
- A change is not "done" until the full gate passes; do not ship with a skipped configuration.
- `cargo fix --all-targets` is not trusted (it can remove imports used by `#[cfg(test)]` code);
  apply fixes manually.

## Porting workflow (per module)

1. Read the upstream TU at the pinned revision and the matching SRS FR group.
2. **Red**: freeze fixtures and write the failing differential tests first (tdd.md §3/§5).
3. **Green**: port → colocate unit tests → make the differential tests pass, outputs
   byte-identical.
4. Run the full gate. Only then move to the next module.
5. Any deviation from upstream behavior or API shape requires an ADR *before* the code lands.
6. Scope changes (new upstream revision, added/removed functions) update the SRS + a superseding
   ADR in the same change.

## Layout

```
src/            one module per upstream TU (design.md §3); math.rs/rasterizer/meshletutils are pub(crate)
docs/           srs.md, design.md, tdd.md, adr/NNNN-*.md (numbered, immutable once Accepted; supersede, don't edit)
tests/data/     golden fixtures (generated; provenance documented beside them)
```
