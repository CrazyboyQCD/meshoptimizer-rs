# ADR-0007: Byte-exact golden vectors generated from upstream; never hand-transcribed

- Status: Accepted
- Date: 2026-10-10

## Context

The codecs (index, vertex, meshlet) define wire formats that must interoperate with upstream
consumers/producers, and the optimizers must match upstream output to be provably correct
ports. Long test vectors hand-typed from a terminal are a proven failure mode: transcription
errors then masquerade as decoder bugs and burn debugging time on the wrong side.

## Decision

1. **Fixture provenance.** Every golden fixture is *generated* by building upstream at the
   pinned revision (ADR-0001) and running its encode/optimize entry points over a committed
   corpus of inputs (regular meshes, degenerate triangles, empty inputs, boundary sizes).
   The generating commands are documented next to the fixtures.
2. **No hand transcription.** Vectors longer than a handful of elements are never typed by hand
   or copied through a human's eyes. Fixture scripts emit Rust literals in short segments; a
   commit adding a fixture must contain the script output verbatim.
3. **Length pins.** Every embedded fixture carries a compile-time length assertion:
   `const _LEN_CHECK: usize = /* expected */; assert_eq!(FIXTURE.len(), _LEN_CHECK);` so a
   truncated or duplicated fixture fails loudly.
4. **Comparison strictness.**
   - Encoders: output must be **byte-identical** to upstream (hard gate).
   - Decoders: must accept every upstream-produced stream, reject malformed ones, and round-trip.
   - Integer-valued outputs (remaps, optimized index buffers, statistics counters):
     **exactly equal**.
   - Float-valued outputs (bounds, errors, tangents): exact where arithmetic is mirrored
     op-for-op; ≤1 ULP tolerance only along paths through `acos`/`log2`/`sin`/`cos` (ADR-0004);
     any larger tolerance must be justified in a comment and an ADR note.
5. **Fixture suspicion rule.** If a decoder fails a fixture, the fixture is re-verified against
   upstream *first* (regenerate, diff); the decoder is only then considered buggy. History shows
   the fixture is the more likely culprit.

## Consequences

- Fixture files are embedded via `include_bytes!` (a `core` macro) so `no_std` test builds keep
  working; large fixtures live in `tests/data/` with short segment-literal copies inside
  colocated test modules where binary files are impractical.
- Regenerating fixtures requires a C++ toolchain once, at re-pin time — not per-test-run.
- The corpus is version-controlled, so differential results are reproducible across machines.
