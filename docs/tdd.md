# Test-Driven Development — meshoptimizer-rs

- Version: 0.1 (2026-10-10)
- Status: Draft for review
- Requirements: [SRS](srs.md) · Design: [design.md](design.md) · Decisions: [ADR index](adr/)

Development on this project is **test-driven**: the upstream implementation at the pinned
revision is the executable specification, expected outputs are frozen as fixtures *before* the
corresponding Rust code exists, and no implementation lands without a failing test that demanded
it. The acceptance criterion throughout is **output-identical to upstream — for everything that
serializes to bytes, literally byte-identical** ([ADR-0007](adr/0007-golden-vectors.md)).

## 1. The byte-identical guarantee

| Output class | Guarantee | Enforcement |
|---|---|---|
| Encoded streams (index, vertex, meshlet codecs; encode filters) | **byte-identical** to upstream v1.3 on every corpus input | hard gate; golden fixtures |
| Decoded outputs and accept/reject decisions | identical decoded bytes; accept every upstream-produced stream; reject malformed input the way upstream does | hard gate; fixtures + hostile-input suite |
| Integer-valued outputs (remaps, optimized index buffers, statistics counters, counts) | element-for-element identical | hard gate; golden fixtures |
| Float-valued outputs (bounds, result errors, tangents, scales) | bit-identical on the fixture-generation platform (`std` backend); ≤1 ULP only through `acos`/`log2`/`sin`/`cos` (§7) | tolerance-mapped; deviations need comment + ADR note |

"Identical" is always judged against upstream at the pinned revision
([ADR-0001](adr/0001-pin-upstream-v1.3.md)), on the committed corpus.

## 2. Why test-first

The correct behavior already exists — upstream compiles and runs. Porting first and testing
later forfeits the only objective oracle and turns review into opinion. Freezing expected
outputs *before* writing Rust code converts "is this a correct port?" into a mechanical check,
and makes every regression on the path to green a named, reproducible failure.

## 3. The red–green–refactor loop (per function)

1. **Red.** Write the colocated `#[cfg(test)]` test(s) asserting the §1 guarantee for one
   function, against a fixture or recorded upstream values. A neutral stub (`0` / `false` /
   identity) may exist to satisfy the compiler. `cargo test` must fail, with the failure naming
   the missing behavior.
2. **Green.** Port the upstream function — minimal, C-parity
   ([ADR-0005](adr/0005-c-parity-api.md)) — until the test passes.
3. **Refactor.** Idiomatic cleanups (modern slice APIs, iterator forms) are allowed only while
   outputs stay identical; the full gate re-runs after every refactor step.

Rules:

- No production code without a failing test that requires it.
- The red state must be reachable and demonstrable before the green implementation is written;
  a single commit may contain both, with the message noting fixture provenance.
- New upstream behavior discovered mid-port (undocumented quirks) becomes a fixture + test
  *first*, then code — never the other way around.

## 4. Test layers (cheapest first)

1. **Unit edge cases** (colocated, written first inside the loop): empty inputs, single-element
   inputs, contract preconditions (`index_count % 3 == 0` as `debug_assert!`), `stride` ≠ struct
   size, `*_bound()` tightness (encode into an exact-size buffer and succeed).
2. **Golden differential**: colocated tests comparing outputs to committed fixtures per §1.
   Every ported public entry point has at least one; FR↔test completeness is a review criterion.
3. **Round-trip**: `decode(encode(x)) == x` across the corpus (current encode version,
   [ADR-0009](adr/0009-encoder-version-hardcoded.md)); decode fixtures cover every upstream
   version (v0/v1); `decodeIndexVersion`/`decodeVertexVersion` probes agree with the stream.
4. **Hostile input**: decoders fed truncated streams at *every* prefix length, bit-flipped
   header/control fields, and empty buffers — assert `false`/`0` returns, never a panic
   (SRS NFR-3).
5. **Configuration gates**: the four-command gate (AGENTS.md §Gate), including the
   `--no-default-features` `no_std` proof ([ADR-0002](adr/0002-no-std-alloc-baseline.md)).

## 5. Fixture workflow (before implementation)

1. Build upstream at the pinned revision with its reference tooling (one-time per re-pin).
2. Run the committed corpus — cube/sphere/grid/subdivided meshes, degenerate triangles, point
   clouds, boundary counts and strides — through upstream's encode/optimize entry points; for
   v0 *decode* fixtures, drive upstream's version setter on the tooling side.
3. Emit fixtures as generated segment literals or binary files; commit the generator script
   beside them. Long vectors are never hand-typed ([ADR-0007](adr/0007-golden-vectors.md)).
4. Embed via `include_bytes!` (or generated literals) with `const _LEN_CHECK` length pins.
5. If a decoder fails a fixture: regenerate and re-verify the fixture against upstream *first*;
   only then suspect the decoder.

## 6. Coverage and evidence

- **Per-module process** follows design.md §9: fixtures and failing tests precede the module's
  implementation step; a module is done only when the full gate passes in both configurations.
- **Traceability**: each module's test module header lists the upstream entry points its tests
  cover; the SRS FR table is the checklist — an FR without a differential test is not done.
- **Regression discipline**: any output difference between two runs of *our* code is a bug
  (determinism, SRS NFR-4); any difference against *upstream* is either a fixture provenance
  error or a port bug — both block merge until classified.

## 7. Float tolerance map

| Path | Functions | Tolerance |
|---|---|---|
| All encoder/decoder bytes; all integer outputs | — | exact (byte/element equality) |
| Correctly-rounded math (`sqrt`, `sqrt_f64`, `floor*`, `ceil*`) | IEEE 754 exact functions | exact on every platform, both backends |
| Non-correctly-rounded math (`acos_f64`, `log2`, `sin_f64`, `cos`, `cos_f64`) — consumers: cluster/meshlet bounds cone, vertex codec size estimate | `libm` vs platform libm may differ in the last unit in the last place | bit-identical on the fixture platform (`std` backend, same libm as upstream); ≤1 ULP cross-platform and under `--no-default-features` |

Any tolerance looser than this table requires a justifying comment at the assertion and a note
in the superseding ADR ([ADR-0004](adr/0004-numeric-via-libm.md)).
