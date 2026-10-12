# ADR-0010: Golden fixtures are generated from upstream's scalar profile

- Status: Accepted
- Date: 2026-10-10
- Refines: [ADR-0007](0007-golden-vectors.md)

## Context

Upstream compiles SIMD (SSE/AVX/NEON/Wasm) kernels for vertex-codec decoding and vertex filters
by default on mainstream platforms. The SIMD *filter* kernels are not output-identical to the
scalar kernels: they use reciprocal-square-root approximations (`_mm_rsqrt_ps`, NEON `vrsqrte`)
and fast float→int renormalization tricks, so decoded filter output differs from the scalar
kernels — and differs between ISA implementations (SSE vs NEON vs Wasm). The vertex *codec*'s
SIMD decode is a bit-level reimplementation and produces byte-identical output to its scalar
kernel; only the filters diverge.

The phase-1 Rust port implements the scalar kernels only (SRS §6.1), and SRS NFR-4 requires
outputs to be deterministic across platforms while NFR-6 requires bit-identical float outputs on
the fixture platform. A fixture corpus generated from a SIMD build would pin ISA-specific
approximations that neither the port nor upstream itself reproduces across platforms.

## Decision

- `tests/fixtures/generate.cmd` compiles `gen_driver.cpp` with `/DMESHOPTIMIZER_NO_SIMD`, so
  every fixture is generated from upstream's documented scalar profile (the same profile the
  SRS NFR-11 informative performance target references).
- The scalar profile is the normative behavioral reference for this port. Encoder output, the
  index codec, and the vertex codec are unaffected by this choice (their scalar and SIMD paths
  already agree); vertex **filter decode** fixtures pin scalar-kernel behavior.
- Filter decode stays scalar in this crate permanently: porting the SIMD filter kernels (phase 2
  or later) would change filter outputs relative to this reference and is out of scope unless a
  superseding ADR redefines the reference profile.

## Consequences

- Fixture generation is reproducible on any x64 MSVC environment regardless of CPU SIMD
  capabilities, and the Rust scalar kernels have a well-defined oracle for every entry point.
- The SIMD dispatch layers of `vertexcodec.cpp`/`vertexfilter.cpp` are intentionally not
  represented in the fixtures; their byte-level control flow has no Rust counterpart in phase 1.
- If upstream ever changes scalar filter behavior, re-pinning updates fixtures and the tolerance
  notes together, per the normal [ADR-0007](0007-golden-vectors.md) workflow.
