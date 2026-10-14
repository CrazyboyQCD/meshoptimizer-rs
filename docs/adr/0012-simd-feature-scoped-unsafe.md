# ADR-0012: SIMD vertex-decode kernels behind an opt-in, unsafe-scoped feature

- Status: Accepted
- Date: 2026-10-13
- Amends: [ADR-0003](0003-forbid-unsafe-deny-panic.md) (crate-wide `forbid(unsafe_code)`
  becomes conditional on the default feature set)
- Refines: [ADR-0010](0010-scalar-profile-fixtures.md) (scalar profile remains the default
  build and the normative oracle; the vertex-codec SIMD decode is byte-identical to the
  scalar kernel and therefore representable), [SRS §6.1](../srs.md)

## Context

The benchmark harness (ADR-0011) measured a 6.9× gap on `decode_vertex_buffer` against
upstream's shipped SSE/AVX2 decode kernels — the single largest remaining gap; the scalar
median ratio is ~1.2× and every scalar-only optimization has been harvested (P2.6–P2.8).
Closing it requires explicit SIMD, and each prior option was rejected:

- `std::arch` intrinsics are `unsafe fn` — blocked by the crate-wide
  `forbid(unsafe_code)` (ADR-0003), reaffirmed 2026-10-13 for *unchecked bounds-check*
  use (that specific ADR-0012 proposal was rejected and is not revisited).
- `core::simd` (portable_simd) is fully safe but nightly-gated — conflicts with the stable
  MSRV 1.88 publish target.
- Third-party safe wrappers move the `unsafe` into dependencies — violates the
  zero-dependency design (libm only).

Upstream's vertex-codec SIMD decode is a bit-level reimplementation that is **byte-identical**
to its scalar kernel ([ADR-0010](0010-scalar-profile-fixtures.md)); this was re-confirmed
empirically when the harness's SIMD build passed the scalar output pins for `vc_decode`.
Upstream's SIMD *filters*, in contrast, are not output-identical (reciprocal-sqrt
approximations) and stay out of scope permanently.

## Decision

1. **Feature shape.** A `simd` cargo feature (`simd = ["std"]`, x86_64 targets only,
   not enabled by default). The runtime-detection macro is `std`-only, hence the feature
   implies `std`; `no_std` builds are unaffected. The SSE2 kernel needs no detection
   (baseline instruction set on x86_64); the AVX2 kernel is selected at runtime with the
   scalar kernel as fallback.

2. **Conditional safety gate.** The crate attribute becomes
   `#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]` with
   `#![deny(unsafe_code)]` otherwise. ADR-0003's machine-enforced guarantee therefore
   holds for **every default build**; `unsafe` exists and compiles only inside the
   feature-gated `vertexcodec` SIMD kernel module, where each `unsafe` block carries a
   safety comment referencing the upstream invariant it preserves.

3. **Kernel scope.** Only the vertex-codec *decode* kernels (`decodeVertex` SSE2/AVX2
   translation units of upstream `vertexcodec.cpp` at the pinned revision), ported
   line-for-line under the same C-parity discipline as the scalar port. Encoders stay
   scalar (upstream has no SIMD encode). Vertex filters stay scalar permanently — their
   SIMD variants are not output-identical (ADR-0010), and the crate's golden oracle is
   byte-exact.

4. **Verification.** The existing golden suite is the SIMD differential oracle:
   `cargo test --features simd` must pass byte-exactly. The gate keeps running in the
   default configuration (no `simd`), where the `forbid` remains absolute; a
   `--features simd` clippy/test pass is added to the SIMD workstream, not to the default
   gate. The bench harness measures `vc_decode` with and without the feature.

## Consequences

- The published default artifact is unchanged: no `unsafe` compiles, no behavioral delta,
  golden-byte-exact, `no_std` intact.
- Opting into `simd` is an explicit acknowledgment that the decode kernel module contains
  scoped `unsafe`; every other module remains forbidden even in that configuration.
- ADR-0003's "crate-wide, unconditional" wording is amended by this ADR; the red-line
  decision against `get_unchecked` bounds-check removal stands unchanged.
- Future SIMD candidates (encode, filters) would require a superseding ADR plus, for
  filters, a redefinition of the behavioral oracle (output is not bit-identical).

## Erratum (2026-10-14, implementation phase)

Factual and scope corrections recorded during the P2.9 implementation; the decision
itself (opt-in scoped-unsafe SIMD decode, scalar oracle) is unchanged. The original
text above is preserved verbatim.

1. **ISA facts.** Upstream's "SSE" decode kernel requires **SSSE3+POPCNT**
   (`pshufb`/`popcnt`), not SSE2; x86_64 baseline is SSE2 only. The second x86 kernel
   is **AVX512-VBMI2** (`SIMD_AVX` gate = VBMI+VBMI2+VL), not AVX2; upstream ships it
   only when compiled with AVX512 flags.
2. **Dispatch is compile-time (supersedes §1's detection language).** Per owner
   decision there is no runtime detection: `simd` assumes the target has
   SSSE3+POPCNT — the same contract as upstream compiled with `-mssse3 -mpopcnt`,
   which also has no cpuid fallback; running on lesser hardware faults.
   `simd-avx512` mirrors the `SIMD_AVX` compile gate and must only run on AVX512
   hardware.
3. **`simd` does not imply `std` (supersedes §1's `simd = ["std"]`).** The kernel is
   pure `core::arch`; the only `std` dependency was the detection macro, which is
   gone. `no_std` + `simd` builds are supported.
4. **MSRV 1.88 → 1.89.** The AVX-512 target features and intrinsics
   (`avx512f/bw/vl/vbmi/vbmi2`) are unstable in 1.88 (verified empirically:
   `stdarch_x86_avx512` E0658) and stabilized in Rust 1.89.0 ("Stabilize the avx512
   target features", PR #138940).
5. **Scalar decode kernels are compiled out under `simd`**, mirroring upstream's
   `#if defined(SIMD_FALLBACK) || no-SIMD` structure (`vertexcodec.cpp` L581): the
   scalar bitstream/`decodeDeltas1`/`decodeVertexBlock` kernels exist only in
   configurations that actually call them.
6. **Safety-gate mechanics (§2 detail).** The `[lints.rust] unsafe_code = "forbid"`
   table entry moved to `lib.rs`
   (`cfg_attr(not(simd) → forbid, simd → deny)`) because the lints table cannot be
   feature-conditional; the kernel module carries the crate's only
   `#[allow(unsafe_code)]`.

Measured result (P2.9): `vc_decode` Rust SIMD = **1.18×** upstream shipped-SIMD
(was 6.9× scalar; scalar-vs-scalar unchanged), outputs hash-identical across
scalar/SIMD builds; `cargo test --features simd` passes the full golden suite
byte-exactly, including the every-prefix hostile fixtures.
