# Changelog

## 0.1.0 — 2026-10-20

- Initial release: full Rust port of meshoptimizer v1.3 (pin `9e1f07b1`), 84 public entry points
- Outputs byte-identical to upstream's scalar profile, pinned by 33 zero-tolerance golden fixtures
- Every public function documents its upstream counterpart; API mirrors upstream 1:1
- Optional `simd` feature: SSSE3+POPCNT / AVX512-VBMI2, NEON, and WASM SIMD128 vertex-decode kernels
- `no_std` + `alloc` capable; the default `std` feature only selects the math backend
- Hostile-input hardening: every-prefix truncation and bit-flip suites on all four decoders
- Library code is `forbid(unsafe_code)` by default; 5 audited `unsafe` sites under the `simd` feature, Miri-audited
- MSRV 1.89 (stable AVX512 intrinsics)
