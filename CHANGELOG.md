# Changelog

All notable changes to this crate are documented in this file, in the spirit of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-10-20

### Added

- Initial public release: safe, `no_std`-capable Rust port of meshoptimizer v1.3 (pin `9e1f07b1`).
- Full upstream API: 84 public entry points, each documenting its upstream counterpart.
- Byte-identical outputs to upstream's scalar profile, pinned by 33 zero-tolerance golden fixtures.
- Optional `simd` feature: SSSE3+POPCNT / AVX512-VBMI2, NEON, and WASM SIMD128 vertex-decode kernels.
- Hostile-input hardening: every-prefix truncation and bit-flip suites on all four decoders.
- `no_std` + `alloc` capable; five Miri-audited `unsafe` sites behind the `simd` feature; MSRV 1.89.
