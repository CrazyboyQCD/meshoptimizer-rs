# Changelog

All notable changes to this crate are documented in this file, in the spirit of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.1] - 2026-10-21

### Changed

- Packaging: exclude the bench harness from the published package.

## [0.1.0] - 2026-10-20

### Added

- Initial public release: safe, `no_std`-capable Rust port of meshoptimizer v1.3 (pin `9e1f07b1`).
- Index processing: remap, welding, sparse filter, index analysis.
- Optimization: vertex cache, overdraw, vertex fetch, stripify, spatial sort, partition.
- Simplification: quadric, sloppy, points, prune.
- Clusterization: meshlets, bounds, KD-tree; meshlet codec.
- Compression: index/vertex codecs; SIMD decode (SSSE3/AVX512, NEON, WASM SIMD128).
- Vertex filters: oct/quat/exp/color; remesher; tangents, normals, opacity maps, quantization.
- Byte-identical to the upstream scalar profile; `no_std` + `alloc`; MSRV 1.89.
