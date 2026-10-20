# Changelog

All notable changes to this crate are documented in this file, in the spirit of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-10-20

### Added

- Initial public release: safe, `no_std`-capable Rust port of meshoptimizer v1.3 (pin `9e1f07b1`).
- Index processing: vertex remap, sparse remap/filter, welding, index analysis.
- Optimization: vertex cache, overdraw, vertex fetch, stripification, spatial sort, partition.
- Simplification: quadric edge collapse (with attributes), sloppy grid, point clouds, prune.
- Clusterization: meshlet building, bounds, KD-tree clustering; meshlet codec for the output.
- Compression: index/vertex codecs; SIMD decode (SSSE3+POPCNT/AVX512-VBMI2, NEON, WASM SIMD128).
- Vertex filters: octahedral, quaternion, exponential, YCoCg color — encode and decode.
- Remesher: voxelization, solidification, marching-cubes polygonization with attributes.
- Tangent space and normals; opacity maps; quantization helpers.
- Byte-identical outputs to upstream's scalar profile (33 golden fixtures); `no_std` + `alloc`; MSRV 1.89.
