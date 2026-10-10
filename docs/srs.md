# Software Requirements Specification — meshoptimizer-rs

- Version: 0.1 (2026-10-10)
- Status: Draft for review
- Upstream reference: [zeux/meshoptimizer](https://github.com/zeux/meshoptimizer) tag `v1.3`
  (commit `9e1f07b159d3cb777f1c67ed31fc11fd117986f4`) — see [ADR-0001](adr/0001-pin-upstream-v1.3.md)
- Related documents: [design.md](design.md) (how it is built), [TDD](tdd.md) (how it is proven,
  test-first), [ADR index](adr/) (why it is built this way)

## 1. Introduction

### 1.1 Purpose

This document specifies the functional and non-functional requirements for a from-scratch Rust
port of the meshoptimizer library (phase 1: full scalar implementation). It is the contract the
implementation and its tests are verified against.

### 1.2 Scope

**In scope (phase 1):** every public entry point of upstream v1.3 that has a Rust-portable
counterpart — 84 functions (82 stable + 2 experimental) across 18 functional groups (§3) — plus
the two internal support translation units those functions share.

**Out of scope (phase 1):** SIMD execution paths (SSE/AVX/NEON variants of the vertex codec and
filters), custom-allocator hooks, C++ STL adapter layer, big-endian targets, and `std`-only
conveniences. See §6.

### 1.3 Definitions

| Term | Meaning |
|---|---|
| index buffer | `&[u32]`, triply-repeating vertex indices forming triangles |
| vertex stream | raw vertex bytes plus a `stride` in bytes, as in upstream |
| wire format | byte-level encoding produced by the encoders; interoperability target |
| golden fixture | committed test vector generated from upstream at the pinned revision |

## 2. Overall description

A single `#![no_std]` library crate exposing mesh-optimization algorithms for GPU consumption:
vertex-cache/overdraw/fetch optimization, meshlet building and compression, index/vertex
compression codecs, simplification, stripification, spatial ordering, quantization, tangent
generation, and mesh analysis. Behavior is specified by upstream v1.3; where this document and
upstream appear to disagree, upstream v1.3 source wins and this document is corrected.

Design decisions and their rationale live in the ADR set; this document states *what* is
required, not *why*.

## 3. Functional requirements

Each FR group below requires: a public Rust API function for every listed upstream entry point
(names/signatures per [ADR-0005](adr/0005-c-parity-api.md)), behavior equivalent to upstream v1.3
(same algorithmic contract, same output), and differential tests per [ADR-0007](adr/0007-golden-vectors.md).

| ID | Group (upstream file) | Functions (upstream names, `meshopt_` prefix elided) | N |
|----|----------------------|--------------------------------------------------------|---|
| FR-01 | Vertex remapping & index generation (`indexgenerator.cpp`) | generateVertexRemap, generateVertexRemapMulti, generateVertexRemapCustom, remapVertexBuffer, remapIndexBuffer, filterIndexBuffer, filterIndexBufferMulti, generateShadowIndexBuffer, generateShadowIndexBufferMulti, generatePositionRemap, generateAdjacencyIndexBuffer, generateTessellationIndexBuffer, generateProvokingIndexBuffer | 13 |
| FR-02 | Vertex cache optimization (`vcacheoptimizer.cpp`) | optimizeVertexCache, optimizeVertexCacheStrip, optimizeVertexCacheFifo | 3 |
| FR-03 | Overdraw optimization (`overdrawoptimizer.cpp`) | optimizeOverdraw | 1 |
| FR-04 | Vertex fetch optimization (`vfetchoptimizer.cpp`) | optimizeVertexFetch, optimizeVertexFetchRemap | 2 |
| FR-05 | Index buffer codec (`indexcodec.cpp`) | encodeIndexBuffer, encodeIndexBufferBound, decodeIndexBuffer, decodeIndexVersion, encodeIndexSequence, encodeIndexSequenceBound, decodeIndexSequence | 7 |
| FR-06 | Meshlet codec (`meshletcodec.cpp`) | encodeMeshlet, encodeMeshletBound, decodeMeshlet, decodeMeshletRaw | 4 |
| FR-07 | Vertex buffer codec (`vertexcodec.cpp`) | encodeVertexBuffer, encodeVertexBufferBound, encodeVertexBufferLevel, decodeVertexBuffer, decodeVertexVersion | 5 |
| FR-08 | Vertex filters (`vertexfilter.cpp`) | decodeFilterOct, decodeFilterQuat, decodeFilterExp, decodeFilterColor, encodeFilterOct, encodeFilterQuat, encodeFilterExp, encodeFilterColor | 8 |
| FR-09 | Simplification (`simplifier.cpp`) | simplify, simplifyWithAttributes, simplifyWithUpdate, simplifySloppy, simplifyPrune, simplifyPoints, simplifyScale | 7 |
| FR-10 | Stripification (`stripifier.cpp`) | stripify, stripifyBound, unstripify, unstripifyBound | 4 |
| FR-11 | Mesh analysis (`indexanalyzer.cpp`) | analyzeVertexCache, analyzeVertexFetch, analyzeOverdraw, analyzeCoverage | 4 |
| FR-12 | Clusterization & meshlets (`clusterizer.cpp`) | buildMeshlets, buildMeshletsScan, buildMeshletsBound, buildMeshletsFlex, buildMeshletsSpatial, optimizeMeshlet, optimizeMeshletLevel, computeClusterBounds, computeMeshletBounds, computeSphereBounds, extractMeshletIndices | 11 |
| FR-13 | Cluster partitioning (`partition.cpp`) | partitionClusters | 1 |
| FR-14 | Spatial ordering (`spatialorder.cpp`) | spatialSortRemap, spatialSortTriangles, spatialClusterPoints | 3 |
| FR-15 | Opacity maps (`opacitymap.cpp`) | opacityMapMeasure, opacityMapRasterize, opacityMapEntrySize, opacityMapCompact | 4 |
| FR-16 | Tangent space (`tangentspace.cpp`) | generateTangents | 1 |
| FR-17 | Quantization (`quantization.cpp`) | quantizeHalf, quantizeFloat, dequantizeHalf, computePositionExponent | 4 |
| FR-18 | Experimental: normals & remeshing (`remesher.cpp`, `tangentspace.cpp`) | generateNormals, remesh | 2 |
| | **Total portable entry points** | | **84** |

Additional requirements per group:

- **FR-05, FR-07**: encoders emit the current wire version (v1) only; upstream's global version
  setters are not ported ([ADR-0009](adr/0009-encoder-version-hardcoded.md)). Decoders accept
  every version upstream produces (v0 and v1), and `decodeIndexVersion`/`decodeVertexVersion`
  stay available as version probes.

- **FR-03, FR-11** depend on the internal software rasterizer (`rasterizer.cpp`, 289 lines);
  it is ported as a `pub(crate)` module with the same rasterization order, since overdraw
  statistics and cluster bounds cones are order-sensitive.
- **FR-06** depends on internal meshlet helpers (`meshletutils.cpp`, 619 lines), ported as
  `pub(crate)`.
- **FR-18** functions are gated upstream behind `MESHOPTIMIZER_EXPERIMENTAL`; they are in scope
  for full coverage but may land last in the phase-1 sequence and may be feature-gated in the
  Rust crate if their upstream contract shifts before re-pin.
- **Excluded from porting** (2 stable/1 experimental entry points + STL layer, documented in
  [ADR-0005](adr/0005-c-parity-api.md)): `setAllocator`, `setAllocatorCache`,
  `meshopt_IndexAdapter`/`meshopt_Buffer` template machinery.

## 4. Non-functional requirements

| ID | Requirement | Verification |
|----|-------------|--------------|
| NFR-1 | `no_std` + `alloc` code paths when built with `--no-default-features` ([ADR-0002](adr/0002-no-std-alloc-baseline.md)) | `cargo clippy --no-default-features --all-targets -- -D warnings` (host build; the crate attribute turns any `std` usage into a compile error) |
| NFR-2 | 100% safe: `unsafe` forbidden crate-wide; no `panic!`/`unwrap`/`expect`/`todo!`/`unimplemented!`/`unreachable!` in library code ([ADR-0003](adr/0003-forbid-unsafe-deny-panic.md)) | `[lints]` table in `Cargo.toml` + `-D warnings` clippy gate |
| NFR-3 | Decoders never panic on arbitrary (including malformed) input; they return failure indicators | Code-review criterion per decode function; fuzz targets (stretch goal) |
| NFR-4 | Deterministic: identical inputs produce identical outputs on every platform, every run | Encoders byte-compared against golden fixtures on CI; no floating-point environment flags are touched |
| NFR-5 | Wire-format compatibility: every stream produced by upstream v1.3 decoders decodes here, and every stream produced here decodes upstream; encoders are byte-exact vs upstream | Golden fixtures ([ADR-0007](adr/0007-golden-vectors.md)) |
| NFR-6 | Numeric agreement: integer outputs exactly equal upstream; float outputs exact except ≤1 ULP on paths through non-correctly-rounded functions (`acos`, `log2`, `sin`, `cos`) | Differential tests; tolerance map in TDD §7 |
| NFR-7 | Lint-clean: `cargo clippy --all-targets --all-features -- -D warnings` passes; the `[lints]` table (including `std_instead_of_core`, `std_instead_of_alloc`, `alloc_instead_of_core`) stays intact | Verification gate (§5) |
| NFR-8 | Toolchain baseline: Rust 2024 edition, MSRV 1.88 (floor for `slice::as_chunks` and the project's modern-API style) | `rust-version` in `Cargo.toml`; gate runs on MSRV |
| NFR-9 | Documentation and comments in English; every public item references its upstream counterpart | Review criterion |
| NFR-10 | Source files stay under ~1000 lines; tests are colocated with implementation in `#[cfg(test)]` modules | Review criterion; enforced at module boundaries |
| NFR-11 | Performance: not a phase-1 gate. Informative target ≤2× upstream compiled with its scalar profile on the reference corpus; SIMD recovery is phase 2 | Deferred benchmark suite |

## 5. Acceptance (verification gates)

A change is accepted only when all of the following pass on the changed tree:

```
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo clippy --no-default-features --all-targets -- -D warnings   # no_std configuration (NFR-1)
cargo test
```

The commands are run exactly as written (same feature/target selection) — see AGENTS.md §Gate.

## 6. Out of scope (phase 1)

1. **SIMD paths** — upstream's SSE/AVX/NEON vertex-codec and filter kernels. Phase 1 is fully
   scalar; wire formats are identical, so SIMD can be added later behind runtime detection
   without API change (phase 2 candidate).
2. **Allocator customization** — `setAllocator`/`setAllocatorCache`; Rust's global allocator
   covers the use case.
3. **`std`-only conveniences** — the default `std` feature selects only the math backend
   ([ADR-0002](adr/0002-no-std-alloc-baseline.md)); APIs that would require `std` (e.g.
   `std::error::Error` impls) stay out of scope.
4. **Big-endian targets** — upstream codecs are developed and tested little-endian-first;
   parity work would lack a reference. Revisit on demand.
5. **Fuzzing as a release gate** — cargo-fuzz targets for decoders are a stretch goal; review
   criteria (NFR-3) are the phase-1 guarantee.

## 7. Traceability

- FR-01…FR-18 → module mapping: design.md §3.
- NFR-1/2/4/5/6 → decisions: ADR-0002/0003/0004/0007; verification method: TDD §1–§7.
- API shape and exclusions → ADR-0005/0009; crate layout → ADR-0006; licensing → ADR-0008.
