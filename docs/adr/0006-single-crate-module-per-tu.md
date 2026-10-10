# ADR-0006: Single crate, one Rust module per upstream translation unit

- Status: Accepted
- Date: 2026-10-10

## Context

Upstream v1.3 `src/` holds 21 implementation files (15,665 lines of C++) with a clean internal
layering: `rasterizer.cpp` and `meshletutils.cpp` are shared internals with no public API; the
rest map to disjoint public-function families. A premature workspace split (e.g.
`meshopt-codec`, `meshopt-geo`) would duplicate shared internals across crates or force a
`core` crate now, before the real seams are proven by working code.

## Decision

One crate (`meshoptimizer-rs`), modules mirroring the upstream file layout:

```
src/
  lib.rs               crate root: no_std, alloc, module wiring, re-exports
  math.rs              internal numeric wrappers over libm (ADR-0004)
  rasterizer.rs        internal software rasterizer (pub(crate)) — used by overdraw/opacity map
  meshletutils.rs      internal meshlet helpers (pub(crate)) — used by meshlet codec
  quantization.rs      indexgenerator.rs        vcacheoptimizer.rs
  overdrawoptimizer.rs vfetchoptimizer.rs       indexcodec.rs
  vertexcodec.rs       vertexfilter.rs          meshletcodec.rs
  clusterizer.rs       partition.rs             spatialorder.rs
  stripifier.rs        indexanalyzer.rs         simplifier.rs
  opacitymap.rs        tangentspace.rs          remesher.rs
```

- Public items are re-exported flat at the crate root (matching the flat C header).
- `allocator.cpp` has no Rust counterpart (ADR-0005).

Splitting seam for the future: the internal modules (`math`, `rasterizer`, `meshletutils`) are
exactly what would move to a shared core crate if a workspace split ever happens. Revisit only
if compile times or API segmentation actually demand it (projected trigger: the port exceeding
~20k lines of Rust).

## Consequences

- No version-skew or dependency-cycle problems between sub-crates in phase 1.
- Per-file size discipline still applies (AGENTS.md): any file drifting past ~1000 lines is
  split along cohesive seams (e.g. `simplifier.cpp`, 2,947 lines upstream, is expected to split
  into adjacency/quadric/collapse-refinement submodules or a `simplifier/` directory).
- Cargo feature surface stays minimal; no inter-crate feature-forwarding chains to maintain.
