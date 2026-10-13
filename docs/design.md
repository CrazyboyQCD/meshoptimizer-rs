# Technical Design Document — meshoptimizer-rs

- Version: 0.2 (2026-10-10)
- Status: Draft for review
- Requirements: [SRS](srs.md) · Test process: [TDD](tdd.md) · Decisions: [ADR index](adr/)

## 1. Design goals, in priority order

1. **Provable correctness.** Every module can be diffed statement-by-statement against its
   upstream translation unit; outputs are identical to upstream — encoders byte-exact —
   enforced test-first ([ADR-0005](adr/0005-c-parity-api.md), [ADR-0007](adr/0007-golden-vectors.md),
   [TDD](tdd.md)).
2. **Total safety.** No `unsafe`, no panicking primitives in library code
   ([ADR-0003](adr/0003-forbid-unsafe-deny-panic.md)); decoders survive arbitrary input.
3. **Portability.** `no_std + alloc`, scalar-only, deterministic across platforms
   ([ADR-0002](adr/0002-no-std-alloc-baseline.md), [ADR-0004](adr/0004-numeric-via-libm.md)).
4. **Auditability beats ergonomics.** C-parity signatures first; convenience layers later.

## 2. Toolchain and dependencies

| Item | Value | Rationale |
|---|---|---|
| Edition | 2024 | current stable; modern idiom required by project style |
| MSRV | 1.88 | edition-2024 floor is 1.85; 1.88 adds `slice::as_chunks`, which the project style mandates |
| Features | `std` (default), `--no-default-features` = `no_std` | backend selection ([ADR-0002](adr/0002-no-std-alloc-baseline.md)) |
| Dependencies | `libm = "0.2"` only | `no_std` math backend ([ADR-0004](adr/0004-numeric-via-libm.md)) |
| Dev-dependencies | none planned for phase 1 | fixtures embedded via `include_bytes!` |
| Lints | `[lints]` table in `Cargo.toml` | mechanical enforcement of SRS NFR-1/2 |

## 3. Crate and module structure

One crate, one module per upstream translation unit ([ADR-0006](adr/0006-single-crate-module-per-tu.md)).
Module tree with upstream provenance and v1.3 size (C++ lines):

| Rust module | Upstream TU | LOC | SRS group | Notes |
|---|---|---|---|---|
| `math` *(internal)* | — | — | all | libm wrappers; see §5 |
| `rasterizer` *(internal)* | rasterizer.cpp | 289 | FR-03/11/15 | `pub(crate)`, fixed-point rasterizer; order-sensitive |
| `meshletutils` *(internal)* | meshletutils.cpp | 619 | FR-06 | `pub(crate)`, meshlet pack/unpack helpers |
| `quantization` | quantization.cpp | 127 | FR-17 | |
| `indexgenerator` | indexgenerator.cpp | 825 | FR-01 | hash remap, adjacency, tessellation |
| `vcacheoptimizer` | vcacheoptimizer.cpp | 467 | FR-02 | tipsify + strip + fifo variants |
| `overdrawoptimizer` | overdrawoptimizer.cpp | 333 | FR-03 | uses `rasterizer` |
| `vfetchoptimizer` | vfetchoptimizer.cpp | 71 | FR-04 | |
| `indexcodec` | indexcodec.cpp | 703 | FR-05 | v0/v1 dec, v1 enc, sequences |
| `vertexcodec` | vertexcodec.cpp | 1880 | FR-07 | v0/v1 dec, v1 enc; scalar kernels only |
| `vertexfilter` | vertexfilter.cpp | 1503 | FR-08 | oct/quat/exp/color filters |
| `meshletcodec` | meshletcodec.cpp | 1051 | FR-06 | uses `meshletutils` |
| `clusterizer` | clusterizer.cpp | 1377 | FR-12 | meshlets + bounds; likely splits (§4) |
| `partition` | partition.cpp | 621 | FR-13 | k-means partitioning |
| `spatialorder` | spatialorder.cpp | 341 | FR-14 | |
| `stripifier` | stripifier.cpp | 296 | FR-10 | |
| `indexanalyzer` | indexanalyzer.cpp | 125 | FR-11 | four statistics structs |
| `simplifier` | simplifier.cpp | 2947 | FR-09 | largest TU; expected to split (§4) |
| `opacitymap` | opacitymap.cpp | 557 | FR-15 | self-contained bilinear texture sampler |
| `tangentspace` | tangentspace.cpp | 730 | FR-16/18 | |
| `remesher` | remesher.cpp | 782 | FR-18 | experimental |

Public items are re-exported flat at the crate root. Internal modules stay `pub(crate)`;
they are the designated seam if a workspace split ever happens ([ADR-0006](adr/0006-single-crate-module-per-tu.md)).

## 4. File size discipline

Upstream's largest units exceed the project's ~1000-line source-file budget. Planned splits
(along algorithmic seams, tests stay colocated):

- `simplifier.rs` → `simplifier/` directory (decided during P1.6): `adjacency.rs`
  (edge/vertex adjacency + position remap + vertex classification), `quadric.rs`
  (quadric construction/evaluation/solve), `collapse.rs` (collapse
  selection/ranking/execution + fold quadrics + post-pass index/loop remapping),
  `solve.rs` (quadric position/attribute solving for `WithUpdate`), `simplify.rs`
  (collapse scheduling driver; the three thin public wrappers live in `mod.rs`
  to keep the driver file under budget), `sloppy.rs` (grid-based indexed-mesh
  simplification), `points.rs` (point-cloud simplification), `prune.rs`
  (connected-component pruning).
- `clusterizer.rs` → `clusterizer/` with `meshlets.rs` (build + flex machinery),
  `kdtree.rs` (kd-tree internals), `spatial.rs` (spatial builder + BVH),
  `optimize.rs` (optimize/extract helpers from `meshletutils.cpp`) and `bounds.rs`
  (cluster/meshlet/sphere bounds).
- `vertexcodec.rs` → `vertexcodec/` with `encode.rs`, `decode.rs`, shared `bitstream.rs`.

Split decisions are made when a file approaches the limit during its port, not preemptively.

## 5. Numeric layer (`math`)

Internal `pub(crate) mod math` wraps the backend; call sites never name `libm` or `std`
directly ([ADR-0004](adr/0004-numeric-via-libm.md)):

```rust
#[inline(always)]
pub(crate) fn sqrt(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        f32::sqrt(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrtf(x)
    }
}
// ... acos_f64, log2, sin_f64, cos, cos_f64, ceil, floor variants — same shape
```

Rules:

- Every `math` wrapper is `#[inline(always)]`: the backend selection must not cost a call.
- Core arithmetic (`+ - * /`, `min`, `max`, `abs`, comparisons, casts) is used directly.
- Mixed `f32`/`f64` usage mirrors upstream exactly (upstream computes cone axes in `f64` via
  `acos`/`sqrt`, quadric error in `f32`) — do not "fix" precision while porting.
- No reassociation, no `mul_add` substitutions: `a*b + c` must stay `a*b + c`. Rust never
  contracts to FMA or reorders float ops, which is what makes upstream-identical output
  achievable at all.

## 6. API conventions (summary)

Full mapping table in [ADR-0005](adr/0005-c-parity-api.md). Points with design content:

- **Strides are `usize` bytes**, same unit as upstream (`vertex_positions_stride` is bytes,
  *not* element count) — porting bugs most often hide here; every stride use must round-trip
  `&data[base..base + stride]`-style arithmetic through `usize` byte offsets.
- **Return contracts** (documented per function, mirroring upstream):
  - encoders → `usize` bytes written; `0` = destination smaller than the corresponding
    `*_bound()` result (the only failure cause);
  - decoders → `bool`; `false` = malformed input; **no panic on any input** (SRS NFR-3);
  - optimizers/remaps → `usize` counts where upstream returns counts, `()` for in-place void fns;
  - `simplify`-family → `usize` output index count, `Option<&mut f32>` for `result_error`.
- **Constants/enums.** Option bit flags (e.g. `SimplifyLockBorder = 1 << 0`) become `pub const`
  `u32`s in the owning module with values copied from the header. `EncodeExpMode` becomes an
  enum with explicit discriminants matching upstream values.
- **Stats/bounds structs** are `#[derive(Clone, Copy, Debug, PartialEq)]`; field names and order
  follow the header (`meshopt_Meshlet`'s fixed arrays become `[u32; 64]` / `[u8; 126]`-style
  fields sized from upstream `kMeshletMax*` constants).
- **Callbacks.** `generateVertexRemapCustom`'s C callback becomes `impl FnMut(u32, u32) -> bool`
  (return `true` = merge, matching upstream's nonzero semantics).
- **Encoder wire version.** Hardcoded to the current version (v1); the upstream global setters
  (`encodeIndexVersion`/`encodeVertexVersion`) are not ported, and `encodeVertexBufferLevel`
  takes only the compression level — [ADR-0009](adr/0009-encoder-version-hardcoded.md).
  Decoders accept every version upstream produces (v0/v1).

## 7. Determinism analysis

- Rust float semantics are strict IEEE: no reassociation, no FMA contraction, no fast-math —
  identical behavior across platforms by construction.
- `libm` is pure Rust: same bits on every target for the same input
  ([ADR-0004](adr/0004-numeric-via-libm.md)).
- Sorting in the port uses *stable* sorts with the same key functions and tie-breaking as
  upstream; upstream's reliance on `std::sort` vs `std::stable_sort` is audited per module
  during porting (checklist item in each module's test module header).
- Hash-based remaps (FR-01) replicate upstream's exact hash function, seed, and probe order —
  byte-exactness of the remap output is part of the fixture suite.
- `partition`'s deterministic RNG is ported algorithm-for-algorithm (no platform `rand`).

## 8. Panic-safety implementation notes

- Decode paths: every read of the encoded buffer goes through a checked helper
  (`bitstream`-style cursor returning `Option`), so one `?`-style guard per field rather than
  scattered index arithmetic.
- Hot loops mirror upstream's hoisted bounds checks: one length validation before the loop body,
  indexed access inside is then total (documented with a short comment where non-obvious).
- `debug_assert!` mirrors every upstream `assert()`; release builds omit them (NDEBUG parity),
  and memory safety makes the omission benign (ADR-0003).

## 9. Phase-1 implementation sequence

Ordered to land shared internals before dependents; each step is a self-contained, gate-passing
commit series driven test-first ([TDD](tdd.md) §3/§6):

| Step | Content | Depends on |
|---|---|---|
| P1.0 | crate skeleton, lints, gate wiring, `math` | — |
| P1.1 | `quantization`, `indexgenerator` | P1.0 |
| P1.2 | `indexanalyzer`, `stripifier`, `vfetchoptimizer`, `vcacheoptimizer` | P1.0 |
| P1.3 | `rasterizer`, `overdrawoptimizer`, `spatialorder`, `partition` | P1.2 |
| P1.4 | `indexcodec`, `vertexcodec`, `vertexfilter` + codec fixtures | P1.0 |
| P1.5 | `meshletutils`, `meshletcodec`, `clusterizer` | P1.4 |
| P1.6 | `simplifier` | P1.1 |
| P1.7 | `tangentspace`, `opacitymap`, `remesher` (experimental) | P1.3 |
| P1.8 | corpus hardening, hostile-input suites, MSRV run | all |

## 10. Open items

- `simplifier`'s exact module split is decided (see §4, landed in P1.6).
- FR-18 cargo feature: decided at P1.7 — **no feature gate**; upstream expands
  `MESHOPTIMIZER_EXPERIMENTAL` to the public API by default and the v1.3 pin froze the contract
  (see SRS §3). `generateNormals`/`remesh` ship flat with experimental-status doc comments.
- crates.io package name verification happens at publish time ([ADR-0008](adr/0008-licensing.md)).
