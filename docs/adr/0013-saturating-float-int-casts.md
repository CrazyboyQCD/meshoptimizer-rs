# ADR-0013: Saturating float-to-int casts on bounded hot paths

- Status: Accepted
- Date: 2026-10-14
- Amends: [ADR-0005](0005-c-parity-api.md) (narrows the bit-parity goal for inputs where
  upstream C behavior is undefined; the x86 `cvttss2si` mirror policy embodied in
  `quantization::cast_i32` is superseded at the named hot-path sites below)
- Refines: [ADR-0011](0011-benchmark-methodology.md) (this change is driven by the
  measured attribution workflow, not applied speculatively)

## Context

Phase-attribution probes (2026-10-14) located the remaining scalar-profile gaps on
`simplify_sloppy` (search loop at 2.19× vs clang O2) and `remesh` (three loops at
1.4–2.8×). Assembly comparison at identical flags (clang `-O2`, rustc opt-level 2, both
baseline x86-64/SSE2) showed rustc emitted fully scalar code for these loops while clang
vectorized the same source. The blocking factor on the Rust side is
`quantization::cast_i32`, which wraps Rust's saturating `f32 as i32` in a branch sequence
mirroring x86 `cvttss2si` (`INT_MIN` on NaN and out-of-range). The per-component
NaN/range branches defeat SLP/loop vectorization; replacing the three casts in
`compute_vertex_ids` with plain `as` recovered −17% on the whole `simplify_sloppy` op
(corpus pins still byte-identical), and the same substitution in `voxelize` recovered
−7% on `remesh`.

The parity question this raises: upstream computes `int(expr)` in these loops. Per C99
§6.3.1.4, a float-to-integer conversion of a NaN or a value outside the range of the
integer type is **undefined behavior**. The `INT_MIN` ("integer indefinite") result on
x86 is an implementation artifact of `cvttss2si`, not a documented contract of
meshoptimizer; no golden fixture exercises it (the corpus contains only finite,
in-range geometry). For all in-range finite values, Rust's `as` and C's `int()` are
both truncation toward zero and bit-identical — the entire observable divergence window
is exactly the UB territory of the upstream source.

## Decision

1. **Hot-path sites with provably bounded valid inputs use saturating `as` casts:**
   - `compute_vertex_ids` (src/simplifier/sloppy.rs): the position quantization loop;
     shared by `simplify_sloppy` and `simplify_points`.
   - `voxelize` h-coordinates (src/remesher/mod.rs): `hx/hy/hz` in the per-sample loop.
2. **The saturation behavior is part of the port's contract on invalid inputs** and is
   pinned by unit tests: NaN casts to `0`, values ≥ 2³¹ to `i32::MAX`, values < −2³¹ to
   `i32::MIN` (Rust scalar semantics on every target, matching the `as` operator — not
   x86-specific).
3. **In `voxelize`, the downstream clamp is unchanged and still bounds ±overflow**
   (`unsigned(x) < unsigned(cutoff)` wraps `i32::MAX`/`i32::MIN` to huge unsigned values
   and clamps to `cutoff`, as upstream). NaN positions land in voxel 0 instead of
   upstream-x86's cutoff voxel; this is the one deliberate divergence and it is inside
   upstream's UB window.
4. **`quantization::cast_i32` is retained** for the public-API quantize paths
   (`quantize_snorm`/`quantize_float` chain) and for cold sites (`min_grid`,
   `next_grid_size`, `tip` in the grid searches — once per pass, no measured cost, no
   reason to widen the divergence surface). New code defaults to `cast_i32` unless this
   ADR's criteria (measured hot path + bounded valid-input range) are met.

## Consequences

- Bench (Zen3, 3-round medians, 2026-10-14): `simplify_sloppy` 12.31 → 10.24 ms
  (1.54× → 1.28× vs cpp O2); `remesh` 12.14 → ~11.0 ms (1.80× → 1.63× fresh-session
  baseline). All 33 golden fixtures remain byte-identical; corpus output pins verified
  per run.
- Invalid-input behavior is deterministic and documented here rather than mimicking one
  x86 implementation artifact; cross-platform determinism (ADR-0002 goals) is
  unaffected — saturating `as` has identical semantics on every target, where the
  emulated `cvttss2si` wrapper was the only thing standing between `no_std` builds and
  x86-specific observable behavior.
- Future "why `as` here but `cast_i32` there" questions resolve to this ADR; removing
  the wrapper entirely is out of scope and not planned.

## Erratum 1 (2026-10-19): dead-branch audit narrows `cast_i32` retention

A full call-chain audit of the remaining `cast_i32` sites found the wrapper's checks
**unreachable for every input** at ten sites — the UB window is already absorbed
upstream of the cast by a total clamp (`quantize_snorm` / `quantize_unorm` clamp in
`>=`-form, converting NaN into range), a downstream clamp plus bounds safeguard
(`sample_texture`, `voxelize` sample count), count-derived value bounds (`bvh_pivot`
fill cost, `min_grid`, `next_grid_size`), or an equivalence in the non-finite case
(`decode_filter_oct` s8 and meshlet `cone_cutoff_s8`: NaN truncates to 0 exactly like
x86 `INT_MIN` after the narrowing cast). These sites now use saturating `as` directly;
outputs are bit-identical for every input including hostile ones (all 33 golden
fixtures unchanged).

`cast_i32` survives only where upstream's C99 §6.3.1.4 UB window is genuinely
reachable: color u16 decode (`as_ == 0` -> division by zero, and extreme components
with small `as_` keep an i32-overflow window the guard cannot close), `encode_filter_exp`
mantissa
(user NaN/inf), opacity-map level/edge resolution and subpixel UV quantization (inf or
huge user UVs), normal-smoothing
passes (smoothing > 2^31), and the grid-search `tip` interpolation. At those sites the
wrapper pins the fixture-generator platform's de facto behavior (x86 `cvttss2si` ->
`INT_MIN`): upstream has no single cross-platform behavior in this window (AArch64
`fcvtzs` saturates with NaN -> 0, RISC-V `fcvt.w.s` maps NaN -> `INT_MAX`, wasm
`i32.trunc_f32_s` traps), and the ADR-0007 golden rig is generated on x86, so the port
selects x86 semantics as the canonical one for the UB window. The color u8 kernel
instead takes an `as_ == 0` -> `ss = 0` guard — byte-identical to upstream's
division-by-zero truncation and pinned by a colocated test — so only the u16 kernel
carries the wrapper among the color paths.
