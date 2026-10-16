# ADR-0014: Unsafe governance and plain float-to-int truncation on the grid-quantization hot path

- Status: Accepted
- Date: 2026-10-14
- Amends: [ADR-0003](0003-forbid-unsafe-deny-panic.md) and [ADR-0012](0012-simd-feature-scoped-unsafe.md)
  (crate-wide `forbid(unsafe_code)` becomes an unconditional `deny(unsafe_code)` with a
  per-site, audited allowance inventory)
- Amends: [ADR-0013](0013-saturating-float-int-casts.md) for `compute_vertex_ids` only
  (its saturating-`as` contract is replaced by a documented finiteness precondition;
  `voxelize` and the quantize paths keep ADR-0013 unchanged)
- Refines: [ADR-0011](0011-benchmark-methodology.md) (change driven by the measured
  attribution workflow)

## Context

ADR-0013 replaced the branching `cast_i32` wrapper with saturating `as` casts on the two
measured hot paths. Phase attribution then showed `compute_vertex_ids`
(simplifier/sloppy.rs; called by both `meshopt_simplifySloppy` and
`meshopt_simplifyPoints`) still 4.6x slower than upstream in isolation: Rust's only safe
conversion lowers to the `fptosi.sat` intrinsic, whose vector lowering on baseline SSE2
scalarizes per lane, while upstream C's `int()` — plain `fptosi`, UB outside the
representable range per C99 6.3.1.4 — vectorizes to a single packed `cvttps2dq`.
Isolated measurements (32768 vertices, 400-round median, identical flags):

| variant | ns/call | relative |
|---|---|---|
| upstream C (UB `int()`, clang -O2) | 18,800 | 1.00x |
| `to_int_unchecked` (plain truncation) | 18,700 | 1.00x |
| saturating `as` (ADR-0013 state) | 86,400 | 4.61x |
| float-clamped + `to_int_unchecked` | 22,800 | 1.21x |

A float-clamped checked variant was evaluated and rejected: it adds 22% at the function
level to buy determinism exactly where upstream is itself undefined — there is no
upstream behavior to stay compatible with in that window.

## Decision

1. **`compute_vertex_ids` uses `to_int_unchecked::<i32>()`** for the three quantization
   conversions, restoring bit-for-bit the truncation semantics of upstream's `int()` and
   full auto-vectorization (packed `mulps/addps/cvttps2dq/pslld` chain).
2. **The precondition is documented, not enforced**: input positions must be finite.
   This is enforced nowhere in upstream either — `int(NaN)` is C UB, so the Rust UB
   window coincides exactly with the upstream UB window. Callers safe for upstream are
   safe for this port; the public entry points (`simplify_sloppy`, `simplify_points`)
   carry the precondition in their doc comments, and `debug_assert!(is_finite)` fires in
   debug builds.
3. **Soundness of the concrete sites** rests on the value range through the only two
   call chains: `rescale_positions` maps finite inputs into [0, 1] (extreme magnitudes
   normalize; a degenerate zero extent collapses to 0), and `cell_scale` is bounded by
   1023 by construction, so the converted argument lies in [0.5, 1023.5]. The two golden
   suites covering both callers pass under Miri, which flags `float_to_int_unchecked`
   out-of-range as UB (verified: a NaN input is caught); the extreme-finite unit test
   keeps that proof executable.
4. **Unsafe governance** (replacing the forbid regime of ADR-0003/0012, which cannot
   express a legitimate always-on unsafe site): `#![deny(unsafe_code)]` crate-wide.
   `deny` remains machine-enforced — new unsafe cannot compile without an explicit
   `allow(unsafe_code)` — while each allowance is a reviewed, SAFETY-commented site.
   The complete inventory is: `simplifier::sloppy::compute_vertex_ids` (this ADR) and
   `vertexcodec::decode_simd` ([ADR-0012]). `grep -rn "allow(unsafe_code)" src/` must
   list exactly these two sites.

## Consequences

- `simplify_sloppy` drops from 10.51 ms to ~9.4-9.7 ms on the bench corpus (1.32x →
  ~1.19x vs cpp O2); outputs on all 33 golden fixtures remain byte-identical.
- The crate is no longer "zero unsafe": it is unsafe-minimal with a two-site inventory,
  each traced to an ADR and argued sound. This is a positioning change and is stated in
  the crate documentation before publishing.
- `voxelize` keeps ADR-0013's saturating `as` (its downstream clamp absorbs saturation,
  and no vectorization benefit was measured); the quantize API paths keep
  `cast_i32`. New hot-path conversion sites must choose between the two ADRs explicitly.

## Erratum 1 (2026-10-15): vertex filter decode kernels join the inventory

The vertex filters that feed the vertex codec are bound by ADR-0010's scalar-output
contract (byte-identical to the scalar oracle), so their decoded bytes are wire-visible
and must match upstream exactly on **every** input — including bytes produced by a
hostile decoder. `meshopt_decodeFilterOct`/`Quat` round three or four floats per element
through `cast_i32`, whose branch blocks loop vectorization even in the chunked kernel
shape (measured 2026-10-15, interleaved 3×3, bench corpus `vf_*`):

| kernel variant | vf_oct | vf_quat |
|---|---|---|
| stride-indexed (pre-chunking) | 5.56x | 2.55x |
| chunked, saturating `as` | 2.64x | 1.48x |
| chunked, `to_int_unchecked` | **2.07x** | **1.15x** |

Decision, per kernel:

1. **`decode_filter_quat_s16`**: plain `to_int_unchecked::<i32>()` with no input
   precondition. For every `i16` input, `sf = wi | 3` is never zero (`|s| >= 1`), so `ss
   = scale / s` is finite; `|x * ss| <= 32767 * (32767 / sqrt(2)) < 2^30` and
   `|w * ss| <= sqrt(2) * scale` up to one rounding step; no NaN is reachable. The
   converted value is always representable — the site is unconditionally sound.
2. **`decode_filter_oct_s16`**: one guard `s = if l == 0.0 { 0.0 } else { max / l }`
   (a compare-select, stays inside the vectorized loop) removes the only NaN path
   (`0 * inf` when the normal is exactly zero). With `l > 0`, each of `|x|, |y|, |z|` is
   bounded by `l` up to three monotone rounding steps, so all three converted arguments
   lie within `max * (1 + 2^-22)`; with `l == 0` the input is exactly zero and the
   arguments are `±0.5`. The converted value is always representable — the site is
   unconditionally sound, **unlike `compute_vertex_ids` it requires no documented
   precondition**: the decode filters accept arbitrary decoded bytes.
3. Output bytes are unchanged for every input: in-bounds truncation is identical to the
   saturating cast, and the former NaN path (`cast_i32 -> i32::MIN -> truncating i16
   cast`) and the guard path (`0 * 0 + ±0.5 -> 0`) both produce zero bytes. A colocated
   differential test pins the kernels against a `cast_i32` reference over zero, extreme,
   and randomized sweeps; the filter goldens and the differential tests run clean under
   Miri (which flags `float_to_int_unchecked` out-of-range as UB).

The inventory grows to: `simplifier::sloppy::compute_vertex_ids`,
`vertexcodec::decode_simd`, `vertexfilter::decode_filter_oct_s16`,
`vertexfilter::decode_filter_quat_s16`. The crate positioning becomes "unsafe-minimal
(4 audited sites), each traced to an ADR and argued sound for every input".

## Erratum 2 (2026-10-16): `simplifier::prune::follow` joins the inventory

`meshopt_simplifyPrune` builds connected components with a union-find whose
`follow` performs path-halving inside a pointer-chase loop. The bounds check on
each `parents[index]` access sits on the dependent-load chain of the chase (every
hop's next address depends on the previous load), so unlike the noise-level
check costs measured on remesh/simplify_sloppy/voxelize (2026-10-14/15), the
check here is serialized with every memory reference. Measured on the bench
corpus (interleaved 3×3, 2026-10-16): removing only the `follow` checks takes
`prune` from 4.29 ms to 3.60 ms (−17.5%, 1.36x → 1.14x vs cpp O2, which indexes
through raw pointers); the remaining unchecked accesses in `build_components`
outside `follow` were worth <0.1 ms and stay bounds-checked.

Decision: `follow` reads and writes `parents` through `get_unchecked` /
`get_unchecked_mut` inside the chase loop. Soundness is argued from a structural
invariant rather than an input precondition, so the site is sound for every
input satisfying the caller contracts:

1. Every value ever stored in `parents` is `< parents.len()`: initialization
   writes `components[i] = i` for `i < vertex_count` (the table is allocated and
   iota-filled inside `build_components` itself); every merge step stores a
   value returned by `follow` (a value previously read from `parents`); no other
   stores exist. By induction, all values are among the initially stored
   `0..vertex_count`, and `vertex_count <= parents.len()` holds at every call
   site (`build_components` allocates `components` with `vertex_count`).
2. Entry indices are in range: the edge loop passes `remap[i0]`/`remap[i1]`
   values from the position-weld table, whose entries are vertex indices
   `< vertex_count` by construction of `build_position_remap` (and the preceding
   `remap[i0 as usize]` load is bounds-checked against `remap`, whose length is
   `vertex_count`); the root loop passes `i as u32` from a `0..vertex_count`
   range. The colocated `follow_path_halves` test table satisfies the invariant.
3. `parents` is the unique `&mut` borrow for the call's duration and is never
   reallocated, so no aliasing or dangling hazard exists.

The colocated prune tests (union-find, measure, and the `sp_prune` golden
fixture) run clean under Miri, which flags out-of-bounds `get_unchecked` as
undefined behavior; a deliberately corrupted table would be reported there.

The inventory grows to: `simplifier::sloppy::compute_vertex_ids`,
`vertexcodec::decode_simd`, `vertexfilter::decode_filter_oct_s16`,
`vertexfilter::decode_filter_quat_s16`, `simplifier::prune::follow`. The crate
positioning becomes "unsafe-minimal (5 audited sites), each traced to an ADR
and argued sound".
