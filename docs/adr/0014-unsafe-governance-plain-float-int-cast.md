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
