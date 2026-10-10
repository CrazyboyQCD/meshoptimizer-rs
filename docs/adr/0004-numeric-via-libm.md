# ADR-0004: Numeric kernels via `libm` / `std` through one `math` module; no hand-written math

- Status: Accepted
- Date: 2026-10-10

## Context

`no_std` removes `f32::sqrt`, `f32::acos`, `f32::floor`, and friends. Measured usage across
upstream v1.3 sources:

| function | call sites | consumers |
|----------|-----------|-----------|
| `sqrtf` / `sqrt` (f64) | 63 + 12 | cluster bounds cones, simplifier quadrics, spatial order, opacity map |
| `fabsf` | 37 | simplifier, clusterizer, filters |
| `floorf` / `floor` | 3 + 1 | quantization, vertex filters, encode filters |
| `ceilf` / `ceil` | 2 + 1 | quantization |
| `acos` (f64) | 1 | cluster bounds cone axis |
| `log2f` | 1 | vertex codec (v1 bit estimation) |
| `sin` / `cos` (f64) | 2 + 4 | cluster bounds cone |
| `cosf` | 1 | cluster bounds |
| `frexpf` | 2 | quantization (`computePositionExponent`) |
| `ldexpf` | 1 | quantization |

Writing scalar approximations by hand (polynomials, `rsqrt` tricks) would break bit-level
agreement with upstream, which calls the platform libm.

## Decision

- Depend on `libm = "0.2"` (pure Rust, `no_std`, deterministic, identical results on every
  platform for a given input) as the `no_std` backend; with the default `std` feature the same
  wrappers call the standard library instead ([ADR-0002](0002-no-std-alloc-baseline.md)). The
  rules in this ADR apply to both backends.
- Call it through one internal `math` module with short wrappers (`math::sqrt(f32)`,
  `math::acos_f64(f64)`, ...). Call sites stay readable and the indirection gives a single place
  to swap or test implementations. All wrappers are `#[inline(always)]`, so the indirection has
  zero call overhead in release builds.
- Plain arithmetic (`+ - * /`, `f32::abs`, `f32::min/max`, comparisons) comes from `core`
  directly — no wrapper, no trait indirection.
- Never hand-approximate a function that upstream takes from libm. If a future module seems to
  need a new libm entry point, add it to the `math` module and note it in the module's porting
  checklist.

## Consequences

- `sqrt`/`floor`/`ceil` are correctly rounded by IEEE 754 — bit-identical to any conforming
  platform libm, so encoders that quantize after `floorf` stay byte-exact (required by ADR-0007).
- `frexp`/`ldexp` are exact functions (bit-level results): `libm` is used for them in *both*
  configurations because std's variants are unstable; results are identical everywhere.
- `acos`, `log2`, `sin`, `cos` are *not* correctly-rounded functions; libm results may differ
  from glibc/MSVC by ≤1 ULP. Differential tests for float outputs route through those functions
  must therefore use a documented tolerance (TDD §7), while pure-arithmetic and integer outputs
  are compared exactly.
- `libm` adds a compile-time dependency but zero runtime cost for `sqrt` (LLVM lowers it to the
  hardware instruction on common targets).
