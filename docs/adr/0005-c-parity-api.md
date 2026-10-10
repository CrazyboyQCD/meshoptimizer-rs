# ADR-0005: C-parity API surface and error semantics

- Status: Accepted
- Date: 2026-10-10

## Context

The port's primary quality instrument is *auditability*: a reviewer must be able to put a C++
translation unit and its Rust module side by side and verify statement-by-statement equivalence.
A redesigned "idiomatic-first" API (builders, iterators, type-state strides) would break that
mapping before the port is even proven correct. Ergonomics can be layered later; correctness
cannot be retro-fitted.

## Decision

The phase-1 public API mirrors upstream 1:1, with mechanical renames only:

| Upstream (C) | Rust |
|---|---|
| `meshopt_optimizeVertexCache(...)` | `optimize_vertex_cache(...)` at crate root (flat namespace, no `meshopt_` prefix) |
| `const unsigned* indices, size_t index_count` | `indices: &[u32]` |
| `const void* vertices, size_t vertex_count, size_t vertex_size/stride` | `vertices: &[u8]`, explicit `stride: usize` (bytes, same meaning as upstream) |
| `const float* vertex_positions, ..., stride` | `vertex_positions: &[f32]`, `vertex_positions_stride: usize` (bytes, same as upstream) |
| `unsigned int* destination` out-param | `destination: &mut [u32]` |
| return `size_t` (0 = failure / output too small) | return `usize` (0 keeps the same meaning, documented per function) |
| return `int` (decode success) | return `bool` |
| `float* result_error` (nullable out) | `result_error: Option<&mut f32>` |
| options bitmask (`meshopt_SimplifyLockBorder`, ...) | `u32` bit-flag constants in the owning module, values matching upstream |
| `enum meshopt_EncodeExpMode` | `enum` with explicit discriminants equal to upstream values |
| `int (*callback)(void*, unsigned, unsigned)` | `impl FnMut(u32, u32) -> bool` parameter |
| `struct meshopt_Stream` / `Meshlet` / `Bounds` / statistics | plain structs, `#[derive(Clone, Copy, Debug, PartialEq)]`, field-for-field |
| `size_t meshopt_*Bound(...)` size upper bounds | same functions; `const fn` where computable without allocation |

Excluded (not portable, no Rust counterpart needed): `meshopt_setAllocator`,
`meshopt_setAllocatorCache`, the encoder version setters (`meshopt_encodeIndexVersion`,
`meshopt_encodeVertexVersion` — see [ADR-0009](0009-encoder-version-hardcoded.md)), and the C++
STL adapter layer (`meshopt_IndexAdapter`, `meshopt_Buffer`) — Rust slices and `Vec` fill those
roles.

Rationale for keeping `usize`/`bool` C-style returns instead of `Result`: encode "failures" are
one cause (destination buffer smaller than the documented `*_bound()`), and decode "failures" are
one cause (malformed input). Neither carries data; `Result` would add no information while
breaking the 1:1 mapping.

## Consequences

- Every public function carries a doc comment naming its upstream counterpart (e.g.
  `Upstream: meshopt_optimizeVertexCache (vcacheoptimizer.cpp)`), so audit and future doc
  tooling can trace provenance.
- Idiomatic convenience wrappers (growable-output variants, typed vertex structs) may be added
  later as a separate layer; they must delegate to, not replace, the parity layer.
- `*_bound()` functions returning `usize` enable callers to allocate exactly; they are the
  sanctioned way to avoid "output too small" results.
