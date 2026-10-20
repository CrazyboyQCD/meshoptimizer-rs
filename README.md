# rmeshopt

A Rust port of
[meshoptimizer](https://github.com/zeux/meshoptimizer) version 1.3.

## Feature flags

| Feature | Effect |
| --- | --- |
| `std` *(default)* | Platform math; drop for `no_std` + `alloc` (pure-Rust `libm` backend) |
| `simd` | SIMD decode kernels (x86_64 / aarch64 / wasm32) |
| `simd-avx512` | AVX512 vertex-decode kernel variant |

## Usages

```rust
use rmeshopt::simplify;

// indices: triangle list; positions: `vertex_count` vertices, 3 floats each
let mut destination = vec![0u32; indices.len()];
let written: usize = simplify(
    &mut destination,
    &indices,
    &positions,
    vertex_count,
    12,                // vertex_positions_stride in bytes
    indices.len() / 2, // target index count
    1e-2,              // target error
    0,                 // options
    None,
);
destination.truncate(written);
```

### Vertex cache optimization

```rust
use rmeshopt::optimize_vertex_cache;

let mut optimized = vec![0u32; indices.len()];
optimize_vertex_cache(&mut optimized, &indices, vertex_count);
```

### Index buffer compression

```rust
use rmeshopt::{decode_index_buffer, encode_index_buffer, encode_index_buffer_bound};

let mut buffer = vec![0u8; encode_index_buffer_bound(indices.len(), vertex_count)];
let compressed = encode_index_buffer(&mut buffer, &indices);

// decode back into 4-byte indices; false on malformed input, never panics
let mut decoded = vec![0u8; indices.len() * 4];
let ok = decode_index_buffer(&mut decoded, indices.len(), 4, &buffer[..compressed]);
```

## Documentation

[`docs/srs.md`](docs/srs.md) (requirements) ·
[`docs/design.md`](docs/design.md) (port design) ·
[`docs/tdd.md`](docs/tdd.md) (fixture methodology) ·
[`docs/adr/`](docs/adr/) (14 decision records).

## License

Distributed under the [MIT license](LICENSE-MIT). This is a derivative work of
meshoptimizer, which is Copyright (c) 2016-2026 Arseny Kapoulkine and also MIT
licensed ([LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY)); not affiliated with or
endorsed by the upstream authors.
