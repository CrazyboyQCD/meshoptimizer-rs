//! Vertex remapping and index buffer generation.
//!
//! Upstream: `indexgenerator.cpp`. Hashing replicates upstream's MurmurHash2-based
//! open addressing byte-for-byte; hash arithmetic uses explicit wrapping operations
//! (C unsigned semantics) so outputs are identical in debug and release builds.

use alloc::vec;
use alloc::vec::Vec;

/// Upstream: `meshopt_Stream` — a vertex attribute stream with byte granularity.
#[derive(Clone, Copy, Debug)]
pub struct Stream<'a> {
    pub data: &'a [u8],
    pub size: usize,
    pub stride: usize,
}

fn hash_update(mut h: u32, key: &[u8]) -> u32 {
    // MurmurHash2
    const M: u32 = 0x5bd1_e995;
    const R: u32 = 24;

    for chunk in key.as_chunks::<4>().0 {
        let mut k = u32::from_le_bytes(*chunk);

        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);

        h = h.wrapping_mul(M);
        h ^= k;
    }

    let len = key.len() % 4;
    if len > 0 {
        let tail = &key[key.len() - len..];
        h ^= u32::from(tail[0]) | u32::from(tail[len / 2]) << 8 | u32::from(tail[len - 1]) << 16;
        h = h.wrapping_mul(M);
        h ^= h >> R;
    }

    h
}

fn hash_buckets(count: usize) -> usize {
    let mut buckets = 1usize;
    while buckets < count + count / 4 {
        buckets *= 2;
    }
    buckets
}

fn filled<T: Copy>(len: usize, value: T) -> Vec<T> {
    vec![value; len]
}

fn hash_lookup<T: Copy + PartialEq>(
    table: &mut [T],
    hash: u32,
    mut eq: impl FnMut(&T, &T) -> bool,
    key: T,
    empty: T,
) -> usize {
    let hashmod = table.len() - 1;
    let mut bucket = hash as usize & hashmod;
    let mut probe = 0usize;

    // termination: hash_buckets caps the load factor at 0.8 and quadratic probing
    // over a power-of-two table visits every slot, so an empty slot always exists
    loop {
        let item = &table[bucket];

        if *item == empty || eq(item, &key) {
            return bucket;
        }

        // hash collision, quadratic probing
        bucket = (bucket + probe + 1) & hashmod;
        probe += 1;
    }
}

#[derive(Clone, Copy, PartialEq)]
struct TriangleKey {
    a: u32,
    b: u32,
    c: u32,
}

trait VertexHasher {
    fn hash(&mut self, index: u32) -> u32;
    fn equal(&mut self, lhs: u32, rhs: u32) -> bool;
}

struct BytesHasher<'a> {
    vertices: &'a [u8],
    size: usize,
    stride: usize,
}

impl VertexHasher for BytesHasher<'_> {
    fn hash(&mut self, index: u32) -> u32 {
        let base = index as usize * self.stride;
        hash_update(0, &self.vertices[base..base + self.size])
    }

    fn equal(&mut self, lhs: u32, rhs: u32) -> bool {
        let l = lhs as usize * self.stride;
        let r = rhs as usize * self.stride;
        self.vertices[l..l + self.size] == self.vertices[r..r + self.size]
    }
}

struct StreamsHasher<'a, 'b> {
    streams: &'b [Stream<'a>],
}

impl VertexHasher for StreamsHasher<'_, '_> {
    fn hash(&mut self, index: u32) -> u32 {
        let mut h = 0u32;

        for s in self.streams {
            let base = index as usize * s.stride;
            h = hash_update(h, &s.data[base..base + s.size]);
        }

        h
    }

    fn equal(&mut self, lhs: u32, rhs: u32) -> bool {
        self.streams.iter().all(|s| {
            let l = lhs as usize * s.stride;
            let r = rhs as usize * s.stride;
            s.data[l..l + s.size] == s.data[r..r + s.size]
        })
    }
}

struct CustomHasher<'a, 'b> {
    positions: &'a [f32],
    stride_float: usize,
    callback: Option<&'b mut dyn FnMut(u32, u32) -> bool>,
}

impl VertexHasher for CustomHasher<'_, '_> {
    fn hash(&mut self, index: u32) -> u32 {
        let base = index as usize * self.stride_float;

        let mut x = self.positions[base].to_bits();
        let mut y = self.positions[base + 1].to_bits();
        let mut z = self.positions[base + 2].to_bits();

        // replace negative zero with zero
        if x == 0x8000_0000 {
            x = 0;
        }
        if y == 0x8000_0000 {
            y = 0;
        }
        if z == 0x8000_0000 {
            z = 0;
        }

        // scramble bits to make sure that integer coordinates have entropy in lower bits
        x ^= x >> 17;
        y ^= y >> 17;
        z ^= z >> 17;

        // Optimized Spatial Hashing for Collision Detection of Deformable Objects
        x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ z.wrapping_mul(83492791)
    }

    fn equal(&mut self, lhs: u32, rhs: u32) -> bool {
        let l = lhs as usize * self.stride_float;
        let r = rhs as usize * self.stride_float;

        if self.positions[l] != self.positions[r]
            || self.positions[l + 1] != self.positions[r + 1]
            || self.positions[l + 2] != self.positions[r + 2]
        {
            return false;
        }

        match &mut self.callback {
            Some(callback) => callback(lhs, rhs),
            None => true,
        }
    }
}

/// Upstream: `generateVertexRemap` — a template there, monomorphized here so the
/// hash and equality probes inline instead of dispatching through `dyn`.
fn generate_vertex_remap_impl<H: VertexHasher>(
    hasher: &mut H,
    destination: &mut [u32],
    indices: Option<&[u32]>,
    index_count: usize,
    vertex_count: usize,
) -> usize {
    destination[..vertex_count].fill(!0);

    let mut table = filled(hash_buckets(vertex_count), !0u32);

    let mut next_vertex = 0u32;

    for i in 0..index_count {
        let index = match indices {
            Some(indices) => indices[i],
            None => i as u32,
        };
        debug_assert!(index < vertex_count as u32);

        if destination[index as usize] != !0 {
            continue;
        }

        let hash = hasher.hash(index);
        let slot = hash_lookup(&mut table, hash, |a, b| hasher.equal(*a, *b), index, !0);

        if table[slot] == !0 {
            table[slot] = index;
            destination[index as usize] = next_vertex;
            next_vertex += 1;
        } else {
            debug_assert!(destination[table[slot] as usize] != !0);
            destination[index as usize] = destination[table[slot] as usize];
        }
    }

    next_vertex as usize
}

fn generate_shadow_impl<H: VertexHasher>(
    hasher: &mut H,
    destination: &mut [u32],
    indices: &[u32],
    vertex_count: usize,
) {
    let mut remap = filled(vertex_count, !0u32);
    let mut table = filled(hash_buckets(vertex_count), !0u32);

    for (i, &index) in indices.iter().enumerate() {
        debug_assert!(index < vertex_count as u32);

        if remap[index as usize] == !0 {
            let hash = hasher.hash(index);
            let slot = hash_lookup(&mut table, hash, |a, b| hasher.equal(*a, *b), index, !0);

            if table[slot] == !0 {
                table[slot] = index;
            }

            remap[index as usize] = table[slot];
        }

        destination[i] = remap[index as usize];
    }
}

fn filter_index_buffer_impl(destination: &mut [u32], indices: &[u32], remap: &[u32]) -> usize {
    let empty = TriangleKey {
        a: !0,
        b: !0,
        c: !0,
    };

    let face_count = indices.len() / 3;
    let mut table = filled(hash_buckets(face_count), empty);

    let mut write = 0usize;

    for &[i0, i1, i2] in indices.as_chunks::<3>().0 {
        let mut ra = remap[i0 as usize];
        let mut rb = remap[i1 as usize];
        let mut rc = remap[i2 as usize];

        // degenerate triangle
        if ra == rb || ra == rc || rb == rc {
            continue;
        }

        // canonicalize triangle wrt winding preserving rotation
        if rb < ra && rb < rc {
            // abc -> bca
            let t = ra;
            ra = rb;
            rb = rc;
            rc = t;
        } else if rc < ra && rc < rb {
            // abc -> cab
            let t = rc;
            rc = rb;
            rb = ra;
            ra = t;
        }

        let key = TriangleKey {
            a: ra,
            b: rb,
            c: rc,
        };
        let hash = key.a.wrapping_mul(73856093)
            ^ key.b.wrapping_mul(19349663)
            ^ key.c.wrapping_mul(83492791);
        let slot = hash_lookup(&mut table, hash, |a, b| a == b, key, empty);

        // duplicate triangle
        if table[slot].a != !0 {
            continue;
        }

        table[slot] = key;

        destination[write] = i0;
        destination[write + 1] = i1;
        destination[write + 2] = i2;
        write += 3;
    }

    write
}

fn build_position_remap(
    remap: &mut [u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) {
    let mut hasher = CustomHasher {
        positions: vertex_positions,
        stride_float: vertex_positions_stride / 4,
        callback: None,
    };

    let mut table = filled(hash_buckets(vertex_count), !0u32);

    for i in 0..vertex_count {
        let index = i as u32;
        let hash = hasher.hash(index);
        let slot = hash_lookup(&mut table, hash, |a, b| hasher.equal(*a, *b), index, !0);

        if table[slot] == !0 {
            table[slot] = index;
        }

        remap[index as usize] = table[slot];
    }
}

struct EdgeHasher<'a> {
    remap: &'a [u32],
}

impl EdgeHasher<'_> {
    fn hash(&self, edge: u64) -> u32 {
        let e0 = (edge >> 32) as u32;
        let e1 = edge as u32;

        let mut h1 = self.remap[e0 as usize];
        let mut h2 = self.remap[e1 as usize];

        const M: u32 = 0x5bd1_e995;

        // MurmurHash64B finalizer
        h1 ^= h2 >> 18;
        h1 = h1.wrapping_mul(M);
        h2 ^= h1 >> 22;
        h2 = h2.wrapping_mul(M);
        h1 ^= h2 >> 17;
        h1 = h1.wrapping_mul(M);
        h2 ^= h1 >> 19;
        h2 = h2.wrapping_mul(M);

        h2
    }

    fn equal(&self, lhs: u64, rhs: u64) -> bool {
        let l0 = (lhs >> 32) as u32 as usize;
        let l1 = lhs as u32 as usize;
        let r0 = (rhs >> 32) as u32 as usize;
        let r1 = rhs as u32 as usize;

        self.remap[l0] == self.remap[r0] && self.remap[l1] == self.remap[r1]
    }
}

/// Upstream: `meshopt_generateVertexRemap`. `indices: None` iterates
/// `0..vertex_count` as the index sequence (upstream's NULL). Returns the number
/// of unique vertices.
pub fn generate_vertex_remap(
    destination: &mut [u32],
    indices: Option<&[u32]>,
    vertices: &[u8],
    vertex_count: usize,
    vertex_size: usize,
) -> usize {
    debug_assert!(vertex_size > 0 && vertex_size <= 256);
    debug_assert!(destination.len() >= vertex_count);
    debug_assert!(vertices.len() >= vertex_count * vertex_size);

    let mut hasher = BytesHasher {
        vertices,
        size: vertex_size,
        stride: vertex_size,
    };
    let index_count = indices.map_or(vertex_count, <[u32]>::len);
    generate_vertex_remap_impl(&mut hasher, destination, indices, index_count, vertex_count)
}

/// Upstream: `meshopt_generateVertexRemapMulti`.
pub fn generate_vertex_remap_multi(
    destination: &mut [u32],
    indices: Option<&[u32]>,
    vertex_count: usize,
    streams: &[Stream<'_>],
) -> usize {
    debug_assert!(!streams.is_empty() && streams.len() <= 16);
    for s in streams {
        debug_assert!(s.size > 0 && s.size <= 256 && s.size <= s.stride);
    }
    debug_assert!(destination.len() >= vertex_count);

    let mut hasher = StreamsHasher { streams };
    let index_count = indices.map_or(vertex_count, <[u32]>::len);
    generate_vertex_remap_impl(&mut hasher, destination, indices, index_count, vertex_count)
}

/// Upstream: `meshopt_generateVertexRemapCustom`. The callback refines position
/// equality: it is invoked only for pairs with equal `x`/`y`/`z` floats and returns
/// `true` when the pair should merge (matching upstream's nonzero convention).
pub fn generate_vertex_remap_custom(
    destination: &mut [u32],
    indices: Option<&[u32]>,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    callback: Option<&mut dyn FnMut(u32, u32) -> bool>,
) -> usize {
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
    debug_assert!(destination.len() >= vertex_count);

    let mut hasher = CustomHasher {
        positions: vertex_positions,
        stride_float: vertex_positions_stride / 4,
        callback,
    };
    let index_count = indices.map_or(vertex_count, <[u32]>::len);
    generate_vertex_remap_impl(&mut hasher, destination, indices, index_count, vertex_count)
}

/// Upstream: `meshopt_remapVertexBuffer`. Rust slices cannot alias, so the
/// in-place form supported upstream is expressed by the caller cloning the
/// vertices first (identical observable behavior).
pub fn remap_vertex_buffer(
    destination: &mut [u8],
    vertices: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    remap: &[u32],
) {
    debug_assert!(vertex_size > 0 && vertex_size <= 256);
    debug_assert!(vertices.len() >= vertex_count * vertex_size);
    debug_assert!(destination.len() >= vertex_count * vertex_size);
    debug_assert!(remap.len() >= vertex_count);

    for (i, &target_index) in remap[..vertex_count].iter().enumerate() {
        if target_index != !0 {
            let target = target_index as usize;
            debug_assert!(target < vertex_count);
            let src = i * vertex_size..i * vertex_size + vertex_size;
            let dst = target * vertex_size..target * vertex_size + vertex_size;
            destination[dst].copy_from_slice(&vertices[src]);
        }
    }
}

/// Upstream: `meshopt_remapIndexBuffer`.
pub fn remap_index_buffer(destination: &mut [u32], indices: Option<&[u32]>, remap: &[u32]) {
    for (i, d) in destination.iter_mut().enumerate() {
        let index = match indices {
            Some(indices) => indices[i],
            None => i as u32,
        };
        debug_assert!(remap[index as usize] != !0);
        *d = remap[index as usize];
    }
}

/// Upstream: `meshopt_filterIndexBuffer`.
pub fn filter_index_buffer(
    destination: &mut [u32],
    indices: &[u32],
    vertices: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    vertex_stride: usize,
) -> usize {
    debug_assert!(indices.len().is_multiple_of(3));
    debug_assert!(vertex_size > 0 && vertex_size <= 256 && vertex_size <= vertex_stride);
    debug_assert!(destination.len() >= indices.len());
    debug_assert!(vertices.len() >= vertex_count * vertex_stride);

    let mut hasher = BytesHasher {
        vertices,
        size: vertex_size,
        stride: vertex_stride,
    };
    let mut remap = filled(vertex_count, !0u32);
    generate_vertex_remap_impl(
        &mut hasher,
        &mut remap,
        Some(indices),
        indices.len(),
        vertex_count,
    );

    filter_index_buffer_impl(destination, indices, &remap)
}

/// Upstream: `meshopt_filterIndexBufferMulti`.
pub fn filter_index_buffer_multi(
    destination: &mut [u32],
    indices: &[u32],
    vertex_count: usize,
    streams: &[Stream<'_>],
) -> usize {
    debug_assert!(indices.len().is_multiple_of(3));
    debug_assert!(!streams.is_empty() && streams.len() <= 16);
    for s in streams {
        debug_assert!(s.size > 0 && s.size <= 256 && s.size <= s.stride);
    }
    debug_assert!(destination.len() >= indices.len());

    let mut hasher = StreamsHasher { streams };
    let mut remap = filled(vertex_count, !0u32);
    generate_vertex_remap_impl(
        &mut hasher,
        &mut remap,
        Some(indices),
        indices.len(),
        vertex_count,
    );

    filter_index_buffer_impl(destination, indices, &remap)
}

/// Upstream: `meshopt_generateShadowIndexBuffer`.
pub fn generate_shadow_index_buffer(
    destination: &mut [u32],
    indices: &[u32],
    vertices: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    vertex_stride: usize,
) {
    debug_assert!(indices.len().is_multiple_of(3));
    debug_assert!(vertex_size > 0 && vertex_size <= 256 && vertex_size <= vertex_stride);
    debug_assert!(destination.len() >= indices.len());
    debug_assert!(vertices.len() >= vertex_count * vertex_stride);

    let mut hasher = BytesHasher {
        vertices,
        size: vertex_size,
        stride: vertex_stride,
    };
    generate_shadow_impl(&mut hasher, destination, indices, vertex_count);
}

/// Upstream: `meshopt_generateShadowIndexBufferMulti`.
pub fn generate_shadow_index_buffer_multi(
    destination: &mut [u32],
    indices: &[u32],
    vertex_count: usize,
    streams: &[Stream<'_>],
) {
    debug_assert!(indices.len().is_multiple_of(3));
    debug_assert!(!streams.is_empty() && streams.len() <= 16);
    for s in streams {
        debug_assert!(s.size > 0 && s.size <= 256 && s.size <= s.stride);
    }
    debug_assert!(destination.len() >= indices.len());

    let mut hasher = StreamsHasher { streams };
    generate_shadow_impl(&mut hasher, destination, indices, vertex_count);
}

/// Upstream: `meshopt_generatePositionRemap`.
pub fn generate_position_remap(
    destination: &mut [u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) {
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
    debug_assert!(destination.len() >= vertex_count);

    build_position_remap(
        destination,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
    );
}

/// Upstream: `meshopt_generateAdjacencyIndexBuffer`.
pub fn generate_adjacency_index_buffer(
    destination: &mut [u32],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
    debug_assert!(destination.len() >= index_count * 2);

    const NEXT: [usize; 4] = [1, 2, 0, 1];

    // build position remap: for each vertex, which other (canonical) vertex does it map to?
    let mut remap = filled(vertex_count, !0u32);
    build_position_remap(
        &mut remap,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
    );

    let edge_hasher = EdgeHasher { remap: &remap };

    let mut edge_table = filled(hash_buckets(index_count), !0u64);
    let mut edge_vertex_table = filled(edge_table.len(), !0u32);

    let mut i = 0;
    while i < index_count {
        for e in 0..3 {
            let i0 = indices[i + e];
            let i1 = indices[i + NEXT[e]];
            let i2 = indices[i + NEXT[e + 1]];
            debug_assert!(
                i0 < vertex_count as u32 && i1 < vertex_count as u32 && i2 < vertex_count as u32
            );

            let edge = (u64::from(i0) << 32) | u64::from(i1);
            let hash = edge_hasher.hash(edge);
            let slot = hash_lookup(
                &mut edge_table,
                hash,
                |a, b| edge_hasher.equal(*a, *b),
                edge,
                !0,
            );

            if edge_table[slot] == !0 {
                edge_table[slot] = edge;

                // store vertex opposite to the edge
                edge_vertex_table[slot] = i2;
            }
        }
        i += 3;
    }

    // build resulting index buffer: 6 indices for each input triangle
    let mut i = 0;
    while i < index_count {
        let mut patch = [0u32; 6];

        for e in 0..3 {
            let i0 = indices[i + e];
            let i1 = indices[i + NEXT[e]];
            debug_assert!(i0 < vertex_count as u32 && i1 < vertex_count as u32);

            // note: this refers to the opposite edge!
            let edge = (u64::from(i1) << 32) | u64::from(i0);
            let hash = edge_hasher.hash(edge);
            let slot = hash_lookup(
                &mut edge_table,
                hash,
                |a, b| edge_hasher.equal(*a, *b),
                edge,
                !0,
            );

            patch[e * 2] = i0;
            patch[e * 2 + 1] = if edge_table[slot] == !0 {
                i0
            } else {
                edge_vertex_table[slot]
            };
        }

        destination[i * 2..i * 2 + 6].copy_from_slice(&patch);
        i += 3;
    }
}

/// Upstream: `meshopt_generateTessellationIndexBuffer`.
pub fn generate_tessellation_index_buffer(
    destination: &mut [u32],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
    debug_assert!(destination.len() >= index_count * 4);

    const NEXT: [usize; 3] = [1, 2, 0];

    // build position remap: for each vertex, which other (canonical) vertex does it map to?
    let mut remap = filled(vertex_count, !0u32);
    build_position_remap(
        &mut remap,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
    );

    let edge_hasher = EdgeHasher { remap: &remap };

    let mut edge_table = filled(hash_buckets(index_count), !0u64);

    let mut i = 0;
    while i < index_count {
        for e in 0..3 {
            let i0 = indices[i + e];
            let i1 = indices[i + NEXT[e]];
            debug_assert!(i0 < vertex_count as u32 && i1 < vertex_count as u32);

            let edge = (u64::from(i0) << 32) | u64::from(i1);
            let hash = edge_hasher.hash(edge);
            let slot = hash_lookup(
                &mut edge_table,
                hash,
                |a, b| edge_hasher.equal(*a, *b),
                edge,
                !0,
            );

            if edge_table[slot] == !0 {
                edge_table[slot] = edge;
            }
        }
        i += 3;
    }

    // build resulting index buffer: 12 indices for each input triangle
    let mut i = 0;
    while i < index_count {
        let mut patch = [0u32; 12];

        for e in 0..3 {
            let i0 = indices[i + e];
            let i1 = indices[i + NEXT[e]];
            debug_assert!(i0 < vertex_count as u32 && i1 < vertex_count as u32);

            // note: this refers to the opposite edge!
            let edge = (u64::from(i1) << 32) | u64::from(i0);
            let hash = edge_hasher.hash(edge);
            let slot = hash_lookup(
                &mut edge_table,
                hash,
                |a, b| edge_hasher.equal(*a, *b),
                edge,
                !0,
            );
            let mut oppe = edge_table[slot];

            // use the same edge if opposite edge doesn't exist (border)
            if oppe == !0 {
                oppe = edge;
            }

            // triangle index (0, 1, 2)
            patch[e] = i0;

            // opposite edge (3, 4; 5, 6; 7, 8)
            patch[3 + e * 2] = oppe as u32;
            patch[3 + e * 2 + 1] = (oppe >> 32) as u32;

            // dominant vertex (9, 10, 11)
            patch[9 + e] = remap[i0 as usize];
        }

        destination[i * 4..i * 4 + 12].copy_from_slice(&patch);
        i += 3;
    }
}

/// Upstream: `meshopt_generateProvokingIndexBuffer`.
pub fn generate_provoking_index_buffer(
    destination: &mut [u32],
    reorder: &mut [u32],
    indices: &[u32],
    vertex_count: usize,
) -> usize {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));
    debug_assert!(destination.len() >= index_count);
    debug_assert!(reorder.len() >= vertex_count + index_count / 3);

    let mut remap = filled(vertex_count, !0u32);

    // compute vertex valence; this is used to prioritize least used corner
    // note: we use 8-bit counters for performance; for outlier vertices the valence is
    // incorrect but that just affects the heuristic, so the wrap is intentional
    let mut valence = filled(vertex_count, 0u8);

    for &index in indices.iter() {
        debug_assert!(index < vertex_count as u32);
        valence[index as usize] = valence[index as usize].wrapping_add(1);
    }

    let mut reorder_offset = 0usize;

    // assign provoking vertices; leave the rest for the next pass
    let mut i = 0;
    while i < index_count {
        let mut a = indices[i];
        let mut b = indices[i + 1];
        let mut c = indices[i + 2];
        debug_assert!(
            a < vertex_count as u32 && b < vertex_count as u32 && c < vertex_count as u32
        );

        // try to rotate triangle such that provoking vertex hasn't been seen before
        // if multiple vertices are new, prioritize the one with least valence
        // this reduces the risk that a future triangle will have all three vertices seen
        let va = if remap[a as usize] == !0 {
            u32::from(valence[a as usize])
        } else {
            !0
        };
        let vb = if remap[b as usize] == !0 {
            u32::from(valence[b as usize])
        } else {
            !0
        };
        let vc = if remap[c as usize] == !0 {
            u32::from(valence[c as usize])
        } else {
            !0
        };

        if vb != !0 && vb <= va && vb <= vc {
            // abc -> bca
            let t = a;
            a = b;
            b = c;
            c = t;
        } else if vc != !0 && vc <= va && vc <= vb {
            // abc -> cab
            let t = c;
            c = b;
            b = a;
            a = t;
        }

        let newidx = reorder_offset as u32;

        // now remap[a] = !0 or all three vertices are old
        // recording remap[a] makes it possible to remap future references to the same
        // index, conserving space
        if remap[a as usize] == !0 {
            remap[a as usize] = newidx;
        }

        // we need to clone the provoking vertex to get a unique index
        // if all three are used the choice is arbitrary since no future triangle will be
        // able to reuse any of these
        reorder[reorder_offset] = a;
        reorder_offset += 1;

        // note: first vertex is final, the other two will be fixed up in next pass
        destination[i] = newidx;
        destination[i + 1] = b;
        destination[i + 2] = c;

        // update vertex valences for corner heuristic
        valence[a as usize] = valence[a as usize].wrapping_sub(1);
        valence[b as usize] = valence[b as usize].wrapping_sub(1);
        valence[c as usize] = valence[c as usize].wrapping_sub(1);

        i += 3;
    }

    // remap or clone non-provoking vertices (iterating to skip provoking vertices)
    let mut step = 1usize;

    let mut i = 1;
    while i < index_count {
        let index = destination[i];

        if remap[index as usize] == !0 {
            // we haven't seen the vertex before as a provoking vertex
            // to maintain the reference to the original vertex we need to clone it
            let newidx = reorder_offset as u32;

            remap[index as usize] = newidx;
            reorder[reorder_offset] = index;
            reorder_offset += 1;
        }

        destination[i] = remap[index as usize];

        i += step;
        step ^= 3;
    }

    debug_assert!(reorder_offset <= vertex_count + index_count / 3);
    reorder_offset
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_generateVertexRemap,
    // meshopt_generateVertexRemapMulti, meshopt_generateVertexRemapCustom,
    // meshopt_remapVertexBuffer, meshopt_remapIndexBuffer,
    // meshopt_filterIndexBuffer, meshopt_filterIndexBufferMulti,
    // meshopt_generateShadowIndexBuffer, meshopt_generateShadowIndexBufferMulti,
    // meshopt_generatePositionRemap, meshopt_generateAdjacencyIndexBuffer,
    // meshopt_generateTessellationIndexBuffer,
    // meshopt_generateProvokingIndexBuffer.
    use super::*;

    struct Cursor<'a> {
        data: &'a [u8],
        pos: usize,
    }

    impl<'a> Cursor<'a> {
        fn new(data: &'a [u8]) -> Self {
            Cursor { data, pos: 0 }
        }

        fn take(&mut self, n: usize) -> &'a [u8] {
            let s = &self.data[self.pos..self.pos + n];
            self.pos += n;
            s
        }

        fn u32(&mut self) -> u32 {
            let mut a = [0u8; 4];
            a.copy_from_slice(self.take(4));
            u32::from_le_bytes(a)
        }

        fn bytes(&mut self, n: usize) -> &'a [u8] {
            self.take(n)
        }

        fn vec_u32(&mut self, n: usize) -> Vec<u32> {
            (0..n).map(|_| self.u32()).collect()
        }

        fn f32s(&mut self, byte_len: usize) -> Vec<f32> {
            let raw = self.take(byte_len);
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect()
        }

        // the driver always records the index sequence (identity values when the
        // upstream call used NULL indices)
        fn indices(&mut self, index_count: usize, has_indices: bool) -> Option<Vec<u32>> {
            let values = self.vec_u32(index_count);
            has_indices.then_some(values)
        }
    }

    fn stream_from(data: &[u8], size: usize, stride: usize) -> Stream<'_> {
        Stream { data, size, stride }
    }

    #[test]
    fn golden_generate_vertex_remap() {
        const _LEN_CHECK: usize = 53820;
        let data = include_bytes!("../tests/data/indexgen_remap.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let vertex_size = c.u32() as usize;
            let index_count = c.u32() as usize;
            let has_indices = c.u32() == 1;
            let vertices = c.bytes(vertex_count * vertex_size);
            let indices = c.indices(index_count, has_indices);
            let expected_remap = c.vec_u32(vertex_count);
            let expected_result = c.u32() as usize;

            let mut destination = vec![0xeeu32; vertex_count];
            let result = generate_vertex_remap(
                &mut destination,
                indices.as_deref(),
                vertices,
                vertex_count,
                vertex_size,
            );
            assert_eq!(destination, expected_remap, "remap v={vertex_count}");
            assert_eq!(result, expected_result, "result v={vertex_count}");
        }
    }

    #[test]
    fn golden_generate_vertex_remap_multi() {
        const _LEN_CHECK: usize = 20776;
        let data = include_bytes!("../tests/data/indexgen_remap_multi.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stream_count = c.u32() as usize;
            let index_count = c.u32() as usize;
            let has_indices = c.u32() == 1;
            let mut streams: Vec<Stream<'_>> = Vec::new();
            for _ in 0..stream_count {
                let size = c.u32() as usize;
                let stride = c.u32() as usize;
                let buf = c.bytes(vertex_count * stride);
                streams.push(stream_from(buf, size, stride));
            }
            let indices = c.indices(index_count, has_indices);
            let expected_remap = c.vec_u32(vertex_count);
            let expected_result = c.u32() as usize;

            let mut destination = vec![0xeeu32; vertex_count];
            let result = generate_vertex_remap_multi(
                &mut destination,
                indices.as_deref(),
                vertex_count,
                &streams,
            );
            assert_eq!(destination, expected_remap, "remap v={vertex_count}");
            assert_eq!(result, expected_result, "result v={vertex_count}");
        }
    }

    #[test]
    fn golden_generate_vertex_remap_custom() {
        const _LEN_CHECK: usize = 7304;
        let data = include_bytes!("../tests/data/indexgen_remap_custom.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let has_indices = c.u32() == 1;
            let positions = c.f32s(vertex_count * stride);
            let indices = c.indices(index_count, has_indices);
            let expected_remap = c.vec_u32(vertex_count);
            let expected_result = c.u32() as usize;

            let mut destination = vec![0xeeu32; vertex_count];
            let result = generate_vertex_remap_custom(
                &mut destination,
                indices.as_deref(),
                &positions,
                vertex_count,
                stride,
                None,
            );
            assert_eq!(destination, expected_remap, "remap v={vertex_count}");
            assert_eq!(result, expected_result, "result v={vertex_count}");
        }
    }

    #[test]
    fn golden_remap_vertex_buffer() {
        const _LEN_CHECK: usize = 15020;
        let data = include_bytes!("../tests/data/indexgen_remap_vb.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let vertex_size = c.u32() as usize;
            let index_count = c.u32() as usize;
            let vertices = c.bytes(vertex_count * vertex_size);
            let _indices = c.vec_u32(index_count);
            let remap = c.vec_u32(vertex_count);
            let expected = c.bytes(vertex_count * vertex_size);

            let mut destination = vec![0xccu8; vertex_count * vertex_size];
            remap_vertex_buffer(
                &mut destination,
                vertices,
                vertex_count,
                vertex_size,
                &remap,
            );
            assert_eq!(destination.as_slice(), expected, "vb v={vertex_count}");
        }
    }

    #[test]
    fn golden_remap_index_buffer() {
        const _LEN_CHECK: usize = 11432;
        let data = include_bytes!("../tests/data/indexgen_remap_ib.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let index_count = c.u32() as usize;
            let has_indices = c.u32() == 1;
            let remap = c.vec_u32(vertex_count);
            // this fixture omits the index sequence entirely for the identity case
            let indices = has_indices.then(|| c.vec_u32(index_count));
            let expected = c.vec_u32(index_count);

            let mut destination = vec![0xeeu32; index_count];
            remap_index_buffer(&mut destination, indices.as_deref(), &remap);
            assert_eq!(destination, expected, "ib ic={index_count}");
        }
    }

    #[test]
    fn golden_filter_index_buffer() {
        const _LEN_CHECK: usize = 14376;
        let data = include_bytes!("../tests/data/indexgen_filter_ib.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let vertex_size = c.u32() as usize;
            let stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let vertices = c.bytes(vertex_count * stride);
            let indices = c.vec_u32(index_count);
            let expected = c.vec_u32(index_count);
            let expected_written = c.u32() as usize;

            let mut destination = vec![0xeeu32; index_count];
            let written = filter_index_buffer(
                &mut destination,
                &indices,
                vertices,
                vertex_count,
                vertex_size,
                stride,
            );
            assert_eq!(destination, expected, "filter v={vertex_count}");
            assert_eq!(written, expected_written, "written v={vertex_count}");
        }
    }

    #[test]
    fn golden_filter_index_buffer_multi() {
        const _LEN_CHECK: usize = 10068;
        let data = include_bytes!("../tests/data/indexgen_filter_multi.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stream_count = c.u32() as usize;
            let index_count = c.u32() as usize;
            let _has_indices = c.u32() == 1;
            let mut streams: Vec<Stream<'_>> = Vec::new();
            for _ in 0..stream_count {
                let size = c.u32() as usize;
                let stride = c.u32() as usize;
                let buf = c.bytes(vertex_count * stride);
                streams.push(stream_from(buf, size, stride));
            }
            let indices = c.vec_u32(index_count);
            let expected = c.vec_u32(index_count);
            let expected_written = c.u32() as usize;

            let mut destination = vec![0xeeu32; index_count];
            let written =
                filter_index_buffer_multi(&mut destination, &indices, vertex_count, &streams);
            assert_eq!(destination, expected, "filter v={vertex_count}");
            assert_eq!(written, expected_written, "written v={vertex_count}");
        }
    }

    #[test]
    fn golden_generate_shadow_index_buffer() {
        const _LEN_CHECK: usize = 14768;
        let data = include_bytes!("../tests/data/indexgen_shadow.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let vertex_size = c.u32() as usize;
            let stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let vertices = c.bytes(vertex_count * stride);
            let indices = c.vec_u32(index_count);
            let expected = c.vec_u32(index_count);

            let mut destination = vec![0xeeu32; index_count];
            generate_shadow_index_buffer(
                &mut destination,
                &indices,
                vertices,
                vertex_count,
                vertex_size,
                stride,
            );
            assert_eq!(destination, expected, "shadow v={vertex_count}");
        }
    }

    #[test]
    fn golden_generate_shadow_index_buffer_multi() {
        const _LEN_CHECK: usize = 10572;
        let data = include_bytes!("../tests/data/indexgen_shadow_multi.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stream_count = c.u32() as usize;
            let index_count = c.u32() as usize;
            let _has_indices = c.u32() == 1;
            let mut streams: Vec<Stream<'_>> = Vec::new();
            for _ in 0..stream_count {
                let size = c.u32() as usize;
                let stride = c.u32() as usize;
                let buf = c.bytes(vertex_count * stride);
                streams.push(stream_from(buf, size, stride));
            }
            let indices = c.vec_u32(index_count);
            let expected = c.vec_u32(index_count);

            let mut destination = vec![0xeeu32; index_count];
            generate_shadow_index_buffer_multi(&mut destination, &indices, vertex_count, &streams);
            assert_eq!(destination, expected, "shadow v={vertex_count}");
        }
    }

    #[test]
    fn golden_generate_position_remap() {
        const _LEN_CHECK: usize = 2980;
        let data = include_bytes!("../tests/data/indexgen_pos_remap.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let positions = c.f32s(vertex_count * stride);
            let expected = c.vec_u32(vertex_count);

            let mut destination = vec![0xeeu32; vertex_count];
            generate_position_remap(&mut destination, &positions, vertex_count, stride);
            assert_eq!(destination, expected, "posremap v={vertex_count}");
        }
    }

    #[test]
    fn golden_generate_adjacency_index_buffer() {
        const _LEN_CHECK: usize = 11344;
        let data = include_bytes!("../tests/data/indexgen_adjacency.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let positions = c.f32s(vertex_count * stride);
            let indices = c.vec_u32(index_count);
            let expected = c.vec_u32(index_count * 2);

            let mut destination = vec![0xeeu32; index_count * 2];
            generate_adjacency_index_buffer(
                &mut destination,
                &indices,
                &positions,
                vertex_count,
                stride,
            );
            assert_eq!(destination, expected, "adjacency v={vertex_count}");
        }
    }

    #[test]
    fn golden_generate_tessellation_index_buffer() {
        const _LEN_CHECK: usize = 17344;
        let data = include_bytes!("../tests/data/indexgen_tess.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let positions = c.f32s(vertex_count * stride);
            let indices = c.vec_u32(index_count);
            let expected = c.vec_u32(index_count * 4);

            let mut destination = vec![0xeeu32; index_count * 4];
            generate_tessellation_index_buffer(
                &mut destination,
                &indices,
                &positions,
                vertex_count,
                stride,
            );
            assert_eq!(destination, expected, "tess v={vertex_count}");
        }
    }

    #[test]
    fn golden_generate_provoking_index_buffer() {
        const _LEN_CHECK: usize = 4504;
        let data = include_bytes!("../tests/data/indexgen_provoking.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let index_count = c.u32() as usize;
            let indices = c.vec_u32(index_count);
            let expected = c.vec_u32(index_count);
            let reorder_offset = c.u32() as usize;
            let expected_reorder = c.vec_u32(reorder_offset);

            let mut destination = vec![0xeeu32; index_count];
            let mut reorder = vec![0xeeu32; vertex_count + index_count / 3 + 1];
            let result = generate_provoking_index_buffer(
                &mut destination,
                &mut reorder,
                &indices,
                vertex_count,
            );
            assert_eq!(destination, expected, "provoking v={vertex_count}");
            assert_eq!(result, reorder_offset, "reorder_offset v={vertex_count}");
            assert_eq!(
                &reorder[..result],
                expected_reorder.as_slice(),
                "reorder v={vertex_count}"
            );
        }
    }

    #[test]
    fn custom_callback_semantics() {
        // v0 and v1 are float-equal; v2 is distinct
        let positions = [1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut keep_distinct = |_: u32, _: u32| false;
        let mut d = vec![0xeeu32; 3];
        let n = generate_vertex_remap_custom(
            &mut d,
            None,
            &positions,
            3,
            12,
            Some(&mut keep_distinct as &mut dyn FnMut(u32, u32) -> bool),
        );
        assert_eq!((n, d.as_slice()), (3, &[0, 1, 2][..]));

        let mut merge = |_: u32, _: u32| true;
        let mut d = vec![0xeeu32; 3];
        let n = generate_vertex_remap_custom(
            &mut d,
            None,
            &positions,
            3,
            12,
            Some(&mut merge as &mut dyn FnMut(u32, u32) -> bool),
        );
        assert_eq!((n, d.as_slice()), (2, &[0, 0, 1][..]));

        // negative zero merges with positive zero via float equality
        let nz = [0.0f32, 0.0, 0.0, -0.0, 0.0, 0.0];
        let mut d = vec![0xeeu32; 2];
        let n = generate_vertex_remap_custom(&mut d, None, &nz, 2, 12, None);
        assert_eq!((n, d.as_slice()), (1, &[0, 0][..]));
    }
}
