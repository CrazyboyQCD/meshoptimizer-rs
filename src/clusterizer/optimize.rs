//! Meshlet optimization and meshlet-local index extraction (`meshletutils.cpp`
//! entry points, re-homed under `clusterizer` per design.md §4).

use crate::meshletutils::{MESHLET_MAX_TRIANGLES, MESHLET_MAX_VERTICES};

/// Upstream counterpart: `meshopt_optimizeMeshlet` (level 0).
pub fn optimize_meshlet(meshlet_vertices: &mut [u32], meshlet_triangles: &mut [u8]) {
    optimize_meshlet_level(meshlet_vertices, meshlet_triangles, 0);
}

/// Upstream counterpart: `meshopt_optimizeMeshletLevel`.
pub fn optimize_meshlet_level(
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    level: i32,
) {
    let vertex_count = meshlet_vertices.len();
    let triangle_count = meshlet_triangles.len() / 3;
    debug_assert!(triangle_count <= MESHLET_MAX_TRIANGLES);
    debug_assert!(vertex_count <= MESHLET_MAX_VERTICES);
    debug_assert!((0..=9).contains(&level));

    let indices = meshlet_triangles;
    let vertices = meshlet_vertices;

    // cache tracks vertex timestamps (corresponding to triangle index! all 3 vertices are added at the same time and never removed)
    let mut cache = [0u8; MESHLET_MAX_VERTICES];

    // note that we start from a value that means all vertices aren't in cache
    let mut cache_last: u8 = 128;
    const CACHE_CUTOFF: u8 = 3; // 3 triangles = ~5..9 vertices depending on reuse

    // vertex valence is used to prioritize triangles for level>0
    // note: we use 8-bit counters for performance; for outlier vertices the valence is incorrect but that just affects the heuristic
    let mut valence = [0u8; MESHLET_MAX_VERTICES];

    for tri in indices[..triangle_count * 3].as_chunks::<3>().0 {
        let a = tri[0] as usize;
        let b = tri[1] as usize;
        let c = tri[2] as usize;

        valence[a] = valence[a].wrapping_add(1);
        valence[b] = valence[b].wrapping_add(1);
        valence[c] = valence[c].wrapping_add(1);
    }

    for i in 0..triangle_count {
        let mut next: i32 = -1;
        let mut next_score: i32 = -1;
        let mut edges: i32 = 0;

        let mut j = i;
        while j < triangle_count {
            let [a, b, c] = indices.as_chunks::<3>().0[j];
            let (a, b, c) = (a as usize, b as usize, c as usize);

            // compute cache distance using unsigned 8-bit subtraction, so cache timestamp overflow is handled gracefully
            let ad = cache_last.wrapping_sub(cache[a]);
            let bd = cache_last.wrapping_sub(cache[b]);
            let cd = cache_last.wrapping_sub(cache[c]);

            let matches = i32::from(ad < CACHE_CUTOFF)
                + i32::from(bd < CACHE_CUTOFF)
                + i32::from(cd < CACHE_CUTOFF);

            if level != 0 {
                // prefer low minimum valence
                let vmin = (valence[a].min(valence[b])).min(valence[c]);

                // prefer vertices with smaller cache distance and valence to improve traversal locality
                let mut score =
                    matches * 1024 + (1023 - i32::from(ad) - i32::from(bd) - i32::from(cd));
                score = score * 256 + (255 - i32::from(vmin));

                if score > next_score {
                    next = j as i32;
                    next_score = score;
                }

                // terminate after finding enough edge matches
                if matches >= 2 {
                    edges += 1;
                    if edges >= level {
                        break;
                    }
                }
            } else {
                let score = matches;

                if score > next_score {
                    next = j as i32;
                    next_score = score;
                }

                // settle for a first edge match, which makes the function ~linear in practice
                if matches >= 2 {
                    break;
                }
            }

            j += 1;
        }

        let next = next as usize;

        let [a, b, c] = indices.as_chunks::<3>().0[next];

        // shift triangles before the next one forward so that we always keep an ordered partition
        // note: this could have swapped triangles [i] and [next] but that distorts the order and may skew the output sequence
        indices.copy_within(i * 3..next * 3, (i + 1) * 3);

        indices[i * 3] = a;
        indices[i * 3 + 1] = b;
        indices[i * 3 + 2] = c;

        let a = a as usize;
        let b = b as usize;
        let c = c as usize;

        // cache timestamp is the same between all vertices of each triangle to reduce overflow
        cache_last = cache_last.wrapping_add(1);
        cache[a] = cache_last;
        cache[b] = cache_last;
        cache[c] = cache_last;

        // update vertex valences for scoring heuristic
        valence[a] = valence[a].wrapping_sub(1);
        valence[b] = valence[b].wrapping_sub(1);
        valence[c] = valence[c].wrapping_sub(1);
    }

    // rotate triangles to maximize compressibility; only done at level >= 1 for compatibility
    if level >= 1 {
        cache[..vertex_count].fill(0);

        let chunks = &mut indices.as_chunks_mut::<3>().0[..triangle_count];
        for i in 0..triangle_count {
            let [mut a, mut b, mut c] = chunks[i];

            // if only the middle vertex has been used, rotate triangle to ensure new vertices are always sequential
            if cache[a as usize] == 0 && cache[b as usize] != 0 && cache[c as usize] == 0 {
                // abc -> bca
                (a, b, c) = (b, c, a);
            } else if cache[a as usize] == 0 && cache[b as usize] == 0 && cache[c as usize] == 0 {
                // out of three edges, the edge ab can not be reused by subsequent triangles in some encodings
                // if subsequent triangles don't share edges ca or bc, we can rotate the triangle to fix this
                let mut needab = false;
                let mut needbc = false;
                let mut needca = false;

                let mut j = i + 1;
                while j < triangle_count && j <= i + CACHE_CUTOFF as usize {
                    let [oa, ob, oc] = chunks[j];

                    // note: edge comparisons are reversed as reused edges are flipped
                    needab |= (oa == b && ob == a) || (ob == b && oc == a) || (oc == b && oa == a);
                    needbc |= (oa == c && ob == b) || (ob == c && oc == b) || (oc == c && oa == b);
                    needca |= (oa == a && ob == c) || (ob == a && oc == c) || (oc == a && oa == c);
                    j += 1;
                }

                if needab && !needbc {
                    // abc -> bca
                    (a, b, c) = (b, c, a);
                } else if needab && !needca {
                    // abc -> cab
                    (a, b, c) = (c, a, b);
                } else if !needab && needbc && !needca {
                    // abc -> cab: put bc at most recent FIFO position
                    (a, b, c) = (c, a, b);
                }
            }

            chunks[i] = [a, b, c];

            cache[a as usize] = 1;
            cache[b as usize] = 1;
            cache[c as usize] = 1;
        }
    }

    // reorder meshlet vertices for access locality assuming index buffer is scanned sequentially
    let mut order = [0u32; MESHLET_MAX_VERTICES];

    let mut remap = [-1i16; MESHLET_MAX_VERTICES];

    let mut vertex_offset = 0usize;

    for idx_byte in &mut indices[..triangle_count * 3] {
        let idx = *idx_byte as usize;
        let r = remap[idx];

        if r < 0 {
            remap[idx] = vertex_offset as i16;
            order[vertex_offset] = vertices[idx];
            vertex_offset += 1;
        }

        *idx_byte = remap[idx] as u8;
    }

    debug_assert!(vertex_offset <= vertex_count);
    vertices[..vertex_offset].copy_from_slice(&order[..vertex_offset]);
}

/// Upstream counterpart: `meshopt_extractMeshletIndices`.
pub fn extract_meshlet_indices(
    vertices: &mut [u32],
    triangles: &mut [u8],
    indices: &[u32],
) -> usize {
    debug_assert!(indices.len().is_multiple_of(3));
    debug_assert!(indices.len() / 3 <= MESHLET_MAX_TRIANGLES);

    let mut unique = 0usize;

    // direct mapped cache for fast lookups based on low index bits; inspired by vk_lod_clusters from NVIDIA
    let mut cache = [-1i16; 1024];

    for (i, &v) in indices.iter().enumerate() {
        let key = v as usize & 1023;
        let c = cache[key];

        // fast path: vertex has been seen before
        if c >= 0 && vertices[c as usize] == v {
            triangles[i] = c as u8;
            continue;
        }

        // fast path: vertex has never been seen before
        if c < 0 {
            debug_assert!(unique < MESHLET_MAX_VERTICES);
            cache[key] = unique as i16;
            triangles[i] = unique as u8;
            vertices[unique] = v;
            unique += 1;
            continue;
        }

        // slow path: collision with a different vertex, so we need to look through all vertices
        let pos = vertices[..unique]
            .iter()
            .position(|&u| u == v)
            .map_or(-1, |p| p as i32);

        if pos < 0 {
            debug_assert!(unique < MESHLET_MAX_VERTICES);
            cache[key] = unique as i16;
            triangles[i] = unique as u8;
            vertices[unique] = v;
            unique += 1;
            continue;
        }

        cache[key] = pos as i16;
        triangles[i] = pos as u8;
    }

    unique
}
