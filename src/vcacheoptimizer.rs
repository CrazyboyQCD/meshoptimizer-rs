//! Vertex cache reordering (Forsyth-style triangle scoring and FIFO variant).
//!
//! Upstream: `vcacheoptimizer.cpp` (`meshopt_optimizeVertexCache`,
//! `meshopt_optimizeVertexCacheStrip`, `meshopt_optimizeVertexCacheFifo`).

use alloc::vec;
use alloc::vec::Vec;

const K_CACHE_SIZE_MAX: usize = 16;
const K_VALENCE_MAX: usize = 8;

struct VertexScoreTable {
    cache: [f32; 1 + K_CACHE_SIZE_MAX],
    live: [f32; 1 + K_VALENCE_MAX],
}

// Tuned to minimize the ACMR of a GPU that has a cache profile similar to NVidia and AMD
static K_VERTEX_SCORE_TABLE: VertexScoreTable = VertexScoreTable {
    cache: [
        0.0, 0.779, 0.791, 0.789, 0.981, 0.843, 0.726, 0.847, 0.882, 0.867, 0.799, 0.642, 0.613,
        0.600, 0.568, 0.372, 0.234,
    ],
    live: [0.0, 0.995, 0.713, 0.450, 0.404, 0.059, 0.005, 0.147, 0.006],
};

// Tuned to minimize the encoded index buffer size
static K_VERTEX_SCORE_TABLE_STRIP: VertexScoreTable = VertexScoreTable {
    cache: [
        0.0, 1.000, 1.000, 1.000, 0.453, 0.561, 0.490, 0.459, 0.179, 0.526, 0.000, 0.227, 0.184,
        0.490, 0.112, 0.050, 0.131,
    ],
    live: [0.0, 0.956, 0.786, 0.577, 0.558, 0.618, 0.549, 0.499, 0.489],
};

struct TriangleAdjacency {
    counts: Vec<u32>,
    offsets: Vec<u32>,
    data: Vec<u32>,
}

fn build_triangle_adjacency(indices: &[u32], vertex_count: usize) -> TriangleAdjacency {
    let index_count = indices.len();
    let face_count = index_count / 3;

    let mut counts = vec![0u32; vertex_count];
    let mut offsets = vec![0u32; vertex_count];
    let mut data = vec![0u32; index_count];

    for &index in indices.iter() {
        debug_assert!(index < vertex_count as u32);
        counts[index as usize] += 1;
    }

    let mut offset = 0u32;
    for i in 0..vertex_count {
        offsets[i] = offset;
        offset += counts[i];
    }

    debug_assert!(offset as usize == index_count);

    for face in 0..face_count {
        let a = indices[face * 3];
        let b = indices[face * 3 + 1];
        let c = indices[face * 3 + 2];

        data[offsets[a as usize] as usize] = face as u32;
        offsets[a as usize] += 1;
        data[offsets[b as usize] as usize] = face as u32;
        offsets[b as usize] += 1;
        data[offsets[c as usize] as usize] = face as u32;
        offsets[c as usize] += 1;
    }

    // fix offsets that have been disturbed by the previous pass
    for i in 0..vertex_count {
        debug_assert!(offsets[i] >= counts[i]);
        offsets[i] -= counts[i];
    }

    TriangleAdjacency {
        counts,
        offsets,
        data,
    }
}

fn vertex_score(table: &VertexScoreTable, cache_position: i32, live_triangles: u32) -> f32 {
    debug_assert!((-1..K_CACHE_SIZE_MAX as i32).contains(&cache_position));

    let live_triangles_clamped = (live_triangles as usize).min(K_VALENCE_MAX);

    table.cache[(1 + cache_position) as usize] + table.live[live_triangles_clamped]
}

fn get_next_vertex_dead_end(
    dead_end: &[u32],
    dead_end_top: &mut usize,
    input_cursor: &mut u32,
    live_triangles: &[u32],
    vertex_count: usize,
) -> u32 {
    // check dead-end stack
    while *dead_end_top != 0 {
        *dead_end_top -= 1;
        let vertex = dead_end[*dead_end_top];

        if live_triangles[vertex as usize] > 0 {
            return vertex;
        }
    }

    // input order
    while (*input_cursor as usize) < vertex_count {
        if live_triangles[*input_cursor as usize] > 0 {
            return *input_cursor;
        }

        *input_cursor += 1;
    }

    !0
}

fn get_next_vertex_neighbor(
    candidates: &[u32],
    live_triangles: &[u32],
    cache_timestamps: &[u32],
    timestamp: u32,
    cache_size: u32,
) -> u32 {
    let mut best_candidate = !0u32;
    let mut best_priority = -1i32;

    for &vertex in candidates.iter() {
        // otherwise we don't need to process it
        if live_triangles[vertex as usize] > 0 {
            let mut priority = 0i32;

            // will it be in cache after fanning?
            if 2 * live_triangles[vertex as usize] + timestamp - cache_timestamps[vertex as usize]
                <= cache_size
            {
                priority = (timestamp - cache_timestamps[vertex as usize]) as i32; // position in cache
            }

            if priority > best_priority {
                best_candidate = vertex;
                best_priority = priority;
            }
        }
    }

    best_candidate
}

fn get_next_triangle_dead_end(
    input_cursor: &mut u32,
    emitted_flags: &[u8],
    face_count: usize,
) -> u32 {
    // input order
    while (*input_cursor as usize) < face_count {
        if emitted_flags[*input_cursor as usize] == 0 {
            return *input_cursor;
        }

        *input_cursor += 1;
    }

    !0
}

fn optimize_vertex_cache_table(
    destination: &mut [u32],
    indices: &[u32],
    vertex_count: usize,
    table: &VertexScoreTable,
) {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));

    // guard for empty meshes
    if index_count == 0 || vertex_count == 0 {
        return;
    }

    let cache_size = K_CACHE_SIZE_MAX;

    let face_count = index_count / 3;

    // build adjacency information; live triangle counts alias the counts array since
    // triangles are removed from adjacency as they are emitted
    let TriangleAdjacency {
        counts: mut live_triangles,
        offsets,
        mut data,
    } = build_triangle_adjacency(indices, vertex_count);

    // emitted flags
    let mut emitted_flags = vec![0u8; face_count];

    // compute initial vertex scores
    let mut vertex_scores = vec![0f32; vertex_count];

    for i in 0..vertex_count {
        vertex_scores[i] = vertex_score(table, -1, live_triangles[i]);
    }

    // compute triangle scores
    let mut triangle_scores = vec![0f32; face_count];

    for i in 0..face_count {
        let a = indices[i * 3];
        let b = indices[i * 3 + 1];
        let c = indices[i * 3 + 2];

        triangle_scores[i] =
            vertex_scores[a as usize] + vertex_scores[b as usize] + vertex_scores[c as usize];
    }

    let mut cache_holder = [0u32; 2 * (K_CACHE_SIZE_MAX + 4)];
    let (cache_a, cache_b) = cache_holder.split_at_mut(K_CACHE_SIZE_MAX + 4);
    let mut cache = cache_a;
    let mut cache_new = cache_b;
    let mut cache_count = 0usize;

    let mut current_triangle = 0u32;
    let mut input_cursor = 1u32;

    let mut output_triangle = 0usize;

    while current_triangle != !0u32 {
        debug_assert!(output_triangle < face_count);

        let a = indices[current_triangle as usize * 3];
        let b = indices[current_triangle as usize * 3 + 1];
        let c = indices[current_triangle as usize * 3 + 2];

        // output indices
        destination[output_triangle * 3] = a;
        destination[output_triangle * 3 + 1] = b;
        destination[output_triangle * 3 + 2] = c;
        output_triangle += 1;

        // update emitted flags
        emitted_flags[current_triangle as usize] = 1;
        triangle_scores[current_triangle as usize] = 0.0;

        // new triangle
        let mut cache_write = 0usize;
        cache_new[cache_write] = a;
        cache_write += 1;
        cache_new[cache_write] = b;
        cache_write += 1;
        cache_new[cache_write] = c;
        cache_write += 1;

        // old triangles
        for &index in cache[..cache_count].iter() {
            cache_new[cache_write] = index;
            cache_write += usize::from(index != a && index != b && index != c);
        }

        core::mem::swap(&mut cache, &mut cache_new);
        cache_count = cache_write.min(cache_size);

        // remove emitted triangle from adjacency data
        // this makes sure that we spend less time traversing these lists on subsequent
        // iterations; live triangle counts are updated as a byproduct of these adjustments
        for k in 0..3 {
            let index = indices[current_triangle as usize * 3 + k] as usize;

            let neighbors_begin = offsets[index] as usize;
            let neighbors_size = live_triangles[index] as usize;

            for i in 0..neighbors_size {
                let tri = data[neighbors_begin + i];

                if tri == current_triangle {
                    data[neighbors_begin + i] = data[neighbors_begin + neighbors_size - 1];
                    live_triangles[index] -= 1;
                    break;
                }
            }
        }

        let mut best_triangle = !0u32;
        let mut best_score = 0f32;

        // update cache positions, vertex scores and triangle scores, and find next best triangle
        for (position, &cache_index) in cache[..cache_write].iter().enumerate() {
            let index = cache_index as usize;

            // no need to update scores if we are never going to use this vertex
            if live_triangles[index] == 0 {
                continue;
            }

            let cache_position = if position >= cache_size {
                -1
            } else {
                position as i32
            };

            // update vertex score
            let score = vertex_score(table, cache_position, live_triangles[index]);
            let score_diff = score - vertex_scores[index];

            vertex_scores[index] = score;

            // update scores of vertex triangles
            let neighbors_begin = offsets[index] as usize;
            let neighbors_size = live_triangles[index] as usize;

            for ni in 0..neighbors_size {
                let tri = data[neighbors_begin + ni];
                debug_assert!(emitted_flags[tri as usize] == 0);

                let tri_score = triangle_scores[tri as usize] + score_diff;
                debug_assert!(tri_score > 0.0);

                if best_score < tri_score {
                    best_triangle = tri;
                    best_score = tri_score;
                }

                triangle_scores[tri as usize] = tri_score;
            }
        }

        // step through input triangles in order if we hit a dead-end
        current_triangle = best_triangle;

        if current_triangle == !0u32 {
            current_triangle =
                get_next_triangle_dead_end(&mut input_cursor, &emitted_flags, face_count);
        }
    }

    debug_assert!(input_cursor as usize == face_count);
    debug_assert!(output_triangle == face_count);
}

/// Upstream: `meshopt_optimizeVertexCache`.
pub fn optimize_vertex_cache(destination: &mut [u32], indices: &[u32], vertex_count: usize) {
    optimize_vertex_cache_table(destination, indices, vertex_count, &K_VERTEX_SCORE_TABLE);
}

/// Upstream: `meshopt_optimizeVertexCacheStrip`.
pub fn optimize_vertex_cache_strip(destination: &mut [u32], indices: &[u32], vertex_count: usize) {
    optimize_vertex_cache_table(
        destination,
        indices,
        vertex_count,
        &K_VERTEX_SCORE_TABLE_STRIP,
    );
}

/// Upstream: `meshopt_optimizeVertexCacheFifo`.
pub fn optimize_vertex_cache_fifo(
    destination: &mut [u32],
    indices: &[u32],
    vertex_count: usize,
    cache_size: u32,
) {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));
    debug_assert!(cache_size >= 3);

    // guard for empty meshes
    if index_count == 0 || vertex_count == 0 {
        return;
    }

    let face_count = index_count / 3;

    // build adjacency information
    let TriangleAdjacency {
        offsets,
        data,
        counts,
    } = build_triangle_adjacency(indices, vertex_count);

    // live triangle counts
    let mut live_triangles = counts.clone();

    // cache time stamps
    let mut cache_timestamps = vec![0u32; vertex_count];

    // dead-end stack
    let mut dead_end = vec![0u32; index_count];
    let mut dead_end_top = 0usize;

    // emitted flags
    let mut emitted_flags = vec![0u8; face_count];

    let mut current_vertex = 0u32;

    let mut timestamp = cache_size + 1;
    let mut input_cursor = 1u32; // vertex to restart from in case of dead-end

    let mut output_triangle = 0usize;

    while current_vertex != !0u32 {
        let next_candidates_begin = dead_end_top;

        // emit all vertex neighbors
        let cv = current_vertex as usize;
        let neighbors_begin = offsets[cv] as usize;
        let neighbors_size = counts[cv] as usize;

        for ni in 0..neighbors_size {
            let triangle = data[neighbors_begin + ni];

            if emitted_flags[triangle as usize] == 0 {
                let a = indices[triangle as usize * 3];
                let b = indices[triangle as usize * 3 + 1];
                let c = indices[triangle as usize * 3 + 2];

                // output indices
                destination[output_triangle * 3] = a;
                destination[output_triangle * 3 + 1] = b;
                destination[output_triangle * 3 + 2] = c;
                output_triangle += 1;

                // update dead-end stack
                dead_end[dead_end_top] = a;
                dead_end[dead_end_top + 1] = b;
                dead_end[dead_end_top + 2] = c;
                dead_end_top += 3;

                // update live triangle counts
                live_triangles[a as usize] -= 1;
                live_triangles[b as usize] -= 1;
                live_triangles[c as usize] -= 1;

                // update cache info
                // if vertex is not in cache, put it in cache
                if timestamp - cache_timestamps[a as usize] > cache_size {
                    cache_timestamps[a as usize] = timestamp;
                    timestamp += 1;
                }

                if timestamp - cache_timestamps[b as usize] > cache_size {
                    cache_timestamps[b as usize] = timestamp;
                    timestamp += 1;
                }

                if timestamp - cache_timestamps[c as usize] > cache_size {
                    cache_timestamps[c as usize] = timestamp;
                    timestamp += 1;
                }

                // update emitted flags
                emitted_flags[triangle as usize] = 1;
            }
        }

        // next candidates are the ones we pushed to dead-end stack just now
        let next_candidates_end = dead_end_top;

        // get next vertex
        current_vertex = get_next_vertex_neighbor(
            &dead_end[next_candidates_begin..next_candidates_end],
            &live_triangles,
            &cache_timestamps,
            timestamp,
            cache_size,
        );

        if current_vertex == !0u32 {
            current_vertex = get_next_vertex_dead_end(
                &dead_end,
                &mut dead_end_top,
                &mut input_cursor,
                &live_triangles,
                vertex_count,
            );
        }
    }

    debug_assert!(output_triangle == face_count);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    struct Cursor<'a> {
        data: &'a [u8],
        pos: usize,
    }

    impl<'a> Cursor<'a> {
        fn new(data: &'a [u8]) -> Self {
            Cursor { data, pos: 0 }
        }

        fn u32(&mut self) -> u32 {
            let mut a = [0u8; 4];
            a.copy_from_slice(&self.data[self.pos..self.pos + 4]);
            self.pos += 4;
            u32::from_le_bytes(a)
        }

        fn vec_u32(&mut self, n: usize) -> Vec<u32> {
            (0..n).map(|_| self.u32()).collect()
        }
    }

    #[test]
    fn golden_optimize_vertex_cache() {
        const _LEN_CHECK: usize = 9944;
        let data = include_bytes!("../tests/data/vcache.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let index_count = c.u32() as usize;
            let vertex_count = c.u32() as usize;
            let indices = c.vec_u32(index_count);
            let expected = c.vec_u32(index_count);

            let mut destination = vec![0xeeu32; index_count];
            optimize_vertex_cache(&mut destination, &indices, vertex_count);
            assert_eq!(destination, expected, "vcache v={vertex_count}");
        }
    }

    #[test]
    fn golden_optimize_vertex_cache_strip() {
        const _LEN_CHECK: usize = 9944;
        let data = include_bytes!("../tests/data/vcache_strip.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let index_count = c.u32() as usize;
            let vertex_count = c.u32() as usize;
            let indices = c.vec_u32(index_count);
            let expected = c.vec_u32(index_count);

            let mut destination = vec![0xeeu32; index_count];
            optimize_vertex_cache_strip(&mut destination, &indices, vertex_count);
            assert_eq!(destination, expected, "vcache_strip v={vertex_count}");
        }
    }

    #[test]
    fn golden_optimize_vertex_cache_fifo() {
        const _LEN_CHECK: usize = 12372;
        let data = include_bytes!("../tests/data/vcache_fifo.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let index_count = c.u32() as usize;
            let vertex_count = c.u32() as usize;
            let cache_size = c.u32();
            let indices = c.vec_u32(index_count);
            let expected = c.vec_u32(index_count);

            let mut destination = vec![0xeeu32; index_count];
            optimize_vertex_cache_fifo(&mut destination, &indices, vertex_count, cache_size);
            assert_eq!(
                destination, expected,
                "vcache_fifo v={vertex_count} cs={cache_size}"
            );
        }
    }
}
