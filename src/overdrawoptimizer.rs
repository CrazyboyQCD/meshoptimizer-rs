//! Overdraw-driven index reordering.
//!
//! Upstream: `overdrawoptimizer.cpp` (`meshopt_optimizeOverdraw`).

use crate::math;
use alloc::vec;

fn quantize_unorm(v: f32, n: i32) -> i32 {
    let scale = ((1 << n) - 1) as f32;

    let v = if v >= 0.0 { v } else { 0.0 };
    let v = if v <= 1.0 { v } else { 1.0 };

    (v * scale + 0.5) as i32
}

#[allow(clippy::needless_range_loop)]
fn calculate_sort_data(
    sort_data: &mut [f32],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    clusters: &[u32],
    cluster_count: usize,
) {
    let vertex_stride_float = vertex_positions_stride / 4;

    let mut mesh_centroid = [0f32; 3];

    for i in 0..vertex_count {
        let p = &vertex_positions[vertex_stride_float * i..][..3];

        mesh_centroid[0] += p[0];
        mesh_centroid[1] += p[1];
        mesh_centroid[2] += p[2];
    }

    mesh_centroid[0] /= vertex_count as f32;
    mesh_centroid[1] /= vertex_count as f32;
    mesh_centroid[2] /= vertex_count as f32;

    for cluster in 0..cluster_count {
        let cluster_begin = clusters[cluster] as usize * 3;
        let cluster_end = if cluster + 1 < cluster_count {
            clusters[cluster + 1] as usize * 3
        } else {
            indices.len()
        };
        debug_assert!(cluster_begin < cluster_end);

        let mut cluster_area = 0f32;
        let mut cluster_centroid = [0f32; 3];
        let mut cluster_normal = [0f32; 3];

        let mut i = cluster_begin;
        while i < cluster_end {
            let p0 = &vertex_positions[vertex_stride_float * indices[i] as usize..][..3];
            let p1 = &vertex_positions[vertex_stride_float * indices[i + 1] as usize..][..3];
            let p2 = &vertex_positions[vertex_stride_float * indices[i + 2] as usize..][..3];

            let p10 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let p20 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

            let normalx = p10[1] * p20[2] - p10[2] * p20[1];
            let normaly = p10[2] * p20[0] - p10[0] * p20[2];
            let normalz = p10[0] * p20[1] - p10[1] * p20[0];

            let area = math::sqrt(normalx * normalx + normaly * normaly + normalz * normalz);

            let area_third = area / 3.0;
            cluster_centroid[0] += (p0[0] + p1[0] + p2[0]) * area_third;
            cluster_centroid[1] += (p0[1] + p1[1] + p2[1]) * area_third;
            cluster_centroid[2] += (p0[2] + p1[2] + p2[2]) * area_third;
            cluster_normal[0] += normalx;
            cluster_normal[1] += normaly;
            cluster_normal[2] += normalz;
            cluster_area += area;

            i += 3;
        }

        let inv_cluster_area = if cluster_area == 0.0 {
            0.0
        } else {
            1.0 / cluster_area
        };

        cluster_centroid[0] *= inv_cluster_area;
        cluster_centroid[1] *= inv_cluster_area;
        cluster_centroid[2] *= inv_cluster_area;

        let cluster_normal_length = math::sqrt(
            cluster_normal[0] * cluster_normal[0]
                + cluster_normal[1] * cluster_normal[1]
                + cluster_normal[2] * cluster_normal[2],
        );
        let inv_cluster_normal_length = if cluster_normal_length == 0.0 {
            0.0
        } else {
            1.0 / cluster_normal_length
        };

        cluster_normal[0] *= inv_cluster_normal_length;
        cluster_normal[1] *= inv_cluster_normal_length;
        cluster_normal[2] *= inv_cluster_normal_length;

        let centroid_vector = [
            cluster_centroid[0] - mesh_centroid[0],
            cluster_centroid[1] - mesh_centroid[1],
            cluster_centroid[2] - mesh_centroid[2],
        ];

        sort_data[cluster] = centroid_vector[0] * cluster_normal[0]
            + centroid_vector[1] * cluster_normal[1]
            + centroid_vector[2] * cluster_normal[2];
    }
}

fn calculate_sort_order_radix(sort_order: &mut [u32], sort_data: &[f32], sort_keys: &mut [u16]) {
    // compute sort data bounds and renormalize, using fixed point snorm
    let mut sort_data_max = 1e-3f32;

    for &d in sort_data.iter() {
        let dpa = d.abs();

        if sort_data_max < dpa {
            sort_data_max = dpa;
        }
    }

    const SORT_BITS: i32 = 11;

    for i in 0..sort_data.len() {
        // note that we flip distribution since high dot product should come first
        let sort_key = 0.5 - 0.5 * (sort_data[i] / sort_data_max);

        sort_keys[i] = (quantize_unorm(sort_key, SORT_BITS) & ((1 << SORT_BITS) - 1)) as u16;
    }

    // fill histogram for counting sort
    let mut histogram = [0u32; 1 << SORT_BITS];

    for &key in sort_keys[..sort_data.len()].iter() {
        histogram[key as usize] += 1;
    }

    // compute offsets based on histogram data
    let mut histogram_sum = 0usize;

    for h in histogram.iter_mut() {
        let count = *h;
        *h = histogram_sum as u32;
        histogram_sum += count as usize;
    }

    debug_assert!(histogram_sum == sort_data.len());

    // compute sort order based on offsets
    for (i, &key) in sort_keys[..sort_data.len()].iter().enumerate() {
        let slot = histogram[key as usize] as usize;
        sort_order[slot] = i as u32;
        histogram[key as usize] += 1;
    }
}

fn update_cache(
    a: u32,
    b: u32,
    c: u32,
    cache_size: u32,
    cache_timestamps: &mut [u32],
    timestamp: &mut u32,
) -> u32 {
    let mut cache_misses = 0;

    // if vertex is not in cache, put it in cache
    if *timestamp - cache_timestamps[a as usize] > cache_size {
        cache_timestamps[a as usize] = *timestamp;
        *timestamp += 1;
        cache_misses += 1;
    }

    if *timestamp - cache_timestamps[b as usize] > cache_size {
        cache_timestamps[b as usize] = *timestamp;
        *timestamp += 1;
        cache_misses += 1;
    }

    if *timestamp - cache_timestamps[c as usize] > cache_size {
        cache_timestamps[c as usize] = *timestamp;
        *timestamp += 1;
        cache_misses += 1;
    }

    cache_misses
}

fn generate_hard_boundaries(
    destination: &mut [u32],
    indices: &[u32],
    vertex_count: usize,
    cache_size: u32,
    cache_timestamps: &mut [u32],
) -> usize {
    cache_timestamps[..vertex_count].fill(0);

    let mut timestamp = cache_size + 1;

    let face_count = indices.len() / 3;

    let mut result = 0usize;

    for i in 0..face_count {
        let m = update_cache(
            indices[i * 3],
            indices[i * 3 + 1],
            indices[i * 3 + 2],
            cache_size,
            cache_timestamps,
            &mut timestamp,
        );

        // when all three vertices are not in the cache it's usually relatively safe to assume
        // that this is a new patch in the mesh that is disjoint from previous vertices; usually
        // the first triangle has 3 misses unless it's degenerate - thus we make sure the first
        // cluster always starts with 0
        if i == 0 || m == 3 {
            destination[result] = i as u32;
            result += 1;
        }
    }

    debug_assert!(result <= indices.len() / 3);

    result
}

fn generate_soft_boundaries(
    destination: &mut [u32],
    indices: &[u32],
    vertex_count: usize,
    clusters: &[u32],
    cache_size: u32,
    threshold: f32,
    cache_timestamps: &mut [u32],
) -> usize {
    let cluster_count = clusters.len();
    cache_timestamps[..vertex_count].fill(0);

    let mut timestamp = 0u32;

    let mut result = 0usize;

    for it in 0..cluster_count {
        let start = clusters[it] as usize;
        let end = if it + 1 < cluster_count {
            clusters[it + 1] as usize
        } else {
            indices.len() / 3
        };
        debug_assert!(start < end);

        // reset cache
        timestamp += cache_size + 1;

        // measure cluster ACMR
        let mut cluster_misses = 0u32;

        for i in start..end {
            let m = update_cache(
                indices[i * 3],
                indices[i * 3 + 1],
                indices[i * 3 + 2],
                cache_size,
                cache_timestamps,
                &mut timestamp,
            );

            cluster_misses += m;
        }

        let cluster_threshold = threshold * (cluster_misses as f32 / (end - start) as f32);

        // first cluster always starts from the hard cluster boundary
        destination[result] = start as u32;
        result += 1;

        // reset cache
        timestamp += cache_size + 1;

        let mut running_misses = 0u32;
        let mut running_faces = 0u32;

        for i in start..end {
            let m = update_cache(
                indices[i * 3],
                indices[i * 3 + 1],
                indices[i * 3 + 2],
                cache_size,
                cache_timestamps,
                &mut timestamp,
            );

            running_misses += m;
            running_faces += 1;

            if running_misses as f32 / running_faces as f32 <= cluster_threshold {
                // we have reached the target ACMR with the current triangle so we need to start
                // a new cluster on the next one; note that this may mean that we add `end` to
                // destination for the last triangle, which will imply that the last cluster is
                // empty; however, the trim after the loop will clean it up
                destination[result] = (i + 1) as u32;
                result += 1;

                // reset cache
                timestamp += cache_size + 1;

                running_misses = 0;
                running_faces = 0;
            }
        }

        // each time we reach the target ACMR we flush the cluster; the last cluster boundary is
        // removed, merging the last complete cluster with the last incomplete one
        if destination[result - 1] != start as u32 {
            result -= 1;
        }
    }

    debug_assert!(result >= cluster_count);
    debug_assert!(result <= indices.len() / 3);

    result
}

fn stride_assert(vertex_positions_stride: usize) {
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
}

/// Upstream: `meshopt_optimizeOverdraw`. Reorders triangles to reduce overdraw
/// using the internal rasterizer (`threshold` trades efficiency for runtime).
pub fn optimize_overdraw(
    destination: &mut [u32],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    threshold: f32,
) {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));
    stride_assert(vertex_positions_stride);

    // guard for empty meshes
    if index_count == 0 || vertex_count == 0 {
        return;
    }

    let cache_size = 16u32;

    let mut cache_timestamps = vec![0u32; vertex_count];

    // generate hard boundaries from full-triangle cache misses
    let mut hard_clusters = vec![0u32; index_count / 3];
    let hard_cluster_count = generate_hard_boundaries(
        &mut hard_clusters,
        indices,
        vertex_count,
        cache_size,
        &mut cache_timestamps,
    );

    // generate soft boundaries
    let mut soft_clusters = vec![0u32; index_count / 3 + 1];
    let soft_cluster_count = generate_soft_boundaries(
        &mut soft_clusters,
        indices,
        vertex_count,
        &hard_clusters[..hard_cluster_count],
        cache_size,
        threshold,
        &mut cache_timestamps,
    );

    let clusters = &soft_clusters[..soft_cluster_count];
    let cluster_count = soft_cluster_count;

    // fill sort data
    let mut sort_data = vec![0f32; cluster_count];
    calculate_sort_data(
        &mut sort_data,
        indices,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        clusters,
        cluster_count,
    );

    // sort clusters using sort data
    let mut sort_keys = vec![0u16; cluster_count];
    let mut sort_order = vec![0u32; cluster_count];
    calculate_sort_order_radix(&mut sort_order, &sort_data, &mut sort_keys);

    // fill output buffer
    let mut offset = 0usize;

    for &cluster in sort_order.iter() {
        let cluster = cluster as usize;
        debug_assert!(cluster < cluster_count);

        let cluster_begin = clusters[cluster] as usize * 3;
        let cluster_end = if cluster + 1 < cluster_count {
            clusters[cluster + 1] as usize * 3
        } else {
            index_count
        };
        debug_assert!(cluster_begin < cluster_end);

        destination[offset..offset + (cluster_end - cluster_begin)]
            .copy_from_slice(&indices[cluster_begin..cluster_end]);
        offset += cluster_end - cluster_begin;
    }

    debug_assert!(offset == index_count);
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_optimizeOverdraw.
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

        fn f32(&mut self) -> f32 {
            f32::from_bits(self.u32())
        }

        fn bytes(&mut self, n: usize) -> &'a [u8] {
            let s = &self.data[self.pos..self.pos + n];
            self.pos += n;
            s
        }

        fn vec_u32(&mut self, n: usize) -> Vec<u32> {
            (0..n).map(|_| self.u32()).collect()
        }

        fn f32s(&mut self, byte_len: usize) -> Vec<f32> {
            let raw = self.bytes(byte_len);
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect()
        }
    }

    #[test]
    fn golden_optimize_overdraw() {
        const _LEN_CHECK: usize = 8848;
        let data = include_bytes!("../tests/data/overdraw.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let positions = c.f32s(vertex_count * stride);
            let indices = c.vec_u32(index_count);
            let threshold = c.f32();
            let expected = c.vec_u32(index_count);

            let mut destination = vec![0xeeu32; index_count];
            optimize_overdraw(
                &mut destination,
                &indices,
                &positions,
                vertex_count,
                stride,
                threshold,
            );
            assert_eq!(
                destination, expected,
                "overdraw v={vertex_count} t={threshold}"
            );
        }
    }
}
