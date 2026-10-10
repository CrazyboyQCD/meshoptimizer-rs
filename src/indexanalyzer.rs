//! Index buffer analysis.
//!
//! Upstream: `indexanalyzer.cpp` (`meshopt_analyzeVertexCache`,
//! `meshopt_analyzeVertexFetch`).

use alloc::vec;

/// Upstream: `meshopt_VertexCacheStatistics`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VertexCacheStatistics {
    pub vertices_transformed: u32,
    pub warps_executed: u32,
    /// transformed vertices / triangle count; best case 0.5, worst case 3.0
    pub acmr: f32,
    /// transformed vertices / vertex count; best case 1.0, worst case 6.0
    pub atvr: f32,
}

/// Upstream: `meshopt_VertexFetchStatistics`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VertexFetchStatistics {
    pub bytes_fetched: u32,
    /// fetched bytes / vertex buffer size; best case 1.0
    pub overfetch: f32,
}

/// Upstream: `meshopt_analyzeVertexCache`.
pub fn analyze_vertex_cache(
    indices: &[u32],
    vertex_count: usize,
    cache_size: u32,
    warp_size: u32,
    primgroup_size: u32,
) -> VertexCacheStatistics {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));
    debug_assert!(cache_size >= 3);
    debug_assert!(warp_size == 0 || warp_size >= 3);

    let mut result = VertexCacheStatistics {
        vertices_transformed: 0,
        warps_executed: 0,
        acmr: 0.0,
        atvr: 0.0,
    };

    let mut warp_offset = 0u32;
    let mut primgroup_offset = 0u32;

    let mut cache_timestamps = vec![0u32; vertex_count];

    let mut timestamp = cache_size + 1;

    let mut i = 0;
    while i < index_count {
        let a = indices[i];
        let b = indices[i + 1];
        let c = indices[i + 2];
        debug_assert!(
            a < vertex_count as u32 && b < vertex_count as u32 && c < vertex_count as u32
        );

        let ac = timestamp - cache_timestamps[a as usize] > cache_size;
        let bc = timestamp - cache_timestamps[b as usize] > cache_size;
        let cc = timestamp - cache_timestamps[c as usize] > cache_size;

        // flush cache if triangle doesn't fit into warp or into the primitive buffer
        if (primgroup_size != 0 && primgroup_offset == primgroup_size)
            || (warp_size != 0
                && warp_offset + u32::from(ac) + u32::from(bc) + u32::from(cc) > warp_size)
        {
            result.warps_executed += u32::from(warp_offset > 0);

            warp_offset = 0;
            primgroup_offset = 0;

            // reset cache
            timestamp += cache_size + 1;
        }

        // update cache and add vertices to warp
        for j in 0..3 {
            let index = indices[i + j];

            if timestamp - cache_timestamps[index as usize] > cache_size {
                cache_timestamps[index as usize] = timestamp;
                timestamp += 1;
                result.vertices_transformed += 1;
                warp_offset += 1;
            }
        }

        primgroup_offset += 1;
        i += 3;
    }

    let mut unique_vertex_count = 0usize;

    for &t in cache_timestamps.iter() {
        unique_vertex_count += usize::from(t > 0);
    }

    result.warps_executed += u32::from(warp_offset > 0);

    result.acmr = if index_count == 0 {
        0.0
    } else {
        result.vertices_transformed as f32 / (index_count / 3) as f32
    };
    result.atvr = if unique_vertex_count == 0 {
        0.0
    } else {
        result.vertices_transformed as f32 / unique_vertex_count as f32
    };

    result
}

/// Upstream: `meshopt_analyzeVertexFetch`.
pub fn analyze_vertex_fetch(
    indices: &[u32],
    vertex_count: usize,
    vertex_size: usize,
) -> VertexFetchStatistics {
    debug_assert!(vertex_size > 0 && vertex_size <= 256);

    let mut result = VertexFetchStatistics {
        bytes_fetched: 0,
        overfetch: 0.0,
    };

    let mut vertex_visited = vec![0u8; vertex_count];

    const K_CACHE_LINE: usize = 64;
    const K_CACHE_SIZE: usize = 128 * 1024;

    // simple direct mapped cache; on typical mesh data this is close to 4-way cache, and this
    // model is a gross approximation anyway
    let mut cache = [0usize; K_CACHE_SIZE / K_CACHE_LINE];

    for &index in indices.iter() {
        debug_assert!(index < vertex_count as u32);

        vertex_visited[index as usize] = 1;

        let start_address = index as usize * vertex_size;
        let end_address = start_address + vertex_size;

        let start_tag = start_address / K_CACHE_LINE;
        let end_tag = end_address.div_ceil(K_CACHE_LINE);

        debug_assert!(start_tag < end_tag);

        for tag in start_tag..end_tag {
            let line = tag % cache.len();

            // we store +1 since cache is filled with 0 by default
            result.bytes_fetched += u32::from(cache[line] != tag + 1) * K_CACHE_LINE as u32;
            cache[line] = tag + 1;
        }
    }

    let mut unique_vertex_count = 0usize;

    for &v in vertex_visited.iter() {
        unique_vertex_count += usize::from(v != 0);
    }

    result.overfetch = if unique_vertex_count == 0 {
        0.0
    } else {
        result.bytes_fetched as f32 / (unique_vertex_count * vertex_size) as f32
    };

    result
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

        fn f32(&mut self) -> f32 {
            f32::from_bits(self.u32())
        }

        fn vec_u32(&mut self, n: usize) -> Vec<u32> {
            (0..n).map(|_| self.u32()).collect()
        }
    }

    #[test]
    fn golden_analyze_vertex_cache() {
        const _LEN_CHECK: usize = 8652;
        let data = include_bytes!("../tests/data/analyze_vcache.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let index_count = c.u32() as usize;
            let vertex_count = c.u32() as usize;
            let cache_size = c.u32();
            let warp_size = c.u32();
            let primgroup_size = c.u32();
            let indices = c.vec_u32(index_count);
            let expected = VertexCacheStatistics {
                vertices_transformed: c.u32(),
                warps_executed: c.u32(),
                acmr: c.f32(),
                atvr: c.f32(),
            };

            let result = analyze_vertex_cache(
                &indices,
                vertex_count,
                cache_size,
                warp_size,
                primgroup_size,
            );
            assert_eq!(
                result, expected,
                "vcache cs={cache_size} ws={warp_size} ps={primgroup_size}"
            );
        }
    }

    #[test]
    fn golden_analyze_vertex_fetch() {
        const _LEN_CHECK: usize = 5004;
        let data = include_bytes!("../tests/data/analyze_vfetch.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let index_count = c.u32() as usize;
            let vertex_count = c.u32() as usize;
            let vertex_size = c.u32() as usize;
            let indices = c.vec_u32(index_count);
            let expected = VertexFetchStatistics {
                bytes_fetched: c.u32(),
                overfetch: c.f32(),
            };

            let result = analyze_vertex_fetch(&indices, vertex_count, vertex_size);
            assert_eq!(result, expected, "vfetch vs={vertex_size}");
        }
    }
}
