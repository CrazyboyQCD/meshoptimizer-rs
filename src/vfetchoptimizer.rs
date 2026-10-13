//! Vertex fetch reordering.
//!
//! Upstream: `vfetchoptimizer.cpp` (`meshopt_optimizeVertexFetchRemap`,
//! `meshopt_optimizeVertexFetch`).

use alloc::vec;

/// Upstream: `meshopt_optimizeVertexFetchRemap`. `destination` covers
/// `vertex_count` entries; returns the number of referenced vertices.
pub fn optimize_vertex_fetch_remap(destination: &mut [u32], indices: &[u32]) -> usize {
    let vertex_count = destination.len();
    destination.fill(!0);

    let mut next_vertex = 0u32;

    for &index in indices.iter() {
        debug_assert!(index < vertex_count as u32);

        if destination[index as usize] == !0 {
            destination[index as usize] = next_vertex;
            next_vertex += 1;
        }
    }

    next_vertex as usize
}

/// Upstream: `meshopt_optimizeVertexFetch`. Reorders vertices into `destination`
/// and rewrites `indices` in place; returns the number of unique vertices written.
/// Rust slices cannot alias, so the in-place form supported upstream is expressed
/// by the caller cloning the vertices first (identical observable behavior).
pub fn optimize_vertex_fetch(
    destination: &mut [u8],
    indices: &mut [u32],
    vertices: &[u8],
    vertex_count: usize,
    vertex_size: usize,
) -> usize {
    debug_assert!(vertex_size > 0 && vertex_size <= 256);
    debug_assert!(destination.len() >= vertex_count * vertex_size);
    debug_assert!(vertices.len() >= vertex_count * vertex_size);

    let mut vertex_remap = vec![!0u32; vertex_count];

    let mut next_vertex = 0usize;

    for index_slot in indices.iter_mut() {
        let index = *index_slot as usize;
        debug_assert!(index < vertex_count);

        if vertex_remap[index] == !0 {
            // vertex was not added to destination VB; add it
            let src = index * vertex_size..index * vertex_size + vertex_size;
            let dst = next_vertex * vertex_size..next_vertex * vertex_size + vertex_size;
            destination[dst].copy_from_slice(&vertices[src]);

            vertex_remap[index] = next_vertex as u32;
            next_vertex += 1;
        }

        // modify indices in place
        *index_slot = vertex_remap[index];
    }

    next_vertex
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_optimizeVertexFetch,
    // meshopt_optimizeVertexFetchRemap.
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

        fn bytes(&mut self, n: usize) -> &'a [u8] {
            let s = &self.data[self.pos..self.pos + n];
            self.pos += n;
            s
        }

        fn vec_u32(&mut self, n: usize) -> Vec<u32> {
            (0..n).map(|_| self.u32()).collect()
        }
    }

    #[test]
    fn golden_optimize_vertex_fetch_remap() {
        const _LEN_CHECK: usize = 6108;
        let data = include_bytes!("../tests/data/vfetch_remap.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let index_count = c.u32() as usize;
            let vertex_count = c.u32() as usize;
            let indices = c.vec_u32(index_count);
            let expected_remap = c.vec_u32(vertex_count);
            let expected_result = c.u32() as usize;

            let mut destination = vec![0xeeu32; vertex_count];
            let result = optimize_vertex_fetch_remap(&mut destination, &indices);
            assert_eq!(destination, expected_remap, "remap v={vertex_count}");
            assert_eq!(result, expected_result, "result v={vertex_count}");
        }
    }

    #[test]
    fn golden_optimize_vertex_fetch() {
        const _LEN_CHECK: usize = 18792;
        let data = include_bytes!("../tests/data/vfetch_fetch.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let index_count = c.u32() as usize;
            let vertex_count = c.u32() as usize;
            let vertex_size = c.u32() as usize;
            let vertices = c.bytes(vertex_count * vertex_size);
            let input_indices = c.vec_u32(index_count);
            let expected_indices = c.vec_u32(index_count);
            let expected_vertices = c.bytes(vertex_count * vertex_size);
            let expected_result = c.u32() as usize;

            let mut destination = vec![0xccu8; vertex_count * vertex_size];
            let mut indices = input_indices.clone();
            let result = optimize_vertex_fetch(
                &mut destination,
                &mut indices,
                vertices,
                vertex_count,
                vertex_size,
            );
            assert_eq!(indices, expected_indices, "indices v={vertex_count}");
            assert_eq!(
                destination.as_slice(),
                expected_vertices,
                "vb v={vertex_count}"
            );
            assert_eq!(result, expected_result, "result v={vertex_count}");
        }
    }
}
