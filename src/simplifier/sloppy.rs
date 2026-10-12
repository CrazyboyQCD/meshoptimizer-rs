//! Grid-based ("sloppy") simplification for indexed meshes and point clouds.
//! Upstream: `meshopt_simplifySloppy`/`meshopt_simplifyPoints`/
//! `meshopt_simplifyScale` (simplifier.cpp L2041-2329, L2646-2947).

use super::adjacency::{hash_buckets2, hash_lookup2, rescale_positions};
use super::quadric::{Quadric, quadric_add, quadric_error, quadric_from_triangle};
use super::simplify::SIMPLIFY_VERTEX_LOCK;
use super::{EMPTY_SLOT, Vector3};
use crate::math;
use crate::quantization::cast_i32;
use alloc::vec;

// MurmurHash2 finalizer (CellHasher)
fn murmur2(mut h: u32) -> u32 {
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1e995);
    h ^= h >> 15;
    h
}

pub(crate) fn compute_vertex_ids(
    vertex_ids: &mut [u32],
    vertex_positions: &[Vector3],
    vertex_lock: Option<&[u8]>,
    vertex_count: usize,
    grid_size: i32,
) {
    debug_assert!((1..=1024).contains(&grid_size));
    let cell_scale = grid_size as f32 - 1.0;

    for i in 0..vertex_count {
        let v = &vertex_positions[i];

        let xi = cast_i32(v.x * cell_scale + 0.5);
        let yi = cast_i32(v.y * cell_scale + 0.5);
        let zi = cast_i32(v.z * cell_scale + 0.5);

        if vertex_lock.is_some_and(|l| l[i] & SIMPLIFY_VERTEX_LOCK != 0) {
            vertex_ids[i] = (1 << 30) | i as u32;
        } else {
            vertex_ids[i] = ((xi << 20) | (yi << 10) | zi) as u32;
        }
    }
}

pub(crate) fn count_triangles(vertex_ids: &[u32], indices: &[u32], index_count: usize) -> usize {
    let mut result = 0usize;

    for i in (0..index_count).step_by(3) {
        let id0 = vertex_ids[indices[i] as usize];
        let id1 = vertex_ids[indices[i + 1] as usize];
        let id2 = vertex_ids[indices[i + 2] as usize];

        result += (id0 != id1 && id0 != id2 && id1 != id2) as usize;
    }

    result
}

pub(crate) fn fill_vertex_cells(
    table: &mut [u32],
    vertex_cells: &mut [u32],
    vertex_ids: &[u32],
    vertex_count: usize,
) -> usize {
    for t in table.iter_mut() {
        *t = EMPTY_SLOT;
    }

    let mut result = 0usize;

    for i in 0..vertex_count {
        let entry = hash_lookup2(
            table,
            murmur2(vertex_ids[i]),
            |lhs, rhs| vertex_ids[lhs as usize] == vertex_ids[rhs as usize],
            i as u32,
        );

        if table[entry] == EMPTY_SLOT {
            table[entry] = i as u32;
            vertex_cells[i] = result as u32;
            result += 1;
        } else {
            vertex_cells[i] = vertex_cells[table[entry] as usize];
        }
    }

    result
}

pub(crate) fn fill_cell_quadrics(
    cell_quadrics: &mut [Quadric],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[Vector3],
    vertex_cells: &[u32],
) {
    for i in (0..index_count).step_by(3) {
        let i0 = indices[i] as usize;
        let i1 = indices[i + 1] as usize;
        let i2 = indices[i + 2] as usize;

        let c0 = vertex_cells[i0];
        let c1 = vertex_cells[i1];
        let c2 = vertex_cells[i2];

        let single_cell = c0 == c1 && c0 == c2;

        let mut q = Quadric::ZERO;
        quadric_from_triangle(
            &mut q,
            vertex_positions[i0],
            vertex_positions[i1],
            vertex_positions[i2],
            if single_cell { 3.0 } else { 1.0 },
        );

        if single_cell {
            quadric_add(&mut cell_quadrics[c0 as usize], &q);
        } else {
            quadric_add(&mut cell_quadrics[c0 as usize], &q);
            quadric_add(&mut cell_quadrics[c1 as usize], &q);
            quadric_add(&mut cell_quadrics[c2 as usize], &q);
        }
    }
}

pub(crate) fn fill_cell_remap(
    cell_remap: &mut [u32],
    cell_errors: &mut [f32],
    cell_count: usize,
    vertex_cells: &[u32],
    cell_quadrics: &[Quadric],
    vertex_positions: &[Vector3],
    vertex_count: usize,
) {
    for r in cell_remap[..cell_count].iter_mut() {
        *r = EMPTY_SLOT;
    }

    for i in 0..vertex_count {
        let cell = vertex_cells[i] as usize;
        let error = quadric_error(&cell_quadrics[cell], &vertex_positions[i]);

        if cell_remap[cell] == EMPTY_SLOT || cell_errors[cell] > error {
            cell_remap[cell] = i as u32;
            cell_errors[cell] = error;
        }
    }
}

pub(crate) fn filter_triangles(
    destination: &mut [u32],
    tritable: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_cells: &[u32],
    cell_remap: &[u32],
) -> usize {
    for t in tritable.iter_mut() {
        *t = EMPTY_SLOT;
    }

    let mut result = 0usize;

    for i in (0..index_count).step_by(3) {
        let c0 = vertex_cells[indices[i] as usize];
        let c1 = vertex_cells[indices[i + 1] as usize];
        let c2 = vertex_cells[indices[i + 2] as usize];

        if c0 != c1 && c0 != c2 && c1 != c2 {
            let mut a = cell_remap[c0 as usize];
            let mut b = cell_remap[c1 as usize];
            let mut c = cell_remap[c2 as usize];

            if b < a && b < c {
                let t = a;
                a = b;
                b = c;
                c = t;
            } else if c < a && c < b {
                let t = c;
                c = b;
                b = a;
                a = t;
            }

            destination[result * 3] = a;
            destination[result * 3 + 1] = b;
            destination[result * 3 + 2] = c;

            // inline TriangleHasher probe over the just-written triple; the loop
            // terminates because hash_buckets2 keeps the table load factor <= 5/4
            // and quadratic probing over a power-of-two table visits every slot
            let hash =
                a.wrapping_mul(73856093) ^ b.wrapping_mul(19349663) ^ c.wrapping_mul(83492791);
            let hashmod = tritable.len() - 1;
            let mut bucket = hash as usize & hashmod;
            let mut probe = 0usize;

            let entry = loop {
                let item = tritable[bucket];

                if item == EMPTY_SLOT {
                    break bucket;
                }

                let t = item as usize * 3;
                if destination[t] == a && destination[t + 1] == b && destination[t + 2] == c {
                    break bucket;
                }

                // hash collision, quadratic probing
                bucket = (bucket + probe + 1) & hashmod;
                probe += 1;
            };

            if tritable[entry] == EMPTY_SLOT {
                tritable[entry] = result as u32;
                result += 1;
            }
        }
    }

    result * 3
}

fn interpolate(y: f32, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    // three point interpolation from "revenge of interpolation search" paper
    let num = (y1 - y) * (x1 - x2) * (x1 - x0) * (y2 - y0);
    let den = (y2 - y) * (x1 - x2) * (y0 - y1) + (y0 - y) * (x1 - x0) * (y1 - y2);
    x1 + if den == 0.0 { 0.0 } else { num / den }
}

/// Upstream: `meshopt_simplifySloppy`. Grid-quantizing simplifier that
/// guarantees the target index count by welding vertices on a uniform grid;
/// `target_error` drives the minimum grid resolution.
// port-shape: mirrors the upstream parameter list (ADR-0005)
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn simplify_sloppy(
    destination: &mut [u32],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_lock: Option<&[u8]>,
    target_index_count: usize,
    target_error: f32,
    result_error: Option<&mut f32>,
) -> usize {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
    debug_assert!(target_index_count <= index_count);
    debug_assert!(target_error >= 0.0);

    // we expect to get ~2 triangles/vertex in the output
    let target_cell_count = target_index_count / 6;

    let mut rescaled = vec![
        Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0
        };
        vertex_count
    ];
    rescale_positions(
        Some(&mut rescaled),
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        None,
        None,
    );

    // find the optimal grid size using guided binary search
    let mut vertex_ids = vec![0u32; vertex_count];

    const K_INTERPOLATION_PASSES: i32 = 5;

    // invariant: # of triangles in min_grid <= target_count
    let inv_error = if target_error < 1e-3 {
        1e-3
    } else if target_error < 1.0 {
        target_error
    } else {
        1.0
    };
    let mut min_grid = cast_i32(1.0 / inv_error);
    let mut max_grid = 1025i32;
    let mut min_triangles = 0usize;
    let mut max_triangles = index_count / 3;

    // when we're error-limited, we compute the triangle count for the min. size; this accelerates convergence and provides the correct answer when we can't use a larger grid
    if min_grid > 1 || vertex_lock.is_some() {
        compute_vertex_ids(
            &mut vertex_ids,
            &rescaled,
            vertex_lock,
            vertex_count,
            min_grid,
        );
        min_triangles = count_triangles(&vertex_ids, indices, index_count);
    }

    // instead of starting in the middle, let's guess as to what the answer might be! triangle count usually grows as a square of grid size...
    let mut next_grid_size = cast_i32(math::sqrt(target_cell_count as f32) + 0.5);

    for pass in 0..10 + K_INTERPOLATION_PASSES {
        if min_triangles >= target_index_count / 3 || max_grid - min_grid <= 1 {
            break;
        }

        // we clamp the prediction of the grid size to make sure that the search converges
        let mut grid_size = next_grid_size;
        grid_size = if grid_size <= min_grid {
            min_grid + 1
        } else if grid_size >= max_grid {
            max_grid - 1
        } else {
            grid_size
        };

        compute_vertex_ids(
            &mut vertex_ids,
            &rescaled,
            vertex_lock,
            vertex_count,
            grid_size,
        );
        let triangles = count_triangles(&vertex_ids, indices, index_count);

        let tip = interpolate(
            (target_index_count / 3) as f32,
            min_grid as f32,
            min_triangles as f32,
            grid_size as f32,
            triangles as f32,
            max_grid as f32,
            max_triangles as f32,
        );

        if triangles <= target_index_count / 3 {
            min_grid = grid_size;
            min_triangles = triangles;
        } else {
            max_grid = grid_size;
            max_triangles = triangles;
        }

        // we start by using interpolation search - it usually converges faster
        // however, interpolation search has a worst case of O(N) so we switch to binary search after a few iterations which converges in O(logN)
        next_grid_size = if pass < K_INTERPOLATION_PASSES {
            cast_i32(tip + 0.5)
        } else {
            (min_grid + max_grid) / 2
        };
    }

    if min_triangles == 0 {
        if let Some(re) = result_error {
            *re = 1.0;
        }

        return 0;
    }

    // build vertex->cell association by mapping all vertices with the same quantized position to the same cell
    let table_size = hash_buckets2(vertex_count);
    let mut table = vec![EMPTY_SLOT; table_size];

    let mut vertex_cells = vec![0u32; vertex_count];

    compute_vertex_ids(
        &mut vertex_ids,
        &rescaled,
        vertex_lock,
        vertex_count,
        min_grid,
    );
    let cell_count = fill_vertex_cells(&mut table, &mut vertex_cells, &vertex_ids, vertex_count);

    // build a quadric for each target cell
    let mut cell_quadrics = vec![Quadric::ZERO; cell_count];

    fill_cell_quadrics(
        &mut cell_quadrics,
        indices,
        index_count,
        &rescaled,
        &vertex_cells,
    );

    // for each target cell, find the vertex with the minimal error
    let mut cell_remap = vec![EMPTY_SLOT; cell_count];
    let mut cell_errors = vec![0.0f32; cell_count];

    fill_cell_remap(
        &mut cell_remap,
        &mut cell_errors,
        cell_count,
        &vertex_cells,
        &cell_quadrics,
        &rescaled,
        vertex_count,
    );

    // compute error
    let mut result_error_value = 0.0f32;

    for i in 0..cell_count {
        result_error_value = if result_error_value < cell_errors[i] {
            cell_errors[i]
        } else {
            result_error_value
        };
    }

    // vertex collapses often result in duplicate triangles; we need a table to filter them out
    let tritable_size = hash_buckets2(min_triangles);
    let mut tritable = vec![EMPTY_SLOT; tritable_size];

    // note: this is the first and last write to destination, which allows aliasing destination with indices
    let write = filter_triangles(
        destination,
        &mut tritable,
        indices,
        index_count,
        &vertex_cells,
        &cell_remap,
    );

    if let Some(re) = result_error {
        *re = math::sqrt(result_error_value);
    }

    write
}

/// Upstream: `meshopt_simplifyScale`. Computes the extent (largest side of the
/// position bounding box); the value `simplify` uses to normalize
/// `target_error`.
pub fn simplify_scale(
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> f32 {
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));

    rescale_positions(
        None,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        None,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
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
            f32::from_le_bytes(self.u32().to_le_bytes())
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

        fn bytes_vec(&mut self, n: usize) -> Vec<u8> {
            self.bytes(n).to_vec()
        }
    }

    #[test]
    fn golden_simplify_sloppy() {
        const _LEN_CHECK: usize = 36104;
        let data = include_bytes!("../../tests/data/sp_sloppy.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let target_index_count = c.u32() as usize;
            let target_error = c.f32();
            let has_lock = c.u32();
            let positions = c.f32s(vertex_count * stride);
            let indices = c.vec_u32(index_count);
            let lock = if has_lock == 1 {
                Some(c.bytes_vec(vertex_count))
            } else {
                None
            };

            let expected_count = c.u32() as usize;
            let expected_indices = c.vec_u32(expected_count);
            let expected_error = c.f32();

            let mut destination = vec![0xeeu32; index_count];
            let mut result_error = 0.0f32;
            let result = simplify_sloppy(
                &mut destination,
                &indices,
                &positions,
                vertex_count,
                stride,
                lock.as_deref(),
                target_index_count,
                target_error,
                Some(&mut result_error),
            );

            assert_eq!(result, expected_count, "simplifySloppy case {case} count");
            assert_eq!(
                &destination[..result],
                &expected_indices[..],
                "simplifySloppy case {case} indices"
            );
            assert_eq!(
                result_error, expected_error,
                "simplifySloppy case {case} error"
            );
            case += 1;
        }
        assert_eq!(case, 6);
    }

    #[test]
    fn golden_simplify_scale() {
        const _LEN_CHECK: usize = 2776;
        let data = include_bytes!("../../tests/data/sp_scale.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let positions = c.f32s(vertex_count * stride);

            let expected_extent = c.f32();

            let extent = simplify_scale(&positions, vertex_count, stride);
            assert_eq!(extent, expected_extent, "simplifyScale case {case}");
            case += 1;
        }
        assert_eq!(case, 4);
    }

    fn v3(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x, y, z }
    }

    #[test]
    #[allow(clippy::identity_op)] // edge-code literals keep upstream's `| 0` shape
    fn vertex_ids_grid_quantization() {
        let positions = vec![v3(0.0, 0.0, 0.0), v3(1.0, 0.5, 0.25), v3(1.0, 1.0, 1.0)];
        let mut ids = vec![0u32; 3];

        compute_vertex_ids(&mut ids, &positions, None, 3, 5);
        // cell_scale = 4: xi/yi/zi = round(p * 4)
        assert_eq!(ids[0], (0 << 20) | (0 << 10) | 0);
        assert_eq!(ids[1], (4 << 20) | (2 << 10) | 1);
        assert_eq!(ids[2], (4 << 20) | (4 << 10) | 4);

        // locked vertices keep their index in the high bits
        let lock = vec![SIMPLIFY_VERTEX_LOCK, 0, 0];
        compute_vertex_ids(&mut ids, &positions, Some(&lock), 3, 5);
        assert_eq!(ids[0], (1 << 30) | 0);
        assert_eq!(ids[1], (4 << 20) | (2 << 10) | 1);
    }

    #[test]
    fn count_triangles_drops_cell_degenerate() {
        let ids = vec![7u32, 7, 9, 12];
        let indices = vec![0, 1, 2, 0, 2, 3];
        // first triangle has id0 == id1 => degenerate in cell space
        assert_eq!(count_triangles(&ids, &indices, indices.len()), 1);
    }

    #[test]
    fn vertex_cells_assign_sequential_cells() {
        let ids = vec![5u32, 5, 9, 9, 9, 1];
        let mut table = vec![EMPTY_SLOT; hash_buckets2(6)];
        let mut cells = vec![0u32; 6];

        let count = fill_vertex_cells(&mut table, &mut cells, &ids, 6);
        assert_eq!(count, 3);
        assert_eq!(cells, vec![0, 0, 1, 1, 1, 2]);
    }

    #[test]
    fn filter_triangles_dedups_and_rotates() {
        // three distinct cells; cell representatives 10, 11, 12
        let vertex_cells = vec![0u32, 1, 2];
        let cell_remap = vec![10u32, 11, 12];
        let indices = vec![0, 1, 2, 0, 1, 2];
        let mut destination = vec![0u32; 6];
        let mut tritable = vec![EMPTY_SLOT; hash_buckets2(2)];

        let written = filter_triangles(
            &mut destination,
            &mut tritable,
            &indices,
            indices.len(),
            &vertex_cells,
            &cell_remap,
        );
        assert_eq!(written, 3);
        assert_eq!(&destination[..3], &[10, 11, 12]);
    }
}
