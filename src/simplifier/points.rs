//! Point-cloud simplification via grid clustering with color-weighted
//! reservoirs. Upstream: `meshopt_simplifyPoints` plus the reservoir helpers
//! (simplifier.cpp L2195-2231, L2250-2276, L2813-2935; split out of `sloppy`
//! for file-size discipline, design §4).

use super::Vector3;
use super::adjacency::{hash_buckets2, rescale_positions};
use super::sloppy::{compute_vertex_ids, fill_vertex_cells};
use crate::math;
use crate::quantization::cast_i32;
use alloc::vec;
use alloc::vec::Vec;

// x, y, z = position accumulator; r, g, b = color accumulator; w = weight
#[derive(Clone, Copy)]
struct Reservoir {
    x: f32,
    y: f32,
    z: f32,
    r: f32,
    g: f32,
    b: f32,
    w: f32,
}

impl Reservoir {
    const ZERO: Reservoir = Reservoir {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        r: 0.0,
        g: 0.0,
        b: 0.0,
        w: 0.0,
    };
}

fn fill_cell_reservoirs(
    cell_reservoirs: &mut [Reservoir],
    cell_count: usize,
    vertex_positions: &[Vector3],
    vertex_colors: Option<&[f32]>,
    vertex_colors_stride: usize,
    vertex_cells: &[u32],
) {
    let dummy_color = [0.0f32; 3];

    // uniform color window: the None path reads the dummy through stride 0, so the
    // loop body carries no colors Option discriminant test (upstream selects the
    // dummy_color pointer per iteration; this folds the select into the stride)
    let colors: &[f32] = match vertex_colors {
        Some(c) => c,
        None => &dummy_color,
    };
    let stride_float = if vertex_colors.is_some() {
        vertex_colors_stride / 4
    } else {
        0
    };

    for (i, (v, &cell)) in vertex_positions.iter().zip(vertex_cells.iter()).enumerate() {
        let color = &colors[i * stride_float..i * stride_float + 3];

        let r = &mut cell_reservoirs[cell as usize];

        r.x += v.x;
        r.y += v.y;
        r.z += v.z;
        r.r += color[0];
        r.g += color[1];
        r.b += color[2];
        r.w += 1.0;
    }

    for r in &mut cell_reservoirs[..cell_count] {
        let iw = if r.w == 0.0 { 0.0 } else { 1.0 / r.w };

        r.x *= iw;
        r.y *= iw;
        r.z *= iw;
        r.r *= iw;
        r.g *= iw;
        r.b *= iw;
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_cell_remap_reservoir(
    cell_remap: &mut [u32],
    cell_errors: &mut [f32],
    cell_count: usize,
    vertex_cells: &[u32],
    cell_reservoirs: &[Reservoir],
    vertex_positions: &[Vector3],
    vertex_colors: Option<&[f32]>,
    vertex_colors_stride: usize,
    color_weight: f32,
) {
    let dummy_color = [0.0f32; 3];

    // uniform color window: the None path reads the dummy through stride 0, so the
    // loop body carries no colors Option discriminant test (upstream selects the
    // dummy_color pointer per iteration; this folds the select into the stride)
    let colors: &[f32] = match vertex_colors {
        Some(c) => c,
        None => &dummy_color,
    };
    let stride_float = if vertex_colors.is_some() {
        vertex_colors_stride / 4
    } else {
        0
    };

    for r in cell_remap[..cell_count].iter_mut() {
        *r = super::EMPTY_SLOT;
    }

    for (i, (v, &cell)) in vertex_positions.iter().zip(vertex_cells.iter()).enumerate() {
        let cell = cell as usize;
        let r = &cell_reservoirs[cell];
        let color = &colors[i * stride_float..i * stride_float + 3];

        let pos_error =
            (v.x - r.x) * (v.x - r.x) + (v.y - r.y) * (v.y - r.y) + (v.z - r.z) * (v.z - r.z);
        let col_error = (color[0] - r.r) * (color[0] - r.r)
            + (color[1] - r.g) * (color[1] - r.g)
            + (color[2] - r.b) * (color[2] - r.b);
        let error = pos_error + color_weight * col_error;

        if cell_remap[cell] == super::EMPTY_SLOT || cell_errors[cell] > error {
            cell_remap[cell] = i as u32;
            cell_errors[cell] = error;
        }
    }
}

/// Upstream: `meshopt_simplifyPoints`. Clusters points on a uniform grid and
/// picks a representative per cell; `vertex_colors`/`color_weight` steer the
/// representative choice.
///
/// `vertex_positions` must be finite; NaN or infinite coordinates are outside
/// the API contract, where upstream C is undefined behavior as well
/// (ADR-0014).
// port-shape: mirrors the upstream parameter list (ADR-0005)
#[allow(clippy::too_many_arguments)]
pub fn simplify_points(
    destination: &mut [u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_colors: Option<&[f32]>,
    vertex_colors_stride: usize,
    color_weight: f32,
    target_vertex_count: usize,
) -> usize {
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
    debug_assert!(vertex_colors_stride == 0 || (12..=256).contains(&vertex_colors_stride));
    debug_assert!(vertex_colors_stride.is_multiple_of(4));
    debug_assert!(vertex_colors.is_none() || vertex_colors_stride != 0);
    debug_assert!(target_vertex_count <= vertex_count);

    let target_cell_count = target_vertex_count;

    if target_cell_count == 0 {
        return 0;
    }

    let mut rescaled = Vec::new();
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

    let table_size = hash_buckets2(vertex_count);
    let mut table = vec![super::EMPTY_SLOT; table_size];

    const K_INTERPOLATION_PASSES: i32 = 5;

    // invariant: # of vertices in min_grid <= target_count
    let mut min_grid = 0i32;
    let mut max_grid = 1025i32;
    let mut min_vertices = 0usize;
    let mut max_vertices = vertex_count;

    // instead of starting in the middle, let's guess as to what the answer might be! surface point count usually grows as a square of grid size...
    let mut next_grid_size = (math::sqrt(target_cell_count as f32) + 0.5) as i32;

    for pass in 0..10 + K_INTERPOLATION_PASSES {
        debug_assert!(min_vertices < target_vertex_count);
        debug_assert!(max_grid - min_grid > 1);

        // we clamp the prediction of the grid size to make sure that the search converges
        let mut grid_size = next_grid_size;
        grid_size = if grid_size <= min_grid {
            min_grid + 1
        } else if grid_size >= max_grid {
            max_grid - 1
        } else {
            grid_size
        };

        compute_vertex_ids(&mut vertex_ids, &rescaled, None, vertex_count, grid_size);
        let vertices = count_vertex_cells(&mut table, &vertex_ids, vertex_count);

        let tip = interpolate(
            target_vertex_count as f32,
            min_grid as f32,
            min_vertices as f32,
            grid_size as f32,
            vertices as f32,
            max_grid as f32,
            max_vertices as f32,
        );

        if vertices <= target_vertex_count {
            min_grid = grid_size;
            min_vertices = vertices;
        } else {
            max_grid = grid_size;
            max_vertices = vertices;
        }

        if vertices == target_vertex_count || max_grid - min_grid <= 1 {
            break;
        }

        // we start by using interpolation search - it usually converges faster
        // however, interpolation search has a worst case of O(N) so we switch to binary search after a few iterations which converges in O(logN)
        next_grid_size = if pass < K_INTERPOLATION_PASSES {
            cast_i32(tip + 0.5)
        } else {
            (min_grid + max_grid) / 2
        };
    }

    if min_vertices == 0 {
        return 0;
    }

    // build vertex->cell association by mapping all vertices with the same quantized position to the same cell
    let mut vertex_cells = vec![0u32; vertex_count];

    compute_vertex_ids(&mut vertex_ids, &rescaled, None, vertex_count, min_grid);
    let cell_count = fill_vertex_cells(&mut table, &mut vertex_cells, &vertex_ids, vertex_count);

    // accumulate points into a reservoir for each target cell
    let mut cell_reservoirs = vec![Reservoir::ZERO; cell_count];

    fill_cell_reservoirs(
        &mut cell_reservoirs,
        cell_count,
        &rescaled,
        vertex_colors,
        vertex_colors_stride,
        &vertex_cells,
    );

    // for each target cell, find the vertex with the minimal error
    let mut cell_remap = vec![super::EMPTY_SLOT; cell_count];
    let mut cell_errors = vec![0.0f32; cell_count];

    // we scale the color weight to bring it to the same scale as position so that error addition makes sense
    let color_weight_scaled = color_weight
        * if min_grid == 1 {
            1.0
        } else {
            1.0 / (min_grid as f32 - 1.0)
        };

    fill_cell_remap_reservoir(
        &mut cell_remap,
        &mut cell_errors,
        cell_count,
        &vertex_cells,
        &cell_reservoirs,
        &rescaled,
        vertex_colors,
        vertex_colors_stride,
        color_weight_scaled * color_weight_scaled,
    );

    // copy results to the output
    debug_assert!(cell_count <= target_vertex_count);
    destination[..cell_count].copy_from_slice(&cell_remap);

    cell_count
}

// MurmurHash2 finalizer (CellHasher with NULL vertex_ids: hash/equal on ids)
fn murmur2(mut h: u32) -> u32 {
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1e995);
    h ^= h >> 15;
    h
}

fn count_vertex_cells(table: &mut [u32], vertex_ids: &[u32], vertex_count: usize) -> usize {
    for t in table.iter_mut() {
        *t = super::EMPTY_SLOT;
    }

    let mut result = 0usize;
    let hashmod = table.len() - 1;

    // termination mirrors hash_lookup2: the load factor is capped at 5/4 and
    // quadratic probing over a power-of-two table visits every slot
    for &id in &vertex_ids[..vertex_count] {
        let mut bucket = murmur2(id) as usize & hashmod;
        let mut probe = 0usize;

        let empty = loop {
            let item = table[bucket];

            if item == super::EMPTY_SLOT || item == id {
                break item == super::EMPTY_SLOT;
            }

            // hash collision, quadratic probing
            probe += 1;
            bucket = (bucket + probe) & hashmod;
        };

        result += empty as usize;
        table[bucket] = id;
    }

    result
}

fn interpolate(y: f32, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    // three point interpolation from "revenge of interpolation search" paper
    let num = (y1 - y) * (x1 - x2) * (x1 - x0) * (y2 - y0);
    let den = (y2 - y) * (x1 - x2) * (y0 - y1) + (y0 - y) * (x1 - x0) * (y1 - y2);
    x1 + if den == 0.0 { 0.0 } else { num / den }
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_simplifyPoints.
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
    }

    #[test]
    fn golden_simplify_points() {
        const _LEN_CHECK: usize = 10432;
        let data = include_bytes!("../../tests/data/sp_points.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let pos_stride = c.u32() as usize;
            let has_colors = c.u32();
            let color_stride = c.u32() as usize;
            let color_weight = c.f32();
            let target_vertex_count = c.u32() as usize;
            let positions = c.f32s(vertex_count * pos_stride);
            let colors = if has_colors == 1 {
                Some(c.f32s(vertex_count * color_stride))
            } else {
                None
            };

            let expected_count = c.u32() as usize;
            let expected = c.vec_u32(expected_count);

            let mut destination = vec![0xeeu32; target_vertex_count.max(expected_count)];
            let result = simplify_points(
                &mut destination,
                &positions,
                vertex_count,
                pos_stride,
                colors.as_deref(),
                if colors.is_some() { color_stride } else { 0 },
                color_weight,
                target_vertex_count,
            );

            assert_eq!(result, expected_count, "simplifyPoints case {case} count");
            assert_eq!(
                &destination[..result],
                &expected[..],
                "simplifyPoints case {case} indices"
            );
            case += 1;
        }
        assert_eq!(case, 5);
    }

    #[test]
    fn interpolate_exact_line() {
        // y = 3x through (0,0), (1,3), (2,6): y = 3 at x = 1
        assert_eq!(interpolate(3.0, 0.0, 0.0, 1.0, 3.0, 2.0, 6.0), 1.0);
        // zero-denominator case falls back to x1
        assert_eq!(interpolate(4.0, 0.0, 1.0, 2.0, 1.0, 4.0, 1.0), 2.0);
    }

    #[test]
    fn point_cells_dedup_by_id() {
        let ids = vec![5u32, 5, 9, 9, 9, 1];
        let mut table = vec![crate::simplifier::EMPTY_SLOT; hash_buckets2(6)];

        let count = count_vertex_cells(&mut table, &ids, 6);
        assert_eq!(count, 3);
    }
}
