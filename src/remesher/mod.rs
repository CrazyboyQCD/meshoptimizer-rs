//! Remeshing (FR-18). Upstream: `meshopt_remesh` (remesher.cpp, entire TU).
//!
//! Marching-cubes style voxelization: triangles are sampled into a voxel grid,
//! interior voxels are solidified by flood fill, per-voxel positions (optionally
//! quadric-solved) are computed, and the isosurface is polygonized. Split along
//! algorithmic seams per design §4: marching-cubes table generation in
//! `tables`, isosurface extraction in `polygonize`.
//!
//! Upstream builds the marching-cubes rotation tables once via C++ static
//! initialization; the port rebuilds them per call (pure function of the base
//! case table, behavior-identical). `meshopt_RemeshInternalDebug` is not
//! exported by the upstream public header and is not ported, mirroring the
//! `meshopt_SimplifyInternalDebug` precedent.

mod polygonize;
mod tables;

use crate::math;
use crate::quantization::cast_i32;
use alloc::vec;

/// Upstream: `meshopt_RemeshShell`. Keep only the voxel shell, skipping the
/// interior solidification pass.
pub const REMESH_SHELL: u32 = 1 << 0;
/// Upstream: `meshopt_RemeshSolve`. Solve per-voxel quadric positions instead
/// of using voxel centroids.
pub const REMESH_SOLVE: u32 = 1 << 1;

const EMPTY_SLOT: u32 = u32::MAX;

/// Upstream: `meshopt_remesh`. Remeshes the triangle soup into a grid of
/// `resolution` voxels per side and returns the number of generated triangles.
///
/// With `destination` == `None` this only counts the triangles; otherwise
/// `destination.len() / 9` is the triangle capacity and generated triangles are
/// written as 9 floats (3 vertices, xyz each). `indices.len()` is the index
/// count and must be a multiple of 3; `vertex_positions_stride` is in bytes
/// (12..=256, multiple of 4); `resolution` must be in 4..=256.
#[allow(clippy::too_many_arguments)] // port-shape: mirrors the upstream parameter list (ADR-0005)
pub fn remesh(
    destination: Option<&mut [f32]>,
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    resolution: i32,
    options: u32,
) -> usize {
    let tables = tables::build_remesh_tables();

    // measure voxel grid to compute position => voxel mapping
    let mut offset = [0f32; 3];
    let scale = measure_grid(
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        resolution,
        &mut offset,
    );

    let res = resolution as usize;

    // rasterize triangles into the voxel grid: in the first pass, this simply tags occupied voxels
    let mut grid = vec![0u8; res * res * res];

    voxelize(
        &mut grid,
        &[],
        None,
        indices,
        vertex_positions,
        vertex_positions_stride,
        resolution,
        scale,
        &offset,
        options,
    );

    // compute row offsets for occupied voxels; also stores ~0u for empty rows to accelerate further processing
    let mut rowmap = vec![0u32; res * res];
    let voxel_count = rowpack(&mut grid, &mut rowmap, resolution);

    // allocate additional voxel data for each occupied voxel; this can be filled in the second pass to compute positions
    // note that we only do this if we need to compute output triangles; counting runs skip it for performance
    let mut voxels = None;

    if destination.is_some() {
        voxels = Some(vec![Voxel::default(); voxel_count]);
    }

    // fill in empty voxels that are not reachable from the grid boundary; the inside empty voxels are marked with 0xff
    // note that these voxels do *not* have the associated Voxel data as they were never voxelized
    if options & REMESH_SHELL == 0 {
        let mut worklist = vec![0u32; res * res];
        let mut queued = vec![0u8; res * res];

        solidify(&mut grid, &rowmap, &mut worklist, &mut queued, resolution);
    }

    // accumulate voxel positions: in the second pass, this computes enough data in each voxel to calculate positions
    if let Some(voxels) = voxels.as_deref_mut() {
        voxelize(
            &mut grid,
            &rowmap,
            Some(voxels),
            indices,
            vertex_positions,
            vertex_positions_stride,
            resolution,
            scale,
            &offset,
            options,
        );
    }

    // compute final voxel positions; each voxel has a single resulting position that will be emitted during polygonization
    if let Some(voxels) = voxels.as_deref_mut() {
        solve(voxels, scale, options);
    }

    // output triangles from the voxel grid; if destination is None, this still counts the number of triangles that would be generated
    polygonize::polygonize(
        destination,
        &grid,
        &rowmap,
        voxels.as_deref(),
        resolution,
        scale,
        &offset,
        options,
        &tables,
    )
}

#[derive(Clone, Default)]
struct Voxel {
    coord: u32,
    octants: u8,

    px: f32,
    py: f32,
    pz: f32,
    w: f32,

    // a00*x^2 + a11*y^2 + a22*z^2 + 2*a10*xy + 2*a20*xz + 2*a21*yz + 2*b0*x + 2*b1*y + 2*b2*z + c
    a00: f32,
    a11: f32,
    a22: f32,
    a10: f32,
    a20: f32,
    a21: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    c: f32,
}

fn voxel_accumulate(vox: &mut Voxel, px: f32, py: f32, pz: f32, w: f32) {
    vox.px += px * w;
    vox.py += py * w;
    vox.pz += pz * w;
    vox.w += w;
}

fn voxel_accumulate_q(vox: &mut Voxel, a: f32, b: f32, c: f32, d: f32, w: f32) {
    let aw = a * w;
    let bw = b * w;
    let cw = c * w;
    let dw = d * w;

    vox.a00 += a * aw;
    vox.a11 += b * bw;
    vox.a22 += c * cw;
    vox.a10 += a * bw;
    vox.a20 += a * cw;
    vox.a21 += b * cw;
    vox.b0 += a * dw;
    vox.b1 += b * dw;
    vox.b2 += c * dw;
    vox.c += d * dw;
}

fn voxel_solve(r: &mut [f32; 3], vox: &Voxel, lambda: f32) -> bool {
    // solve A*p = -b where A is the quadric matrix and b is the linear term
    let rw = lambda * vox.w;
    let a00 = rw + vox.a00;
    let a11 = rw + vox.a11;
    let a22 = rw + vox.a22;
    let a10 = vox.a10;
    let a20 = vox.a20;
    let a21 = vox.a21;
    let x0 = lambda * vox.px - vox.b0;
    let x1 = lambda * vox.py - vox.b1;
    let x2 = lambda * vox.pz - vox.b2;

    let eps = 1e-6f32 * vox.w;

    // LDL decomposition: A = LDL^T
    let d0 = a00;
    let l10 = a10 / d0;
    let l20 = a20 / d0;

    let d1 = a11 - a10 * l10;
    let dl21 = a21 - a20 * l10;
    let l21 = dl21 / d1;

    let d2 = a22 - a20 * l20 - dl21 * l21;

    // solve L*y = x
    let y0 = x0;
    let y1 = x1 - l10 * y0;
    let y2 = x2 - l20 * y0 - l21 * y1;

    // solve D*z = y
    let z0 = y0 / d0;
    let z1 = y1 / d1;
    let z2 = y2 / d2;

    // substitute L^T*p = z
    r[2] = z2;
    r[1] = z1 - l21 * r[2];
    r[0] = z0 - l10 * r[1] - l20 * r[2];

    d0.abs() > eps && d1.abs() > eps && d2.abs() > eps
}

fn measure_grid(
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    resolution: i32,
    out_offset: &mut [f32; 3],
) -> f32 {
    let vertex_stride_float = vertex_positions_stride / 4;

    let mut minv = [f32::MAX; 3];
    let mut maxv = [-f32::MAX; 3];

    for i in 0..vertex_count {
        let v = &vertex_positions[i * vertex_stride_float..];

        for j in 0..3 {
            let vj = v[j];

            minv[j] = if minv[j] > vj { vj } else { minv[j] };
            maxv[j] = if maxv[j] < vj { vj } else { maxv[j] };
        }
    }

    let mut extent = 0.0f32;

    extent = extent.max(maxv[0] - minv[0]);
    extent = extent.max(maxv[1] - minv[1]);
    extent = extent.max(maxv[2] - minv[2]);

    // small subvoxel padding keeps vertices inside the last valid voxel to avoid clamping correctness issues
    extent *= (resolution as f32 + 1e-2) / resolution as f32;

    // rescale extents to [0..resolution - 2], because the first and last voxels are used as empty padding
    let scale = if extent == 0.0 {
        0.0
    } else {
        (resolution - 2) as f32 / extent
    };

    // center mesh in the grid; this improves voxelization for symmetric inputs
    for j in 0..3 {
        out_offset[j] = minv[j] - (extent - (maxv[j] - minv[j])) * 0.5;
    }

    scale
}

#[allow(clippy::too_many_arguments)]
fn voxelize(
    grid: &mut [u8],
    rowmap: &[u32],
    mut voxels: Option<&mut [Voxel]>,
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_positions_stride: usize,
    resolution: i32,
    scale: f32,
    offset: &[f32; 3],
    options: u32,
) {
    let vertex_stride_float = vertex_positions_stride / 4;

    for t in 0..indices.len() / 3 {
        let a = indices[t * 3] as usize;
        let b = indices[t * 3 + 1] as usize;
        let c = indices[t * 3 + 2] as usize;

        let va = &vertex_positions[a * vertex_stride_float..];
        let vb = &vertex_positions[b * vertex_stride_float..];
        let vc = &vertex_positions[c * vertex_stride_float..];

        let ex = vb[0] - va[0];
        let ey = vb[1] - va[1];
        let ez = vb[2] - va[2];
        let fx = vc[0] - va[0];
        let fy = vc[1] - va[1];
        let fz = vc[2] - va[2];
        let gx = vc[0] - vb[0];
        let gy = vc[1] - vb[1];
        let gz = vc[2] - vb[2];

        // use maximum edge length to establish sampling rate
        let el2 = ex * ex + ey * ey + ez * ez;
        let fl2 = fx * fx + fy * fy + fz * fz;
        let gl2 = gx * gx + gy * gy + gz * gz;

        let mut max_edge = if el2 > fl2 { el2 } else { fl2 };
        max_edge = if max_edge > gl2 { max_edge } else { gl2 };
        let max_edge = math::sqrt(max_edge);

        // we target 2 samples per voxel edge which should be enough to hit all voxels at any rotation
        let samples = cast_i32(max_edge * scale * 2.0).clamp(1, resolution * 2);

        // normal is used to compute area for weighting as well as for the quadric
        let mut nx = ey * fz - ez * fy;
        let mut ny = ez * fx - ex * fz;
        let mut nz = ex * fy - ey * fx;
        let area = math::sqrt(nx * nx + ny * ny + nz * nz);

        // skip degenerate triangles; they don't contribute to voxelization and may result in NaN when computing voxel centroid
        if area == 0.0 {
            continue;
        }

        let ns = 1.0 / area;
        nx *= ns;
        ny *= ns;
        nz *= ns;

        let sx = va[0] - offset[0];
        let sy = va[1] - offset[1];
        let sz = va[2] - offset[2];
        let weight = area / ((samples + 1) * (samples + 2)) as f32;

        // this division is on a critical path, and for dense input meshes the branch is almost never taken so it's worth skipping it
        let sr = if samples > 1 {
            1.0 / samples as f32
        } else {
            1.0
        };

        for u in 0..=samples {
            for v in 0..=samples - u {
                let su = u as f32 * sr;
                let sv = v as f32 * sr;
                let px = sx + su * ex + sv * fx;
                let py = sy + su * ey + sv * fy;
                let pz = sz + su * ez + sv * fz;

                // compute coordinates with an extra bit to use in octant mask
                let hx = cast_i32(px * (scale * 2.0));
                let hy = cast_i32(py * (scale * 2.0));
                let hz = cast_i32(pz * (scale * 2.0));
                let mut x = hx >> 1;
                let mut y = hy >> 1;
                let mut z = hz >> 1;

                // safety: rounding errors and non-finite inputs may produce out of bounds coordinates, so we clamp them
                let cutoff = resolution - 3;
                x = if (x as u32) < (cutoff as u32) {
                    x
                } else {
                    cutoff
                };
                y = if (y as u32) < (cutoff as u32) {
                    y
                } else {
                    cutoff
                };
                z = if (z as u32) < (cutoff as u32) {
                    z
                } else {
                    cutoff
                };

                let res = resolution as usize;
                let row = (y as usize + 1) + res * (z as usize + 1);
                let idx = (x as usize + 1) + res * row;

                if let Some(voxels) = voxels.as_deref_mut() {
                    let vox = &mut voxels[rowmap[row] as usize + grid[idx] as usize - 1];

                    vox.coord = ((x as u32) << 20) | ((y as u32) << 10) | z as u32;
                    vox.octants |= 1u8 << ((hx & 1) | ((hy & 1) << 1) | ((hz & 1) << 2));

                    voxel_accumulate(vox, px, py, pz, weight);

                    if options & REMESH_SOLVE != 0 {
                        let distance = nx * px + ny * py + nz * pz;
                        voxel_accumulate_q(vox, nx, ny, nz, -distance, weight);
                    }
                } else {
                    grid[idx] = 1;
                }
            }
        }
    }
}

fn rowpack(grid: &mut [u8], rowmap: &mut [u32], resolution: i32) -> usize {
    let mut result = 0usize;
    let res = resolution as usize;
    let slice = res * res;

    for i in 0..slice {
        let data = &mut grid[i * res..(i + 1) * res];

        if data.iter().all(|&b| b == 0) {
            // mark empty rows with a sentinel to accelerate further processing
            rowmap[i] = EMPTY_SLOT;
            continue;
        }

        let mut count = 0u32;

        for b in data.iter_mut() {
            // voxelize only produces 0/1 values; packed offsets use a single byte
            // with 0 reserved for empty voxels and 0xff reserved for interior voxels
            count += u32::from(*b != 0);
            *b = if *b != 0 { count as u8 } else { 0 };
        }

        rowmap[i] = result as u32;
        result += count as usize;
    }

    result
}

fn solidify_queue(row: usize, worklist: &mut [u32], queued: &mut [u8], pending: &mut usize) {
    if queued[row] != 0 {
        return;
    }

    queued[row] = 1;
    worklist[*pending] = row as u32;
    *pending += 1;
}

fn solidify(
    grid: &mut [u8],
    rowmap: &[u32],
    worklist: &mut [u32],
    queued: &mut [u8],
    resolution: i32,
) {
    let mut pending = 0usize;
    let res = resolution as usize;
    queued[..res * res].fill(0);

    // mark the interior empty voxels as 'inside'; we will propagate 'empty' state from the boundary
    for z in 1..resolution - 1 {
        for y in 1..resolution - 1 {
            let yz = y as usize + res * z as usize;

            // rows without occupied voxels shortcircuit the interior propagation and are kept as 'empty'
            if rowmap[yz] == EMPTY_SLOT {
                continue;
            }

            let base = res * yz;

            for x in 1..res - 1 {
                grid[base + x] = if grid[base + x] == 0 {
                    0xff
                } else {
                    grid[base + x]
                };
            }
        }
    }

    // queue all rows for processing; note that boundary rows are always empty but we need them to propagate into neighboring rows
    for z in 0..resolution {
        for y in 0..resolution {
            solidify_queue(
                y as usize + res * z as usize,
                worklist,
                queued,
                &mut pending,
            );
        }
    }

    while pending > 0 {
        pending -= 1;
        let row = worklist[pending] as usize;
        queued[row] = 0;

        let base = res * row;

        // propagate outside state to the interior within row; rows without occupied voxels have no 'inside' voxels
        if rowmap[row] != EMPTY_SLOT {
            for x in 1..res - 1 {
                grid[base + x] = if grid[base + x] == 0xff && grid[base + x - 1] == 0 {
                    0
                } else {
                    grid[base + x]
                };
            }

            for x in (1..res - 1).rev() {
                grid[base + x] = if grid[base + x] == 0xff && grid[base + x + 1] == 0 {
                    0
                } else {
                    grid[base + x]
                };
            }
        }

        // propagate outside state to the interior into neighboring rows
        let y = row % res;
        let z = row / res;

        for k in 0..4 {
            let yn = y as i32
                + if k == 0 {
                    -1
                } else if k == 1 {
                    1
                } else {
                    0
                };
            let zn = z as i32
                + if k == 2 {
                    -1
                } else if k == 3 {
                    1
                } else {
                    0
                };

            if yn < 1 || yn >= resolution - 1 || zn < 1 || zn >= resolution - 1 {
                continue;
            }

            let yzn = yn as usize + res * zn as usize;

            // neighboring rows without occupied voxels have no 'inside' voxels and can be skipped
            if rowmap[yzn] == EMPTY_SLOT {
                continue;
            }

            let nbase = res * yzn;
            let mut changed = 0u8;

            for x in 1..res - 1 {
                let rep = if grid[base + x] == 0 && grid[nbase + x] == 0xff {
                    0
                } else {
                    grid[nbase + x]
                };
                changed |= rep ^ grid[nbase + x];
                grid[nbase + x] = rep;
            }

            if changed != 0 {
                solidify_queue(yzn, worklist, queued, &mut pending);
            }
        }
    }
}

fn solve(voxels: &mut [Voxel], scale: f32, options: u32) {
    // regularization factor is a tradeoff between solve stability/uniformity and adherence to the original surface
    let factor = 3e-2f32;

    // cutoff is a squared distance in model space; sqrt(3)/scale is the voxel diagonal
    let cutoff = 3.0 / (scale * scale);
    let rscale = 1.0 / scale;

    for vox in voxels.iter_mut() {
        let mut px = vox.px / vox.w;
        let mut py = vox.py / vox.w;
        let mut pz = vox.pz / vox.w;

        // voxel corner, used for clamping and debug visualization
        let cx = ((vox.coord >> 20) & 1023) as f32 * rscale;
        let cy = ((vox.coord >> 10) & 1023) as f32 * rscale;
        let cz = (vox.coord & 1023) as f32 * rscale;

        if options & REMESH_SOLVE != 0 {
            // compute minimizing point; this includes regularization to stabilize the solve
            let mut s = [0f32; 3];
            if voxel_solve(&mut s, vox, factor) {
                let d = (s[0] - px) * (s[0] - px)
                    + (s[1] - py) * (s[1] - py)
                    + (s[2] - pz) * (s[2] - pz);

                // reject solutions that move the vertex too far from voxel centroid; this is a safety net in case regularization is insufficient
                if d < cutoff {
                    // use a clamped solution to reduce intersections
                    px = s[0].clamp(cx, cx + rscale);
                    py = s[1].clamp(cy, cy + rscale);
                    pz = s[2].clamp(cz, cz + rscale);
                }
            }
        }

        vox.px = px;
        vox.py = py;
        vox.pz = pz;
    }
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_remesh.
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

        fn i32(&mut self) -> i32 {
            self.u32() as i32
        }

        fn bytes(&mut self, n: usize) -> &'a [u8] {
            let s = &self.data[self.pos..self.pos + n];
            self.pos += n;
            s
        }

        fn f32s(&mut self, count: usize) -> Vec<f32> {
            let raw = self.bytes(count * 4);
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect()
        }

        fn vec_u32(&mut self, n: usize) -> Vec<u32> {
            (0..n).map(|_| self.u32()).collect()
        }
    }

    #[test]
    fn golden_remesh() {
        const _LEN_CHECK: usize = 1085224;
        let data = include_bytes!("../../tests/data/rm_remesh.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let pos_stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let resolution = c.i32();
            let options = c.u32();
            let positions = c.f32s(vertex_count * pos_stride / 4);
            let indices = c.vec_u32(index_count);

            let expected_bound = c.u32();
            let expected_result = c.u32();
            let expected = c.f32s(expected_result as usize * 9);

            let bound = remesh(
                None,
                &indices,
                &positions,
                vertex_count,
                pos_stride,
                resolution,
                options,
            );
            assert_eq!(bound, expected_bound as usize, "remesh case {case} bound");

            let mut destination = vec![0.0f32; bound * 9];
            let result = remesh(
                Some(&mut destination),
                &indices,
                &positions,
                vertex_count,
                pos_stride,
                resolution,
                options,
            );

            assert_eq!(
                result, expected_result as usize,
                "remesh case {case} result"
            );
            assert_eq!(
                &destination[..result * 9],
                expected.as_slice(),
                "remesh case {case} (resolution {resolution}, options {options:#x})"
            );
            case += 1;
        }
        assert_eq!(case, 10);
    }

    #[test]
    fn voxel_solve_rejects_singular_system() {
        // w == 0 makes eps == 0 and d0 == 0, so no pivot can exceed it
        let vox = Voxel::default();
        let mut r = [0f32; 3];
        assert!(!voxel_solve(&mut r, &vox, 3e-2));

        // identity quadric with unit weight solves to the linear term
        let vox = Voxel {
            w: 1.0,
            a00: 1.0,
            a11: 1.0,
            a22: 1.0,
            b0: 2.0,
            b1: 4.0,
            b2: 6.0,
            ..Voxel::default()
        };
        assert!(voxel_solve(&mut r, &vox, 0.0));
        assert_eq!(r, [-2.0, -4.0, -6.0]);
    }

    #[test]
    fn measure_grid_centers_and_scales() {
        // single vertex: zero extent -> zero scale, offset centers on the vertex
        let mut offset = [0f32; 3];
        let scale = measure_grid(&[1.0, 2.0, 3.0], 1, 12, 16, &mut offset);
        assert_eq!(scale, 0.0);
        assert_eq!(offset, [1.0, 2.0, 3.0]);

        // unit cube: extent padded by (resolution + 1e-2) / resolution
        let positions = [
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            1.0, 1.0, 1.0,
        ];
        let scale = measure_grid(&positions, 4, 12, 16, &mut offset);
        let extent = 1.0 * (16.0 + 1e-2) / 16.0;
        assert_eq!(scale, 14.0 / extent);
        assert!((offset[0] - (1.0 - extent) * 0.5).abs() < 1e-6);
    }
}
