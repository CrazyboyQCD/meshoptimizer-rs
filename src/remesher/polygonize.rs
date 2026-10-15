//! Isosurface extraction for the remesher: per-voxel position emission,
//! alternate-configuration deciders, and marching-cubes polygonization.
//! Upstream: `emitVertex` (inlined into `polygonize` with a per-cell corner
//! cache), `octantDecider`, `quadricDecider`, `polygonize` (remesher.cpp).

use super::tables::RemeshTables;
use super::{EMPTY_SLOT, REMESH_SOLVE, Voxel};

fn voxel_error(vox: &Voxel, x: f32, y: f32, z: f32) -> f32 {
    let rx = (vox.b0 + vox.a10 * y) * 2.0 + vox.a00 * x;
    let ry = (vox.b1 + vox.a21 * z) * 2.0 + vox.a11 * y;
    let rz = (vox.b2 + vox.a20 * x) * 2.0 + vox.a22 * z;

    (vox.c + rx * x + ry * y + rz * z).abs()
}

#[allow(clippy::too_many_arguments)]
fn octant_decider(
    x: i32,
    y: i32,
    z: i32,
    cube: i32,
    grid: &[u8],
    rowmap: &[u32],
    voxels: &[Voxel],
    resolution: i32,
) -> bool {
    let res = resolution as usize;

    for c in 0..8 {
        if cube & (1 << c) != 0 {
            let ox = c & 1;
            let oy = (c >> 1) & 1;
            let oz = (c >> 2) & 1;

            let row = (y as usize + oy as usize) + res * (z as usize + oz as usize);
            let idx = (x as usize + ox as usize) + res * row;

            // note: unlike vertex emission, which never consults interior occupied corners, here we need to skip corners without an occupied voxel
            if grid[idx] == 0xff {
                continue;
            }

            let vox = &voxels[rowmap[row] as usize + grid[idx] as usize - 1];

            // test octant contained within the cell (opposite of the corner index)
            if vox.octants & (1u8 << (7 - c)) != 0 {
                return false;
            }
        }
    }

    // select alternate configuration if *all* octants are empty
    true
}

#[allow(clippy::too_many_arguments)]
fn quadric_decider(
    x: i32,
    y: i32,
    z: i32,
    cube: i32,
    grid: &[u8],
    rowmap: &[u32],
    voxels: &[Voxel],
    resolution: i32,
    rscale: f32,
    tables: &RemeshTables,
) -> bool {
    let res = resolution as usize;

    // quads are encoded implicitly as 0xabc 0xcbd
    let quad = (u32::from(tables.triangles[cube as usize][0][0]) << 4)
        | (u32::from(tables.triangles[cube as usize][0][1]) & 0xf);

    let vidx: [usize; 4] = core::array::from_fn(|i| {
        let c = ((quad >> (12 - i * 4)) & 0xf) as usize;
        let row = (y as usize + ((c >> 1) & 1)) + res * (z as usize + ((c >> 2) & 1));
        let idx = (x as usize + (c & 1)) + res * row;

        rowmap[row] as usize + grid[idx] as usize - 1
    });

    let (va, vb, vc, vd) = (
        &voxels[vidx[0]],
        &voxels[vidx[1]],
        &voxels[vidx[2]],
        &voxels[vidx[3]],
    );

    // accumulate the quadrics of all four corners; we sum (a+d)+(b+c) to get the same results on both sides of thin sheets
    let sum = Voxel {
        a00: (va.a00 + vd.a00) + (vb.a00 + vc.a00),
        a11: (va.a11 + vd.a11) + (vb.a11 + vc.a11),
        a22: (va.a22 + vd.a22) + (vb.a22 + vc.a22),
        a10: (va.a10 + vd.a10) + (vb.a10 + vc.a10),
        a20: (va.a20 + vd.a20) + (vb.a20 + vc.a20),
        a21: (va.a21 + vd.a21) + (vb.a21 + vc.a21),
        b0: (va.b0 + vd.b0) + (vb.b0 + vc.b0),
        b1: (va.b1 + vd.b1) + (vb.b1 + vc.b1),
        b2: (va.b2 + vd.b2) + (vb.b2 + vc.b2),
        c: (va.c + vd.c) + (vb.c + vc.c),
        w: (va.w + vd.w) + (vb.w + vc.w),
        ..Voxel::default()
    };

    // evaluate error for midpoints of primary (bc) and alternate (ad) diagonal
    let error0 = voxel_error(
        &sum,
        (vb.px + vc.px) * 0.5,
        (vb.py + vc.py) * 0.5,
        (vb.pz + vc.pz) * 0.5,
    );
    let error1 = voxel_error(
        &sum,
        (va.px + vd.px) * 0.5,
        (va.py + vd.py) * 0.5,
        (va.pz + vd.pz) * 0.5,
    );

    // select alternate configuration if it is clearly better and if the primary error is not too small, to avoid excessive flips
    error0 > sum.w * (0.1 * 0.1 * rscale * rscale) && error1 < error0 * (0.85 * 0.85)
}

#[allow(clippy::too_many_arguments)]
fn polygonize_count(grid: &[u8], rowmap: &[u32], resolution: i32, tables: &RemeshTables) -> usize {
    let mut result = 0usize;
    let res = resolution as usize;
    let slice = res * res;

    // counting run: scan plus per-cube triangle counts; no deciders or emission
    for z in 0..resolution - 1 {
        for y in 0..resolution - 1 {
            let row = y as usize + res * z as usize;

            // skip processing if all four rows are entirely empty as that guarantees empty output (even three out of four would be enough but that's more expensive to check)
            if (rowmap[row] & rowmap[row + 1] & rowmap[row + res] & rowmap[row + res + 1])
                == EMPTY_SLOT
            {
                continue;
            }

            let base = res * row;

            // four grid rows feeding the cube codes; hoisting them as slices removes
            // the per-cell bounds checks in the scan loop below
            let r0 = &grid[base..base + res];
            let r1 = &grid[base + res..base + 2 * res];
            let r2 = &grid[base + slice..base + slice + res];
            let r3 = &grid[base + slice + res..base + slice + 2 * res];

            // we track each slice as a 8-bit code (matching cube indexing) as we iterate through the row to avoid extra lookups
            let mut last = (r0[0] != 0) as i32
                | ((r1[0] != 0) as i32) << 2
                | ((r2[0] != 0) as i32) << 4
                | ((r3[0] != 0) as i32) << 6;

            for (g0, (g1, (g2, g3))) in r0[1..]
                .iter()
                .zip(r1[1..].iter().zip(r2[1..].iter().zip(r3[1..].iter())))
            {
                let next = ((*g0 != 0) as i32)
                    | ((*g1 != 0) as i32) << 2
                    | ((*g2 != 0) as i32) << 4
                    | ((*g3 != 0) as i32) << 6;

                // next slice bits are in position for the next iteration so we need to shift them to retrieve current cube code
                let cube = last | (next << 1);
                last = next;

                if cube == 0 || cube == 0xff {
                    continue;
                }

                // fast path: we can statically determine the upper bound on triangles the loop below will output based on cube configuration
                result += tables.count[cube as usize] as usize;
            }
        }
    }

    result
}

#[allow(clippy::too_many_arguments)]
pub(super) fn polygonize(
    destination: Option<&mut [f32]>,
    grid: &[u8],
    rowmap: &[u32],
    voxels: Option<&[Voxel]>,
    resolution: i32,
    scale: f32,
    offset: &[f32; 3],
    options: u32,
    tables: &RemeshTables,
) -> usize {
    let mut result = 0usize;
    let res = resolution as usize;
    let slice = res * res;
    let rscale = 1.0 / scale;

    // the destination is fixed per call, so the count/fill dispatch is hoisted out of
    // the scan instead of re-testing the Option per non-empty cell; a separate count
    // function keeps its codegen independent of the fat emit body
    let Some(dest) = destination else {
        return polygonize_count(grid, rowmap, resolution, tables);
    };

    // counting runs never reach deciders or emission, so the empty fallback is
    // never dereferenced there
    let voxel_data: &[Voxel] = voxels.unwrap_or(&[]);
    let max_triangle_count = dest.len() / 9;

    for z in 0..resolution - 1 {
        for y in 0..resolution - 1 {
            let row = y as usize + res * z as usize;

            let (rm0, rm1, rm2, rm3) = (
                rowmap[row],
                rowmap[row + 1],
                rowmap[row + res],
                rowmap[row + res + 1],
            );

            // skip processing if all four rows are entirely empty as that guarantees empty output (even three out of four would be enough but that's more expensive to check)
            if (rm0 & rm1 & rm2 & rm3) == EMPTY_SLOT {
                continue;
            }
            let base = res * row;

            // the grid bases of the 2x2 row block, precomputed as scalars so the emit
            // path's per-corner select stays on registers (selecting between slice
            // references would force a jump table)
            let (go1, go2, go3) = (base + res, base + slice, base + slice + res);

            // four grid rows feeding the cube codes; hoisting them as slices removes
            // the per-cell bounds checks in the scan loop below
            let r0 = &grid[base..base + res];
            let r1 = &grid[base + res..base + 2 * res];
            let r2 = &grid[base + slice..base + slice + res];
            let r3 = &grid[base + slice + res..base + slice + 2 * res];

            // we track each slice as a 8-bit code (matching cube indexing) as we iterate through the row to avoid extra lookups
            let mut last = (r0[0] != 0) as i32
                | ((r1[0] != 0) as i32) << 2
                | ((r2[0] != 0) as i32) << 4
                | ((r3[0] != 0) as i32) << 6;

            for (x, (g0, (g1, (g2, g3)))) in (0..resolution - 1).zip(
                r0[1..]
                    .iter()
                    .zip(r1[1..].iter().zip(r2[1..].iter().zip(r3[1..].iter()))),
            ) {
                let next = ((*g0 != 0) as i32)
                    | ((*g1 != 0) as i32) << 2
                    | ((*g2 != 0) as i32) << 4
                    | ((*g3 != 0) as i32) << 6;

                // next slice bits are in position for the next iteration so we need to shift them to retrieve current cube code
                let cube = last | (next << 1);
                last = next;

                if cube == 0 || cube == 0xff {
                    continue;
                }

                // deciders are only evaluated for cells with an alternate configuration to reduce overhead
                let mut alt = 0usize;

                if tables.alt[cube as usize] == 1 {
                    alt = octant_decider(x, y, z, cube, grid, rowmap, voxel_data, resolution)
                        as usize;
                } else if tables.alt[cube as usize] == 2 && options & REMESH_SOLVE != 0 {
                    alt = quadric_decider(
                        x, y, z, cube, grid, rowmap, voxel_data, resolution, rscale, tables,
                    ) as usize;
                }

                let tris = &tables.triangles[cube as usize][alt];

                for tri in tris.iter() {
                    if *tri == 0 {
                        break;
                    }

                    // note: we only emit the triangle if we have space for it, but we count it regardless for consistent capacity estimation
                    if result < max_triangle_count {
                        let tri = *tri;

                        let slot = &mut dest[result * 9..result * 9 + 9];
                        let (s0, rest) = slot.split_at_mut(3);
                        let (s1, s2) = rest.split_at_mut(3);

                        for (corner, dst) in [
                            (i32::from((tri >> 8) & 0xf), s0),
                            (i32::from((tri >> 4) & 0xf), s1),
                            (i32::from(tri & 0xf), s2),
                        ] {
                            // the row's grid bases and rowmap offsets are preloaded as
                            // scalars, so the gather is one select plus two loads. Emitted
                            // corners are always occupied (never 0 or 0xff interior; see
                            // the note in `octant_decider`), so the byte is a valid
                            // 1-based voxel index
                            let (go, rmv) = match (corner >> 1) & 3 {
                                0 => (base, rm0),
                                1 => (go1, rm1),
                                2 => (go2, rm2),
                                _ => (go3, rm3),
                            };

                            let b = grid[go + x as usize + (corner & 1) as usize];
                            let vox = &voxel_data[rmv as usize + b as usize - 1];

                            dst[0] = vox.px + offset[0];
                            dst[1] = vox.py + offset[1];
                            dst[2] = vox.pz + offset[2];
                        }
                    }

                    result += 1;
                }
            }
        }
    }

    result
}
