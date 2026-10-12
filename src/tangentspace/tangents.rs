//! `meshopt_generateTangents` (FR-16; tangentspace.cpp). Per-corner tangent
//! groups are merged along orientation-consistent strips, accumulated with
//! incident-angle weighting, and finalized through face orientation groups.

use super::{
    TANGENT_COMPATIBLE, TANGENT_ZERO_FALLBACK, build_corner_adjacency, build_vertex_remap,
    face_index, follow2, optacos,
};
use crate::math;
use alloc::vec;
use alloc::vec::Vec;

#[derive(Clone, Copy)]
struct Tangent {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

#[allow(clippy::needless_range_loop)]
fn compute_face_tangents(
    result: &mut [Tangent],
    indices: Option<&[u32]>,
    vertex_positions: &[f32],
    vertex_positions_stride: usize,
    vertex_uvs: &[f32],
    vertex_uvs_stride: usize,
) {
    let pos_stride = vertex_positions_stride / 4;
    let uv_stride = vertex_uvs_stride / 4;

    for i in 0..result.len() {
        let a = face_index(indices, i * 3) as usize;
        let b = face_index(indices, i * 3 + 1) as usize;
        let c = face_index(indices, i * 3 + 2) as usize;

        let pa = &vertex_positions[a * pos_stride..];
        let pb = &vertex_positions[b * pos_stride..];
        let pc = &vertex_positions[c * pos_stride..];
        let ta = &vertex_uvs[a * uv_stride..];
        let tb = &vertex_uvs[b * uv_stride..];
        let tc = &vertex_uvs[c * uv_stride..];

        // compute face tangents using chain rule, see eq. 42
        let dp1x = pb[0] - pa[0];
        let dp1y = pb[1] - pa[1];
        let dp1z = pb[2] - pa[2];
        let dp2x = pc[0] - pa[0];
        let dp2y = pc[1] - pa[1];
        let dp2z = pc[2] - pa[2];
        let dt1x = tb[0] - ta[0];
        let dt1y = tb[1] - ta[1];
        let dt2x = tc[0] - ta[0];
        let dt2y = tc[1] - ta[1];

        let rx = dt2y * dp1x - dt1y * dp2x;
        let ry = dt2y * dp1y - dt1y * dp2y;
        let rz = dt2y * dp1z - dt1y * dp2z;

        // .w stores orientation (+1/-1) or 0 for degenerate triangles, see eq. 48
        let det = dt1x * dt2y - dt1y * dt2x;
        let mut rw = if det == 0.0 {
            0.0
        } else if det > 0.0 {
            1.0
        } else {
            -1.0
        };

        // treat zero area triangles as degenerate even if their UVs aren't for consistency
        rw = if pa[0] == pb[0] && pa[1] == pb[1] && pa[2] == pb[2] {
            0.0
        } else {
            rw
        };
        rw = if pa[0] == pc[0] && pa[1] == pc[1] && pa[2] == pc[2] {
            0.0
        } else {
            rw
        };
        rw = if pb[0] == pc[0] && pb[1] == pc[1] && pb[2] == pc[2] {
            0.0
        } else {
            rw
        };

        let rl = math::sqrt(rx * rx + ry * ry + rz * rz);
        let rs = if rl != 0.0 { rw / rl } else { 0.0 };

        result[i] = Tangent {
            x: rx * rs,
            y: ry * rs,
            z: rz * rs,
            w: rw,
        };
    }
}

#[allow(clippy::needless_range_loop)]
fn merge_tangent_groups(
    groups: &mut [u32],
    facegroups: &mut [u32],
    facesign: &mut [u8],
    data: &[u32],
    indices: Option<&[u32]>,
    remap: &[u32],
) {
    const NEXT: [usize; 4] = [1, 2, 0, 1];

    for i in 0..data.len() {
        let ci = (data[i] & 3) as usize;
        let ti = (data[i] >> 2) as usize;
        let cib = ti * 3 + NEXT[ci];
        let cic = ti * 3 + NEXT[ci + 1];

        let nib = remap[face_index(indices, cib) as usize];
        let nic = remap[face_index(indices, cic) as usize];

        for j in i + 1..data.len() {
            let cj = (data[j] & 3) as usize;
            let tj = (data[j] >> 2) as usize;
            let cjb = tj * 3 + NEXT[cj];
            let cjc = tj * 3 + NEXT[cj + 1];

            let njb = remap[face_index(indices, cjb) as usize];
            let njc = remap[face_index(indices, cjc) as usize];

            // merge tangent groups if triangles are adjacent and orientations agree or either one is degenerate
            if (njb == nic || njc == nib) && (facesign[ti] | facesign[tj]) != 3 {
                // if either triangle is degenerate, it could have been assigned to a concrete orientation; we need to confirm that via facegroups
                if (facesign[ti] & facesign[tj]) == 0 {
                    let gti = follow2(facegroups, ti as u32);
                    let gtj = follow2(facegroups, tj as u32);

                    // skip merge if orientations diverge; otherwise, union the groups with gti as the root
                    if gti != gtj {
                        if (facesign[gti as usize] | facesign[gtj as usize]) == 3 {
                            continue;
                        }

                        facegroups[gtj as usize] = gti;
                        facesign[gti as usize] |= facesign[gtj as usize];
                    }
                }

                // union tangent groups for individual corners with gi as the root; the orientations must match due to facegroups check above
                let gi = follow2(groups, (ti * 3 + ci) as u32);
                let gj = follow2(groups, (tj * 3 + cj) as u32);

                if gi != gj {
                    groups[gj as usize] = gi;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_tangents(
    result: &mut [f32],
    groups: &[u32],
    indices: Option<&[u32]>,
    index_count: usize,
    face_tangents: &[Tangent],
    vertex_positions: &[f32],
    vertex_positions_stride: usize,
    vertex_normals: &[f32],
    vertex_normals_stride: usize,
    options: u32,
) {
    const NEXT: [usize; 4] = [1, 2, 0, 1];

    let pos_stride = vertex_positions_stride / 4;
    let nrm_stride = vertex_normals_stride / 4;

    let face_count = index_count / 3;

    for i in 0..face_count {
        let t = &face_tangents[i];

        if t.w == 0.0 {
            continue;
        }

        for k in 0..3 {
            let base = groups[i * 3 + k] as usize * 4;

            let a = face_index(indices, i * 3 + k) as usize;
            let b = face_index(indices, i * 3 + NEXT[k]) as usize;
            let c = face_index(indices, i * 3 + NEXT[k + 1]) as usize;

            let pa = &vertex_positions[a * pos_stride..];
            let pb = &vertex_positions[b * pos_stride..];
            let pc = &vertex_positions[c * pos_stride..];
            let n = &vertex_normals[a * nrm_stride..];

            // reproject tangent vector onto vertex normal
            let sd = t.x * n[0] + t.y * n[1] + t.z * n[2];
            let sx = t.x - n[0] * sd;
            let sy = t.y - n[1] * sd;
            let sz = t.z - n[2] * sd;
            let sl = math::sqrt(sx * sx + sy * sy + sz * sz);

            // compute incident angle for weighting, in tangent plane
            // note: this step is absent from the paper, reference implementation computes angle after projecting edges to tangent plane...
            let mut dp1x = pb[0] - pa[0];
            let mut dp1y = pb[1] - pa[1];
            let mut dp1z = pb[2] - pa[2];
            let mut dp2x = pc[0] - pa[0];
            let mut dp2y = pc[1] - pa[1];
            let mut dp2z = pc[2] - pa[2];

            let dp1d = dp1x * n[0] + dp1y * n[1] + dp1z * n[2];
            let dp2d = dp2x * n[0] + dp2y * n[1] + dp2z * n[2];

            dp1x -= n[0] * dp1d;
            dp1y -= n[1] * dp1d;
            dp1z -= n[2] * dp1d;
            dp2x -= n[0] * dp2d;
            dp2y -= n[1] * dp2d;
            dp2z -= n[2] * dp2d;

            let dp1l = dp1x * dp1x + dp1y * dp1y + dp1z * dp1z;
            let dp2l = dp2x * dp2x + dp2y * dp2y + dp2z * dp2z;
            let dpl = math::sqrt(dp1l * dp2l);

            let cosa = (dp1x * dp2x + dp1y * dp2y + dp1z * dp2z)
                * (if dpl == 0.0 { 0.0 } else { 1.0 / dpl });
            let angle = optacos(cosa); // optacos handles clamping to [-1..1]

            // accumulate renormalized tangent weighted by angle
            let mut w = angle * (if sl == 0.0 { 0.0 } else { 1.0 / sl });

            // weight larger adjacent triangles more to reduce interpolation artifacts from slivers; this deviates from reference implementation for higher quality
            if options & TANGENT_COMPATIBLE == 0 {
                w *= dpl;
            }

            result[base] += sx * w;
            result[base + 1] += sy * w;
            result[base + 2] += sz * w;
        }
    }
}

// port-shape: mirrors the upstream parameter list (ADR-0005)
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
#[allow(clippy::approx_constant)] // upstream's literal 3.1415927 bound in the assert
pub fn generate_tangents(
    result: &mut [f32],
    indices: Option<&[u32]>,
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_normals: &[f32],
    vertex_normals_stride: usize,
    vertex_uvs: &[f32],
    vertex_uvs_stride: usize,
    options: u32,
) {
    debug_assert!(indices.is_some() || index_count == vertex_count);
    debug_assert!(index_count.is_multiple_of(3));
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!((12..=256).contains(&vertex_normals_stride));
    debug_assert!((8..=256).contains(&vertex_uvs_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
    debug_assert!(vertex_normals_stride.is_multiple_of(4));
    debug_assert!(vertex_uvs_stride.is_multiple_of(4));
    debug_assert!(result.len() >= index_count * 4);

    let face_count = index_count / 3;

    // compute vertex remap to unique vertex index
    let mut remap = vec![0u32; vertex_count];
    build_vertex_remap::<false>(
        &mut remap,
        vertex_positions,
        vertex_positions_stride,
        vertex_normals,
        vertex_normals_stride,
        vertex_uvs,
        vertex_uvs_stride,
    );

    // build adjacency information
    let adjacency = build_corner_adjacency(indices, index_count, &remap, vertex_count);

    // compute per-triangle tangents
    let mut face_tangents = vec![
        Tangent {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0
        };
        face_count
    ];
    compute_face_tangents(
        &mut face_tangents,
        indices,
        vertex_positions,
        vertex_positions_stride,
        vertex_uvs,
        vertex_uvs_stride,
    );

    // compute per-corner tangent groups: triangles adjacent to the same vertex with the same orientation are merged in strips
    let mut groups: Vec<u32> = (0..index_count as u32).collect();

    // compute per-face orientation groups: degenerate triangles are merged with adjacent non-degenerate ones and inherit their orientation, stored as a bitmask
    let mut facegroups: Vec<u32> = (0..face_count as u32).collect();
    let mut facesign = vec![0u8; face_count];

    for i in 0..face_count {
        facesign[i] =
            u8::from(face_tangents[i].w > 0.0) | (u8::from(face_tangents[i].w < 0.0) << 1);
    }

    for i in 0..vertex_count {
        if adjacency.offsets[i + 1] != adjacency.offsets[i] {
            merge_tangent_groups(
                &mut groups,
                &mut facegroups,
                &mut facesign,
                &adjacency.data[adjacency.offsets[i] as usize..adjacency.offsets[i + 1] as usize],
                indices,
                &remap,
            );
        }
    }

    for i in 0..index_count {
        let root = follow2(&mut groups, i as u32);
        groups[i] = root;
    }

    // accumulate tangents into their own respective groups
    result[..index_count * 4].fill(0.0);
    accumulate_tangents(
        result,
        &groups,
        indices,
        index_count,
        &face_tangents,
        vertex_positions,
        vertex_positions_stride,
        vertex_normals,
        vertex_normals_stride,
        options,
    );

    // finalize tangent signs for each face using facegroups
    for i in 0..face_count {
        let fg = follow2(&mut facegroups, i as u32) as usize;
        let w = if facesign[fg] & 1 != 0 { 1.0 } else { -1.0 }; // note: if face is degenerate, we canonicalize to -1 for consistency

        result[(i * 3) * 4 + 3] = w;
        result[(i * 3 + 1) * 4 + 3] = w;
        result[(i * 3 + 2) * 4 + 3] = w;
    }

    // finalize tangent vectors by normalizing roots and propagating the rest
    for i in 0..index_count {
        if groups[i] as usize == i {
            let r = i * 4;
            let l = math::sqrt(
                result[r] * result[r]
                    + result[r + 1] * result[r + 1]
                    + result[r + 2] * result[r + 2],
            );
            let s = if l == 0.0 { 0.0 } else { 1.0 / l };

            result[r] *= s;
            result[r + 1] *= s;
            result[r + 2] *= s;

            // for isolated degenerate triangles, use (1, 0, 0) tangent for consistency
            if l == 0.0 && options & TANGENT_ZERO_FALLBACK == 0 {
                result[r] = 1.0;
            }
        }
    }

    for i in 0..index_count {
        if groups[i] as usize != i {
            let g = groups[i] as usize * 4;
            result[i * 4] = result[g];
            result[i * 4 + 1] = result[g + 1];
            result[i * 4 + 2] = result[g + 2]; // .w was set per face earlier
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        fn f32s(&mut self, count: usize) -> Vec<f32> {
            let raw = self.bytes(count * 4);
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect()
        }
    }

    #[test]
    fn golden_generate_tangents() {
        const _LEN_CHECK: usize = 85924;
        let data = include_bytes!("../../tests/data/tan_tangents.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let index_count = c.u32() as usize;
            let has_indices = c.u32();
            let pos_stride = c.u32() as usize;
            let nrm_stride = c.u32() as usize;
            let uv_stride = c.u32() as usize;
            let options = c.u32();
            let positions = c.f32s(vertex_count * pos_stride / 4);
            let normals = c.f32s(vertex_count * nrm_stride / 4);
            let uvs = c.f32s(vertex_count * uv_stride / 4);
            let indices = if has_indices == 1 {
                Some(c.vec_u32(index_count))
            } else {
                None
            };

            let expected = c.f32s(index_count * 4);

            let mut result = vec![0.0f32; index_count * 4];
            generate_tangents(
                &mut result,
                indices.as_deref(),
                index_count,
                &positions,
                vertex_count,
                pos_stride,
                &normals,
                nrm_stride,
                &uvs,
                uv_stride,
                options,
            );

            assert_eq!(
                result.as_slice(),
                expected.as_slice(),
                "generateTangents case {case} (options {options:#x})"
            );
            case += 1;
        }
        assert_eq!(case, 11);
    }
}
