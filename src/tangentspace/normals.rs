//! `meshopt_generateNormals` (FR-18, experimental upstream; tangentspace.cpp).
//! Per-corner normal groups merged by crease angle, angle-weighted
//! accumulation, and optional iterative normal smoothing.

use super::{build_corner_adjacency, build_vertex_remap, face_index, follow2, optacos};
use crate::math;
use crate::quantization::cast_i32;
use alloc::vec;
use alloc::vec::Vec;

#[derive(Clone, Copy)]
struct Normal {
    x: f32,
    y: f32,
    z: f32,
    id: u32,
}

#[allow(clippy::needless_range_loop)]
fn compute_face_normals(
    result: &mut [Normal],
    indices: Option<&[u32]>,
    vertex_positions: &[f32],
    vertex_positions_stride: usize,
) {
    let pos_stride = vertex_positions_stride / 4;

    for i in 0..result.len() {
        let a = face_index(indices, i * 3) as usize;
        let b = face_index(indices, i * 3 + 1) as usize;
        let c = face_index(indices, i * 3 + 2) as usize;

        let pa = &vertex_positions[a * pos_stride..];
        let pb = &vertex_positions[b * pos_stride..];
        let pc = &vertex_positions[c * pos_stride..];

        let dp1x = pb[0] - pa[0];
        let dp1y = pb[1] - pa[1];
        let dp1z = pb[2] - pa[2];
        let dp2x = pc[0] - pa[0];
        let dp2y = pc[1] - pa[1];
        let dp2z = pc[2] - pa[2];

        let nx = dp1y * dp2z - dp1z * dp2y;
        let ny = dp1z * dp2x - dp1x * dp2z;
        let nz = dp1x * dp2y - dp1y * dp2x;

        let nl = math::sqrt(nx * nx + ny * ny + nz * nz);
        let ns = if nl != 0.0 { 1.0 / nl } else { 0.0 };

        result[i] = Normal {
            x: nx * ns,
            y: ny * ns,
            z: nz * ns,
            id: u32::from(nl != 0.0),
        };
    }
}

#[allow(clippy::needless_range_loop)]
fn merge_normal_groups(
    groups: &mut [u32],
    data: &[u32],
    indices: Option<&[u32]>,
    remap: &[u32],
    face_normals: &[Normal],
    crease_cutoff: f32,
) {
    const NEXT: [usize; 4] = [1, 2, 0, 1];

    for i in 0..data.len() {
        let ci = (data[i] & 3) as usize;
        let ti = (data[i] >> 2) as usize;
        let cib = ti * 3 + NEXT[ci];
        let cic = ti * 3 + NEXT[ci + 1];

        let nib = remap[face_index(indices, cib) as usize];
        let nic = remap[face_index(indices, cic) as usize];

        let fi = &face_normals[ti];

        for j in i + 1..data.len() {
            let cj = (data[j] & 3) as usize;
            let tj = (data[j] >> 2) as usize;
            let cjb = tj * 3 + NEXT[cj];
            let cjc = tj * 3 + NEXT[cj + 1];

            let njb = remap[face_index(indices, cjb) as usize];
            let njc = remap[face_index(indices, cjc) as usize];

            let fj = &face_normals[tj];

            // merge normal groups if triangles are adjacent and normal angle is below crease threshold
            if (njb == nic || njc == nib)
                && fi.x * fj.x + fi.y * fj.y + fi.z * fj.z > crease_cutoff
                && fi.id == fj.id
            {
                // union normal groups for individual corners with gi as the root
                let gi = follow2(groups, (ti * 3 + ci) as u32);
                let gj = follow2(groups, (tj * 3 + cj) as u32);

                if gi != gj {
                    groups[gj as usize] = gi;
                }
            }
        }
    }
}

fn accumulate_normals(
    result: &mut [f32],
    groups: &[u32],
    indices: Option<&[u32]>,
    index_count: usize,
    face_normals: &[Normal],
    vertex_positions: &[f32],
    vertex_positions_stride: usize,
) {
    const NEXT: [usize; 4] = [1, 2, 0, 1];

    let pos_stride = vertex_positions_stride / 4;

    let face_count = index_count / 3;

    for i in 0..face_count {
        let n = &face_normals[i];

        for k in 0..3 {
            let base = groups[i * 3 + k] as usize * 3;

            let a = face_index(indices, i * 3 + k) as usize;
            let b = face_index(indices, i * 3 + NEXT[k]) as usize;
            let c = face_index(indices, i * 3 + NEXT[k + 1]) as usize;

            let pa = &vertex_positions[a * pos_stride..];
            let pb = &vertex_positions[b * pos_stride..];
            let pc = &vertex_positions[c * pos_stride..];

            // compute incident angle for weighting
            let dp1x = pb[0] - pa[0];
            let dp1y = pb[1] - pa[1];
            let dp1z = pb[2] - pa[2];
            let dp2x = pc[0] - pa[0];
            let dp2y = pc[1] - pa[1];
            let dp2z = pc[2] - pa[2];

            let dp1l = dp1x * dp1x + dp1y * dp1y + dp1z * dp1z;
            let dp2l = dp2x * dp2x + dp2y * dp2y + dp2z * dp2z;
            let dpl = math::sqrt(dp1l * dp2l);

            let cosa = (dp1x * dp2x + dp1y * dp2y + dp1z * dp2z)
                * (if dpl == 0.0 { 0.0 } else { 1.0 / dpl });
            let angle = optacos(cosa); // optacos handles clamping to [-1..1]

            // accumulate normal weighted by angle and edge length product; this weighting scheme matches the default scheme used for tangents
            let w = angle * dpl;

            result[base] += n.x * w;
            result[base + 1] += n.y * w;
            result[base + 2] += n.z * w;
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn smooth_normals(
    result: &mut [f32],
    scratch: &mut [f32],
    groups: &[u32],
    index_count: usize,
    alpha: f32,
) {
    const NEXT: [usize; 4] = [1, 2, 0, 1];

    let face_count = index_count / 3;

    scratch[..index_count * 4].fill(0.0);

    // for each triangle, accumulate normal deltas alongside each edge in both directions
    for i in 0..face_count {
        for k in 0..3 {
            let ga = groups[i * 3 + k] as usize;
            let gb = groups[i * 3 + NEXT[k]] as usize;

            let na = ga * 3;
            let nb = gb * 3;

            // normal alignment is symmetric; we compute it once per edge and use dp^2 for stronger alignment
            let dp = result[na] * result[nb]
                + result[na + 1] * result[nb + 1]
                + result[na + 2] * result[nb + 2];
            let w = if dp > 0.0 { dp * dp } else { 0.0 };

            let nx = (result[nb] - result[na]) * w;
            let ny = (result[nb + 1] - result[na + 1]) * w;
            let nz = (result[nb + 2] - result[na + 2]) * w;

            // accumulate deltas; the last component is used to compute delta averages below
            let sa = ga * 4;
            let sb = gb * 4;

            scratch[sa] += nx;
            scratch[sa + 1] += ny;
            scratch[sa + 2] += nz;
            scratch[sa + 3] += 1.0;

            scratch[sb] -= nx;
            scratch[sb + 1] -= ny;
            scratch[sb + 2] -= nz;
            scratch[sb + 3] += 1.0;
        }
    }

    // apply average delta to group roots and renormalize the results
    for i in 0..index_count {
        if groups[i] as usize == i && scratch[i * 4 + 3] > 0.0 {
            let r = i * 3;
            let s = i * 4;

            let sw = alpha / scratch[s + 3];
            let rx = result[r] + scratch[s] * sw;
            let ry = result[r + 1] + scratch[s + 1] * sw;
            let rz = result[r + 2] + scratch[s + 2] * sw;

            let rl = math::sqrt(rx * rx + ry * ry + rz * rz);
            let rs = if rl == 0.0 { 0.0 } else { 1.0 / rl };

            result[r] = rx * rs;
            result[r + 1] = ry * rs;
            result[r + 2] = rz * rs;
        }
    }
}

// port-shape: mirrors the upstream parameter list (ADR-0005)
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
#[allow(clippy::approx_constant)] // upstream's literal 3.1415927 bound in the assert
pub fn generate_normals(
    result: &mut [f32],
    indices: Option<&[u32]>,
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    crease_angle: f32,
    smoothing: f32,
) {
    debug_assert!(indices.is_some() || index_count == vertex_count);
    debug_assert!(index_count.is_multiple_of(3));
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
    debug_assert!((0.0..=3.1415927).contains(&crease_angle));
    debug_assert!(smoothing >= 0.0);
    debug_assert!(result.len() >= index_count * 3);

    let face_count = index_count / 3;
    let crease_cutoff = math::cos(crease_angle);

    // compute vertex remap to unique vertex index
    let mut remap = vec![0u32; vertex_count];
    build_vertex_remap::<true>(
        &mut remap,
        vertex_positions,
        vertex_positions_stride,
        &[],
        0,
        &[],
        0,
    );

    // build adjacency information
    let adjacency = build_corner_adjacency(indices, index_count, &remap, vertex_count);

    // compute per-triangle normals
    let mut face_normals = vec![
        Normal {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            id: 0
        };
        face_count
    ];
    compute_face_normals(
        &mut face_normals,
        indices,
        vertex_positions,
        vertex_positions_stride,
    );

    // compute per-corner normal groups: triangles adjacent to the same vertex with similar normals are merged in strips
    let mut groups: Vec<u32> = (0..index_count as u32).collect();

    for i in 0..vertex_count {
        if adjacency.offsets[i + 1] != adjacency.offsets[i] {
            merge_normal_groups(
                &mut groups,
                &adjacency.data[adjacency.offsets[i] as usize..adjacency.offsets[i + 1] as usize],
                indices,
                &remap,
                &face_normals,
                crease_cutoff,
            );
        }
    }

    for i in 0..index_count {
        let root = follow2(&mut groups, i as u32);
        groups[i] = root;
    }

    // accumulate normals into their own respective groups
    result[..index_count * 3].fill(0.0);
    accumulate_normals(
        result,
        &groups,
        indices,
        index_count,
        &face_normals,
        vertex_positions,
        vertex_positions_stride,
    );

    // normalize accumulated normals; we need to do this before smoothing as smoothing expects normalized inputs
    for i in 0..index_count {
        if groups[i] as usize == i {
            let r = i * 3;
            let l = math::sqrt(
                result[r] * result[r]
                    + result[r + 1] * result[r + 1]
                    + result[r + 2] * result[r + 2],
            );
            let s = if l == 0.0 { 0.0 } else { 1.0 / l };

            result[r] *= s;
            result[r + 1] *= s;
            result[r + 2] *= s;
        }
    }

    if smoothing > 0.0 {
        let mut passes = cast_i32(math::ceil(smoothing));
        passes = if passes < 10 { passes } else { 10 }; // clamp to avoid pathological cases

        let mut scratch = vec![0.0f32; index_count * 4];

        for pass in 0..passes {
            let rem = smoothing - pass as f32;
            let alpha = 0.5 * (if rem < 1.0 { rem } else { 1.0 });

            smooth_normals(result, &mut scratch, &groups, index_count, alpha);
        }
    }

    // finalize normal vectors by propagating the rest
    for i in 0..index_count {
        if groups[i] as usize != i {
            let g = groups[i] as usize * 3;
            result[i * 3] = result[g];
            result[i * 3 + 1] = result[g + 1];
            result[i * 3 + 2] = result[g + 2];
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
    fn golden_generate_normals() {
        const _LEN_CHECK: usize = 53332;
        let data = include_bytes!("../../tests/data/tan_normals.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let index_count = c.u32() as usize;
            let has_indices = c.u32();
            let pos_stride = c.u32() as usize;
            let crease_angle = c.f32();
            let smoothing = c.f32();
            let positions = c.f32s(vertex_count * pos_stride / 4);
            let indices = if has_indices == 1 {
                Some(c.vec_u32(index_count))
            } else {
                None
            };

            let expected = c.f32s(index_count * 3);

            let mut result = vec![0.0f32; index_count * 3];
            generate_normals(
                &mut result,
                indices.as_deref(),
                index_count,
                &positions,
                vertex_count,
                pos_stride,
                crease_angle,
                smoothing,
            );

            assert_eq!(
                result.as_slice(),
                expected.as_slice(),
                "generateNormals case {case} (crease {crease_angle}, smoothing {smoothing})"
            );
            case += 1;
        }
        assert_eq!(case, 8);
    }
}
