//! Internal meshlet helpers (`meshletutils.cpp`).
//!
//! Shared kernel for the cluster/meshlet/sphere bounds entry points (re-homed under
//! `clusterizer`, design.md §4) and the meshlet size limits.

use crate::clusterizer::Bounds;

/// Must be <= 256 since meshlet indices are stored as bytes (`kMeshletMaxVertices`).
pub(crate) const MESHLET_MAX_VERTICES: usize = 256;

/// A reasonable limit is around 2*max_vertices or less (`kMeshletMaxTriangles`).
pub(crate) const MESHLET_MAX_TRIANGLES: usize = 512;

/// Upstream counterpart: `meshopt::computeBoundingSphere` (meshletutils.cpp).
pub(crate) fn compute_bounding_sphere(
    count: usize,
    points: &[f32],
    points_stride: usize,
    radii: &[f32],
    radii_stride: usize,
    axis_count: usize,
    indices: Option<&[u32]>,
) -> [f32; 4] {
    const AXES: [[f32; 3]; 7] = [
        // X, Y, Z
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        // XYZ, -XYZ, X-YZ, XY-Z; normalized to unit length
        [0.57735026, 0.57735026, 0.57735026],
        [-0.57735026, 0.57735026, 0.57735026],
        [0.57735026, -0.57735026, 0.57735026],
        [0.57735026, 0.57735026, -0.57735026],
    ];

    // C asserts count > 0; a zero count is unreachable from valid callers, so we
    // return a degenerate sphere instead of reading out of bounds
    if count == 0 {
        return [0.0; 4];
    }

    let points_stride_float = points_stride / 4;
    let radii_stride_float = radii_stride / 4;

    // find extremum points along all axes; for each axis we get a pair of points with min/max coordinates
    let mut pmin = [0u32; 7];
    let mut pmax = [0u32; 7];
    let mut tmin = [f32::MAX; 7];
    let mut tmax = [-f32::MAX; 7];

    for i in 0..count {
        let v = match indices {
            Some(indices) => indices[i],
            None => i as u32,
        };
        let p = &points[v as usize * points_stride_float..];
        let r = radii[v as usize * radii_stride_float];

        for (axis, ax) in AXES.iter().enumerate().take(axis_count) {
            let tp = ax[0] * p[0] + ax[1] * p[1] + ax[2] * p[2];
            let tpmin = tp - r;
            let tpmax = tp + r;

            pmin[axis] = if tpmin < tmin[axis] { v } else { pmin[axis] };
            pmax[axis] = if tpmax > tmax[axis] { v } else { pmax[axis] };
            tmin[axis] = if tpmin < tmin[axis] {
                tpmin
            } else {
                tmin[axis]
            };
            tmax[axis] = if tpmax > tmax[axis] {
                tpmax
            } else {
                tmax[axis]
            };
        }
    }

    // find the pair of points with largest distance
    let mut paxis = 0;
    let mut paxisdr = 0.0f32;

    for axis in 0..axis_count {
        let p1 = &points[pmin[axis] as usize * points_stride_float..];
        let p2 = &points[pmax[axis] as usize * points_stride_float..];
        let r1 = radii[pmin[axis] as usize * radii_stride_float];
        let r2 = radii[pmax[axis] as usize * radii_stride_float];

        let d2 = (p2[0] - p1[0]) * (p2[0] - p1[0])
            + (p2[1] - p1[1]) * (p2[1] - p1[1])
            + (p2[2] - p1[2]) * (p2[2] - p1[2]);
        let dr = crate::math::sqrt(d2) + r1 + r2;

        if dr > paxisdr {
            paxisdr = dr;
            paxis = axis;
        }
    }

    // use the longest segment as the initial sphere diameter
    let p1 = &points[pmin[paxis] as usize * points_stride_float..];
    let p2 = &points[pmax[paxis] as usize * points_stride_float..];
    let r1 = radii[pmin[paxis] as usize * radii_stride_float];
    let r2 = radii[pmax[paxis] as usize * radii_stride_float];

    let paxisd = crate::math::sqrt(
        (p2[0] - p1[0]) * (p2[0] - p1[0])
            + (p2[1] - p1[1]) * (p2[1] - p1[1])
            + (p2[2] - p1[2]) * (p2[2] - p1[2]),
    );
    let paxisk = if paxisd > 0.0 {
        (paxisd + r2 - r1) / (2.0 * paxisd)
    } else {
        0.0
    };

    let mut center = [
        p1[0] + (p2[0] - p1[0]) * paxisk,
        p1[1] + (p2[1] - p1[1]) * paxisk,
        p1[2] + (p2[2] - p1[2]) * paxisk,
    ];
    let mut radius = paxisdr / 2.0;

    // iteratively adjust the sphere up until all points fit
    for i in 0..count {
        let v = match indices {
            Some(indices) => indices[i],
            None => i as u32,
        };
        let p = &points[v as usize * points_stride_float..];
        let r = radii[v as usize * radii_stride_float];

        let d2 = (p[0] - center[0]) * (p[0] - center[0])
            + (p[1] - center[1]) * (p[1] - center[1])
            + (p[2] - center[2]) * (p[2] - center[2]);
        let d = crate::math::sqrt(d2);

        if d + r > radius {
            let k = if d > 0.0 {
                (d + r - radius) / (2.0 * d)
            } else {
                0.0
            };

            center[0] += k * (p[0] - center[0]);
            center[1] += k * (p[1] - center[1]);
            center[2] += k * (p[2] - center[2]);
            radius = (radius + d + r) / 2.0;
        }
    }

    [center[0], center[1], center[2], radius]
}

/// Upstream counterpart: `meshopt::computeClusterBounds` (meshletutils.cpp, static).
///
/// `corners` holds the deduplicated corner vertex list; only the first
/// `corner_count` entries participate in the bounding sphere.
pub(crate) fn compute_cluster_bounds_core(
    indices: &[u32],
    corners: &[u32],
    corner_count: usize,
    vertex_positions: &[f32],
    vertex_positions_stride: usize,
) -> Bounds {
    let vertex_stride_float = vertex_positions_stride / 4;

    // compute triangle normals (.w completes plane equation)
    let mut normals = [[0.0f32; 4]; MESHLET_MAX_TRIANGLES];
    let mut triangles = 0usize;

    for &[a, b, c] in indices.as_chunks::<3>().0 {
        let (a, b, c) = (a as usize, b as usize, c as usize);

        let p0 = &vertex_positions[vertex_stride_float * a..];
        let p1 = &vertex_positions[vertex_stride_float * b..];
        let p2 = &vertex_positions[vertex_stride_float * c..];

        let p10 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let p20 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        let mut normalx = p10[1] * p20[2] - p10[2] * p20[1];
        let mut normaly = p10[2] * p20[0] - p10[0] * p20[2];
        let mut normalz = p10[0] * p20[1] - p10[1] * p20[0];

        let area = crate::math::sqrt(normalx * normalx + normaly * normaly + normalz * normalz);

        // no need to include degenerate triangles - they will be invisible anyway
        if area == 0.0 {
            continue;
        }

        normalx /= area;
        normaly /= area;
        normalz /= area;

        // record triangle normals; normal and corner 0 define a plane equation
        normals[triangles] = [
            normalx,
            normaly,
            normalz,
            -(normalx * p0[0] + normaly * p0[1] + normalz * p0[2]),
        ];
        triangles += 1;
    }

    let mut bounds = Bounds {
        center: [0.0; 3],
        radius: 0.0,
        cone_apex: [0.0; 3],
        cone_axis: [0.0; 3],
        cone_cutoff: 0.0,
        cone_axis_s8: [0; 4],
        cone_cutoff_s8: 0,
    };

    // degenerate cluster, no valid triangles => trivial reject (cone data is 0)
    if triangles == 0 {
        return bounds;
    }

    const RZERO: [f32; 1] = [0.0];

    // compute cluster bounding sphere; we'll use the center to determine normal cone apex as well
    let psphere = compute_bounding_sphere(
        corner_count,
        vertex_positions,
        vertex_positions_stride,
        &RZERO,
        0,
        7,
        Some(corners),
    );

    let center = [psphere[0], psphere[1], psphere[2]];

    // treating triangle normals as points, find the bounding sphere - the sphere center determines the optimal cone axis
    let nsphere =
        compute_bounding_sphere(triangles, normals.as_flattened(), 16, &RZERO, 0, 3, None);

    let mut axis = [nsphere[0], nsphere[1], nsphere[2]];
    let axislength = crate::math::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    let invaxislength = if axislength == 0.0 {
        0.0
    } else {
        1.0 / axislength
    };

    axis[0] *= invaxislength;
    axis[1] *= invaxislength;
    axis[2] *= invaxislength;

    // compute a tight cone around all normals, mindp = cos(angle/2)
    let mut mindp = 1.0f32;

    for n in &normals[..triangles] {
        let dp = n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2];

        mindp = if dp < mindp { dp } else { mindp };
    }

    // fill bounding sphere info; note that below we can return bounds without cone information for degenerate cones
    bounds.center = center;
    bounds.radius = psphere[3];

    // degenerate cluster, normal cone is larger than a hemisphere => trivial accept
    // note that if mindp is positive but close to 0, the triangle intersection code below gets less stable
    // we arbitrarily decide that if a normal cone is ~168 degrees wide or more, the cone isn't useful
    if mindp <= 0.1 {
        bounds.cone_cutoff = 1.0;
        bounds.cone_cutoff_s8 = 127;
        return bounds;
    }

    let mut maxt = 0.0f32;

    // we need to find the point on center-t*axis ray that lies in negative half-space of all triangles
    for n in &normals[..triangles] {
        // dot(center-t*axis-corner, trinormal) = 0
        // dot(center-corner, trinormal) - t * dot(axis, trinormal) = 0
        let dc = center[0] * n[0] + center[1] * n[1] + center[2] * n[2] + n[3];
        let dn = axis[0] * n[0] + axis[1] * n[1] + axis[2] * n[2];

        // dn should be larger than mindp cutoff above
        let t = dc / dn;

        maxt = if t > maxt { t } else { maxt };
    }

    // cone apex should be in the negative half-space of all cluster triangles by construction
    bounds.cone_apex = [
        center[0] - axis[0] * maxt,
        center[1] - axis[1] * maxt,
        center[2] - axis[2] * maxt,
    ];

    // note: this axis is the axis of the normal cone, but our test for perspective camera effectively negates the axis
    bounds.cone_axis = axis;

    // cos(a) for normal cone is mindp; we need to add 90 degrees on both sides and invert the cone
    // which gives us -cos(a+90) = -(-sin(a)) = sin(a) = sqrt(1 - cos^2(a))
    bounds.cone_cutoff = crate::math::sqrt(1.0 - mindp * mindp);

    // quantize axis & cutoff to 8-bit SNORM format
    bounds.cone_axis_s8[0] = crate::quantization::quantize_snorm(bounds.cone_axis[0], 8) as i8;
    bounds.cone_axis_s8[1] = crate::quantization::quantize_snorm(bounds.cone_axis[1], 8) as i8;
    bounds.cone_axis_s8[2] = crate::quantization::quantize_snorm(bounds.cone_axis[2], 8) as i8;

    // for the 8-bit test to be conservative, we need to adjust the cutoff by measuring the max. error
    let cone_axis_s8_e0 = (f32::from(bounds.cone_axis_s8[0]) / 127.0 - bounds.cone_axis[0]).abs();
    let cone_axis_s8_e1 = (f32::from(bounds.cone_axis_s8[1]) / 127.0 - bounds.cone_axis[1]).abs();
    let cone_axis_s8_e2 = (f32::from(bounds.cone_axis_s8[2]) / 127.0 - bounds.cone_axis[2]).abs();

    // note that we need to round this up instead of rounding to nearest, hence +1
    let cone_cutoff_s8 = crate::quantization::cast_i32(
        127.0 * (bounds.cone_cutoff + cone_axis_s8_e0 + cone_axis_s8_e1 + cone_axis_s8_e2) + 1.0,
    );

    bounds.cone_cutoff_s8 = if cone_cutoff_s8 > 127 {
        127
    } else {
        cone_cutoff_s8 as i8
    };

    bounds
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

        fn bytes(&mut self, n: usize) -> &'a [u8] {
            let s = &self.data[self.pos..self.pos + n];
            self.pos += n;
            s
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
    fn golden_compute_bounding_sphere() {
        const _LEN_CHECK: usize = 24404;
        let data = include_bytes!("../tests/data/clusterbounds_sphere.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let count = c.u32() as usize;
            let stride = c.u32() as usize;
            let has_radii = c.u32() == 1;
            let radii_stride = c.u32() as usize;
            let points = c.f32s(count * stride);
            let no_radii = [0.0f32];
            let with_radii;
            let radii: &[f32] = if has_radii {
                with_radii = c.f32s(count * radii_stride);
                &with_radii
            } else {
                &no_radii
            };
            let mut expected = [0.0f32; 4];
            expected[0] = c.f32();
            expected[1] = c.f32();
            expected[2] = c.f32();
            expected[3] = c.f32();

            let out = compute_bounding_sphere(count, &points, stride, radii, radii_stride, 7, None);
            for k in 0..4 {
                assert_eq!(
                    out[k].to_bits(),
                    expected[k].to_bits(),
                    "component {k} mismatch (count {count}, stride {stride}, radii {has_radii})"
                );
            }
        }
    }
}
