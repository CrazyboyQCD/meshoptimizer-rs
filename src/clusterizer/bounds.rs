//! Cluster/meshlet/sphere bounds (`meshletutils.cpp` entry points).
//!
//! Upstream counterpart: `meshopt_computeClusterBounds`, `meshopt_computeMeshletBounds`,
//! `meshopt_computeSphereBounds` and the shared `meshopt_Bounds` struct.

use crate::meshletutils::{
    MESHLET_MAX_TRIANGLES, compute_bounding_sphere, compute_cluster_bounds_core,
};

/// Upstream counterpart: `meshopt_Bounds`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub center: [f32; 3],
    pub radius: f32,
    pub cone_apex: [f32; 3],
    pub cone_axis: [f32; 3],
    pub cone_cutoff: f32,
    pub cone_axis_s8: [i8; 4],
    pub cone_cutoff_s8: i8,
}

/// Upstream counterpart: `meshopt_computeClusterBounds`.
pub fn compute_cluster_bounds(
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Bounds {
    debug_assert!(indices.len().is_multiple_of(3));
    debug_assert!(indices.len() / 3 <= MESHLET_MAX_TRIANGLES);
    debug_assert!(
        (12..=256).contains(&vertex_positions_stride) && vertex_positions_stride.is_multiple_of(4)
    );
    // upstream only uses vertex_count for asserts; index bounds are enforced by slice indexing
    let _ = vertex_count;

    let mut cache = [u32::MAX; 512];
    let mut corners = [0u32; MESHLET_MAX_TRIANGLES * 3 + 1]; // +1 for branchless slot
    let mut corner_count = 0usize;

    for &v in indices {
        let c = &mut cache[(v & 511) as usize];

        // branchless append if vertex isn't in cache
        corners[corner_count] = v;
        corner_count += usize::from(*c != v);
        *c = v;
    }

    compute_cluster_bounds_core(
        indices,
        &corners,
        corner_count,
        vertex_positions,
        vertex_positions_stride,
    )
}

/// Upstream counterpart: `meshopt_computeMeshletBounds`.
pub fn compute_meshlet_bounds(
    meshlet_vertices: &[u32],
    meshlet_triangles: &[u8],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Bounds {
    let triangle_count = meshlet_triangles.len() / 3;
    debug_assert!(triangle_count <= MESHLET_MAX_TRIANGLES);
    debug_assert!(
        (12..=256).contains(&vertex_positions_stride) && vertex_positions_stride.is_multiple_of(4)
    );
    let _ = vertex_count;

    let mut indices = [0u32; MESHLET_MAX_TRIANGLES * 3];
    let mut corner_count = 0usize;

    for (i, &t) in meshlet_triangles.iter().enumerate() {
        let index = meshlet_vertices[t as usize];
        indices[i] = index;

        // meshlet_vertices[] slice should only contain vertices used by triangle indices, which is the case for any well formed meshlet
        corner_count = if t as usize >= corner_count {
            t as usize + 1
        } else {
            corner_count
        };
    }

    compute_cluster_bounds_core(
        &indices,
        meshlet_vertices,
        corner_count,
        vertex_positions,
        vertex_positions_stride,
    )
}

/// Upstream counterpart: `meshopt_computeSphereBounds`.
pub fn compute_sphere_bounds(
    positions: &[f32],
    count: usize,
    positions_stride: usize,
    radii: Option<&[f32]>,
    radii_stride: usize,
) -> Bounds {
    debug_assert!((12..=256).contains(&positions_stride) && positions_stride.is_multiple_of(4));
    debug_assert!(
        radii.is_none() || ((4..=256).contains(&radii_stride) && radii_stride.is_multiple_of(4))
    );

    // degenerate cluster, no valid triangles => trivial reject (cone data is 0)
    if count == 0 {
        return Bounds {
            center: [0.0; 3],
            radius: 0.0,
            cone_apex: [0.0; 3],
            cone_axis: [0.0; 3],
            cone_cutoff: 0.0,
            cone_axis_s8: [0; 4],
            cone_cutoff_s8: 0,
        };
    }

    const RZERO: [f32; 1] = [0.0];

    let psphere = compute_bounding_sphere(
        count,
        positions,
        positions_stride,
        radii.unwrap_or(&RZERO),
        if radii.is_some() { radii_stride } else { 0 },
        7,
        None,
    );

    Bounds {
        center: [psphere[0], psphere[1], psphere[2]],
        radius: psphere[3],
        cone_apex: [0.0; 3],
        cone_axis: [0.0; 3],
        cone_cutoff: 0.0,
        cone_axis_s8: [0; 4],
        cone_cutoff_s8: 0,
    }
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

        fn bytes(&mut self, n: usize) -> Vec<u8> {
            let s = &self.data[self.pos..self.pos + n];
            self.pos += n;
            s.to_vec()
        }

        fn vec_u32(&mut self, n: usize) -> Vec<u32> {
            (0..n).map(|_| self.u32()).collect()
        }

        fn bounds(&mut self) -> Bounds {
            let mut center = [0.0f32; 3];
            for v in &mut center {
                *v = self.f32();
            }
            let radius = self.f32();
            let mut cone_apex = [0.0f32; 3];
            for v in &mut cone_apex {
                *v = self.f32();
            }
            let mut cone_axis = [0.0f32; 3];
            for v in &mut cone_axis {
                *v = self.f32();
            }
            let cone_cutoff = self.f32();
            let mut cone_axis_s8 = [0i8; 4];
            for v in &mut cone_axis_s8[..3] {
                *v = self.u32() as u8 as i8;
            }
            let cone_cutoff_s8 = self.u32() as u8 as i8;
            Bounds {
                center,
                radius,
                cone_apex,
                cone_axis,
                cone_cutoff,
                cone_axis_s8,
                cone_cutoff_s8,
            }
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
    fn golden_compute_cluster_bounds() {
        const _LEN_CHECK: usize = 15952;
        let data = include_bytes!("../../tests/data/cb_cluster.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let index_count = c.u32() as usize;
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let positions = c.f32s(vertex_count * stride);
            let indices = c.vec_u32(index_count);
            let expected = c.bounds();

            let out = compute_cluster_bounds(&indices, &positions, vertex_count, stride);
            assert_eq!(out, expected, "cluster bounds mismatch (ic={index_count})");
        }
    }

    #[test]
    fn golden_compute_meshlet_bounds() {
        const _LEN_CHECK: usize = 5246;
        let data = include_bytes!("../../tests/data/cb_meshlet.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let triangle_count = c.u32() as usize;
            let vertex_total = c.u32() as usize;
            let stride = c.u32() as usize;
            let meshlet_vertices = c.vec_u32(vertex_count);
            let meshlet_triangles = c.bytes(triangle_count * 3);
            let positions = c.f32s(vertex_total * stride);
            let expected = c.bounds();

            let out = compute_meshlet_bounds(
                &meshlet_vertices,
                &meshlet_triangles,
                &positions,
                vertex_total,
                stride,
            );
            assert_eq!(
                out, expected,
                "meshlet bounds mismatch (v={vertex_count}, t={triangle_count})"
            );
        }
    }

    #[test]
    fn golden_compute_sphere_bounds() {
        const _LEN_CHECK: usize = 24404;
        let data = include_bytes!("../../tests/data/clusterbounds_sphere.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let count = c.u32() as usize;
            let stride = c.u32() as usize;
            let has_radii = c.u32() == 1;
            let radii_stride = c.u32() as usize;
            let positions = c.f32s(count * stride);
            let no_radii = [0.0f32];
            let with_radii;
            let radii: &[f32] = if has_radii {
                with_radii = c.f32s(count * radii_stride);
                &with_radii
            } else {
                &no_radii
            };
            let mut center = [0.0f32; 3];
            center[0] = c.f32();
            center[1] = c.f32();
            center[2] = c.f32();
            let radius = c.f32();

            let radii_opt = if has_radii { Some(radii) } else { None };
            let out = compute_sphere_bounds(&positions, count, stride, radii_opt, radii_stride);
            assert_eq!(out.center, center, "sphere center mismatch (count {count})");
            assert_eq!(
                out.radius.to_bits(),
                radius.to_bits(),
                "sphere radius mismatch (count {count})"
            );
            assert_eq!(out.cone_apex, [0.0; 3]);
            assert_eq!(out.cone_axis, [0.0; 3]);
            assert_eq!(out.cone_cutoff, 0.0);
            assert_eq!(out.cone_axis_s8, [0; 4]);
            assert_eq!(out.cone_cutoff_s8, 0);
        }
    }
}
