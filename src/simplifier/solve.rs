//! Quadric-based position and attribute solving for `simplifyWithUpdate`.
//! Upstream: `simplifier.cpp` L1696-1830 (`solvePositions`/`solveAttributes`;
//! split out of `simplify` for file-size discipline, design §4).

use super::Vector3;
use super::adjacency::{EdgeAdjacency, KIND_COMPLEX, KIND_FRINGE, KIND_MANIFOLD};
use super::collapse::{get_neighborhood_radius, has_triangle_flips_simple};
use super::quadric::{
    Quadric, QuadricGrad, quadric_add, quadric_error, quadric_from_point,
    quadric_reduce_attributes, quadric_solve,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_positions(
    vertex_positions: &mut [Vector3],
    vertex_count: usize,
    vertex_quadrics: &[Quadric],
    volume_gradients: Option<&[QuadricGrad]>,
    attribute_quadrics: Option<&[Quadric]>,
    attribute_gradients: Option<&[QuadricGrad]>,
    attribute_count: usize,
    remap: &[u32],
    wedge: &[u32],
    adjacency: &EdgeAdjacency,
    vertex_kind: &[u8],
    vertex_update: &[u8],
) {
    for i in 0..vertex_count {
        if vertex_update[i] == 0 {
            continue;
        }

        // moving vertices on an attribute discontinuity may result in extrapolating UV outside of the chart bounds
        // moving vertices on a border requires a stronger edge quadric to preserve the border geometry
        if vertex_kind[i] != KIND_MANIFOLD && vertex_kind[i] != KIND_COMPLEX {
            continue;
        }

        if remap[i] != i as u32 {
            vertex_positions[i] = vertex_positions[remap[i] as usize];
            continue;
        }

        let vp = vertex_positions[i];

        let mut q = vertex_quadrics[i];
        let mut gv = QuadricGrad::ZERO;

        // add a point quadric for regularization to stabilize the solution
        let mut r = Quadric::ZERO;
        quadric_from_point(&mut r, vp.x, vp.y, vp.z, q.w * 1e-4);
        quadric_add(&mut q, &r);

        if attribute_count > 0 {
            if let (Some(aq), Some(ag)) = (attribute_quadrics, attribute_gradients) {
                // optimal point simultaneously minimizes attribute quadrics for all wedges
                let mut v = i as u32;
                loop {
                    quadric_reduce_attributes(
                        &mut q,
                        &aq[v as usize],
                        &ag[v as usize * attribute_count..],
                        attribute_count,
                    );

                    v = wedge[v as usize];
                    if v == i as u32 {
                        break;
                    }
                }
            }

            // minimizing attribute quadrics results in volume loss so we incorporate volume gradient as a constraint
            if let Some(vg) = volume_gradients {
                gv = vg[i];
            }
        }

        let mut p = Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        if !quadric_solve(&mut p, &q, &gv) {
            continue;
        }

        // reject updates that move the vertex too far from its neighborhood
        // this detects and fixes most cases when the quadric is not well-defined
        let nr = get_neighborhood_radius(adjacency, vertex_positions, i as u32);
        let dp =
            (p.x - vp.x) * (p.x - vp.x) + (p.y - vp.y) * (p.y - vp.y) + (p.z - vp.z) * (p.z - vp.z);

        if dp > nr * nr {
            continue;
        }

        // reject updates that would flip a neighboring triangle, as we do for edge collapse
        if has_triangle_flips_simple(adjacency, vertex_positions, i as u32, &p) {
            continue;
        }

        // reject updates that increase positional error too much; allow some tolerance to improve attribute quality
        if quadric_error(&vertex_quadrics[i], &p)
            > quadric_error(&vertex_quadrics[i], &vp) * 1.5 + 1e-6
        {
            continue;
        }

        vertex_positions[i] = p;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_attributes(
    vertex_positions: &[Vector3],
    vertex_attributes: &mut [f32],
    vertex_count: usize,
    attribute_quadrics: &[Quadric],
    attribute_gradients: &[QuadricGrad],
    attribute_count: usize,
    remap: &[u32],
    wedge: &[u32],
    vertex_kind: &[u8],
    vertex_update: &[u8],
) {
    for i in 0..vertex_count {
        if vertex_update[i] == 0 {
            continue;
        }

        if remap[i] != i as u32 {
            continue;
        }

        for k in 0..attribute_count {
            let mut shared = super::EMPTY_SLOT;

            // for complex vertices, preserve attribute continuity and use highest weight wedge if values were shared
            if vertex_kind[i] == KIND_COMPLEX || vertex_kind[i] == KIND_FRINGE {
                shared = i as u32;

                let mut v = wedge[i];
                while v != i as u32 {
                    if vertex_attributes[v as usize * attribute_count + k]
                        != vertex_attributes[i * attribute_count + k]
                    {
                        shared = super::EMPTY_SLOT;
                    } else if shared != super::EMPTY_SLOT
                        && attribute_quadrics[v as usize].w > attribute_quadrics[shared as usize].w
                    {
                        shared = v;
                    }

                    v = wedge[v as usize];
                }
            }

            // update attributes for all wedges
            let mut v = i as u32;
            loop {
                let r = if shared == super::EMPTY_SLOT {
                    v
                } else {
                    shared
                };

                let p = &vertex_positions[i]; // same for all wedges
                let a = &attribute_quadrics[r as usize];
                let g = attribute_gradients[r as usize * attribute_count + k];

                let iw = if a.w == 0.0 { 0.0 } else { 1.0 / a.w };
                let av = (g.gx * p.x + g.gy * p.y + g.gz * p.z + g.gw) * iw;

                vertex_attributes[v as usize * attribute_count + k] = av;

                v = wedge[v as usize];
                if v == i as u32 {
                    break;
                }
            }
        }
    }
}
