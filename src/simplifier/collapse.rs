//! Collapse selection, ranking, and execution machinery for the simplifier,
//! plus fold-line quadrics and post-pass index/loop remapping. Upstream:
//! `simplifier.cpp` L1146-1206, L1230-2039 (meshlet-adjacent sections of the
//! same TU; split out of `simplify` for file-size discipline, design §4).

use super::adjacency::{
    EdgeAdjacency, K_CAN_COLLAPSE, K_HAS_OPPOSITE, KIND_BORDER, KIND_COMPLEX, KIND_FRINGE,
    KIND_MANIFOLD, KIND_SEAM,
};
use super::quadric::{
    K_MAX_ATTRIBUTES, Quadric, QuadricGrad, quadric_add, quadric_add_grad, quadric_add_grad_n,
    quadric_error, quadric_error_attr, quadric_from_triangle_edge,
};
use super::{EMPTY_SLOT, Vector3};
use crate::math;

#[derive(Clone, Copy)]
pub(crate) struct Collapse {
    pub(crate) v0: u32,
    pub(crate) v1: u32,
    pub(crate) bidi: u32,
    // upstream stores error/bidi in a union; bidi is only read during ranking,
    // before error is written, so separate fields preserve the semantics
    pub(crate) error: f32,
}

// bundled attribute data for the ranking/solving paths (all `None` upstream)
pub(crate) struct AttributeData<'a> {
    pub(crate) attributes: &'a [f32],
    pub(crate) quadrics: &'a [Quadric],
    pub(crate) gradients: &'a [QuadricGrad],
}

fn has_triangle_flip(a: &Vector3, b: &Vector3, c: &Vector3, d: &Vector3) -> bool {
    let eb = Vector3 {
        x: b.x - a.x,
        y: b.y - a.y,
        z: b.z - a.z,
    };
    let ec = Vector3 {
        x: c.x - a.x,
        y: c.y - a.y,
        z: c.z - a.z,
    };
    let ed = Vector3 {
        x: d.x - a.x,
        y: d.y - a.y,
        z: d.z - a.z,
    };

    let nbc = Vector3 {
        x: eb.y * ec.z - eb.z * ec.y,
        y: eb.z * ec.x - eb.x * ec.z,
        z: eb.x * ec.y - eb.y * ec.x,
    };
    let nbd = Vector3 {
        x: eb.y * ed.z - eb.z * ed.y,
        y: eb.z * ed.x - eb.x * ed.z,
        z: eb.x * ed.y - eb.y * ed.x,
    };

    let ndp = nbc.x * nbd.x + nbc.y * nbd.y + nbc.z * nbd.z;
    let abc = nbc.x * nbc.x + nbc.y * nbc.y + nbc.z * nbc.z;
    let abd = nbd.x * nbd.x + nbd.y * nbd.y + nbd.z * nbd.z;

    // scale is cos(angle); somewhat arbitrarily set to ~75 degrees
    // note that the "pure" check is ndp <= 0 (90 degree cutoff) but that allows flipping through a series of close-to-90 collapses
    ndp <= 0.25 * math::sqrt(abc * abd)
}

pub(crate) fn has_triangle_flips(
    adjacency: &EdgeAdjacency,
    vertex_positions: &[Vector3],
    collapse_remap: &[u32],
    i0: u32,
    i1: u32,
) -> bool {
    debug_assert!(collapse_remap[i0 as usize] == i0);
    debug_assert!(collapse_remap[i1 as usize] == i1);

    let v0 = &vertex_positions[i0 as usize];
    let v1 = &vertex_positions[i1 as usize];

    let start = adjacency.offsets[i0 as usize] as usize;
    let count = (adjacency.offsets[i0 as usize + 1] - adjacency.offsets[i0 as usize]) as usize;

    for e in &adjacency.data[start..start + count] {
        let a = collapse_remap[e.next as usize];
        let b = collapse_remap[e.prev as usize];

        // skip triangles that will get collapsed by i0->i1 collapse or already got collapsed previously
        if a == i1 || b == i1 || a == b {
            continue;
        }

        // early-out when at least one triangle flips due to a collapse
        if has_triangle_flip(
            &vertex_positions[a as usize],
            &vertex_positions[b as usize],
            v0,
            v1,
        ) {
            return true;
        }
    }

    false
}

pub(crate) fn has_triangle_flips_simple(
    adjacency: &EdgeAdjacency,
    vertex_positions: &[Vector3],
    i0: u32,
    v1: &Vector3,
) -> bool {
    let v0 = &vertex_positions[i0 as usize];

    let start = adjacency.offsets[i0 as usize] as usize;
    let count = (adjacency.offsets[i0 as usize + 1] - adjacency.offsets[i0 as usize]) as usize;

    for e in &adjacency.data[start..start + count] {
        let a = e.next;
        let b = e.prev;

        if has_triangle_flip(
            &vertex_positions[a as usize],
            &vertex_positions[b as usize],
            v0,
            v1,
        ) {
            return true;
        }
    }

    false
}

pub(crate) fn get_neighborhood_radius(
    adjacency: &EdgeAdjacency,
    vertex_positions: &[Vector3],
    i0: u32,
) -> f32 {
    let v0 = &vertex_positions[i0 as usize];

    let start = adjacency.offsets[i0 as usize] as usize;
    let count = (adjacency.offsets[i0 as usize + 1] - adjacency.offsets[i0 as usize]) as usize;

    let mut result = 0.0f32;

    for e in &adjacency.data[start..start + count] {
        let va = &vertex_positions[e.next as usize];
        let vb = &vertex_positions[e.prev as usize];

        let da = (va.x - v0.x) * (va.x - v0.x)
            + (va.y - v0.y) * (va.y - v0.y)
            + (va.z - v0.z) * (va.z - v0.z);
        let db = (vb.x - v0.x) * (vb.x - v0.x)
            + (vb.y - v0.y) * (vb.y - v0.y)
            + (vb.z - v0.z) * (vb.z - v0.z);

        result = if result < da { da } else { result };
        result = if result < db { db } else { result };
    }

    math::sqrt(result)
}

fn get_complex_target(
    v: u32,
    target: u32,
    remap: &[u32],
    edge_loop: &[u32],
    loopback: &[u32],
) -> u32 {
    let r = remap[target as usize];

    // use loop metadata to guide complex collapses towards the correct wedge
    // this works for edges on attribute discontinuities because loop/loopback track the single half-edge without a pair, similar to seams
    if edge_loop[v as usize] != EMPTY_SLOT && remap[edge_loop[v as usize] as usize] == r {
        edge_loop[v as usize]
    } else if loopback[v as usize] != EMPTY_SLOT && remap[loopback[v as usize] as usize] == r {
        loopback[v as usize]
    } else {
        target
    }
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn bound_edge_collapses(
    adjacency: &EdgeAdjacency,
    vertex_count: usize,
    index_count: usize,
    vertex_kind: &[u8],
) -> usize {
    let mut dual_count = 0usize;

    for i in 0..vertex_count {
        let k = vertex_kind[i];
        let e = (adjacency.offsets[i + 1] - adjacency.offsets[i]) as usize;

        dual_count += if k == KIND_MANIFOLD || k == KIND_SEAM {
            e
        } else {
            0
        };
    }

    debug_assert!(dual_count <= index_count);

    // pad capacity by 3 so that we can check for overflow once per triangle instead of once per edge
    (index_count - dual_count / 2) + 3
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn pick_edge_collapses(
    collapses: &mut [Collapse],
    indices: &[u32],
    index_count: usize,
    remap: &[u32],
    vertex_kind: &[u8],
    edge_loop: &[u32],
    loopback: &[u32],
) -> usize {
    let collapse_capacity = collapses.len();
    let mut collapse_count = 0usize;

    const NEXT: [usize; 3] = [1, 2, 0];

    for i in (0..index_count).step_by(3) {
        // this should never happen as boundEdgeCollapses should give an upper bound for the collapse count, but in an unlikely event it does we can just drop extra collapses
        if collapse_count + 3 > collapse_capacity {
            break;
        }

        for e in 0..3 {
            let i0 = indices[i + e];
            let i1 = indices[i + NEXT[e]];

            // this can happen either when input has a zero-length edge, or when we perform collapses for complex
            // topology w/seams and collapse a manifold vertex that connects to both wedges onto one of them
            // we leave edges like this alone since they may be important for preserving mesh integrity
            if remap[i0 as usize] == remap[i1 as usize] {
                continue;
            }

            let k0 = vertex_kind[i0 as usize] as usize;
            let k1 = vertex_kind[i1 as usize] as usize;

            // the edge has to be collapsible in at least one direction
            if K_CAN_COLLAPSE[k0][k1] | K_CAN_COLLAPSE[k1][k0] == 0 {
                continue;
            }

            // manifold and seam edges should occur twice (i0->i1 and i1->i0) - skip redundant edges
            if K_HAS_OPPOSITE[k0][k1] != 0 && remap[i1 as usize] > remap[i0 as usize] {
                continue;
            }

            // two vertices are on a border or a seam, but there's no direct edge between them
            // this indicates that they belong to two different edge loops and we should not collapse this edge
            // loop[] and loopback[] track half edges so we only need to check one of them
            if (vertex_kind[i0 as usize] == KIND_BORDER || vertex_kind[i0 as usize] == KIND_SEAM)
                && vertex_kind[i1 as usize] != KIND_MANIFOLD
                && edge_loop[i0 as usize] != i1
            {
                continue;
            }
            if (vertex_kind[i1 as usize] == KIND_BORDER || vertex_kind[i1 as usize] == KIND_SEAM)
                && vertex_kind[i0 as usize] != KIND_MANIFOLD
                && loopback[i1 as usize] != i0
            {
                continue;
            }

            // edge can be collapsed in either direction - we will pick the one with minimum error
            // note: we evaluate error later during collapse ranking, here we just tag the edge as bidirectional
            if K_CAN_COLLAPSE[k0][k1] & K_CAN_COLLAPSE[k1][k0] != 0 {
                collapses[collapse_count] = Collapse {
                    v0: i0,
                    v1: i1,
                    bidi: 1,
                    error: 0.0,
                };
                collapse_count += 1;
            } else {
                // edge can only be collapsed in one direction
                let (e0, e1) = if K_CAN_COLLAPSE[k0][k1] != 0 {
                    (i0, i1)
                } else {
                    (i1, i0)
                };

                collapses[collapse_count] = Collapse {
                    v0: e0,
                    v1: e1,
                    bidi: 0,
                    error: 0.0,
                };
                collapse_count += 1;
            }
        }
    }

    collapse_count
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn rank_edge_collapses(
    collapses: &mut [Collapse],
    collapse_count: usize,
    vertex_positions: &[Vector3],
    attribute_data: Option<&AttributeData<'_>>,
    attribute_count: usize,
    vertex_quadrics: &[Quadric],
    remap: &[u32],
    wedge: &[u32],
    vertex_kind: &[u8],
    edge_loop: &[u32],
    loopback: &[u32],
    clamped: bool,
) {
    for i in 0..collapse_count {
        let c = collapses[i];

        let i0 = c.v0;
        let i1 = c.v1;
        let bidi = c.bidi != 0;

        let mut ei = quadric_error(
            &vertex_quadrics[remap[i0 as usize] as usize],
            &vertex_positions[i1 as usize],
        );
        let mut ej = if bidi {
            quadric_error(
                &vertex_quadrics[remap[i1 as usize] as usize],
                &vertex_positions[i0 as usize],
            )
        } else {
            f32::MAX
        };

        if let Some(attrs) = attribute_data {
            ei += quadric_error_attr(
                &attrs.quadrics[i0 as usize],
                &attrs.gradients[i0 as usize * attribute_count..],
                attribute_count,
                &vertex_positions[i1 as usize],
                &attrs.attributes[i1 as usize * attribute_count..],
                clamped,
            );
            ej += if bidi {
                quadric_error_attr(
                    &attrs.quadrics[i1 as usize],
                    &attrs.gradients[i1 as usize * attribute_count..],
                    attribute_count,
                    &vertex_positions[i0 as usize],
                    &attrs.attributes[i0 as usize * attribute_count..],
                    clamped,
                )
            } else {
                0.0
            };

            // seam edges need to aggregate attribute errors between primary and secondary edges, as attribute quadrics are separate
            if vertex_kind[i0 as usize] == KIND_SEAM {
                // for seam collapses we need to find the seam pair; this is a bit tricky since we need to rely on edge loops as target vertex may be locked (and thus have more than two wedges)
                let s0 = wedge[i0 as usize];
                let mut s1 = if edge_loop[i0 as usize] == i1 {
                    loopback[s0 as usize]
                } else {
                    edge_loop[s0 as usize]
                };

                debug_assert!(wedge[s0 as usize] == i0); // s0 may be equal to i0 for half-seams
                debug_assert!(s1 != EMPTY_SLOT && remap[s1 as usize] == remap[i1 as usize]);

                // note: this should never happen due to the assertion above, but when disabled if we ever hit this case we'll get a memory safety issue; for now play it safe
                s1 = if s1 != EMPTY_SLOT {
                    s1
                } else {
                    wedge[i1 as usize]
                };

                ei += quadric_error_attr(
                    &attrs.quadrics[s0 as usize],
                    &attrs.gradients[s0 as usize * attribute_count..],
                    attribute_count,
                    &vertex_positions[s1 as usize],
                    &attrs.attributes[s1 as usize * attribute_count..],
                    clamped,
                );
                ej += if bidi {
                    quadric_error_attr(
                        &attrs.quadrics[s1 as usize],
                        &attrs.gradients[s1 as usize * attribute_count..],
                        attribute_count,
                        &vertex_positions[s0 as usize],
                        &attrs.attributes[s0 as usize * attribute_count..],
                        clamped,
                    )
                } else {
                    0.0
                };
            } else {
                // complex edges can have multiple wedges, so we need to aggregate errors for all wedges based on the selected target
                if vertex_kind[i0 as usize] == KIND_COMPLEX
                    || vertex_kind[i0 as usize] == KIND_FRINGE
                {
                    let mut v = wedge[i0 as usize];
                    while v != i0 {
                        let t = get_complex_target(v, i1, remap, edge_loop, loopback);

                        ei += quadric_error_attr(
                            &attrs.quadrics[v as usize],
                            &attrs.gradients[v as usize * attribute_count..],
                            attribute_count,
                            &vertex_positions[t as usize],
                            &attrs.attributes[t as usize * attribute_count..],
                            clamped,
                        );

                        v = wedge[v as usize];
                    }
                }

                if (vertex_kind[i1 as usize] == KIND_COMPLEX
                    || vertex_kind[i1 as usize] == KIND_FRINGE)
                    && bidi
                {
                    let mut v = wedge[i1 as usize];
                    while v != i1 {
                        let t = get_complex_target(v, i0, remap, edge_loop, loopback);

                        ej += quadric_error_attr(
                            &attrs.quadrics[v as usize],
                            &attrs.gradients[v as usize * attribute_count..],
                            attribute_count,
                            &vertex_positions[t as usize],
                            &attrs.attributes[t as usize * attribute_count..],
                            clamped,
                        );

                        v = wedge[v as usize];
                    }
                }
            }
        }

        // pick edge direction with minimal error (branchless)
        let rev = bidi && ej < ei;

        collapses[i].v0 = if rev { i1 } else { i0 };
        collapses[i].v1 = if rev { i0 } else { i1 };
        collapses[i].error = if ej < ei { ej } else { ei };
    }
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn sort_edge_collapses(
    sort_order: &mut [u32],
    collapses: &[Collapse],
    collapse_count: usize,
) {
    // we use counting sort to order collapses by error; since the exact sort order is not as critical,
    // only top 12 bits of exponent+mantissa (8 bits of exponent and 4 bits of mantissa) are used.
    // to avoid excessive stack usage, we clamp the exponent range as collapses with errors much higher than 1 are not useful.
    const SORT_BITS: u32 = 12;
    const SORT_BINS: usize = 2048 + 512; // exponent range [-127, 32)

    // fill histogram for counting sort
    let mut histogram = [0usize; SORT_BINS];

    for i in 0..collapse_count {
        // skip sign bit since error is non-negative
        let error = collapses[i].error.to_bits();
        let mut key = ((error << 1) >> (32 - SORT_BITS)) as usize;
        key = if key < SORT_BINS { key } else { SORT_BINS - 1 };

        histogram[key] += 1;
    }

    // compute offsets based on histogram data
    let mut histogram_sum = 0usize;

    for h in histogram.iter_mut().take(SORT_BINS) {
        let count = *h;
        *h = histogram_sum;
        histogram_sum += count;
    }

    debug_assert_eq!(histogram_sum, collapse_count);

    // compute sort order based on offsets
    for i in 0..collapse_count {
        // skip sign bit since error is non-negative
        let error = collapses[i].error.to_bits();
        let mut key = ((error << 1) >> (32 - SORT_BITS)) as usize;
        key = if key < SORT_BINS { key } else { SORT_BINS - 1 };

        sort_order[histogram[key]] = i as u32;
        histogram[key] += 1;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn perform_edge_collapses(
    collapse_remap: &mut [u32],
    collapse_locked: &mut [u8],
    collapses: &[Collapse],
    collapse_count: usize,
    collapse_order: &[u32],
    remap: &[u32],
    wedge: &[u32],
    vertex_kind: &[u8],
    edge_loop: &[u32],
    loopback: &[u32],
    vertex_positions: &[Vector3],
    adjacency: &EdgeAdjacency,
    triangle_collapse_goal: usize,
    error_limit: f32,
    result_error: &mut f32,
) -> usize {
    let mut edge_collapses = 0usize;
    let mut triangle_collapses = 0usize;

    // most collapses remove 2 triangles; use this to establish a bound on the pass in terms of error limit
    // note that edge_collapse_goal is an estimate; triangle_collapse_goal will be used to actually limit collapses
    let mut edge_collapse_goal = triangle_collapse_goal / 2;

    for i in 0..collapse_count {
        let c = collapses[collapse_order[i] as usize];

        if c.error > error_limit {
            break;
        }

        if triangle_collapses >= triangle_collapse_goal {
            break;
        }

        // we limit the error in each pass based on the error of optimal last collapse; since many collapses will be locked
        // as they will share vertices with other successful collapses, we need to increase the acceptable error by some factor
        let error_goal = if edge_collapse_goal < collapse_count {
            1.5 * collapses[collapse_order[edge_collapse_goal] as usize].error
        } else {
            f32::MAX
        };

        // on average, each collapse is expected to lock 6 other collapses; to avoid degenerate passes on meshes with odd
        // topology, we only abort if we got over 1/6 collapses accordingly.
        if c.error > error_goal
            && c.error > *result_error
            && triangle_collapses > triangle_collapse_goal / 6
        {
            break;
        }

        let i0 = c.v0;
        let i1 = c.v1;

        let r0 = remap[i0 as usize];
        let r1 = remap[i1 as usize];

        let kind = vertex_kind[i0 as usize];

        // we don't collapse vertices that had source or target vertex involved in a collapse
        // it's important to not move the vertices twice since it complicates the tracking/remapping logic
        // it's important to not move other vertices towards a moved vertex to preserve error since we don't re-rank collapses mid-pass
        if collapse_locked[r0 as usize] | collapse_locked[r1 as usize] != 0 {
            continue;
        }

        if has_triangle_flips(adjacency, vertex_positions, collapse_remap, r0, r1) {
            // adjust collapse goal since this collapse is invalid and shouldn't factor into error goal
            edge_collapse_goal += 1;

            continue;
        }

        debug_assert!(collapse_remap[r0 as usize] == r0);
        debug_assert!(collapse_remap[r1 as usize] == r1);

        if kind == KIND_COMPLEX || kind == KIND_FRINGE {
            collapse_remap[i0 as usize] = i1;

            // remap all vertices in the complex to the target vertex, using the same ranking that we used to evaluate the collapses
            let mut v = wedge[i0 as usize];
            while v != i0 {
                collapse_remap[v as usize] = get_complex_target(v, i1, remap, edge_loop, loopback);

                v = wedge[v as usize];
            }
        } else if kind == KIND_SEAM {
            // for seam collapses we need to move the seam pair together; this is a bit tricky since we need to rely on edge loops as target vertex may be locked (and thus have more than two wedges)
            let s0 = wedge[i0 as usize];
            let mut s1 = if edge_loop[i0 as usize] == i1 {
                loopback[s0 as usize]
            } else {
                edge_loop[s0 as usize]
            };
            debug_assert!(wedge[s0 as usize] == i0); // s0 may be equal to i0 for half-seams
            debug_assert!(s1 != EMPTY_SLOT && remap[s1 as usize] == r1);

            // note: this should never happen due to the assertion above, but when disabled if we ever hit this case we'll get a memory safety issue; for now play it safe
            s1 = if s1 != EMPTY_SLOT {
                s1
            } else {
                wedge[i1 as usize]
            };

            collapse_remap[i0 as usize] = i1;
            collapse_remap[s0 as usize] = s1;
        } else {
            debug_assert!(wedge[i0 as usize] == i0);

            collapse_remap[i0 as usize] = i1;
        }

        // note: we technically don't need to lock r1 if it's a locked vertex, as it can't move and its quadric won't be used
        // however, this results in slightly worse error on some meshes because the locked collapses get an unfair advantage wrt scheduling
        collapse_locked[r0 as usize] = 1;
        collapse_locked[r1 as usize] = 1;

        // border edges collapse 1 triangle, other edges collapse 2 or more
        // fringe edges often collapse 1 but may also collapse 2 because they don't respect loops, so we conservatively assume 2
        triangle_collapses += if kind == KIND_BORDER { 1 } else { 2 };
        edge_collapses += 1;

        *result_error = if *result_error < c.error {
            c.error
        } else {
            *result_error
        };
    }

    edge_collapses
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn update_quadrics(
    collapse_remap: &[u32],
    vertex_count: usize,
    vertex_quadrics: &mut [Quadric],
    mut volume_gradients: Option<&mut [QuadricGrad]>,
    mut attribute_quadrics: Option<&mut [Quadric]>,
    mut attribute_gradients: Option<&mut [QuadricGrad]>,
    attribute_count: usize,
    vertex_positions: &[Vector3],
    remap: &[u32],
    vertex_error: &mut f32,
) {
    for i in 0..vertex_count {
        if collapse_remap[i] == i as u32 {
            continue;
        }

        let i0 = i as u32;
        let i1 = collapse_remap[i];

        let r0 = remap[i0 as usize];
        let r1 = remap[i1 as usize];

        // ensure we only update vertex_quadrics once: primary vertex must be moved if any wedge is moved
        if i0 == r0 {
            let src = vertex_quadrics[r0 as usize];
            quadric_add(&mut vertex_quadrics[r1 as usize], &src);

            if let Some(vg) = volume_gradients.as_mut() {
                let src = vg[r0 as usize];
                quadric_add_grad(&mut vg[r1 as usize], &src);
            }
        }

        if let (Some(aq), Some(ag)) = (attribute_quadrics.as_mut(), attribute_gradients.as_mut()) {
            let src = aq[i0 as usize];
            quadric_add(&mut aq[i1 as usize], &src);

            let mut src = [QuadricGrad::ZERO; K_MAX_ATTRIBUTES];
            src[..attribute_count]
                .copy_from_slice(&ag[i0 as usize * attribute_count..][..attribute_count]);
            quadric_add_grad_n(
                &mut ag[i1 as usize * attribute_count..],
                &src,
                attribute_count,
            );

            if i0 == r0 {
                // when attributes are used, distance error needs to be recomputed as collapses don't track it; it is safe to do this after the quadric adjustment
                let derr = quadric_error(
                    &vertex_quadrics[r0 as usize],
                    &vertex_positions[r1 as usize],
                );
                *vertex_error = if *vertex_error < derr {
                    derr
                } else {
                    *vertex_error
                };
            }
        }
    }
}

pub(crate) fn remap_index_buffer(
    indices: &mut [u32],
    index_count: usize,
    collapse_remap: &[u32],
    remap: &[u32],
) -> usize {
    let mut write = 0usize;

    for i in (0..index_count).step_by(3) {
        let v0 = collapse_remap[indices[i] as usize];
        let v1 = collapse_remap[indices[i + 1] as usize];
        let v2 = collapse_remap[indices[i + 2] as usize];

        // we never move the vertex twice during a single pass
        debug_assert!(collapse_remap[v0 as usize] == v0);
        debug_assert!(collapse_remap[v1 as usize] == v1);
        debug_assert!(collapse_remap[v2 as usize] == v2);

        // collapse zero area triangles even if they are not topologically degenerate
        // this is required to cleanup manifold->seam collapses when a vertex is collapsed onto a seam pair
        // as well as complex collapses and some other cases where cross wedge collapses are performed
        let r0 = remap[v0 as usize];
        let r1 = remap[v1 as usize];
        let r2 = remap[v2 as usize];

        if r0 != r1 && r0 != r2 && r1 != r2 {
            indices[write] = v0;
            indices[write + 1] = v1;
            indices[write + 2] = v2;
            write += 3;
        }
    }

    write
}

pub(crate) fn remap_edge_loops(edge_loop: &mut [u32], vertex_count: usize, collapse_remap: &[u32]) {
    for i in 0..vertex_count {
        // note: this is a no-op for vertices that were remapped
        // ideally we would clear the loop entries for those for consistency, even though they aren't going to be used
        // however, the remapping process needs loop information for remapped vertices, so this would require a separate pass
        if edge_loop[i] != EMPTY_SLOT {
            let l = edge_loop[i];
            let r = collapse_remap[l as usize];

            // i == r is a special case when the seam edge is collapsed in a direction opposite to where loop goes
            if i as u32 == r {
                let ll = edge_loop[l as usize];
                edge_loop[i] = if ll != EMPTY_SLOT {
                    collapse_remap[ll as usize]
                } else {
                    EMPTY_SLOT
                };
            } else {
                edge_loop[i] = r;
            }
        }
    }
}

fn opposite_vertex(adjacency: &EdgeAdjacency, a: u32, b: u32, remap: &[u32], wedge: &[u32]) -> u32 {
    let mut v = a;
    let mut r = EMPTY_SLOT;

    loop {
        let count = (adjacency.offsets[v as usize + 1] - adjacency.offsets[v as usize]) as usize;
        let start = adjacency.offsets[v as usize] as usize;

        for e in &adjacency.data[start..start + count] {
            if remap[e.next as usize] == remap[b as usize] {
                r = if r == EMPTY_SLOT { e.prev } else { a };
            }
        }

        v = wedge[v as usize];
        if v == a {
            break;
        }
    }

    // we only return the opposite vertex if it's unique
    if r == a { EMPTY_SLOT } else { r }
}

pub(crate) fn fill_fold_quadrics(
    vertex_quadrics: &mut [Quadric],
    adjacency: &EdgeAdjacency,
    vertex_positions: &[Vector3],
    vertex_count: usize,
    remap: &[u32],
    wedge: &[u32],
) {
    for i0 in 0..vertex_count {
        for i in adjacency.offsets[i0]..adjacency.offsets[i0 + 1] {
            let i1 = adjacency.data[i as usize].next;
            let i2 = adjacency.data[i as usize].prev;

            // since we only process paired edges each edge should occur twice; skip redundant edges
            if remap[i1 as usize] > remap[i0] {
                continue;
            }

            let i3 = opposite_vertex(adjacency, i1, i0 as u32, remap, wedge);

            if i3 == EMPTY_SLOT {
                continue;
            }

            let p0 = vertex_positions[i0];
            let p1 = vertex_positions[i1 as usize];
            let p2 = vertex_positions[i2 as usize];
            let p3 = vertex_positions[i3 as usize];

            let p10 = Vector3 {
                x: p1.x - p0.x,
                y: p1.y - p0.y,
                z: p1.z - p0.z,
            };
            let p20 = Vector3 {
                x: p2.x - p0.x,
                y: p2.y - p0.y,
                z: p2.z - p0.z,
            };
            let p30 = Vector3 {
                x: p3.x - p0.x,
                y: p3.y - p0.y,
                z: p3.z - p0.z,
            };

            let normal = Vector3 {
                x: p10.y * p20.z - p10.z * p20.y,
                y: p10.z * p20.x - p10.x * p20.z,
                z: p10.x * p20.y - p10.y * p20.x,
            };
            let opposite = Vector3 {
                x: p30.y * p10.z - p30.z * p10.y,
                y: p30.z * p10.x - p30.x * p10.z,
                z: p30.x * p10.y - p30.y * p10.x,
            };
            let nl = math::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
            let ol = math::sqrt(
                opposite.x * opposite.x + opposite.y * opposite.y + opposite.z * opposite.z,
            );

            // identify folds by checking dihedral angle against a somewhat arbitrary ~30 degree cutoff
            if normal.x * opposite.x + normal.y * opposite.y + normal.z * opposite.z
                >= -0.85 * nl * ol
            {
                continue;
            }

            // use both triangles for symmetry since we won't visit this edge again
            let mut q = Quadric::ZERO;
            quadric_from_triangle_edge(&mut q, p0, p1, p2, 1.0);
            let mut qo = Quadric::ZERO;
            quadric_from_triangle_edge(&mut qo, p0, p1, p3, 1.0);
            quadric_add(&mut q, &qo);

            quadric_add(&mut vertex_quadrics[remap[i0] as usize], &q);
            quadric_add(&mut vertex_quadrics[remap[i1 as usize] as usize], &q);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn v3(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x, y, z }
    }

    #[test]
    fn triangle_flip_detection() {
        // CCW triangle in xy: replacing c with d below the plane flips
        let a = v3(0.0, 0.0, 0.0);
        let b = v3(1.0, 0.0, 0.0);
        let c = v3(0.0, 1.0, 0.0);
        let d = v3(0.25, 0.25, -1.0);

        assert!(has_triangle_flip(&a, &b, &c, &d));
        assert!(!has_triangle_flip(&a, &b, &c, &v3(0.25, 0.25, 0.1)));
    }

    #[test]
    fn sort_edge_collapses_orders_by_error() {
        let collapses = [
            Collapse {
                v0: 0,
                v1: 1,
                bidi: 0,
                error: 1.0,
            },
            Collapse {
                v0: 2,
                v1: 3,
                bidi: 0,
                error: 0.5,
            },
            Collapse {
                v0: 4,
                v1: 5,
                bidi: 0,
                error: 2.0,
            },
        ];
        let mut order = vec![0u32; 3];

        sort_edge_collapses(&mut order, &collapses, 3);

        assert_eq!(order, vec![1, 0, 2]);
    }

    #[test]
    fn remap_edge_loops_follows_collapse() {
        // loop 1 -> 3; collapsing 3 onto 2 moves the loop target
        let mut edge_loop = vec![EMPTY_SLOT, 3, EMPTY_SLOT, EMPTY_SLOT];
        let collapse_remap = vec![0u32, 1, 2, 2];

        remap_edge_loops(&mut edge_loop, 4, &collapse_remap);
        assert_eq!(edge_loop[1], 2);
    }

    #[test]
    fn remap_index_buffer_drops_degenerate() {
        let mut indices = vec![0u32, 1, 2];
        let index_count = indices.len();
        let collapse_remap = vec![0u32, 0, 2]; // 1 collapses onto 0
        let remap = vec![0u32, 1, 2];

        let write = remap_index_buffer(&mut indices, index_count, &collapse_remap, &remap);
        assert_eq!(write, 0);
    }

    #[test]
    fn get_complex_target_uses_loop_metadata() {
        let remap = vec![0u32, 1, 2, 3];
        // v has a loop half-edge to a wedge of target 2 (same remap slot)
        let edge_loop = vec![EMPTY_SLOT, 3, EMPTY_SLOT, EMPTY_SLOT];
        let loopback = vec![EMPTY_SLOT; 4];

        // remap[3] != remap[2]: falls back to target
        assert_eq!(get_complex_target(1, 2, &remap, &edge_loop, &loopback), 2);

        // if 3 welded onto 2's slot, the loop target wins
        let remap = vec![0u32, 1, 2, 2];
        assert_eq!(get_complex_target(1, 2, &remap, &edge_loop, &loopback), 3);
    }
}
