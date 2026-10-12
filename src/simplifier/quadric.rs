//! Quadric error metric construction, evaluation, and solving for the
//! simplifier. Upstream: `simplifier.cpp` L653-1228 (fold-line quadrics and
//! `oppositeVertex` live in `collapse`).

use super::Vector3;
use super::adjacency::{KIND_BORDER, KIND_COMPLEX, KIND_FRINGE, KIND_MANIFOLD, KIND_SEAM};
use super::simplify::{SIMPLIFY_REGULARIZE, SIMPLIFY_REGULARIZE_LIGHT, SIMPLIFY_VERTEX_PRIORITY};
use crate::math;
use alloc::vec;

pub(crate) const K_MAX_ATTRIBUTES: usize = 32;

// a00*x^2 + a11*y^2 + a22*z^2 + 2*a10*xy + 2*a20*xz + 2*a21*yz + 2*b0*x + 2*b1*y + 2*b2*z + c
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Quadric {
    pub(crate) a00: f32,
    pub(crate) a11: f32,
    pub(crate) a22: f32,
    pub(crate) a10: f32,
    pub(crate) a20: f32,
    pub(crate) a21: f32,
    pub(crate) b0: f32,
    pub(crate) b1: f32,
    pub(crate) b2: f32,
    pub(crate) c: f32,
    pub(crate) w: f32,
}

impl Quadric {
    pub(crate) const ZERO: Quadric = Quadric {
        a00: 0.0,
        a11: 0.0,
        a22: 0.0,
        a10: 0.0,
        a20: 0.0,
        a21: 0.0,
        b0: 0.0,
        b1: 0.0,
        b2: 0.0,
        c: 0.0,
        w: 0.0,
    };
}

// gx*x + gy*y + gz*z + gw
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct QuadricGrad {
    pub(crate) gx: f32,
    pub(crate) gy: f32,
    pub(crate) gz: f32,
    pub(crate) gw: f32,
}

impl QuadricGrad {
    pub(crate) const ZERO: QuadricGrad = QuadricGrad {
        gx: 0.0,
        gy: 0.0,
        gz: 0.0,
        gw: 0.0,
    };
}

fn normalize(v: &mut Vector3) -> f32 {
    let length = math::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

    if length > 0.0 {
        v.x /= length;
        v.y /= length;
        v.z /= length;
    }

    length
}

pub(crate) fn quadric_add(q: &mut Quadric, r: &Quadric) {
    q.a00 += r.a00;
    q.a11 += r.a11;
    q.a22 += r.a22;
    q.a10 += r.a10;
    q.a20 += r.a20;
    q.a21 += r.a21;
    q.b0 += r.b0;
    q.b1 += r.b1;
    q.b2 += r.b2;
    q.c += r.c;
    q.w += r.w;
}

pub(crate) fn quadric_add_grad(g: &mut QuadricGrad, r: &QuadricGrad) {
    g.gx += r.gx;
    g.gy += r.gy;
    g.gz += r.gz;
    g.gw += r.gw;
}

pub(crate) fn quadric_add_grad_n(g: &mut [QuadricGrad], r: &[QuadricGrad], attribute_count: usize) {
    for k in 0..attribute_count {
        g[k].gx += r[k].gx;
        g[k].gy += r[k].gy;
        g[k].gz += r[k].gz;
        g[k].gw += r[k].gw;
    }
}

fn quadric_eval(q: &Quadric, v: &Vector3) -> f32 {
    let mut rx = (q.b0 + q.a10 * v.y) * 2.0;
    let mut ry = (q.b1 + q.a21 * v.z) * 2.0;
    let mut rz = (q.b2 + q.a20 * v.x) * 2.0;

    rx += q.a00 * v.x;
    ry += q.a11 * v.y;
    rz += q.a22 * v.z;

    q.c + rx * v.x + ry * v.y + rz * v.z
}

pub(crate) fn quadric_error(q: &Quadric, v: &Vector3) -> f32 {
    let r = quadric_eval(q, v);
    let s = if q.w == 0.0 { 0.0 } else { 1.0 / q.w };

    r.abs() * s
}

pub(crate) fn quadric_error_attr(
    q: &Quadric,
    g: &[QuadricGrad],
    attribute_count: usize,
    v: &Vector3,
    va: &[f32],
    clamped: bool,
) -> f32 {
    let mut r = quadric_eval(q, v);

    // see quadricFromAttributes for general derivation; here we need to add the parts of (eval(pos) - attr)^2 that depend on attr
    for k in 0..attribute_count {
        let a = va[k];
        let gv = v.x * g[k].gx + v.y * g[k].gy + v.z * g[k].gz + g[k].gw;

        r += a * (a * q.w - 2.0 * gv);
    }

    // note: unlike position error, we do not normalize by Q.w to retain edge scaling as described in quadricFromAttributes
    // we do support clamping, with the error limited to the cumulative weight (= area), which makes perceptual error map to position error more closely
    let e = r.abs();
    if clamped {
        if e < q.w { e } else { q.w }
    } else {
        e
    }
}

fn quadric_from_plane(q: &mut Quadric, a: f32, b: f32, c: f32, d: f32, w: f32) {
    let aw = a * w;
    let bw = b * w;
    let cw = c * w;
    let dw = d * w;

    q.a00 = a * aw;
    q.a11 = b * bw;
    q.a22 = c * cw;
    q.a10 = a * bw;
    q.a20 = a * cw;
    q.a21 = b * cw;
    q.b0 = a * dw;
    q.b1 = b * dw;
    q.b2 = c * dw;
    q.c = d * dw;
    q.w = w;
}

pub(crate) fn quadric_from_point(q: &mut Quadric, x: f32, y: f32, z: f32, w: f32) {
    *q = Quadric {
        a00: w,
        a11: w,
        a22: w,
        a10: 0.0,
        a20: 0.0,
        a21: 0.0,
        b0: -x * w,
        b1: -y * w,
        b2: -z * w,
        c: (x * x + y * y + z * z) * w,
        w,
    };
}

pub(crate) fn quadric_from_triangle(
    q: &mut Quadric,
    p0: Vector3,
    p1: Vector3,
    p2: Vector3,
    weight: f32,
) {
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

    // normal = cross(p1 - p0, p2 - p0)
    let mut normal = Vector3 {
        x: p10.y * p20.z - p10.z * p20.y,
        y: p10.z * p20.x - p10.x * p20.z,
        z: p10.x * p20.y - p10.y * p20.x,
    };
    let area = normalize(&mut normal);

    let distance = normal.x * p0.x + normal.y * p0.y + normal.z * p0.z;

    // we use sqrtf(area) so that the error is scaled linearly; this tends to improve silhouettes
    quadric_from_plane(
        q,
        normal.x,
        normal.y,
        normal.z,
        -distance,
        math::sqrt(area) * weight,
    );
}

pub(crate) fn quadric_from_triangle_edge(
    q: &mut Quadric,
    p0: Vector3,
    p1: Vector3,
    p2: Vector3,
    weight: f32,
) {
    let p10 = Vector3 {
        x: p1.x - p0.x,
        y: p1.y - p0.y,
        z: p1.z - p0.z,
    };

    // edge length; keep squared length around for projection correction
    let lengthsq = p10.x * p10.x + p10.y * p10.y + p10.z * p10.z;
    let length = math::sqrt(lengthsq);

    // p20p = length of projection of p2-p0 onto p1-p0; note that p10 is unnormalized so we need to correct it later
    let p20 = Vector3 {
        x: p2.x - p0.x,
        y: p2.y - p0.y,
        z: p2.z - p0.z,
    };
    let p20p = p20.x * p10.x + p20.y * p10.y + p20.z * p10.z;

    // perp = perpendicular vector from p2 to line segment p1-p0
    // note: since p10 is unnormalized we need to correct the projection; we scale p20 instead to take advantage of normalize below
    let mut perp = Vector3 {
        x: p20.x * lengthsq - p10.x * p20p,
        y: p20.y * lengthsq - p10.y * p20p,
        z: p20.z * lengthsq - p10.z * p20p,
    };
    normalize(&mut perp);

    let distance = perp.x * p0.x + perp.y * p0.y + perp.z * p0.z;

    // note: the weight is scaled linearly with edge length; this has to match the triangle weight
    quadric_from_plane(q, perp.x, perp.y, perp.z, -distance, length * weight);
}

// port-shape: mirrors the upstream parameter list (ADR-0005)
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn quadric_from_attributes(
    q: &mut Quadric,
    g: &mut [QuadricGrad],
    p0: Vector3,
    p1: Vector3,
    p2: Vector3,
    va0: &[f32],
    va1: &[f32],
    va2: &[f32],
    attribute_count: usize,
) {
    // for each attribute we want to encode the following function into the quadric:
    // (eval(pos) - attr)^2
    // where eval(pos) interpolates attribute across the triangle like so:
    // eval(pos) = pos.x * gx + pos.y * gy + pos.z * gz + gw
    // where gx/gy/gz/gw are gradients
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

    // normal = cross(p1 - p0, p2 - p0)
    let normal = Vector3 {
        x: p10.y * p20.z - p10.z * p20.y,
        y: p10.z * p20.x - p10.x * p20.z,
        z: p10.x * p20.y - p10.y * p20.x,
    };
    let area = math::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z) * 0.5;

    // quadric is weighted with the square of edge length (= area)
    // this equalizes the units with the positional error (which, after normalization, is a square of distance)
    // as a result, a change in weighted attribute of 1 along distance d is approximately equivalent to a change in position of d
    let w = area;

    // we compute gradients using barycentric coordinates; barycentric coordinates can be computed as follows:
    // v = (d11 * d20 - d01 * d21) / denom
    // w = (d00 * d21 - d01 * d20) / denom
    // u = 1 - v - w
    // here v0, v1 are triangle edge vectors, v2 is a vector from point to triangle corner, and dij = dot(vi, vj)
    // note: v2 and d20/d21 can not be evaluated here as v2 is effectively an unknown variable; we need these only as variables for derivation of gradients
    let v0 = &p10;
    let v1 = &p20;
    let d00 = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z;
    let d01 = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
    let d11 = v1.x * v1.x + v1.y * v1.y + v1.z * v1.z;
    let denom = d00 * d11 - d01 * d01;
    let denomr = if denom == 0.0 { 0.0 } else { 1.0 / denom };

    // precompute gradient factors
    // these are derived by directly computing derivative of eval(pos) = a0 * u + a1 * v + a2 * w and factoring out expressions that are shared between attributes
    let gx1 = (d11 * v0.x - d01 * v1.x) * denomr;
    let gx2 = (d00 * v1.x - d01 * v0.x) * denomr;
    let gy1 = (d11 * v0.y - d01 * v1.y) * denomr;
    let gy2 = (d00 * v1.y - d01 * v0.y) * denomr;
    let gz1 = (d11 * v0.z - d01 * v1.z) * denomr;
    let gz2 = (d00 * v1.z - d01 * v0.z) * denomr;

    *q = Quadric {
        a00: 0.0,
        a11: 0.0,
        a22: 0.0,
        a10: 0.0,
        a20: 0.0,
        a21: 0.0,
        b0: 0.0,
        b1: 0.0,
        b2: 0.0,
        c: 0.0,
        w,
    };

    for k in 0..attribute_count {
        let (a0, a1, a2) = (va0[k], va1[k], va2[k]);

        // compute gradient of eval(pos) for x/y/z/w
        // the formulas below are obtained by directly computing derivative of eval(pos) = a0 * u + a1 * v + a2 * w
        let gx = gx1 * (a1 - a0) + gx2 * (a2 - a0);
        let gy = gy1 * (a1 - a0) + gy2 * (a2 - a0);
        let gz = gz1 * (a1 - a0) + gz2 * (a2 - a0);
        let gw = a0 - p0.x * gx - p0.y * gy - p0.z * gz;

        // quadric encodes (eval(pos)-attr)^2; this means that the resulting expansion needs to compute, for example, pos.x * pos.y * K
        // since quadrics already encode factors for pos.x * pos.y, we can accumulate almost everything in basic quadric fields
        // note: for simplicity we scale all factors by weight here instead of outside the loop
        q.a00 += w * (gx * gx);
        q.a11 += w * (gy * gy);
        q.a22 += w * (gz * gz);

        q.a10 += w * (gy * gx);
        q.a20 += w * (gz * gx);
        q.a21 += w * (gz * gy);

        q.b0 += w * (gx * gw);
        q.b1 += w * (gy * gw);
        q.b2 += w * (gz * gw);

        q.c += w * (gw * gw);

        // the only remaining sum components are ones that depend on attr; these will be added during error evaluation, see quadricError
        g[k] = QuadricGrad {
            gx: w * gx,
            gy: w * gy,
            gz: w * gz,
            gw: w * gw,
        };
    }
}

pub(crate) fn quadric_volume_gradient(g: &mut QuadricGrad, p0: Vector3, p1: Vector3, p2: Vector3) {
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

    // normal = cross(p1 - p0, p2 - p0)
    let mut normal = Vector3 {
        x: p10.y * p20.z - p10.z * p20.y,
        y: p10.z * p20.x - p10.x * p20.z,
        z: p10.x * p20.y - p10.y * p20.x,
    };
    let area = normalize(&mut normal) * 0.5;

    *g = QuadricGrad {
        gx: normal.x * area,
        gy: normal.y * area,
        gz: normal.z * area,
        gw: (-p0.x * normal.x - p0.y * normal.y - p0.z * normal.z) * area,
    };
}

pub(crate) fn quadric_solve(p: &mut Vector3, q: &Quadric, gv: &QuadricGrad) -> bool {
    // solve A*p = -b where A is the quadric matrix and b is the linear term
    let a00 = q.a00;
    let a11 = q.a11;
    let a22 = q.a22;
    let a10 = q.a10;
    let a20 = q.a20;
    let a21 = q.a21;
    let x0 = -q.b0;
    let x1 = -q.b1;
    let x2 = -q.b2;

    let eps = 1e-6 * q.w;

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

    // augment system with linear constraint GV using Lagrange multiplier
    let a30 = gv.gx;
    let a31 = gv.gy;
    let a32 = gv.gz;
    let x3 = -gv.gw;

    let l30 = a30 / d0;
    let dl31 = a31 - a30 * l10;
    let l31 = dl31 / d1;
    let dl32 = a32 - a30 * l20 - dl31 * l21;
    let l32 = dl32 / d2;
    let d3 = 0.0 - a30 * l30 - dl31 * l31 - dl32 * l32;

    let y3 = x3 - l30 * y0 - l31 * y1 - l32 * y2;
    let z3 = if d3.abs() > eps { y3 / d3 } else { 0.0 }; // if d3 is zero, we can ignore the constraint

    // substitute L^T*p = z
    let lambda = z3;
    let pz = z2 - l32 * lambda;
    let py = z1 - l21 * pz - l31 * lambda;
    let px = z0 - l10 * py - l20 * pz - l30 * lambda;

    *p = Vector3 {
        x: px,
        y: py,
        z: pz,
    };

    d0.abs() > eps && d1.abs() > eps && d2.abs() > eps
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn quadric_reduce_attributes(
    q: &mut Quadric,
    a: &Quadric,
    g: &[QuadricGrad],
    attribute_count: usize,
) {
    // update vertex quadric with attribute quadric; multiply by vertex weight to minimize normalized error
    q.a00 += a.a00 * q.w;
    q.a11 += a.a11 * q.w;
    q.a22 += a.a22 * q.w;
    q.a10 += a.a10 * q.w;
    q.a20 += a.a20 * q.w;
    q.a21 += a.a21 * q.w;
    q.b0 += a.b0 * q.w;
    q.b1 += a.b1 * q.w;
    q.b2 += a.b2 * q.w;

    let iaw = if a.w == 0.0 { 0.0 } else { q.w / a.w };

    // update linear system based on attribute gradients (BB^T/a)
    for k in 0..attribute_count {
        let gk = &g[k];

        q.a00 -= (gk.gx * gk.gx) * iaw;
        q.a11 -= (gk.gy * gk.gy) * iaw;
        q.a22 -= (gk.gz * gk.gz) * iaw;
        q.a10 -= (gk.gx * gk.gy) * iaw;
        q.a20 -= (gk.gx * gk.gz) * iaw;
        q.a21 -= (gk.gy * gk.gz) * iaw;

        q.b0 -= (gk.gx * gk.gw) * iaw;
        q.b1 -= (gk.gy * gk.gw) * iaw;
        q.b2 -= (gk.gz * gk.gw) * iaw;
    }
}

pub(crate) fn fill_face_quadrics(
    vertex_quadrics: &mut [Quadric],
    mut volume_gradients: Option<&mut [QuadricGrad]>,
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[Vector3],
    remap: &[u32],
) {
    for i in (0..index_count).step_by(3) {
        let i0 = indices[i];
        let i1 = indices[i + 1];
        let i2 = indices[i + 2];

        let mut q = Quadric::ZERO;
        quadric_from_triangle(
            &mut q,
            vertex_positions[i0 as usize],
            vertex_positions[i1 as usize],
            vertex_positions[i2 as usize],
            1.0,
        );

        quadric_add(&mut vertex_quadrics[remap[i0 as usize] as usize], &q);
        quadric_add(&mut vertex_quadrics[remap[i1 as usize] as usize], &q);
        quadric_add(&mut vertex_quadrics[remap[i2 as usize] as usize], &q);

        if let Some(volume_gradients) = volume_gradients.as_mut() {
            let mut gv = QuadricGrad::ZERO;
            quadric_volume_gradient(
                &mut gv,
                vertex_positions[i0 as usize],
                vertex_positions[i1 as usize],
                vertex_positions[i2 as usize],
            );

            quadric_add_grad(&mut volume_gradients[remap[i0 as usize] as usize], &gv);
            quadric_add_grad(&mut volume_gradients[remap[i1 as usize] as usize], &gv);
            quadric_add_grad(&mut volume_gradients[remap[i2 as usize] as usize], &gv);
        }
    }
}

pub(crate) fn fill_vertex_quadrics(
    vertex_quadrics: &mut [Quadric],
    vertex_positions: &[Vector3],
    vertex_count: usize,
    remap: &[u32],
    vertex_lock: Option<&[u8]>,
    sparse_remap: Option<&[u32]>,
    options: u32,
) {
    // by default, we use a very small weight to improve triangulation and numerical stability without affecting the shape or error
    let factor = if options & SIMPLIFY_REGULARIZE_LIGHT != 0 {
        1e-2
    } else if options & SIMPLIFY_REGULARIZE != 0 {
        1e-1
    } else {
        1e-7
    };

    for i in 0..vertex_count {
        if remap[i] != i as u32 {
            continue;
        }

        // increase regularization weight for vertices marked as priority; for now we only examine the primary vertex
        let ri = sparse_remap.map_or(i as u32, |r| r[i]) as usize;
        let priority = vertex_lock.is_some_and(|l| l[ri] & SIMPLIFY_VERTEX_PRIORITY != 0);

        let p = &vertex_positions[i];
        let w = vertex_quadrics[i].w * if priority { 1.0 } else { factor };

        let mut q = Quadric::ZERO;
        quadric_from_point(&mut q, p.x, p.y, p.z, w);

        quadric_add(&mut vertex_quadrics[i], &q);
    }
}

// port-shape: mirrors the upstream parameter list (ADR-0005)
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_edge_quadrics(
    vertex_quadrics: &mut [Quadric],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[Vector3],
    remap: &[u32],
    vertex_kind: &[u8],
    edge_loop: &[u32],
    loopback: &[u32],
) {
    const NEXT: [usize; 4] = [1, 2, 0, 1];

    for i in (0..index_count).step_by(3) {
        for e in 0..3 {
            let i0 = indices[i + e];
            let i1 = indices[i + NEXT[e]];

            let k0 = vertex_kind[i0 as usize];
            let k1 = vertex_kind[i1 as usize];

            // early out: manifold vertices never lie on a border
            if k0 == KIND_MANIFOLD || k1 == KIND_MANIFOLD {
                continue;
            }

            // one of the vertices must be border/seam/fringe; additional checks help validate the border
            if k0 != KIND_BORDER
                && k0 != KIND_SEAM
                && k0 != KIND_FRINGE
                && k1 != KIND_BORDER
                && k1 != KIND_SEAM
                && k1 != KIND_FRINGE
            {
                continue;
            }

            // if i0/i1 are border/seam we check if they are on the same edge loop
            // note that we need to add the error even for edges that connect e.g. border & locked
            // if we don't do that, the adjacent border->border edge won't have correct errors for corners
            if (k0 == KIND_BORDER || k0 == KIND_SEAM) && edge_loop[i0 as usize] != i1 {
                continue;
            }

            if (k1 == KIND_BORDER || k1 == KIND_SEAM) && loopback[i1 as usize] != i0 {
                continue;
            }

            // fringe vertices don't have proper edge loops, so we need to add edge quadrics to fringe-fringe or fringe-locked edges
            // note that border-fringe edges are still processed, guarded by the loop check above
            if (k0 == KIND_FRINGE || k1 == KIND_FRINGE)
                && (k0 == KIND_COMPLEX || k1 == KIND_COMPLEX)
            {
                continue;
            }

            let i2 = indices[i + NEXT[e + 1]];

            // we try hard to maintain border edge geometry; seam edges can move more freely
            // due to topological restrictions on collapses, seam quadrics slightly improve collapse structure but aren't critical
            const K_EDGE_WEIGHT_SEAM: f32 = 0.5; // applied twice due to opposite edges
            const K_EDGE_WEIGHT_BORDER: f32 = 10.0;

            let edge_weight = if k0 == KIND_SEAM || k1 == KIND_SEAM {
                K_EDGE_WEIGHT_SEAM
            } else {
                K_EDGE_WEIGHT_BORDER
            };

            let mut q = Quadric::ZERO;
            quadric_from_triangle_edge(
                &mut q,
                vertex_positions[i0 as usize],
                vertex_positions[i1 as usize],
                vertex_positions[i2 as usize],
                edge_weight,
            );

            let mut qt = Quadric::ZERO;
            quadric_from_triangle(
                &mut qt,
                vertex_positions[i0 as usize],
                vertex_positions[i1 as usize],
                vertex_positions[i2 as usize],
                edge_weight,
            );

            // mix edge quadric with triangle quadric to stabilize collapses in both directions; both quadrics inherit edge weight so that their error is added
            qt.w = 0.0;
            quadric_add(&mut q, &qt);

            quadric_add(&mut vertex_quadrics[remap[i0 as usize] as usize], &q);
            quadric_add(&mut vertex_quadrics[remap[i1 as usize] as usize], &q);
        }
    }
}

pub(crate) fn fill_attribute_quadrics(
    attribute_quadrics: &mut [Quadric],
    attribute_gradients: &mut [QuadricGrad],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[Vector3],
    vertex_attributes: &[f32],
    attribute_count: usize,
) {
    let mut qa = Quadric::ZERO;
    let mut g = vec![QuadricGrad::ZERO; attribute_count];

    for i in (0..index_count).step_by(3) {
        let i0 = indices[i];
        let i1 = indices[i + 1];
        let i2 = indices[i + 2];

        quadric_from_attributes(
            &mut qa,
            &mut g,
            vertex_positions[i0 as usize],
            vertex_positions[i1 as usize],
            vertex_positions[i2 as usize],
            &vertex_attributes[i0 as usize * attribute_count..],
            &vertex_attributes[i1 as usize * attribute_count..],
            &vertex_attributes[i2 as usize * attribute_count..],
            attribute_count,
        );

        quadric_add(&mut attribute_quadrics[i0 as usize], &qa);
        quadric_add(&mut attribute_quadrics[i1 as usize], &qa);
        quadric_add(&mut attribute_quadrics[i2 as usize], &qa);

        quadric_add_grad_n(
            &mut attribute_gradients[i0 as usize * attribute_count..],
            &g,
            attribute_count,
        );
        quadric_add_grad_n(
            &mut attribute_gradients[i1 as usize * attribute_count..],
            &g,
            attribute_count,
        );
        quadric_add_grad_n(
            &mut attribute_gradients[i2 as usize * attribute_count..],
            &g,
            attribute_count,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn v3(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x, y, z }
    }

    #[test]
    fn quadric_from_plane_distance_error() {
        // plane x = 2 (normal +x, d = -2), weight 3
        let mut q = Quadric::ZERO;
        quadric_from_plane(&mut q, 1.0, 0.0, 0.0, -2.0, 3.0);

        assert_eq!(q.a00, 3.0);
        assert_eq!(q.b0, -6.0);
        assert_eq!(q.c, 12.0);
        assert_eq!(q.w, 3.0);

        // error at origin: squared distance (4) scaled by weight (3) then normalized
        let e = quadric_error(&q, &v3(0.0, 0.0, 0.0));
        assert_eq!(e, 4.0);
        // error on the plane is zero
        let e = quadric_error(&q, &v3(2.0, 7.0, -3.0));
        assert_eq!(e, 0.0);
    }

    #[test]
    fn quadric_from_triangle_normal() {
        // right triangle in xy plane
        let mut q = Quadric::ZERO;
        quadric_from_triangle(
            &mut q,
            v3(0.0, 0.0, 0.0),
            v3(1.0, 0.0, 0.0),
            v3(0.0, 1.0, 0.0),
            1.0,
        );

        // |cross| = 1 for this triangle, so weight = sqrt(1); plane z = 0
        assert_eq!(q.w, 1.0);
        assert_eq!(q.a22, 1.0); // c=1: a22 = c*cw = 1*(1*1)
        assert_eq!(q.c, 0.0);

        let e = quadric_error(&q, &v3(3.0, 4.0, 0.25));
        assert_eq!(e, 0.0625);
    }

    #[test]
    fn quadric_from_point_minimizer() {
        let mut q = Quadric::ZERO;
        quadric_from_point(&mut q, 3.0, -2.0, 0.5, 2.0);

        // the minimizer of a point quadric is the point itself
        let mut p = v3(0.0, 0.0, 0.0);
        assert!(quadric_solve(&mut p, &q, &QuadricGrad::ZERO));
        assert_eq!(p, v3(3.0, -2.0, 0.5));
    }

    #[test]
    fn quadric_solve_sum_of_planes_and_point() {
        // three axis planes (weight 1) + point quadric at (3,3,3) with weight 3:
        // A = diag(4), b = (-9,-9,-9) => minimizer (9/4, 9/4, 9/4), exact in f32
        let mut q = Quadric::ZERO;
        quadric_from_plane(&mut q, 1.0, 0.0, 0.0, 0.0, 1.0);
        let mut r = Quadric::ZERO;
        quadric_from_plane(&mut r, 0.0, 1.0, 0.0, 0.0, 1.0);
        quadric_add(&mut q, &r);
        quadric_from_plane(&mut r, 0.0, 0.0, 1.0, 0.0, 1.0);
        quadric_add(&mut q, &r);
        quadric_from_point(&mut r, 3.0, 3.0, 3.0, 3.0);
        quadric_add(&mut q, &r);

        let mut p = v3(0.0, 0.0, 0.0);
        assert!(quadric_solve(&mut p, &q, &QuadricGrad::ZERO));
        assert_eq!(p, v3(2.25, 2.25, 2.25));
    }

    #[test]
    fn quadric_solve_degenerate_returns_false() {
        // zero quadric: A = 0, solve fails (d0 = 0)
        let mut p = v3(1.0, 1.0, 1.0);
        assert!(!quadric_solve(&mut p, &Quadric::ZERO, &QuadricGrad::ZERO));
    }

    #[test]
    fn quadric_from_attributes_interpolation_error_zero_at_vertices() {
        // triangle in xy plane; attribute interpolates linearly (eval = 0.5 + 0.5x + 0.5y)
        let p0 = v3(0.0, 0.0, 0.0);
        let p1 = v3(1.0, 0.0, 0.0);
        let p2 = v3(0.0, 1.0, 0.0);
        let a0 = [0.5f32];
        let a1 = [1.0f32];
        let a2 = [1.0f32];

        let mut q = Quadric::ZERO;
        let mut g = vec![QuadricGrad::ZERO; 1];
        quadric_from_attributes(&mut q, &mut g, p0, p1, p2, &a0, &a1, &a2, 1);

        // attribute part of the error vanishes at all three vertices
        for (p, a) in [(p0, &a0), (p1, &a1), (p2, &a2)] {
            let e = quadric_error_attr(&q, &g, 1, &p, a, false);
            assert!(e.abs() < 1e-6, "attribute error at vertex not zero: {e}");
        }

        // ... and is positive away from the linear field
        let e = quadric_error_attr(&q, &g, 1, &v3(0.25, 0.25, 1.0), &[0.0], false);
        assert!(e > 1e-3);
    }

    #[test]
    fn volume_gradient_of_xy_triangle_points_up() {
        let mut g = QuadricGrad::ZERO;
        quadric_volume_gradient(
            &mut g,
            v3(0.0, 0.0, 0.0),
            v3(1.0, 0.0, 0.0),
            v3(0.0, 1.0, 0.0),
        );

        // normalize returns |cross| = 1; area = 1 * 0.5; gradient = normal * area = (0, 0, 0.5)
        assert_eq!(g.gx, 0.0);
        assert_eq!(g.gy, 0.0);
        assert_eq!(g.gz, 0.5);
        assert_eq!(g.gw, 0.0);
    }

    fn unit_quad() -> (Vec<u32>, Vec<Vector3>) {
        (
            vec![0, 1, 2, 0, 2, 3],
            vec![
                v3(0.0, 0.0, 0.0),
                v3(1.0, 0.0, 0.0),
                v3(1.0, 1.0, 0.0),
                v3(0.0, 1.0, 0.0),
            ],
        )
    }

    #[test]
    fn fill_face_quadrics_accumulates_per_wedge() {
        let (indices, positions) = unit_quad();
        let remap = vec![0u32, 1, 2, 3];
        let mut quadrics = vec![Quadric::ZERO; 4];

        fill_face_quadrics(
            &mut quadrics,
            None,
            &indices,
            indices.len(),
            &positions,
            &remap,
        );

        // the shared diagonal edge (0,2) contributes to both adjacent triangles,
        // so vertices 0 and 2 have double weight
        assert_eq!(quadrics[0].w, 2.0);
        assert_eq!(quadrics[2].w, 2.0);
        assert_eq!(quadrics[1].w, 1.0);
        assert_eq!(quadrics[3].w, 1.0);

        // volume gradients: closed book when None is requested
        assert!(quadrics.iter().all(|q| q.w > 0.0));
    }

    #[test]
    fn fill_vertex_quadrics_regularization_weights() {
        let (indices, positions) = unit_quad();
        let remap = vec![0u32, 1, 2, 3];
        let mut quadrics = vec![Quadric::ZERO; 4];
        fill_face_quadrics(
            &mut quadrics,
            None,
            &indices,
            indices.len(),
            &positions,
            &remap,
        );

        let quadrics_default = {
            let mut q = quadrics;
            fill_vertex_quadrics(&mut q, &positions, 4, &remap, None, None, 0);
            // default factor 1e-7 against face weights (impl-exact: w + w * factor)
            assert_eq!(q[0].w, 2.0 + 2.0 * 1e-7);
            assert_eq!(q[1].w, 1.0 + 1.0 * 1e-7);
            q
        };

        let mut quadrics_regularize = quadrics_default;
        fill_vertex_quadrics(
            &mut quadrics_regularize,
            &positions,
            4,
            &remap,
            None,
            None,
            SIMPLIFY_REGULARIZE,
        );
        // regularization re-adds w * 1e-1 on top of the default fill
        let w1 = 1.0f32 + 1.0 * 1e-7;
        assert_eq!(quadrics_regularize[1].w, w1 + w1 * 1e-1);
    }

    #[test]
    fn fill_edge_quadrics_border_weight() {
        let (indices, positions) = unit_quad();
        let remap = vec![0u32, 1, 2, 3];
        let mut quadrics = vec![Quadric::ZERO; 4];
        fill_face_quadrics(
            &mut quadrics,
            None,
            &indices,
            indices.len(),
            &positions,
            &remap,
        );

        // open quad: every vertex is on the border
        let kind = vec![KIND_BORDER; 4];
        // loop data must reflect the actual border half-edges: 0->1, 1->2, 2->3, 3->0
        let edge_loop = vec![1u32, 2, 3, 0];
        let loopback = vec![3u32, 0, 1, 2];

        fill_edge_quadrics(
            &mut quadrics,
            &indices,
            indices.len(),
            &positions,
            &remap,
            &kind,
            &edge_loop,
            &loopback,
        );

        // each vertex keeps its face-quadric weight (2 for the diagonal-shared
        // vertices, 1 otherwise) plus two unit border edges at w = length * 10
        assert_eq!(quadrics[0].w, 22.0);
        assert_eq!(quadrics[1].w, 21.0);
        assert_eq!(quadrics[2].w, 22.0);
        assert_eq!(quadrics[3].w, 21.0);
    }
}
