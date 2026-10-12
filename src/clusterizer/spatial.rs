//! Spatial meshlet builder (`meshopt_buildMeshletsSpatial` and its BVH internals,
//! from `clusterizer.cpp`).

use super::meshlets::{Meshlet, append_meshlet, clear_used};

#[derive(Clone, Copy)]
struct BvhBox {
    min: [f32; 3],
    max: [f32; 3],
}

#[derive(Clone, Copy)]
struct BvhBoxT {
    min: [f32; 3],
    max: [f32; 3],
}

fn box_merge(bvbox: &mut BvhBoxT, other: &BvhBox) -> f32 {
    for k in 0..3 {
        bvbox.min[k] = if other.min[k] < bvbox.min[k] {
            other.min[k]
        } else {
            bvbox.min[k]
        };
        bvbox.max[k] = if other.max[k] > bvbox.max[k] {
            other.max[k]
        } else {
            bvbox.max[k]
        };
    }

    let sx = bvbox.max[0] - bvbox.min[0];
    let sy = bvbox.max[1] - bvbox.min[1];
    let sz = bvbox.max[2] - bvbox.min[2];
    sx * sy + sy * sz + sz * sx
}

fn radix_float(v: u32) -> u32 {
    // if sign bit is 0, flip sign bit
    // if sign bit is 1, flip everything
    let mask = ((v as i32) >> 31) as u32 | 0x8000_0000;
    v ^ mask
}

fn compute_histogram(hist: &mut [[u32; 3]; 1024], data: &[f32]) {
    *hist = [[0u32; 3]; 1024];

    // compute 3 10-bit histograms in parallel (dropping 2 LSB)
    for &x in data {
        let id = radix_float(x.to_bits());

        hist[((id >> 2) & 1023) as usize][0] += 1;
        hist[((id >> 12) & 1023) as usize][1] += 1;
        hist[((id >> 22) & 1023) as usize][2] += 1;
    }

    let (mut sum0, mut sum1, mut sum2) = (0u32, 0u32, 0u32);

    // replace histogram data with prefix histogram sums in-place
    for h in hist.iter_mut() {
        let (hx, hy, hz) = (h[0], h[1], h[2]);

        h[0] = sum0;
        h[1] = sum1;
        h[2] = sum2;

        sum0 += hx;
        sum1 += hy;
        sum2 += hz;
    }
}

fn radix_pass(
    destination: &mut [u32],
    source: &[u32],
    keys: &[f32],
    hist: &mut [[u32; 3]; 1024],
    pass: usize,
) {
    let bitoff = pass * 10 + 2; // drop 2 LSB to be able to use 3 10-bit passes

    for &s in source {
        let id = (radix_float(keys[s as usize].to_bits()) >> bitoff) & 1023;

        let h = &mut hist[id as usize][pass];
        destination[*h as usize] = s;
        *h += 1;
    }
}

fn bvh_prepare(
    boxes: &mut [BvhBox],
    centroids: &mut [f32],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_stride_float: usize,
) {
    let face_count = indices.len() / 3;

    for i in 0..face_count {
        let a = indices[i * 3] as usize;
        let b = indices[i * 3 + 1] as usize;
        let c = indices[i * 3 + 2] as usize;

        let va = &vertex_positions[vertex_stride_float * a..];
        let vb = &vertex_positions[vertex_stride_float * b..];
        let vc = &vertex_positions[vertex_stride_float * c..];

        let bvbox = &mut boxes[i];

        for k in 0..3 {
            bvbox.min[k] = if va[k] < vb[k] { va[k] } else { vb[k] };
            bvbox.min[k] = if vc[k] < bvbox.min[k] {
                vc[k]
            } else {
                bvbox.min[k]
            };

            bvbox.max[k] = if va[k] > vb[k] { va[k] } else { vb[k] };
            bvbox.max[k] = if vc[k] > bvbox.max[k] {
                vc[k]
            } else {
                bvbox.max[k]
            };

            centroids[i + face_count * k] = (bvbox.min[k] + bvbox.max[k]) / 2.0;
        }
    }
}

fn bvh_count_vertices(
    order: &[u32],
    used: &mut [i16],
    indices: &[u32],
    mut out: Option<&mut [u32]>,
) -> usize {
    // count number of unique vertices
    let mut used_vertices = 0usize;
    for (i, &index) in order.iter().enumerate() {
        let a = indices[index as usize * 3] as usize;
        let b = indices[index as usize * 3 + 1] as usize;
        let c = indices[index as usize * 3 + 2] as usize;

        used_vertices +=
            usize::from(used[a] < 0) + usize::from(used[b] < 0) + usize::from(used[c] < 0);
        used[a] = 1;
        used[b] = 1;
        used[c] = 1;

        if let Some(out) = &mut out {
            out[i] = used_vertices as u32;
        }
    }

    // reset used[] for future invocations
    for &index in order {
        let a = indices[index as usize * 3] as usize;
        let b = indices[index as usize * 3 + 1] as usize;
        let c = indices[index as usize * 3 + 2] as usize;

        used[a] = -1;
        used[b] = -1;
        used[c] = -1;
    }

    used_vertices
}

fn bvh_pack_leaf(boundary: &mut [u8], count: usize) {
    // mark meshlet boundary for future reassembly
    debug_assert!(count > 0);

    boundary[0] = 1;
    boundary[1..count].fill(0);
}

fn bvh_pack_tail(
    boundary: &mut [u8],
    order: &[u32],
    used: &mut [i16],
    indices: &[u32],
    max_vertices: usize,
    max_triangles: usize,
) {
    let mut i = 0usize;
    while i < order.len() {
        let chunk = if i + max_triangles <= order.len() {
            max_triangles
        } else {
            order.len() - i
        };

        if bvh_count_vertices(&order[i..i + chunk], used, indices, None) <= max_vertices {
            bvh_pack_leaf(&mut boundary[i..], chunk);
            i += chunk;
            continue;
        }

        // chunk is vertex bound, split it into smaller meshlets
        debug_assert!(chunk > max_vertices / 3);

        bvh_pack_leaf(&mut boundary[i..], max_vertices / 3);
        i += max_vertices / 3;
    }
}

fn bvh_divisible(count: usize, min: usize, max: usize) -> bool {
    // count is representable as a sum of values in [min..max] if it is in range of [k*min..k*min+k*(max-min)]
    // equivalent to ceil(count / max) <= floor(count / min), but the form below allows using idiv (see nv_cluster_builder)
    // we avoid expensive integer divisions in the common case where min is <= max/2
    if min * 2 <= max {
        count >= min
    } else {
        count % min <= (count / min) * (max - min)
    }
}

fn bvh_compute_area(areas: &mut [f32], boxes: &[BvhBox], order: &[u32]) {
    let mut accuml = BvhBoxT {
        min: [f32::MAX; 3],
        max: [-f32::MAX; 3],
    };
    let mut accumr = accuml;

    let count = order.len();

    for i in 0..count {
        let larea = box_merge(&mut accuml, &boxes[order[i] as usize]);
        let rarea = box_merge(&mut accumr, &boxes[order[count - 1 - i] as usize]);

        areas[i] = larea;
        areas[i + count] = rarea;
    }
}

// port-shape: mirrors the upstream parameter list (ADR-0005)
#[allow(clippy::too_many_arguments)]
fn bvh_pivot(
    areas: &[f32],
    vertices: Option<&[u32]>,
    count: usize,
    step: usize,
    min: usize,
    max: usize,
    fill: f32,
    maxfill: usize,
    out_cost: &mut f32,
) -> usize {
    let aligned = count >= min * 2 && bvh_divisible(count, min, max);
    let end = if aligned { count - min } else { count - 1 };

    let rmaxfill = 1.0 / maxfill as f32;

    // find best split that minimizes SAH
    let mut bestsplit = 0usize;
    let mut bestcost = f32::MAX;

    let mut i = min - 1;
    while i < end {
        let lsplit = i + 1;
        let rsplit = count - (i + 1);

        if !bvh_divisible(lsplit, min, max) {
            i += step;
            continue;
        }
        if aligned && !bvh_divisible(rsplit, min, max) {
            i += step;
            continue;
        }

        // areas[x] = inclusive surface area of boxes[0..x]
        // areas[count-1-x] = inclusive surface area of boxes[x..count-1]
        let larea = areas[i];
        let rarea = areas[(count - 1 - (i + 1)) + count];
        let mut cost = larea * (lsplit as f32) + rarea * (rsplit as f32);

        if cost > bestcost {
            i += step;
            continue;
        }

        // use vertex fill when splitting vertex limited clusters; note that we use the same (left->right) vertex count
        // using bidirectional vertex counts is a little more expensive to compute and produces slightly worse results in practice
        let (lfill, rfill) = match vertices {
            Some(vertices) => (vertices[i] as usize, vertices[i] as usize),
            None => (lsplit, rsplit),
        };

        // fill cost; use floating point math to round up to maxfill to avoid expensive integer modulo
        let lrest = crate::quantization::cast_i32((lfill + maxfill - 1) as f32 * rmaxfill)
            * maxfill as i32
            - lfill as i32;
        let rrest = crate::quantization::cast_i32((rfill + maxfill - 1) as f32 * rmaxfill)
            * maxfill as i32
            - rfill as i32;

        cost += fill * (lrest as f32 * larea + rrest as f32 * rarea);

        if cost < bestcost {
            bestcost = cost;
            bestsplit = i + 1;
        }

        i += step;
    }

    *out_cost = bestcost;
    bestsplit
}

fn bvh_partition(target: &mut [u32], order: &[u32], sides: &[u8], split: usize) {
    let mut l = 0usize;
    let mut r = split;

    for &o in order {
        let side = sides[o as usize];
        if side != 0 {
            target[r] = o;
            r += 1;
        } else {
            target[l] = o;
            l += 1;
        }
    }

    debug_assert!(l == split && r == order.len());
}

// To avoid excessive recursion for malformed inputs, we limit the maximum depth of the tree
const MESHLET_MAX_TREE_DEPTH: i32 = 50;

#[allow(clippy::too_many_arguments)]
fn bvh_split(
    boxes: &[BvhBox],
    orderx: &mut [u32],
    ordery: &mut [u32],
    orderz: &mut [u32],
    boundary: &mut [u8],
    count: usize,
    depth: i32,
    areas: &mut [f32],
    vertices: &mut [u32],
    sides: &mut [u8],
    temp: &mut [u32],
    used: &mut [i16],
    indices: &[u32],
    max_vertices: usize,
    min_triangles: usize,
    max_triangles: usize,
    fill_weight: f32,
) {
    // C passes raw pointers and only touches the first `count` entries of each axis;
    // slice here so the recursive calls see the exact sub-problem size
    let orderx = &mut orderx[..count];
    let ordery = &mut ordery[..count];
    let orderz = &mut orderz[..count];
    let boundary = &mut boundary[..count];

    if count <= max_triangles
        && bvh_count_vertices(&orderx[..count], used, indices, None) <= max_vertices
    {
        return bvh_pack_leaf(boundary, count);
    }

    // we can use step=1 unconditionally but to reduce the cost for min=max case we use step=max
    let step = if min_triangles == max_triangles && count > max_triangles {
        max_triangles
    } else {
        1
    };

    // if we could not pack the meshlet, we must be vertex bound
    let mint = if count <= max_triangles && max_vertices / 3 < min_triangles {
        max_vertices / 3
    } else {
        min_triangles
    };
    let maxfill = if count <= max_triangles {
        max_vertices
    } else {
        max_triangles
    };

    // find best split that minimizes SAH
    let mut bestk: i32 = -1;
    let mut bestsplit = 0usize;
    let mut bestcost = f32::MAX;

    for k in 0..3usize {
        let axis: &mut [u32] = match k {
            0 => orderx,
            1 => ordery,
            _ => orderz,
        };

        bvh_compute_area(areas, boxes, axis);

        let mut vertices_ref: Option<&[u32]> = None;

        if count <= max_triangles {
            // for vertex bound clusters, count number of unique vertices for each split
            bvh_count_vertices(axis, used, indices, Some(&mut vertices[..count]));
            vertices_ref = Some(&vertices[..count]);
        }

        let mut axiscost = f32::MAX;
        let axissplit = bvh_pivot(
            areas,
            vertices_ref,
            count,
            step,
            mint,
            max_triangles,
            fill_weight,
            maxfill,
            &mut axiscost,
        );

        if axissplit != 0 && axiscost < bestcost {
            bestk = k as i32;
            bestcost = axiscost;
            bestsplit = axissplit;
        }
    }

    // this may happen if SAH costs along the admissible splits are NaN, or due to imbalanced splits on pathological inputs
    if bestk < 0 || depth >= MESHLET_MAX_TREE_DEPTH {
        return bvh_pack_tail(boundary, orderx, used, indices, max_vertices, max_triangles);
    }

    // mark sides of split for partitioning
    let best_axis: &[u32] = match bestk {
        0 => orderx,
        1 => ordery,
        _ => orderz,
    };

    for i in 0..bestsplit {
        sides[best_axis[i] as usize] = 0;
    }

    for i in bestsplit..count {
        sides[best_axis[i] as usize] = 1;
    }

    // partition all axes into two sides, maintaining order
    for k in 0..3usize {
        if k as i32 == bestk {
            continue;
        }

        let axis: &mut [u32] = match k {
            0 => orderx,
            1 => ordery,
            _ => orderz,
        };
        temp[..count].copy_from_slice(&axis[..count]);
        bvh_partition(axis, &temp[..count], sides, bestsplit);
    }

    // recursion depth is bounded due to max depth check above
    bvh_split(
        boxes,
        orderx,
        ordery,
        orderz,
        boundary,
        bestsplit,
        depth + 1,
        areas,
        vertices,
        sides,
        temp,
        used,
        indices,
        max_vertices,
        min_triangles,
        max_triangles,
        fill_weight,
    );
    bvh_split(
        boxes,
        &mut orderx[bestsplit..],
        &mut ordery[bestsplit..],
        &mut orderz[bestsplit..],
        &mut boundary[bestsplit..],
        count - bestsplit,
        depth + 1,
        areas,
        vertices,
        sides,
        temp,
        used,
        indices,
        max_vertices,
        min_triangles,
        max_triangles,
        fill_weight,
    );
}

/// Upstream counterpart: `meshopt_buildMeshletsSpatial`.
#[allow(clippy::too_many_arguments)]
pub fn build_meshlets_spatial(
    meshlets: &mut [Meshlet],
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    max_vertices: usize,
    min_triangles: usize,
    max_triangles: usize,
    fill_weight: f32,
) -> usize {
    debug_assert!(indices.len().is_multiple_of(3));
    debug_assert!(
        (12..=256).contains(&vertex_positions_stride) && vertex_positions_stride.is_multiple_of(4)
    );
    debug_assert!((3..=256).contains(&max_vertices));
    debug_assert!(
        (1..=512).contains(&min_triangles)
            && min_triangles <= max_triangles
            && max_triangles <= 512
    );
    debug_assert!(fill_weight >= 0.0);

    let index_count = indices.len();
    if index_count == 0 {
        return 0;
    }

    let face_count = index_count / 3;
    let vertex_stride_float = vertex_positions_stride / 4;

    // compute bounding boxes and centroids for sorting
    let mut boxes = alloc::vec![
        BvhBox {
            min: [0.0; 3],
            max: [0.0; 3]
        };
        face_count
    ];
    let mut centroids = alloc::vec![0.0f32; face_count * 3];
    bvh_prepare(
        &mut boxes,
        &mut centroids,
        indices,
        vertex_positions,
        vertex_stride_float,
    );

    let mut axes = alloc::vec![0u32; face_count * 3];
    let mut temp = alloc::vec![0u32; face_count];

    for k in 0..3usize {
        let keys = &centroids[k * face_count..(k + 1) * face_count];

        let mut hist = [[0u32; 3]; 1024];
        compute_histogram(&mut hist, keys);

        // 3-pass radix sort computes the resulting order into axes
        for (i, t) in temp[..face_count].iter_mut().enumerate() {
            *t = i as u32;
        }

        let order = &mut axes[k * face_count..(k + 1) * face_count];
        radix_pass(order, &temp, keys, &mut hist, 0);
        radix_pass(&mut temp, order, keys, &mut hist, 1);
        radix_pass(order, &temp, keys, &mut hist, 2);
    }

    // index of the vertex in the meshlet, -1 if the vertex isn't used
    let mut used = alloc::vec![0i16; vertex_count];
    clear_used(&mut used, indices);

    let mut boundary = alloc::vec![0u8; face_count];

    {
        let mut areas = alloc::vec![0.0f32; face_count * 2];
        let mut vertices_scratch = alloc::vec![0u32; face_count];
        let mut sides = alloc::vec![0u8; face_count];

        let (orderx, rest) = axes.split_at_mut(face_count);
        let (ordery, orderz) = rest.split_at_mut(face_count);

        bvh_split(
            &boxes,
            orderx,
            ordery,
            orderz,
            &mut boundary,
            face_count,
            0,
            &mut areas,
            &mut vertices_scratch,
            &mut sides,
            &mut temp,
            &mut used,
            indices,
            max_vertices,
            min_triangles,
            max_triangles,
            fill_weight,
        );
    }

    // compute the desired number of meshlets; note that on some meshes with a lot of vertex bound clusters this might go over the bound
    let mut meshlet_count = 0usize;
    for &b in &boundary[..face_count] {
        debug_assert!(b <= 1);
        meshlet_count += b as usize;
    }

    let meshlet_bound =
        super::meshlets::build_meshlets_bound(index_count, max_vertices, min_triangles);

    // pack triangles into meshlets according to the order and boundaries marked by bvhSplit
    let mut meshlet = Meshlet {
        vertex_offset: 0,
        triangle_offset: 0,
        vertex_count: 0,
        triangle_count: 0,
    };
    let mut meshlet_offset = 0usize;
    let mut meshlet_pending = meshlet_count;

    for i in 0..face_count {
        debug_assert!(boundary[i] <= 1);
        let mut split = i > 0 && boundary[i] == 1;

        // while we are over the limit, we ignore boundary[] data and disable splits until we free up enough space
        if split
            && meshlet_count > meshlet_bound
            && meshlet_offset + meshlet_pending >= meshlet_bound
        {
            split = false;
        }

        let index = axes[i] as usize;
        debug_assert!(index < face_count);

        let a = indices[index * 3];
        let b = indices[index * 3 + 1];
        let c = indices[index * 3 + 2];

        // appends triangle to the meshlet and writes previous meshlet to the output if full
        meshlet_offset += append_meshlet(
            &mut meshlet,
            a,
            b,
            c,
            &mut used,
            meshlets,
            meshlet_vertices,
            meshlet_triangles,
            meshlet_offset,
            max_vertices,
            max_triangles,
            split,
        ) as usize;
        meshlet_pending -= boundary[i] as usize;
    }

    if meshlet.triangle_count != 0 {
        meshlets[meshlet_offset] = meshlet;
        meshlet_offset += 1;
    }

    meshlet_offset
}
