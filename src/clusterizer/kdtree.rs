//! kd-tree nearest-neighbor search over triangle cones (`clusterizer.cpp`
//! internals, used by `meshopt_buildMeshletsFlex`).

use super::meshlets::{Cone, MESHLET_MAX_TREE_DEPTH};

#[derive(Clone, Copy)]
pub(crate) struct KDNode {
    pub index: u32,
    pub split: f32,
    // leaves: axis = 3, children = number of points including this one
    // branches: axis != 3, left subtree = skip 1, right subtree = skip 1+children
    pub axis: u32,
    pub children: u32,
}

impl KDNode {
    const LEAF: u32 = 3;

    pub fn zero() -> KDNode {
        KDNode {
            index: 0,
            split: 0.0,
            axis: 0,
            children: 0,
        }
    }
}

fn kdtree_partition(indices: &mut [u32], points: &[Cone], axis: usize, pivot: f32) -> usize {
    let mut m = 0usize;

    // invariant: elements in range [0, m) are < pivot, elements in range [m, i) are >= pivot
    for i in 0..indices.len() {
        let v = cone_coord(&points[indices[i] as usize], axis);

        // swap(m, i) unconditionally
        indices.swap(m, i);

        // when v >= pivot, we swap i with m without advancing it, preserving invariants
        m += usize::from(v < pivot);
    }

    m
}

fn cone_coord(point: &Cone, axis: usize) -> f32 {
    match axis {
        0 => point.px,
        1 => point.py,
        _ => point.pz,
    }
}

fn kdtree_build_leaf(offset: usize, nodes: &mut [KDNode], indices: &[u32], count: usize) -> usize {
    debug_assert!(offset + count <= nodes.len());

    nodes[offset] = KDNode {
        index: indices[0],
        split: 0.0,
        axis: KDNode::LEAF,
        children: count as u32,
    };

    // all remaining points are stored in nodes immediately following the leaf
    for i in 1..count {
        nodes[offset + i] = KDNode {
            index: indices[i],
            split: 0.0,
            axis: KDNode::LEAF,
            children: 0x3fff_ffff, // bogus value to prevent misuse
        };
    }

    offset + count
}

pub(crate) fn kdtree_build(
    offset: usize,
    nodes: &mut [KDNode],
    points: &[Cone],
    indices: &mut [u32],
    count: usize,
    leaf_size: usize,
    depth: i32,
) -> usize {
    debug_assert!(count > 0);
    debug_assert!(offset < nodes.len());

    if count <= leaf_size {
        return kdtree_build_leaf(offset, nodes, indices, count);
    }

    let mut mean = [0.0f32; 3];
    let mut vars = [0.0f32; 3];
    let mut runc = 1.0f32;
    let mut runs = 1.0f32;

    // gather statistics on the points in the subtree using Welford's algorithm
    for i in 0..count {
        let point = &points[indices[i] as usize];

        for k in 0..3 {
            let pk = cone_coord(point, k);
            let delta = pk - mean[k];
            mean[k] += delta * runs;
            vars[k] += delta * (pk - mean[k]);
        }

        runc += 1.0;
        runs = 1.0 / runc;
    }

    // split axis is one where the variance is largest
    let axis = if vars[0] >= vars[1] && vars[0] >= vars[2] {
        0
    } else if vars[1] >= vars[2] {
        1
    } else {
        2
    };

    let split = mean[axis];
    let middle = kdtree_partition(indices, points, axis, split);

    // when the partition is degenerate simply consolidate the points into a single node
    // this also ensures recursion depth is bounded on pathological inputs
    if middle <= leaf_size / 2 || middle >= count - leaf_size / 2 || depth >= MESHLET_MAX_TREE_DEPTH
    {
        return kdtree_build_leaf(offset, nodes, indices, count);
    }

    nodes[offset] = KDNode {
        index: 0,
        split,
        axis: axis as u32,
        children: 0,
    };

    // left subtree is right after our node
    let next_offset = kdtree_build(
        offset + 1,
        nodes,
        points,
        &mut indices[..middle],
        middle,
        leaf_size,
        depth + 1,
    );

    // distance to the right subtree is represented explicitly
    debug_assert!(next_offset - offset > 1);
    nodes[offset].children = (next_offset - offset - 1) as u32;

    kdtree_build(
        next_offset,
        nodes,
        points,
        &mut indices[middle..],
        count - middle,
        leaf_size,
        depth + 1,
    )
}

pub(crate) fn kdtree_nearest(
    nodes: &mut [KDNode],
    root: usize,
    points: &[Cone],
    emitted_flags: &[u8],
    position: [f32; 3],
    result: &mut u32,
    limit: &mut f32,
) {
    if nodes[root].children == 0 {
        return;
    }

    if nodes[root].axis == KDNode::LEAF {
        // leaf
        let mut inactive = true;

        for i in 0..nodes[root].children as usize {
            let index = nodes[root + i].index;

            if emitted_flags[index as usize] != 0 {
                continue;
            }

            inactive = false;

            let point = &points[index as usize];

            let dx = point.px - position[0];
            let dy = point.py - position[1];
            let dz = point.pz - position[2];
            let distance = crate::math::sqrt(dx * dx + dy * dy + dz * dz);

            if distance < *limit {
                *result = index;
                *limit = distance;
            }
        }

        // deactivate leaves that no longer have items to emit
        if inactive {
            nodes[root].children = 0;
        }
    } else {
        // branch; we order recursion to process the node that search position is in first
        let axis = nodes[root].axis as usize;
        let children = nodes[root].children as usize;
        let delta = position[axis] - nodes[root].split;
        let first = if delta <= 0.0 { 0 } else { children };
        let second = first ^ children;

        // deactivate branches that no longer have items to emit to accelerate traversal
        // note that we do this *before* recursing which delays deactivation but keeps tail calls
        if nodes[root + 1 + first].children | nodes[root + 1 + second].children == 0 {
            nodes[root].children = 0;
        }

        // recursion depth is bounded by tree depth (which is limited by construction)
        kdtree_nearest(
            nodes,
            root + 1 + first,
            points,
            emitted_flags,
            position,
            result,
            limit,
        );

        // only process the other node if it can have a match based on closest distance so far
        if delta.abs() <= *limit {
            kdtree_nearest(
                nodes,
                root + 1 + second,
                points,
                emitted_flags,
                position,
                result,
                limit,
            );
        }
    }
}
