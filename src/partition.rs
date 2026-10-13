//! Cluster partitioning by adjacency merging.
//!
//! Upstream: `partition.cpp` (`meshopt_partitionClusters`).
//! Based on Takio Kurita, An efficient agglomerative clustering algorithm using a heap, 1991.

use crate::math;
use alloc::vec;
use alloc::vec::Vec;

// To avoid excessive recursion for malformed inputs, we switch to bisection after some depth
const K_MERGE_DEPTH_CUTOFF: i32 = 40;

struct ClusterAdjacency {
    offsets: Vec<u32>,
    clusters: Vec<u32>,
    shared: Vec<u32>,
}

fn filter_cluster_indices(
    data: &mut [u32],
    offsets: &mut [u32],
    cluster_indices: &[u32],
    cluster_index_counts: &[u32],
    used: &mut [u8],
) {
    let (mut cluster_start, mut cluster_write) = (0usize, 0usize);

    for (i, &count) in cluster_index_counts.iter().enumerate() {
        offsets[i] = cluster_write as u32;

        // copy cluster indices, skipping duplicates
        for j in 0..count as usize {
            let v = cluster_indices[cluster_start + j] as usize;
            debug_assert!(v < used.len());

            data[cluster_write] = v as u32;
            cluster_write += usize::from(used[v] == 0);
            used[v] = 1;
        }

        // reset used flags for the next cluster
        for slot in data[offsets[i] as usize..cluster_write].iter() {
            used[*slot as usize] = 0;
        }

        cluster_start += count as usize;
    }

    debug_assert!(cluster_write <= cluster_indices.len());
    offsets[cluster_index_counts.len()] = cluster_write as u32;
}

fn compute_cluster_bounds(
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_stride_float: usize,
    out_center: &mut [f32; 3],
) -> f32 {
    let index_count = indices.len();
    let mut center = [0f32; 3];

    // approximate center of the cluster by averaging all vertex positions
    for &index in indices.iter() {
        let p = &vertex_positions[index as usize * vertex_stride_float..][..3];

        center[0] += p[0];
        center[1] += p[1];
        center[2] += p[2];
    }

    // note: technically clusters can't be empty per meshopt_partitionCluster but we check for a
    // division by zero in case that changes
    if index_count != 0 {
        let n = index_count as f32;
        center[0] /= n;
        center[1] /= n;
        center[2] /= n;
    }

    // compute radius of the bounding sphere for each cluster
    let mut radiussq = 0f32;

    for &index in indices.iter() {
        let p = &vertex_positions[index as usize * vertex_stride_float..][..3];

        let d2 = (p[0] - center[0]) * (p[0] - center[0])
            + (p[1] - center[1]) * (p[1] - center[1])
            + (p[2] - center[2]) * (p[2] - center[2]);

        if radiussq < d2 {
            radiussq = d2;
        }
    }

    *out_center = center;
    math::sqrt(radiussq)
}

fn build_cluster_adjacency(
    cluster_indices: &[u32],
    cluster_offsets: &[u32],
    cluster_count: usize,
    vertex_count: usize,
) -> ClusterAdjacency {
    let mut ref_offsets = vec![0u32; vertex_count + 1];

    // compute number of clusters referenced by each vertex
    for i in 0..cluster_count {
        for &v in &cluster_indices[cluster_offsets[i] as usize..cluster_offsets[i + 1] as usize] {
            ref_offsets[v as usize] += 1;
        }
    }

    // compute (worst-case) number of adjacent clusters for each cluster
    let mut total_adjacency = 0usize;

    for i in 0..cluster_count {
        let mut count = 0usize;

        // worst case is every vertex has a disjoint cluster list
        for &v in &cluster_indices[cluster_offsets[i] as usize..cluster_offsets[i + 1] as usize] {
            count += ref_offsets[v as usize] as usize - 1;
        }

        // ... but only every other cluster can be adjacent in the end
        total_adjacency += if count < cluster_count - 1 {
            count
        } else {
            cluster_count - 1
        };
    }

    // we can now allocate adjacency buffers
    let mut offsets = vec![0u32; cluster_count + 1];
    let mut clusters = vec![0u32; total_adjacency];
    let mut shared = vec![0u32; total_adjacency];

    // convert ref counts to offsets
    let mut total_refs = 0usize;

    for r in ref_offsets[..vertex_count].iter_mut() {
        let count = *r as usize;
        *r = total_refs as u32;
        total_refs += count;
    }

    let mut ref_data = vec![0u32; total_refs];

    // fill cluster refs for each vertex
    for i in 0..cluster_count {
        for &v in &cluster_indices[cluster_offsets[i] as usize..cluster_offsets[i + 1] as usize] {
            let slot = ref_offsets[v as usize] as usize;
            ref_data[slot] = i as u32;
            ref_offsets[v as usize] += 1;
        }
    }

    // after the previous pass, ref_offsets contain the end of the data for each vertex; shift it
    // forward to get the start
    ref_offsets.copy_within(0..vertex_count, 1);
    ref_offsets[0] = 0;

    let mut slots = vec![!0u32; cluster_count];

    // fill cluster adjacency for each cluster...
    offsets[0] = 0;

    for i in 0..cluster_count {
        let adj_begin = offsets[i] as usize;
        let mut count = 0usize;

        for &v in &cluster_indices[cluster_offsets[i] as usize..cluster_offsets[i + 1] as usize] {
            // merge the entire cluster list of each vertex into current list
            for k in ref_offsets[v as usize]..ref_offsets[v as usize + 1] {
                let c = ref_data[k as usize];
                debug_assert!(c < cluster_count as u32);

                if c == i as u32 {
                    continue;
                }

                // if the cluster is already in the list, increment the shared count
                // we do not need to clear slots across clusters due to adj[slot] confirmation
                // check
                let slot = slots[c as usize];
                if slot < count as u32 && clusters[adj_begin + slot as usize] == c {
                    shared[adj_begin + slot as usize] += 1;
                } else {
                    slots[c as usize] = count as u32;
                    clusters[adj_begin + count] = c;
                    shared[adj_begin + count] = 1;
                    count += 1;
                }
            }
        }

        // mark the end of the adjacency list; the next cluster will start there as well
        offsets[i + 1] = offsets[i] + count as u32;
    }

    debug_assert!(offsets[cluster_count] as usize <= total_adjacency);

    ClusterAdjacency {
        offsets,
        clusters,
        shared,
    }
}

#[derive(Clone, Copy)]
struct ClusterGroup {
    group: i32,
    next: i32,
    size: u32, // 0 unless root
    vertices: u32,

    center: [f32; 3],
    radius: f32,
}

#[derive(Clone, Copy)]
struct GroupOrder {
    id: u32,
    order: i32,
}

/// Returns disjoint mutable references to elements `i` and `j` (i != j), order-independent.
fn two_mut<T>(slice: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    debug_assert!(i != j);
    if i < j {
        let (lo, hi) = slice.split_at_mut(j);
        (&mut lo[i], &mut hi[0])
    } else {
        let (lo, hi) = slice.split_at_mut(i);
        (&mut hi[0], &mut lo[j])
    }
}

fn heap_push(heap: &mut [GroupOrder], size: usize, item: GroupOrder) -> usize {
    // insert a new element at the end (breaks heap invariant)
    heap[size] = item;
    let size = size + 1;

    // bubble up the new element to its correct position
    let mut i = size - 1;
    while i > 0 && heap[i].order < heap[(i - 1) / 2].order {
        let p = (i - 1) / 2;

        heap.swap(i, p);
        i = p;
    }

    size
}

fn heap_pop(heap: &mut [GroupOrder], size: usize) -> (GroupOrder, usize) {
    debug_assert!(size > 0);
    let top = heap[0];
    let size = size - 1;

    // move the last element to the top (breaks heap invariant)
    heap[0] = heap[size];

    // bubble down the new top element to its correct position
    let mut i = 0usize;
    while i * 2 + 1 < size {
        // find the smallest child
        let mut j = i * 2 + 1;
        if j + 1 < size && heap[j + 1].order < heap[j].order {
            j += 1;
        }

        // if the parent is already smaller than both children, we're done
        if heap[j].order >= heap[i].order {
            break;
        }

        // otherwise, swap the parent and child and continue
        heap.swap(i, j);
        i = j;
    }

    (top, size)
}

fn merge_bounds(target: &mut ClusterGroup, source: &ClusterGroup) {
    let (r1, r2) = (target.radius, source.radius);
    let dx = source.center[0] - target.center[0];
    let dy = source.center[1] - target.center[1];
    let dz = source.center[2] - target.center[2];
    let d = math::sqrt(dx * dx + dy * dy + dz * dz);

    if d + r1 < r2 {
        target.center = source.center;
        target.radius = source.radius;
        return;
    }

    if d + r2 > r1 {
        let k = if d > 0.0 {
            (d + r2 - r1) / (2.0 * d)
        } else {
            0.0
        };

        target.center[0] += dx * k;
        target.center[1] += dy * k;
        target.center[2] += dz * k;
        target.radius = (d + r2 + r1) / 2.0;
    }
}

fn bounds_score(target: &ClusterGroup, source: &ClusterGroup) -> f32 {
    let (r1, r2) = (target.radius, source.radius);
    let dx = source.center[0] - target.center[0];
    let dy = source.center[1] - target.center[1];
    let dz = source.center[2] - target.center[2];
    let d = math::sqrt(dx * dx + dy * dy + dz * dz);

    let mr = if d + r1 < r2 {
        r2
    } else if d + r2 < r1 {
        r1
    } else {
        (d + r2 + r1) / 2.0
    };

    if mr > 0.0 { r1 / mr } else { 0.0 }
}

fn pick_group_to_merge(
    groups: &[ClusterGroup],
    id: usize,
    adjacency: &ClusterAdjacency,
    max_partition_size: usize,
    use_bounds: bool,
    shared_acc: &mut [u32],
) -> (i32, u32) {
    debug_assert!(groups[id].size > 0);

    let group_rsqrt = 1.0 / math::sqrt(groups[id].vertices as f32);

    let mut best_group = -1i32;
    let mut best_score = 0f32;
    let mut best_shared = 0u32;

    // compute shared count based on underlying clusters for every adjacent group
    // note: we assume shared_acc only contains zeros on input
    let mut ci = id as i32;
    while ci >= 0 {
        let begin = adjacency.offsets[ci as usize];
        let end = adjacency.offsets[ci as usize + 1];

        for adj in begin..end {
            let other = groups[adjacency.clusters[adj as usize] as usize].group;
            if other >= 0 {
                shared_acc[other as usize] += adjacency.shared[adj as usize];
            }
        }

        ci = groups[ci as usize].next;
    }

    // score each adjacent group and pick the best one; resets shared_acc in the process
    let mut ci = id as i32;
    while ci >= 0 {
        let begin = adjacency.offsets[ci as usize];
        let end = adjacency.offsets[ci as usize + 1];

        for adj in begin..end {
            let other = groups[adjacency.clusters[adj as usize] as usize].group;
            if other < 0 || shared_acc[other as usize] == 0 {
                continue;
            }

            // note: we reset shared_acc here to make sure the next invocation sees zero inputs;
            // it's important that this is done before any early outs
            let shared = shared_acc[other as usize];
            shared_acc[other as usize] = 0;

            debug_assert!(groups[other as usize].size > 0);
            if groups[id].size + groups[other as usize].size > max_partition_size as u32 {
                continue;
            }

            let other_rsqrt = 1.0 / math::sqrt(groups[other as usize].vertices as f32);

            // normalize shared count by the expected boundary of each group (+ keeps scoring
            // symmetric)
            let mut score = shared as f32 * (group_rsqrt + other_rsqrt);

            // incorporate spatial score to favor merging nearby groups
            if use_bounds {
                score *= 1.0 + 0.4 * bounds_score(&groups[id], &groups[other as usize]);
            }

            if score > best_score {
                best_group = other;
                best_score = score;
                best_shared = shared;
            }
        }

        ci = groups[ci as usize].next;
    }

    (best_group, best_shared)
}

fn merge_leaf(
    groups: &mut [ClusterGroup],
    order: &[u32],
    target_partition_size: usize,
    max_partition_size: usize,
) {
    for &id in order.iter() {
        let id = id as usize;
        if groups[id].size == 0 || groups[id].size >= target_partition_size as u32 {
            continue;
        }

        let mut best_score = -1f32;
        let mut best_group = -1i32;

        for &other in order.iter() {
            let other = other as usize;
            if id == other || groups[other].size == 0 {
                continue;
            }

            if groups[id].size + groups[other].size > max_partition_size as u32 {
                continue;
            }

            // favor merging nearby groups
            let score = bounds_score(&groups[id], &groups[other]);

            if score > best_score {
                best_score = score;
                best_group = other as i32;
            }
        }

        // merge id *into* best_group; that way, we may merge more groups into the same
        // best_group, maximizing the chance of reaching target
        if best_group != -1 {
            let best_group = best_group as usize;

            // combine groups by linking them together
            let mut tail = best_group;
            while groups[tail].next >= 0 {
                tail = groups[tail].next as usize;
            }

            groups[tail].next = id as i32;

            // update group sizes; note, we omit vertices update for simplicity as it's not used
            // for spatial merge
            groups[best_group].size += groups[id].size;
            groups[id].size = 0;

            // merge bounding spheres
            let (target, source) = two_mut(groups, best_group, id);
            merge_bounds(target, source);
            groups[id].radius = 0.0;
        }
    }
}

fn merge_partition(order: &mut [u32], groups: &[ClusterGroup], axis: usize, pivot: f32) -> usize {
    let mut m = 0usize;

    // invariant: elements in range [0, m) are < pivot, elements in range [m, i) are >= pivot
    for i in 0..order.len() {
        let v = groups[order[i] as usize].center[axis];

        // swap(m, i) unconditionally
        order.swap(m, i);

        // when v >= pivot, we swap i with m without advancing it, preserving invariants
        m += usize::from(v < pivot);
    }

    m
}

fn merge_spatial(
    groups: &mut [ClusterGroup],
    order: &mut [u32],
    target_partition_size: usize,
    max_partition_size: usize,
    leaf_size: usize,
    depth: i32,
) {
    let count = order.len();
    let mut total = 0usize;
    for &id in order.iter() {
        total += groups[id as usize].size as usize;
    }

    if total <= max_partition_size || count <= leaf_size {
        return merge_leaf(groups, order, target_partition_size, max_partition_size);
    }

    let mut mean = [0f32; 3];
    let mut vars = [0f32; 3];
    let (mut runc, mut runs) = (1f32, 1f32);

    // gather statistics on the points in the subtree using Welford's algorithm
    for &id in order.iter() {
        let point = &groups[id as usize].center;

        for k in 0..3 {
            let delta = point[k] - mean[k];
            mean[k] += delta * runs;
            vars[k] += delta * (point[k] - mean[k]);
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
    let mut middle = merge_partition(order, groups, axis, split);

    // enforce balance for degenerate partitions
    // this also ensures recursion depth is bounded on pathological inputs
    if middle <= leaf_size / 2 || count - middle <= leaf_size / 2 || depth >= K_MERGE_DEPTH_CUTOFF {
        middle = count / 2;
    }

    // recursion depth is logarithmic and bounded due to max depth check above
    let (order_l, order_r) = order.split_at_mut(middle);
    merge_spatial(
        groups,
        order_l,
        target_partition_size,
        max_partition_size,
        leaf_size,
        depth + 1,
    );
    merge_spatial(
        groups,
        order_r,
        target_partition_size,
        max_partition_size,
        leaf_size,
        depth + 1,
    );
}

/// Upstream: `meshopt_partitionClusters`. `vertex_positions: None` skips spatial
/// scoring (upstream's NULL).
pub fn partition_clusters(
    destination: &mut [u32],
    cluster_indices: &[u32],
    cluster_index_counts: &[u32],
    vertex_positions: Option<&[f32]>,
    vertex_count: usize,
    vertex_positions_stride: usize,
    target_partition_size: usize,
) -> usize {
    let cluster_count = cluster_index_counts.len();
    debug_assert!(vertex_positions.is_none() || (12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
    debug_assert!(target_partition_size > 0);

    let max_partition_size = target_partition_size + target_partition_size / 3;

    let mut used = vec![0u8; vertex_count];

    let mut cluster_newindices = vec![0u32; cluster_indices.len()];
    let mut cluster_offsets = vec![0u32; cluster_count + 1];

    // make new cluster index list that filters out duplicate indices
    filter_cluster_indices(
        &mut cluster_newindices,
        &mut cluster_offsets,
        cluster_indices,
        cluster_index_counts,
        &mut used,
    );
    let cluster_indices = &cluster_newindices[..];

    // build cluster adjacency along with edge weights (shared vertex count)
    let adjacency = build_cluster_adjacency(
        cluster_indices,
        &cluster_offsets,
        cluster_count,
        vertex_count,
    );

    let mut groups = vec![
        ClusterGroup {
            group: 0,
            next: 0,
            size: 0,
            vertices: 0,
            center: [0.0; 3],
            radius: 0.0,
        };
        cluster_count
    ];

    let mut order: Vec<GroupOrder> = vec![GroupOrder { id: 0, order: 0 }; cluster_count];
    let mut pending = 0usize;

    let mut shared_acc = vec![0u32; cluster_count];

    let vertex_stride_float = vertex_positions_stride / 4;

    // create a singleton group for each cluster and order them by priority
    for i in 0..cluster_count {
        groups[i].group = i as i32;
        groups[i].next = -1;
        groups[i].size = 1;
        groups[i].vertices = cluster_offsets[i + 1] - cluster_offsets[i];
        debug_assert!(groups[i].vertices > 0);

        // compute bounding sphere for each cluster if positions are provided
        if let Some(positions) = vertex_positions {
            let (center, radius) = {
                let mut center = [0f32; 3];
                let radius = compute_cluster_bounds(
                    &cluster_indices[cluster_offsets[i] as usize..cluster_offsets[i + 1] as usize],
                    positions,
                    vertex_stride_float,
                    &mut center,
                );
                (center, radius)
            };
            groups[i].center = center;
            groups[i].radius = radius;
        }

        let item = GroupOrder {
            id: i as u32,
            order: groups[i].vertices as i32,
        };

        pending = heap_push(&mut order, pending, item);
    }

    // iteratively merge the smallest group with the best group
    while pending != 0 {
        let (top, new_pending) = heap_pop(&mut order, pending);
        pending = new_pending;

        // this group was merged into another group earlier
        if groups[top.id as usize].size == 0 {
            continue;
        }

        // disassociate clusters from the group to prevent them from being merged again; we will
        // re-associate them if the group is reinserted
        let mut i = top.id as i32;
        while i >= 0 {
            debug_assert!(groups[i as usize].group == top.id as i32);
            groups[i as usize].group = -1;
            i = groups[i as usize].next;
        }

        // the group is large enough, emit as is
        if groups[top.id as usize].size >= target_partition_size as u32 {
            continue;
        }

        let (best_group, best_shared) = pick_group_to_merge(
            &groups,
            top.id as usize,
            &adjacency,
            max_partition_size,
            vertex_positions.is_some(),
            &mut shared_acc,
        );

        // we can't grow the group any more, emit as is
        if best_group == -1 {
            continue;
        }
        let best_group = best_group as usize;

        // combine groups by linking them together
        let mut tail = top.id as usize;
        while groups[tail].next >= 0 {
            tail = groups[tail].next as usize;
        }

        groups[tail].next = best_group as i32;

        // update group sizes; note, the vertex update is a O(1) approximation which avoids
        // recomputing the true size
        groups[top.id as usize].size += groups[best_group].size;
        groups[top.id as usize].vertices += groups[best_group].vertices;
        groups[top.id as usize].vertices = if groups[top.id as usize].vertices > best_shared {
            groups[top.id as usize].vertices - best_shared
        } else {
            1
        };

        groups[best_group].size = 0;
        groups[best_group].vertices = 0;

        // merge bounding spheres if bounds are available
        if vertex_positions.is_some() {
            let (target, source) = two_mut(&mut groups, top.id as usize, best_group);
            merge_bounds(target, source);
            groups[best_group].radius = 0.0;
        }

        // re-associate all clusters back to the merged group
        let mut i = top.id as i32;
        while i >= 0 {
            groups[i as usize].group = top.id as i32;
            i = groups[i as usize].next;
        }

        let top = GroupOrder {
            order: groups[top.id as usize].vertices as i32,
            ..top
        };
        pending = heap_push(&mut order, pending, top);
    }

    // if vertex positions are provided, we do a final pass to see if we can merge small groups
    // based on spatial locality alone
    if vertex_positions.is_some() {
        let mut merge_order: Vec<u32> = vec![0u32; cluster_count];
        let mut merge_offset = 0usize;

        for (i, group) in groups.iter().enumerate() {
            if group.size != 0 {
                merge_order[merge_offset] = i as u32;
                merge_offset += 1;
            }
        }

        merge_spatial(
            &mut groups,
            &mut merge_order[..merge_offset],
            target_partition_size,
            max_partition_size,
            8, // leaf_size
            0,
        );
    }

    // output each remaining group
    let mut next_group = 0u32;

    for (i, group) in groups.iter().enumerate() {
        if group.size == 0 {
            continue;
        }

        let mut j = i as i32;
        while j >= 0 {
            destination[j as usize] = next_group;
            j = groups[j as usize].next;
        }

        next_group += 1;
    }

    debug_assert!(next_group as usize <= cluster_count);
    next_group as usize
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_partitionClusters.
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

        fn bytes(&mut self, n: usize) -> &'a [u8] {
            let s = &self.data[self.pos..self.pos + n];
            self.pos += n;
            s
        }

        fn vec_u32(&mut self, n: usize) -> Vec<u32> {
            (0..n).map(|_| self.u32()).collect()
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
    fn golden_partition_clusters() {
        const _LEN_CHECK: usize = 10056;
        let data = include_bytes!("../tests/data/partition.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let total_index_count = c.u32() as usize;
            let cluster_count = c.u32() as usize;
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let target = c.u32() as usize;
            let with_positions = c.u32() == 1;
            let counts = c.vec_u32(cluster_count);
            let indices = c.vec_u32(total_index_count);
            let positions = if with_positions {
                Some(c.f32s(vertex_count * stride))
            } else {
                None
            };
            let expected = c.vec_u32(cluster_count);
            let expected_result = c.u32() as usize;

            let mut destination = vec![0xeeu32; cluster_count];
            let result = partition_clusters(
                &mut destination,
                &indices,
                &counts,
                positions.as_deref(),
                vertex_count,
                stride,
                target,
            );
            assert_eq!(
                destination, expected,
                "partition cc={cluster_count} t={target}"
            );
            assert_eq!(
                result, expected_result,
                "partition result cc={cluster_count}"
            );
        }
    }
}
