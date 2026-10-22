//! Connected-component pruning for simplification. Upstream:
//! `meshopt_simplifyPrune` plus the component machinery shared with
//! `meshopt_simplifyEdge`'s `SimplifyPrune` option (simplifier.cpp
//! L1887-2039, L2779-2811).

use super::Vector3;
use super::adjacency::{build_position_remap, rescale_positions};
use alloc::vec;
use alloc::vec::Vec;

#[allow(unsafe_code)]
pub(crate) fn follow(parents: &mut [u32], index: u32) -> u32 {
    let mut index = index;

    // ADR-0014 erratum 2: audited unchecked-index site. SAFETY: every value ever stored
    // in `parents` is < `parents.len()` — callers initialize `components[i] = i` for
    // i < vertex_count, merge steps store only values previously read from `parents`
    // (follow returns such a value), and the entry indices come from the position-weld
    // table (remap values are vertex indices < vertex_count) or from a range-bound
    // loop counter. By induction every index used here is in bounds; the slice is
    // never reallocated and `parents` is the unique &mut borrow. The bounds check on
    // the chase loop sits on the dependent-load chain and costs ~17% of
    // meshopt_simplifyPrune (measured 2026-10-16, see bench/results baseline).
    unsafe {
        while index != *parents.get_unchecked(index as usize) {
            let parent = *parents.get_unchecked(index as usize);
            *parents.get_unchecked_mut(index as usize) = *parents.get_unchecked(parent as usize);
            index = parent;
        }
    }

    index
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn build_components(
    vertex_count: usize,
    indices: &[u32],
    remap: &[u32],
) -> (Vec<u32>, usize) {
    // collect allocates and fills in one pass — no pre-zeroing (upstream's raw
    // malloc writes the iota table without clearing either)
    let mut components: Vec<u32> = (0..vertex_count as u32).collect();

    // compute a unique (but not sequential!) index for each component via union-find
    const NEXT: [usize; 4] = [1, 2, 0, 1];

    for corners in indices.as_chunks::<3>().0 {
        for e in 0..3 {
            let i0 = corners[e];
            let i1 = corners[NEXT[e]];

            let r0 = remap[i0 as usize];
            let r1 = remap[i1 as usize];

            let r0 = follow(&mut components, r0);
            let r1 = follow(&mut components, r1);

            // merge components with larger indices into components with smaller indices
            // this guarantees that the root of the component is always the one with the smallest index
            if r0 != r1 {
                let (lo, hi) = if r0 < r1 { (r0, r1) } else { (r1, r0) };
                components[hi as usize] = lo;
            }
        }
    }

    // make sure each element points to the component root *before* we renumber the components
    for i in 0..vertex_count {
        if remap[i] == i as u32 {
            components[i] = follow(&mut components, i as u32);
        }
    }

    let mut next_component = 0u32;

    // renumber components using sequential indices
    // a sequential pass is sufficient because component root always has the smallest index
    // note: it is unsafe to use follow() in this pass because we're replacing component links with sequential indices inplace
    for i in 0..vertex_count {
        if remap[i] == i as u32 {
            let root = components[i];
            debug_assert!(root <= i as u32); // make sure we already computed the component for non-roots
            if root == i as u32 {
                components[i] = next_component;
                next_component += 1;
            } else {
                components[i] = components[root as usize];
            }
        } else {
            debug_assert!(remap[i] < i as u32); // make sure we already computed the component
            components[i] = components[remap[i] as usize];
        }
    }

    (components, next_component as usize)
}

pub(crate) fn measure_components(
    component_errors: &mut [f32],
    component_count: usize,
    components: &[u32],
    vertex_positions: &[Vector3],
    vertex_count: usize,
) {
    for e in component_errors[..component_count * 4].iter_mut() {
        *e = 0.0;
    }

    // compute approximate sphere center for each component as an average
    for i in 0..vertex_count {
        let c = components[i] as usize;
        debug_assert!(c < component_count);

        let v = vertex_positions[i]; // copy avoids aliasing issues

        component_errors[c * 4] += v.x;
        component_errors[c * 4 + 1] += v.y;
        component_errors[c * 4 + 2] += v.z;
        component_errors[c * 4 + 3] += 1.0; // weight
    }

    // complete the center computation, and reinitialize [3] as a radius
    for i in 0..component_count {
        let w = component_errors[i * 4 + 3];
        let iw = if w == 0.0 { 0.0 } else { 1.0 / w };

        component_errors[i * 4] *= iw;
        component_errors[i * 4 + 1] *= iw;
        component_errors[i * 4 + 2] *= iw;
        component_errors[i * 4 + 3] = 0.0; // radius
    }

    // compute squared radius for each component
    for i in 0..vertex_count {
        let c = components[i] as usize;

        let dx = vertex_positions[i].x - component_errors[c * 4];
        let dy = vertex_positions[i].y - component_errors[c * 4 + 1];
        let dz = vertex_positions[i].z - component_errors[c * 4 + 2];
        let r = dx * dx + dy * dy + dz * dz;

        component_errors[c * 4 + 3] = if component_errors[c * 4 + 3] < r {
            r
        } else {
            component_errors[c * 4 + 3]
        };
    }

    // we've used the output buffer as scratch space, so we need to move the results to proper indices
    // note: we keep the squared error to make it match quadric error metric
    for i in 0..component_count {
        component_errors[i] = component_errors[i * 4 + 3];
    }
}

pub(crate) fn prune_components(
    indices: &mut [u32],
    index_count: usize,
    components: &[u32],
    component_errors: &[f32],
    error_cutoff: f32,
    nexterror: &mut f32,
) -> usize {
    let mut write = 0usize;
    let mut min_error = f32::MAX;

    for i in (0..index_count).step_by(3) {
        let v0 = indices[i];
        let v1 = indices[i + 1];
        let v2 = indices[i + 2];
        let c = components[v0 as usize] as usize;
        debug_assert!(
            c == components[v1 as usize] as usize && c == components[v2 as usize] as usize
        );

        if component_errors[c] > error_cutoff {
            min_error = if min_error > component_errors[c] {
                component_errors[c]
            } else {
                min_error
            };

            indices[write] = v0;
            indices[write + 1] = v1;
            indices[write + 2] = v2;
            write += 3;
        }
    }

    // update next error with the smallest error of the remaining components
    *nexterror = min_error;
    write
}

/// Upstream: `meshopt_simplifyPrune`. Removes connected components whose
/// bounding-sphere error (squared, in mesh-relative space) is below
/// `target_error * target_error`.
pub fn simplify_prune(
    destination: &mut [u32],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    target_error: f32,
) -> usize {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
    debug_assert!(target_error >= 0.0);

    destination[..index_count].copy_from_slice(indices);

    // build position remap that maps each vertex to the one with identical position
    let mut remap = vec![0u32; vertex_count];
    build_position_remap(
        &mut remap,
        None,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        None,
    );

    let mut rescaled = Vec::new();
    rescale_positions(
        Some(&mut rescaled),
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        None,
        None,
    );

    let (components, component_count) = build_components(vertex_count, indices, &remap);

    // overallocate for temporary use inside measureComponents
    let mut component_errors = vec![0.0f32; component_count * 4];
    measure_components(
        &mut component_errors,
        component_count,
        &components,
        &rescaled,
        vertex_count,
    );

    let mut component_nexterror = 0.0f32;
    prune_components(
        destination,
        index_count,
        &components,
        &component_errors,
        target_error * target_error,
        &mut component_nexterror,
    )
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_simplifyPrune.
    use super::*;
    use alloc::vec;
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
    fn golden_simplify_prune() {
        const _LEN_CHECK: usize = 60056;
        let data = include_bytes!("../../tests/data/sp_prune.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let target_error = c.f32();
            let positions = c.f32s(vertex_count * stride);
            let indices = c.vec_u32(index_count);

            let expected_count = c.u32() as usize;
            let expected_indices = c.vec_u32(expected_count);

            let mut destination = vec![0xeeu32; index_count];
            let result = simplify_prune(
                &mut destination,
                &indices,
                &positions,
                vertex_count,
                stride,
                target_error,
            );

            assert_eq!(result, expected_count, "simplifyPrune case {case} count");
            assert_eq!(
                &destination[..result],
                &expected_indices[..],
                "simplifyPrune case {case} indices"
            );
            case += 1;
        }
        assert_eq!(case, 7);
    }

    fn v3(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x, y, z }
    }

    #[test]
    fn follow_path_halves() {
        // chain 0 -> 1 -> 2 -> 3 (parents point up)
        let mut parents = vec![1u32, 2, 3, 3];
        let root = follow(&mut parents, 0);
        assert_eq!(root, 3);
        // path halving links each visited node to its grandparent
        assert_eq!(parents[0], 2);
        assert_eq!(parents[1], 3);
    }

    #[test]
    fn build_components_two_islands() {
        // triangle (0,1,2) welded onto position-shared (3,4,5), far from (6,7,8)
        let indices = vec![0, 1, 2, 6, 7, 8];
        let remap = vec![0u32, 1, 2, 0, 1, 2, 6, 7, 8]; // 3..5 share positions with 0..2

        let (components, count) = build_components(9, &indices, &remap);
        assert_eq!(count, 2);
        assert_eq!(components, vec![0, 0, 0, 0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn measure_and_prune_two_islands() {
        // component A: small triangle around origin; component B: triangle far out
        let positions = vec![
            v3(0.0, 0.0, 0.0),
            v3(1.0, 0.0, 0.0),
            v3(0.0, 1.0, 0.0),
            v3(10.0, 10.0, 0.0),
            v3(11.0, 10.0, 0.0),
            v3(10.0, 11.0, 0.0),
        ];
        let indices = vec![0, 1, 2, 3, 4, 5];
        let remap = vec![0u32, 1, 2, 3, 4, 5];
        let (components, count) = build_components(6, &indices, &remap);
        assert_eq!(count, 2);

        let mut errors = vec![0.0f32; count * 4];
        measure_components(&mut errors, count, &components, &positions, 6);

        // component centers (1/3, 1/3, 0) / (10+1/3, 10+1/3, 0); squared radius = 5/9
        assert!((errors[0] - 5.0 / 9.0).abs() < 1e-6);
        assert!((errors[1] - 5.0 / 9.0).abs() < 1e-6);

        let mut indices_mut = indices;
        let index_count = indices_mut.len();
        let mut nexterror = 0.0f32;
        // cutoff below both radii keeps everything
        let kept = prune_components(
            &mut indices_mut,
            index_count,
            &components,
            &errors,
            0.1,
            &mut nexterror,
        );
        assert_eq!(kept, 6);
        assert!((nexterror - 5.0 / 9.0).abs() < 1e-6);

        // cutoff above both removes everything
        let kept = prune_components(
            &mut indices_mut,
            index_count,
            &components,
            &errors,
            1.0,
            &mut nexterror,
        );
        assert_eq!(kept, 0);
        assert_eq!(nexterror, f32::MAX);
    }
}
