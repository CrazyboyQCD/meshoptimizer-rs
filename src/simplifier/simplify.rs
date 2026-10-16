//! Quadric-error mesh simplification entry points. Upstream:
//! `meshopt_simplifyEdge`/`meshopt_simplify`/`meshopt_simplifyWithAttributes`/
//! `meshopt_simplifyWithUpdate` (simplifier.cpp L1230-2644).
//!
//! The `meshopt_SimplifyInternalDebug` visualization path is not ported: it is
//! unreachable from any public entry point upstream ("do not use").

use super::adjacency::{
    EdgeAdjacency, build_position_remap, build_sparse_remap, classify_vertices, finalize_vertices,
    rescale_attributes, rescale_positions,
};
use super::collapse::{
    AttributeData, Collapse, bound_edge_collapses, fill_fold_quadrics, perform_edge_collapses,
    pick_edge_collapses, rank_edge_collapses, remap_edge_loops, remap_index_buffer,
    sort_edge_collapses, update_quadrics,
};
use super::prune::{build_components, measure_components, prune_components};
use super::quadric::{
    K_MAX_ATTRIBUTES, Quadric, QuadricGrad, fill_attribute_quadrics, fill_edge_quadrics,
    fill_face_quadrics, fill_vertex_quadrics,
};
use super::solve::{solve_attributes, solve_positions};
use crate::math;
use alloc::vec;
use alloc::vec::Vec;

/// Upstream: `meshopt_SimplifyLockBorder`. Lock vertices on the geometric border.
pub const SIMPLIFY_LOCK_BORDER: u32 = 1 << 0;
/// Upstream: `meshopt_SimplifySparse`. Assume indices are a sparse subset of the mesh.
pub const SIMPLIFY_SPARSE: u32 = 1 << 1;
/// Upstream: `meshopt_SimplifyErrorAbsolute`. Treat error as absolute instead of relative.
pub const SIMPLIFY_ERROR_ABSOLUTE: u32 = 1 << 2;
/// Upstream: `meshopt_SimplifyPrune`. Remove disconnected components incrementally.
pub const SIMPLIFY_PRUNE: u32 = 1 << 3;
/// Upstream: `meshopt_SimplifyRegularize`. Produce more regular triangles at some quality cost.
pub const SIMPLIFY_REGULARIZE: u32 = 1 << 4;
/// Upstream: `meshopt_SimplifyPermissive`. Allow collapses across attribute discontinuities.
pub const SIMPLIFY_PERMISSIVE: u32 = 1 << 5;
/// Upstream: `meshopt_SimplifyRegularizeLight`. Milder variant of `SIMPLIFY_REGULARIZE`.
pub const SIMPLIFY_REGULARIZE_LIGHT: u32 = 1 << 6;
/// Upstream: `meshopt_SimplifyPreserveFolds`. Try to preserve fold lines (experimental upstream).
pub const SIMPLIFY_PRESERVE_FOLDS: u32 = 1 << 7;
/// Upstream: `meshopt_SimplifyErrorClamped`. Clamp attribute error to position error scale (experimental upstream).
pub const SIMPLIFY_ERROR_CLAMPED: u32 = 1 << 8;

/// Upstream: `meshopt_SimplifyVertex_Lock`. Do not move this vertex.
pub const SIMPLIFY_VERTEX_LOCK: u8 = 1 << 0;
/// Upstream: `meshopt_SimplifyVertex_Protect`. Protect attribute discontinuity (with `SIMPLIFY_PERMISSIVE`).
pub const SIMPLIFY_VERTEX_PROTECT: u8 = 1 << 1;
/// Upstream: `meshopt_SimplifyVertex_Priority`. Make this vertex more likely to be preserved.
pub const SIMPLIFY_VERTEX_PRIORITY: u8 = 1 << 2;

// internal option bit; set by simplifyWithUpdate (upstream anonymous enum)
pub(crate) const SIMPLIFY_INTERNAL_SOLVE: u32 = 1 << 29;

#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn simplify_edge(
    result: &mut [u32],
    index_count: usize,
    vertex_positions_data: &mut [f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_attributes_data: Option<&mut [f32]>,
    vertex_attributes_stride: usize,
    attribute_weights: &[f32],
    vertex_lock: Option<&[u8]>,
    target_index_count: usize,
    target_error: f32,
    options: u32,
    out_result_error: Option<&mut f32>,
) -> usize {
    debug_assert!(index_count.is_multiple_of(3));
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
    debug_assert!(target_index_count <= index_count);
    debug_assert!(target_error >= 0.0);
    debug_assert!(
        vertex_attributes_stride >= attribute_weights.len() * 4 && vertex_attributes_stride <= 256
    );
    debug_assert!(vertex_attributes_stride.is_multiple_of(4));
    debug_assert!(attribute_weights.len() <= K_MAX_ATTRIBUTES);
    for w in attribute_weights {
        debug_assert!(*w >= 0.0);
    }

    let mut attribute_count = 0usize;

    // build an index remap and update indices/vertex_count to minimize the subsequent work
    // note: as a consequence, errors will be computed relative to the subset extent
    let mut vertex_count_current = vertex_count;
    let mut sparse_remap: Option<Vec<u32>> = None;
    if options & SIMPLIFY_SPARSE != 0 {
        let (remap, unique) = build_sparse_remap(result, index_count, vertex_count);
        vertex_count_current = unique;
        sparse_remap = Some(remap);
    }
    let sparse = sparse_remap.as_deref();

    // build adjacency information
    let mut adjacency = EdgeAdjacency::prepare(index_count, vertex_count_current);
    adjacency.update(result, index_count, vertex_count_current, None);

    // build position remap that maps each vertex to the one with identical position
    // wedge table stores next vertex with identical position for each vertex
    let mut remap = vec![0u32; vertex_count_current];
    let mut wedge = vec![0u32; vertex_count_current];
    build_position_remap(
        &mut remap,
        Some(&mut wedge),
        vertex_positions_data,
        vertex_count_current,
        vertex_positions_stride,
        sparse,
    );

    // classify vertices; vertex kind determines collapse rules, see kCanCollapse
    let mut vertex_kind = vec![0u8; vertex_count_current];
    let mut edge_loop = vec![0u32; vertex_count_current];
    let mut loopback = vec![0u32; vertex_count_current];
    classify_vertices(
        &mut vertex_kind,
        &mut edge_loop,
        &mut loopback,
        vertex_count_current,
        &adjacency,
        &remap,
        &wedge,
        vertex_lock,
        sparse,
        options,
    );

    let mut vertex_positions = Vec::new();
    let mut vertex_offset = [0.0f32; 3];
    let vertex_scale = rescale_positions(
        Some(&mut vertex_positions),
        vertex_positions_data,
        vertex_count_current,
        vertex_positions_stride,
        sparse,
        Some(&mut vertex_offset),
    );

    let mut vertex_attributes: Vec<f32> = Vec::new();
    let mut attribute_remap = [0u32; K_MAX_ATTRIBUTES];

    if let Some(attrs) = vertex_attributes_data.as_deref() {
        // remap attributes to only include ones with weight > 0 to minimize memory/compute overhead for quadrics
        let mut attributes_used = 0usize;
        for (i, &w) in attribute_weights.iter().enumerate() {
            if w > 0.0 {
                attribute_remap[attributes_used] = i as u32;
                attributes_used += 1;
            }
        }

        attribute_count = attributes_used;

        if attribute_count > 0 {
            vertex_attributes = vec![0.0; vertex_count_current * attribute_count];
            rescale_attributes(
                &mut vertex_attributes,
                attrs,
                vertex_count_current,
                vertex_attributes_stride,
                attribute_weights,
                attribute_count,
                &attribute_remap,
                sparse,
            );
        }
    }

    let mut vertex_quadrics = vec![Quadric::ZERO; vertex_count_current];

    let mut attribute_quadrics: Vec<Quadric> = Vec::new();
    let mut attribute_gradients: Vec<QuadricGrad> = Vec::new();
    let mut volume_gradients: Vec<QuadricGrad> = Vec::new();

    if attribute_count > 0 {
        attribute_quadrics = vec![Quadric::ZERO; vertex_count_current];
        attribute_gradients = vec![QuadricGrad::ZERO; vertex_count_current * attribute_count];

        if options & SIMPLIFY_INTERNAL_SOLVE != 0 {
            volume_gradients = vec![QuadricGrad::ZERO; vertex_count_current];
        }
    }

    fill_face_quadrics(
        &mut vertex_quadrics,
        if volume_gradients.is_empty() {
            None
        } else {
            Some(&mut volume_gradients[..])
        },
        result,
        index_count,
        &vertex_positions,
        &remap,
    );
    fill_vertex_quadrics(
        &mut vertex_quadrics,
        &vertex_positions,
        vertex_count_current,
        &remap,
        vertex_lock,
        sparse,
        options,
    );
    fill_edge_quadrics(
        &mut vertex_quadrics,
        result,
        index_count,
        &vertex_positions,
        &remap,
        &vertex_kind,
        &edge_loop,
        &loopback,
    );

    if options & SIMPLIFY_PRESERVE_FOLDS != 0 {
        fill_fold_quadrics(
            &mut vertex_quadrics,
            &adjacency,
            &vertex_positions,
            vertex_count_current,
            &remap,
            &wedge,
        );
    }

    if attribute_count > 0 {
        fill_attribute_quadrics(
            &mut attribute_quadrics,
            &mut attribute_gradients,
            result,
            index_count,
            &vertex_positions,
            &vertex_attributes,
            attribute_count,
        );
    }

    let mut components: Vec<u32> = Vec::new();
    let mut component_errors: Vec<f32> = Vec::new();
    let mut component_count = 0usize;
    let mut component_nexterror = 0.0f32;

    if options & SIMPLIFY_PRUNE != 0 {
        let (built, count) = build_components(vertex_count_current, result, &remap);
        components = built;
        component_count = count;

        // overallocate for temporary use inside measureComponents
        component_errors = vec![0.0; component_count * 4];
        measure_components(
            &mut component_errors,
            component_count,
            &components,
            &vertex_positions,
            vertex_count_current,
        );

        component_nexterror = f32::MAX;
        for i in 0..component_count {
            component_nexterror = if component_nexterror > component_errors[i] {
                component_errors[i]
            } else {
                component_nexterror
            };
        }
    }

    let collapse_capacity =
        bound_edge_collapses(&adjacency, vertex_count_current, index_count, &vertex_kind);

    let mut edge_collapses = vec![
        Collapse {
            v0: 0,
            v1: 0,
            bidi: 0,
            error: 0.0
        };
        collapse_capacity
    ];
    let mut collapse_order = vec![0u32; collapse_capacity];
    let mut collapse_remap = vec![0u32; vertex_count_current];
    let mut collapse_locked = vec![0u8; vertex_count_current];

    let mut result_count = index_count;
    let mut result_error = 0.0f32;
    let mut vertex_error = 0.0f32;

    // target_error input is linear; we need to adjust it to match quadricError units
    let error_scale = if options & SIMPLIFY_ERROR_ABSOLUTE != 0 {
        vertex_scale
    } else {
        1.0
    };
    let error_limit = (target_error * target_error) / (error_scale * error_scale);

    while result_count > target_index_count {
        // note: throughout the simplification process adjacency structure reflects welded topology for result-in-progress
        adjacency.update(result, result_count, vertex_count_current, Some(&remap));

        let edge_collapse_count = pick_edge_collapses(
            &mut edge_collapses,
            result,
            result_count,
            &remap,
            &vertex_kind,
            &edge_loop,
            &loopback,
        );

        // no edges can be collapsed any more due to topology restrictions
        if edge_collapse_count == 0 {
            break;
        }

        let attribute_data = if attribute_count > 0 {
            Some(AttributeData {
                attributes: &vertex_attributes,
                quadrics: &attribute_quadrics,
                gradients: &attribute_gradients,
            })
        } else {
            None
        };

        rank_edge_collapses(
            &mut edge_collapses,
            edge_collapse_count,
            &vertex_positions,
            attribute_data.as_ref(),
            attribute_count,
            &vertex_quadrics,
            &remap,
            &wedge,
            &vertex_kind,
            &edge_loop,
            &loopback,
            options & SIMPLIFY_ERROR_CLAMPED != 0,
        );

        sort_edge_collapses(&mut collapse_order, &edge_collapses, edge_collapse_count);

        let triangle_collapse_goal = (result_count - target_index_count) / 3;

        for i in 0..vertex_count_current {
            collapse_remap[i] = i as u32;
        }

        for l in collapse_locked[..vertex_count_current].iter_mut() {
            *l = 0;
        }

        let collapses_performed = perform_edge_collapses(
            &mut collapse_remap,
            &mut collapse_locked,
            &edge_collapses,
            edge_collapse_count,
            &collapse_order,
            &remap,
            &wedge,
            &vertex_kind,
            &edge_loop,
            &loopback,
            &vertex_positions,
            &adjacency,
            triangle_collapse_goal,
            error_limit,
            &mut result_error,
        );

        // no edges can be collapsed any more due to hitting the error limit or triangle collapse limit
        if collapses_performed == 0 {
            break;
        }

        let volume_ref = if volume_gradients.is_empty() {
            None
        } else {
            Some(&mut volume_gradients[..])
        };
        let aq_ref = if attribute_count > 0 {
            Some(&mut attribute_quadrics[..])
        } else {
            None
        };
        let ag_ref = if attribute_count > 0 {
            Some(&mut attribute_gradients[..])
        } else {
            None
        };

        update_quadrics(
            &collapse_remap,
            vertex_count_current,
            &mut vertex_quadrics,
            volume_ref,
            aq_ref,
            ag_ref,
            attribute_count,
            &vertex_positions,
            &remap,
            &mut vertex_error,
        );

        // updateQuadrics will update vertex error if we use attributes, but if we don't then result_error and vertex_error are equivalent
        vertex_error = if attribute_count == 0 {
            result_error
        } else {
            vertex_error
        };

        // note: we update loops following edge collapses, but after this we might still have stale loop data
        // this can happen when a triangle with a loop edge gets collapsed along a non-loop edge
        // that works since a loop that points to a vertex that is no longer connected is not affecting collapse logic
        remap_edge_loops(&mut edge_loop, vertex_count_current, &collapse_remap);
        remap_edge_loops(&mut loopback, vertex_count_current, &collapse_remap);

        result_count = remap_index_buffer(result, result_count, &collapse_remap, &remap);

        if options & SIMPLIFY_PRUNE != 0
            && result_count > target_index_count
            && component_nexterror <= vertex_error
        {
            result_count = prune_components(
                result,
                result_count,
                &components,
                &component_errors,
                vertex_error,
                &mut component_nexterror,
            );
        }
    }

    // at this point, component_nexterror might be stale: component it references may have been removed through a series of edge collapses
    let mut component_nextstale = true;

    // we're done with the regular simplification but we're still short of the target; try pruning more aggressively towards error_limit
    while options & SIMPLIFY_PRUNE != 0
        && result_count > target_index_count
        && component_nexterror <= error_limit
    {
        let component_cutoff = if component_nexterror * 1.5 < error_limit {
            component_nexterror * 1.5
        } else {
            error_limit
        };

        // track maximum error in eligible components as we are increasing resulting error
        let mut component_maxerror = 0.0f32;
        for i in 0..component_count {
            if component_errors[i] > component_maxerror && component_errors[i] <= component_cutoff {
                component_maxerror = component_errors[i];
            }
        }

        let new_count = prune_components(
            result,
            result_count,
            &components,
            &component_errors,
            component_cutoff,
            &mut component_nexterror,
        );
        if new_count == result_count && !component_nextstale {
            break;
        }

        component_nextstale = false; // pruneComponents guarantees next error is up to date
        result_count = new_count;
        result_error = if result_error < component_maxerror {
            component_maxerror
        } else {
            result_error
        };
        vertex_error = if vertex_error < component_maxerror {
            component_maxerror
        } else {
            vertex_error
        };
    }

    // if solve is requested, update input buffers destructively from internal data
    if options & SIMPLIFY_INTERNAL_SOLVE != 0 {
        let mut vertex_update = vec![0u8; vertex_count_current];

        // limit quadric solve to vertices that are still used in the result
        for i in 0..result_count {
            let v = result[i];

            // mark the vertex for finalizeVertices and root vertex for solve*
            vertex_update[remap[v as usize] as usize] = 1;
            vertex_update[v as usize] = 1;
        }

        // edge adjacency may be stale as we haven't updated it after last series of edge collapses
        adjacency.update(result, result_count, vertex_count_current, Some(&remap));

        solve_positions(
            &mut vertex_positions,
            vertex_count_current,
            &vertex_quadrics,
            if volume_gradients.is_empty() {
                None
            } else {
                Some(&volume_gradients[..])
            },
            if attribute_count > 0 {
                Some(&attribute_quadrics[..])
            } else {
                None
            },
            if attribute_count > 0 {
                Some(&attribute_gradients[..])
            } else {
                None
            },
            attribute_count,
            &remap,
            &wedge,
            &adjacency,
            &vertex_kind,
            &vertex_update,
        );

        if attribute_count > 0 {
            solve_attributes(
                &vertex_positions,
                &mut vertex_attributes,
                vertex_count_current,
                &attribute_quadrics,
                &attribute_gradients,
                attribute_count,
                &remap,
                &wedge,
                &vertex_kind,
                &vertex_update,
            );
        }

        let mut empty_attributes: [f32; 0] = [];
        finalize_vertices(
            vertex_positions_data,
            vertex_positions_stride,
            vertex_attributes_data.unwrap_or(&mut empty_attributes),
            vertex_attributes_stride,
            attribute_weights,
            attribute_count,
            vertex_count_current,
            &vertex_positions,
            &vertex_attributes,
            sparse,
            &attribute_remap,
            vertex_scale,
            &vertex_offset,
            &vertex_kind,
            &vertex_update,
            vertex_lock,
        );
    }

    // convert resulting indices back into the dense space of the larger mesh
    if let Some(sparse) = sparse {
        for i in 0..result_count {
            result[i] = sparse[result[i] as usize];
        }
    }

    // result_error is quadratic; we need to remap it back to linear
    if let Some(re) = out_result_error {
        *re = math::sqrt(result_error) * error_scale;
    }

    result_count
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_simplify,
    // meshopt_simplifyWithAttributes, meshopt_simplifyWithUpdate.
    use super::*;
    use crate::simplifier::{simplify, simplify_with_attributes, simplify_with_update};
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

        fn bytes_vec(&mut self, n: usize) -> Vec<u8> {
            self.bytes(n).to_vec()
        }
    }

    #[test]
    fn option_flag_values_match_upstream() {
        assert_eq!(SIMPLIFY_LOCK_BORDER, 1);
        assert_eq!(SIMPLIFY_SPARSE, 2);
        assert_eq!(SIMPLIFY_ERROR_ABSOLUTE, 4);
        assert_eq!(SIMPLIFY_PRUNE, 8);
        assert_eq!(SIMPLIFY_REGULARIZE, 16);
        assert_eq!(SIMPLIFY_PERMISSIVE, 32);
        assert_eq!(SIMPLIFY_REGULARIZE_LIGHT, 64);
        assert_eq!(SIMPLIFY_PRESERVE_FOLDS, 128);
        assert_eq!(SIMPLIFY_ERROR_CLAMPED, 256);

        assert_eq!(SIMPLIFY_VERTEX_LOCK, 1);
        assert_eq!(SIMPLIFY_VERTEX_PROTECT, 2);
        assert_eq!(SIMPLIFY_VERTEX_PRIORITY, 4);
    }

    #[test]
    fn golden_simplify() {
        const _LEN_CHECK: usize = 104248;
        let data = include_bytes!("../../tests/data/sp_simplify.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let options = c.u32();
            let target_index_count = c.u32() as usize;
            let target_error = c.f32();
            let has_lock = c.u32();
            let positions = c.f32s(vertex_count * stride);
            let indices = c.vec_u32(index_count);
            // sp_simplify fixtures record the lock bytes for layout symmetry, but
            // meshopt_simplify itself takes no vertex_lock parameter
            let _lock = if has_lock == 1 {
                Some(c.bytes_vec(vertex_count))
            } else {
                None
            };

            let expected_count = c.u32() as usize;
            let expected_indices = c.vec_u32(expected_count);
            let expected_error = c.f32();

            let mut destination = vec![0xeeu32; index_count];
            let mut result_error = 0.0f32;
            let result = simplify(
                &mut destination,
                &indices,
                &positions,
                vertex_count,
                stride,
                target_index_count,
                target_error,
                options,
                Some(&mut result_error),
            );

            assert_eq!(result, expected_count, "simplify case {case} count");
            assert_eq!(
                &destination[..result],
                &expected_indices[..],
                "simplify case {case} indices"
            );
            assert_eq!(
                result_error, expected_error,
                "simplify case {case} error (options {options:#x})"
            );
            case += 1;
        }
        assert_eq!(case, 16);
    }

    #[test]
    fn golden_simplify_with_attributes() {
        const _LEN_CHECK: usize = 56049;
        let data = include_bytes!("../../tests/data/sp_attrs.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let pos_stride = c.u32() as usize;
            let attr_stride = c.u32() as usize;
            let attr_count = c.u32() as usize;
            let index_count = c.u32() as usize;
            let options = c.u32();
            let target_index_count = c.u32() as usize;
            let target_error = c.f32();
            let has_lock = c.u32();
            let positions = c.f32s(vertex_count * pos_stride);
            let attributes = c.f32s(vertex_count * attr_stride);
            let mut weights = Vec::new();
            for _ in 0..attr_count {
                weights.push(c.f32());
            }
            let indices = c.vec_u32(index_count);
            let lock = if has_lock == 1 {
                Some(c.bytes_vec(vertex_count))
            } else {
                None
            };

            let expected_count = c.u32() as usize;
            let expected_indices = c.vec_u32(expected_count);
            let expected_error = c.f32();

            let mut destination = vec![0xeeu32; index_count];
            let mut result_error = 0.0f32;
            let result = simplify_with_attributes(
                &mut destination,
                &indices,
                &positions,
                vertex_count,
                pos_stride,
                &attributes,
                attr_stride,
                &weights,
                lock.as_deref(),
                target_index_count,
                target_error,
                options,
                Some(&mut result_error),
            );

            assert_eq!(
                result, expected_count,
                "simplifyWithAttributes case {case} count"
            );
            assert_eq!(
                &destination[..result],
                &expected_indices[..],
                "simplifyWithAttributes case {case} indices"
            );
            assert_eq!(
                result_error, expected_error,
                "simplifyWithAttributes case {case} error (options {options:#x})"
            );
            case += 1;
        }
        assert_eq!(case, 7);
    }

    #[test]
    fn golden_simplify_with_update() {
        const _LEN_CHECK: usize = 24944;
        let data = include_bytes!("../../tests/data/sp_update.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let pos_stride = c.u32() as usize;
            let attr_stride = c.u32() as usize;
            let attr_count = c.u32() as usize;
            let index_count = c.u32() as usize;
            let options = c.u32();
            let target_index_count = c.u32() as usize;
            let target_error = c.f32();
            let has_lock = c.u32();
            let positions = c.f32s(vertex_count * pos_stride);
            let attributes = c.f32s(vertex_count * attr_stride);
            let mut weights = Vec::new();
            for _ in 0..attr_count {
                weights.push(c.f32());
            }
            let mut indices = c.vec_u32(index_count);
            let lock = if has_lock == 1 {
                Some(c.bytes_vec(vertex_count))
            } else {
                None
            };

            let expected_count = c.u32() as usize;
            let expected_indices = c.vec_u32(expected_count);
            let expected_error = c.f32();
            let expected_positions = c.f32s(vertex_count * pos_stride);
            let expected_attributes = c.f32s(vertex_count * attr_stride);

            let mut result_error = 0.0f32;
            let mut positions = positions;
            let mut attributes = attributes;
            let result = simplify_with_update(
                &mut indices,
                &mut positions,
                vertex_count,
                pos_stride,
                &mut attributes,
                attr_stride,
                &weights,
                lock.as_deref(),
                target_index_count,
                target_error,
                options,
                Some(&mut result_error),
            );

            assert_eq!(
                result, expected_count,
                "simplifyWithUpdate case {case} count"
            );
            assert_eq!(
                &indices[..result],
                &expected_indices[..],
                "simplifyWithUpdate case {case} indices"
            );
            assert_eq!(
                result_error, expected_error,
                "simplifyWithUpdate case {case} error"
            );
            assert_eq!(
                positions, expected_positions,
                "simplifyWithUpdate case {case} positions"
            );
            assert_eq!(
                attributes, expected_attributes,
                "simplifyWithUpdate case {case} attributes"
            );
            case += 1;
        }
        assert_eq!(case, 3);
    }
}
