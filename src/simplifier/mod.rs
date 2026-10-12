//! `simplifier.cpp` port (FR-09), split along algorithmic seams per design §4:
//! `adjacency` (edge adjacency + position remap + vertex classification),
//! `quadric` (quadric construction/evaluation/solve), `collapse` (collapse
//! selection/ranking/execution + fold quadrics + post-pass remapping),
//! `solve` (quadric position/attribute solving), `simplify` (collapse
//! scheduling loop + public entry points), `sloppy` (grid-based simplification
//! for indexed meshes), `points` (point-cloud simplification), `prune`
//! (connected-component pruning).

mod adjacency;
mod collapse;
mod points;
mod prune;
mod quadric;
mod simplify;
mod sloppy;
mod solve;

/// Internal position triple mirroring upstream `meshopt::Vector3`; field order
/// and f32 width are load-bearing for quadric math parity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Vector3 {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
}

/// Shared empty-slot sentinel for the simplifier's open-addressing tables
/// (upstream `~0u`).
pub(crate) const EMPTY_SLOT: u32 = u32::MAX;

pub use points::simplify_points;
pub use prune::simplify_prune;
pub use simplify::{
    SIMPLIFY_ERROR_ABSOLUTE, SIMPLIFY_ERROR_CLAMPED, SIMPLIFY_LOCK_BORDER, SIMPLIFY_PERMISSIVE,
    SIMPLIFY_PRESERVE_FOLDS, SIMPLIFY_PRUNE, SIMPLIFY_REGULARIZE, SIMPLIFY_REGULARIZE_LIGHT,
    SIMPLIFY_SPARSE, SIMPLIFY_VERTEX_LOCK, SIMPLIFY_VERTEX_PRIORITY, SIMPLIFY_VERTEX_PROTECT,
};
pub use sloppy::{simplify_scale, simplify_sloppy};

use simplify::{SIMPLIFY_INTERNAL_SOLVE, simplify_edge};

/// Upstream: `meshopt_simplify`. Reduces the triangle count towards
/// `target_index_count`, stopping early on topology restrictions or when the
/// quadric error exceeds `target_error` (relative to mesh extents unless
/// `SIMPLIFY_ERROR_ABSOLUTE`).
// port-shape: mirrors the upstream parameter list (ADR-0005)
#[allow(clippy::too_many_arguments)]
pub fn simplify(
    destination: &mut [u32],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    target_index_count: usize,
    target_error: f32,
    options: u32,
    result_error: Option<&mut f32>,
) -> usize {
    debug_assert!(options & SIMPLIFY_INTERNAL_SOLVE == 0); // use simplify_with_update instead

    let index_count = indices.len();
    let mut result = indices.to_vec();

    // the driver writes back moved vertices only for the WithUpdate path; the
    // clone keeps the caller's buffer read-only as upstream's const contract
    let mut positions = vertex_positions.to_vec();

    let written = simplify_edge(
        &mut result,
        index_count,
        &mut positions,
        vertex_count,
        vertex_positions_stride,
        None,
        0,
        &[],
        None,
        target_index_count,
        target_error,
        options,
        result_error,
    );

    destination.copy_from_slice(&result);

    written
}

/// Upstream: `meshopt_simplifyWithAttributes`. Like `simplify`, but collapses
/// also weigh attribute error (`attribute_weights`; zero weights are skipped)
/// and `vertex_lock` can lock/protect/prioritize vertices.
#[allow(clippy::too_many_arguments)]
pub fn simplify_with_attributes(
    destination: &mut [u32],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_attributes: &[f32],
    vertex_attributes_stride: usize,
    attribute_weights: &[f32],
    vertex_lock: Option<&[u8]>,
    target_index_count: usize,
    target_error: f32,
    options: u32,
    result_error: Option<&mut f32>,
) -> usize {
    debug_assert!(options & SIMPLIFY_INTERNAL_SOLVE == 0); // use simplify_with_update instead

    let index_count = indices.len();
    let mut result = indices.to_vec();
    let mut positions = vertex_positions.to_vec();
    let mut attributes = vertex_attributes.to_vec();

    let written = simplify_edge(
        &mut result,
        index_count,
        &mut positions,
        vertex_count,
        vertex_positions_stride,
        Some(&mut attributes),
        vertex_attributes_stride,
        attribute_weights,
        vertex_lock,
        target_index_count,
        target_error,
        options,
        result_error,
    );

    destination.copy_from_slice(&result);

    written
}

/// Upstream: `meshopt_simplifyWithUpdate`. Simplifies in place: `indices` is
/// rewritten with the result and positions/attributes are destructively
/// updated for vertices that moved during simplification.
#[allow(clippy::too_many_arguments)]
pub fn simplify_with_update(
    indices: &mut [u32],
    vertex_positions: &mut [f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_attributes: &mut [f32],
    vertex_attributes_stride: usize,
    attribute_weights: &[f32],
    vertex_lock: Option<&[u8]>,
    target_index_count: usize,
    target_error: f32,
    options: u32,
    result_error: Option<&mut f32>,
) -> usize {
    let index_count = indices.len();

    simplify_edge(
        indices,
        index_count,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        Some(vertex_attributes),
        vertex_attributes_stride,
        attribute_weights,
        vertex_lock,
        target_index_count,
        target_error,
        options | SIMPLIFY_INTERNAL_SOLVE,
        result_error,
    )
}
