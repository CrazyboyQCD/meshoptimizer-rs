//! Safe, deterministic Rust port of the
//! [meshoptimizer](https://github.com/zeux/meshoptimizer) library, pinned to upstream
//! v1.3 (commit `9e1f07b1`).
//!
//! The crate builds in two configurations: with the default `std` feature (platform
//! math) and with `--no-default-features` for `no_std` + `alloc` targets (`libm` math).
//! See `docs/srs.md` for the requirement set, `docs/design.md` for the design,
//! `docs/tdd.md` for the test-driven process, and `docs/adr/` for decision records.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod clusterizer;
mod indexanalyzer;
mod indexcodec;
mod indexgenerator;
mod math;
mod meshletcodec;
mod meshletutils;
mod opacitymap;
mod overdrawoptimizer;
mod partition;
mod quantization;
mod rasterizer;
mod simplifier;
mod spatialorder;
mod stripifier;
mod tangentspace;
mod vcacheoptimizer;
mod vertexcodec;
mod vertexfilter;
mod vfetchoptimizer;

pub use clusterizer::{
    Bounds, Meshlet, build_meshlets, build_meshlets_bound, build_meshlets_flex,
    build_meshlets_scan, build_meshlets_spatial, compute_cluster_bounds, compute_meshlet_bounds,
    compute_sphere_bounds, extract_meshlet_indices, optimize_meshlet, optimize_meshlet_level,
};
pub use indexanalyzer::{
    VertexCacheStatistics, VertexFetchStatistics, analyze_vertex_cache, analyze_vertex_fetch,
};
pub use indexcodec::{
    decode_index_buffer, decode_index_sequence, decode_index_version, encode_index_buffer,
    encode_index_buffer_bound, encode_index_sequence, encode_index_sequence_bound,
};
pub use indexgenerator::*;
pub use meshletcodec::{decode_meshlet, decode_meshlet_raw, encode_meshlet, encode_meshlet_bound};
pub use opacitymap::{
    opacity_map_compact, opacity_map_entry_size, opacity_map_measure, opacity_map_rasterize,
};
pub use overdrawoptimizer::optimize_overdraw;
pub use partition::partition_clusters;
pub use quantization::{compute_position_exponent, dequantize_half, quantize_float, quantize_half};
pub use rasterizer::{CoverageStatistics, OverdrawStatistics, analyze_coverage, analyze_overdraw};
pub use simplifier::{
    SIMPLIFY_ERROR_ABSOLUTE, SIMPLIFY_ERROR_CLAMPED, SIMPLIFY_LOCK_BORDER, SIMPLIFY_PERMISSIVE,
    SIMPLIFY_PRESERVE_FOLDS, SIMPLIFY_PRUNE, SIMPLIFY_REGULARIZE, SIMPLIFY_REGULARIZE_LIGHT,
    SIMPLIFY_SPARSE, SIMPLIFY_VERTEX_LOCK, SIMPLIFY_VERTEX_PRIORITY, SIMPLIFY_VERTEX_PROTECT,
    simplify, simplify_points, simplify_prune, simplify_scale, simplify_sloppy,
    simplify_with_attributes, simplify_with_update,
};
pub use spatialorder::{spatial_cluster_points, spatial_sort_remap, spatial_sort_triangles};
pub use stripifier::{stripify, stripify_bound, unstripify, unstripify_bound};
pub use tangentspace::{
    TANGENT_COMPATIBLE, TANGENT_ZERO_FALLBACK, generate_normals, generate_tangents,
};
pub use vcacheoptimizer::{
    optimize_vertex_cache, optimize_vertex_cache_fifo, optimize_vertex_cache_strip,
};
pub use vertexcodec::{
    decode_vertex_buffer, decode_vertex_version, encode_vertex_buffer, encode_vertex_buffer_bound,
    encode_vertex_buffer_level,
};
pub use vertexfilter::{
    EncodeExpMode, decode_filter_color, decode_filter_exp, decode_filter_oct, decode_filter_quat,
    encode_filter_color, encode_filter_exp, encode_filter_oct, encode_filter_quat,
};
pub use vfetchoptimizer::{optimize_vertex_fetch, optimize_vertex_fetch_remap};

#[cfg(test)]
extern crate std;
