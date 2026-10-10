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

mod indexanalyzer;
mod indexgenerator;
mod math;
mod overdrawoptimizer;
mod partition;
mod quantization;
mod rasterizer;
mod spatialorder;
mod stripifier;
mod vcacheoptimizer;
mod vfetchoptimizer;

pub use indexanalyzer::{
    VertexCacheStatistics, VertexFetchStatistics, analyze_vertex_cache, analyze_vertex_fetch,
};
pub use indexgenerator::*;
pub use overdrawoptimizer::optimize_overdraw;
pub use partition::partition_clusters;
pub use quantization::{compute_position_exponent, dequantize_half, quantize_float, quantize_half};
pub use rasterizer::{CoverageStatistics, OverdrawStatistics, analyze_coverage, analyze_overdraw};
pub use spatialorder::{spatial_cluster_points, spatial_sort_remap, spatial_sort_triangles};
pub use stripifier::{stripify, stripify_bound, unstripify, unstripify_bound};
pub use vcacheoptimizer::{
    optimize_vertex_cache, optimize_vertex_cache_fifo, optimize_vertex_cache_strip,
};
pub use vfetchoptimizer::{optimize_vertex_fetch, optimize_vertex_fetch_remap};

#[cfg(test)]
extern crate std;
