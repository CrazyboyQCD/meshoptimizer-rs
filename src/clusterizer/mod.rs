//! Clusterization & meshlets (`clusterizer.cpp`, FR-12) plus the cluster/meshlet/
//! sphere bounds entry points from `meshletutils.cpp`.

mod bounds;
mod kdtree;
mod meshlets;
mod optimize;
mod spatial;

pub use bounds::{Bounds, compute_cluster_bounds, compute_meshlet_bounds, compute_sphere_bounds};
pub use meshlets::{
    Meshlet, build_meshlets, build_meshlets_bound, build_meshlets_flex, build_meshlets_scan,
};
pub use optimize::{extract_meshlet_indices, optimize_meshlet, optimize_meshlet_level};
pub use spatial::build_meshlets_spatial;
