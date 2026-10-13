//! Meshlet building and optimization (`clusterizer.cpp` entry points plus the
//! optimize/extract helpers from `meshletutils.cpp`).

use alloc::vec::Vec;

/// Upstream counterpart: `meshopt_Meshlet`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Meshlet {
    pub vertex_offset: u32,
    pub triangle_offset: u32,
    pub vertex_count: u32,
    pub triangle_count: u32,
}

// We keep a limited number of seed triangles and add a few triangles per finished meshlet
const MESHLET_MAX_SEEDS: usize = 256;
const MESHLET_ADD_SEEDS: usize = 4;

// To avoid excessive recursion for malformed inputs, we limit the maximum depth of the tree
pub(crate) const MESHLET_MAX_TREE_DEPTH: i32 = 50;

use super::kdtree::{KDNode, kdtree_build, kdtree_nearest};

/// Upstream counterpart: `meshopt_buildMeshletsBound`.
pub fn build_meshlets_bound(
    index_count: usize,
    max_vertices: usize,
    max_triangles: usize,
) -> usize {
    debug_assert!(index_count.is_multiple_of(3));
    debug_assert!((3..=256).contains(&max_vertices));
    debug_assert!((1..=512).contains(&max_triangles));

    // meshlet construction is limited by max vertices and max triangles per meshlet
    // the worst case is that the input is an unindexed stream since this equally stresses both limits
    // note that we assume that in the worst case, we leave 2 vertices unpacked in each meshlet - if we have space for 3 we can pack any triangle
    let max_vertices_conservative = max_vertices - 2;
    let meshlet_limit_vertices = index_count.div_ceil(max_vertices_conservative);
    let meshlet_limit_triangles = (index_count / 3).div_ceil(max_triangles);

    meshlet_limit_vertices.max(meshlet_limit_triangles)
}

struct TriangleAdjacency {
    counts: Vec<u32>,
    offsets: Vec<u32>,
    data: Vec<u32>,
}

fn build_triangle_adjacency(indices: &[u32], vertex_count: usize) -> TriangleAdjacency {
    let face_count = indices.len() / 3;

    let mut counts = alloc::vec![0u32; vertex_count];
    let mut offsets = alloc::vec![0u32; vertex_count];
    let mut data = alloc::vec![0u32; indices.len()];

    // fill triangle counts
    for &v in indices {
        counts[v as usize] += 1;
    }

    // fill offset table
    let mut offset = 0u32;

    for i in 0..vertex_count {
        offsets[i] = offset;
        offset += counts[i];
    }

    debug_assert!(offset as usize == indices.len());

    // fill triangle data
    for f in 0..face_count {
        let a = indices[f * 3] as usize;
        let b = indices[f * 3 + 1] as usize;
        let c = indices[f * 3 + 2] as usize;

        data[offsets[a] as usize] = f as u32;
        offsets[a] += 1;
        data[offsets[b] as usize] = f as u32;
        offsets[b] += 1;
        data[offsets[c] as usize] = f as u32;
        offsets[c] += 1;
    }

    // fix offsets that have been disturbed by the previous pass
    for i in 0..vertex_count {
        debug_assert!(offsets[i] >= counts[i]);
        offsets[i] -= counts[i];
    }

    TriangleAdjacency {
        counts,
        offsets,
        data,
    }
}

fn build_triangle_adjacency_sparse(indices: &[u32], vertex_count: usize) -> TriangleAdjacency {
    let face_count = indices.len() / 3;

    // sparse mode can build adjacency more quickly by ignoring unused vertices, using a bit to mark visited vertices
    const SPARSE_SEEN: u32 = 1 << 31;
    debug_assert!(indices.len() < SPARSE_SEEN as usize);

    let mut counts = alloc::vec![0u32; vertex_count];
    let mut offsets = alloc::vec![0u32; vertex_count];
    let mut data = alloc::vec![0u32; indices.len()];

    // fill triangle counts
    for &v in indices {
        counts[v as usize] += 1;
    }

    // fill offset table; uses sparse_seen bit to tag visited vertices
    let mut offset = 0u32;

    for &v in indices {
        let c = &mut counts[v as usize];

        if *c & SPARSE_SEEN == 0 {
            offsets[v as usize] = offset;
            offset += *c;
            *c |= SPARSE_SEEN;
        }
    }

    debug_assert!(offset as usize == indices.len());

    // fill triangle data
    for f in 0..face_count {
        let a = indices[f * 3] as usize;
        let b = indices[f * 3 + 1] as usize;
        let c = indices[f * 3 + 2] as usize;

        data[offsets[a] as usize] = f as u32;
        offsets[a] += 1;
        data[offsets[b] as usize] = f as u32;
        offsets[b] += 1;
        data[offsets[c] as usize] = f as u32;
        offsets[c] += 1;
    }

    // fix offsets that have been disturbed by the previous pass
    // also fix counts (that were marked with sparse_seen by the first pass)
    for &v in indices {
        let c = &mut counts[v as usize];

        if *c & SPARSE_SEEN != 0 {
            *c &= !SPARSE_SEEN;

            debug_assert!(offsets[v as usize] >= *c);
            offsets[v as usize] -= *c;
        }
    }

    TriangleAdjacency {
        counts,
        offsets,
        data,
    }
}

pub(crate) fn clear_used(used: &mut [i16], indices: &[u32]) {
    // for sparse inputs, it's faster to only clear vertices referenced by the index buffer
    if used.len() <= indices.len() {
        used.fill(-1);
    } else {
        for &v in indices {
            used[v as usize] = -1;
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Cone {
    pub px: f32,
    pub py: f32,
    pub pz: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
}

impl Cone {
    pub const ZERO: Cone = Cone {
        px: 0.0,
        py: 0.0,
        pz: 0.0,
        nx: 0.0,
        ny: 0.0,
        nz: 0.0,
    };
}

fn get_meshlet_score(distance: f32, spread: f32, cone_weight: f32, expected_radius: f32) -> f32 {
    let cone = 1.0 - spread * cone_weight;
    let cone_clamped = if cone < 1e-3 { 1e-3 } else { cone };

    (1.0 + distance / expected_radius * (1.0 - cone_weight)) * cone_clamped
}

fn get_meshlet_cone(acc: Cone, triangle_count: usize) -> Cone {
    let mut result = acc;

    let center_scale = if triangle_count == 0 {
        0.0
    } else {
        1.0 / triangle_count as f32
    };

    result.px *= center_scale;
    result.py *= center_scale;
    result.pz *= center_scale;

    let axis_length = result.nx * result.nx + result.ny * result.ny + result.nz * result.nz;
    let axis_scale = if axis_length == 0.0 {
        0.0
    } else {
        1.0 / crate::math::sqrt(axis_length)
    };

    result.nx *= axis_scale;
    result.ny *= axis_scale;
    result.nz *= axis_scale;

    result
}

fn compute_triangle_cones(
    triangles: &mut [Cone],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_positions_stride: usize,
) -> f32 {
    let vertex_stride_float = vertex_positions_stride / 4;
    let face_count = indices.len() / 3;

    let mut mesh_area = 0.0f32;

    for i in 0..face_count {
        let a = indices[i * 3] as usize;
        let b = indices[i * 3 + 1] as usize;
        let c = indices[i * 3 + 2] as usize;

        let p0 = &vertex_positions[vertex_stride_float * a..];
        let p1 = &vertex_positions[vertex_stride_float * b..];
        let p2 = &vertex_positions[vertex_stride_float * c..];

        let p10 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let p20 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        let normalx = p10[1] * p20[2] - p10[2] * p20[1];
        let normaly = p10[2] * p20[0] - p10[0] * p20[2];
        let normalz = p10[0] * p20[1] - p10[1] * p20[0];

        let area = crate::math::sqrt(normalx * normalx + normaly * normaly + normalz * normalz);
        let invarea = if area == 0.0 { 0.0 } else { 1.0 / area };

        triangles[i] = Cone {
            px: (p0[0] + p1[0] + p2[0]) / 3.0,
            py: (p0[1] + p1[1] + p2[1]) / 3.0,
            pz: (p0[2] + p1[2] + p2[2]) / 3.0,
            nx: normalx * invarea,
            ny: normaly * invarea,
            nz: normalz * invarea,
        };

        mesh_area += area;
    }

    mesh_area
}

// port-shape: mirrors the upstream parameter list (ADR-0005)
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_meshlet(
    meshlet: &mut Meshlet,
    a: u32,
    b: u32,
    c: u32,
    used: &mut [i16],
    meshlets: &mut [Meshlet],
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    meshlet_offset: usize,
    max_vertices: usize,
    max_triangles: usize,
    split: bool,
) -> bool {
    let mut result = false;

    let used_extra = i32::from(used[a as usize] < 0)
        + i32::from(used[b as usize] < 0)
        + i32::from(used[c as usize] < 0);

    if meshlet.vertex_count as usize + used_extra as usize > max_vertices
        || meshlet.triangle_count as usize >= max_triangles
        || split
    {
        meshlets[meshlet_offset] = *meshlet;

        for j in 0..meshlet.vertex_count as usize {
            used[meshlet_vertices[meshlet.vertex_offset as usize + j] as usize] = -1;
        }

        meshlet.vertex_offset += meshlet.vertex_count;
        meshlet.triangle_offset += meshlet.triangle_count * 3;
        meshlet.vertex_count = 0;
        meshlet.triangle_count = 0;

        result = true;
    }

    // C's av/bv/cv are references into used[]: a branch writing the slot is visible
    // to the later branches when corners coincide (degenerate triangles)
    if used[a as usize] < 0 {
        used[a as usize] = meshlet.vertex_count as i16;
        meshlet_vertices[meshlet.vertex_offset as usize + meshlet.vertex_count as usize] = a;
        meshlet.vertex_count += 1;
    }

    if used[b as usize] < 0 {
        used[b as usize] = meshlet.vertex_count as i16;
        meshlet_vertices[meshlet.vertex_offset as usize + meshlet.vertex_count as usize] = b;
        meshlet.vertex_count += 1;
    }

    if used[c as usize] < 0 {
        used[c as usize] = meshlet.vertex_count as i16;
        meshlet_vertices[meshlet.vertex_offset as usize + meshlet.vertex_count as usize] = c;
        meshlet.vertex_count += 1;
    }

    let av = used[a as usize];
    let bv = used[b as usize];
    let cv = used[c as usize];

    meshlet_triangles[meshlet.triangle_offset as usize + meshlet.triangle_count as usize * 3] =
        av as u8;
    meshlet_triangles[meshlet.triangle_offset as usize + meshlet.triangle_count as usize * 3 + 1] =
        bv as u8;
    meshlet_triangles[meshlet.triangle_offset as usize + meshlet.triangle_count as usize * 3 + 2] =
        cv as u8;
    meshlet.triangle_count += 1;

    result
}

#[allow(clippy::too_many_arguments)]
fn get_neighbor_triangle(
    meshlet: &Meshlet,
    meshlet_cone: &Cone,
    meshlet_vertices: &[u32],
    indices: &[u32],
    adjacency: &TriangleAdjacency,
    triangles: &[Cone],
    live_triangles: &[u32],
    used: &[i16],
    meshlet_expected_radius: f32,
    cone_weight: f32,
) -> u32 {
    let mut best_triangle = u32::MAX;
    let mut best_priority = 5;
    let mut best_score = f32::MAX;

    for i in 0..meshlet.vertex_count as usize {
        let index = meshlet_vertices[meshlet.vertex_offset as usize + i] as usize;

        let neighbors = &adjacency.data[adjacency.offsets[index] as usize..];
        let neighbors_size = adjacency.counts[index] as usize;

        for &triangle in &neighbors[..neighbors_size] {
            let a = indices[triangle as usize * 3] as usize;
            let b = indices[triangle as usize * 3 + 1] as usize;
            let c = indices[triangle as usize * 3 + 2] as usize;

            let extra = i32::from(used[a] < 0) + i32::from(used[b] < 0) + i32::from(used[c] < 0);

            let priority = if extra == 0 {
                // triangles that don't add new vertices to meshlets are max. priority
                0
            } else if live_triangles[a] == 1 || live_triangles[b] == 1 || live_triangles[c] == 1 {
                // artificially increase the priority of dangling triangles as they're expensive to add to new meshlets
                1
            } else if i32::from(live_triangles[a] == 2)
                + i32::from(live_triangles[b] == 2)
                + i32::from(live_triangles[c] == 2)
                >= 2
            {
                // if two vertices have live count of 2, removing this triangle will make another triangle dangling which is good for overall flow
                1 + extra
            } else {
                // otherwise adjust priority to be after the above cases, 3 or 4 based on used[] count
                2 + extra
            };

            // since topology-based priority is always more important than the score, we can skip scoring in some cases
            if priority > best_priority {
                continue;
            }

            let tri_cone = &triangles[triangle as usize];

            let dx = tri_cone.px - meshlet_cone.px;
            let dy = tri_cone.py - meshlet_cone.py;
            let dz = tri_cone.pz - meshlet_cone.pz;
            let distance = crate::math::sqrt(dx * dx + dy * dy + dz * dz);
            let spread = tri_cone.nx * meshlet_cone.nx
                + tri_cone.ny * meshlet_cone.ny
                + tri_cone.nz * meshlet_cone.nz;

            let score = get_meshlet_score(distance, spread, cone_weight, meshlet_expected_radius);

            // note that topology-based priority is always more important than the score
            // this helps maintain reasonable effectiveness of meshlet data and reduces scoring cost
            if priority < best_priority || score < best_score {
                best_triangle = triangle;
                best_priority = priority;
                best_score = score;
            }
        }
    }

    best_triangle
}

#[allow(clippy::too_many_arguments)]
fn append_seed_triangles(
    seeds: &mut [u32],
    meshlet: &Meshlet,
    meshlet_vertices: &[u32],
    indices: &[u32],
    adjacency: &TriangleAdjacency,
    triangles: &[Cone],
    live_triangles: &[u32],
    cornerx: f32,
    cornery: f32,
    cornerz: f32,
) -> usize {
    let mut best_seeds = [u32::MAX; MESHLET_ADD_SEEDS];
    let mut best_live = [u32::MAX; MESHLET_ADD_SEEDS];
    let mut best_score = [f32::MAX; MESHLET_ADD_SEEDS];

    for i in 0..meshlet.vertex_count as usize {
        let index = meshlet_vertices[meshlet.vertex_offset as usize + i] as usize;

        let mut best_neighbor = u32::MAX;
        let mut best_neighbor_live = u32::MAX;

        // find the neighbor with the smallest live metric
        let neighbors = &adjacency.data[adjacency.offsets[index] as usize..];
        let neighbors_size = adjacency.counts[index] as usize;

        for &triangle in &neighbors[..neighbors_size] {
            let a = indices[triangle as usize * 3] as usize;
            let b = indices[triangle as usize * 3 + 1] as usize;
            let c = indices[triangle as usize * 3 + 2] as usize;

            let live = live_triangles[a] + live_triangles[b] + live_triangles[c];

            if live < best_neighbor_live {
                best_neighbor = triangle;
                best_neighbor_live = live;
            }
        }

        // add the neighbor to the list of seeds; the list is unsorted and the replacement criteria is approximate
        if best_neighbor == u32::MAX {
            continue;
        }

        let dx = triangles[best_neighbor as usize].px - cornerx;
        let dy = triangles[best_neighbor as usize].py - cornery;
        let dz = triangles[best_neighbor as usize].pz - cornerz;
        let best_neighbor_score = crate::math::sqrt(dx * dx + dy * dy + dz * dz);

        for j in 0..MESHLET_ADD_SEEDS {
            // non-strict comparison reduces the number of duplicate seeds (triangles adjacent to multiple vertices)
            if best_neighbor_live < best_live[j]
                || (best_neighbor_live == best_live[j] && best_neighbor_score <= best_score[j])
            {
                best_seeds[j] = best_neighbor;
                best_live[j] = best_neighbor_live;
                best_score[j] = best_neighbor_score;
                break;
            }
        }
    }

    // add surviving seeds to the meshlet
    let mut seed_count = 0usize;

    for &seed in &best_seeds {
        if seed != u32::MAX {
            seeds[seed_count] = seed;
            seed_count += 1;
        }
    }

    seed_count
}

fn prune_seed_triangles(seeds: &mut [u32], seed_count: usize, emitted_flags: &[u8]) -> usize {
    let mut result = 0usize;

    for i in 0..seed_count {
        let index = seeds[i];

        seeds[result] = index;
        result += usize::from(emitted_flags[index as usize] == 0);
    }

    result
}

fn select_seed_triangle(
    seeds: &[u32],
    indices: &[u32],
    triangles: &[Cone],
    live_triangles: &[u32],
    cornerx: f32,
    cornery: f32,
    cornerz: f32,
) -> u32 {
    let mut best_seed = u32::MAX;
    let mut best_live = u32::MAX;
    let mut best_score = f32::MAX;

    for &index in seeds {
        let a = indices[index as usize * 3] as usize;
        let b = indices[index as usize * 3 + 1] as usize;
        let c = indices[index as usize * 3 + 2] as usize;

        let live = live_triangles[a] + live_triangles[b] + live_triangles[c];
        let dx = triangles[index as usize].px - cornerx;
        let dy = triangles[index as usize].py - cornery;
        let dz = triangles[index as usize].pz - cornerz;
        let score = crate::math::sqrt(dx * dx + dy * dy + dz * dz);

        if live < best_live || (live == best_live && score < best_score) {
            best_seed = index;
            best_live = live;
            best_score = score;
        }
    }

    best_seed
}

/// Upstream counterpart: `meshopt_buildMeshletsFlex`.
#[allow(clippy::too_many_arguments)]
pub fn build_meshlets_flex(
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
    cone_weight: f32,
    split_factor: f32,
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
    debug_assert!((0.0..=1.0).contains(&cone_weight));
    debug_assert!(split_factor >= 0.0);

    let index_count = indices.len();
    if index_count == 0 {
        return 0;
    }

    let mut adjacency = if vertex_count > index_count && index_count < (1u32 << 31) as usize {
        build_triangle_adjacency_sparse(indices, vertex_count)
    } else {
        build_triangle_adjacency(indices, vertex_count)
    };

    // live triangle counts; note, we alias adjacency.counts as we remove triangles after emitting them so the counts always match
    let face_count = index_count / 3;

    let mut emitted_flags = alloc::vec![0u8; face_count];

    // for each triangle, precompute centroid & normal to use for scoring
    let mut triangles_cones = alloc::vec![Cone::ZERO; face_count];
    let mesh_area = compute_triangle_cones(
        &mut triangles_cones,
        indices,
        vertex_positions,
        vertex_positions_stride,
    );

    // assuming each meshlet is a square patch, expected radius is sqrt(expected area)
    let triangle_area_avg = if face_count == 0 {
        0.0
    } else {
        mesh_area / face_count as f32 * 0.5
    };
    let meshlet_expected_radius = crate::math::sqrt(triangle_area_avg * max_triangles as f32) * 0.5;

    // build a kd-tree for nearest neighbor lookup
    let mut kdindices = (0..face_count as u32).collect::<alloc::vec::Vec<u32>>();

    let mut nodes = alloc::vec![KDNode::zero(); face_count * 2];
    kdtree_build(
        0,
        &mut nodes,
        &triangles_cones,
        &mut kdindices,
        face_count,
        8, /* leaf_size */
        0,
    );

    // find a specific corner of the mesh to use as a starting point for meshlet flow
    let mut cornerx = f32::MAX;
    let mut cornery = f32::MAX;
    let mut cornerz = f32::MAX;

    for tri in &triangles_cones {
        cornerx = if cornerx > tri.px { tri.px } else { cornerx };
        cornery = if cornery > tri.py { tri.py } else { cornery };
        cornerz = if cornerz > tri.pz { tri.pz } else { cornerz };
    }

    // index of the vertex in the meshlet, -1 if the vertex isn't used
    let mut used = alloc::vec![0i16; vertex_count];
    clear_used(&mut used, indices);

    // initial seed triangle is the one closest to the corner
    let mut initial_seed = u32::MAX;
    let mut initial_score = f32::MAX;

    for (i, tri) in triangles_cones.iter().enumerate() {
        let dx = tri.px - cornerx;
        let dy = tri.py - cornery;
        let dz = tri.pz - cornerz;
        let score = crate::math::sqrt(dx * dx + dy * dy + dz * dz);

        if initial_seed == u32::MAX || score < initial_score {
            initial_seed = i as u32;
            initial_score = score;
        }
    }

    // seed triangles to continue meshlet flow
    let mut seeds = [0u32; MESHLET_MAX_SEEDS];
    let mut seed_count = 0usize;

    let mut meshlet = Meshlet {
        vertex_offset: 0,
        triangle_offset: 0,
        vertex_count: 0,
        triangle_count: 0,
    };
    let mut meshlet_offset = 0usize;

    let mut meshlet_cone_acc = Cone::ZERO;

    loop {
        let meshlet_cone = get_meshlet_cone(meshlet_cone_acc, meshlet.triangle_count as usize);

        // for the first triangle, we don't have a meshlet cone yet, so we use the initial seed
        // to continue the meshlet, we select an adjacent triangle based on connectivity and spatial scoring
        let mut best_triangle = if meshlet_offset == 0 && meshlet.triangle_count == 0 {
            initial_seed
        } else {
            get_neighbor_triangle(
                &meshlet,
                &meshlet_cone,
                meshlet_vertices,
                indices,
                &adjacency,
                &triangles_cones,
                &adjacency.counts,
                &used,
                meshlet_expected_radius,
                cone_weight,
            )
        };

        let mut split = false;

        // when we run out of adjacent triangles we need to switch to spatial search; we currently just pick the closest triangle irrespective of connectivity
        if best_triangle == u32::MAX {
            let position = [meshlet_cone.px, meshlet_cone.py, meshlet_cone.pz];
            let mut index = u32::MAX;
            let mut distance = f32::MAX;

            kdtree_nearest(
                &mut nodes,
                0,
                &triangles_cones,
                &emitted_flags,
                position,
                &mut index,
                &mut distance,
            );

            best_triangle = index;
            split = meshlet.triangle_count as usize >= min_triangles
                && split_factor > 0.0
                && distance > meshlet_expected_radius * split_factor;
        }

        if best_triangle == u32::MAX {
            break;
        }

        let best_extra = i32::from(used[indices[best_triangle as usize * 3] as usize] < 0)
            + i32::from(used[indices[best_triangle as usize * 3 + 1] as usize] < 0)
            + i32::from(used[indices[best_triangle as usize * 3 + 2] as usize] < 0);

        // if the best triangle doesn't fit into current meshlet, we re-select using seeds to maintain global flow
        if split
            || meshlet.vertex_count as usize + best_extra as usize > max_vertices
            || meshlet.triangle_count as usize >= max_triangles
        {
            seed_count = prune_seed_triangles(&mut seeds, seed_count, &emitted_flags);
            seed_count = if seed_count + MESHLET_ADD_SEEDS <= MESHLET_MAX_SEEDS {
                seed_count
            } else {
                MESHLET_MAX_SEEDS - MESHLET_ADD_SEEDS
            };
            seed_count += append_seed_triangles(
                &mut seeds[seed_count..],
                &meshlet,
                meshlet_vertices,
                indices,
                &adjacency,
                &triangles_cones,
                &adjacency.counts,
                cornerx,
                cornery,
                cornerz,
            );

            let best_seed = select_seed_triangle(
                &seeds[..seed_count],
                indices,
                &triangles_cones,
                &adjacency.counts,
                cornerx,
                cornery,
                cornerz,
            );

            // we may not find a valid seed triangle if the mesh is disconnected as seeds are based on adjacency
            best_triangle = if best_seed != u32::MAX {
                best_seed
            } else {
                best_triangle
            };
        }

        let a = indices[best_triangle as usize * 3];
        let b = indices[best_triangle as usize * 3 + 1];
        let c = indices[best_triangle as usize * 3 + 2];

        // add meshlet to the output; when the current meshlet is full we reset the accumulated bounds
        if append_meshlet(
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
        ) {
            meshlet_offset += 1;
            meshlet_cone_acc = Cone::ZERO;
        }

        // remove emitted triangle from adjacency data
        // this makes sure that we spend less time traversing these lists on subsequent iterations
        // live triangle counts are updated as a byproduct of these adjustments
        for k in 0..3usize {
            let index = indices[best_triangle as usize * 3 + k] as usize;

            let offset = adjacency.offsets[index] as usize;
            let neighbors_size = adjacency.counts[index] as usize;

            for i in 0..neighbors_size {
                if adjacency.data[offset + i] == best_triangle {
                    adjacency.data[offset + i] = adjacency.data[offset + neighbors_size - 1];
                    adjacency.counts[index] -= 1;
                    break;
                }
            }
        }

        // update aggregated meshlet cone data for scoring subsequent triangles
        let tri_cone = triangles_cones[best_triangle as usize];
        meshlet_cone_acc.px += tri_cone.px;
        meshlet_cone_acc.py += tri_cone.py;
        meshlet_cone_acc.pz += tri_cone.pz;
        meshlet_cone_acc.nx += tri_cone.nx;
        meshlet_cone_acc.ny += tri_cone.ny;
        meshlet_cone_acc.nz += tri_cone.nz;

        emitted_flags[best_triangle as usize] = 1;
    }

    if meshlet.triangle_count != 0 {
        meshlets[meshlet_offset] = meshlet;
        meshlet_offset += 1;
    }

    meshlet_offset
}

/// Upstream counterpart: `meshopt_buildMeshlets`.
#[allow(clippy::too_many_arguments)]
pub fn build_meshlets(
    meshlets: &mut [Meshlet],
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    max_vertices: usize,
    max_triangles: usize,
    cone_weight: f32,
) -> usize {
    build_meshlets_flex(
        meshlets,
        meshlet_vertices,
        meshlet_triangles,
        indices,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        max_vertices,
        max_triangles,
        max_triangles,
        cone_weight,
        0.0,
    )
}

/// Upstream counterpart: `meshopt_buildMeshletsScan`.
pub fn build_meshlets_scan(
    meshlets: &mut [Meshlet],
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    indices: &[u32],
    vertex_count: usize,
    max_vertices: usize,
    max_triangles: usize,
) -> usize {
    debug_assert!(indices.len().is_multiple_of(3));
    debug_assert!((3..=256).contains(&max_vertices));
    debug_assert!((1..=512).contains(&max_triangles));

    // index of the vertex in the meshlet, -1 if the vertex isn't used
    let mut used = alloc::vec![0i16; vertex_count];
    clear_used(&mut used, indices);

    let mut meshlet = Meshlet {
        vertex_offset: 0,
        triangle_offset: 0,
        vertex_count: 0,
        triangle_count: 0,
    };
    let mut meshlet_offset = 0usize;

    for i in (0..indices.len()).step_by(3) {
        let a = indices[i];
        let b = indices[i + 1];
        let c = indices[i + 2];

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
            false,
        ) as usize;
    }

    if meshlet.triangle_count != 0 {
        meshlets[meshlet_offset] = meshlet;
        meshlet_offset += 1;
    }

    meshlet_offset
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_buildMeshlets,
    // meshopt_buildMeshletsScan, meshopt_buildMeshletsBound,
    // meshopt_buildMeshletsFlex, meshopt_buildMeshletsSpatial,
    // meshopt_optimizeMeshlet, meshopt_optimizeMeshletLevel,
    // meshopt_extractMeshletIndices.
    use super::super::build_meshlets_spatial;
    use super::super::optimize::{
        extract_meshlet_indices, optimize_meshlet, optimize_meshlet_level,
    };
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
            f32::from_bits(self.u32())
        }

        fn bytes(&mut self, n: usize) -> Vec<u8> {
            let s = &self.data[self.pos..self.pos + n];
            self.pos += n;
            s.to_vec()
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

    fn zero_meshlets(n: usize) -> Vec<Meshlet> {
        vec![
            Meshlet {
                vertex_offset: 0,
                triangle_offset: 0,
                vertex_count: 0,
                triangle_count: 0,
            };
            n
        ]
    }

    #[test]
    fn golden_optimize_meshlet() {
        const _LEN_CHECK: usize = 4946;
        let data = include_bytes!("../../tests/data/ms_optimize.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let triangle_count = c.u32() as usize;
            let level = c.u32() as i32;
            let mut vertices = c.vec_u32(vertex_count);
            let mut triangles = c.bytes(triangle_count * 3);
            let expected_vertices = c.vec_u32(vertex_count);
            let expected_triangles = c.bytes(triangle_count * 3);

            if level == 0 {
                optimize_meshlet(&mut vertices, &mut triangles);
            } else {
                optimize_meshlet_level(&mut vertices, &mut triangles, level);
            }
            assert_eq!(
                vertices, expected_vertices,
                "optimize vertices (level {level})"
            );
            assert_eq!(
                triangles, expected_triangles,
                "optimize triangles (level {level})"
            );
        }
    }

    #[test]
    fn golden_extract_meshlet_indices() {
        const _LEN_CHECK: usize = 4335;
        let data = include_bytes!("../../tests/data/ms_extract.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let index_count = c.u32() as usize;
            let unique = c.u32() as usize;
            let indices = c.vec_u32(index_count);
            let expected_vertices = c.vec_u32(unique);
            let expected_triangles = c.bytes(index_count);

            let mut vertices = vec![0u32; 256];
            let mut triangles = vec![0u8; index_count];
            let result = extract_meshlet_indices(&mut vertices, &mut triangles, &indices);
            assert_eq!(result, unique, "extract unique mismatch (ic={index_count})");
            assert_eq!(&vertices[..unique], &expected_vertices[..]);
            assert_eq!(triangles, expected_triangles);
        }
    }

    #[test]
    fn golden_build_meshlets() {
        const _LEN_CHECK: usize = 53204;
        let data = include_bytes!("../../tests/data/ms_build.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let index_count = c.u32() as usize;
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let variant = c.u32();
            let max_vertices = c.u32() as usize;
            let min_triangles = c.u32() as usize;
            let max_triangles = c.u32() as usize;
            let weight = c.f32();
            let positions = c.f32s(vertex_count * stride);
            let indices = c.vec_u32(index_count);

            let bound = build_meshlets_bound(index_count, max_vertices, min_triangles);
            let mut meshlets = zero_meshlets(bound);
            let mut meshlet_vertices = vec![0u32; index_count];
            let mut meshlet_triangles = vec![0u8; index_count];

            let count = match variant {
                0 => build_meshlets(
                    &mut meshlets,
                    &mut meshlet_vertices,
                    &mut meshlet_triangles,
                    &indices,
                    &positions,
                    vertex_count,
                    stride,
                    max_vertices,
                    max_triangles,
                    weight,
                ),
                1 => build_meshlets_scan(
                    &mut meshlets,
                    &mut meshlet_vertices,
                    &mut meshlet_triangles,
                    &indices,
                    vertex_count,
                    max_vertices,
                    max_triangles,
                ),
                2 => build_meshlets_flex(
                    &mut meshlets,
                    &mut meshlet_vertices,
                    &mut meshlet_triangles,
                    &indices,
                    &positions,
                    vertex_count,
                    stride,
                    max_vertices,
                    min_triangles,
                    max_triangles,
                    weight,
                    1.0,
                ),
                _ => build_meshlets_spatial(
                    &mut meshlets,
                    &mut meshlet_vertices,
                    &mut meshlet_triangles,
                    &indices,
                    &positions,
                    vertex_count,
                    stride,
                    max_vertices,
                    min_triangles,
                    max_triangles,
                    weight,
                ),
            };

            let expected_count = c.u32() as usize;
            assert_eq!(count, expected_count, "meshlet count (variant {variant})");
            for m in &mut meshlets[..count] {
                let vertex_offset = c.u32();
                let triangle_offset = c.u32();
                let vc = c.u32();
                let tc = c.u32();
                assert_eq!(
                    (
                        m.vertex_offset,
                        m.triangle_offset,
                        m.vertex_count,
                        m.triangle_count
                    ),
                    (vertex_offset, triangle_offset, vc, tc),
                    "meshlet record (variant {variant})"
                );
            }
            let expected_vertices = c.vec_u32(index_count);
            let expected_triangles = c.bytes(index_count);
            assert_eq!(meshlet_vertices, expected_vertices, "meshlet vertices");
            assert_eq!(meshlet_triangles, expected_triangles, "meshlet triangles");
        }
    }
}
