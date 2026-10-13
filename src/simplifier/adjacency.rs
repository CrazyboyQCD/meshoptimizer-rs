//! Edge adjacency, position remap hashing, and vertex classification for the
//! simplifier. Upstream: `simplifier.cpp` L34-651.

use super::simplify::{
    SIMPLIFY_LOCK_BORDER, SIMPLIFY_PERMISSIVE, SIMPLIFY_VERTEX_LOCK, SIMPLIFY_VERTEX_PROTECT,
};
use super::{EMPTY_SLOT, Vector3};
use alloc::vec;
use alloc::vec::Vec;

pub(crate) const KIND_MANIFOLD: u8 = 0; // not on an attribute seam, not on any boundary
pub(crate) const KIND_BORDER: u8 = 1; // not on an attribute seam, has exactly two open edges
pub(crate) const KIND_SEAM: u8 = 2; // on an attribute seam with exactly two attribute seam edges
pub(crate) const KIND_COMPLEX: u8 = 3; // none of the above; can move as long as all wedges move to the target vertex
pub(crate) const KIND_FRINGE: u8 = 4; // complex vertex on a geometric border; can only move to another fringe vertex
pub(crate) const KIND_LOCKED: u8 = 5; // none of the above; can't move

// manifold vertices can collapse onto anything
// border/seam vertices can collapse onto border/seam respectively, or locked
// complex vertices can collapse onto complex/locked
// fringe vertices can only collapse onto fringe vertices; border, manifold and complex vertices can collapse onto them
// a rule of thumb is that collapsing kind A into kind B preserves the kind B in the target vertex
// for example, while we could collapse Complex into Manifold, this would mean the target vertex isn't Manifold anymore
pub(crate) const K_CAN_COLLAPSE: [[u8; 6]; 6] = [
    [1, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 1, 1],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
];

// if a vertex is manifold or seam, adjoining edges are guaranteed to have an opposite edge
// note that for seam edges, the opposite edge isn't present in the attribute-based topology
// but is present if you consider a position-only mesh variant
// while many complex collapses have the opposite edge, since complex vertices collapse to the
// same wedge, keeping opposite edges separate improves the quality by considering both targets
pub(crate) const K_HAS_OPPOSITE: [[u8; 6]; 6] = [
    [1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
];

#[derive(Clone, Copy)]
pub(crate) struct Edge {
    pub(crate) next: u32,
    pub(crate) prev: u32,
}

pub(crate) struct EdgeAdjacency {
    pub(crate) offsets: Vec<u32>,
    pub(crate) data: Vec<Edge>,
}

impl EdgeAdjacency {
    pub(crate) fn prepare(index_count: usize, vertex_count: usize) -> Self {
        EdgeAdjacency {
            offsets: vec![0; vertex_count + 1],
            data: vec![Edge { next: 0, prev: 0 }; index_count],
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub(crate) fn update(
        &mut self,
        indices: &[u32],
        index_count: usize,
        vertex_count: usize,
        remap: Option<&[u32]>,
    ) {
        let face_count = index_count / 3;
        let data = &mut self.data;
        let offsets = &mut self.offsets[1..];

        // fill edge counts
        for offset in offsets[..vertex_count].iter_mut() {
            *offset = 0;
        }

        for &index in &indices[..index_count] {
            let v = remap.map_or(index, |r| r[index as usize]);
            debug_assert!(v < vertex_count as u32);

            offsets[v as usize] += 1;
        }

        // fill offset table
        let mut offset = 0u32;

        for o in offsets.iter_mut() {
            let count = *o;
            *o = offset;
            offset += count;
        }

        debug_assert_eq!(offset as usize, index_count);

        // fill edge data
        for tri in &indices.as_chunks::<3>().0[..face_count] {
            let [mut a, mut b, mut c] = *tri;

            if let Some(r) = remap {
                a = r[a as usize];
                b = r[b as usize];
                c = r[c as usize];
            }

            data[offsets[a as usize] as usize] = Edge { next: b, prev: c };
            offsets[a as usize] += 1;

            data[offsets[b as usize] as usize] = Edge { next: c, prev: a };
            offsets[b as usize] += 1;

            data[offsets[c as usize] as usize] = Edge { next: a, prev: b };
            offsets[c as usize] += 1;
        }

        // finalize offsets
        self.offsets[0] = 0;
        debug_assert_eq!(self.offsets[vertex_count] as usize, index_count);
    }
}

pub(crate) fn has_edge(adjacency: &EdgeAdjacency, a: u32, b: u32) -> bool {
    let count = (adjacency.offsets[a as usize + 1] - adjacency.offsets[a as usize]) as usize;
    let start = adjacency.offsets[a as usize] as usize;

    adjacency.data[start..start + count]
        .iter()
        .any(|e| e.next == b)
}

pub(crate) fn has_edge_remapped(
    adjacency: &EdgeAdjacency,
    a: u32,
    b: u32,
    remap: &[u32],
    wedge: &[u32],
) -> bool {
    let mut v = a;

    loop {
        let count = (adjacency.offsets[v as usize + 1] - adjacency.offsets[v as usize]) as usize;
        let start = adjacency.offsets[v as usize] as usize;

        for e in &adjacency.data[start..start + count] {
            if remap[e.next as usize] == remap[b as usize] {
                return true;
            }
        }

        v = wedge[v as usize];
        if v == a {
            break;
        }
    }

    false
}

pub(crate) fn hash_buckets2(count: usize) -> usize {
    let mut buckets = 1usize;
    while buckets < count + count / 4 {
        buckets *= 2;
    }

    buckets
}

pub(crate) fn hash_lookup2(
    table: &mut [u32],
    hash: u32,
    mut eq: impl FnMut(u32, u32) -> bool,
    key: u32,
) -> usize {
    let hashmod = table.len() - 1;
    let mut bucket = hash as usize & hashmod;
    let mut probe = 0usize;

    // termination: hash_buckets2 caps the load factor at 5/4 and quadratic probing
    // over a power-of-two table visits every slot, so an empty slot always exists
    loop {
        let item = table[bucket];

        if item == EMPTY_SLOT || eq(item, key) {
            return bucket;
        }

        // hash collision, quadratic probing
        bucket = (bucket + probe + 1) & hashmod;
        probe += 1;
    }
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn build_position_remap(
    remap: &mut [u32],
    wedge: Option<&mut [u32]>,
    vertex_positions_data: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    sparse_remap: Option<&[u32]>,
) {
    let vertex_stride_float = vertex_positions_stride / 4;

    let table_size = hash_buckets2(vertex_count);
    let mut table = vec![EMPTY_SLOT; table_size];

    // build forward remap: for each vertex, which other (canonical) vertex does it map to?
    // we use position equivalence for this, and remap vertices to other existing vertices
    let hash = |index: u32| -> u32 {
        let ri = sparse_remap.map_or(index, |r| r[index as usize]);
        let key = &vertex_positions_data[ri as usize * vertex_stride_float..];

        let mut x = key[0].to_bits();
        let mut y = key[1].to_bits();
        let mut z = key[2].to_bits();

        // replace negative zero with zero
        x = if x == 0x8000_0000 { 0 } else { x };
        y = if y == 0x8000_0000 { 0 } else { y };
        z = if z == 0x8000_0000 { 0 } else { z };

        // scramble bits to make sure that integer coordinates have entropy in lower bits
        x ^= x >> 17;
        y ^= y >> 17;
        z ^= z >> 17;

        // Optimized Spatial Hashing for Collision Detection of Deformable Objects
        x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ z.wrapping_mul(83492791)
    };

    let mut eq = |lhs: u32, rhs: u32| -> bool {
        let li = sparse_remap.map_or(lhs, |r| r[lhs as usize]) as usize;
        let ri = sparse_remap.map_or(rhs, |r| r[rhs as usize]) as usize;

        let lv = &vertex_positions_data[li * vertex_stride_float..];
        let rv = &vertex_positions_data[ri * vertex_stride_float..];

        lv[0] == rv[0] && lv[1] == rv[1] && lv[2] == rv[2]
    };

    for i in 0..vertex_count {
        let index = i as u32;
        let entry = hash_lookup2(&mut table, hash(index), &mut eq, index);

        if table[entry] == EMPTY_SLOT {
            table[entry] = index;
        }

        remap[index as usize] = table[entry];
    }

    let Some(wedge) = wedge else { return };

    // build wedge table: for each vertex, which other vertex is the next wedge that also maps to the same vertex?
    // entries in table form a (cyclic) wedge loop per vertex; for manifold vertices, wedge[i] == remap[i] == i
    for i in 0..vertex_count {
        wedge[i] = i as u32;
    }

    for i in 0..vertex_count {
        if remap[i] != i as u32 {
            let r = remap[i] as usize;

            let wr = wedge[r];
            wedge[i] = wr;
            wedge[r] = i as u32;
        }
    }
}

pub(crate) fn build_sparse_remap(
    indices: &mut [u32],
    index_count: usize,
    vertex_count: usize,
) -> (Vec<u32>, usize) {
    // use a bit set to compute the precise number of unique vertices
    let mut filter = vec![0u8; vertex_count.div_ceil(8)];

    let mut unique = 0usize;
    for &index in &indices[..index_count] {
        debug_assert!(index < vertex_count as u32);
        let bit = &mut filter[(index / 8) as usize];
        unique += if *bit & (1 << (index % 8)) == 0 { 1 } else { 0 };
        *bit |= 1 << (index % 8);
    }

    let mut remap = vec![0u32; unique];
    let mut offset = 0usize;

    // temporary map dense => sparse; the table stores original vertex indices so the
    // equality probe needs no access to the (concurrently written) remap; this yields
    // the same probe sequence as upstream's RemapHasher because hash(id) is injective
    let revremap_size = hash_buckets2(unique);
    let mut revremap = vec![EMPTY_SLOT; revremap_size]; // slot -> original vertex
    let mut dense = vec![0u32; revremap_size]; // slot -> dense id

    // fill remap, using revremap as a helper, and rewrite indices in the same pass
    for index in indices.iter_mut().take(index_count) {
        let original = *index;
        let hash = original.wrapping_mul(0x5bd1e995);
        let entry = hash_lookup2(&mut revremap, hash, |lhs, rhs| lhs == rhs, original);

        if revremap[entry] == EMPTY_SLOT {
            revremap[entry] = original;
            dense[entry] = offset as u32;
            remap[offset] = original;
            offset += 1;
        }

        *index = dense[entry];
    }

    debug_assert_eq!(offset, unique);
    (remap, offset)
}

// port-shape: mirrors the upstream parameter list (ADR-0005)
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn classify_vertices(
    result: &mut [u8],
    edge_loop: &mut [u32],
    loopback: &mut [u32],
    vertex_count: usize,
    adjacency: &EdgeAdjacency,
    remap: &[u32],
    wedge: &[u32],
    vertex_lock: Option<&[u8]>,
    sparse_remap: Option<&[u32]>,
    options: u32,
) {
    for i in 0..vertex_count {
        edge_loop[i] = EMPTY_SLOT;
        loopback[i] = EMPTY_SLOT;
    }

    // incoming & outgoing open edges: ~0u if no open edges, i if there are more than 1
    // note that this is the same data as required in loop[] arrays; loop[] data is only used for border/seam by default
    // in permissive mode we also use it to guide complex-complex collapses, so we fill it for all vertices
    // (upstream aliases openinc=loopback, openout=loop; the data doubles as edge loops afterwards)
    for i in 0..vertex_count {
        let vertex = i as u32;

        let count =
            (adjacency.offsets[vertex as usize + 1] - adjacency.offsets[vertex as usize]) as usize;
        let start = adjacency.offsets[vertex as usize] as usize;

        for e in &adjacency.data[start..start + count] {
            let target = e.next;

            if target == vertex {
                // degenerate triangles have two distinct edges instead of three, and the self edge
                // is bi-directional by definition; this can break border/seam classification by "closing"
                // the open edge from another triangle and falsely marking the vertex as manifold
                // instead we mark the vertex as having >1 open edges which turns it into locked/complex
                loopback[vertex as usize] = vertex;
                edge_loop[vertex as usize] = vertex;
            } else if !has_edge(adjacency, target, vertex) {
                loopback[target as usize] = if loopback[target as usize] == EMPTY_SLOT {
                    vertex
                } else {
                    target
                };
                edge_loop[vertex as usize] = if edge_loop[vertex as usize] == EMPTY_SLOT {
                    target
                } else {
                    vertex
                };
            }
        }
    }

    for i in 0..vertex_count {
        if remap[i] == i as u32 {
            if wedge[i] == i as u32 {
                // no attribute seam, need to check if it's manifold
                let openi = loopback[i];
                let openo = edge_loop[i];

                // note: we classify any vertices with no open edges as manifold
                // this is technically incorrect - if 4 triangles share an edge, we'll classify vertices as manifold
                // it's unclear if this is a problem in practice
                if openi == EMPTY_SLOT && openo == EMPTY_SLOT {
                    result[i] = KIND_MANIFOLD;
                } else if openi != EMPTY_SLOT
                    && openo != EMPTY_SLOT
                    && remap[openi as usize] == remap[openo as usize]
                    && openi != i as u32
                {
                    // classify half-seams as seams (the branch below would mis-classify them as borders)
                    // half-seam is a single vertex that connects to both vertices of a potential seam
                    // treating these as seams allows collapsing the "full" seam vertex onto them
                    result[i] = KIND_SEAM;
                } else if openi != i as u32 && openo != i as u32 {
                    result[i] = KIND_BORDER;
                } else {
                    result[i] = KIND_LOCKED;
                }
            } else if wedge[wedge[i] as usize] == i as u32 {
                // attribute seam; need to distinguish between Seam and Locked
                let w = wedge[i] as usize;
                let openiv = loopback[i];
                let openov = edge_loop[i];
                let openiw = loopback[w];
                let openow = edge_loop[w];

                // seam should have one open half-edge for each vertex, and the edges need to "connect" - point to the same vertex post-remap
                if openiv != EMPTY_SLOT
                    && openiv != i as u32
                    && openov != EMPTY_SLOT
                    && openov != i as u32
                    && openiw != EMPTY_SLOT
                    && openiw != w as u32
                    && openow != EMPTY_SLOT
                    && openow != w as u32
                    && remap[openiv as usize] == remap[openow as usize]
                    && remap[openov as usize] == remap[openiw as usize]
                    && remap[openiv as usize] != remap[openov as usize]
                {
                    result[i] = KIND_SEAM;
                } else {
                    // mismatched seam endpoints or disconnected seam; we don't have classification available
                    result[i] = KIND_LOCKED;
                }
            } else {
                // more than one vertex maps to this one; we don't have classification available
                result[i] = KIND_LOCKED;
            }
        } else {
            debug_assert!(remap[i] < i as u32);

            result[i] = result[remap[i] as usize];
        }
    }

    if options & SIMPLIFY_PERMISSIVE != 0 {
        for i in 0..vertex_count {
            if result[i] == KIND_SEAM || result[i] == KIND_LOCKED {
                if remap[i] != i as u32 {
                    // only process primary vertices; wedges will be updated to match the primary vertex
                    result[i] = result[remap[i] as usize];
                    continue;
                }

                let mut v = i as u32;

                // vertex_lock may protect any wedge, not just the primary vertex, so we switch to complex only if no wedges are protected
                let mut protect = false;
                loop {
                    let rv = sparse_remap.map_or(v, |r| r[v as usize]);
                    protect |=
                        vertex_lock.is_some_and(|l| l[rv as usize] & SIMPLIFY_VERTEX_PROTECT != 0);
                    v = wedge[v as usize];
                    if v == i as u32 {
                        break;
                    }
                }

                // check if any adjoining edge doesn't have an opposite edge (indicating vertex is on the border)
                let mut border = false;
                loop {
                    let count = (adjacency.offsets[v as usize + 1] - adjacency.offsets[v as usize])
                        as usize;
                    let start = adjacency.offsets[v as usize] as usize;

                    for e in &adjacency.data[start..start + count] {
                        border |= !has_edge_remapped(adjacency, e.next, v, remap, wedge);
                    }
                    v = wedge[v as usize];
                    if v == i as u32 {
                        break;
                    }
                }

                // border vertices with attribute discontinuities can only collapse along the border, so they need a separate tagging
                result[i] = if protect {
                    result[i]
                } else if border {
                    KIND_FRINGE
                } else {
                    KIND_COMPLEX
                };
            }
        }
    }

    if let Some(lock) = vertex_lock {
        // vertex_lock may lock any wedge, not just the primary vertex, so we need to lock the primary vertex and relock any wedges
        for i in 0..vertex_count {
            let ri = sparse_remap.map_or(i as u32, |r| r[i]) as usize;

            if lock[ri] & SIMPLIFY_VERTEX_LOCK != 0 {
                result[remap[i] as usize] = KIND_LOCKED;
            }
        }

        for i in 0..vertex_count {
            if result[remap[i] as usize] == KIND_LOCKED {
                result[i] = KIND_LOCKED;
            }
        }
    }

    if options & SIMPLIFY_LOCK_BORDER != 0 {
        for i in 0..vertex_count {
            if result[i] == KIND_BORDER || result[i] == KIND_FRINGE {
                result[i] = KIND_LOCKED;
            }
        }
    }
}

pub(crate) fn rescale_positions(
    mut result: Option<&mut [Vector3]>,
    vertex_positions_data: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    sparse_remap: Option<&[u32]>,
    mut out_offset: Option<&mut [f32; 3]>,
) -> f32 {
    let vertex_stride_float = vertex_positions_stride / 4;

    let mut minv = [f32::MAX; 3];
    let mut maxv = [-f32::MAX; 3];

    for i in 0..vertex_count {
        let ri = sparse_remap.map_or(i as u32, |r| r[i]) as usize;
        let v = &vertex_positions_data[ri * vertex_stride_float..];

        if let Some(r) = result.as_mut() {
            r[i] = Vector3 {
                x: v[0],
                y: v[1],
                z: v[2],
            };
        }

        for j in 0..3 {
            let vj = v[j];

            minv[j] = if minv[j] > vj { vj } else { minv[j] };
            maxv[j] = if maxv[j] < vj { vj } else { maxv[j] };
        }
    }

    let mut extent = 0.0f32;

    extent = if maxv[0] - minv[0] < extent {
        extent
    } else {
        maxv[0] - minv[0]
    };
    extent = if maxv[1] - minv[1] < extent {
        extent
    } else {
        maxv[1] - minv[1]
    };
    extent = if maxv[2] - minv[2] < extent {
        extent
    } else {
        maxv[2] - minv[2]
    };

    if let Some(r) = result.as_mut() {
        let scale = if extent == 0.0 { 0.0 } else { 1.0 / extent };

        for i in 0..vertex_count {
            r[i].x = (r[i].x - minv[0]) * scale;
            r[i].y = (r[i].y - minv[1]) * scale;
            r[i].z = (r[i].z - minv[2]) * scale;
        }
    }

    if let Some(offset) = out_offset.as_mut() {
        **offset = [minv[0], minv[1], minv[2]];
    }

    extent
}

// port-shape: mirrors the upstream parameter list (ADR-0005)
#[allow(clippy::too_many_arguments)]
pub(crate) fn rescale_attributes(
    result: &mut [f32],
    vertex_attributes_data: &[f32],
    vertex_count: usize,
    vertex_attributes_stride: usize,
    attribute_weights: &[f32],
    attribute_count: usize,
    attribute_remap: &[u32],
    sparse_remap: Option<&[u32]>,
) {
    let vertex_attributes_stride_float = vertex_attributes_stride / 4;

    for i in 0..vertex_count {
        let ri = sparse_remap.map_or(i as u32, |r| r[i]) as usize;

        for k in 0..attribute_count {
            let rk = attribute_remap[k] as usize;
            let a = vertex_attributes_data[ri * vertex_attributes_stride_float + rk];

            result[i * attribute_count + k] = a * attribute_weights[rk];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_vertices(
    vertex_positions_data: &mut [f32],
    vertex_positions_stride: usize,
    vertex_attributes_data: &mut [f32],
    vertex_attributes_stride: usize,
    attribute_weights: &[f32],
    attribute_count: usize,
    vertex_count: usize,
    vertex_positions: &[Vector3],
    vertex_attributes: &[f32],
    sparse_remap: Option<&[u32]>,
    attribute_remap: &[u32],
    vertex_scale: f32,
    vertex_offset: &[f32; 3],
    vertex_kind: &[u8],
    vertex_update: &[u8],
    vertex_lock: Option<&[u8]>,
) {
    let vertex_positions_stride_float = vertex_positions_stride / 4;
    let vertex_attributes_stride_float = vertex_attributes_stride / 4;

    for i in 0..vertex_count {
        if vertex_update[i] == 0 {
            continue;
        }

        let ri = sparse_remap.map_or(i as u32, |r| r[i]) as usize;

        // updating externally locked vertices is not allowed
        if vertex_lock.is_some_and(|l| l[ri] & SIMPLIFY_VERTEX_LOCK != 0) {
            continue;
        }

        // moving locked vertices may result in floating point drift
        if vertex_kind[i] != KIND_LOCKED {
            let p = &vertex_positions[i];
            let v = &mut vertex_positions_data[ri * vertex_positions_stride_float..];

            v[0] = p.x * vertex_scale + vertex_offset[0];
            v[1] = p.y * vertex_scale + vertex_offset[1];
            v[2] = p.z * vertex_scale + vertex_offset[2];
        }

        if attribute_count > 0 {
            let sa = &vertex_attributes[i * attribute_count..];
            let va = &mut vertex_attributes_data[ri * vertex_attributes_stride_float..];

            for k in 0..attribute_count {
                let rk = attribute_remap[k] as usize;

                va[rk] = sa[k] / attribute_weights[rk];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: none (internal adjacency build; pinned
    // through the simplifier fixtures).
    use super::*;
    use alloc::vec;

    fn quad_mesh() -> (Vec<u32>, Vec<f32>) {
        // two CCW triangles sharing edge (1,2); vertices 0-3
        (
            vec![0, 1, 2, 1, 3, 2],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        )
    }

    #[test]
    fn edge_adjacency_offsets() {
        let (indices, _) = quad_mesh();
        let mut adjacency = EdgeAdjacency::prepare(indices.len(), 4);
        adjacency.update(&indices, indices.len(), 4, None);

        // directed edge counts: v0 1, v1 2, v2 2, v3 1
        assert_eq!(adjacency.offsets, vec![0, 1, 3, 5, 6]);

        // v0: edge (0->1, prev 2); v1: edges (1->2, prev 0) and (1->3, prev 2)
        assert_eq!(adjacency.data[0].next, 1);
        assert_eq!(adjacency.data[0].prev, 2);
        assert_eq!(adjacency.data[1].next, 2);
        assert_eq!(adjacency.data[1].prev, 0);
        assert_eq!(adjacency.data[2].next, 3);
        assert_eq!(adjacency.data[2].prev, 2);
    }

    #[test]
    fn has_edge_variants() {
        let (indices, _) = quad_mesh();
        let mut adjacency = EdgeAdjacency::prepare(indices.len(), 4);
        adjacency.update(&indices, indices.len(), 4, None);

        assert!(has_edge(&adjacency, 0, 1));
        assert!(has_edge(&adjacency, 2, 1)); // shared edge (1,2) is present in both directions
        assert!(has_edge(&adjacency, 1, 2));
        assert!(!has_edge(&adjacency, 1, 0)); // border edge only exists one way here
        assert!(!has_edge(&adjacency, 0, 3));

        let remap = vec![0, 1, 2, 3];
        let wedge = vec![0, 1, 2, 3];
        assert!(has_edge_remapped(&adjacency, 0, 1, &remap, &wedge));
        assert!(!has_edge_remapped(&adjacency, 0, 3, &remap, &wedge));
    }

    #[test]
    fn position_remap_welds_duplicates() {
        // v3 duplicates v1; v2 is unique
        let positions = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            1.0, 0.0, 0.0, //
        ];
        let mut remap = vec![0u32; 4];
        let mut wedge = vec![0u32; 4];
        build_position_remap(&mut remap, Some(&mut wedge), &positions, 4, 12, None);

        assert_eq!(remap, vec![0, 1, 2, 1]);
        // wedge loop for position 1: 1 -> 3 -> 1
        assert_eq!(wedge[1], 3);
        assert_eq!(wedge[3], 1);
        assert_eq!(wedge[0], 0);
        assert_eq!(wedge[2], 2);
    }

    #[test]
    fn position_remap_negative_zero() {
        // -0.0 hashes as 0.0 but compares equal only to itself here; both vertices
        // have distinct x so no weld happens
        let mut positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        positions[0] = -0.0;

        let mut remap = vec![0u32; 2];
        build_position_remap(&mut remap, None, &positions, 2, 12, None);
        assert_eq!(remap, vec![0, 1]);

        // -0.0 vs 0.0 in the same slot welds (bit-normalized in the hash)
        let positions = vec![-0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut remap = vec![0u32; 2];
        build_position_remap(&mut remap, None, &positions, 2, 12, None);
        assert_eq!(remap, vec![0, 0]);
    }

    #[test]
    fn sparse_remap_compacts_used_vertices() {
        let mut indices = vec![7, 3, 7, 1];
        let index_count = indices.len();
        let (remap, unique) = build_sparse_remap(&mut indices, index_count, 8);

        assert_eq!(unique, 3);
        assert_eq!(remap, vec![7, 3, 1]);
        assert_eq!(indices, vec![0, 1, 0, 2]);
    }

    #[test]
    fn hash_buckets2_shapes() {
        assert_eq!(hash_buckets2(0), 1);
        // count 1 needs buckets >= 1 + 0 => already satisfied at 1
        assert_eq!(hash_buckets2(1), 1);
        assert_eq!(hash_buckets2(4), 8);
        // load factor 5/4: 8 count needs > 8+2 => 16
        assert_eq!(hash_buckets2(8), 16);
    }

    #[test]
    fn classify_quad_border_and_manifold() {
        // closed square from four boundary triangles: all vertices manifold
        let indices = vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 3, 2, 1];
        let positions = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            1.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, //
        ];
        let mut remap = vec![0u32; 4];
        let mut wedge = vec![0u32; 4];
        build_position_remap(&mut remap, Some(&mut wedge), &positions, 4, 12, None);

        let mut adjacency = EdgeAdjacency::prepare(indices.len(), 4);
        adjacency.update(&indices, indices.len(), 4, None);

        let mut kind = vec![0u8; 4];
        let mut edge_loop = vec![0u32; 4];
        let mut loopback = vec![0u32; 4];
        classify_vertices(
            &mut kind,
            &mut edge_loop,
            &mut loopback,
            4,
            &adjacency,
            &remap,
            &wedge,
            None,
            None,
            0,
        );

        assert_eq!(kind, vec![KIND_MANIFOLD; 4]);

        // open quad (two triangles): border ring
        let indices = vec![0, 1, 2, 0, 2, 3];
        let mut adjacency = EdgeAdjacency::prepare(indices.len(), 4);
        adjacency.update(&indices, indices.len(), 4, None);

        let mut kind = vec![0u8; 4];
        let mut edge_loop = vec![0u32; 4];
        let mut loopback = vec![0u32; 4];
        classify_vertices(
            &mut kind,
            &mut edge_loop,
            &mut loopback,
            4,
            &adjacency,
            &remap,
            &wedge,
            None,
            None,
            0,
        );

        assert_eq!(kind, vec![KIND_BORDER; 4]);

        // with lock border option all borders become locked
        classify_vertices(
            &mut kind,
            &mut edge_loop,
            &mut loopback,
            4,
            &adjacency,
            &remap,
            &wedge,
            None,
            None,
            SIMPLIFY_LOCK_BORDER,
        );
        assert_eq!(kind, vec![KIND_LOCKED; 4]);
    }

    #[test]
    fn rescale_positions_extent_and_offset() {
        let positions = vec![1.0, 2.0, 3.0, -1.0, -2.0, 5.0];
        let mut scaled = vec![
            Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            };
            2
        ];
        let mut offset = [0.0f32; 3];
        let extent = rescale_positions(
            Some(&mut scaled),
            &positions,
            2,
            12,
            None,
            Some(&mut offset),
        );

        assert_eq!(extent, 4.0);
        assert_eq!(offset, [-1.0, -2.0, 3.0]);
        assert_eq!(
            scaled[0],
            Vector3 {
                x: 0.5,
                y: 1.0,
                z: 0.0
            }
        );
        assert_eq!(
            scaled[1],
            Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.5
            }
        );

        // NULL result computes extent only
        let extent = rescale_positions(None, &positions, 2, 12, None, None);
        assert_eq!(extent, 4.0);
    }

    #[test]
    fn rescale_and_finalize_attributes_roundtrip() {
        // attribute stream: [a, b] per vertex at stride 8; weight for a = 0.5, b unused (0)
        let attributes = vec![2.0, 9.0, 4.0, 9.0];
        let weights = [0.5, 0.0];
        let attribute_remap = [0u32]; // only attribute 0 has weight > 0

        let mut scaled = vec![0.0f32; 2];
        rescale_attributes(
            &mut scaled,
            &attributes,
            2,
            8,
            &weights,
            1,
            &attribute_remap,
            None,
        );
        assert_eq!(scaled, vec![1.0, 2.0]);

        // finalize divides by weight and writes back at the remapped slot
        let mut out_attributes = vec![0.0f32; 4];
        let positions = vec![
            Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            };
            2
        ];
        let mut out_positions = vec![0.0f32; 6];
        let kind = [KIND_MANIFOLD, KIND_MANIFOLD];
        let update = [1u8, 1];
        finalize_vertices(
            &mut out_positions,
            12,
            &mut out_attributes,
            8,
            &weights,
            1,
            2,
            &positions,
            &scaled,
            None,
            &attribute_remap,
            1.0,
            &[0.0, 0.0, 0.0],
            &kind,
            &update,
            None,
        );
        assert_eq!(out_attributes[0], 2.0);
        assert_eq!(out_attributes[2], 4.0);
    }
}
