//! Tangent space computation (FR-16, FR-18). Upstream: `meshopt_generateTangents`
//! / `meshopt_generateNormals` (tangentspace.cpp, entire TU), split along
//! algorithmic seams per design §4: shared vertex hashing and corner adjacency
//! plumbing here, `tangents` (generateTangents), `normals` (generateNormals).
//!
//! `optacos` is upstream's own polynomial approximation (max error ~5e-4); it is
//! part of the algorithm, not a stand-in for `libm` `acos`, and is ported
//! verbatim so outputs stay byte-identical.

mod normals;
mod tangents;

use crate::math;
use alloc::vec;
use alloc::vec::Vec;

pub use normals::generate_normals;
pub use tangents::generate_tangents;

/// Upstream: `meshopt_TangentCompatible`. Accumulate tangents without the
/// triangle-size weighting, matching the reference implementation.
pub const TANGENT_COMPATIBLE: u32 = 1 << 0;
/// Upstream: `meshopt_TangentZeroFallback`. Output zero tangents for vertices
/// only connected to degenerate triangles instead of `(1, 0, 0)`.
pub const TANGENT_ZERO_FALLBACK: u32 = 1 << 1;

const EMPTY_SLOT: u32 = u32::MAX;

fn face_index(indices: Option<&[u32]>, corner: usize) -> u32 {
    match indices {
        Some(ix) => ix[corner],
        None => corner as u32,
    }
}

fn hash_buckets4(count: usize) -> usize {
    let mut buckets = 1usize;
    while buckets < count + count / 4 {
        buckets *= 2;
    }

    buckets
}

fn hash_lookup4(
    table: &mut [u32],
    hash: u32,
    mut eq: impl FnMut(u32, u32) -> bool,
    key: u32,
) -> usize {
    let hashmod = table.len() - 1;
    let mut bucket = hash as usize & hashmod;
    let mut probe = 0usize;

    // termination: hash_buckets4 caps the load factor at 5/4 and quadratic probing
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

struct VertexHasherF<'a, const POSITION_ONLY: bool> {
    vertex_positions: &'a [f32],
    vertex_positions_stride: usize,
    vertex_normals: &'a [f32],
    vertex_normals_stride: usize,
    vertex_uvs: &'a [f32],
    vertex_uvs_stride: usize,
}

impl<const POSITION_ONLY: bool> VertexHasherF<'_, POSITION_ONLY> {
    fn hash(&self, index: u32) -> u32 {
        let p = self.vertex_positions_stride * index as usize;
        let mut x = self.vertex_positions[p].to_bits();
        let mut y = self.vertex_positions[p + 1].to_bits();
        let mut z = self.vertex_positions[p + 2].to_bits();

        // replace negative zero with zero
        x = if x == 0x8000_0000 { 0 } else { x };
        y = if y == 0x8000_0000 { 0 } else { y };
        z = if z == 0x8000_0000 { 0 } else { z };

        // scramble bits to make sure that integer coordinates have entropy in lower bits
        x ^= x >> 17;
        y ^= y >> 17;
        z ^= z >> 17;

        if POSITION_ONLY {
            return x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ z.wrapping_mul(83492791);
        }

        let n = self.vertex_normals_stride * index as usize;
        let t = self.vertex_uvs_stride * index as usize;

        let nx = self.vertex_normals[n].to_bits();
        let ny = self.vertex_normals[n + 1].to_bits();
        let nz = self.vertex_normals[n + 2].to_bits();
        let tu = self.vertex_uvs[t].to_bits();
        let tv = self.vertex_uvs[t + 1].to_bits();

        // mix in normal bits
        x ^= if nx == 0x8000_0000 { 0 } else { nx >> 15 };
        y ^= if ny == 0x8000_0000 { 0 } else { ny >> 15 };
        z ^= if nz == 0x8000_0000 { 0 } else { nz >> 15 };

        // collect texture coordinate bits (simplified -0 handling and scrambling)
        let mut w = (tu ^ tv) & 0x7fff_ffff;
        w ^= w >> 13;

        // Optimized Spatial Hashing for Collision Detection of Deformable Objects
        x.wrapping_mul(73856093)
            ^ y.wrapping_mul(19349663)
            ^ z.wrapping_mul(83492791)
            ^ w.wrapping_mul(50331653)
    }

    fn equal(&self, lhs: u32, rhs: u32) -> bool {
        let lp = self.vertex_positions_stride * lhs as usize;
        let rp = self.vertex_positions_stride * rhs as usize;
        let p = self.vertex_positions;

        if POSITION_ONLY {
            return p[lp] == p[rp] && p[lp + 1] == p[rp + 1] && p[lp + 2] == p[rp + 2];
        }

        let ln = self.vertex_normals_stride * lhs as usize;
        let lt = self.vertex_uvs_stride * lhs as usize;
        let rn = self.vertex_normals_stride * rhs as usize;
        let rt = self.vertex_uvs_stride * rhs as usize;

        p[lp] == p[rp]
            && p[lp + 1] == p[rp + 1]
            && p[lp + 2] == p[rp + 2]
            && self.vertex_normals[ln] == self.vertex_normals[rn]
            && self.vertex_normals[ln + 1] == self.vertex_normals[rn + 1]
            && self.vertex_normals[ln + 2] == self.vertex_normals[rn + 2]
            && self.vertex_uvs[lt] == self.vertex_uvs[rt]
            && self.vertex_uvs[lt + 1] == self.vertex_uvs[rt + 1]
    }
}

fn build_vertex_remap<const POSITION_ONLY: bool>(
    remap: &mut [u32],
    vertex_positions: &[f32],
    vertex_positions_stride: usize,
    vertex_normals: &[f32],
    vertex_normals_stride: usize,
    vertex_uvs: &[f32],
    vertex_uvs_stride: usize,
) {
    let vertex_hasher = VertexHasherF::<POSITION_ONLY> {
        vertex_positions,
        vertex_positions_stride: vertex_positions_stride / 4,
        vertex_normals,
        vertex_normals_stride: vertex_normals_stride / 4,
        vertex_uvs,
        vertex_uvs_stride: vertex_uvs_stride / 4,
    };

    let vertex_table_size = hash_buckets4(remap.len());
    let mut vertex_table = vec![EMPTY_SLOT; vertex_table_size];

    for i in 0..remap.len() {
        let index = i as u32;
        let entry = hash_lookup4(
            &mut vertex_table,
            vertex_hasher.hash(index),
            |lhs, rhs| vertex_hasher.equal(lhs, rhs),
            index,
        );

        if vertex_table[entry] == EMPTY_SLOT {
            vertex_table[entry] = index;
        }

        remap[index as usize] = vertex_table[entry];
    }
}

struct CornerAdjacency {
    offsets: Vec<u32>,
    data: Vec<u32>,
}

fn build_corner_adjacency(
    indices: Option<&[u32]>,
    index_count: usize,
    remap: &[u32],
    vertex_count: usize,
) -> CornerAdjacency {
    let mut offsets = vec![0u32; vertex_count + 1];
    let mut data = vec![0u32; index_count];

    let face_count = index_count / 3;

    // fill corner counts
    for i in 0..index_count {
        let v = remap[face_index(indices, i) as usize];
        offsets[v as usize + 1] += 1;
    }

    // fill offset table
    let mut offset = 0u32;

    for v in 0..vertex_count {
        let count = offsets[v + 1];
        offsets[v + 1] = offset;
        offset += count;
    }

    // fill corner data
    for i in 0..face_count {
        let a = remap[face_index(indices, i * 3) as usize] as usize;
        let b = remap[face_index(indices, i * 3 + 1) as usize] as usize;
        let c = remap[face_index(indices, i * 3 + 2) as usize] as usize;

        // encode corner index in the low 2 bits
        data[offsets[a + 1] as usize] = (i as u32) << 2;
        offsets[a + 1] += 1;
        data[offsets[b + 1] as usize] = (i as u32) << 2 | 1;
        offsets[b + 1] += 1;
        data[offsets[c + 1] as usize] = (i as u32) << 2 | 2;
        offsets[c + 1] += 1;
    }

    CornerAdjacency { offsets, data }
}

fn follow2(parents: &mut [u32], index: u32) -> u32 {
    let mut index = index;

    while index != parents[index as usize] {
        let parent = parents[index as usize];
        parents[index as usize] = parents[parent as usize];
        index = parent;
    }

    index
}

// upstream's literal 3.1415926, not f32::consts::PI; changing it would change the approximation
#[allow(clippy::approx_constant)]
fn optacos(x: f32) -> f32 {
    // approximation for acos(x) = sqrt(1 - abs(x)) * 2-degree polynomial, max error ~5e-4
    let mut ax = x.abs();
    ax = if ax > 1.0 { 1.0 } else { ax };
    let mut r = 1.570337f32 + ax * (-0.2053972f32 + ax * 0.05147786f32);
    r *= math::sqrt(1.0 - ax);
    if x < 0.0 { 3.1415926 - r } else { r }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangent_option_flag_values_match_upstream() {
        assert_eq!(TANGENT_COMPATIBLE, 1);
        assert_eq!(TANGENT_ZERO_FALLBACK, 2);
    }

    #[test]
    fn hash_buckets4_caps_load_factor() {
        assert_eq!(hash_buckets4(0), 1);
        assert_eq!(hash_buckets4(1), 1); // count + count/4 truncates: 1 + 0
        assert_eq!(hash_buckets4(2), 2);
        assert_eq!(hash_buckets4(8), 16);
        assert_eq!(hash_buckets4(9), 16);
    }

    #[test]
    fn follow2_halves_paths_to_root() {
        // chain 3 -> 2 -> 1 -> 0; follow2(3) flattens the chain towards the root
        let mut parents = alloc::vec![0u32, 0, 1, 2];
        assert_eq!(follow2(&mut parents, 3), 0);
        assert_eq!(parents[3], 1); // path halving: 3 now points at its grandparent
        assert_eq!(parents[2], 0);
    }

    #[test]
    #[allow(clippy::approx_constant)] // upstream's literal, asserted bit-exact
    fn optacos_matches_upstream_at_boundaries() {
        assert_eq!(optacos(1.0), 0.0);
        assert_eq!(optacos(0.0), 1.570337);
        assert_eq!(optacos(-1.0), 3.1415926);
        assert_eq!(optacos(2.0), 0.0); // clamped to 1 before evaluation
    }
}
