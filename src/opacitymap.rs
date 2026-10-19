//! Opacity micromaps (FR-15). Upstream: `meshopt_opacityMapMeasure`,
//! `meshopt_opacityMapEntrySize`, `meshopt_opacityMapRasterize`,
//! `meshopt_opacityMapCompact` (opacitymap.cpp, entire TU).
//!
//! The TU is self-contained: it brings its own bilinear texture sampler
//! (`sampleTexture`) rather than depending on `rasterizer.cpp`.

use crate::math;
use crate::quantization::cast_i32;
use alloc::vec;

const EMPTY_SLOT: u32 = u32::MAX;

/// Upstream: `meshopt_opacityMapMeasure`. Deduplicates triangle rasterization
/// requests by quantized UV and assigns each triangle a subdivision level.
///
/// `indices.len()` is the index count and must be a multiple of 3;
/// `omm_indices.len() == indices.len() / 3`; `levels` and `sources` must be at
/// least as long as the number of unique triangles (at most
/// `indices.len() / 3`). `vertex_uvs_stride` is in bytes (8..=256, multiple of
/// 4). `max_level` must be in 0..=12, `target_edge >= 0`; texture dimensions
/// are at most 16384. Returns the unique OMM count.
#[allow(clippy::too_many_arguments)] // port-shape: mirrors the upstream parameter list (ADR-0005)
pub fn opacity_map_measure(
    levels: &mut [u8],
    sources: &mut [u32],
    omm_indices: &mut [i32],
    indices: &[u32],
    vertex_uvs: &[f32],
    vertex_count: usize,
    vertex_uvs_stride: usize,
    texture_width: u32,
    texture_height: u32,
    max_level: i32,
    target_edge: f32,
) -> usize {
    let _ = vertex_count; // only used by upstream asserts
    let vertex_stride_float = vertex_uvs_stride / 4;
    let texture_area = texture_width as f32 * texture_height as f32;

    let triangle_count = indices.len() / 3;

    // hash map used to deduplicate triangle rasterization requests based on UV
    let table_size = hash_buckets3(triangle_count);
    let mut table = vec![EMPTY_SLOT; table_size];
    let mut triangles = vec![
        TriangleOmm {
            uvs: [0; 6],
            level: 0,
        };
        triangle_count
    ];

    let mut result = 0usize;

    for (t, &[a, b, c]) in indices.as_chunks::<3>().0.iter().enumerate() {
        let u0 = vertex_uvs[a as usize * vertex_stride_float];
        let v0 = vertex_uvs[a as usize * vertex_stride_float + 1];
        let u1 = vertex_uvs[b as usize * vertex_stride_float];
        let v1 = vertex_uvs[b as usize * vertex_stride_float + 1];
        let u2 = vertex_uvs[c as usize * vertex_stride_float];
        let v2 = vertex_uvs[c as usize * vertex_stride_float + 1];

        let mut level = max_level;

        if target_edge > 0.0 {
            // compute ratio of edge length (in texels) to target and determine subdivision level
            let uvarea = ((u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)).abs() * 0.5 * texture_area;
            let ratio = math::sqrt(uvarea) / target_edge;
            let levelf = math::log2(if ratio > 1.0 { ratio } else { 1.0 });

            // round to nearest and clamp
            level = cast_i32(levelf + 0.5);
            level = level.clamp(0, max_level);
        }

        // deduplicate rasterization requests based on UV
        let su0 = quantize_subpixel(u0, texture_width);
        let sv0 = quantize_subpixel(v0, texture_height);
        let su1 = quantize_subpixel(u1, texture_width);
        let sv1 = quantize_subpixel(v1, texture_height);
        let su2 = quantize_subpixel(u2, texture_width);
        let sv2 = quantize_subpixel(v2, texture_height);

        let tri = TriangleOmm {
            uvs: [su0, sv0, su1, sv1, su2, sv2],
            level,
        };
        triangles[result] = tri; // speculatively write triangle data to give hasher a way to compare it

        let hash = |index: u32| -> u32 {
            let tri = &triangles[index as usize];

            // upstream reinterprets the int[6] array as native-endian bytes
            let mut bytes = [0u8; 24];
            for (i, v) in tri.uvs.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }

            hash_update4u(tri.level as u32, &bytes)
        };
        let equal = |lhs: u32, rhs: u32| -> bool {
            let lt = &triangles[lhs as usize];
            let rt = &triangles[rhs as usize];

            lt.level == rt.level && lt.uvs == rt.uvs
        };

        let entry = hash_lookup3(&table, hash, equal, result as u32, EMPTY_SLOT);

        if table[entry] == EMPTY_SLOT {
            table[entry] = result as u32;
            levels[result] = level as u8;
            sources[result] = t as u32;
            result += 1;
        }

        omm_indices[t] = table[entry] as i32;
    }

    result
}

/// Upstream: `meshopt_opacityMapEntrySize`. Returns the encoded size in bytes
/// of a level-`level` opacity micromap with `states` states (2 or 4);
/// `level` must be in 0..=12.
pub fn opacity_map_entry_size(level: i32, states: i32) -> usize {
    get_level_size(level, states)
}

/// Upstream: `meshopt_opacityMapRasterize`. Rasterizes the opacity micromap for
/// one triangle by recursively subdividing it and sampling `texture_data`.
///
/// `result` must be at least [`opacity_map_entry_size(level, states)`] bytes
/// long; only that prefix is written (zero-filled first). `texture_stride` is
/// the number of bytes per texel (1..=4) and `texture_pitch` is the row pitch
/// (`>= texture_stride * texture_width`); `uv0`/`uv1`/`uv2` are `[u, v]`
/// pairs.
#[allow(clippy::too_many_arguments)]
pub fn opacity_map_rasterize(
    result: &mut [u8],
    level: i32,
    states: i32,
    uv0: &[f32],
    uv1: &[f32],
    uv2: &[f32],
    texture_data: &[u8],
    texture_stride: usize,
    texture_pitch: usize,
    texture_width: u32,
    texture_height: u32,
) {
    let size = get_level_size(level, states);
    result[..size].fill(0);

    let texture = Texture {
        data: texture_data,
        stride: texture_stride,
        pitch: texture_pitch,
        width: texture_width,
        height: texture_height,
        widthf: texture_width as i32 as f32 * 256.0,
        heightf: texture_height as i32 as f32 * 256.0,
    };

    // determine number of edge samples for conservative state estimation
    let texture_area = texture_width as i32 as f32 * texture_height as i32 as f32;
    let uvarea = ((uv1[0] - uv0[0]) * (uv2[1] - uv0[1]) - (uv2[0] - uv0[0]) * (uv1[1] - uv0[1]))
        .abs()
        * 0.5
        * texture_area;
    let uvedge = math::sqrt(uvarea) / (1 << level) as f32;

    // target ~2px distance between edge samples (assuming equilateral microtriangles)
    let edgeres = cast_i32(uvedge * 0.75);
    let edgeres = edgeres.clamp(0, 7);

    // rasterize all micro triangles recursively, passing corner data down to reduce redundant sampling
    let c0 = [uv0[0], uv0[1], sample_texture(&texture, uv0[0], uv0[1])];
    let c1 = [uv1[0], uv1[1], sample_texture(&texture, uv1[0], uv1[1])];
    let c2 = [uv2[0], uv2[1], sample_texture(&texture, uv2[0], uv2[1])];

    if states == 2 {
        rasterize_opacity_rec::<2>(result, 0, level, edgeres, &c0, &c1, &c2, &texture);
    } else {
        rasterize_opacity_rec::<4>(result, 0, level, edgeres, &c0, &c1, &c2, &texture);
    }
}

/// Upstream: `meshopt_opacityMapCompact`. Deduplicates opacity micromap data in
/// place and converts entries whose micro-triangles all share one state into
/// special indices (`-1`..`-4`).
///
/// `data.len()` is the data size, `offsets.len()` is the OMM count (`levels`
/// must be the same length), and `omm_indices.len()` is the triangle count.
/// Each `offsets[i]` plus the entry size for `levels[i]` must fit within
/// `data`. Returns the compacted OMM count.
pub fn opacity_map_compact(
    data: &mut [u8],
    levels: &mut [u8],
    offsets: &mut [u32],
    omm_indices: &mut [i32],
    states: i32,
) -> usize {
    let omm_count = offsets.len();

    let data_old = data.to_vec();

    // hash map used to deduplicate opacity micromap entries based on contents
    let table_size = hash_buckets3(omm_count);
    let mut table = vec![EMPTY_SLOT; table_size];

    let mut remap = vec![0i32; omm_count];

    let mut next = 0usize;
    let mut offset = 0usize;

    for i in 0..omm_count {
        let level = levels[i] as i32;

        let old = &data_old[offsets[i] as usize..];
        let size = get_level_size(level, states);

        // try to convert to a special index if all micro-triangle states are the same
        let special = get_special_index(old, level, states);
        if special < 0 {
            remap[i] = special;
            continue;
        }

        // speculatively write data to give hasher a way to compare it
        data[offset..offset + size].copy_from_slice(&old[..size]);
        offsets[next] = offset as u32;
        levels[next] = level as u8;

        let hash = |index: u32| -> u32 {
            let index = index as usize;
            let key = &data[offsets[index] as usize..];
            let size = get_level_size(levels[index] as i32, states);

            let mut h = levels[index] as u32;

            // MurmurHash2 for large keys, simple fold for small; note that size is a power of two
            if size < 4 {
                h ^= key[0] as u32 | (key[size - 1] as u32) << 8;
            } else {
                h = hash_update4u(h, &key[..size]);
            }

            // MurmurHash2 finalizer
            h ^= h >> 13;
            h = h.wrapping_mul(0x5bd1_e995);
            h ^= h >> 15;
            h
        };
        let equal = |lhs: u32, rhs: u32| -> bool {
            let lhs = lhs as usize;
            let rhs = rhs as usize;
            let size = get_level_size(levels[lhs] as i32, states);

            levels[lhs] == levels[rhs]
                && data[offsets[lhs] as usize..][..size] == data[offsets[rhs] as usize..][..size]
        };

        let entry = hash_lookup3(&table, hash, equal, next as u32, EMPTY_SLOT);

        if table[entry] == EMPTY_SLOT {
            table[entry] = next as u32;
            next += 1;
            offset += size;
        }

        remap[i] = table[entry] as i32;
    }

    // remap triangle indices to new indices or special indices
    for idx in omm_indices.iter_mut() {
        *idx = if *idx < 0 { *idx } else { remap[*idx as usize] };
    }

    next
}

// opacity micromaps use a "bird" traversal order which recursively subdivides the triangles:
// https://docs.vulkan.org/spec/latest/_images/micromap-subd.svg
// note that triangles 0 and 2 have the same winding as the source triangle, however triangles 1 (flipped)
// and 3 (upright) have flipped winding; this is obvious from the level 2 subdivision in the diagram above
fn get_level_size(level: i32, states: i32) -> usize {
    // 1-bit 2-state or 2-bit 4-state per micro triangle, rounded up to whole bytes
    ((1usize << (level as u32 * 2)) * (states as usize >> 1) + 7) >> 3
}

struct Texture<'a> {
    data: &'a [u8],
    stride: usize,
    pitch: usize,
    width: u32,
    height: u32,
    widthf: f32,  // width * 256.0
    heightf: f32, // height * 256.0
}

fn sample_texture(texture: &Texture, u: f32, v: f32) -> f32 {
    // wrap texture coordinates; floor is expensive so only call it if we're outside of [0, 1] range (+eps)
    let u = if (u - 0.5).abs() > 0.5 {
        u - math::floor(u)
    } else {
        u
    };
    let v = if (v - 0.5).abs() > 0.5 {
        v - math::floor(v)
    } else {
        v
    };

    // convert from [0, 1] to 16.8 fixed point coordinates (rounded to nearest subpixel) with texel centers on integer grid
    // (`as` matches upstream for every input: non-finite UVs saturate into the
    // `.max(0)` clamp and the inbounds safeguard below, like upstream's INT_MIN)
    let uf = (u * texture.widthf - 127.5) as i32;
    let vf = (v * texture.heightf - 127.5) as i32;

    // clamp to avoid extrapolation past left/top edge since we don't wrap across the edge
    let uf = uf.max(0);
    let vf = vf.max(0);

    // x/y are texel coordinates, rx/ry are subpixel offsets
    let x = uf >> 8;
    let y = vf >> 8;
    let rx = uf & 255;
    let ry = vf & 255;

    // safeguard: this should not happen but if it ever does, ensure the accesses are inbounds
    if x as u32 >= texture.width || y as u32 >= texture.height {
        return 0.0;
    }

    // clamp the offsets instead of wrapping for simplicity and performance
    let offset = y as usize * texture.pitch + x as usize * texture.stride;
    let offsetx = if x + 1 < texture.width as i32 {
        texture.stride
    } else {
        0
    };
    let offsety = if y + 1 < texture.height as i32 {
        texture.pitch
    } else {
        0
    };

    let a00 = texture.data[offset] as i32;
    let a10 = texture.data[offset + offsetx] as i32;
    let a01 = texture.data[offset + offsety] as i32;
    let a11 = texture.data[offset + offsetx + offsety] as i32;

    // bilinear interpolation in integer space: result is 8.16 fixed point
    let ax0 = a00 * 256 + (a10 - a00) * rx;
    let ax1 = a01 * 256 + (a11 - a01) * rx;
    let axy = ax0 * 256 + (ax1 - ax0) * ry;

    axy as f32 * (1.0 / (255.0 * 65536.0))
}

// upstream hashes the raw int[6] bytes via memcpy (native endian); LE is pinned
// via to_le_bytes so the hash is deterministic across platforms
fn hash_update4u(mut h: u32, key: &[u8]) -> u32 {
    const M: u32 = 0x5bd1_e995;

    for word in key.as_chunks::<4>().0 {
        let mut k = u32::from_le_bytes(*word);

        k = k.wrapping_mul(M);
        k ^= k >> 24;
        k = k.wrapping_mul(M);

        h = h.wrapping_mul(M);
        h ^= k;
    }

    h
}

#[derive(Clone)]
struct TriangleOmm {
    uvs: [i32; 6],
    level: i32,
}

fn hash_buckets3(count: usize) -> usize {
    let mut buckets = 1usize;
    while buckets < count + count / 4 {
        buckets *= 2;
    }

    buckets
}

fn hash_lookup3(
    table: &[u32],
    hash: impl Fn(u32) -> u32,
    equal: impl Fn(u32, u32) -> bool,
    key: u32,
    empty: u32,
) -> usize {
    let hashmod = table.len() - 1;
    let mut bucket = hash(key) as usize & hashmod;
    let mut probe = 0usize;

    // termination: hash_buckets3 caps the load factor at 0.8 and quadratic probing
    // over a power-of-two table visits every slot, so an empty slot always exists
    loop {
        let item = table[bucket];

        if item == empty || equal(item, key) {
            return bucket;
        }

        // hash collision, quadratic probing
        bucket = (bucket + probe + 1) & hashmod;
        probe += 1;
    }
}

fn quantize_subpixel(v: f32, size: u32) -> i32 {
    cast_i32(v * (size as i32 * 4) as f32 + if v >= 0.0 { 0.5 } else { -0.5 })
}

fn rasterize_edge(u0: f32, v0: f32, u1: f32, v1: f32, edgeres: i32, texture: &Texture) -> i32 {
    let edgestep = 1.0 / (edgeres + 1) as f32;

    let ud = (u1 - u0) * edgestep;
    let vd = (v1 - v0) * edgestep;
    let mut u = u0;
    let mut v = v0;

    let mut mask = 0i32;
    let mut count = 0i32;

    for i in 0..edgeres {
        u += ud;
        v += vd;

        let a = sample_texture(texture, u, v);
        mask |= ((a >= 0.5) as i32) << i;
        count += (a >= 0.5) as i32;
    }

    mask | (count << 16)
}

#[allow(clippy::too_many_arguments)]
fn rasterize_opacity_0<const STATES: i32>(
    result: &mut [u8],
    index: usize,
    a0: f32,
    a1: f32,
    a2: f32,
    ac: f32,
    e0: i32,
    e1: i32,
    e2: i32,
    edgeres: i32,
) {
    // basic coverage estimator from center and corner values; trained to minimize error
    let mut coverage = (a0 + a1 + a2) * 0.12 + ac * 0.64;

    if edgeres != 0 {
        let edgescale = 1.0 / edgeres as f32;

        // if we have edge samples, we can get a better coverage estimate by including them; trained to minimize error
        coverage = ac * 0.22
            + ((e0 >> 16) + (e1 >> 16) + (e2 >> 16)) as f32 * edgescale * 0.23
            + (a0 + a1 + a2) * 0.03;
    }

    if STATES == 2 {
        result[index / 8] |= ((coverage >= 0.5) as u8) << (index % 8);
        return;
    }

    let transp = (a0 < 0.5) as i32 & (a1 < 0.5) as i32 & (a2 < 0.5) as i32 & (ac < 0.5) as i32;
    let opaque = (a0 > 0.5) as i32 & (a1 > 0.5) as i32 & (a2 > 0.5) as i32 & (ac > 0.5) as i32;

    // treat state as known if thresholding of corners & centers against wider bounds is consistent
    // for unknown states, we currently use the same formula as the 2-state opacity for better consistency with forced 2-state
    let unknown = 2 + (coverage >= 0.5) as i32;
    let mut state = if (transp | opaque) != 0 {
        opaque
    } else {
        unknown
    };

    if edgeres != 0 && (transp | opaque) != 0 {
        // if we have edge samples, ensure they are consistent too, falling back to unknown if not
        let exp = if opaque != 0 { (1 << edgeres) - 1 } else { 0 };
        let eok = ((e0 & 0xffff) == exp) & ((e1 & 0xffff) == exp) & ((e2 & 0xffff) == exp);

        state = if eok { state } else { unknown };
    }

    result[index / 4] |= (state as u8) << ((index % 4) * 2);
}

fn rasterize_opacity_1<const STATES: i32>(
    result: &mut [u8],
    index: usize,
    edgeres: i32,
    c0: &[f32; 3],
    c1: &[f32; 3],
    c2: &[f32; 3],
    texture: &Texture,
) {
    // compute each edge midpoint & sample
    let mut c01 = [(c0[0] + c1[0]) / 2.0, (c0[1] + c1[1]) / 2.0, 0.0];
    let mut c12 = [(c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0, 0.0];
    let mut c20 = [(c2[0] + c0[0]) / 2.0, (c2[1] + c0[1]) / 2.0, 0.0];

    c01[2] = sample_texture(texture, c01[0], c01[1]);
    c12[2] = sample_texture(texture, c12[0], c12[1]);
    c20[2] = sample_texture(texture, c20[0], c20[1]);

    // corner tables for each edge, and corner + edge tables for each triangle
    // edges are numbered counter clockwise, 6 outer first, 3 inner last; triangle vertex and edge references are in triangle winding order
    const EDGES: [[u8; 2]; 9] = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 0],
        [5, 1],
        [1, 3],
        [3, 5],
    ];
    const TRIANGLES: [[u8; 6]; 4] = [
        [0, 1, 5, 0, 6, 5],
        [5, 3, 1, 8, 7, 6],
        [1, 2, 3, 1, 2, 7],
        [3, 5, 4, 8, 4, 3],
    ];

    let points: [&[f32; 3]; 6] = [c0, &c01, c1, &c12, c2, &c20];

    let mut em = [0i32; 9];

    // sample additional points on the edges to improve state estimation
    if edgeres > 0 {
        for (i, edge) in EDGES.iter().enumerate() {
            em[i] = rasterize_edge(
                points[edge[0] as usize][0],
                points[edge[0] as usize][1],
                points[edge[1] as usize][0],
                points[edge[1] as usize][1],
                edgeres,
                texture,
            );
        }
    }

    for (i, tri) in TRIANGLES.iter().enumerate() {
        let p0 = points[tri[0] as usize];
        let p1 = points[tri[1] as usize];
        let p2 = points[tri[2] as usize];

        // compute triangle center & sample
        let uc = (p0[0] + p1[0] + p2[0]) * (1.0 / 3.0);
        let vc = (p0[1] + p1[1] + p2[1]) * (1.0 / 3.0);
        let ac = sample_texture(texture, uc, vc);

        // rasterize opacity state based on alpha values in corners and center (and optionally edges)
        rasterize_opacity_0::<STATES>(
            result,
            index * 4 + i,
            p0[2],
            p1[2],
            p2[2],
            ac,
            em[tri[3] as usize],
            em[tri[4] as usize],
            em[tri[5] as usize],
            edgeres,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn rasterize_opacity_rec<const STATES: i32>(
    result: &mut [u8],
    index: usize,
    level: i32,
    edgeres: i32,
    c0: &[f32; 3],
    c1: &[f32; 3],
    c2: &[f32; 3],
    texture: &Texture,
) {
    if level == 0 {
        // compute triangle center & sample
        let uc = (c0[0] + c1[0] + c2[0]) * (1.0 / 3.0);
        let vc = (c0[1] + c1[1] + c2[1]) * (1.0 / 3.0);
        let ac = sample_texture(texture, uc, vc);

        let mut e0 = 0;
        let mut e1 = 0;
        let mut e2 = 0;

        if edgeres > 0 {
            // sample additional points on the edges to improve state estimation
            e0 = rasterize_edge(c0[0], c0[1], c1[0], c1[1], edgeres, texture);
            e1 = rasterize_edge(c1[0], c1[1], c2[0], c2[1], edgeres, texture);
            e2 = rasterize_edge(c2[0], c2[1], c0[0], c0[1], edgeres, texture);
        }

        // rasterize opacity state based on alpha values in corners and center (and optionally edges)
        rasterize_opacity_0::<STATES>(result, index, c0[2], c1[2], c2[2], ac, e0, e1, e2, edgeres);
        return;
    }

    // fast path: equivalent to recursive rasterization, but reuses edge data to reduce sample count
    if level == 1 && edgeres > 0 {
        rasterize_opacity_1::<STATES>(result, index, edgeres, c0, c1, c2, texture);
        return;
    }

    // compute each edge midpoint & sample
    let mut c01 = [(c0[0] + c1[0]) / 2.0, (c0[1] + c1[1]) / 2.0, 0.0];
    let mut c12 = [(c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0, 0.0];
    let mut c20 = [(c2[0] + c0[0]) / 2.0, (c2[1] + c0[1]) / 2.0, 0.0];

    c01[2] = sample_texture(texture, c01[0], c01[1]);
    c12[2] = sample_texture(texture, c12[0], c12[1]);
    c20[2] = sample_texture(texture, c20[0], c20[1]);

    // recursively rasterize each triangle
    // note: triangles 1 and 3 have flipped winding, and 1 is flipped upside down
    #[allow(clippy::identity_op)] // upstream literal index arithmetic
    rasterize_opacity_rec::<STATES>(
        result,
        index * 4 + 0,
        level - 1,
        edgeres,
        c0,
        &c01,
        &c20,
        texture,
    );
    rasterize_opacity_rec::<STATES>(
        result,
        index * 4 + 1,
        level - 1,
        edgeres,
        &c20,
        &c12,
        &c01,
        texture,
    );
    rasterize_opacity_rec::<STATES>(
        result,
        index * 4 + 2,
        level - 1,
        edgeres,
        &c01,
        c1,
        &c12,
        texture,
    );
    rasterize_opacity_rec::<STATES>(
        result,
        index * 4 + 3,
        level - 1,
        edgeres,
        &c12,
        &c20,
        c2,
        texture,
    );
}

fn get_special_index(data: &[u8], level: i32, states: i32) -> i32 {
    let first = (data[0] & if states == 2 { 1u8 } else { 3u8 }) as i32;
    let special = -(1 + first);

    // at level 0, every micromap can be converted to a special index
    if level == 0 {
        return special;
    }

    // at level 1 with 2 states, the byte is partially filled so we need a separate check
    if level == 1 && states == 2 {
        return if (data[0] as i32 & 15) == ((-first) & 15) {
            special
        } else {
            0
        };
    }

    // otherwise we need to check that all bytes are consistent with the first value and we can do this byte-wise
    let expected = first * if states == 2 { 0xff } else { 0x55 };
    let size = get_level_size(level, states);

    for &byte in &data[..size] {
        if byte as i32 != expected {
            return 0;
        }
    }

    special
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_opacityMapMeasure,
    // meshopt_opacityMapRasterize, meshopt_opacityMapEntrySize,
    // meshopt_opacityMapCompact.
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

        fn i32(&mut self) -> i32 {
            self.u32() as i32
        }

        fn bytes(&mut self, n: usize) -> &'a [u8] {
            let s = &self.data[self.pos..self.pos + n];
            self.pos += n;
            s
        }

        fn f32s(&mut self, count: usize) -> Vec<f32> {
            let raw = self.bytes(count * 4);
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect()
        }

        fn vec_u32(&mut self, n: usize) -> Vec<u32> {
            (0..n).map(|_| self.u32()).collect()
        }

        fn vec_i32(&mut self, n: usize) -> Vec<i32> {
            (0..n).map(|_| self.i32()).collect()
        }
    }

    #[test]
    fn golden_opacity_map_entry_size() {
        const _LEN_CHECK: usize = 312;
        let data = include_bytes!("../tests/data/om_entrysize.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let level = c.i32();
            let states = c.i32();
            let expected = c.u32();

            assert_eq!(
                opacity_map_entry_size(level, states),
                expected as usize,
                "entrySize case {case} (level {level}, states {states})"
            );
            case += 1;
        }
        assert_eq!(case, 26);
    }

    #[test]
    fn golden_opacity_map_rasterize() {
        const _LEN_CHECK: usize = 71471;
        let data = include_bytes!("../tests/data/om_rasterize.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let level = c.i32();
            let states = c.i32();
            let width = c.u32();
            let height = c.u32();
            let stride = c.u32() as usize;
            let pitch = c.u32() as usize;
            let uvs = c.f32s(6);
            let texture = c.bytes(pitch * height as usize);
            let expected = c.bytes(opacity_map_entry_size(level, states));

            let mut result = vec![0u8; expected.len()];
            opacity_map_rasterize(
                &mut result,
                level,
                states,
                &uvs[0..2],
                &uvs[2..4],
                &uvs[4..6],
                texture,
                stride,
                pitch,
                width,
                height,
            );

            assert_eq!(
                result.as_slice(),
                expected,
                "rasterize case {case} (level {level}, states {states})"
            );
            case += 1;
        }
        assert_eq!(case, 14);
    }

    #[test]
    fn golden_opacity_map_measure() {
        const _LEN_CHECK: usize = 21413;
        let data = include_bytes!("../tests/data/om_measure.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let uvs_stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let texture_width = c.u32();
            let texture_height = c.u32();
            let max_level = c.i32();
            let target_edge = c.f32s(1)[0];
            let uvs = c.f32s(vertex_count * uvs_stride / 4);
            let indices = c.vec_u32(index_count);

            let tri_count = index_count / 3;
            let expected_indices = c.vec_i32(tri_count);
            let expected_sources = c.vec_u32(tri_count);
            let expected_levels = c.bytes(tri_count).to_vec();
            let expected_result = c.u32();

            let mut levels = vec![0u8; tri_count];
            let mut sources = vec![0u32; tri_count];
            let mut omm_indices = vec![0i32; tri_count];
            let result = opacity_map_measure(
                &mut levels,
                &mut sources,
                &mut omm_indices,
                &indices,
                &uvs,
                vertex_count,
                uvs_stride,
                texture_width,
                texture_height,
                max_level,
                target_edge,
            );

            assert_eq!(
                result, expected_result as usize,
                "measure case {case} result"
            );
            assert_eq!(
                omm_indices, expected_indices,
                "measure case {case} omm_indices"
            );
            assert_eq!(sources, expected_sources, "measure case {case} sources");
            assert_eq!(levels, expected_levels, "measure case {case} levels");
            case += 1;
        }
        assert_eq!(case, 6);
    }

    #[test]
    fn golden_opacity_map_compact() {
        const _LEN_CHECK: usize = 788;
        let data = include_bytes!("../tests/data/om_compact.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        let mut case = 0;
        while c.pos < data.len() {
            let states = c.i32();
            let omm_count = c.u32() as usize;
            let data_size = c.u32() as usize;
            let triangle_count = c.u32() as usize;
            let mut levels = c.bytes(omm_count).to_vec();
            let mut offsets = c.vec_u32(omm_count);
            let mut buffer = c.bytes(data_size).to_vec();
            let mut omm_indices = c.vec_i32(triangle_count);

            let expected_levels = c.bytes(omm_count).to_vec();
            let expected_offsets = c.vec_u32(omm_count);
            let expected_data = c.bytes(data_size).to_vec();
            let expected_indices = c.vec_i32(triangle_count);
            let expected_next = c.u32();

            let next = opacity_map_compact(
                &mut buffer,
                &mut levels,
                &mut offsets,
                &mut omm_indices,
                states,
            );

            // mirror the generator contract: entries/tails beyond the compacted
            // result hold unspecified speculative writes and are zero-filled
            levels[next..].fill(0);
            offsets[next..].fill(0);
            let end = if next > 0 {
                offsets[next - 1] as usize + opacity_map_entry_size(levels[next - 1] as i32, states)
            } else {
                0
            };
            buffer[end..].fill(0);

            assert_eq!(next, expected_next as usize, "compact case {case} next");
            assert_eq!(levels, expected_levels, "compact case {case} levels");
            assert_eq!(offsets, expected_offsets, "compact case {case} offsets");
            assert_eq!(buffer, expected_data, "compact case {case} data");
            assert_eq!(
                omm_indices, expected_indices,
                "compact case {case} omm_indices"
            );
            case += 1;
        }
        assert_eq!(case, 2);
    }

    #[test]
    fn hash_buckets3_caps_load_factor() {
        assert_eq!(hash_buckets3(0), 1);
        assert_eq!(hash_buckets3(1), 1); // count + count/4 truncates: 1 + 0
        assert_eq!(hash_buckets3(2), 2);
        assert_eq!(hash_buckets3(8), 16);
        assert_eq!(hash_buckets3(9), 16);
    }

    #[test]
    fn level_size_matches_bit_packing() {
        assert_eq!(get_level_size(0, 2), 1);
        assert_eq!(get_level_size(0, 4), 1);
        assert_eq!(get_level_size(1, 2), 1);
        assert_eq!(get_level_size(1, 4), 1);
        assert_eq!(get_level_size(2, 2), 2);
        assert_eq!(get_level_size(2, 4), 4);
        assert_eq!(get_level_size(5, 4), 256);
        assert_eq!(get_level_size(12, 4), 4194304);
    }

    #[test]
    fn special_index_classifies_constant_entries() {
        // level 0: every entry is convertible
        assert_eq!(get_special_index(&[0x00], 0, 2), -1);
        assert_eq!(get_special_index(&[0x01], 0, 2), -2);
        assert_eq!(get_special_index(&[0x02], 0, 4), -3);
        assert_eq!(get_special_index(&[0x03], 0, 4), -4);
        assert_eq!(get_special_index(&[0x55], 0, 4), -2);

        // level 1, 2-state: half byte must be consistent
        assert_eq!(get_special_index(&[0x0f], 1, 2), -2);
        assert_eq!(get_special_index(&[0x1e], 1, 2), 0);
        assert_eq!(get_special_index(&[0x11], 1, 2), 0);

        // deeper levels: byte-wise consistency against the packed first nibble
        assert_eq!(get_special_index(&[0xff, 0xff, 0xff, 0xff], 2, 2), -2);
        assert_eq!(get_special_index(&[0xff, 0xfe, 0xff, 0xff], 2, 2), 0);
        assert_eq!(get_special_index(&[0x55, 0x55, 0x55, 0x55], 2, 4), -2);
        assert_eq!(get_special_index(&[0x00, 0x00, 0x00, 0x00], 2, 4), -1);
    }

    #[test]
    fn quantize_subpixel_rounds_away_from_zero() {
        let w = 64u32;
        assert_eq!(quantize_subpixel(0.0, w), 0);
        assert_eq!(quantize_subpixel(1.0, w), 256);
        assert_eq!(quantize_subpixel(-0.001, w), 0); // -0.256 + 0.5 truncates to 0
        assert_eq!(quantize_subpixel(0.5, w), 128);
    }
}
