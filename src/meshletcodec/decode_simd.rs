//! SIMD meshlet-decode kernels (ADR-0012).
//!
//! Upstream: `meshletcodec.cpp` (`decodeBuildTables`, `decodeTriangleGroup`,
//! `decodeVertexGroup`, `decodeTrianglesSimd`, `decodeVerticesSimd`,
//! `decodeMeshletSimd`) at the pinned revision, SSE4.1 variant ported
//! line-for-line. Upstream ships this kernel by default on x86 (SSE4.1 gate,
//! cpuid fallback in portable builds); the `simd` feature assumes SSE4.1 is
//! present, mirroring a `-msse4.1` build without the fallback. Byte-identical
//! to the scalar decoder: the golden `mc_*` fixtures are the oracle.
//!
//! Two deliberate divergences from the upstream SIMD write pattern, both
//! invisible in the specified output region:
//! - upstream overwrites up to 3 bytes past the logical triangle/vertex tail
//!   to amortize stores ("32-bit aligned size available"); Rust slice bounds
//!   forbid that, so tails write exactly the logical elements
//! - upstream `Raw` decoding rounds counts up and writes the padded outputs;
//!   like the scalar decoder here, exact counts are decoded (same pointer
//!   progression, same accept/reject) and only the logical region is written
//!
//! This module may contain `unsafe` (ADR-0012); each block carries a safety
//! comment referencing the upstream invariant it preserves.

#![allow(unsafe_code)]

use core::arch::x86_64::*;

// SIMD triangle-decode state is stored in a single 16b register:
// 0..5: 6 next extra bytes; 6..14: 9 bytes = 3 triangles of index data;
// 15: 'next' byte. The tables pack next[10..15] + shuf[6..15] per code.
// Upstream builds these at static-init time (decodeBuildTables); a
// const-evaluated table is behavior-identical.
type MeshletTables = ([[u8; 16]; 256], [u8; 256], [[u8; 16]; 256], [u8; 256]);

const fn build_tables() -> MeshletTables {
    let mut masks = [[0u8; 16]; 256];
    let mut extra = [0u8; 256];
    let mut verts = [[0u8; 16]; 256];
    let mut length = [0u8; 256];

    // triangle decoding tables for each combination of two triangle codes
    let mut code = 0usize;
    while code < 256 {
        let mut shuf = [0u8; 16];
        let mut next = [0u8; 16];
        let mut extra_count = 0usize;
        let mut nextoff = 0usize;

        // state 0..5 is refilled every iteration; 6..8 always contain the last
        // decoded triangle; 15 will contain next
        shuf[6] = 12;
        shuf[7] = 13;
        shuf[8] = 14;
        shuf[15] = 15;

        let mut k = 0usize;
        while k < 2 {
            let tri = (code >> (k * 4)) & 0xf;

            if tri < 12 {
                if k == 1 && tri / 4 == 0 {
                    // decode one of two edges from the triangle decoded earlier
                    let s0 = if tri & 2 != 0 { 2 } else { 0 };
                    shuf[9 + k * 3] = shuf[9 + s0];
                    next[9 + k * 3] = next[9 + s0];

                    let s1 = if tri & 2 != 0 { 1 } else { 2 };
                    shuf[10 + k * 3] = shuf[9 + s1];
                    next[10 + k * 3] = next[9 + s1];
                } else {
                    // reuse: edge comes from the history, offset because the
                    // last history triangle was consumed by tri0
                    let trioff = 6 + k * 3 + (2 - tri / 4) * 3;

                    shuf[9 + k * 3] = (trioff + if tri & 2 != 0 { 2 } else { 0 }) as u8;
                    shuf[10 + k * 3] = (trioff + if tri & 2 != 0 { 1 } else { 2 }) as u8;
                }

                // third vertex is either next or comes from extra
                let ec = tri & 1;
                shuf[11 + k * 3] = if ec != 0 { extra_count as u8 } else { 15 };
                next[11 + k * 3] = if ec != 0 { 0 } else { nextoff as u8 };
                extra_count += ec;
                nextoff += 1 - ec;
            } else {
                // restart: three vertices, each from next or extra
                let fe = [
                    (tri > 12) as usize,
                    (tri > 13) as usize,
                    (tri > 14) as usize,
                ];
                let mut j = 0usize;
                while j < 3 {
                    let ec = fe[j];
                    shuf[9 + k * 3 + j] = if ec != 0 { extra_count as u8 } else { 15 };
                    next[9 + k * 3 + j] = if ec != 0 { 0 } else { nextoff as u8 };
                    extra_count += ec;
                    nextoff += 1 - ec;
                    j += 1;
                }
            }

            k += 1;
        }

        next[15] = nextoff as u8;

        let mut j = 0usize;
        while j < 6 {
            masks[code][j] = next[10 + j];
            j += 1;
        }
        let mut j = 0usize;
        while j < 10 {
            masks[code][6 + j] = shuf[6 + j];
            j += 1;
        }
        extra[code] = extra_count as u8;
        code += 1;
    }

    // vertex decoding tables: unpack 4 values with 0-4 bytes each
    let mut i = 0usize;
    while i < 256 {
        let mut shuf = [0u8; 16];
        let mut offset = 0usize;

        let mut k = 0usize;
        while k < 4 {
            let code = ((i >> k) & 1) | ((i >> (k + 3)) & 2);
            let len = if i == 0xff { 4 } else { code }; // 0/1/2/3 bytes, or all 4

            shuf[k * 4] = if len > 0 { offset as u8 } else { 0x80 };
            shuf[k * 4 + 1] = if len > 1 { (offset + 1) as u8 } else { 0x80 };
            shuf[k * 4 + 2] = if len > 2 { (offset + 2) as u8 } else { 0x80 };
            shuf[k * 4 + 3] = if len > 3 { (offset + 3) as u8 } else { 0x80 };

            offset += len;
            k += 1;
        }

        verts[i] = shuf;
        length[i] = offset as u8;
        i += 1;
    }

    (masks, extra, verts, length)
}

static TABLE_MASKS: [[u8; 16]; 256] = build_tables().0;
static TABLE_EXTRA: [u8; 256] = build_tables().1;
static TABLE_VERTS: [[u8; 16]; 256] = build_tables().2;
static TABLE_LENGTH: [u8; 256] = build_tables().3;

#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn decode_triangle_group(
    state: __m128i,
    code: usize,
    buffer: *const u8,
    extra: &mut usize,
) -> __m128i {
    // SAFETY: the 8-byte load at `extra` is covered by the caller's
    // `extra > bound` check plus the encoder-guaranteed 16-byte gap
    // (`bound + 16 <= buffer end`), mirroring upstream decodeTriangleGroup;
    // `code` is a nibble < 256 so table indices are in range; SSE4.1 is the
    // `simd` feature's documented requirement.
    unsafe {
        let shuf = _mm_loadu_si128(TABLE_MASKS[code].as_ptr().cast());
        let next = _mm_slli_si128::<10>(shuf);

        // patch first 6 bytes with current extra and roll state forward
        let ext = _mm_loadl_epi64(buffer.add(*extra).cast());
        let state = _mm_blend_epi16::<7>(state, ext);
        let state = _mm_add_epi8(_mm_shuffle_epi8(state, shuf), next);

        *extra += TABLE_EXTRA[code] as usize;

        state
    }
}

#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn decode_vertex_group(
    last: __m128i,
    code: usize,
    buffer: *const u8,
    data: &mut usize,
) -> __m128i {
    // SAFETY: the 16-byte load at `data` is covered by the caller's
    // `data > bound` check plus the 16-byte gap (`bound + 16 <= buffer end`),
    // mirroring upstream decodeVertexGroup; `code` < 256 keeps indices in
    // range; SSE4.1 is the `simd` feature's documented requirement.
    unsafe {
        let word = _mm_loadu_si128(buffer.add(*data).cast());
        let shuf = _mm_loadu_si128(TABLE_VERTS[code].as_ptr().cast());

        let v = _mm_shuffle_epi8(word, shuf);

        // unzigzag+1
        let xl = _mm_sub_epi32(_mm_setzero_si128(), _mm_and_si128(v, _mm_set1_epi32(1)));
        let xr = _mm_srli_epi32::<1>(v);
        let x = _mm_add_epi32(_mm_xor_si128(xl, xr), _mm_set1_epi32(1));

        // prefix sum
        let x = _mm_add_epi32(x, _mm_slli_si128::<8>(x));
        let x = _mm_add_epi32(x, _mm_slli_si128::<4>(x));
        let x = _mm_add_epi32(x, _mm_shuffle_epi32::<0xff>(last));

        *data += TABLE_LENGTH[code] as usize;

        x
    }
}

// extracts both triangles of a pair as [a,b,c,pad] dwords (state bytes 9..15)
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn repack_tri_pair(state: __m128i) -> (u32, u32) {
    let repack = _mm_setr_epi8(9, 10, 11, -1, 12, 13, 14, -1, 0, 0, 0, 0, 0, 0, 0, 0);
    let r = _mm_shuffle_epi8(state, repack);
    (
        _mm_cvtsi128_si32(r) as u32,
        _mm_extract_epi32::<1>(r) as u32,
    )
}

#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn extract_tri(state: __m128i) -> u32 {
    let repack = _mm_setr_epi8(9, 10, 11, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    _mm_cvtsi128_si32(_mm_shuffle_epi8(state, repack)) as u32
}

#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn extract_vertex_lanes(v: __m128i) -> (u32, u32, u32, u32) {
    (
        _mm_cvtsi128_si32(v) as u32,
        _mm_extract_epi32::<1>(v) as u32,
        _mm_extract_epi32::<2>(v) as u32,
        _mm_extract_epi32::<3>(v) as u32,
    )
}

// Upstream: `decodeTrianglesSimd` (unsigned int output variant). Decodes
// triangle pairs; triangle values are passed to `write` in the same shifted
// format the scalar core uses ((tri << 8) undoes the scalar `tri >> 8`).
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_triangles_simd(
    triangle_count: usize,
    codes: &[u8],
    mut extra: usize,
    bound: usize,
    buffer: &[u8],
    mut write: impl FnMut(usize, u32),
) -> Option<usize> {
    let base = buffer.as_ptr();
    // SAFETY: pure register init
    let mut state = unsafe { _mm_setzero_si128() };

    let pairs = triangle_count / 2;

    let mut i = 0usize;
    while i < pairs {
        // one table entry per byte: both nibbles decode in a single call
        let code = codes[i] as usize;

        // a triangle pair reads at most 6 extra bytes; the 16-byte gap covers
        // the loads (see kernel-level safety comments)
        if extra > bound {
            return None;
        }

        // SAFETY: bound check above; see decode_triangle_group
        state = unsafe { decode_triangle_group(state, code, base, &mut extra) };
        // SAFETY: pure register ops
        let (t0, t1) = unsafe { repack_tri_pair(state) };
        write(i * 2, t0 << 8);
        write(i * 2 + 1, t1 << 8);

        i += 1;
    }

    // 1-triangle tail: the full byte decodes (the high nibble is encoder
    // padding) but only the first triangle is output, as upstream
    if triangle_count & 1 != 0 {
        let code = codes[pairs] as usize;

        if extra > bound {
            return None;
        }

        // SAFETY: bound check above; see decode_triangle_group
        state = unsafe { decode_triangle_group(state, code, base, &mut extra) };
        // SAFETY: pure register ops
        let t0 = unsafe { extract_tri(state) };
        write(pairs * 2, t0 << 8);
    }

    Some(extra)
}

// Upstream: `decodeVerticesSimd` (unsigned int and unsigned short output
// variants share the kernel; the write callback formats the lanes).
pub(super) fn decode_vertices_simd(
    vertex_count: usize,
    ctrl: &[u8],
    mut data: usize,
    bound: usize,
    buffer: &[u8],
    mut write: impl FnMut(usize, u32),
) -> Option<usize> {
    let base = buffer.as_ptr();
    // SAFETY: pure register init (last = ~0 as in upstream)
    let mut last = unsafe { _mm_set1_epi32(-1) };

    let groups = vertex_count / 4;

    let mut i = 0usize;
    while i < groups {
        let code = ctrl[i] as usize;

        if data > bound {
            return None;
        }

        // SAFETY: bound check above; see decode_vertex_group
        last = unsafe { decode_vertex_group(last, code, base, &mut data) };
        // SAFETY: pure register ops
        let (v0, v1, v2, v3) = unsafe { extract_vertex_lanes(last) };
        write(i * 4, v0);
        write(i * 4 + 1, v1);
        write(i * 4 + 2, v2);
        write(i * 4 + 3, v3);

        i += 1;
    }

    // 1-3 vertex tail
    if vertex_count & 3 != 0 {
        let code = ctrl[groups] as usize;

        if data > bound {
            return None;
        }

        // SAFETY: bound check above; see decode_vertex_group
        last = unsafe { decode_vertex_group(last, code, base, &mut data) };
        // SAFETY: pure register ops
        let (v0, v1, v2, _) = unsafe { extract_vertex_lanes(last) };
        let tail = vertex_count & 3;
        write(groups * 4, v0);
        if tail > 1 {
            write(groups * 4 + 1, v1);
        }
        if tail > 2 {
            write(groups * 4 + 2, v2);
        }
    }

    Some(data)
}
