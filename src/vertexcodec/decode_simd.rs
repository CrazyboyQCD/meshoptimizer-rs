//! SIMD vertex-decode kernels, x86_64 (ADR-0012).
//!
//! Upstream: `vertexcodec.cpp` (`decodeBytesGroupSimd`, `decodeBytesSimd`,
//! `decodeDeltas4Simd`, `decodeVertexBlockSimd`, `transpose8`, `unzigzag8/16`,
//! `rotate32`) at the pinned revision, ported line-for-line. The SSSE3+POPCNT
//! kernel assumes the ISA is present (upstream compiled with `-mssse3
//! -mpopcnt`); the AVX512-VBMI2 kernel mirrors the `SIMD_AVX` compile gate
//! behind the `simd-avx512` feature. Both kernels are byte-identical to the
//! scalar decoder ([ADR-0010](../../docs/adr/0010-scalar-profile-fixtures.md));
//! the golden suite is the differential oracle.

#![allow(unsafe_code)]

use core::arch::x86_64::*;

#[cfg(not(feature = "simd-avx512"))]
use super::decode_tables::GROUP_SHUFFLE;
use super::decode_tables::{HBTN, LANES};
use super::{
    BYTE_GROUP_DECODE_LIMIT, BYTE_GROUP_SIZE, VERTEX_BLOCK_MAX_SIZE, VERTEX_BLOCK_SIZE_BYTES,
};

// sent mask, replicating shuffle, and two multipliers (even/odd) for
// multishift emulation
#[cfg(not(feature = "simd-avx512"))]
const GROUP_CONFIG_SSE: [[[u8; 16]; 4]; 9] = [
    [[1; 16], [128; 16], [0; 16], [0; 16]],
    [
        [3; 16],
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [4, 0, 64, 0, 4, 0, 64, 0, 4, 0, 64, 0, 4, 0, 64, 0],
        [16, 0, 0, 1, 16, 0, 0, 1, 16, 0, 0, 1, 16, 0, 0, 1],
    ],
    [
        [15; 16],
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
        [16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    [[0; 16], [128; 16], [0; 16], [0; 16]],
    [[1; 16], [128; 16], [0; 16], [0; 16]],
    [
        [1; 16],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 64, 0, 16, 0, 4, 0, 0, 1, 64, 0, 16, 0, 4, 0],
        [128, 0, 32, 0, 8, 0, 2, 0, 128, 0, 32, 0, 8, 0, 2, 0],
    ],
    [
        [3; 16],
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [4, 0, 64, 0, 4, 0, 64, 0, 4, 0, 64, 0, 4, 0, 64, 0],
        [16, 0, 0, 1, 16, 0, 0, 1, 16, 0, 0, 1, 16, 0, 0, 1],
    ],
    [
        [15; 16],
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
        [16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    [[0; 16], [128; 16], [0; 16], [0; 16]],
];

// sent mask and multishift control
#[cfg(feature = "simd-avx512")]
const GROUP_CONFIG_AVX512: [[[u8; 16]; 2]; 9] = [
    [[0; 16], [0; 16]],
    [
        [3; 16],
        [6, 4, 2, 0, 14, 12, 10, 8, 22, 20, 18, 16, 30, 28, 26, 24],
    ],
    [
        [15; 16],
        [4, 0, 12, 8, 20, 16, 28, 24, 36, 32, 44, 40, 52, 48, 60, 56],
    ],
    [[0; 16], [0; 16]],
    [[0; 16], [0; 16]],
    [
        [1; 16],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ],
    [
        [3; 16],
        [6, 4, 2, 0, 14, 12, 10, 8, 22, 20, 18, 16, 30, 28, 26, 24],
    ],
    [
        [15; 16],
        [4, 0, 12, 8, 20, 16, 28, 24, 36, 32, 44, 40, 52, 48, 60, 56],
    ],
    [[0; 16], [0; 16]],
];

// the kernel reads at most `kByteGroupDecodeLimit` (24) bytes from `data`: the
// sent/sel loads touch [0, 8) and the skip-relative rest load ends at
// data + ((2 << n) & 15) + 16 <= data + 24; every caller checks the same bound
// per group (or 4x24 shared in the fast path, since a group advances at most
// 23 bytes), which is exactly upstream's decodeBytesSimd invariant
#[cfg(not(feature = "simd-avx512"))]
#[target_feature(enable = "ssse3,popcnt")]
#[inline]
unsafe fn decode_bytes_group_sse(data: *const u8, buffer: *mut u8, hbits: u32) -> *const u8 {
    // SAFETY: caller guarantees 24 readable bytes at `data` and 16 writable
    // bytes at `buffer` (upstream decodeBytesSimd invariant, see above);
    // `hbits` is in 0..=8 by construction (v0: hshift 0, v1: hshift 4/5, plus
    // a 2-bit header selector), so all table indices are in range; the
    // SSSE3+POPCNT contract is the `simd` feature's documented requirement.
    unsafe {
        let n = HBTN[hbits as usize];

        let mut data64 = data.cast::<u64>().read_unaligned();
        data64 &= data64 >> n;
        data64 &= data64 >> (n >> 1);
        let datacnt = (data64 & LANES[hbits as usize]).count_ones();

        // for 8-bit groups, instead of loading the bytes through 'data', we
        // load them through 'skip' as they are easier to preserve; for 0-bit
        // groups, the load results get discarded because mask is always 0; in
        // both cases the shift wraps to zero
        let skip = data.add(((2u32 << n) & 15) as usize);

        let selb = _mm_loadu_si64(data);
        let rest = _mm_loadu_si128(skip.cast::<__m128i>());

        // unpack 1, 2 or 4-bit values: shuffle replicates each source byte into
        // both halves of a 16-bit lane; mulhi extracts even and odd fields into
        // the low byte; the results are interleaved back with shift/or
        let cfg = &GROUP_CONFIG_SSE[hbits as usize];
        let selw = _mm_shuffle_epi8(selb, _mm_loadu_si128(cfg[1].as_ptr().cast()));
        let sel0 = _mm_mulhi_epu16(selw, _mm_loadu_si128(cfg[2].as_ptr().cast()));
        let sel1 = _mm_mulhi_epu16(selw, _mm_loadu_si128(cfg[3].as_ptr().cast()));
        let seli = _mm_or_si128(sel0, _mm_slli_epi16::<8>(sel1));

        // the interleaved fields are masked by the bit count (special handling:
        // for 0/8-bit values, mul produces 0)
        let sent = _mm_loadu_si128(cfg[0].as_ptr().cast());
        let sel = _mm_and_si128(seli, sent);

        // compare sel to sentinel; returns 0 for 0-bit (mul produces 0, sent is
        // 1), 1 for 8-bit (mul produces 0, sent is 0)
        let mask = _mm_cmpeq_epi8(sel, sent);
        let mask16 = _mm_movemask_epi8(mask) as u16;
        let mask0 = mask16 as u8;
        let mask1 = (mask16 >> 8) as u8;

        // decode shuffle mask from two halves; second half needs to be shifted
        // by popcount(mask0)
        let sm0 = _mm_loadl_epi64(GROUP_SHUFFLE[mask0 as usize].as_ptr().cast());
        let sm1 = _mm_loadl_epi64(GROUP_SHUFFLE[mask1 as usize].as_ptr().cast());

        // each lane of mask is 0x00 or 0xff; sad yields 255*popcount(mask0) in
        // the low word => low byte is -popcount(mask0)
        let npops = _mm_sad_epu8(mask, _mm_setzero_si128());
        let sm1r = _mm_sub_epi8(sm1, _mm_shuffle_epi8(npops, _mm_setzero_si128()));
        let shuf = _mm_unpacklo_epi64(sm0, sm1r);

        // expand rest via shuffle mask and combine with sel; shuffle mask
        // zeroes out bytes that are replaced by sel
        let result = _mm_or_si128(_mm_shuffle_epi8(rest, shuf), _mm_andnot_si128(mask, sel));

        _mm_storeu_si128(buffer.cast::<__m128i>(), result);

        // datacnt is 0 for 8-bit groups so we can't use skip to advance; 0-bit
        // groups wrap the shift to zero
        data.add((((2u32 << n) & 31) + datacnt) as usize)
    }
}

#[cfg(feature = "simd-avx512")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
#[inline]
unsafe fn decode_bytes_group_avx512(data: *const u8, buffer: *mut u8, hbits: u32) -> *const u8 {
    // SAFETY: same read/write bounds as the SSE kernel; `hbits` in 0..=8 by
    // construction; the AVX512-VBMI2 contract is the `simd-avx512` feature's
    // documented requirement.
    unsafe {
        let n = HBTN[hbits as usize];

        let mut data64 = data.cast::<u64>().read_unaligned();
        data64 &= data64 >> n;
        data64 &= data64 >> (n >> 1);
        let datacnt = (data64 & LANES[hbits as usize]).count_ones();

        // for 8-bit groups, instead of loading the bytes through 'data', we
        // load them through 'skip' as they are easier to preserve; for 0-bit
        // groups, we use a masked load so that we load zero bytes; in both
        // cases the shift wraps to zero
        let skip = data.add(((2u32 << n) & 15) as usize);

        let selb = _mm_loadu_si64(data);
        let loadmask = ((n >> 2).wrapping_sub(1)) as u16 as __mmask16;
        let rest = _mm_maskz_loadu_epi8(loadmask, skip.cast::<i8>());

        let cfg = &GROUP_CONFIG_AVX512[hbits as usize];
        let sent = _mm_loadu_si128(cfg[0].as_ptr().cast());
        let ctrl = _mm_loadu_si128(cfg[1].as_ptr().cast());

        // unpack 1, 2 or 4-bit values using multishift and mask the result; for
        // 0/8-bit values, sel is always 0
        let selw = _mm_shuffle_epi32::<0x44>(selb);
        let sel = _mm_and_si128(sent, _mm_multishift_epi64_epi8(ctrl, selw));

        // compare sel to sentinel and combine; mask is 0 for 0/8-bit as sent is 0
        let mask16 = _mm_cmp_epi8_mask(sel, sent, _MM_CMPINT_EQ);
        let result = _mm_mask_expand_epi8(sel, mask16, rest);

        _mm_storeu_si128(buffer.cast::<__m128i>(), result);

        // datacnt is 0 for 8-bit groups so we can't use skip to advance; 0-bit
        // groups wrap the shift to zero
        data.add((((2u32 << n) & 31) + datacnt) as usize)
    }
}

trait GroupDecoder {
    unsafe fn group(data: *const u8, pos: usize, buffer: *mut u8, hbits: u32) -> usize;
}

#[cfg(not(feature = "simd-avx512"))]
struct GroupSse;

#[cfg(not(feature = "simd-avx512"))]
impl GroupDecoder for GroupSse {
    #[inline(always)]
    unsafe fn group(data: *const u8, pos: usize, buffer: *mut u8, hbits: u32) -> usize {
        // SAFETY: see decode_bytes_group_sse
        unsafe {
            let next = decode_bytes_group_sse(data.add(pos), buffer, hbits);
            next as usize - data as usize
        }
    }
}

#[cfg(feature = "simd-avx512")]
struct GroupAvx512;

#[cfg(feature = "simd-avx512")]
impl GroupDecoder for GroupAvx512 {
    #[inline(always)]
    unsafe fn group(data: *const u8, pos: usize, buffer: *mut u8, hbits: u32) -> usize {
        // SAFETY: see decode_bytes_group_avx512
        unsafe {
            let next = decode_bytes_group_avx512(data.add(pos), buffer, hbits);
            next as usize - data as usize
        }
    }
}

// Upstream: `decodeBytesSimd`. Position-based port of the shared 4-group loop;
// the ISA-specific group decoder is monomorphized through `D`.
#[inline]
unsafe fn decode_bytes_simd<D: GroupDecoder>(
    base: *const u8,
    mut pos: usize,
    data_end: usize,
    buffer: *mut u8,
    buffer_size: usize,
    hshift: u32,
) -> Option<usize> {
    // round number of groups to 4 to get number of header bytes
    let header_size = (buffer_size / BYTE_GROUP_SIZE).div_ceil(4);
    if data_end - pos < header_size {
        return None;
    }

    let header = pos;
    pos += header_size;

    let mut i = 0usize;

    // fast-path: process 4 groups at a time, do a shared bounds check
    while i + BYTE_GROUP_SIZE * 4 <= buffer_size && data_end - pos >= BYTE_GROUP_DECODE_LIMIT * 4 {
        let header_offset = i / BYTE_GROUP_SIZE;
        // SAFETY: in-bounds reads of `base` (see kernel-level bounds comment)
        let header_byte = unsafe { *base.add(header + header_offset / 4) };

        // very-fast-path: for consecutive 4 groups that are all 0-bit (v0/0,
        // v1/0/0000) or 8-bit (v0/3333, v1/1/3333), the branchless decoders are
        // slower than branching over the decoding of 4 groups and issuing a few
        // load/store ops
        if hshift != 5 && header_byte == 0 {
            // SAFETY: i + 64 <= buffer_size, scratch has buffer_size bytes
            unsafe { buffer.add(i).write_bytes(0, BYTE_GROUP_SIZE * 4) };
        } else if hshift != 4 && header_byte == 255 {
            // SAFETY: the shared check guarantees 64 payload bytes remain
            unsafe {
                core::ptr::copy_nonoverlapping(base.add(pos), buffer.add(i), BYTE_GROUP_SIZE * 4);
            }
            pos += BYTE_GROUP_SIZE * 4;
        } else {
            // SAFETY: each group reads/writes at most kByteGroupDecodeLimit
            // bytes and advances at most as much, within the shared 4x check
            unsafe {
                pos = D::group(
                    base,
                    pos,
                    buffer.add(i),
                    hshift + u32::from(header_byte & 3),
                );
                pos = D::group(
                    base,
                    pos,
                    buffer.add(i + BYTE_GROUP_SIZE),
                    hshift + u32::from((header_byte >> 2) & 3),
                );
                pos = D::group(
                    base,
                    pos,
                    buffer.add(i + BYTE_GROUP_SIZE * 2),
                    hshift + u32::from((header_byte >> 4) & 3),
                );
                pos = D::group(
                    base,
                    pos,
                    buffer.add(i + BYTE_GROUP_SIZE * 3),
                    hshift + u32::from((header_byte >> 6) & 3),
                );
            }
        }

        i += BYTE_GROUP_SIZE * 4;
    }

    // slow-path: process remaining groups
    while i < buffer_size {
        if data_end - pos < BYTE_GROUP_DECODE_LIMIT {
            return None;
        }

        let header_offset = i / BYTE_GROUP_SIZE;
        // SAFETY: in-bounds reads of `base` (see kernel-level bounds comment)
        let header_byte = unsafe { *base.add(header + header_offset / 4) };

        // SAFETY: per-group kByteGroupDecodeLimit bound checked above
        pos = unsafe {
            D::group(
                base,
                pos,
                buffer.add(i),
                hshift + u32::from((header_byte >> ((header_offset % 4) * 2)) & 3),
            )
        };

        i += BYTE_GROUP_SIZE;
    }

    Some(pos)
}

// upstream rotates by a runtime amount; the C compilers emit pslld/psrld with a
// variable count, which the SSE2 xmm-count shift intrinsics model directly
#[inline(always)]
unsafe fn rotate32(v: __m128i, r: u32) -> __m128i {
    // SAFETY: baseline SSE2 intrinsics, no feature requirement on x86_64
    unsafe {
        _mm_or_si128(
            _mm_sll_epi32(v, _mm_cvtsi32_si128(r as i32)),
            _mm_srl_epi32(v, _mm_cvtsi32_si128((32 - r) as i32)),
        )
    }
}

#[inline(always)]
unsafe fn unzigzag8(v: __m128i) -> __m128i {
    // SAFETY: baseline SSE2 intrinsics
    unsafe {
        let xl = _mm_sub_epi8(_mm_setzero_si128(), _mm_and_si128(v, _mm_set1_epi8(1)));
        let xr = _mm_and_si128(_mm_srli_epi16::<1>(v), _mm_set1_epi8(127));

        _mm_xor_si128(xl, xr)
    }
}

#[inline(always)]
unsafe fn unzigzag16(v: __m128i) -> __m128i {
    // SAFETY: baseline SSE2 intrinsics
    unsafe {
        let xl = _mm_sub_epi16(_mm_setzero_si128(), _mm_and_si128(v, _mm_set1_epi16(1)));
        let xr = _mm_srli_epi16::<1>(v);

        _mm_xor_si128(xl, xr)
    }
}

#[inline(always)]
unsafe fn transpose8(x0: &mut __m128i, x1: &mut __m128i, x2: &mut __m128i, x3: &mut __m128i) {
    // SAFETY: baseline SSE2 intrinsics
    unsafe {
        let t0 = _mm_unpacklo_epi8(*x0, *x1);
        let t1 = _mm_unpackhi_epi8(*x0, *x1);
        let t2 = _mm_unpacklo_epi8(*x2, *x3);
        let t3 = _mm_unpackhi_epi8(*x2, *x3);

        *x0 = _mm_unpacklo_epi16(t0, t2);
        *x1 = _mm_unpackhi_epi16(t0, t2);
        *x2 = _mm_unpacklo_epi16(t1, t3);
        *x3 = _mm_unpackhi_epi16(t1, t3);
    }
}

#[inline(always)]
unsafe fn unzr<const CHANNEL: u32>(r: __m128i, rot: u32) -> __m128i {
    // SAFETY: dispatches to baseline-SSE2 helpers only
    unsafe {
        match CHANNEL {
            0 => unzigzag8(r),
            1 => unzigzag16(r),
            _ => rotate32(r, rot),
        }
    }
}

#[inline(always)]
unsafe fn fixd<const CHANNEL: u32>(pi: __m128i, t: __m128i) -> __m128i {
    // SAFETY: baseline SSE2 intrinsics
    unsafe {
        match CHANNEL {
            0 => _mm_add_epi8(pi, t),
            1 => _mm_add_epi16(pi, t),
            _ => _mm_xor_si128(pi, t),
        }
    }
}

// Upstream: `decodeDeltas4Simd`. Decodes 16 vertices per iteration: transposes
// the four byte columns back into dword lanes, unzigzags/rotates, and folds
// them into the running delta chain.
#[inline]
unsafe fn decode_deltas4_simd<const CHANNEL: u32>(
    buffer: *const u8,
    transposed: *mut u8,
    vertex_count_aligned: usize,
    vertex_size: usize,
    last_vertex: *const u8,
    rot: u32,
) {
    // SAFETY: `buffer` exposes 4 * vertex_count_aligned readable bytes (the
    // scratch columns), `transposed` has kVertexBlockSizeBytes writable bytes
    // from offset k < vertex_size, and `last_vertex` has 4 readable bytes at
    // +k — all established by decode_vertex_block_simd, mirroring upstream.
    unsafe {
        let mut pi = _mm_loadu_si32(last_vertex);
        let mut savep = transposed;

        let mut j = 0usize;
        while j < vertex_count_aligned {
            let mut r0 = _mm_loadu_si128(buffer.add(j).cast::<__m128i>());
            let mut r1 = _mm_loadu_si128(buffer.add(j + vertex_count_aligned).cast::<__m128i>());
            let mut r2 =
                _mm_loadu_si128(buffer.add(j + vertex_count_aligned * 2).cast::<__m128i>());
            let mut r3 =
                _mm_loadu_si128(buffer.add(j + vertex_count_aligned * 3).cast::<__m128i>());

            transpose8(&mut r0, &mut r1, &mut r2, &mut r3);

            r0 = unzr::<CHANNEL>(r0, rot);
            let t0 = fixd::<CHANNEL>(pi, r0);
            pi = t0;
            let t1 = fixd::<CHANNEL>(pi, _mm_shuffle_epi32::<1>(r0));
            pi = t1;
            let t2 = fixd::<CHANNEL>(pi, _mm_shuffle_epi32::<2>(r0));
            pi = t2;
            let t3 = fixd::<CHANNEL>(pi, _mm_shuffle_epi32::<3>(r0));
            pi = t3;
            save_dword(&mut savep, vertex_size, t0);
            save_dword(&mut savep, vertex_size, t1);
            save_dword(&mut savep, vertex_size, t2);
            save_dword(&mut savep, vertex_size, t3);

            r1 = unzr::<CHANNEL>(r1, rot);
            let t0 = fixd::<CHANNEL>(pi, r1);
            pi = t0;
            let t1 = fixd::<CHANNEL>(pi, _mm_shuffle_epi32::<1>(r1));
            pi = t1;
            let t2 = fixd::<CHANNEL>(pi, _mm_shuffle_epi32::<2>(r1));
            pi = t2;
            let t3 = fixd::<CHANNEL>(pi, _mm_shuffle_epi32::<3>(r1));
            pi = t3;
            save_dword(&mut savep, vertex_size, t0);
            save_dword(&mut savep, vertex_size, t1);
            save_dword(&mut savep, vertex_size, t2);
            save_dword(&mut savep, vertex_size, t3);

            r2 = unzr::<CHANNEL>(r2, rot);
            let t0 = fixd::<CHANNEL>(pi, r2);
            pi = t0;
            let t1 = fixd::<CHANNEL>(pi, _mm_shuffle_epi32::<1>(r2));
            pi = t1;
            let t2 = fixd::<CHANNEL>(pi, _mm_shuffle_epi32::<2>(r2));
            pi = t2;
            let t3 = fixd::<CHANNEL>(pi, _mm_shuffle_epi32::<3>(r2));
            pi = t3;
            save_dword(&mut savep, vertex_size, t0);
            save_dword(&mut savep, vertex_size, t1);
            save_dword(&mut savep, vertex_size, t2);
            save_dword(&mut savep, vertex_size, t3);

            r3 = unzr::<CHANNEL>(r3, rot);
            let t0 = fixd::<CHANNEL>(pi, r3);
            pi = t0;
            let t1 = fixd::<CHANNEL>(pi, _mm_shuffle_epi32::<1>(r3));
            pi = t1;
            let t2 = fixd::<CHANNEL>(pi, _mm_shuffle_epi32::<2>(r3));
            pi = t2;
            let t3 = fixd::<CHANNEL>(pi, _mm_shuffle_epi32::<3>(r3));
            pi = t3;
            save_dword(&mut savep, vertex_size, t0);
            save_dword(&mut savep, vertex_size, t1);
            save_dword(&mut savep, vertex_size, t2);
            save_dword(&mut savep, vertex_size, t3);

            j += 16;
        }
    }
}

#[inline(always)]
unsafe fn save_dword(savep: &mut *mut u8, vertex_size: usize, t: __m128i) {
    // SAFETY: the save region is transposed[k + i*vertex_size .. +4] for
    // i < vertex_count_aligned, within the kVertexBlockSizeBytes scratch
    unsafe {
        (*savep).cast::<i32>().write_unaligned(_mm_cvtsi128_si32(t));
        *savep = (*savep).add(vertex_size);
    }
}

// Upstream: `decodeVertexBlockSimd`. Control flow mirrors the scalar
// decode_vertex_block with two differences: the varint columns decode through
// the SIMD group kernel over the aligned scratch layout, and the delta kernel
// processes 16 vertices per iteration.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn decode_vertex_block_simd<D: GroupDecoder>(
    data: &[u8],
    pos: &mut usize,
    data_end: usize,
    vertex_data: &mut [u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &mut [u8; 256],
    channels: Option<&[u8]>,
    version: i32,
) -> bool {
    debug_assert!(vertex_count > 0 && vertex_count <= VERTEX_BLOCK_MAX_SIZE);

    let mut buffer = [0u8; VERTEX_BLOCK_MAX_SIZE * 4];
    let mut transposed = [0u8; VERTEX_BLOCK_SIZE_BYTES];

    let vertex_count_aligned = (vertex_count + BYTE_GROUP_SIZE - 1) & !(BYTE_GROUP_SIZE - 1);
    debug_assert!(vertex_count <= vertex_count_aligned);

    let control_size = if version == 0 { 0 } else { vertex_size / 4 };
    if data_end - *pos < control_size {
        return false;
    }

    let control = *pos;
    *pos += control_size;

    let base = data.as_ptr();
    let bufp = buffer.as_mut_ptr();

    let mut k = 0usize;
    while k < vertex_size {
        let ctrl_byte = if version == 0 {
            0
        } else {
            data[control + k / 4]
        };

        for j in 0..4usize {
            let ctrl = (u32::from(ctrl_byte) >> (j * 2)) & 3;
            let column = j * vertex_count_aligned;

            if ctrl == 3 {
                // literal encoding; safe to over-copy due to tail
                if data_end - *pos < vertex_count_aligned {
                    return false;
                }

                // SAFETY: the check above guarantees vertex_count_aligned
                // readable bytes; the scratch column has as many writable ones
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        base.add(*pos),
                        bufp.add(column),
                        vertex_count_aligned,
                    );
                }
                *pos += vertex_count;
            } else if ctrl == 2 {
                // zero encoding
                // SAFETY: the scratch column has vertex_count_aligned bytes
                unsafe { bufp.add(column).write_bytes(0, vertex_count_aligned) };
            } else {
                // for v0, headers are mapped to 0..3; for v1, headers are mapped to 4..8
                let hshift = if version == 0 { 0 } else { 4 + ctrl };

                // SAFETY: decode_bytes_simd only ever reads within the
                // position it returns or reports failure
                let next = unsafe {
                    decode_bytes_simd::<D>(
                        base,
                        *pos,
                        data_end,
                        bufp.add(column),
                        vertex_count_aligned,
                        hshift,
                    )
                };
                let Some(next) = next else { return false };
                *pos = next;
            }
        }

        let channel = channels.map_or(0, |c| c[k / 4]);

        // channels come from the stream; an invalid channel type rejects the stream
        // SAFETY: scratch/transposed/last_vertex offsets are bounded by
        // vertex_count_aligned/vertex_size <= 256, as in the scalar kernel
        unsafe {
            match channel & 3 {
                0 => {
                    decode_deltas4_simd::<0>(
                        bufp,
                        transposed.as_mut_ptr().add(k),
                        vertex_count_aligned,
                        vertex_size,
                        last_vertex.as_ptr().add(k),
                        0,
                    );
                }
                1 => {
                    decode_deltas4_simd::<1>(
                        bufp,
                        transposed.as_mut_ptr().add(k),
                        vertex_count_aligned,
                        vertex_size,
                        last_vertex.as_ptr().add(k),
                        0,
                    );
                }
                2 => {
                    decode_deltas4_simd::<2>(
                        bufp,
                        transposed.as_mut_ptr().add(k),
                        vertex_count_aligned,
                        vertex_size,
                        last_vertex.as_ptr().add(k),
                        u32::from((32 - (channel >> 4)) & 31),
                    );
                }
                _ => return false, // invalid channel type
            }
        }

        k += 4;
    }

    vertex_data[..vertex_count * vertex_size]
        .copy_from_slice(&transposed[..vertex_count * vertex_size]);

    last_vertex[..vertex_size]
        .copy_from_slice(&transposed[vertex_size * (vertex_count - 1)..][..vertex_size]);

    true
}

// port shape: mirrors the upstream parameter list (C-parity, ADR-0005)
#[cfg(not(feature = "simd-avx512"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "ssse3,popcnt")]
unsafe fn decode_vertex_block_sse(
    data: &[u8],
    pos: &mut usize,
    data_end: usize,
    vertex_data: &mut [u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &mut [u8; 256],
    channels: Option<&[u8]>,
    version: i32,
) -> bool {
    // SAFETY: the SSSE3+POPCNT contract is the `simd` feature's requirement
    unsafe {
        decode_vertex_block_simd::<GroupSse>(
            data,
            pos,
            data_end,
            vertex_data,
            vertex_count,
            vertex_size,
            last_vertex,
            channels,
            version,
        )
    }
}

// port shape: mirrors the upstream parameter list (C-parity, ADR-0005)
#[cfg(feature = "simd-avx512")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "ssse3,popcnt,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2")]
unsafe fn decode_vertex_block_avx512(
    data: &[u8],
    pos: &mut usize,
    data_end: usize,
    vertex_data: &mut [u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &mut [u8; 256],
    channels: Option<&[u8]>,
    version: i32,
) -> bool {
    // SAFETY: the AVX512-VBMI2 contract is the `simd-avx512` feature's requirement
    unsafe {
        decode_vertex_block_simd::<GroupAvx512>(
            data,
            pos,
            data_end,
            vertex_data,
            vertex_count,
            vertex_size,
            last_vertex,
            channels,
            version,
        )
    }
}

/// Compile-time kernel selection (ADR-0012): `simd` assumes the target has
/// SSSE3+POPCNT (upstream compiled with `-mssse3 -mpopcnt` has no cpuid
/// fallback either); `simd-avx512` mirrors the `SIMD_AVX` compile gate and
/// must only run on AVX512 hardware.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_vertex_block(
    data: &[u8],
    pos: &mut usize,
    data_end: usize,
    vertex_data: &mut [u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &mut [u8; 256],
    channels: Option<&[u8]>,
    version: i32,
) -> bool {
    // SAFETY: the ISA contract is the enabling feature's documented requirement
    unsafe {
        decode_vertex_block_kernel(
            data,
            pos,
            data_end,
            vertex_data,
            vertex_count,
            vertex_size,
            last_vertex,
            channels,
            version,
        )
    }
}

#[cfg(not(feature = "simd-avx512"))]
use decode_vertex_block_sse as decode_vertex_block_kernel;

#[cfg(feature = "simd-avx512")]
use decode_vertex_block_avx512 as decode_vertex_block_kernel;
