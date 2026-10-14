//! SIMD vertex-decode kernels, wasm32 SIMD128 (ADR-0012).
//!
//! Upstream: `vertexcodec.cpp` WASM variants (`decodeShuffleMask`,
//! `wasmMoveMask`, `decodeBytesGroupSimd`, `transpose8`, `unzigzag8/16`,
//! `rotate32`, `decodeBytesSimd`, `decodeDeltas4Simd`,
//! `decodeVertexBlockSimd`) at the pinned revision, ported line-for-line.
//! SIMD128 is not part of the wasm baseline, so the kernel requires the
//! `simd128` target feature at compile time (`#[target_feature]`), mirroring
//! upstream's `__wasm_simd128__` gate; there is no runtime detection.
//! Byte-identical to the scalar decoder
//! ([ADR-0010](../../docs/adr/0010-scalar-profile-fixtures.md)); the golden
//! suite is the differential oracle, run in the GitHub CI tier.

#![allow(unsafe_code)]

use core::arch::wasm32::*;

use super::decode_tables::{GROUP_COUNT, GROUP_SHUFFLE};
use super::{
    BYTE_GROUP_DECODE_LIMIT, BYTE_GROUP_SIZE, VERTEX_BLOCK_MAX_SIZE, VERTEX_BLOCK_SIZE_BYTES,
};

// upstream `decodeShuffleMask`: combines the per-half shuffle masks; the
// second half is shifted by popcount(mask0) via a splat of the count table
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn decode_shuffle_mask(mask0: u8, mask1: u8) -> v128 {
    // SAFETY: table indices are mask bytes < 256; the 16-byte table loads
    // overread the 8-byte rows within the 2048-byte table, mirroring upstream
    unsafe {
        let sm0 = v128_load(GROUP_SHUFFLE[mask0 as usize].as_ptr().cast::<v128>());
        let sm1 = v128_load(GROUP_SHUFFLE[mask1 as usize].as_ptr().cast::<v128>());

        let sm1off = v128_load8_splat(GROUP_COUNT.as_ptr().add(mask0 as usize));
        let sm1r = i8x16_add(sm1, sm1off);

        i64x2_shuffle::<0, 2>(sm0, sm1r)
    }
}

// upstream `wasmMoveMask`: magic constant found using z3 SMT assuming mask has
// 8 groups of 0xff or 0x00
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn wasm_move_mask(mask: v128) -> (u8, u8) {
    const MAGIC: u64 = 0x0001_0307_0f1f_3f80;

    let m0 =
        (u64::from_ne_bytes(i64x2_extract_lane::<0>(mask).to_ne_bytes()).wrapping_mul(MAGIC)) >> 56;
    let m1 =
        (u64::from_ne_bytes(i64x2_extract_lane::<1>(mask).to_ne_bytes()).wrapping_mul(MAGIC)) >> 56;

    (m0 as u8, m1 as u8)
}

// the kernel reads at most `kByteGroupDecodeLimit` (24) bytes from `data`:
// every case's rest loads end at data + 24 or less; every caller checks the
// same bound per group (or 4x24 shared in the fast path, since a group
// advances at most 23 bytes), which is exactly upstream's decodeBytesSimd
// invariant
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn decode_bytes_group_simd(data: *const u8, buffer: *mut u8, hbits: u32) -> *const u8 {
    // SAFETY: caller guarantees 24 readable bytes at `data` and 16 writable
    // bytes at `buffer` (upstream decodeBytesSimd invariant, see above);
    // `hbits` is in 0..=8 by construction so the match is total; the simd128
    // contract is the compile-time target requirement.
    unsafe {
        match hbits {
            0 | 4 => {
                let result = i8x16_splat(0);

                v128_store(buffer.cast::<v128>(), result);

                data
            }

            1 | 6 => {
                let sel2 = v128_load(data.cast::<v128>());
                let rest = v128_load(data.add(4).cast::<v128>());

                let sel22 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                    u16x8_shr(sel2, 4),
                    sel2,
                );
                let sel2222 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                    u16x8_shr(sel22, 2),
                    sel22,
                );
                let sel = v128_and(sel2222, i8x16_splat(3));

                let mask = i8x16_eq(sel, i8x16_splat(3));

                let (mask0, mask1) = wasm_move_mask(mask);

                let shuf = decode_shuffle_mask(mask0, mask1);
                let result = v128_bitselect(i8x16_swizzle(rest, shuf), sel, mask);

                v128_store(buffer.cast::<v128>(), result);

                data.add(
                    4 + GROUP_COUNT[mask0 as usize] as usize + GROUP_COUNT[mask1 as usize] as usize,
                )
            }

            2 | 7 => {
                let sel4 = v128_load(data.cast::<v128>());
                let rest = v128_load(data.add(8).cast::<v128>());

                let sel44 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                    u16x8_shr(sel4, 4),
                    sel4,
                );
                let sel = v128_and(sel44, i8x16_splat(15));

                let mask = i8x16_eq(sel, i8x16_splat(15));

                let (mask0, mask1) = wasm_move_mask(mask);

                let shuf = decode_shuffle_mask(mask0, mask1);
                let result = v128_bitselect(i8x16_swizzle(rest, shuf), sel, mask);

                v128_store(buffer.cast::<v128>(), result);

                data.add(
                    8 + GROUP_COUNT[mask0 as usize] as usize + GROUP_COUNT[mask1 as usize] as usize,
                )
            }

            3 | 8 => {
                let result = v128_load(data.cast::<v128>());

                v128_store(buffer.cast::<v128>(), result);

                data.add(16)
            }

            5 => {
                let rest = v128_load(data.add(2).cast::<v128>());

                let mask0 = *data;
                let mask1 = *data.add(1);

                let shuf = decode_shuffle_mask(mask0, mask1);
                let result = i8x16_swizzle(rest, shuf);

                v128_store(buffer.cast::<v128>(), result);

                data.add(
                    2 + GROUP_COUNT[mask0 as usize] as usize + GROUP_COUNT[mask1 as usize] as usize,
                )
            }

            // unreachable: hbits is in 0..=8 by construction
            _ => data,
        }
    }
}

// Upstream: `decodeBytesSimd` (WASM variant: no 0-bit/8-bit fast paths, those
// are SSE/AVX only). Position-based.
#[inline]
unsafe fn decode_bytes_simd(
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

        // SAFETY: each group reads/writes at most kByteGroupDecodeLimit bytes
        // and advances at most as much, within the shared 4x check
        unsafe {
            pos = decode_bytes_group_simd(
                base.add(pos),
                buffer.add(i),
                hshift + u32::from(header_byte & 3),
            ) as usize
                - base as usize;
            pos = decode_bytes_group_simd(
                base.add(pos),
                buffer.add(i + BYTE_GROUP_SIZE),
                hshift + u32::from((header_byte >> 2) & 3),
            ) as usize
                - base as usize;
            pos = decode_bytes_group_simd(
                base.add(pos),
                buffer.add(i + BYTE_GROUP_SIZE * 2),
                hshift + u32::from((header_byte >> 4) & 3),
            ) as usize
                - base as usize;
            pos = decode_bytes_group_simd(
                base.add(pos),
                buffer.add(i + BYTE_GROUP_SIZE * 3),
                hshift + u32::from((header_byte >> 6) & 3),
            ) as usize
                - base as usize;
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
            decode_bytes_group_simd(
                base.add(pos),
                buffer.add(i),
                hshift + u32::from((header_byte >> ((header_offset % 4) * 2)) & 3),
            ) as usize
                - base as usize
        };

        i += BYTE_GROUP_SIZE;
    }

    Some(pos)
}

// upstream `transpose8` via the wasmx unpack shuffles
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn transpose8(x0: &mut v128, x1: &mut v128, x2: &mut v128, x3: &mut v128) {
    let t0 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(*x0, *x1);
    let t1 =
        i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(*x0, *x1);
    let t2 = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(*x2, *x3);
    let t3 =
        i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(*x2, *x3);

    *x0 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(t0, t2);
    *x1 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(t0, t2);
    *x2 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(t1, t3);
    *x3 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(t1, t3);
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn unzigzag8(v: v128) -> v128 {
    let xl = i8x16_neg(v128_and(v, i8x16_splat(1)));
    let xr = u8x16_shr(v, 1);

    v128_xor(xl, xr)
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn unzigzag16(v: v128) -> v128 {
    let xl = i16x8_neg(v128_and(v, i16x8_splat(1)));
    let xr = u16x8_shr(v, 1);

    v128_xor(xl, xr)
}

// upstream rotates by a runtime amount through wasm_i32x4_shl/v128_or with
// variable shift counts; wasm shift semantics (count >= 32 yields zero) match
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn rotate32(v: v128, r: u32) -> v128 {
    v128_or(i32x4_shl(v, r), i32x4_shr(v, 32 - r))
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn unzr<const CHANNEL: u32>(r: v128, rot: u32) -> v128 {
    // SAFETY: dispatches to register-only helpers
    unsafe {
        match CHANNEL {
            0 => unzigzag8(r),
            1 => unzigzag16(r),
            _ => rotate32(r, rot),
        }
    }
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn fixd<const CHANNEL: u32>(pi: v128, t: v128) -> v128 {
    match CHANNEL {
        0 => i8x16_add(pi, t),
        1 => i16x8_add(pi, t),
        _ => v128_xor(pi, t),
    }
}

// Upstream: `decodeDeltas4Simd` (WASM macro body). Each 16-vertex iteration
// runs four UNZR/GRP4/FIXD/SAVE levels, one per dword lane group.
#[target_feature(enable = "simd128")]
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
    // from offset k < vertex_size, and `last_vertex` has 16 readable bytes at
    // +k within the 256-byte array — all established by
    // decode_vertex_block_simd, mirroring upstream.
    unsafe {
        let mut pi = v128_load(last_vertex.cast::<v128>());
        let mut savep = transposed;

        let mut j = 0usize;
        while j < vertex_count_aligned {
            let mut r0 = v128_load(buffer.add(j).cast::<v128>());
            let mut r1 = v128_load(buffer.add(j + vertex_count_aligned).cast::<v128>());
            let mut r2 = v128_load(buffer.add(j + vertex_count_aligned * 2).cast::<v128>());
            let mut r3 = v128_load(buffer.add(j + vertex_count_aligned * 3).cast::<v128>());

            transpose8(&mut r0, &mut r1, &mut r2, &mut r3);

            r0 = unzr::<CHANNEL>(r0, rot);
            let t0 = r0;
            pi = fixd::<CHANNEL>(pi, t0);
            let t1 = i32x4_shuffle::<1, 1, 1, 1>(r0, r0);
            pi = fixd::<CHANNEL>(pi, t1);
            let t2 = i32x4_shuffle::<2, 2, 2, 2>(r0, r0);
            pi = fixd::<CHANNEL>(pi, t2);
            let t3 = i32x4_shuffle::<3, 3, 3, 3>(r0, r0);
            pi = fixd::<CHANNEL>(pi, t3);
            save_dword(&mut savep, vertex_size, t0);
            save_dword(&mut savep, vertex_size, t1);
            save_dword(&mut savep, vertex_size, t2);
            save_dword(&mut savep, vertex_size, t3);

            r1 = unzr::<CHANNEL>(r1, rot);
            let t0 = r1;
            pi = fixd::<CHANNEL>(pi, t0);
            let t1 = i32x4_shuffle::<1, 1, 1, 1>(r1, r1);
            pi = fixd::<CHANNEL>(pi, t1);
            let t2 = i32x4_shuffle::<2, 2, 2, 2>(r1, r1);
            pi = fixd::<CHANNEL>(pi, t2);
            let t3 = i32x4_shuffle::<3, 3, 3, 3>(r1, r1);
            pi = fixd::<CHANNEL>(pi, t3);
            save_dword(&mut savep, vertex_size, t0);
            save_dword(&mut savep, vertex_size, t1);
            save_dword(&mut savep, vertex_size, t2);
            save_dword(&mut savep, vertex_size, t3);

            r2 = unzr::<CHANNEL>(r2, rot);
            let t0 = r2;
            pi = fixd::<CHANNEL>(pi, t0);
            let t1 = i32x4_shuffle::<1, 1, 1, 1>(r2, r2);
            pi = fixd::<CHANNEL>(pi, t1);
            let t2 = i32x4_shuffle::<2, 2, 2, 2>(r2, r2);
            pi = fixd::<CHANNEL>(pi, t2);
            let t3 = i32x4_shuffle::<3, 3, 3, 3>(r2, r2);
            pi = fixd::<CHANNEL>(pi, t3);
            save_dword(&mut savep, vertex_size, t0);
            save_dword(&mut savep, vertex_size, t1);
            save_dword(&mut savep, vertex_size, t2);
            save_dword(&mut savep, vertex_size, t3);

            r3 = unzr::<CHANNEL>(r3, rot);
            let t0 = r3;
            pi = fixd::<CHANNEL>(pi, t0);
            let t1 = i32x4_shuffle::<1, 1, 1, 1>(r3, r3);
            pi = fixd::<CHANNEL>(pi, t1);
            let t2 = i32x4_shuffle::<2, 2, 2, 2>(r3, r3);
            pi = fixd::<CHANNEL>(pi, t2);
            let t3 = i32x4_shuffle::<3, 3, 3, 3>(r3, r3);
            pi = fixd::<CHANNEL>(pi, t3);
            save_dword(&mut savep, vertex_size, t0);
            save_dword(&mut savep, vertex_size, t1);
            save_dword(&mut savep, vertex_size, t2);
            save_dword(&mut savep, vertex_size, t3);

            j += 16;
        }
    }
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn save_dword(savep: &mut *mut u8, vertex_size: usize, t: v128) {
    // SAFETY: the save region is transposed[k + i*vertex_size .. +4] for
    // i < vertex_count_aligned, within the kVertexBlockSizeBytes scratch
    unsafe {
        v128_store32_lane::<0>(t, (*savep).cast::<u32>());
        *savep = (*savep).add(vertex_size);
    }
}

// Upstream: `decodeVertexBlockSimd`. Control flow mirrors the scalar
// decode_vertex_block with two differences: the varint columns decode through
// the SIMD group kernel over the aligned scratch layout, and the delta kernel
// processes 16 vertices per iteration.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn decode_vertex_block_simd(
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
                    decode_bytes_simd(
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

/// Compile-time kernel selection for wasm32: mirrors upstream's
/// `__wasm_simd128__` gate; the target must be compiled with `simd128`.
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
    // SAFETY: simd128 is the wasm32 target's compile-time requirement
    unsafe {
        decode_vertex_block_simd(
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
