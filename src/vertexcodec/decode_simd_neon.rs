//! SIMD vertex-decode kernels, aarch64 NEON (ADR-0012).
//!
//! Upstream: `vertexcodec.cpp` NEON variants (`decodeBytesGroupSimd`,
//! `neonMoveMask`, `shuffleBytes`, `transpose8`, `unzigzag8/16`, `rotate32`,
//! `rebase`, `decodeBytesSimd`, `decodeDeltas4Simd`, `decodeVertexBlockSimd`)
//! at the pinned revision, ported line-for-line. NEON is baseline on aarch64,
//! so like upstream's `__aarch64__` build the kernel needs no detection; the
//! `simd` feature assumes it. Byte-identical to the scalar decoder
//! ([ADR-0010](../../docs/adr/0010-scalar-profile-fixtures.md)); the golden
//! suite is the differential oracle.
//!
//! This module contains `unsafe` (ADR-0012); each block carries a safety
//! comment referencing the upstream invariant it preserves. On x86-only hosts
//! this code compiles only under a cross-check target; hardware coverage is
//! the GitHub CI tier.

#![allow(unsafe_code)]

use core::arch::aarch64::*;

use super::decode_tables::{GROUP_COUNT, GROUP_SHUFFLE};
use super::{
    BYTE_GROUP_DECODE_LIMIT, BYTE_GROUP_SIZE, VERTEX_BLOCK_MAX_SIZE, VERTEX_BLOCK_SIZE_BYTES,
};

// the kernel reads at most `kByteGroupDecodeLimit` (24) bytes from `data`:
// every case's mask/rest loads end at data + 24 or less; every caller checks
// the same bound per group (or 4x24 shared in the fast path, since a group
// advances at most 23 bytes), which is exactly upstream's decodeBytesSimd
// invariant
#[inline]
unsafe fn decode_bytes_group_simd(data: *const u8, buffer: *mut u8, hbits: u32) -> *const u8 {
    // SAFETY: caller guarantees 24 readable bytes at `data` and 16 writable
    // bytes at `buffer` (upstream decodeBytesSimd invariant, see above);
    // `hbits` is in 0..=8 by construction so the switch below is total.
    unsafe {
        match hbits {
            0 | 4 => {
                let result = vdupq_n_u8(0);

                vst1q_u8(buffer, result);

                data
            }

            1 | 6 => {
                let data32 = data.cast::<u32>().read_unaligned();
                let data32 = data32 & (data32 >> 1);

                // arrange bits such that low bits of nibbles of data64 contain
                // all 2-bit elements of data32
                let data64 = ((data32 as u64) << 30) | data32 as u64;

                // adds all 1-bit nibbles together; the sum fits in 4 bits
                // because datacnt=16 would have used a cheaper mode
                let datacnt = (((data64 & 0x1111_1111_1111_1111)
                    .wrapping_mul(0x1111_1111_1111_1111))
                    >> 60) as usize;

                let sel2 = vld1_u8(data);
                let sel22 = vzip1_u8(vshr_n_u8::<4>(sel2), sel2);
                let sel_lo = vzip1_u8(vshr_n_u8::<2>(sel22), sel22);
                let sel_hi = vzip2_u8(vshr_n_u8::<2>(sel22), sel22);
                let sel = vandq_u8(vcombine_u8(sel_lo, sel_hi), vdupq_n_u8(3));

                let mask = vceqq_u8(sel, vdupq_n_u8(3));
                let (mask0, mask1) = neon_move_mask(mask);

                let rest0 = vld1_u8(data.add(4));
                let rest1 = vld1_u8(data.add(4 + GROUP_COUNT[mask0 as usize] as usize));

                let result = vbslq_u8(mask, shuffle_bytes(mask0, mask1, rest0, rest1), sel);

                vst1q_u8(buffer, result);

                data.add(4 + datacnt)
            }

            2 | 7 => {
                let mut data64 = data.cast::<u64>().read_unaligned();
                data64 &= data64 >> 1;
                data64 &= data64 >> 2;

                // adds all 1-bit nibbles together; the sum fits in 4 bits
                // because datacnt=16 would have used a cheaper mode
                let datacnt = (((data64 & 0x1111_1111_1111_1111)
                    .wrapping_mul(0x1111_1111_1111_1111))
                    >> 60) as usize;

                let sel4 = vld1_u8(data);
                let sel_lo = vzip1_u8(vshr_n_u8::<4>(sel4), vand_u8(sel4, vdup_n_u8(15)));
                let sel_hi = vzip2_u8(vshr_n_u8::<4>(sel4), vand_u8(sel4, vdup_n_u8(15)));
                let sel = vcombine_u8(sel_lo, sel_hi);

                let mask = vceqq_u8(sel, vdupq_n_u8(15));
                let (mask0, mask1) = neon_move_mask(mask);

                let rest0 = vld1_u8(data.add(8));
                let rest1 = vld1_u8(data.add(8 + GROUP_COUNT[mask0 as usize] as usize));

                let result = vbslq_u8(mask, shuffle_bytes(mask0, mask1, rest0, rest1), sel);

                vst1q_u8(buffer, result);

                data.add(8 + datacnt)
            }

            3 | 8 => {
                let result = vld1q_u8(data);

                vst1q_u8(buffer, result);

                data.add(16)
            }

            5 => {
                let mask0 = *data;
                let mask1 = *data.add(1);

                let rest0 = vld1_u8(data.add(2));
                let rest1 = vld1_u8(data.add(2 + GROUP_COUNT[mask0 as usize] as usize));

                let result = shuffle_bytes(mask0, mask1, rest0, rest1);

                vst1q_u8(buffer, result);

                data.add(
                    2 + GROUP_COUNT[mask0 as usize] as usize + GROUP_COUNT[mask1 as usize] as usize,
                )
            }

            // unreachable: hbits is in 0..=8 by construction
            _ => data,
        }
    }
}

// upstream `shuffleBytes`: vtbl1_u8 does not exist on aarch64; the paired
// 8-byte table lookups are expressed with vqtbl1q over zero-padded halves
// (out-of-range indices read as zero, same as vtbl)
#[inline]
unsafe fn shuffle_bytes(mask0: u8, mask1: u8, rest0: uint8x8_t, rest1: uint8x8_t) -> uint8x16_t {
    // SAFETY: pure register ops on values derived above
    unsafe {
        let sm0 = vld1_u8(GROUP_SHUFFLE[mask0 as usize].as_ptr());
        let sm1 = vld1_u8(GROUP_SHUFFLE[mask1 as usize].as_ptr());

        let zero = vdup_n_u8(0);
        let r0 = vget_low_u8(vqtbl1q_u8(vcombine_u8(rest0, zero), vcombine_u8(sm0, zero)));
        let r1 = vget_low_u8(vqtbl1q_u8(vcombine_u8(rest1, zero), vcombine_u8(sm1, zero)));

        vcombine_u8(r0, r1)
    }
}

// upstream `neonMoveMask`: magic constant found using z3 SMT assuming mask has
// 8 groups of 0xff or 0x00
#[inline]
unsafe fn neon_move_mask(mask: uint8x16_t) -> (u8, u8) {
    // SAFETY: pure register ops
    unsafe {
        const MAGIC: u64 = 0x0001_0307_0f1f_3f80;
        let mask2 = vreinterpretq_u64_u8(mask);

        let m0 = (vgetq_lane_u64::<0>(mask2).wrapping_mul(MAGIC)) >> 56;
        let m1 = (vgetq_lane_u64::<1>(mask2).wrapping_mul(MAGIC)) >> 56;

        (m0 as u8, m1 as u8)
    }
}

// Upstream: `decodeBytesSimd` (NEON variant: no 0-bit/8-bit fast paths, those
// are SSE/AVX only). Position-based; the group decoder is monomorphized in.
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

#[inline]
unsafe fn transpose8(
    x0: &mut uint8x16_t,
    x1: &mut uint8x16_t,
    x2: &mut uint8x16_t,
    x3: &mut uint8x16_t,
) {
    // SAFETY: pure register ops
    unsafe {
        let t01_lo = vzip1q_u8(*x0, *x1);
        let t01_hi = vzip2q_u8(*x0, *x1);
        let t23_lo = vzip1q_u8(*x2, *x3);
        let t23_hi = vzip2q_u8(*x2, *x3);

        *x0 = vreinterpretq_u8_u16(vzip1q_u16(
            vreinterpretq_u16_u8(t01_lo),
            vreinterpretq_u16_u8(t23_lo),
        ));
        *x1 = vreinterpretq_u8_u16(vzip2q_u16(
            vreinterpretq_u16_u8(t01_lo),
            vreinterpretq_u16_u8(t23_lo),
        ));
        *x2 = vreinterpretq_u8_u16(vzip1q_u16(
            vreinterpretq_u16_u8(t01_hi),
            vreinterpretq_u16_u8(t23_hi),
        ));
        *x3 = vreinterpretq_u8_u16(vzip2q_u16(
            vreinterpretq_u16_u8(t01_hi),
            vreinterpretq_u16_u8(t23_hi),
        ));
    }
}

#[inline]
unsafe fn unzigzag8(v: uint8x16_t) -> uint8x16_t {
    // SAFETY: pure register ops
    unsafe {
        let xl = vsubq_u8(vdupq_n_u8(0), vandq_u8(v, vdupq_n_u8(1)));
        let xr = vshrq_n_u8::<1>(v);

        veorq_u8(xl, xr)
    }
}

#[inline]
unsafe fn unzigzag16(v: uint8x16_t) -> uint8x16_t {
    // SAFETY: pure register ops
    unsafe {
        let vv = vreinterpretq_u16_u8(v);
        let xl = vsubq_u16(vdupq_n_u16(0), vandq_u16(vv, vdupq_n_u16(1)));
        let xr = vshrq_n_u16::<1>(vv);

        vreinterpretq_u8_u16(veorq_u16(xl, xr))
    }
}

// upstream shifts by a variable amount through vshlq with a signed lane count;
// a negative count shifts right
#[inline]
unsafe fn rotate32(v: uint8x16_t, r: u32) -> uint8x16_t {
    // SAFETY: pure register ops
    unsafe {
        let v32 = vreinterpretq_u32_u8(v);
        vreinterpretq_u8_u32(vorrq_u32(
            vshlq_u32(v32, vdupq_n_s32(r as i32)),
            vshlq_u32(v32, vdupq_n_s32(r as i32 - 32)),
        ))
    }
}

#[inline]
unsafe fn unzr<const CHANNEL: u32>(r: uint8x16_t, rot: u32) -> uint8x16_t {
    // SAFETY: dispatches to register-only helpers
    unsafe {
        match CHANNEL {
            0 => unzigzag8(r),
            1 => unzigzag16(r),
            _ => rotate32(r, rot),
        }
    }
}

#[inline]
unsafe fn fixd<const CHANNEL: u32>(pi: uint8x8_t, t: uint8x8_t) -> uint8x8_t {
    // SAFETY: pure register ops
    unsafe {
        match CHANNEL {
            0 => vadd_u8(pi, t),
            1 => vreinterpret_u8_u16(vadd_u16(vreinterpret_u16_u8(pi), vreinterpret_u16_u8(t))),
            _ => veor_u8(pi, t),
        }
    }
}

// upstream `rebase`: recomputes the accumulator from scratch instead of
// relying on the chained pi, shortening the dependency between iterations
// (SIMD_LATENCYOPT on Apple/Win32 aarch64 targets)
#[cfg(any(target_vendor = "apple", target_env = "msvc"))]
#[inline]
unsafe fn rebase<const CHANNEL: u32>(
    npi: uint8x8_t,
    r0: uint8x16_t,
    r1: uint8x16_t,
    r2: uint8x16_t,
    r3: uint8x16_t,
) -> uint8x8_t {
    // SAFETY: pure register ops
    unsafe {
        match CHANNEL {
            0 => {
                let rsum = vaddq_u8(vaddq_u8(r0, r1), vaddq_u8(r2, r3));
                let rsumx = vadd_u8(vget_low_u8(rsum), vget_high_u8(rsum));
                vadd_u8(vadd_u8(npi, rsumx), vext_u8(rsumx, rsumx, 4))
            }
            1 => {
                let rsum = vaddq_u16(
                    vaddq_u16(vreinterpretq_u16_u8(r0), vreinterpretq_u16_u8(r1)),
                    vaddq_u16(vreinterpretq_u16_u8(r2), vreinterpretq_u16_u8(r3)),
                );
                let rsumx = vadd_u16(vget_low_u16(rsum), vget_high_u16(rsum));
                vreinterpret_u8_u16(vadd_u16(
                    vadd_u16(vreinterpret_u16_u8(npi), rsumx),
                    vext_u16(rsumx, rsumx, 2),
                ))
            }
            _ => {
                let rsum = veorq_u8(veorq_u8(r0, r1), veorq_u8(r2, r3));
                let rsumx = veor_u8(vget_low_u8(rsum), vget_high_u8(rsum));
                veor_u8(veor_u8(npi, rsumx), vext_u8(rsumx, rsumx, 4))
            }
        }
    }
}

// Upstream: `decodeDeltas4Simd` (NEON macro body). Each 16-vertex iteration
// runs four UNZR/GRP4/FIXD/SAVE levels, one per dword lane group. On the
// rebase targets the final chained pi is overwritten by rebase — the same
// dead store upstream has.
#[cfg_attr(
    any(target_vendor = "apple", target_env = "msvc"),
    allow(unused_assignments)
)]
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
        let mut pi = vreinterpret_u8_u32(vdup_n_u32(last_vertex.cast::<u32>().read_unaligned()));
        let mut savep = transposed;

        let mut j = 0usize;
        while j < vertex_count_aligned {
            let mut r0 = vld1q_u8(buffer.add(j));
            let mut r1 = vld1q_u8(buffer.add(j + vertex_count_aligned));
            let mut r2 = vld1q_u8(buffer.add(j + vertex_count_aligned * 2));
            let mut r3 = vld1q_u8(buffer.add(j + vertex_count_aligned * 3));

            transpose8(&mut r0, &mut r1, &mut r2, &mut r3);

            #[cfg(any(target_vendor = "apple", target_env = "msvc"))]
            let npi = pi;

            r0 = unzr::<CHANNEL>(r0, rot);
            let t0 = vget_low_u8(r0);
            let t1 = vreinterpret_u8_u32(vdup_lane_u32::<1>(vreinterpret_u32_u8(t0)));
            let t2 = vget_high_u8(r0);
            let t3 = vreinterpret_u8_u32(vdup_lane_u32::<1>(vreinterpret_u32_u8(t2)));
            // upstream FIXD assigns both: t##i = pi = op(pi, t##i); the saved
            // dword is the post-FIXD t, i.e. the accumulated prefix
            let t0 = fixd::<CHANNEL>(pi, t0);
            let t1 = fixd::<CHANNEL>(t0, t1);
            let t2 = fixd::<CHANNEL>(t1, t2);
            let t3 = fixd::<CHANNEL>(t2, t3);
            pi = t3;
            save_dword(&mut savep, vertex_size, t0);
            save_dword(&mut savep, vertex_size, t1);
            save_dword(&mut savep, vertex_size, t2);
            save_dword(&mut savep, vertex_size, t3);

            r1 = unzr::<CHANNEL>(r1, rot);
            let t0 = vget_low_u8(r1);
            let t1 = vreinterpret_u8_u32(vdup_lane_u32::<1>(vreinterpret_u32_u8(t0)));
            let t2 = vget_high_u8(r1);
            let t3 = vreinterpret_u8_u32(vdup_lane_u32::<1>(vreinterpret_u32_u8(t2)));
            let t0 = fixd::<CHANNEL>(pi, t0);
            let t1 = fixd::<CHANNEL>(t0, t1);
            let t2 = fixd::<CHANNEL>(t1, t2);
            let t3 = fixd::<CHANNEL>(t2, t3);
            pi = t3;
            save_dword(&mut savep, vertex_size, t0);
            save_dword(&mut savep, vertex_size, t1);
            save_dword(&mut savep, vertex_size, t2);
            save_dword(&mut savep, vertex_size, t3);

            r2 = unzr::<CHANNEL>(r2, rot);
            let t0 = vget_low_u8(r2);
            let t1 = vreinterpret_u8_u32(vdup_lane_u32::<1>(vreinterpret_u32_u8(t0)));
            let t2 = vget_high_u8(r2);
            let t3 = vreinterpret_u8_u32(vdup_lane_u32::<1>(vreinterpret_u32_u8(t2)));
            let t0 = fixd::<CHANNEL>(pi, t0);
            let t1 = fixd::<CHANNEL>(t0, t1);
            let t2 = fixd::<CHANNEL>(t1, t2);
            let t3 = fixd::<CHANNEL>(t2, t3);
            pi = t3;
            save_dword(&mut savep, vertex_size, t0);
            save_dword(&mut savep, vertex_size, t1);
            save_dword(&mut savep, vertex_size, t2);
            save_dword(&mut savep, vertex_size, t3);

            r3 = unzr::<CHANNEL>(r3, rot);
            let t0 = vget_low_u8(r3);
            let t1 = vreinterpret_u8_u32(vdup_lane_u32::<1>(vreinterpret_u32_u8(t0)));
            let t2 = vget_high_u8(r3);
            let t3 = vreinterpret_u8_u32(vdup_lane_u32::<1>(vreinterpret_u32_u8(t2)));
            let t0 = fixd::<CHANNEL>(pi, t0);
            let t1 = fixd::<CHANNEL>(t0, t1);
            let t2 = fixd::<CHANNEL>(t1, t2);
            let t3 = fixd::<CHANNEL>(t2, t3);
            pi = t3;
            save_dword(&mut savep, vertex_size, t0);
            save_dword(&mut savep, vertex_size, t1);
            save_dword(&mut savep, vertex_size, t2);
            save_dword(&mut savep, vertex_size, t3);

            #[cfg(any(target_vendor = "apple", target_env = "msvc"))]
            {
                pi = rebase::<CHANNEL>(npi, r0, r1, r2, r3);
            }

            j += 16;
        }
    }
}

#[inline]
unsafe fn save_dword(savep: &mut *mut u8, vertex_size: usize, t: uint8x8_t) {
    // SAFETY: the save region is transposed[k + i*vertex_size .. +4] for
    // i < vertex_count_aligned, within the kVertexBlockSizeBytes scratch
    unsafe {
        vst1_lane_u32::<0>((*savep).cast::<u32>(), vreinterpret_u32_u8(t));
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

/// Compile-time kernel selection for aarch64: NEON is baseline, so like
/// upstream's `__aarch64__` build there is no detection and no fallback.
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
    // SAFETY: NEON is architecturally guaranteed on aarch64
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
