//! Byte-group bit packing shared by the vertex codec encoder and decoder.
//!
//! Upstream: `vertexcodec.cpp` (`encodeBytesGroup*`, `encodeBytes`,
//! `decodeBytesGroup`, `decodeBytes`).

use super::{BYTE_GROUP_DECODE_LIMIT, BYTE_GROUP_SIZE};

// 1-bit groups are stored in reverse bit order so they decode alongside
// other groups with the same trick; C relies on unsigned wraparound and the
// >> 32 only reads bits below the wrap point, so wrapping_mul mirrors it
pub(crate) fn reverse_byte_bits(byte: u8) -> u8 {
    ((((byte as u64).wrapping_mul(0x8020_0802) & 0x0008_8442_2110).wrapping_mul(0x0001_0101_0101))
        >> 32) as u8
}

pub(crate) fn encode_bytes_group_zero(buffer: &[u8]) -> bool {
    buffer[..BYTE_GROUP_SIZE] == [0; BYTE_GROUP_SIZE]
}

pub(crate) fn encode_bytes_group_measure(buffer: &[u8], bits: usize) -> usize {
    debug_assert!(bits <= 8);

    if bits == 0 {
        return if encode_bytes_group_zero(buffer) {
            0
        } else {
            usize::MAX
        };
    }

    if bits == 8 {
        return BYTE_GROUP_SIZE;
    }

    let mut result = BYTE_GROUP_SIZE * bits / 8;

    let sentinel = ((1u16 << bits) - 1) as u8;

    for &b in &buffer[..BYTE_GROUP_SIZE] {
        result += usize::from(b >= sentinel);
    }

    result
}

pub(crate) fn encode_bytes_group(out: &mut [u8], pos: &mut usize, buffer: &[u8], bits: usize) {
    debug_assert!(bits <= 8);
    const _: () = assert!(BYTE_GROUP_SIZE.is_multiple_of(8));

    if bits == 0 {
        return;
    }

    if bits == 8 {
        out[*pos..*pos + BYTE_GROUP_SIZE].copy_from_slice(&buffer[..BYTE_GROUP_SIZE]);
        *pos += BYTE_GROUP_SIZE;
        return;
    }

    let byte_size = 8 / bits;
    debug_assert!(BYTE_GROUP_SIZE.is_multiple_of(byte_size));

    // fixed portion: bits bits for each value
    // variable portion: full byte for each out-of-range value (using 1...1 as sentinel)
    let sentinel = ((1u16 << bits) - 1) as u8;

    let mut i = 0usize;
    while i < BYTE_GROUP_SIZE {
        let mut byte = 0u8;

        for k in 0..byte_size {
            let enc = if buffer[i + k] >= sentinel {
                sentinel
            } else {
                buffer[i + k]
            };

            byte <<= bits;
            byte |= enc;
        }

        // encode 1-bit groups in reverse bit order
        // this makes them faster to decode alongside other groups
        if bits == 1 {
            byte = reverse_byte_bits(byte);
        }

        out[*pos] = byte;
        *pos += 1;
        i += byte_size;
    }

    for &v in &buffer[..BYTE_GROUP_SIZE] {
        // branchless append of out-of-range values
        if v >= sentinel {
            out[*pos] = v;
            *pos += 1;
        }
    }
}

/// Packs `buffer` (a multiple of [`BYTE_GROUP_SIZE`]) into `out` at `*pos`.
/// Returns false if the output limit is hit.
pub(crate) fn encode_bytes(
    out: &mut [u8],
    pos: &mut usize,
    out_limit: usize,
    buffer: &[u8],
    buffer_size: usize,
    bits: &[usize; 4],
) -> bool {
    debug_assert!(buffer_size.is_multiple_of(BYTE_GROUP_SIZE));

    let header_pos = *pos;

    // round number of groups to 4 to get number of header bytes
    let header_size = (buffer_size / BYTE_GROUP_SIZE).div_ceil(4);

    if out_limit - *pos < header_size {
        return false;
    }

    *pos += header_size;

    let mut last_bits = -1i32;

    let mut i = 0usize;
    while i < buffer_size {
        if out_limit - *pos < BYTE_GROUP_DECODE_LIMIT {
            return false;
        }

        let mut best_bitk = 3usize;
        let mut best_size = encode_bytes_group_measure(&buffer[i..], bits[best_bitk]);

        for bitk in 0..3usize {
            let size = encode_bytes_group_measure(&buffer[i..], bits[bitk]);

            // favor consistent bit selection across groups, but never replace literals
            if size < best_size
                || (size == best_size && bits[bitk] as i32 == last_bits && bits[best_bitk] != 8)
            {
                best_bitk = bitk;
                best_size = size;
            }
        }

        let header_offset = i / BYTE_GROUP_SIZE;
        out[header_pos + header_offset / 4] |= (best_bitk as u8) << ((header_offset % 4) * 2);

        let best_bits = bits[best_bitk];
        encode_bytes_group(out, pos, &buffer[i..], best_bits);
        last_bits = best_bits as i32;

        i += BYTE_GROUP_SIZE;
    }

    true
}

// upstream compiles the scalar decode kernels only without SIMD or with a
// cpuid fallback (vertexcodec.cpp L581); the simd feature mirrors the former
#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
pub(crate) fn decode_bytes_group(
    data: &[u8],
    pos: &mut usize,
    buffer: &mut [u8],
    bpos: &mut usize,
    bits: usize,
) {
    match bits {
        0 => {
            buffer[*bpos..*bpos + BYTE_GROUP_SIZE].fill(0);
            *bpos += BYTE_GROUP_SIZE;
        }
        1 => {
            // 2 groups with 8 1-bit values in each byte (reversed from the order in other groups)
            let mut var = *pos + 2;

            for g in 0..2 {
                let mut byte = reverse_byte_bits(data[*pos + g]);

                for _ in 0..8 {
                    let enc = byte >> 7;
                    byte <<= 1;
                    let sentinel = enc == 1;
                    buffer[*bpos] = if sentinel { data[var] } else { enc };
                    *bpos += 1;
                    var += usize::from(sentinel);
                }
            }

            *pos = var;
        }
        2 => {
            // 4 groups with 4 2-bit values in each byte
            let mut var = *pos + 4;

            for g in 0..4 {
                let mut byte = data[*pos + g];

                for _ in 0..4 {
                    let enc = byte >> 6;
                    byte <<= 2;
                    let sentinel = enc == 3;
                    buffer[*bpos] = if sentinel { data[var] } else { enc };
                    *bpos += 1;
                    var += usize::from(sentinel);
                }
            }

            *pos = var;
        }
        4 => {
            // 8 groups with 2 4-bit values in each byte
            let mut var = *pos + 8;

            for g in 0..8 {
                let mut byte = data[*pos + g];

                for _ in 0..2 {
                    let enc = byte >> 4;
                    byte <<= 4;
                    let sentinel = enc == 15;
                    buffer[*bpos] = if sentinel { data[var] } else { enc };
                    *bpos += 1;
                    var += usize::from(sentinel);
                }
            }

            *pos = var;
        }
        8 => {
            buffer[*bpos..*bpos + BYTE_GROUP_SIZE]
                .copy_from_slice(&data[*pos..*pos + BYTE_GROUP_SIZE]);
            *pos += BYTE_GROUP_SIZE;
            *bpos += BYTE_GROUP_SIZE;
        }
        // bits are drawn from the kBits tables, which only contain 0/1/2/4/8
        _ => {}
    }
}

/// Unpacks `buffer_size` bytes from `data` at `*pos`. Returns false if the
/// input limit is hit.
#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
pub(crate) fn decode_bytes(
    data: &[u8],
    pos: &mut usize,
    data_end: usize,
    buffer: &mut [u8],
    buffer_size: usize,
    bits: &[usize],
) -> bool {
    debug_assert!(buffer_size.is_multiple_of(BYTE_GROUP_SIZE));

    // round number of groups to 4 to get number of header bytes
    let header_size = (buffer_size / BYTE_GROUP_SIZE).div_ceil(4);
    if data_end - *pos < header_size {
        return false;
    }

    let header_pos = *pos;
    *pos += header_size;

    let mut bpos = 0usize;

    let mut i = 0usize;
    while i < buffer_size {
        if data_end - *pos < BYTE_GROUP_DECODE_LIMIT {
            return false;
        }

        let header_offset = i / BYTE_GROUP_SIZE;
        let bitsk = (data[header_pos + header_offset / 4] >> ((header_offset % 4) * 2)) & 3;

        decode_bytes_group(data, pos, buffer, &mut bpos, bits[bitsk as usize]);

        i += BYTE_GROUP_SIZE;
    }

    true
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: encode/decodeBytesGroup*, encode/decodeBytes
    // (internal kernels; behavior pinned through the public vertex codec fixtures
    // and the roundtrip property here).
    use super::*;
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    use alloc::vec;

    fn reverse_reference(mut v: u8) -> u8 {
        let mut r = 0u8;
        for _ in 0..8 {
            r = (r << 1) | (v & 1);
            v >>= 1;
        }
        r
    }

    #[test]
    fn reverse_byte_bits_matches_reference() {
        for v in 0..=255u8 {
            assert_eq!(reverse_byte_bits(v), reverse_reference(v), "byte {v:#04x}");
        }
    }

    #[test]
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    fn group_roundtrip_all_widths() {
        // deterministic pseudo-random payload with sentinel-heavy segments
        let mut state = 0x1234_5678u32;
        let mut lcg = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };

        for bits in [0usize, 1, 2, 4, 8] {
            let mut src = [0u8; BYTE_GROUP_SIZE];
            for b in &mut src {
                // bias towards the sentinel to exercise the variable portion
                *b = if lcg() % 4 == 0 {
                    ((1u32 << bits.min(8)) - 1) as u8
                } else {
                    (lcg() & 0xff) as u8
                };
            }
            if bits == 0 {
                src = [0; BYTE_GROUP_SIZE];
            }

            let mut packed = [0u8; BYTE_GROUP_DECODE_LIMIT];
            let mut pos = 0usize;
            encode_bytes_group(&mut packed, &mut pos, &src, bits);
            assert!(pos <= BYTE_GROUP_DECODE_LIMIT);

            let mut dst = [0u8; BYTE_GROUP_SIZE];
            let mut dpos = 0usize;
            let mut bpos = 0usize;
            decode_bytes_group(&packed, &mut dpos, &mut dst, &mut bpos, bits);

            assert_eq!(dpos, pos, "consumed size (bits={bits})");
            assert_eq!(dst, src, "roundtrip (bits={bits})");
        }
    }

    #[test]
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    fn bytes_roundtrip_mixed_groups() {
        let mut state = 0xfeed_beefu32;
        let mut lcg = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };

        let size = BYTE_GROUP_SIZE * 8;
        let mut src = vec![0u8; size];
        for (i, b) in src.iter_mut().enumerate() {
            *b = if i % 5 == 0 {
                0
            } else if i % 7 == 0 {
                255
            } else {
                (lcg() & 0x0f) as u8
            };
        }

        let mut packed = vec![0u8; size * 2];
        let mut pos = 0usize;
        let limit = packed.len();
        assert!(encode_bytes(
            &mut packed,
            &mut pos,
            limit,
            &src,
            size,
            &[1, 2, 4, 8]
        ));

        let mut dst = vec![0u8; size];
        let mut dpos = 0usize;
        assert!(decode_bytes(
            &packed,
            &mut dpos,
            packed.len(),
            &mut dst,
            size,
            &[1, 2, 4, 8]
        ));
        assert_eq!(dpos, pos);
        assert_eq!(dst, src);
    }
}
