//! Vertex codec decoder half.
//!
//! Upstream: `vertexcodec.cpp` (`meshopt_decodeVertexBuffer`,
//! `meshopt_decodeVertexVersion` and the scalar decode kernels). Decoders
//! accept wire versions 0 and 1 and never panic on malformed input.

use super::bitstream::decode_bytes;
use super::{
    BYTE_GROUP_SIZE, DECODE_VERTEX_VERSION, K_BITS_V0, K_BITS_V1, TAIL_MIN_SIZE_V0,
    TAIL_MIN_SIZE_V1, VERTEX_BLOCK_MAX_SIZE, VERTEX_BLOCK_SIZE_BYTES, VERTEX_HEADER,
    get_vertex_block_size, rotate, unzigzag8, unzigzag16,
};

// the scratch buffer is reused per 4-byte group: decodeDeltas1 walks its
// columns internally (buffer += vertex_count * sizeof(T) per sub-iteration),
// so `buffer` is always the full scratch and tbase/lbase are the group offset
fn decode_deltas1_u8(
    buffer: &[u8],
    transposed: &mut [u8],
    tbase: usize,
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &[u8],
    lbase: usize,
) {
    let mut buf_off = 0usize;
    let mut k = 0usize;
    while k < 4 {
        let mut p = last_vertex[lbase + k];

        for i in 0..vertex_count {
            let v = buffer[buf_off + i];
            let r = unzigzag8(v).wrapping_add(p);

            transposed[tbase + k + i * vertex_size] = r;
            p = r;
        }

        buf_off += vertex_count;
        k += 1;
    }
}

fn decode_deltas1_u16(
    buffer: &[u8],
    transposed: &mut [u8],
    tbase: usize,
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &[u8],
    lbase: usize,
) {
    let mut buf_off = 0usize;
    let mut k = 0usize;
    while k < 4 {
        let mut p = u16::from_le_bytes([last_vertex[lbase + k], last_vertex[lbase + k + 1]]);

        for i in 0..vertex_count {
            let v = u16::from_le_bytes([buffer[buf_off + i], buffer[buf_off + i + vertex_count]]);
            let r = unzigzag16(v).wrapping_add(p);

            transposed[tbase + k + i * vertex_size] = r as u8;
            transposed[tbase + k + 1 + i * vertex_size] = (r >> 8) as u8;
            p = r;
        }

        buf_off += vertex_count * 2;
        k += 2;
    }
}

// port shape: mirrors the upstream function signature 1:1 (C-parity, ADR-0005)
#[allow(clippy::too_many_arguments)]
fn decode_deltas1_u32_xor(
    buffer: &[u8],
    transposed: &mut [u8],
    tbase: usize,
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &[u8],
    lbase: usize,
    rot: u32,
) {
    let mut p = u32::from_le_bytes([
        last_vertex[lbase],
        last_vertex[lbase + 1],
        last_vertex[lbase + 2],
        last_vertex[lbase + 3],
    ]);

    for i in 0..vertex_count {
        let v = u32::from_le_bytes([
            buffer[i],
            buffer[i + vertex_count],
            buffer[i + vertex_count * 2],
            buffer[i + vertex_count * 3],
        ]);
        let r = rotate(v, rot) ^ p;

        transposed[tbase + i * vertex_size] = r as u8;
        transposed[tbase + 1 + i * vertex_size] = (r >> 8) as u8;
        transposed[tbase + 2 + i * vertex_size] = (r >> 16) as u8;
        transposed[tbase + 3 + i * vertex_size] = (r >> 24) as u8;
        p = r;
    }
}

// port shape: mirrors the upstream function signature 1:1 (C-parity, ADR-0005)
#[allow(clippy::too_many_arguments)]
fn decode_vertex_block(
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

    let mut k = 0usize;
    while k < vertex_size {
        let ctrl_byte = if version == 0 {
            0
        } else {
            data[control + k / 4]
        };

        for j in 0..4usize {
            let ctrl = ((ctrl_byte >> (j * 2)) & 3) as usize;
            let column = j * vertex_count;

            if ctrl == 3 {
                // literal encoding
                if data_end - *pos < vertex_count {
                    return false;
                }

                buffer[column..column + vertex_count]
                    .copy_from_slice(&data[*pos..*pos + vertex_count]);
                *pos += vertex_count;
            } else if ctrl == 2 {
                // zero encoding
                buffer[column..column + vertex_count].fill(0);
            } else {
                let bits: &[usize] = if version == 0 {
                    &K_BITS_V0
                } else {
                    &K_BITS_V1[ctrl..ctrl + 4]
                };
                if !decode_bytes(
                    data,
                    pos,
                    data_end,
                    &mut buffer[column..column + vertex_count_aligned],
                    vertex_count_aligned,
                    bits,
                ) {
                    return false;
                }
            }
        }

        let channel = channels.map_or(0, |c| c[k / 4]);

        // channels come from the stream; an invalid channel type rejects the stream
        match channel & 3 {
            0 => decode_deltas1_u8(
                &buffer,
                &mut transposed,
                k,
                vertex_count,
                vertex_size,
                last_vertex,
                k,
            ),
            1 => decode_deltas1_u16(
                &buffer,
                &mut transposed,
                k,
                vertex_count,
                vertex_size,
                last_vertex,
                k,
            ),
            2 => decode_deltas1_u32_xor(
                &buffer,
                &mut transposed,
                k,
                vertex_count,
                vertex_size,
                last_vertex,
                k,
                u32::from((32 - (channel >> 4)) & 31),
            ),
            _ => return false, // invalid channel type
        }

        k += 4;
    }

    vertex_data[..vertex_count * vertex_size]
        .copy_from_slice(&transposed[..vertex_count * vertex_size]);

    last_vertex[..vertex_size]
        .copy_from_slice(&transposed[vertex_size * (vertex_count - 1)..][..vertex_size]);

    true
}

/// Upstream: `meshopt_decodeVertexVersion`. Returns the stream version
/// (`0..=1`), or -1 if the buffer does not carry a valid vertex codec header.
pub fn decode_vertex_version(buffer: &[u8]) -> i32 {
    if buffer.is_empty() {
        return -1;
    }

    let header = buffer[0];

    if header & 0xf0 != VERTEX_HEADER {
        return -1;
    }

    let version = i32::from(header & 0x0f);
    if version > DECODE_VERTEX_VERSION {
        return -1;
    }

    version
}

/// Upstream: `meshopt_decodeVertexBuffer`.
///
/// `destination` holds `vertex_count` vertices of `vertex_size` bytes.
/// Returns false on malformed input; the destination contents are unspecified.
pub fn decode_vertex_buffer(
    destination: &mut [u8],
    vertex_count: usize,
    vertex_size: usize,
    buffer: &[u8],
) -> bool {
    debug_assert!(vertex_size > 0 && vertex_size <= 256);
    debug_assert!(vertex_size.is_multiple_of(4));

    let mut last_vertex = [0u8; 256];

    if buffer.is_empty() {
        return false;
    }

    let data_header = buffer[0];

    if data_header & 0xf0 != VERTEX_HEADER {
        return false;
    }

    let version = i32::from(data_header & 0x0f);
    if version > DECODE_VERTEX_VERSION {
        return false;
    }

    let tail_size = vertex_size + if version == 0 { 0 } else { vertex_size / 4 };
    let tail_size_min = if version == 0 {
        TAIL_MIN_SIZE_V0
    } else {
        TAIL_MIN_SIZE_V1
    };
    let tail_size_pad = tail_size.max(tail_size_min);

    if buffer.len() - 1 < tail_size_pad {
        return false;
    }

    let tail = &buffer[buffer.len() - tail_size..];
    last_vertex[..vertex_size].copy_from_slice(&tail[..vertex_size]);

    let channels = if version == 0 {
        None
    } else {
        Some(&tail[vertex_size..vertex_size + vertex_size / 4])
    };

    let vertex_block_size = get_vertex_block_size(vertex_size);

    let mut pos = 1usize;
    let data_end = buffer.len();

    let mut vertex_offset = 0usize;

    while vertex_offset < vertex_count {
        let block_size = if vertex_offset + vertex_block_size < vertex_count {
            vertex_block_size
        } else {
            vertex_count - vertex_offset
        };

        let base = vertex_offset * vertex_size;
        if !decode_vertex_block(
            buffer,
            &mut pos,
            data_end,
            &mut destination[base..],
            block_size,
            vertex_size,
            &mut last_vertex,
            channels,
            version,
        ) {
            return false;
        }

        vertex_offset += block_size;
    }

    if data_end - pos != tail_size_pad {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_decodeVertexBuffer,
    // meshopt_decodeVertexVersion.
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

        fn take(&mut self, n: usize) -> &'a [u8] {
            let s = &self.data[self.pos..self.pos + n];
            self.pos += n;
            s
        }

        fn u32(&mut self) -> u32 {
            let mut a = [0u8; 4];
            a.copy_from_slice(self.take(4));
            u32::from_le_bytes(a)
        }

        fn i32(&mut self) -> i32 {
            self.u32() as i32
        }

        fn bytes(&mut self, n: usize) -> &'a [u8] {
            self.take(n)
        }
    }

    #[test]
    fn golden_decode_vertex_buffer() {
        let data = include_bytes!("../../tests/data/vc_dec.bin");
        let mut c = Cursor::new(data);

        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let vertex_size = c.u32() as usize;
            let buffer_size = c.u32() as usize;
            let expected_version = c.i32();
            let stream = c.bytes(buffer_size).to_vec();
            let result = c.i32();
            let expected = if result == 0 {
                c.bytes(vertex_count * vertex_size).to_vec()
            } else {
                Vec::new()
            };

            assert_eq!(
                decode_vertex_version(&stream),
                expected_version,
                "decode_vertex_version (n={vertex_count}, s={vertex_size})"
            );

            let mut destination = vec![0u8; vertex_count * vertex_size];
            let ok = decode_vertex_buffer(&mut destination, vertex_count, vertex_size, &stream);
            assert_eq!(
                ok,
                result == 0,
                "decode result (n={vertex_count}, s={vertex_size})"
            );
            if ok {
                assert_eq!(
                    destination, expected,
                    "decoded bytes (n={vertex_count}, s={vertex_size})"
                );
            }
        }
    }

    #[test]
    fn hostile_decode_vertex_buffer() {
        const _LEN_CHECK: usize = 86773;
        let data = include_bytes!("../../tests/data/vc_hostile.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);

        while c.pos < data.len() {
            let buffer_size = c.u32() as usize;
            let vertex_count = c.u32() as usize;
            let vertex_size = c.u32() as usize;
            let stream = c.bytes(buffer_size).to_vec();
            let result = c.i32();
            let expected = if result == 0 {
                c.bytes(vertex_count * vertex_size).to_vec()
            } else {
                Vec::new()
            };

            let mut destination = vec![0u8; vertex_count * vertex_size];
            let ok = decode_vertex_buffer(&mut destination, vertex_count, vertex_size, &stream);
            assert_eq!(ok, result == 0, "hostile result (size={buffer_size})");
            if ok {
                assert_eq!(destination, expected, "hostile bytes (size={buffer_size})");
            }
        }
    }

    #[test]
    fn decode_vertex_version_probe() {
        assert_eq!(decode_vertex_version(&[]), -1);
        assert_eq!(decode_vertex_version(&[0x00]), -1);
        assert_eq!(decode_vertex_version(&[0xa2]), -1);
        assert_eq!(decode_vertex_version(&[0xa1]), 1);
        assert_eq!(decode_vertex_version(&[0xa0]), 0);
        assert_eq!(decode_vertex_version(&[0xb1]), -1);
        assert_eq!(decode_vertex_version(&[0xe1]), -1);
    }

    #[test]
    fn decode_vertex_buffer_minimal_rejects() {
        assert!(!decode_vertex_buffer(&mut [], 0, 4, &[]));
        assert!(!decode_vertex_buffer(&mut [0u8; 64], 1, 4, &[0xa1]));
    }
}
