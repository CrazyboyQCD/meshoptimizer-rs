//! Vertex codec encoder half.
//!
//! Upstream: `vertexcodec.cpp` (`meshopt_encodeVertexBuffer`,
//! `meshopt_encodeVertexBufferBound`, `meshopt_encodeVertexBufferLevel` and
//! the encode-side kernels). Emits wire version 1 only
//! ([ADR-0009](../../docs/adr/0009-encoder-version-hardcoded.md)).

use super::bitstream::{encode_bytes, encode_bytes_group_measure, encode_bytes_group_zero};
use super::{
    BYTE_GROUP_SIZE, ENCODE_DEFAULT_LEVEL, K_BITS_V1, VERTEX_BLOCK_MAX_SIZE, get_vertex_block_size,
    rotate, zigzag8, zigzag16,
};

fn encode_deltas_u8(
    buffer: &mut [u8],
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &[u8; 256],
    k: usize,
) {
    let mut p = last_vertex[k];
    let mut vertex = k;

    for slot in buffer.iter_mut().take(vertex_count) {
        let v = vertex_data[vertex];
        let d = zigzag8(v.wrapping_sub(p));

        *slot = d;
        p = v;
        vertex += vertex_size;
    }
}

fn encode_deltas_u16(
    buffer: &mut [u8],
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &[u8; 256],
    k: usize,
) {
    let k0 = k & !1;
    let ks = (k & 1) * 8;

    let mut p = u16::from_le_bytes([last_vertex[k0], last_vertex[k0 + 1]]);
    let mut vertex = k0;

    for slot in buffer.iter_mut().take(vertex_count) {
        let v = u16::from_le_bytes([vertex_data[vertex], vertex_data[vertex + 1]]);
        let d = zigzag16(v.wrapping_sub(p));

        *slot = (d >> ks) as u8;
        p = v;
        vertex += vertex_size;
    }
}

fn encode_deltas_u32_xor(
    buffer: &mut [u8],
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &[u8; 256],
    k: usize,
    rot: u32,
) {
    let k0 = k & !3;
    let ks = (k & 3) * 8;

    let mut p = u32::from_le_bytes([
        last_vertex[k0],
        last_vertex[k0 + 1],
        last_vertex[k0 + 2],
        last_vertex[k0 + 3],
    ]);
    let mut vertex = k0;

    for slot in buffer.iter_mut().take(vertex_count) {
        let v = u32::from_le_bytes([
            vertex_data[vertex],
            vertex_data[vertex + 1],
            vertex_data[vertex + 2],
            vertex_data[vertex + 3],
        ]);
        let d = rotate(v ^ p, rot);

        *slot = (d >> ks) as u8;
        p = v;
        vertex += vertex_size;
    }
}

fn encode_deltas(
    buffer: &mut [u8],
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &[u8; 256],
    k: usize,
    channel: u8,
) {
    // channels are produced by estimate_channel (0, 1, or 2 | rot << 4)
    match channel & 3 {
        0 => encode_deltas_u8(
            buffer,
            vertex_data,
            vertex_count,
            vertex_size,
            last_vertex,
            k,
        ),
        1 => encode_deltas_u16(
            buffer,
            vertex_data,
            vertex_count,
            vertex_size,
            last_vertex,
            k,
        ),
        2 => encode_deltas_u32_xor(
            buffer,
            vertex_data,
            vertex_count,
            vertex_size,
            last_vertex,
            k,
            u32::from(channel >> 4),
        ),
        _ => {}
    }
}

fn estimate_bits(v: u8) -> usize {
    if v <= 15 {
        if v <= 3 { usize::from(v != 0) * 2 } else { 4 }
    } else {
        8
    }
}

fn estimate_rotate(
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    k: usize,
    group_size: usize,
) -> u32 {
    let mut sizes = [0usize; 8];

    let mut vertex = k;
    let mut last = u32::from_le_bytes([
        vertex_data[vertex],
        vertex_data[vertex + 1],
        vertex_data[vertex + 2],
        vertex_data[vertex + 3],
    ]);

    let mut i = 0usize;
    while i < vertex_count {
        let mut bitg = 0u32;

        // calculate bit consistency mask for the group
        let mut j = 0usize;
        while j < group_size && i + j < vertex_count {
            let v = u32::from_le_bytes([
                vertex_data[vertex],
                vertex_data[vertex + 1],
                vertex_data[vertex + 2],
                vertex_data[vertex + 3],
            ]);
            let d = v ^ last;

            bitg |= d;
            last = v;
            vertex += vertex_size;
            j += 1;
        }

        for (rot, size) in sizes.iter_mut().enumerate() {
            let bitr = rotate(bitg, rot as u32);

            *size += estimate_bits(bitr as u8)
                + estimate_bits((bitr >> 8) as u8)
                + estimate_bits((bitr >> 16) as u8)
                + estimate_bits((bitr >> 24) as u8);
        }

        i += group_size;
    }

    let mut best_rot = 0u32;
    for rot in 1..8u32 {
        best_rot = if sizes[rot as usize] < sizes[best_rot as usize] {
            rot
        } else {
            best_rot
        };
    }

    best_rot
}

// port shape: mirrors the upstream function signature 1:1 (C-parity, ADR-0005)
#[allow(clippy::too_many_arguments)]
fn estimate_channel(
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    k: usize,
    vertex_block_size: usize,
    block_skip: usize,
    max_channel: usize,
    xor_rot: u32,
) -> u8 {
    let mut block = [0u8; VERTEX_BLOCK_MAX_SIZE];
    let mut last_vertex = [0u8; 256];

    debug_assert!(max_channel <= 3);

    let mut sizes = [0usize; 3];

    let mut i = 0usize;
    while i < vertex_count {
        let block_size = if i + vertex_block_size < vertex_count {
            vertex_block_size
        } else {
            vertex_count - i
        };
        let block_size_aligned = (block_size + BYTE_GROUP_SIZE - 1) & !(BYTE_GROUP_SIZE - 1);

        let lv_src = if i == 0 { 0 } else { i - 1 } * vertex_size;
        last_vertex[..vertex_size].copy_from_slice(&vertex_data[lv_src..lv_src + vertex_size]);

        // we sometimes encode elements we didn't fill when rounding to kByteGroupSize
        if block_size < block_size_aligned {
            block[block_size..block_size_aligned].fill(0);
        }

        for (channel, size) in sizes.iter_mut().enumerate().take(max_channel) {
            for j in 0..4usize {
                encode_deltas(
                    &mut block,
                    &vertex_data[i * vertex_size..],
                    block_size,
                    vertex_size,
                    &last_vertex,
                    k + j,
                    (channel as u8) | ((xor_rot as u8) << 4),
                );

                let mut ig = 0usize;
                while ig < block_size {
                    // to maximize encoding performance we only evaluate 1/2/4/8 bit groups
                    let size1 = encode_bytes_group_measure(&block[ig..], 1);
                    let size2 = encode_bytes_group_measure(&block[ig..], 2);
                    let size4 = encode_bytes_group_measure(&block[ig..], 4);
                    let size8 = encode_bytes_group_measure(&block[ig..], 8);

                    let best_size = size1.min(size2).min(size4).min(size8);

                    *size += best_size;
                    ig += BYTE_GROUP_SIZE;
                }
            }
        }

        i += vertex_block_size * block_skip;
    }

    let mut best_channel = 0usize;
    for (channel, &size) in sizes.iter().enumerate().take(max_channel).skip(1) {
        best_channel = if size < sizes[best_channel] {
            channel
        } else {
            best_channel
        };
    }

    if best_channel == 2 {
        best_channel as u8 | ((xor_rot as u8) << 4)
    } else {
        best_channel as u8
    }
}

fn estimate_control_zero(buffer: &[u8], vertex_count_aligned: usize) -> bool {
    let mut i = 0usize;
    while i < vertex_count_aligned {
        if !encode_bytes_group_zero(&buffer[i..]) {
            return false;
        }
        i += BYTE_GROUP_SIZE;
    }

    true
}

fn estimate_control(
    buffer: &[u8],
    vertex_count: usize,
    vertex_count_aligned: usize,
    level: i32,
) -> usize {
    if estimate_control_zero(buffer, vertex_count_aligned) {
        return 2; // zero encoding
    }

    if level == 0 {
        return 1; // 1248 encoding in level 0 for encoding speed
    }

    // round number of groups to 4 to get number of header bytes
    let header_size = (vertex_count_aligned / BYTE_GROUP_SIZE).div_ceil(4);

    let mut est_bytes0 = header_size;
    let mut est_bytes1 = header_size;

    let mut i = 0usize;
    while i < vertex_count_aligned {
        // assumes kBitsV1[] = {0, 1, 2, 4, 8} for performance
        let size0 = encode_bytes_group_measure(&buffer[i..], 0);
        let size1 = encode_bytes_group_measure(&buffer[i..], 1);
        let size2 = encode_bytes_group_measure(&buffer[i..], 2);
        let size4 = encode_bytes_group_measure(&buffer[i..], 4);
        let size8 = encode_bytes_group_measure(&buffer[i..], 8);

        // both control modes have access to 1/2/4 bit encoding
        let size12 = size1.min(size2);
        let size124 = size12.min(size4);

        // each control mode has access to 0/8 bit encoding respectively
        est_bytes0 += size124.min(size0);
        est_bytes1 += size124.min(size8);

        i += BYTE_GROUP_SIZE;
    }

    // pick shortest control entry but prefer literal encoding
    if est_bytes0 < vertex_count || est_bytes1 < vertex_count {
        usize::from(est_bytes0 >= est_bytes1)
    } else {
        3 // literal encoding
    }
}

// port shape: mirrors the upstream function signature 1:1 (C-parity, ADR-0005)
#[allow(clippy::too_many_arguments)]
fn encode_vertex_block(
    out: &mut [u8],
    pos: &mut usize,
    out_limit: usize,
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &mut [u8; 256],
    channels: &[u8; 64],
    level: i32,
) -> bool {
    debug_assert!(vertex_count > 0 && vertex_count <= VERTEX_BLOCK_MAX_SIZE);
    debug_assert!(vertex_size.is_multiple_of(4));

    let mut buffer = [0u8; VERTEX_BLOCK_MAX_SIZE];

    let vertex_count_aligned = (vertex_count + BYTE_GROUP_SIZE - 1) & !(BYTE_GROUP_SIZE - 1);

    let control_size = vertex_size / 4;
    if out_limit - *pos < control_size {
        return false;
    }

    let control = *pos;
    *pos += control_size;

    for k in 0..vertex_size {
        encode_deltas(
            &mut buffer,
            vertex_data,
            vertex_count,
            vertex_size,
            last_vertex,
            k,
            channels[k / 4],
        );

        let ctrl = estimate_control(&buffer, vertex_count, vertex_count_aligned, level);

        debug_assert!(ctrl < 4);
        out[control + k / 4] |= (ctrl as u8) << ((k % 4) * 2);

        if ctrl == 3 {
            // literal encoding
            if out_limit - *pos < vertex_count {
                return false;
            }

            out[*pos..*pos + vertex_count].copy_from_slice(&buffer[..vertex_count]);
            *pos += vertex_count;
        } else if ctrl != 2 {
            // non-zero encoding
            let bits = [
                K_BITS_V1[ctrl],
                K_BITS_V1[ctrl + 1],
                K_BITS_V1[ctrl + 2],
                K_BITS_V1[ctrl + 3],
            ];
            if !encode_bytes(out, pos, out_limit, &buffer, vertex_count_aligned, &bits) {
                return false;
            }
        }
    }

    last_vertex[..vertex_size]
        .copy_from_slice(&vertex_data[vertex_size * (vertex_count - 1)..][..vertex_size]);

    true
}

/// Upstream: `meshopt_encodeVertexBufferLevel`. Encodes with the given
/// compression level (`0..=9`; only 0-3 change the output). Emits wire
/// version 1 (ADR-0009).
///
/// Returns the number of bytes written, or 0 if `buffer` is smaller than
/// [`encode_vertex_buffer_bound`].
pub fn encode_vertex_buffer_level(
    buffer: &mut [u8],
    vertices: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    level: i32,
) -> usize {
    debug_assert!(vertex_size > 0 && vertex_size <= 256);
    debug_assert!(vertex_size.is_multiple_of(4));
    debug_assert!((0..=9).contains(&level));

    let vertex_data = vertices;

    if buffer.is_empty() {
        return 0;
    }

    buffer[0] = super::VERTEX_HEADER | 1;

    let mut first_vertex = [0u8; 256];
    if vertex_count > 0 {
        first_vertex[..vertex_size].copy_from_slice(&vertex_data[..vertex_size]);
    }

    let mut last_vertex = [0u8; 256];
    last_vertex[..vertex_size].copy_from_slice(&first_vertex[..vertex_size]);

    let vertex_block_size = get_vertex_block_size(vertex_size);

    let mut channels = [0u8; 64];
    if level > 1 && vertex_count > 1 {
        for k in (0..vertex_size).step_by(4) {
            let rot = if level >= 3 {
                estimate_rotate(vertex_data, vertex_count, vertex_size, k, 16)
            } else {
                0
            };
            let channel = estimate_channel(
                vertex_data,
                vertex_count,
                vertex_size,
                k,
                vertex_block_size,
                3,
                if level >= 3 { 3 } else { 2 },
                rot,
            );

            debug_assert!(u32::from(channel) < 2 || (channel & 3) == 2);
            channels[k / 4] = channel;
        }
    }

    let out_limit = buffer.len();
    let mut pos = 1usize;

    let mut vertex_offset = 0usize;

    while vertex_offset < vertex_count {
        let block_size = if vertex_offset + vertex_block_size < vertex_count {
            vertex_block_size
        } else {
            vertex_count - vertex_offset
        };

        let base = vertex_offset * vertex_size;
        if !encode_vertex_block(
            buffer,
            &mut pos,
            out_limit,
            &vertex_data[base..],
            block_size,
            vertex_size,
            &mut last_vertex,
            &channels,
            level,
        ) {
            return 0;
        }

        vertex_offset += block_size;
    }

    let tail_size = vertex_size + vertex_size / 4;
    let tail_size_pad = tail_size.max(super::TAIL_MIN_SIZE_V1);

    if out_limit - pos < tail_size_pad {
        return 0;
    }

    if tail_size < tail_size_pad {
        buffer[pos..pos + tail_size_pad - tail_size].fill(0);
        pos += tail_size_pad - tail_size;
    }

    buffer[pos..pos + vertex_size].copy_from_slice(&first_vertex[..vertex_size]);
    pos += vertex_size;

    buffer[pos..pos + vertex_size / 4].copy_from_slice(&channels[..vertex_size / 4]);
    pos += vertex_size / 4;

    pos
}

/// Upstream: `meshopt_encodeVertexBuffer`. Encodes with the default
/// compression level (2). Emits wire version 1 (ADR-0009).
///
/// Returns the number of bytes written, or 0 if `buffer` is smaller than
/// [`encode_vertex_buffer_bound`].
pub fn encode_vertex_buffer(
    buffer: &mut [u8],
    vertices: &[u8],
    vertex_count: usize,
    vertex_size: usize,
) -> usize {
    encode_vertex_buffer_level(
        buffer,
        vertices,
        vertex_count,
        vertex_size,
        ENCODE_DEFAULT_LEVEL,
    )
}

/// Upstream: `meshopt_encodeVertexBufferBound`.
pub fn encode_vertex_buffer_bound(vertex_count: usize, vertex_size: usize) -> usize {
    debug_assert!(vertex_size > 0 && vertex_size <= 256);
    debug_assert!(vertex_size.is_multiple_of(4));

    let vertex_block_size = get_vertex_block_size(vertex_size);
    let vertex_block_count = vertex_count.div_ceil(vertex_block_size);

    let vertex_block_control_size = vertex_size / 4;
    let vertex_byte_header_size = (vertex_block_size / BYTE_GROUP_SIZE).div_ceil(4);
    let vertex_byte_data_size = vertex_block_size;

    let tail_size = vertex_size + vertex_size / 4;
    let tail_size_pad = tail_size.max(super::TAIL_MIN_SIZE_V0.max(super::TAIL_MIN_SIZE_V1));

    1 + vertex_block_count
        * (vertex_block_control_size
            + vertex_size * (vertex_byte_header_size + vertex_byte_data_size))
        + tail_size_pad
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_encodeVertexBuffer,
    // meshopt_encodeVertexBufferBound, meshopt_encodeVertexBufferLevel.
    use super::*;
    use alloc::vec;

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

        fn bytes(&mut self, n: usize) -> &'a [u8] {
            self.take(n)
        }
    }

    #[test]
    fn golden_encode_vertex_buffer() {
        let data = include_bytes!("../../tests/data/vc_enc.bin");
        let mut c = Cursor::new(data);

        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let vertex_size = c.u32() as usize;
            let level = c.u32() as i32;
            let vertices = c.bytes(vertex_count * vertex_size).to_vec();
            let size = c.u32() as usize;
            let encoded = c.bytes(size).to_vec();

            let mut buffer = vec![0u8; encode_vertex_buffer_bound(vertex_count, vertex_size)];
            let result = encode_vertex_buffer_level(
                &mut buffer,
                &vertices,
                vertex_count,
                vertex_size,
                level,
            );

            assert_eq!(
                result, size,
                "size (n={vertex_count}, s={vertex_size}, l={level})"
            );
            assert_eq!(
                &buffer[..size],
                &encoded[..],
                "bytes (n={vertex_count}, s={vertex_size}, l={level})"
            );
        }
    }

    #[test]
    fn encode_vertex_buffer_default_level_is_two() {
        // level 2 must match meshopt_encodeVertexBuffer byte for byte
        let data = include_bytes!("../../tests/data/vc_enc.bin");
        let mut c = Cursor::new(data);

        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let vertex_size = c.u32() as usize;
            let level = c.u32() as i32;
            let vertices = c.bytes(vertex_count * vertex_size).to_vec();
            let size = c.u32() as usize;
            c.bytes(size);

            if level != 2 {
                continue;
            }

            let mut buffer = vec![0u8; encode_vertex_buffer_bound(vertex_count, vertex_size)];
            let result = encode_vertex_buffer(&mut buffer, &vertices, vertex_count, vertex_size);

            assert_eq!(
                result, size,
                "default-level size (n={vertex_count}, s={vertex_size})"
            );
        }
    }

    #[test]
    fn encode_vertex_buffer_empty() {
        let bound = encode_vertex_buffer_bound(0, 12);
        let mut buffer = vec![0u8; bound];
        let size = encode_vertex_buffer(&mut buffer, &[], 0, 12);
        assert_ne!(size, 0);
        assert_eq!(buffer[0], super::super::VERTEX_HEADER | 1);
    }

    #[test]
    fn encode_vertex_buffer_too_small() {
        let mut buffer = vec![0u8; 8];
        assert_eq!(encode_vertex_buffer(&mut buffer, &[0u8; 12], 1, 12), 0);
    }

    #[test]
    fn encode_vertex_buffer_roundtrip() {
        let data = include_bytes!("../../tests/data/vc_enc.bin");
        let mut c = Cursor::new(data);

        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let vertex_size = c.u32() as usize;
            let level = c.u32() as i32;
            let vertices = c.bytes(vertex_count * vertex_size).to_vec();
            let size = c.u32() as usize;
            c.bytes(size);

            let mut buffer = vec![0u8; encode_vertex_buffer_bound(vertex_count, vertex_size)];
            let size = encode_vertex_buffer_level(
                &mut buffer,
                &vertices,
                vertex_count,
                vertex_size,
                level,
            );
            assert_ne!(size, 0);

            let mut decoded = vec![0u8; vertex_count * vertex_size];
            let ok = crate::vertexcodec::decode_vertex_buffer(
                &mut decoded,
                vertex_count,
                vertex_size,
                &buffer[..size],
            );
            assert!(
                ok,
                "roundtrip decode (n={vertex_count}, s={vertex_size}, l={level})"
            );
            assert_eq!(
                decoded, vertices,
                "roundtrip bytes (n={vertex_count}, s={vertex_size}, l={level})"
            );
        }
    }
}
