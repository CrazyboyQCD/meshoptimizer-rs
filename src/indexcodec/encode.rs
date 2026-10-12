//! Index codec encoder half.
//!
//! Upstream: `indexcodec.cpp` (`meshopt_encodeIndexBuffer`,
//! `meshopt_encodeIndexBufferBound`, `meshopt_encodeIndexSequence`,
//! `meshopt_encodeIndexSequenceBound`). Emits wire version 1 only
//! ([ADR-0009](../../docs/adr/0009-encoder-version-hardcoded.md)).

// generated from symbol frequencies on a training mesh set; the decoder
// mirrors this table, so the wire bytes depend on it verbatim
const CODE_AUX_ENCODING_TABLE: [u8; 16] = [
    0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9, 0x86, 0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0, 0,
];

// C size_t arithmetic wraps before the & 15 cycle; wrapping_sub mirrors that
fn get_edge_fifo(fifo: &[[u32; 2]; 16], a: u32, b: u32, c: u32, offset: usize) -> i32 {
    for i in 0..16usize {
        let index = offset.wrapping_sub(1 + i) & 15;

        let e0 = fifo[index][0];
        let e1 = fifo[index][1];

        if e0 == a && e1 == b {
            return (i as i32) << 2;
        }
        if e0 == b && e1 == c {
            return ((i as i32) << 2) | 1;
        }
        if e0 == c && e1 == a {
            return ((i as i32) << 2) | 2;
        }
    }

    -1
}

fn push_edge_fifo(fifo: &mut [[u32; 2]; 16], offset: &mut usize, a: u32, b: u32) {
    fifo[*offset][0] = a;
    fifo[*offset][1] = b;
    *offset = (*offset + 1) & 15;
}

fn get_vertex_fifo(fifo: &[u32; 16], v: u32, offset: usize) -> i32 {
    for i in 0..16usize {
        let index = offset.wrapping_sub(1 + i) & 15;

        if fifo[index] == v {
            return i as i32;
        }
    }

    -1
}

fn push_vertex_fifo(fifo: &mut [u32; 16], offset: &mut usize, v: u32, cond: usize) {
    fifo[*offset] = v;
    *offset = (*offset + cond) & 15;
}

fn encode_vbyte(data: &mut [u8], pos: &mut usize, v: u32) {
    // encode 32-bit value in up to 5 7-bit groups
    let mut v = v;
    loop {
        data[*pos] = ((v & 127) | if v > 127 { 128 } else { 0 }) as u8;
        *pos += 1;
        v >>= 7;
        if v == 0 {
            break;
        }
    }
}

fn encode_index(data: &mut [u8], pos: &mut usize, index: u32, last: u32) {
    let d = index.wrapping_sub(last);
    let v = (d << 1) ^ ((d as i32 >> 31) as u32);

    encode_vbyte(data, pos, v);
}

fn rotate_triangle(a: u32, b: u32, c: u32, next: u32) -> usize {
    let _ = a;

    if b == next {
        1
    } else if c == next {
        2
    } else {
        0
    }
}

fn get_code_aux_index(v: u8, table: &[u8; 16]) -> i32 {
    table.iter().position(|&t| t == v).map_or(-1, |i| i as i32)
}

/// Upstream: `meshopt_encodeIndexBuffer`.
///
/// Returns the number of bytes written, or 0 if `buffer` is smaller than
/// [`encode_index_buffer_bound`]. Emits wire version 1 (ADR-0009).
pub fn encode_index_buffer(buffer: &mut [u8], indices: &[u32]) -> usize {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));

    // the minimum valid encoding is header, 1 byte per triangle and a 16-byte codeaux table
    if buffer.len() < 1 + index_count / 3 + 16 {
        return 0;
    }

    buffer[0] = super::INDEX_HEADER | 1;

    let mut edgefifo = [[u32::MAX; 2]; 16];
    let mut vertexfifo = [u32::MAX; 16];

    let mut edgefifooffset = 0usize;
    let mut vertexfifooffset = 0usize;

    let mut next = 0u32;
    let mut last = 0u32;

    let mut code = 1usize;
    let mut data = 1 + index_count / 3;
    let data_safe_end = buffer.len() - 16;

    // v1 narrows the vertex fifo encoding window to make room for last-1/last+1 codes
    let fecmax = 13i32;

    const ROTATIONS: [usize; 5] = [0, 1, 2, 0, 1];
    let codeaux_table = &CODE_AUX_ENCODING_TABLE;

    let mut i = 0usize;
    while i < index_count {
        // make sure we have enough space to write a triangle
        // each triangle writes at most 16 bytes: 1b for codeaux and 5b for each free index
        if data > data_safe_end {
            return 0;
        }

        let fer = get_edge_fifo(
            &edgefifo,
            indices[i],
            indices[i + 1],
            indices[i + 2],
            edgefifooffset,
        );

        if fer >= 0 && (fer >> 2) < 15 {
            // note: getEdgeFifo implicitly rotates triangles by matching a/b to existing edge
            let order = &ROTATIONS[(fer & 3) as usize..];

            let a = indices[i + order[0]];
            let b = indices[i + order[1]];
            let c = indices[i + order[2]];

            // encode edge index and vertex fifo index, next or free index
            let fe = fer >> 2;
            let fc = get_vertex_fifo(&vertexfifo, c, vertexfifooffset);

            let mut fec = if (1..fecmax).contains(&fc) {
                fc
            } else if c == next {
                next += 1;
                0
            } else {
                15
            };

            if fec == 15 {
                // encode last-1 and last+1 to optimize strip-like sequences
                if c.wrapping_add(1) == last {
                    fec = 13;
                    last = c;
                }
                if c == last.wrapping_add(1) {
                    fec = 14;
                    last = c;
                }
            }

            buffer[code] = ((fe << 4) | fec) as u8;
            code += 1;

            // note that we need to update the last index since free indices are delta-encoded
            if fec == 15 {
                encode_index(buffer, &mut data, c, last);
                last = c;
            }

            // we only need to push third vertex since first two are likely already in the vertex fifo
            if fec == 0 || fec >= fecmax {
                push_vertex_fifo(&mut vertexfifo, &mut vertexfifooffset, c, 1);
            }

            // we only need to push two new edges to edge fifo since the third one is already there
            push_edge_fifo(&mut edgefifo, &mut edgefifooffset, c, b);
            push_edge_fifo(&mut edgefifo, &mut edgefifooffset, a, c);
        } else {
            let rotation = rotate_triangle(indices[i], indices[i + 1], indices[i + 2], next);
            let order = &ROTATIONS[rotation..];

            let a = indices[i + order[0]];
            let b = indices[i + order[1]];
            let c = indices[i + order[2]];

            // if a/b/c are 0/1/2, we emit a reset code
            let mut reset = false;

            if a == 0 && b == 1 && c == 2 && next > 0 {
                reset = true;
                next = 0;

                // reset vertex fifo to make sure we don't accidentally reference vertices from that in the future
                // this makes sure next continues to get incremented instead of being stuck
                vertexfifo = [u32::MAX; 16];
            }

            let fb = get_vertex_fifo(&vertexfifo, b, vertexfifooffset);
            let fc = get_vertex_fifo(&vertexfifo, c, vertexfifooffset);

            // after rotation, a is almost always equal to next, so we don't waste bits on FIFO encoding for a
            // note: decoder implicitly assumes that if feb=fec=0, then fea=0 (reset code); this is enforced by rotation
            let fea = if a == next {
                next += 1;
                0
            } else {
                15
            };
            let feb = if (0..14).contains(&fb) {
                fb + 1
            } else if b == next {
                next += 1;
                0
            } else {
                15
            };
            let fec = if (0..14).contains(&fc) {
                fc + 1
            } else if c == next {
                next += 1;
                0
            } else {
                15
            };

            // we encode feb & fec in 4 bits using a table if possible, and as a full byte otherwise
            let codeaux = ((feb << 4) | fec) as u8;
            let codeauxindex = get_code_aux_index(codeaux, codeaux_table);

            // <14 encodes an index into codeaux table, 14 encodes fea=0, 15 encodes fea=15
            if fea == 0 && (0..14).contains(&codeauxindex) && !reset {
                buffer[code] = ((15 << 4) | codeauxindex) as u8;
                code += 1;
            } else {
                buffer[code] = ((15 << 4) | 14 | fea) as u8;
                code += 1;
                buffer[data] = codeaux;
                data += 1;
            }

            // note that we need to update the last index since free indices are delta-encoded
            if fea == 15 {
                encode_index(buffer, &mut data, a, last);
                last = a;
            }

            if feb == 15 {
                encode_index(buffer, &mut data, b, last);
                last = b;
            }

            if fec == 15 {
                encode_index(buffer, &mut data, c, last);
                last = c;
            }

            // only push vertices that weren't already in fifo
            if fea == 0 || fea == 15 {
                push_vertex_fifo(&mut vertexfifo, &mut vertexfifooffset, a, 1);
            }

            if feb == 0 || feb == 15 {
                push_vertex_fifo(&mut vertexfifo, &mut vertexfifooffset, b, 1);
            }

            if fec == 0 || fec == 15 {
                push_vertex_fifo(&mut vertexfifo, &mut vertexfifooffset, c, 1);
            }

            // all three edges aren't in the fifo; pushing all of them is important so that we can match them for later triangles
            push_edge_fifo(&mut edgefifo, &mut edgefifooffset, b, a);
            push_edge_fifo(&mut edgefifo, &mut edgefifooffset, c, b);
            push_edge_fifo(&mut edgefifo, &mut edgefifooffset, a, c);
        }

        i += 3;
    }

    // make sure we have enough space to write codeaux table
    if data > data_safe_end {
        return 0;
    }

    // add codeaux encoding table to the end of the stream; this is used for decoding codeaux *and* as padding
    // we need padding for decoding to be able to assume that each triangle is encoded as <= 16 bytes of extra data
    buffer[data..data + 16].copy_from_slice(codeaux_table);
    data += 16;

    data
}

/// Upstream: `meshopt_encodeIndexBufferBound`.
pub const fn encode_index_buffer_bound(index_count: usize, vertex_count: usize) -> usize {
    // compute number of bits required for each index
    let mut vertex_bits = 1usize;

    while vertex_bits < 32 && vertex_count > (1usize << vertex_bits) {
        vertex_bits += 1;
    }

    // worst-case encoding is 2 header bytes + 3 varint-7 encoded index deltas
    let vertex_groups = (vertex_bits + 1).div_ceil(7);

    1 + (index_count / 3) * (2 + 3 * vertex_groups) + 16
}

/// Upstream: `meshopt_encodeIndexSequence`.
///
/// Returns the number of bytes written, or 0 if `buffer` is smaller than
/// [`encode_index_sequence_bound`]. Emits wire version 1 (ADR-0009).
pub fn encode_index_sequence(buffer: &mut [u8], indices: &[u32]) -> usize {
    // the minimum valid encoding is header, 1 byte per index and a 4-byte tail
    if buffer.len() < 1 + indices.len() + 4 {
        return 0;
    }

    buffer[0] = super::SEQUENCE_HEADER | 1;

    let mut last = [0u32; 2];
    let mut current = 0usize;

    let mut data = 1usize;
    let data_safe_end = buffer.len() - 4;

    for &index in indices {
        // make sure we have enough data to write
        // each index writes at most 5 bytes of data; there's a 4 byte tail after data_safe_end
        if data >= data_safe_end {
            return 0;
        }

        // this is a heuristic that switches between baselines when the delta grows too large
        // we want the encoded delta to fit into one byte (7 bits), but 2 bits are used for sign and baseline index
        let cd = index.wrapping_sub(last[current]) as i32;
        current ^= usize::from(cd.wrapping_abs() >= 30);

        // encode delta from the last index
        let d = index.wrapping_sub(last[current]);
        let v = (d << 1) ^ ((d as i32 >> 31) as u32);

        // note: low bit encodes the index of the last baseline which will be used for reconstruction
        encode_vbyte(buffer, &mut data, (v << 1) | current as u32);

        // update last for the next iteration that uses it
        last[current] = index;
    }

    // make sure we have enough space to write tail
    if data > data_safe_end {
        return 0;
    }

    buffer[data..data + 4].copy_from_slice(&[0; 4]);
    data += 4;

    data
}

/// Upstream: `meshopt_encodeIndexSequenceBound`.
pub const fn encode_index_sequence_bound(index_count: usize, vertex_count: usize) -> usize {
    // compute number of bits required for each index
    let mut vertex_bits = 1usize;

    while vertex_bits < 32 && vertex_count > (1usize << vertex_bits) {
        vertex_bits += 1;
    }

    // worst-case encoding is 1 varint-7 encoded index delta for a K bit value and an extra bit
    let vertex_groups = (vertex_bits + 2).div_ceil(7);

    1 + index_count * vertex_groups + 4
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_encodeIndexBuffer,
    // meshopt_encodeIndexBufferBound, meshopt_encodeIndexSequence,
    // meshopt_encodeIndexSequenceBound.
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

        fn bytes(&mut self, n: usize) -> &'a [u8] {
            self.take(n)
        }
    }

    fn read_indices(c: &mut Cursor, count: u32) -> Vec<u32> {
        (0..count).map(|_| c.u32()).collect()
    }

    #[test]
    fn golden_encode_index_buffer() {
        let data = include_bytes!("../../tests/data/ic_enc_ib.bin");
        let mut c = Cursor::new(data);

        while c.pos < data.len() {
            let index_count = c.u32();
            let vertex_count = c.u32();
            let indices = read_indices(&mut c, index_count);
            let size = c.u32() as usize;
            let encoded = c.bytes(size).to_vec();

            let mut buffer =
                vec![0u8; encode_index_buffer_bound(index_count as usize, vertex_count as usize)];
            let result = encode_index_buffer(&mut buffer, &indices);

            assert_eq!(result, size, "encode_index_buffer size (n={index_count})");
            assert_eq!(
                &buffer[..size],
                &encoded[..],
                "encode_index_buffer bytes (n={index_count})"
            );
        }
    }

    #[test]
    fn golden_encode_index_sequence() {
        let data = include_bytes!("../../tests/data/ic_enc_seq.bin");
        let mut c = Cursor::new(data);

        while c.pos < data.len() {
            let index_count = c.u32();
            let vertex_count = c.u32();
            let indices = read_indices(&mut c, index_count);
            let size = c.u32() as usize;
            let encoded = c.bytes(size).to_vec();

            let mut buffer =
                vec![0u8; encode_index_sequence_bound(index_count as usize, vertex_count as usize)];
            let result = encode_index_sequence(&mut buffer, &indices);

            assert_eq!(result, size, "encode_index_sequence size (n={index_count})");
            assert_eq!(
                &buffer[..size],
                &encoded[..],
                "encode_index_sequence bytes (n={index_count})"
            );
        }
    }

    #[test]
    fn encode_index_buffer_roundtrip() {
        // The codec preserves triangle order and winding but may cyclically
        // rotate triangles that hit the next-vertex path (verified against
        // upstream at the pin); compare triangles up to cyclic rotation.
        let data = include_bytes!("../../tests/data/ic_enc_ib.bin");
        let mut c = Cursor::new(data);

        while c.pos < data.len() {
            let index_count = c.u32();
            let vertex_count = c.u32();
            let indices = read_indices(&mut c, index_count);
            let size = c.u32() as usize;
            c.bytes(size);

            let mut buffer =
                vec![0u8; encode_index_buffer_bound(index_count as usize, vertex_count as usize)];
            let size = encode_index_buffer(&mut buffer, &indices);
            assert_ne!(size, 0);

            let mut decoded = vec![0u8; indices.len() * 4];
            let ok = crate::indexcodec::decode_index_buffer(
                &mut decoded,
                indices.len(),
                4,
                &buffer[..size],
            );
            assert!(ok, "roundtrip decode (n={index_count})");

            let decoded: Vec<u32> = decoded
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect();

            for t in 0..decoded.len() / 3 {
                let a = &decoded[t * 3..t * 3 + 3];
                let b = &indices[t * 3..t * 3 + 3];
                let rotated = a == b || a == [b[1], b[2], b[0]] || a == [b[2], b[0], b[1]];
                assert!(rotated, "roundtrip triangle {t}: {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn encode_index_sequence_roundtrip() {
        let data = include_bytes!("../../tests/data/ic_enc_seq.bin");
        let mut c = Cursor::new(data);

        while c.pos < data.len() {
            let index_count = c.u32();
            let vertex_count = c.u32();
            let indices = read_indices(&mut c, index_count);
            let size = c.u32() as usize;
            c.bytes(size);

            let mut buffer =
                vec![0u8; encode_index_sequence_bound(index_count as usize, vertex_count as usize)];
            let size = encode_index_sequence(&mut buffer, &indices);
            assert_ne!(size, 0);

            let mut decoded = vec![0u8; indices.len() * 4];
            let ok = crate::indexcodec::decode_index_sequence(
                &mut decoded,
                indices.len(),
                4,
                &buffer[..size],
            );
            assert!(ok, "roundtrip decode (n={index_count})");

            let decoded: Vec<u32> = decoded
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect();
            assert_eq!(decoded, indices, "roundtrip indices (n={index_count})");
        }
    }

    #[test]
    fn encode_index_buffer_empty() {
        let bound = encode_index_buffer_bound(0, 1);
        let mut buffer = vec![0u8; bound];
        let size = encode_index_buffer(&mut buffer, &[]);
        assert_eq!(size, 17);
        assert_eq!(buffer[0], super::super::INDEX_HEADER | 1);
    }

    #[test]
    fn encode_index_buffer_too_small() {
        let mut buffer = vec![0u8; 16];
        assert_eq!(encode_index_buffer(&mut buffer, &[]), 0);
    }

    #[test]
    fn encode_index_sequence_empty() {
        let bound = encode_index_sequence_bound(0, 1);
        let mut buffer = vec![0u8; bound];
        let size = encode_index_sequence(&mut buffer, &[]);
        assert_eq!(size, 5);
        assert_eq!(buffer[0], super::super::SEQUENCE_HEADER | 1);
    }

    #[test]
    fn encode_index_sequence_too_small() {
        let mut buffer = vec![0u8; 4];
        assert_eq!(encode_index_sequence(&mut buffer, &[]), 0);
    }
}
