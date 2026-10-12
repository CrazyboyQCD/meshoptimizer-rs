//! Index codec decoder half.
//!
//! Upstream: `indexcodec.cpp` (`meshopt_decodeIndexBuffer`,
//! `meshopt_decodeIndexVersion`, `meshopt_decodeIndexSequence`).
//! Decoders accept wire versions 0 and 1 and never panic on malformed input.

use super::{DECODE_INDEX_VERSION, INDEX_HEADER, SEQUENCE_HEADER};

fn decode_vbyte(data: &[u8], pos: &mut usize) -> u32 {
    let lead = data[*pos];
    *pos += 1;

    // fast path: single byte
    if lead < 128 {
        return u32::from(lead);
    }

    // slow path: up to 4 extra bytes
    // note that this loop always terminates, which is important for malformed data
    let mut result = u32::from(lead & 127);
    let mut shift = 7u32;

    for _ in 0..4 {
        let group = data[*pos];
        *pos += 1;
        result |= u32::from(group & 127) << shift;
        shift += 7;

        if group < 128 {
            break;
        }
    }

    result
}

fn decode_index(data: &[u8], pos: &mut usize, last: u32) -> u32 {
    let v = decode_vbyte(data, pos);
    let d = (v >> 1) ^ (v & 1).wrapping_neg();

    last.wrapping_add(d)
}

fn push_edge_fifo(fifo: &mut [[u32; 2]; 16], offset: &mut usize, a: u32, b: u32) {
    fifo[*offset][0] = a;
    fifo[*offset][1] = b;
    *offset = (*offset + 1) & 15;
}

fn push_vertex_fifo(fifo: &mut [u32; 16], offset: &mut usize, v: u32, cond: usize) {
    fifo[*offset] = v;
    *offset = (*offset + cond) & 15;
}

fn write_triangle(dest: &mut [u8], pos: &mut usize, index_size: usize, a: u32, b: u32, c: u32) {
    if index_size == 2 {
        let tri = &mut dest[*pos..*pos + 6];
        tri[0..2].copy_from_slice(&(a as u16).to_le_bytes());
        tri[2..4].copy_from_slice(&(b as u16).to_le_bytes());
        tri[4..6].copy_from_slice(&(c as u16).to_le_bytes());
    } else {
        let tri = &mut dest[*pos..*pos + 12];
        tri[0..4].copy_from_slice(&a.to_le_bytes());
        tri[4..8].copy_from_slice(&b.to_le_bytes());
        tri[8..12].copy_from_slice(&c.to_le_bytes());
    }

    *pos += 3 * index_size;
}

/// Upstream: `meshopt_decodeIndexVersion`. Returns the stream version
/// (`0..=1`), or -1 if the buffer does not carry a valid index codec header.
pub fn decode_index_version(buffer: &[u8]) -> i32 {
    if buffer.is_empty() {
        return -1;
    }

    let header = buffer[0];

    if header & 0xf0 != INDEX_HEADER && header & 0xf0 != SEQUENCE_HEADER {
        return -1;
    }

    let version = i32::from(header & 0x0f);
    if version > DECODE_INDEX_VERSION {
        return -1;
    }

    version
}

/// Upstream: `meshopt_decodeIndexBuffer`.
///
/// `destination` holds `index_count` indices of `index_size` bytes (2 or 4).
/// Returns false on malformed input; the destination contents are unspecified.
pub fn decode_index_buffer(
    destination: &mut [u8],
    index_count: usize,
    index_size: usize,
    buffer: &[u8],
) -> bool {
    debug_assert!(index_count.is_multiple_of(3));
    debug_assert!(index_size == 2 || index_size == 4);

    // the minimum valid encoding is header, 1 byte per triangle and a 16-byte codeaux table
    if buffer.len() < 1 + index_count / 3 + 16 {
        return false;
    }

    if buffer[0] & 0xf0 != INDEX_HEADER {
        return false;
    }

    let version = i32::from(buffer[0] & 0x0f);
    if version > DECODE_INDEX_VERSION {
        return false;
    }

    let mut edgefifo = [[u32::MAX; 2]; 16];
    let mut vertexfifo = [u32::MAX; 16];

    let mut edgefifooffset = 0usize;
    let mut vertexfifooffset = 0usize;

    let mut next = 0u32;
    let mut last = 0u32;

    // v1 narrows the vertex fifo encoding window to make room for last-1/last+1 codes
    let fecmax = if version >= 1 { 13usize } else { 15usize };

    // since we store 16-byte codeaux table at the end, triangle data has to begin before data_safe_end
    let mut code = 1usize;
    let code_end = code + index_count / 3;
    let mut data = code_end;

    // each triangle reads at most 16 bytes of data: 1b for codeaux and 5b for each free index
    let data_safe_end = buffer.len() - 16;
    let codeaux_table = data_safe_end;

    let mut out = 0usize;

    while code < code_end {
        let codetri = buffer[code];
        code += 1;

        if codetri < 0xf0 {
            let fe = (codetri >> 4) as usize;

            // fifo reads are wrapped around 16 entry buffer
            let edge = edgefifo[(edgefifooffset.wrapping_sub(1 + fe)) & 15];
            let a = edge[0];
            let b = edge[1];
            let c;

            let fec = (codetri & 15) as usize;

            // note: this is the most common path in the entire decoder
            if fec < fecmax {
                let cf = vertexfifo[(vertexfifooffset.wrapping_sub(1 + fec)) & 15];

                c = if fec == 0 { next } else { cf };

                let fec0 = u32::from(fec == 0);
                next += fec0;

                // push vertex fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
                push_vertex_fifo(&mut vertexfifo, &mut vertexfifooffset, c, fec0 as usize);
            } else {
                // make sure we have enough data to read for a triangle; this check covers worst case advance
                if data > data_safe_end {
                    return false;
                }

                // fec * 2 - 27 decodes 13, 14 into -1, 1
                // note that we need to update the last index since free indices are delta-encoded
                c = if fec != 15 {
                    last = last.wrapping_add((fec as i32 * 2 - 27) as u32);
                    last
                } else {
                    last = decode_index(buffer, &mut data, last);
                    last
                };

                // push vertex/edge fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
                push_vertex_fifo(&mut vertexfifo, &mut vertexfifooffset, c, 1);
            }

            // push edge fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
            push_edge_fifo(&mut edgefifo, &mut edgefifooffset, c, b);
            push_edge_fifo(&mut edgefifo, &mut edgefifooffset, a, c);

            // output triangle
            write_triangle(destination, &mut out, index_size, a, b, c);
        } else if codetri < 0xfe {
            // fast path: read codeaux from the table
            let codeaux = buffer[codeaux_table + (codetri & 15) as usize];

            // note: table can't contain feb/fec=15
            let feb = (codeaux >> 4) as usize;
            let fec = (codeaux & 15) as usize;

            // fifo reads are wrapped around 16 entry buffer
            // also note that we increment next for all three vertices before decoding indices - this matches encoder behavior
            let a = next;
            next += 1;

            let bf = vertexfifo[(vertexfifooffset.wrapping_sub(feb)) & 15];
            let b = if feb == 0 { next } else { bf };

            let feb0 = u32::from(feb == 0);
            next += feb0;

            let cf = vertexfifo[(vertexfifooffset.wrapping_sub(fec)) & 15];
            let c = if fec == 0 { next } else { cf };

            let fec0 = u32::from(fec == 0);
            next += fec0;

            // output triangle
            write_triangle(destination, &mut out, index_size, a, b, c);

            // push vertex/edge fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
            push_vertex_fifo(&mut vertexfifo, &mut vertexfifooffset, a, 1);
            push_vertex_fifo(&mut vertexfifo, &mut vertexfifooffset, b, feb0 as usize);
            push_vertex_fifo(&mut vertexfifo, &mut vertexfifooffset, c, fec0 as usize);

            push_edge_fifo(&mut edgefifo, &mut edgefifooffset, b, a);
            push_edge_fifo(&mut edgefifo, &mut edgefifooffset, c, b);
            push_edge_fifo(&mut edgefifo, &mut edgefifooffset, a, c);
        } else {
            // make sure we have enough data to read for a triangle; this check covers worst case advance
            if data > data_safe_end {
                return false;
            }

            // slow path: read a full byte for codeaux instead of using a table lookup
            let codeaux = buffer[data];
            data += 1;

            let fea = if codetri == 0xfe { 0usize } else { 15usize };
            let feb = (codeaux >> 4) as usize;
            let fec = (codeaux & 15) as usize;

            // reset: codeaux is 0 but encoded as not-a-table
            if codeaux == 0 {
                next = 0;
            }

            // fifo reads are wrapped around 16 entry buffer
            // also note that we increment next for all three vertices before decoding indices - this matches encoder behavior
            let mut a = if fea == 0 {
                let v = next;
                next += 1;
                v
            } else {
                0
            };
            let mut b = if feb == 0 {
                let v = next;
                next += 1;
                v
            } else {
                vertexfifo[(vertexfifooffset.wrapping_sub(feb)) & 15]
            };
            let mut c = if fec == 0 {
                let v = next;
                next += 1;
                v
            } else {
                vertexfifo[(vertexfifooffset.wrapping_sub(fec)) & 15]
            };

            // note that we need to update the last index since free indices are delta-encoded
            if fea == 15 {
                a = decode_index(buffer, &mut data, last);
                last = a;
            }

            if feb == 15 {
                b = decode_index(buffer, &mut data, last);
                last = b;
            }

            if fec == 15 {
                c = decode_index(buffer, &mut data, last);
                last = c;
            }

            // output triangle
            write_triangle(destination, &mut out, index_size, a, b, c);

            // push vertex/edge fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
            push_vertex_fifo(&mut vertexfifo, &mut vertexfifooffset, a, 1);
            push_vertex_fifo(
                &mut vertexfifo,
                &mut vertexfifooffset,
                b,
                usize::from(feb == 0 || feb == 15),
            );
            push_vertex_fifo(
                &mut vertexfifo,
                &mut vertexfifooffset,
                c,
                usize::from(fec == 0 || fec == 15),
            );

            push_edge_fifo(&mut edgefifo, &mut edgefifooffset, b, a);
            push_edge_fifo(&mut edgefifo, &mut edgefifooffset, c, b);
            push_edge_fifo(&mut edgefifo, &mut edgefifooffset, a, c);
        }
    }

    // we should've read all data bytes and stopped at the boundary between data and codeaux table
    if data != data_safe_end {
        return false;
    }

    true
}

/// Upstream: `meshopt_decodeIndexSequence`.
///
/// `destination` holds `index_count` indices of `index_size` bytes (2 or 4).
/// Returns false on malformed input; the destination contents are unspecified.
pub fn decode_index_sequence(
    destination: &mut [u8],
    index_count: usize,
    index_size: usize,
    buffer: &[u8],
) -> bool {
    // the minimum valid encoding is header, 1 byte per index and a 4-byte tail
    if buffer.len() < 1 + index_count + 4 {
        return false;
    }

    if buffer[0] & 0xf0 != SEQUENCE_HEADER {
        return false;
    }

    let version = i32::from(buffer[0] & 0x0f);
    if version > DECODE_INDEX_VERSION {
        return false;
    }

    let mut data = 1usize;
    let data_safe_end = buffer.len() - 4;

    let mut last = [0u32; 2];

    for i in 0..index_count {
        // make sure we have enough data to read
        // each index reads at most 5 bytes of data; there's a 4 byte tail after data_safe_end
        if data >= data_safe_end {
            return false;
        }

        let mut v = decode_vbyte(buffer, &mut data);

        // decode the index of the last baseline
        let current = (v & 1) as usize;
        v >>= 1;

        // reconstruct index as a delta
        let d = (v >> 1) ^ (v & 1).wrapping_neg();
        let index = last[current].wrapping_add(d);

        // update last for the next iteration that uses it
        last[current] = index;

        if index_size == 2 {
            destination[i * 2..i * 2 + 2].copy_from_slice(&(index as u16).to_le_bytes());
        } else {
            destination[i * 4..i * 4 + 4].copy_from_slice(&index.to_le_bytes());
        }
    }

    // we should've read all data bytes and stopped at the boundary between data and tail
    if data != data_safe_end {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_decodeIndexBuffer,
    // meshopt_decodeIndexVersion, meshopt_decodeIndexSequence.
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
    fn golden_decode_index_buffer() {
        let data = include_bytes!("../../tests/data/ic_dec_ib.bin");
        let mut c = Cursor::new(data);

        while c.pos < data.len() {
            let index_count = c.u32() as usize;
            let index_size = c.u32() as usize;
            let buffer_size = c.u32() as usize;
            let expected_version = c.i32();
            let stream = c.bytes(buffer_size).to_vec();
            let result = c.i32();
            let expected = if result == 0 {
                c.bytes(index_count * index_size).to_vec()
            } else {
                Vec::new()
            };

            assert_eq!(
                decode_index_version(&stream),
                expected_version,
                "decode_index_version"
            );

            let mut destination = vec![0u8; index_count * index_size];
            let ok = decode_index_buffer(&mut destination, index_count, index_size, &stream);
            assert_eq!(
                ok,
                result == 0,
                "decode_index_buffer result (n={index_count})"
            );
            if ok {
                assert_eq!(
                    destination, expected,
                    "decode_index_buffer bytes (n={index_count})"
                );
            }
        }
    }

    #[test]
    fn golden_decode_index_sequence() {
        let data = include_bytes!("../../tests/data/ic_dec_seq.bin");
        let mut c = Cursor::new(data);

        while c.pos < data.len() {
            let index_count = c.u32() as usize;
            let index_size = c.u32() as usize;
            let buffer_size = c.u32() as usize;
            let expected_version = c.i32();
            let stream = c.bytes(buffer_size).to_vec();
            let result = c.i32();
            let expected = if result == 0 {
                c.bytes(index_count * index_size).to_vec()
            } else {
                Vec::new()
            };

            assert_eq!(
                decode_index_version(&stream),
                expected_version,
                "decode_index_version"
            );

            let mut destination = vec![0u8; index_count * index_size];
            let ok = decode_index_sequence(&mut destination, index_count, index_size, &stream);
            assert_eq!(
                ok,
                result == 0,
                "decode_index_sequence result (n={index_count})"
            );
            if ok {
                assert_eq!(
                    destination, expected,
                    "decode_index_sequence bytes (n={index_count})"
                );
            }
        }
    }

    #[test]
    fn hostile_decode_index_buffer() {
        let data = include_bytes!("../../tests/data/ic_hostile_ib.bin");
        let mut c = Cursor::new(data);

        while c.pos < data.len() {
            let buffer_size = c.u32() as usize;
            let index_count = c.u32() as usize;
            let index_size = c.u32() as usize;
            let stream = c.bytes(buffer_size).to_vec();
            let result = c.i32();
            let expected = if result == 0 {
                c.bytes(index_count * index_size).to_vec()
            } else {
                Vec::new()
            };

            let mut destination = vec![0u8; index_count * index_size];
            let ok = decode_index_buffer(&mut destination, index_count, index_size, &stream);
            assert_eq!(ok, result == 0, "hostile result (size={buffer_size})");
            if ok {
                assert_eq!(destination, expected, "hostile bytes (size={buffer_size})");
            }
        }
    }

    #[test]
    fn hostile_decode_index_sequence() {
        let data = include_bytes!("../../tests/data/ic_hostile_seq.bin");
        let mut c = Cursor::new(data);

        while c.pos < data.len() {
            let buffer_size = c.u32() as usize;
            let index_count = c.u32() as usize;
            let index_size = c.u32() as usize;
            let stream = c.bytes(buffer_size).to_vec();
            let result = c.i32();
            let expected = if result == 0 {
                c.bytes(index_count * index_size).to_vec()
            } else {
                Vec::new()
            };

            let mut destination = vec![0u8; index_count * index_size];
            let ok = decode_index_sequence(&mut destination, index_count, index_size, &stream);
            assert_eq!(ok, result == 0, "hostile result (size={buffer_size})");
            if ok {
                assert_eq!(destination, expected, "hostile bytes (size={buffer_size})");
            }
        }
    }

    #[test]
    fn decode_index_version_probe() {
        assert_eq!(decode_index_version(&[]), -1);
        assert_eq!(decode_index_version(&[0x00]), -1);
        assert_eq!(decode_index_version(&[0xe2]), -1);
        assert_eq!(decode_index_version(&[0xe1]), 1);
        assert_eq!(decode_index_version(&[0xe0]), 0);
        assert_eq!(decode_index_version(&[0xd1]), 1);
        assert_eq!(decode_index_version(&[0xd0]), 0);
        assert_eq!(decode_index_version(&[0xf1]), -1);
    }

    #[test]
    fn decode_index_buffer_minimal_rejects() {
        assert!(!decode_index_buffer(&mut [], 0, 4, &[]));
        assert!(!decode_index_buffer(&mut [], 0, 4, &[0xe1]));
    }

    #[test]
    fn decode_index_sequence_minimal_rejects() {
        assert!(!decode_index_sequence(&mut [], 0, 4, &[]));
        assert!(!decode_index_sequence(&mut [], 0, 4, &[0xd1]));
    }
}
