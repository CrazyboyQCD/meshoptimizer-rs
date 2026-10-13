//! Meshlet codec (`meshletcodec.cpp`, FR-06).
//!
//! Encodes a single optimized meshlet (up to 256 vertices / 256 triangles) into a
//! byte stream: 4-bit triangle codes with an 8-edge FIFO, extra vertex bytes, and
//! grouped-varint zigzag vertex deltas. The scalar profile is normative (ADR-0010);
//! upstream's SIMD decode paths are not ported. Encoded output is byte-identical
//! to upstream; decoding requires exact stream consumption.

pub fn encode_meshlet_bound(max_vertices: usize, max_triangles: usize) -> usize {
    let codes_size = max_triangles.div_ceil(2);
    let extra_size = max_triangles * 3;

    let ctrl_size = max_vertices.div_ceil(4);
    let data_size = max_vertices.div_ceil(4) * 16; // worst case: 16 bytes per vertex group

    let gap_size = 16usize.saturating_sub(codes_size + ctrl_size);

    codes_size + extra_size + ctrl_size + data_size + gap_size
}

fn rotate_triangle(a: u32, b: u32, c: u32) -> usize {
    if a > b && a > c {
        1
    } else if b > c {
        2
    } else {
        0
    }
}

fn get_edge_fifo8(fifo: &[[u32; 2]; 8], a: u32, b: u32, c: u32, offset: usize) -> i32 {
    for i in 0..8usize {
        let index = offset.wrapping_sub(1).wrapping_sub(i) & 7;

        let e0 = fifo[index][0];
        let e1 = fifo[index][1];

        if e0 == a && e1 == b {
            return (i << 2) as i32;
        }
        if e0 == b && e1 == c {
            return ((i << 2) | 1) as i32;
        }
        if e0 == c && e1 == a {
            return ((i << 2) | 2) as i32;
        }
    }

    -1
}

fn push_edge_fifo8(fifo: &mut [[u32; 2]; 8], a: u32, b: u32, offset: &mut usize) {
    fifo[*offset][0] = a;
    fifo[*offset][1] = b;
    *offset = (*offset + 1) & 7;
}

fn encode_triangles(
    codes: &mut [u8],
    extra: &mut [u8],
    triangles: &[u8],
    triangle_count: usize,
) -> usize {
    let mut edgefifo = [[u32::MAX; 2]; 8];

    let mut edgefifooffset = 0usize;

    let mut next: u32 = 0;

    const ROTATIONS: [usize; 5] = [0, 1, 2, 0, 1];

    let mut extra_pos = 0usize;

    for (i, tri) in triangles.as_chunks::<3>().0[..triangle_count]
        .iter()
        .enumerate()
    {
        let t0 = tri[0] as u32;
        let t1 = tri[1] as u32;
        let t2 = tri[2] as u32;

        let fer = get_edge_fifo8(&edgefifo, t0, t1, t2, edgefifooffset);

        if fer >= 0 && (fer >> 2) < 6 {
            // getEdgeFifo8 implicitly rotates triangles by matching a/b to existing edge
            let order = &ROTATIONS[(fer & 3) as usize..];

            let a = tri[order[0]] as u32;
            let b = tri[order[1]] as u32;
            let c = tri[order[2]] as u32;

            let fec = if c == next {
                next += 1;
                0
            } else {
                1
            };

            let code = (fer >> 2) as u32 * 2 + fec;

            codes[i / 2] |= (code << ((i & 1) * 4)) as u8;

            if fec == 1 {
                extra[extra_pos] = c as u8;
                extra_pos += 1;
            }

            push_edge_fifo8(&mut edgefifo, c, b, &mut edgefifooffset);
            push_edge_fifo8(&mut edgefifo, a, c, &mut edgefifooffset);
        } else {
            // rotate triangles to minimize the need for extra vertices
            let rotation = rotate_triangle(t0, t1, t2);
            let order = &ROTATIONS[rotation..];

            let a = triangles[i * 3 + order[0]] as u32;
            let b = triangles[i * 3 + order[1]] as u32;
            let c = triangles[i * 3 + order[2]] as u32;

            // fe must be continuous: once a vertex is encoded with next, further vertices must also be encoded with next
            let fea = if a == next && b == next + 1 && c == next + 2 {
                next += 1;
                0
            } else {
                1
            };
            let feb = if b == next && c == next + 1 {
                next += 1;
                0
            } else {
                1
            };
            let fec = if c == next {
                next += 1;
                0
            } else {
                1
            };

            let code = 12 + fea + feb + fec;

            codes[i / 2] |= (code << ((i & 1) * 4)) as u8;

            if fea == 1 {
                extra[extra_pos] = a as u8;
                extra_pos += 1;
            }
            if feb == 1 {
                extra[extra_pos] = b as u8;
                extra_pos += 1;
            }
            if fec == 1 {
                extra[extra_pos] = c as u8;
                extra_pos += 1;
            }

            push_edge_fifo8(&mut edgefifo, c, b, &mut edgefifooffset);
            push_edge_fifo8(&mut edgefifo, a, c, &mut edgefifooffset);
        }
    }

    extra_pos
}

fn encode_vertices(
    ctrl: &mut [u8],
    data: &mut [u8],
    vertices: &[u32],
    vertex_count: usize,
) -> usize {
    let mut data_pos = 0usize;

    let mut last: u32 = u32::MAX;

    let mut i = 0;
    while i < vertex_count {
        let mut gv = [0u32; 4];

        for (k, g) in gv.iter_mut().enumerate() {
            if i + k >= vertex_count {
                break;
            }
            let d = vertices[i + k].wrapping_sub(last).wrapping_sub(1);
            let v = (d << 1) ^ ((d as i32 >> 31) as u32);

            *g = v;
            last = vertices[i + k];
        }

        // if any value needs 4 bytes, or if *all* values need 3 bytes, we use 4 bytes for all values
        // this allows us to encode most 3-byte deltas with 3 bytes which saves space overall
        let use4 = (gv[0] | gv[1] | gv[2] | gv[3]) > 0xffffff
            || (gv[0] > 0xffff && gv[1] > 0xffff && gv[2] > 0xffff && gv[3] > 0xffff);

        for (k, &v) in gv.iter().enumerate() {
            // 0/1/2/3 bytes per value, or all 4 values use 4 bytes
            let code = if use4 {
                3
            } else if v == 0 {
                0
            } else if v < 256 {
                1
            } else if v < 65536 {
                2
            } else {
                3
            };

            if code > 0 {
                data[data_pos] = v as u8;
                data_pos += 1;
            }
            if code > 1 {
                data[data_pos] = (v >> 8) as u8;
                data_pos += 1;
            }
            if code > 2 {
                data[data_pos] = (v >> 16) as u8;
                data_pos += 1;
            }
            if use4 {
                data[data_pos] = (v >> 24) as u8;
                data_pos += 1;
            }

            // split low and high bits into two nibbles for better packing
            ctrl[i / 4] |= (((code & 1) << k) | ((code >> 1) << (k + 4))) as u8;
        }

        i += 4;
    }

    data_pos
}

fn decode_vertices_core(
    vertex_count: usize,
    ctrl: &[u8],
    mut data: usize,
    bound: usize,
    buffer: &[u8],
    mut write: impl FnMut(usize, u32),
) -> Option<usize> {
    let mut last: u32 = u32::MAX;

    let mut i = 0;
    while i < vertex_count {
        if data > bound {
            return None;
        }

        let code4 = ctrl[i / 4];

        for k in 0..4usize {
            let code = (((code4 >> k) & 1) | ((code4 >> (k + 3)) & 2)) as usize;
            let length = if code4 == 0xff { 4 } else { code };

            // branchlessly read up to 4 bytes
            let mask: u32 = if length == 4 {
                u32::MAX
            } else {
                (1u32 << (8 * length)) - 1
            };
            let mut word = [0u8; 4];
            word.copy_from_slice(&buffer[data..data + 4]);
            let v = u32::from_le_bytes(word) & mask;

            // unzigzag + 1
            let d = (v >> 1) ^ 0u32.wrapping_sub(v & 1);
            let r = last.wrapping_add(d).wrapping_add(1);

            if i + k < vertex_count {
                write(i + k, r);
            }

            data += length;
            last = r;
        }

        i += 4;
    }

    Some(data)
}

fn decode_triangles_core(
    triangle_count: usize,
    codes: &[u8],
    mut extra: usize,
    bound: usize,
    buffer: &[u8],
    mut write: impl FnMut(usize, u32),
) -> Option<usize> {
    let mut next: u32 = 0;
    let mut fifo = [0u32; 3]; // two edge fifo entries in one uint: 0xcbac

    for i in 0..triangle_count {
        if extra > bound {
            return None;
        }

        let code = u32::from((codes[i / 2] >> ((i & 1) * 4)) & 0xF);

        let tri = if code < 12 {
            // reuse
            let mut edge = fifo[(code / 4) as usize];
            edge >>= (code << 3) & 16; // shift by 16 if bit 1 is set (odd edge for each triangle)

            // branchlessly read next or extra vertex and advance pointers
            let e = u32::from(buffer[extra]);
            let c = if code & 1 == 1 { e } else { next };
            extra += (code & 1) as usize;
            next += 1 - (code & 1);

            // repack triangle into edge format (0xcbac)
            ((edge & 0xff) << 16) | (edge & 0xff00) | c | (c << 24)
        } else {
            // restart
            let fea = u32::from(code > 12);
            let feb = u32::from(code > 13);
            let fec = u32::from(code > 14);

            let mut take = |ec: u32| -> u32 {
                let e = u32::from(buffer[extra]);
                let v = if ec == 1 { e } else { next };
                extra += ec as usize;
                next += 1 - ec;
                v
            };

            // repack triangle into edge format (0xcbac)
            let a = take(fea);
            let b = take(feb);
            let c = take(fec);
            c | (a << 8) | (b << 16) | (c << 24)
        };

        // output triangle is stored without extra edge vertex (0xcbac => 0xcba)
        write(i, tri);

        fifo[2] = fifo[1];
        fifo[1] = fifo[0];
        fifo[0] = tri;
    }

    Some(extra)
}

pub fn encode_meshlet(buffer: &mut [u8], vertices: &[u32], triangles: &[u8]) -> usize {
    let vertex_count = vertices.len();
    let triangle_count = triangles.len() / 3;
    debug_assert!(triangle_count <= 256 && vertex_count <= 256);

    // 4 bits per triangle + up to three bytes of extra data
    let mut codes = [0u8; 128];
    let mut extra = [0u8; 256 * 3];
    let codes_size = triangle_count.div_ceil(2);
    let extra_size = encode_triangles(&mut codes, &mut extra, triangles, triangle_count);

    // 2 bits per vertex + up to 4 bytes of actual data
    let mut ctrl = [0u8; 64];
    let mut data = [0u8; 256 * 4];
    let ctrl_size = vertex_count.div_ceil(4);
    let data_size = encode_vertices(&mut ctrl, &mut data, vertices, vertex_count);

    // we need to ensure that up to 16 bytes after extra+data are available for SIMD decoding
    // to minimize overhead, we place fixed-size codes+control at the end of the buffer
    let gap_size = 16usize.saturating_sub(codes_size + ctrl_size);

    let result = codes_size + extra_size + ctrl_size + data_size + gap_size;

    if result > buffer.len() {
        return 0;
    }

    let mut pos = 0;

    // variable-size data first
    buffer[pos..pos + data_size].copy_from_slice(&data[..data_size]);
    pos += data_size;
    buffer[pos..pos + extra_size].copy_from_slice(&extra[..extra_size]);
    pos += extra_size;

    // gap (for accelerated decoding) separates variable-size and fixed-size data
    buffer[pos..pos + gap_size].fill(0);
    pos += gap_size;

    // fixed-size data last; it can be located from buffer end during decoding
    buffer[pos..pos + ctrl_size].copy_from_slice(&ctrl[..ctrl_size]);
    pos += ctrl_size;
    buffer[pos..pos + codes_size].copy_from_slice(&codes[..codes_size]);

    result
}

pub fn decode_meshlet(
    vertices: &mut [u8],
    vertex_count: usize,
    vertex_size: usize,
    triangles: &mut [u8],
    triangle_count: usize,
    triangle_size: usize,
    buffer: &[u8],
) -> bool {
    debug_assert!(triangle_count <= 256 && vertex_count <= 256);
    debug_assert!(vertex_size == 4 || vertex_size == 2);
    debug_assert!(triangle_size == 4 || triangle_size == 3);

    // layout must match encoding
    let codes_size = triangle_count.div_ceil(2);
    let ctrl_size = vertex_count.div_ceil(4);
    let gap_size = 16usize.saturating_sub(codes_size + ctrl_size);

    if buffer.len() < codes_size + ctrl_size + gap_size {
        return false;
    }

    let codes_start = buffer.len() - codes_size;
    let ctrl_start = codes_start - ctrl_size;
    let data_start = 0;

    // gap ensures we have at least 16 bytes available after bound; this allows SIMD decoders to over-read safely
    let bound = ctrl_start - gap_size;
    debug_assert!(bound >= data_start && bound + 16 <= buffer.len());

    let data = match vertex_size {
        4 => decode_vertices_core(
            vertex_count,
            &buffer[ctrl_start..],
            data_start,
            bound,
            buffer,
            &mut |idx: usize, r: u32| {
                vertices[idx * 4..idx * 4 + 4].copy_from_slice(&r.to_le_bytes());
            },
        ),
        _ => decode_vertices_core(
            vertex_count,
            &buffer[ctrl_start..],
            data_start,
            bound,
            buffer,
            &mut |idx: usize, r: u32| {
                vertices[idx * 2..idx * 2 + 2].copy_from_slice(&(r as u16).to_le_bytes());
            },
        ),
    };
    let Some(data) = data else {
        return false;
    };

    let data = match triangle_size {
        4 => decode_triangles_core(
            triangle_count,
            &buffer[codes_start..],
            data,
            bound,
            buffer,
            &mut |i: usize, tri: u32| {
                triangles[i * 4..i * 4 + 4].copy_from_slice(&(tri >> 8).to_le_bytes());
            },
        ),
        _ => decode_triangles_core(
            triangle_count,
            &buffer[codes_start..],
            data,
            bound,
            buffer,
            &mut |i: usize, tri: u32| {
                let t = tri >> 8;
                triangles[i * 3..i * 3 + 3].copy_from_slice(&[
                    t as u8,
                    (t >> 8) as u8,
                    (t >> 16) as u8,
                ]);
            },
        ),
    };
    let Some(data) = data else {
        return false;
    };

    data == bound
}

pub fn decode_meshlet_raw(
    vertices: &mut [u32],
    vertex_count: usize,
    triangles: &mut [u32],
    triangle_count: usize,
    buffer: &[u8],
) -> bool {
    debug_assert!(triangle_count <= 256 && vertex_count <= 256);

    // layout must match encoding
    let codes_size = triangle_count.div_ceil(2);
    let ctrl_size = vertex_count.div_ceil(4);
    let gap_size = 16usize.saturating_sub(codes_size + ctrl_size);

    if buffer.len() < codes_size + ctrl_size + gap_size {
        return false;
    }

    let codes_start = buffer.len() - codes_size;
    let ctrl_start = codes_start - ctrl_size;

    let bound = ctrl_start - gap_size;
    debug_assert!(bound + 16 <= buffer.len());

    let Some(data) = decode_vertices_core(
        vertex_count,
        &buffer[ctrl_start..],
        0,
        bound,
        buffer,
        &mut |idx: usize, r: u32| {
            vertices[idx] = r;
        },
    ) else {
        return false;
    };

    let Some(data) = decode_triangles_core(
        triangle_count,
        &buffer[codes_start..],
        data,
        bound,
        buffer,
        &mut |i: usize, tri: u32| {
            triangles[i] = tri >> 8;
        },
    ) else {
        return false;
    };

    data == bound
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_encodeMeshlet,
    // meshopt_encodeMeshletBound, meshopt_decodeMeshlet, meshopt_decodeMeshletRaw.
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

        fn u32(&mut self) -> u32 {
            let mut a = [0u8; 4];
            a.copy_from_slice(&self.data[self.pos..self.pos + 4]);
            self.pos += 4;
            u32::from_le_bytes(a)
        }

        fn i32(&mut self) -> i32 {
            self.u32() as i32
        }

        fn bytes(&mut self, n: usize) -> Vec<u8> {
            let s = &self.data[self.pos..self.pos + n];
            self.pos += n;
            s.to_vec()
        }

        fn vec_u32(&mut self, n: usize) -> Vec<u32> {
            (0..n).map(|_| self.u32()).collect()
        }
    }

    #[test]
    fn golden_encode_meshlet() {
        const _LEN_CHECK: usize = 6489;
        let data = include_bytes!("../tests/data/mc_enc.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let triangle_count = c.u32() as usize;
            let vertices = c.vec_u32(vertex_count);
            let triangles = c.bytes(triangle_count * 3);
            let encoded_size = c.u32() as usize;
            let encoded = c.bytes(encoded_size);

            let mut buffer = vec![0u8; encode_meshlet_bound(256, 512)];
            let result = encode_meshlet(&mut buffer, &vertices, &triangles);
            assert_eq!(
                result, encoded_size,
                "encode size mismatch (v={vertex_count}, t={triangle_count})"
            );
            assert_eq!(&buffer[..result], &encoded[..], "encode bytes mismatch");
        }
    }

    #[test]
    fn golden_decode_meshlet() {
        const _LEN_CHECK: usize = 12868;
        let data = include_bytes!("../tests/data/mc_dec.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let vertex_size = c.u32() as usize;
            let triangle_count = c.u32() as usize;
            let triangle_size = c.u32() as usize;
            let buffer_size = c.u32() as usize;
            let buffer = c.bytes(buffer_size);
            let result = c.i32();

            let mut vertices = vec![0u8; vertex_count * vertex_size];
            let mut triangles = vec![0u8; triangle_count * triangle_size];
            let ok = decode_meshlet(
                &mut vertices,
                vertex_count,
                vertex_size,
                &mut triangles,
                triangle_count,
                triangle_size,
                &buffer,
            );
            assert_eq!(
                ok,
                result == 0,
                "decode result mismatch (expected {result}, v={vertex_count}, t={triangle_count}, vs={vertex_size}, ts={triangle_size})"
            );
            if result == 0 {
                assert_eq!(
                    vertices.as_slice(),
                    &c.bytes(vertices.len())[..],
                    "vertices mismatch"
                );
                assert_eq!(
                    triangles.as_slice(),
                    &c.bytes(triangles.len())[..],
                    "triangles mismatch"
                );
            }
        }
    }

    #[test]
    fn golden_decode_meshlet_raw() {
        const _LEN_CHECK: usize = 3788;
        let data = include_bytes!("../tests/data/mc_dec_raw.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let triangle_count = c.u32() as usize;
            let buffer_size = c.u32() as usize;
            let buffer = c.bytes(buffer_size);
            let result = c.i32();

            let expected = if result == 0 {
                let vertices = c.vec_u32((vertex_count + 3) & !3);
                let triangles = c.vec_u32((triangle_count + 1) & !1);
                Some((vertices, triangles))
            } else {
                None
            };

            let mut vertices = vec![0u32; (vertex_count + 3) & !3];
            let mut triangles = vec![0u32; (triangle_count + 1) & !1];
            let ok = decode_meshlet_raw(
                &mut vertices,
                vertex_count,
                &mut triangles,
                triangle_count,
                &buffer,
            );
            assert_eq!(
                ok,
                result == 0,
                "raw decode result mismatch (expected {result}, v={vertex_count}, t={triangle_count})"
            );
            if let Some((expected_v, expected_t)) = expected {
                assert_eq!(vertices, expected_v, "raw vertices mismatch");
                assert_eq!(triangles, expected_t, "raw triangles mismatch");
            }
        }
    }

    #[test]
    fn golden_decode_meshlet_hostile() {
        const _LEN_CHECK: usize = 140051;
        let data = include_bytes!("../tests/data/mc_hostile.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let buffer_size = c.u32() as usize;
            let vertex_count = c.u32() as usize;
            let triangle_count = c.u32() as usize;
            let raw = c.u32() == 1;
            let buffer = c.bytes(buffer_size);
            let result = c.i32();

            let expected = if result == 0 {
                let vertex_size = 4usize;
                let triangle_size = if raw { 4 } else { 3 };
                let vertices = c.bytes(vertex_count * vertex_size);
                let triangles = c.bytes(triangle_count * triangle_size);
                Some((vertices, triangles))
            } else {
                None
            };

            let ok = if raw {
                let mut vertices = vec![0u32; vertex_count];
                let mut triangles = vec![0u32; triangle_count];
                let ok = decode_meshlet_raw(
                    &mut vertices,
                    vertex_count,
                    &mut triangles,
                    triangle_count,
                    &buffer,
                );
                if let Some((expected_v, expected_t)) = &expected {
                    for (dst, src) in vertices.iter().zip(expected_v.as_chunks::<4>().0) {
                        assert_eq!(dst.to_le_bytes(), *src, "hostile raw vertices mismatch");
                    }
                    for (dst, src) in triangles.iter().zip(expected_t.as_chunks::<4>().0) {
                        assert_eq!(dst.to_le_bytes(), *src, "hostile raw triangles mismatch");
                    }
                }
                ok
            } else {
                let mut vertices = vec![0u8; vertex_count * 4];
                let mut triangles = vec![0u8; triangle_count * 3];
                let ok = decode_meshlet(
                    &mut vertices,
                    vertex_count,
                    4,
                    &mut triangles,
                    triangle_count,
                    3,
                    &buffer,
                );
                if let Some((expected_v, expected_t)) = &expected {
                    assert_eq!(vertices.as_slice(), &expected_v[..]);
                    assert_eq!(triangles.as_slice(), &expected_t[..]);
                }
                ok
            };
            assert_eq!(
                ok,
                result == 0,
                "hostile decode result mismatch (expected {result}, raw={raw}, v={vertex_count}, t={triangle_count})"
            );
        }
    }
}
