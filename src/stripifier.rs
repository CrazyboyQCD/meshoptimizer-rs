//! Triangle stripification.
//!
//! Upstream: `stripifier.cpp` (`meshopt_stripify`, `meshopt_unstripify`).

use alloc::vec;

const BUFFER_CAPACITY: usize = 8;

fn find_strip_first(buffer: &[[u32; 3]], buffer_size: usize, valence: &[u8]) -> usize {
    let mut index = 0usize;
    let mut iv = !0u32;

    for i in 0..buffer_size {
        let va = u32::from(valence[buffer[i][0] as usize]);
        let vb = u32::from(valence[buffer[i][1] as usize]);
        let vc = u32::from(valence[buffer[i][2] as usize]);
        let v = if va < vb && va < vc {
            va
        } else if vb < vc {
            vb
        } else {
            vc
        };

        if v < iv {
            index = i;
            iv = v;
        }
    }

    index
}

fn find_strip_next(buffer: &[[u32; 3]], buffer_size: usize, e0: u32, e1: u32) -> i32 {
    for (i, tri) in buffer.iter().enumerate().take(buffer_size) {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];

        if e0 == a && e1 == b {
            return ((i as i32) << 2) | 2;
        } else if e0 == b && e1 == c {
            // (int(i) << 2) | 0 upstream; the `| 0` is a no-op
            return (i as i32) << 2;
        } else if e0 == c && e1 == a {
            return ((i as i32) << 2) | 1;
        }
    }

    -1
}

/// Upstream: `meshopt_stripify`. Converts a triangle list to a strip, stitching
/// with `restart_index` (0 = degenerate triangles).
pub fn stripify(
    destination: &mut [u32],
    indices: &[u32],
    vertex_count: usize,
    restart_index: u32,
) -> usize {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));
    debug_assert!(destination.len() >= stripify_bound(index_count));

    let mut buffer = [[0u32; 3]; BUFFER_CAPACITY];
    let mut buffer_size = 0usize;

    let mut index_offset = 0usize;

    let mut strip = [0u32; 2];
    let mut parity = 0u32;

    let mut strip_size = 0usize;

    // compute vertex valence; this is used to prioritize starting triangle for strips
    // note: we use 8-bit counters for performance; for outlier vertices the valence is
    // incorrect but that just affects the heuristic, so the wrap is intentional
    let mut valence = vec![0u8; vertex_count];

    for &index in indices.iter() {
        debug_assert!(index < vertex_count as u32);
        valence[index as usize] = valence[index as usize].wrapping_add(1);
    }

    let mut next = -1i32;

    while buffer_size > 0 || index_offset < index_count {
        debug_assert!(next < 0 || ((next >> 2) as usize) < buffer_size && (next & 3) < 3);

        // fill triangle buffer
        while buffer_size < BUFFER_CAPACITY && index_offset < index_count {
            buffer[buffer_size][0] = indices[index_offset];
            buffer[buffer_size][1] = indices[index_offset + 1];
            buffer[buffer_size][2] = indices[index_offset + 2];

            buffer_size += 1;
            index_offset += 3;
        }

        debug_assert!(buffer_size > 0);

        if next >= 0 {
            let i = (next >> 2) as usize;
            let a = buffer[i][0];
            let b = buffer[i][1];
            let c = buffer[i][2];
            let v = buffer[i][(next & 3) as usize];

            // ordered removal from the buffer
            buffer.copy_within(i + 1..buffer_size, i);
            buffer_size -= 1;

            // update vertex valences for strip start heuristic
            valence[a as usize] = valence[a as usize].wrapping_sub(1);
            valence[b as usize] = valence[b as usize].wrapping_sub(1);
            valence[c as usize] = valence[c as usize].wrapping_sub(1);

            // find next triangle (note that edge order flips on every iteration)
            // in some cases we need to perform a swap to pick a different outgoing triangle edge
            // for [a b c], the default strip edge is [b c], but we might want to use [a c]
            let cont = if valence[strip[1] as usize] != 0 {
                find_strip_next(
                    &buffer,
                    buffer_size,
                    if parity != 0 { strip[1] } else { v },
                    if parity != 0 { v } else { strip[1] },
                )
            } else {
                -1
            };
            let swap = if cont < 0 && valence[strip[0] as usize] != 0 {
                find_strip_next(
                    &buffer,
                    buffer_size,
                    if parity != 0 { v } else { strip[0] },
                    if parity != 0 { strip[0] } else { v },
                )
            } else {
                -1
            };

            if cont < 0 && swap >= 0 {
                // [a b c] => [a b a c]
                destination[strip_size] = strip[0];
                strip_size += 1;
                destination[strip_size] = v;
                strip_size += 1;

                // next strip has same winding
                // ? a b => b a v
                strip[1] = v;

                next = swap;
            } else {
                // emit the next vertex in the strip
                destination[strip_size] = v;
                strip_size += 1;

                // next strip has flipped winding
                strip[0] = strip[1];
                strip[1] = v;
                parity ^= 1;

                next = cont;
            }
        } else {
            // if we didn't find anything, we need to find the next new triangle
            // we use a heuristic to maximize the strip length
            let i = find_strip_first(&buffer, buffer_size, &valence);
            let mut a = buffer[i][0];
            let mut b = buffer[i][1];
            let mut c = buffer[i][2];

            // ordered removal from the buffer
            buffer.copy_within(i + 1..buffer_size, i);
            buffer_size -= 1;

            // update vertex valences for strip start heuristic
            valence[a as usize] = valence[a as usize].wrapping_sub(1);
            valence[b as usize] = valence[b as usize].wrapping_sub(1);
            valence[c as usize] = valence[c as usize].wrapping_sub(1);

            // we need to pre-rotate the triangle so that we will find a match in the existing
            // buffer on the next iteration
            let ea = find_strip_next(&buffer, buffer_size, c, b);
            let eb = find_strip_next(&buffer, buffer_size, a, c);
            let ec = find_strip_next(&buffer, buffer_size, b, a);

            // in some cases we can have several matching edges; since we can pick any edge, we
            // pick the one with the smallest triangle index in the buffer. this reduces the
            // effect of stripification on ACMR and additionally - for unclear reasons -
            // slightly improves the stripification efficiency
            let mut mine = i32::MAX;
            mine = if ea >= 0 && mine > ea { ea } else { mine };
            mine = if eb >= 0 && mine > eb { eb } else { mine };
            mine = if ec >= 0 && mine > ec { ec } else { mine };

            if ea == mine {
                // keep abc
                next = ea;
            } else if eb == mine {
                // abc -> bca
                let t = a;
                a = b;
                b = c;
                c = t;

                next = eb;
            } else if ec == mine {
                // abc -> cab
                let t = c;
                c = b;
                b = a;
                a = t;

                next = ec;
            }

            if restart_index != 0 {
                if strip_size != 0 {
                    destination[strip_size] = restart_index;
                    strip_size += 1;
                }

                destination[strip_size] = a;
                strip_size += 1;
                destination[strip_size] = b;
                strip_size += 1;
                destination[strip_size] = c;
                strip_size += 1;

                // new strip always starts with the same edge winding
                strip[0] = b;
                strip[1] = c;
                parity = 1;
            } else {
                if strip_size != 0 {
                    // connect last strip using degenerate triangles
                    destination[strip_size] = strip[1];
                    strip_size += 1;
                    destination[strip_size] = a;
                    strip_size += 1;
                }

                // note that we may need to flip the emitted triangle based on parity
                // we always end up with outgoing edge "cb" in the end
                let e0 = if parity != 0 { c } else { b };
                let e1 = if parity != 0 { b } else { c };

                destination[strip_size] = a;
                strip_size += 1;
                destination[strip_size] = e0;
                strip_size += 1;
                destination[strip_size] = e1;
                strip_size += 1;

                strip[0] = e0;
                strip[1] = e1;
                parity ^= 1;
            }
        }
    }

    strip_size
}

/// Upstream: `meshopt_stripifyBound`.
pub const fn stripify_bound(index_count: usize) -> usize {
    (index_count / 3) * 5
}

/// Upstream: `meshopt_unstripify`. Converts a strip back to a triangle list.
pub fn unstripify(destination: &mut [u32], indices: &[u32], restart_index: u32) -> usize {
    let index_count = indices.len();
    debug_assert!(destination.len() >= unstripify_bound(index_count));

    let mut offset = 0usize;
    let mut start = 0usize;

    for i in 0..index_count {
        if restart_index != 0 && indices[i] == restart_index {
            start = i + 1;
        } else if i >= start + 2 {
            let mut a = indices[i - 2];
            let mut b = indices[i - 1];
            let c = indices[i];

            // flip winding for odd triangles
            if (i - start) & 1 != 0 {
                core::mem::swap(&mut a, &mut b);
            }

            // although we use restart indices, strip swaps still produce degenerate
            // triangles, so skip them
            if a != b && a != c && b != c {
                destination[offset] = a;
                destination[offset + 1] = b;
                destination[offset + 2] = c;
                offset += 3;
            }
        }
    }

    offset
}

/// Upstream: `meshopt_unstripifyBound`.
pub const fn unstripify_bound(index_count: usize) -> usize {
    if index_count == 0 {
        0
    } else {
        (index_count - 2) * 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        fn vec_u32(&mut self, n: usize) -> Vec<u32> {
            (0..n).map(|_| self.u32()).collect()
        }
    }

    #[test]
    fn bounds() {
        assert_eq!(stripify_bound(0), 0);
        assert_eq!(stripify_bound(3), 5);
        assert_eq!(stripify_bound(300), 500);
        assert_eq!(unstripify_bound(0), 0);
        assert_eq!(unstripify_bound(3), 3);
        assert_eq!(unstripify_bound(300), 894);
    }

    #[test]
    fn golden_stripify() {
        const _LEN_CHECK: usize = 11816;
        let data = include_bytes!("../tests/data/stripify.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let index_count = c.u32() as usize;
            let vertex_count = c.u32() as usize;
            let restart_index = c.u32();
            let indices = c.vec_u32(index_count);
            let strip_len = c.u32() as usize;
            let expected = c.vec_u32(strip_len);

            let mut destination = vec![0xeeu32; stripify_bound(index_count)];
            let result = stripify(&mut destination, &indices, vertex_count, restart_index);
            assert_eq!(
                result, strip_len,
                "stripify v={vertex_count} r={restart_index}"
            );
            assert_eq!(
                &destination[..result],
                expected.as_slice(),
                "stripify v={vertex_count} r={restart_index}"
            );
        }
    }

    #[test]
    fn golden_unstripify() {
        const _LEN_CHECK: usize = 10544;
        let data = include_bytes!("../tests/data/unstripify.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let strip_len = c.u32() as usize;
            let restart_index = c.u32();
            let strip = c.vec_u32(strip_len);
            let tri_count = c.u32() as usize;
            let expected = c.vec_u32(tri_count);

            let mut destination = vec![0xeeu32; unstripify_bound(strip_len)];
            let result = unstripify(&mut destination, &strip, restart_index);
            assert_eq!(result, tri_count, "unstripify r={restart_index}");
            assert_eq!(
                &destination[..result],
                expected.as_slice(),
                "unstripify r={restart_index}"
            );
        }
    }
}
