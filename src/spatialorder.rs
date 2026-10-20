//! Morton-code spatial reordering.
//!
//! Upstream: `spatialorder.cpp` (`meshopt_spatialSortRemap`,
//! `meshopt_spatialSortTriangles`, `meshopt_spatialClusterPoints`).

use alloc::vec;

// "Insert" two 0 bits after each of the 20 low bits of x
fn part1by2(mut x: u64) -> u64 {
    x &= 0x000fffff;
    x = (x ^ (x << 32)) & 0x000f00000000ffff;
    x = (x ^ (x << 16)) & 0x000f0000ff0000ff;
    x = (x ^ (x << 8)) & 0x000f00f00f00f00f;
    x = (x ^ (x << 4)) & 0x00c30c30c30c30c3;
    x = (x ^ (x << 2)) & 0x0249249249249249;
    x
}

fn compute_order(
    result: &mut [u64],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    morton: bool,
) {
    let vertex_stride_float = vertex_positions_stride / 4;

    let mut minv = [f32::MAX; 3];
    let mut maxv = [-f32::MAX; 3];

    let vertex_count_positions = &vertex_positions[..vertex_count * vertex_stride_float];

    for p in vertex_count_positions.chunks_exact(vertex_stride_float) {
        let v = &p[..3];

        for j in 0..3 {
            let vj = v[j];

            if minv[j] > vj {
                minv[j] = vj;
            }
            if maxv[j] < vj {
                maxv[j] = vj;
            }
        }
    }

    let mut extent = 0f32;

    for j in 0..3 {
        let e = maxv[j] - minv[j];
        extent = if e < extent { extent } else { e };
    }

    // rescale each axis to 16 bits to get 48-bit Morton codes
    let scale = if extent == 0.0 { 0.0 } else { 65535.0 / extent };

    // generate Morton order based on the position inside a unit cube
    for (i, p) in vertex_count_positions
        .chunks_exact(vertex_stride_float)
        .enumerate()
    {
        let v = &p[..3];

        let x = (v[0] - minv[0]) * scale + 0.5;
        let y = (v[1] - minv[1]) * scale + 0.5;
        let z = (v[2] - minv[2]) * scale + 0.5;

        let (x, y, z) = (x as i32, y as i32, z as i32);

        result[i] = if morton {
            part1by2(x as u64) | (part1by2(y as u64) << 1) | (part1by2(z as u64) << 2)
        } else {
            (x as u64) | ((y as u64) << 20) | ((z as u64) << 40)
        };
    }
}

fn radix_sort10(destination: &mut [u32], source: &[u32], keys: &[u16]) {
    let mut hist = [0u32; 1024];

    // compute histogram (assume keys are 10-bit)
    for &key in keys.iter() {
        hist[key as usize] += 1;
    }

    let mut sum = 0u32;

    // replace histogram data with prefix histogram sums in-place
    for h in hist.iter_mut() {
        let v = *h;
        *h = sum;
        sum += v;
    }

    debug_assert!(sum as usize == source.len());

    // reorder values
    for &src in source.iter() {
        let id = keys[src as usize] as usize;

        destination[hist[id] as usize] = src;
        hist[id] += 1;
    }
}

fn compute_histogram(hist: &mut [[u32; 2]; 256], data: &[u16]) {
    hist.fill([0; 2]);

    // compute 2 8-bit histograms in parallel
    for &id in data.iter() {
        let id = id as u64;

        hist[(id & 255) as usize][0] += 1;
        hist[((id >> 8) & 255) as usize][1] += 1;
    }

    let (mut sum0, mut sum1) = (0u32, 0u32);

    // replace histogram data with prefix histogram sums in-place
    for h in hist.iter_mut() {
        let (h0, h1) = (h[0], h[1]);

        h[0] = sum0;
        h[1] = sum1;

        sum0 += h0;
        sum1 += h1;
    }

    debug_assert!(sum0 as usize == data.len() && sum1 as usize == data.len());
}

fn radix_pass(
    destination: &mut [u32],
    source: &[u32],
    keys: &[u16],
    hist: &mut [[u32; 2]; 256],
    pass: usize,
) {
    let bitoff = pass * 8;

    for &src in source.iter() {
        let id = (((keys[src as usize] >> bitoff) & 255) as u32) as usize;

        let slot = hist[id][pass] as usize;
        destination[slot] = src;
        hist[id][pass] += 1;
    }
}

fn partition_points(target: &mut [u32], order: &[u32], sides: &[u8], split: usize) {
    let count = order.len();
    let (mut l, mut r) = (0usize, split);

    for &point in order.iter() {
        let side = u32::from(sides[point as usize] != 0);
        let slot = if side != 0 { r } else { l };
        target[slot] = point;
        l += 1;
        l -= side as usize;
        r += side as usize;
    }

    debug_assert!(l == split && r == count);
}

// mirrors the upstream pointer-triplet recursion shape; the axis count is intrinsic
#[allow(clippy::too_many_arguments)]
fn split_points(
    destination: &mut [u32],
    orderx: &mut [u32],
    ordery: &mut [u32],
    orderz: &mut [u32],
    keys: &[u64],
    scratch: &mut [u32],
    sides: &mut [u8],
    cluster_size: usize,
) {
    let count = destination.len();

    if count <= cluster_size {
        destination[..count].copy_from_slice(&orderx[..count]);
        return;
    }

    let mut bestk = 0usize;
    let mut bestdim = 0u32;

    for k in 0..3usize {
        let axis: &[u32] = match k {
            0 => orderx,
            1 => ordery,
            _ => orderz,
        };
        let mask = (1u32 << 20) - 1;
        let hi = ((keys[axis[count - 1] as usize] >> (k * 20)) as u32) & mask;
        let lo = ((keys[axis[0] as usize] >> (k * 20)) as u32) & mask;
        let dim = hi.wrapping_sub(lo);

        if dim >= bestdim {
            bestk = k;
            bestdim = dim;
        }
    }

    // split roughly in half, with the left split always being aligned to cluster size
    let split = (count / 2).div_ceil(cluster_size) * cluster_size;
    debug_assert!(split > 0 && split < count);

    // mark sides of split for partitioning; the sides array covers the full vertex id space
    let best_axis: &[u32] = match bestk {
        0 => orderx,
        1 => ordery,
        _ => orderz,
    };
    for &point in best_axis[..split].iter() {
        sides[point as usize] = 0;
    }
    for &point in best_axis[split..count].iter() {
        sides[point as usize] = 1;
    }

    // partition all axes into two sides, maintaining order
    let temp = &mut scratch[..count];

    for k in 0..3usize {
        if k == bestk {
            continue;
        }

        match k {
            0 => {
                temp.copy_from_slice(&orderx[..count]);
                partition_points(orderx, temp, sides, split);
            }
            1 => {
                temp.copy_from_slice(&ordery[..count]);
                partition_points(ordery, temp, sides, split);
            }
            _ => {
                temp.copy_from_slice(&orderz[..count]);
                partition_points(orderz, temp, sides, split);
            }
        }
    }

    // recursion depth is logarithmic and bounded as we always split in approximately half
    // note: keys stay global since order entries are global vertex ids
    let (dest_l, dest_r) = destination.split_at_mut(split);
    let (ox_l, ox_r) = orderx.split_at_mut(split);
    let (oy_l, oy_r) = ordery.split_at_mut(split);
    let (oz_l, oz_r) = orderz.split_at_mut(split);
    let (s_l, s_r) = scratch.split_at_mut(split);

    split_points(dest_l, ox_l, oy_l, oz_l, keys, s_l, sides, cluster_size);
    split_points(dest_r, ox_r, oy_r, oz_r, keys, s_r, sides, cluster_size);
}

fn stride_assert(vertex_positions_stride: usize) {
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
}

/// Upstream: `meshopt_spatialSortRemap`.
pub fn spatial_sort_remap(
    destination: &mut [u32],
    vertex_positions: &[f32],
    vertex_positions_stride: usize,
) {
    let vertex_count = destination.len();
    stride_assert(vertex_positions_stride);

    let mut keys = vec![0u64; vertex_count];
    compute_order(
        &mut keys,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        true,
    );

    let mut scratch = vec![0u32; vertex_count]; // ordering scratch
    let mut keyk = vec![0u16; vertex_count];

    for (i, d) in destination.iter_mut().enumerate() {
        *d = i as u32;
    }

    // 5-pass radix sort computes the resulting order into scratch
    for k in 0..5i32 {
        // copy 10-bit key segments into keyk to reduce cache pressure during radix pass
        for i in 0..vertex_count {
            keyk[i] = ((keys[i] >> (k * 10)) & 1023) as u16;
        }

        if k % 2 == 0 {
            radix_sort10(&mut scratch, destination, &keyk);
        } else {
            radix_sort10(destination, &scratch, &keyk);
        }
    }

    // since our remap table is mapping old=>new, we need to reverse it
    for i in 0..vertex_count {
        destination[scratch[i] as usize] = i as u32;
    }
}

/// Upstream: `meshopt_spatialSortTriangles`.
pub fn spatial_sort_triangles(
    destination: &mut [u32],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));
    stride_assert(vertex_positions_stride);

    let face_count = index_count / 3;
    let vertex_stride_float = vertex_positions_stride / 4;

    let mut centroids = vec![0f32; face_count * 3];

    for (tri_out, tri_in) in centroids
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(indices.as_chunks::<3>().0)
    {
        let [a, b, c] = *tri_in;
        let (a, b, c) = (a as usize, b as usize, c as usize);
        debug_assert!(a < vertex_count && b < vertex_count && c < vertex_count);

        let va = &vertex_positions[a * vertex_stride_float..][..3];
        let vb = &vertex_positions[b * vertex_stride_float..][..3];
        let vc = &vertex_positions[c * vertex_stride_float..][..3];

        tri_out[0] = (va[0] + vb[0] + vc[0]) / 3.0;
        tri_out[1] = (va[1] + vb[1] + vc[1]) / 3.0;
        tri_out[2] = (va[2] + vb[2] + vc[2]) / 3.0;
    }

    let mut remap = vec![0u32; face_count];

    spatial_sort_remap(&mut remap, &centroids, 12);

    for (tri, &r) in indices.as_chunks::<3>().0.iter().zip(remap.iter()) {
        let [a, b, c] = *tri;
        let r = r as usize;

        destination[r * 3] = a;
        destination[r * 3 + 1] = b;
        destination[r * 3 + 2] = c;
    }
}

/// Upstream: `meshopt_spatialClusterPoints`.
pub fn spatial_cluster_points(
    destination: &mut [u32],
    vertex_positions: &[f32],
    vertex_positions_stride: usize,
    cluster_size: usize,
) {
    let vertex_count = destination.len();
    stride_assert(vertex_positions_stride);
    debug_assert!(cluster_size > 0);

    let mut keys = vec![0u64; vertex_count];
    compute_order(
        &mut keys,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        false,
    );

    let mut order = vec![0u32; vertex_count * 3];
    let mut scratch = vec![0u32; vertex_count];
    let mut keyk = vec![0u16; vertex_count];

    for k in 0..3usize {
        // copy 16-bit key segments into keyk to reduce cache pressure during radix pass
        for i in 0..vertex_count {
            keyk[i] = (keys[i] >> (k * 20)) as u16;
        }

        let mut hist = [[0u32; 2]; 256];
        compute_histogram(&mut hist, &keyk);

        for (i, o) in order[k * vertex_count..(k + 1) * vertex_count]
            .iter_mut()
            .enumerate()
        {
            *o = i as u32;
        }

        let axis = &mut order[k * vertex_count..(k + 1) * vertex_count];

        radix_pass(&mut scratch, axis, &keyk, &mut hist, 0);
        radix_pass(axis, &scratch, &keyk, &mut hist, 1);
    }

    let mut sides = vec![0u8; vertex_count];

    let (ox, rest) = order.split_at_mut(vertex_count);
    let (oy, oz) = rest.split_at_mut(vertex_count);

    split_points(
        destination,
        ox,
        oy,
        oz,
        &keys,
        &mut scratch,
        &mut sides,
        cluster_size,
    );
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_spatialSortRemap,
    // meshopt_spatialSortTriangles, meshopt_spatialClusterPoints.
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

        fn bytes(&mut self, n: usize) -> &'a [u8] {
            let s = &self.data[self.pos..self.pos + n];
            self.pos += n;
            s
        }

        fn vec_u32(&mut self, n: usize) -> Vec<u32> {
            (0..n).map(|_| self.u32()).collect()
        }

        fn f32s(&mut self, byte_len: usize) -> Vec<f32> {
            let raw = self.bytes(byte_len);
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect()
        }
    }

    #[test]
    fn golden_spatial_sort_remap() {
        const _LEN_CHECK: usize = 22188;
        let data = include_bytes!("../tests/data/spatial_remap.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let positions = c.f32s(vertex_count * stride);
            let expected = c.vec_u32(vertex_count);

            let mut destination = vec![0xeeu32; vertex_count];
            spatial_sort_remap(&mut destination, &positions, stride);
            assert_eq!(destination, expected, "spatial remap v={vertex_count}");
        }
    }

    #[test]
    fn golden_spatial_sort_triangles() {
        const _LEN_CHECK: usize = 8344;
        let data = include_bytes!("../tests/data/spatial_tri.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let positions = c.f32s(vertex_count * stride);
            let indices = c.vec_u32(index_count);
            let expected = c.vec_u32(index_count);

            let mut destination = vec![0xeeu32; index_count];
            spatial_sort_triangles(&mut destination, &indices, &positions, vertex_count, stride);
            assert_eq!(destination, expected, "spatial tri v={vertex_count}");
        }
    }

    #[test]
    fn golden_spatial_cluster_points() {
        const _LEN_CHECK: usize = 8248;
        let data = include_bytes!("../tests/data/spatial_cluster.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let cluster_size = c.u32() as usize;
            let positions = c.f32s(vertex_count * stride);
            let expected = c.vec_u32(vertex_count);

            let mut destination = vec![0xeeu32; vertex_count];
            spatial_cluster_points(&mut destination, &positions, stride, cluster_size);
            assert_eq!(
                destination, expected,
                "spatial cluster v={vertex_count} cs={cluster_size}"
            );
        }
    }
}
