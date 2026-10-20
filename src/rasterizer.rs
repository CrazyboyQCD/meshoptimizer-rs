//! Internal fixed-point software rasterizer with the overdraw and coverage
//! analyzers built on top of it.
//!
//! Upstream: `rasterizer.cpp` (`meshopt_analyzeOverdraw`, `meshopt_analyzeCoverage`).
//! Based on Nicolas Capens, Advanced Rasterization, 2004.

use alloc::vec;
use alloc::vec::Vec;

const K_VIEWPORT: f32 = 256.0;
const K_VIEWPORT_SIZE: usize = 256;

/// Upstream: `meshopt_OverdrawStatistics`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverdrawStatistics {
    pub pixels_covered: u32,
    pub pixels_shaded: u32,
    /// shaded pixels / covered pixels; best case 1.0
    pub overdraw: f32,
}

/// Upstream: `meshopt_CoverageStatistics`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverageStatistics {
    /// ratio of viewport pixels covered from each axis
    pub coverage: [f32; 3],
    /// viewport size in mesh coordinates
    pub extent: f32,
}

struct OverdrawBuffer {
    // index layout mirrors upstream: [y][x][s], s in {0, 1}
    z: Vec<f32>,
    overdraw: Vec<u32>,
}

impl OverdrawBuffer {
    fn new() -> Self {
        let len = K_VIEWPORT_SIZE * K_VIEWPORT_SIZE * 2;
        OverdrawBuffer {
            z: vec![0.0; len],
            overdraw: vec![0; len],
        }
    }

    fn clear(&mut self) {
        self.z.fill(0.0);
        self.overdraw.fill(0);
    }
}

fn compute_depth_gradients(v1: &[f32; 3], v2: &[f32; 3], v3: &[f32; 3]) -> (f32, f32, f32) {
    let (x1, y1, z1) = (v1[0], v1[1], v1[2]);
    let (x2, y2, z2) = (v2[0], v2[1], v2[2]);
    let (x3, y3, z3) = (v3[0], v3[1], v3[2]);

    // z2 = z1 + dzdx * (x2 - x1) + dzdy * (y2 - y1)
    // z3 = z1 + dzdx * (x3 - x1) + dzdy * (y3 - y1)
    // we'll solve it with Cramer's rule
    let det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
    let invdet = if det == 0.0 { 0.0 } else { 1.0 / det };

    let dzdx = ((z2 - z1) * (y3 - y1) - (y2 - y1) * (z3 - z1)) * invdet;
    let dzdy = ((x2 - x1) * (z3 - z1) - (z2 - z1) * (x3 - x1)) * invdet;

    (det, dzdx, dzdy)
}

// half-space fixed point triangle rasterizer
fn rasterize(buffer: &mut OverdrawBuffer, v1: [f32; 3], mut v2: [f32; 3], mut v3: [f32; 3]) {
    // compute depth gradients
    let (det, mut dzx, mut dzy) = compute_depth_gradients(&v1, &v2, &v3);
    let sign = usize::from(det > 0.0);
    let mut v1z = v1[2];

    // flip backfacing triangles to simplify rasterization logic
    if sign == 1 {
        // flipping v2 & v3 preserves depth gradients since they're based on v1; only v1z is
        // used below
        core::mem::swap(&mut v2[0], &mut v3[0]);
        core::mem::swap(&mut v2[1], &mut v3[1]);

        // flip depth since we rasterize backfacing triangles to second buffer with reverse Z;
        // only v1z is used below
        v1z = K_VIEWPORT - v1z;
        dzx = -dzx;
        dzy = -dzy;
    }

    // coordinates, 28.4 fixed point
    let x1 = (16.0 * v1[0] + 0.5) as i32;
    let x2 = (16.0 * v2[0] + 0.5) as i32;
    let x3 = (16.0 * v3[0] + 0.5) as i32;

    let y1 = (16.0 * v1[1] + 0.5) as i32;
    let y2 = (16.0 * v2[1] + 0.5) as i32;
    let y3 = (16.0 * v3[1] + 0.5) as i32;

    // bounding rectangle, clipped against viewport
    // since we rasterize pixels with covered centers, min >0.5 should round up
    // as for max, due to top-left filling convention we will never rasterize right/bottom
    // edges so max >= 0.5 should round down for inclusive bounds, and up for exclusive (ours)
    let mut minx = x1.min(x2).min(x3);
    minx = (minx + 7) >> 4;
    minx = minx.max(0);

    let mut miny = y1.min(y2).min(y3);
    miny = (miny + 7) >> 4;
    miny = miny.max(0);

    let mut maxx = x1.max(x2).max(x3);
    maxx = (maxx + 7) >> 4;
    maxx = maxx.min(K_VIEWPORT_SIZE as i32);

    let mut maxy = y1.max(y2).max(y3);
    maxy = (maxy + 7) >> 4;
    maxy = maxy.min(K_VIEWPORT_SIZE as i32);

    // deltas, 28.4 fixed point
    let dx12 = x1 - x2;
    let dx23 = x2 - x3;
    let dx31 = x3 - x1;

    let dy12 = y1 - y2;
    let dy23 = y2 - y3;
    let dy31 = y3 - y1;

    // fill convention correction
    let tl1 = i32::from(dy12 < 0 || (dy12 == 0 && dx12 > 0));
    let tl2 = i32::from(dy23 < 0 || (dy23 == 0 && dx23 > 0));
    let tl3 = i32::from(dy31 < 0 || (dy31 == 0 && dx31 > 0));

    // half edge equations, 24.8 fixed point
    // note that we offset minx/miny by half pixel since we want to rasterize pixels with
    // covered centers
    let fx = (minx << 4) + 8;
    let fy = (miny << 4) + 8;
    let mut cy1 = dx12 * (fy - y1) - dy12 * (fx - x1) + tl1 - 1;
    let mut cy2 = dx23 * (fy - y2) - dy23 * (fx - x2) + tl2 - 1;
    let mut cy3 = dx31 * (fy - y3) - dy31 * (fx - x3) + tl3 - 1;
    let mut zy = v1z + (dzx * (fx - x1) as f32 + dzy * (fy - y1) as f32) * 0.0625;

    for y in miny..maxy {
        let mut cx1 = cy1;
        let mut cx2 = cy2;
        let mut cx3 = cy3;
        let mut zx = zy;

        for x in minx..maxx {
            // check if all CXn are non-negative
            if (cx1 | cx2 | cx3) >= 0 {
                let idx = (y as usize * K_VIEWPORT_SIZE + x as usize) * 2 + sign;

                if zx >= buffer.z[idx] {
                    buffer.z[idx] = zx;
                    buffer.overdraw[idx] += 1;
                }
            }

            // signed left shift is UB for negative numbers in C so use unsigned casts
            cx1 -= ((dy12 as u32) << 4) as i32;
            cx2 -= ((dy23 as u32) << 4) as i32;
            cx3 -= ((dy31 as u32) << 4) as i32;
            zx += dzx;
        }

        cy1 += ((dx12 as u32) << 4) as i32;
        cy2 += ((dx23 as u32) << 4) as i32;
        cy3 += ((dx31 as u32) << 4) as i32;
        zy += dzy;
    }
}

fn transform_triangles(
    triangles: &mut [f32],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> f32 {
    let vertex_stride_float = vertex_positions_stride / 4;

    let mut minv = [f32::MAX; 3];
    let mut maxv = [-f32::MAX; 3];

    for i in 0..vertex_count {
        let v = &vertex_positions[i * vertex_stride_float..][..3];

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

    let scale = if extent == 0.0 {
        0.0
    } else {
        K_VIEWPORT / extent
    };

    for (&index, t) in indices.iter().zip(triangles.as_chunks_mut::<3>().0) {
        debug_assert!(index < vertex_count as u32);

        let v = &vertex_positions[index as usize * vertex_stride_float..][..3];

        t[0] = (v[0] - minv[0]) * scale;
        t[1] = (v[1] - minv[1]) * scale;
        t[2] = (v[2] - minv[2]) * scale;
    }

    extent
}

fn rasterize_triangles(
    buffer: &mut OverdrawBuffer,
    triangles: &[f32],
    index_count: usize,
    axis: usize,
) {
    let mut i = 0;
    while i < index_count {
        let v0 = [triangles[i * 3], triangles[i * 3 + 1], triangles[i * 3 + 2]];
        let v1 = [
            triangles[(i + 1) * 3],
            triangles[(i + 1) * 3 + 1],
            triangles[(i + 1) * 3 + 2],
        ];
        let v2 = [
            triangles[(i + 2) * 3],
            triangles[(i + 2) * 3 + 1],
            triangles[(i + 2) * 3 + 2],
        ];

        match axis {
            0 => rasterize(
                buffer,
                [v0[2], v0[1], v0[0]],
                [v1[2], v1[1], v1[0]],
                [v2[2], v2[1], v2[0]],
            ),
            1 => rasterize(
                buffer,
                [v0[0], v0[2], v0[1]],
                [v1[0], v1[2], v1[1]],
                [v2[0], v2[2], v2[1]],
            ),
            _ => rasterize(
                buffer,
                [v0[1], v0[0], v0[2]],
                [v1[1], v1[0], v1[2]],
                [v2[1], v2[0], v2[2]],
            ),
        }

        i += 3;
    }
}

fn stride_assert(vertex_positions_stride: usize) {
    debug_assert!((12..=256).contains(&vertex_positions_stride));
    debug_assert!(vertex_positions_stride.is_multiple_of(4));
}

/// Upstream: `meshopt_analyzeOverdraw`.
// port-shape: ported loops keep upstream index style
#[allow(clippy::needless_range_loop)]
pub fn analyze_overdraw(
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> OverdrawStatistics {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));
    stride_assert(vertex_positions_stride);

    let mut result = OverdrawStatistics {
        pixels_covered: 0,
        pixels_shaded: 0,
        overdraw: 0.0,
    };

    let mut triangles = vec![0f32; index_count * 3];
    transform_triangles(
        &mut triangles,
        indices,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
    );

    let mut buffer = OverdrawBuffer::new();

    for axis in 0..3 {
        buffer.clear();
        rasterize_triangles(&mut buffer, &triangles, index_count, axis);

        for idx in 0..K_VIEWPORT_SIZE * K_VIEWPORT_SIZE * 2 {
            let overdraw = buffer.overdraw[idx];

            result.pixels_covered += u32::from(overdraw > 0);
            result.pixels_shaded += overdraw;
        }
    }

    result.overdraw = if result.pixels_covered != 0 {
        result.pixels_shaded as f32 / result.pixels_covered as f32
    } else {
        0.0
    };

    result
}

/// Upstream: `meshopt_analyzeCoverage`.
// port-shape: ported loops keep upstream index style
#[allow(clippy::needless_range_loop)]
pub fn analyze_coverage(
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> CoverageStatistics {
    let index_count = indices.len();
    debug_assert!(index_count.is_multiple_of(3));
    stride_assert(vertex_positions_stride);

    let mut result = CoverageStatistics {
        coverage: [0.0; 3],
        extent: 0.0,
    };

    let mut triangles = vec![0f32; index_count * 3];
    result.extent = transform_triangles(
        &mut triangles,
        indices,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
    );

    let mut buffer = OverdrawBuffer::new();

    for axis in 0..3 {
        buffer.clear();
        rasterize_triangles(&mut buffer, &triangles, index_count, axis);

        let mut covered = 0u32;

        for idx in 0..K_VIEWPORT_SIZE * K_VIEWPORT_SIZE {
            covered += u32::from(buffer.overdraw[idx * 2] | buffer.overdraw[idx * 2 + 1] > 0);
        }

        result.coverage[axis] = covered as f32 / (K_VIEWPORT_SIZE * K_VIEWPORT_SIZE) as f32;
    }

    result
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_analyzeOverdraw,
    // meshopt_analyzeCoverage (FR-11; rasterization order is part of the
    // contract, so the internal rasterizer is pinned through these).
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

        fn f32(&mut self) -> f32 {
            f32::from_bits(self.u32())
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
    fn golden_analyze_overdraw() {
        const _LEN_CHECK: usize = 5728;
        let data = include_bytes!("../tests/data/analyze_overdraw.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let positions = c.f32s(vertex_count * stride);
            let indices = c.vec_u32(index_count);
            let expected = OverdrawStatistics {
                pixels_covered: c.u32(),
                pixels_shaded: c.u32(),
                overdraw: c.f32(),
            };

            let result = analyze_overdraw(&indices, &positions, vertex_count, stride);
            assert_eq!(result, expected, "overdraw v={vertex_count}");
        }
    }

    #[test]
    fn golden_analyze_coverage() {
        const _LEN_CHECK: usize = 5740;
        let data = include_bytes!("../tests/data/analyze_coverage.bin");
        assert_eq!(data.len(), _LEN_CHECK);
        let mut c = Cursor::new(data);
        while c.pos < data.len() {
            let vertex_count = c.u32() as usize;
            let stride = c.u32() as usize;
            let index_count = c.u32() as usize;
            let positions = c.f32s(vertex_count * stride);
            let indices = c.vec_u32(index_count);
            let expected = CoverageStatistics {
                coverage: [c.f32(), c.f32(), c.f32()],
                extent: c.f32(),
            };

            let result = analyze_coverage(&indices, &positions, vertex_count, stride);
            assert_eq!(result, expected, "coverage v={vertex_count}");
        }
    }
}
