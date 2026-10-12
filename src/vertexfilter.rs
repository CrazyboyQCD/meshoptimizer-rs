//! Vertex filters for post-vertex-codec reconstruction and pre-encode packing.
//!
//! Upstream: `vertexfilter.cpp` (`meshopt_decodeFilterOct`, `meshopt_decodeFilterQuat`,
//! `meshopt_decodeFilterExp`, `meshopt_decodeFilterColor`, `meshopt_encodeFilterOct`,
//! `meshopt_encodeFilterQuat`, `meshopt_encodeFilterExp`, `meshopt_encodeFilterColor`).
//! Scalar kernels only; upstream's SIMD filter kernels are not output-identical and
//! are out of scope permanently ([ADR-0010](../docs/adr/0010-scalar-profile-fixtures.md)).

use crate::math;

/// Upstream: `enum meshopt_EncodeExpMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeExpMode {
    /// Use separate values for each component (maximum quality).
    Separate = 0,
    /// Use a shared value for all components of each vector (better compression).
    SharedVector = 1,
    /// Use a shared value for each component of all vectors (best compression).
    SharedComponent = 2,
    /// Use separate values for each component, clamped to 0.
    Clamped = 3,
}

// C float->int casts truncate toward zero and produce INT_MIN for NaN,
// +-inf and out-of-range values on x86 (cvttss2si); Rust saturates instead,
// so hostile filter inputs would diverge without this wrapper. 2^31 itself
// is not representable as i32 and casts to INT_MIN on x86 as well.
fn cast_i32(v: f32) -> i32 {
    if v.is_nan() || v >= 2147483648.0 || v < -2147483648.0 {
        i32::MIN
    } else {
        v as i32
    }
}

// upstream header inlines (meshoptimizer.h): meshopt_quantizeSnorm/Unorm
fn quantize_unorm(v: f32, n: i32) -> i32 {
    let scale = ((1i32 << n) - 1) as f32;

    let v = if v >= 0.0 { v } else { 0.0 };
    let v = if v <= 1.0 { v } else { 1.0 };

    cast_i32(v * scale + 0.5)
}

fn quantize_snorm(v: f32, n: i32) -> i32 {
    let scale = ((1i32 << (n - 1)) - 1) as f32;

    let round = if v >= 0.0 { 0.5 } else { -0.5 };

    let v = if v >= -1.0 { v } else { -1.0 };
    let v = if v <= 1.0 { v } else { 1.0 };

    cast_i32(v * scale + round)
}

fn decode_filter_oct_s8(data: &mut [u8], count: usize) {
    let max = 127f32;

    for i in 0..count {
        // convert x and y to floats and reconstruct z; this assumes zf encodes 1.f at the same bit count
        let x = f32::from(data[i * 4] as i8);
        let y = f32::from(data[i * 4 + 1] as i8);
        let z = f32::from(data[i * 4 + 2] as i8) - x.abs() - y.abs();

        // fixup octahedral coordinates for z<0
        let t = if z >= 0.0 { 0.0 } else { z };

        let x = x + if x >= 0.0 { t } else { -t };
        let y = y + if y >= 0.0 { t } else { -t };

        // compute normal length & scale
        let l = math::sqrt(x * x + y * y + z * z);
        let s = max / l;

        // rounded signed float->int
        let xf = cast_i32(x * s + if x >= 0.0 { 0.5 } else { -0.5 });
        let yf = cast_i32(y * s + if y >= 0.0 { 0.5 } else { -0.5 });
        let zf = cast_i32(z * s + if z >= 0.0 { 0.5 } else { -0.5 });

        data[i * 4] = xf as i8 as u8;
        data[i * 4 + 1] = yf as i8 as u8;
        data[i * 4 + 2] = zf as i8 as u8;
    }
}

fn decode_filter_oct_s16(data: &mut [u8], count: usize) {
    let max = 32767f32;

    for i in 0..count {
        let xi = i16::from_le_bytes([data[i * 8], data[i * 8 + 1]]);
        let yi = i16::from_le_bytes([data[i * 8 + 2], data[i * 8 + 3]]);
        let zi = i16::from_le_bytes([data[i * 8 + 4], data[i * 8 + 5]]);

        let x = f32::from(xi);
        let y = f32::from(yi);
        let z = f32::from(zi) - x.abs() - y.abs();

        let t = if z >= 0.0 { 0.0 } else { z };

        let x = x + if x >= 0.0 { t } else { -t };
        let y = y + if y >= 0.0 { t } else { -t };

        let l = math::sqrt(x * x + y * y + z * z);
        let s = max / l;

        let xf = cast_i32(x * s + if x >= 0.0 { 0.5 } else { -0.5 });
        let yf = cast_i32(y * s + if y >= 0.0 { 0.5 } else { -0.5 });
        let zf = cast_i32(z * s + if z >= 0.0 { 0.5 } else { -0.5 });

        data[i * 8..i * 8 + 2].copy_from_slice(&(xf as i16).to_le_bytes());
        data[i * 8 + 2..i * 8 + 4].copy_from_slice(&(yf as i16).to_le_bytes());
        data[i * 8 + 4..i * 8 + 6].copy_from_slice(&(zf as i16).to_le_bytes());
    }
}

fn decode_filter_quat_s16(data: &mut [u8], count: usize) {
    let scale = 32767f32 / math::sqrt(2.0);

    for i in 0..count {
        let wi = i16::from_le_bytes([data[i * 8 + 6], data[i * 8 + 7]]);

        // recover scale from the high byte/word of the component
        let sf = i32::from(wi) | 3;
        let s = sf as f32;

        // convert x/y/z to floating point (unscaled! implied scale of 1/sqrt(2.f) * 1/sf)
        let x = f32::from(i16::from_le_bytes([data[i * 8], data[i * 8 + 1]]));
        let y = f32::from(i16::from_le_bytes([data[i * 8 + 2], data[i * 8 + 3]]));
        let z = f32::from(i16::from_le_bytes([data[i * 8 + 4], data[i * 8 + 5]]));

        // reconstruct w as a square root (unscaled); we clamp to 0.f to avoid NaN due to precision errors
        let ws = s * s;
        let ww = ws * 2.0 - x * x - y * y - z * z;
        let w = math::sqrt(if ww >= 0.0 { ww } else { 0.0 });

        // compute final scale; note that all computations above are unscaled
        let ss = scale / s;

        // rounded signed float->int
        let xf = cast_i32(x * ss + if x >= 0.0 { 0.5 } else { -0.5 });
        let yf = cast_i32(y * ss + if y >= 0.0 { 0.5 } else { -0.5 });
        let zf = cast_i32(z * ss + if z >= 0.0 { 0.5 } else { -0.5 });
        let wf = cast_i32(w * ss + 0.5);

        let qc = (i32::from(wi) & 3) as usize;

        // output order is dictated by input index
        let w0 = (xf as i16).to_le_bytes();
        let w1 = (yf as i16).to_le_bytes();
        let w2 = (zf as i16).to_le_bytes();
        let w3 = (wf as i16).to_le_bytes();

        data[i * 8 + ((qc + 1) & 3) * 2..i * 8 + ((qc + 1) & 3) * 2 + 2].copy_from_slice(&w0);
        data[i * 8 + ((qc + 2) & 3) * 2..i * 8 + ((qc + 2) & 3) * 2 + 2].copy_from_slice(&w1);
        data[i * 8 + ((qc + 3) & 3) * 2..i * 8 + ((qc + 3) & 3) * 2 + 2].copy_from_slice(&w2);
        data[i * 8 + (qc & 3) * 2..i * 8 + (qc & 3) * 2 + 2].copy_from_slice(&w3);
    }
}

fn decode_filter_exp_u32(data: &mut [u8], count: usize) {
    for i in 0..count {
        let v = u32::from_le_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]);

        // decode mantissa and exponent
        let m = ((v << 8) as i32) >> 8;
        let e = (v as i32) >> 24;

        // optimized version of ldexp(float(m), e)
        let u = f32::from_bits(((e + 127) as u32) << 23) * m as f32;

        data[i * 4..i * 4 + 4].copy_from_slice(&u.to_le_bytes());
    }
}

fn decode_filter_color_u8(data: &mut [u8], count: usize) {
    let max = 255f32;

    for i in 0..count {
        let a0 = i32::from(data[i * 4 + 3]);

        // recover scale from alpha high bit
        let as_ = {
            let mut as_ = a0;
            as_ |= as_ >> 1;
            as_ |= as_ >> 2;
            as_ |= as_ >> 4;
            as_
        };

        // convert to RGB in fixed point (co/cg are sign extended)
        let y = i32::from(data[i * 4]);
        let co = i32::from(data[i * 4 + 1] as i8);
        let cg = i32::from(data[i * 4 + 2] as i8);

        let r = y + co - cg;
        let g = y + cg;
        let b = y - co - cg;

        // expand alpha by one bit to match other components
        let a = ((a0 << 1) & as_) | (a0 & 1);

        // compute scaling factor
        let ss = max / as_ as f32;

        // rounded float->int
        let rf = cast_i32(r as f32 * ss + 0.5);
        let gf = cast_i32(g as f32 * ss + 0.5);
        let bf = cast_i32(b as f32 * ss + 0.5);
        let af = cast_i32(a as f32 * ss + 0.5);

        data[i * 4] = rf as u8;
        data[i * 4 + 1] = gf as u8;
        data[i * 4 + 2] = bf as u8;
        data[i * 4 + 3] = af as u8;
    }
}

fn decode_filter_color_u16(data: &mut [u8], count: usize) {
    let max = 65535f32;

    for i in 0..count {
        let read = |j: usize| u16::from_le_bytes([data[i * 8 + j * 2], data[i * 8 + j * 2 + 1]]);

        let a0 = i32::from(read(3));

        // recover scale from alpha high bit
        let as_ = {
            let mut as_ = a0;
            as_ |= as_ >> 1;
            as_ |= as_ >> 2;
            as_ |= as_ >> 4;
            as_ |= as_ >> 8; // noop for 8-bit
            as_
        };

        // convert to RGB in fixed point (co/cg are sign extended)
        let y = i32::from(read(0));
        let co = i32::from(read(1) as i16);
        let cg = i32::from(read(2) as i16);

        let r = y + co - cg;
        let g = y + cg;
        let b = y - co - cg;

        // expand alpha by one bit to match other components
        let a = ((a0 << 1) & as_) | (a0 & 1);

        let ss = max / as_ as f32;

        let rf = cast_i32(r as f32 * ss + 0.5);
        let gf = cast_i32(g as f32 * ss + 0.5);
        let bf = cast_i32(b as f32 * ss + 0.5);
        let af = cast_i32(a as f32 * ss + 0.5);

        for (j, v) in [rf, gf, bf, af].into_iter().enumerate() {
            data[i * 8 + j * 2..i * 8 + j * 2 + 2].copy_from_slice(&(v as u16).to_le_bytes());
        }
    }
}

/// Upstream: `meshopt_decodeFilterOct`. `stride` is 4 (8-bit components) or
/// 8 (16-bit components); `buffer` holds `count * stride` bytes.
pub fn decode_filter_oct(buffer: &mut [u8], count: usize, stride: usize) {
    debug_assert!(stride == 4 || stride == 8);

    if stride == 4 {
        decode_filter_oct_s8(buffer, count);
    } else {
        decode_filter_oct_s16(buffer, count);
    }
}

/// Upstream: `meshopt_decodeFilterQuat`. `stride` is 8; `buffer` holds
/// `count * stride` bytes.
pub fn decode_filter_quat(buffer: &mut [u8], count: usize, stride: usize) {
    debug_assert!(stride == 8);

    decode_filter_quat_s16(buffer, count);
}

/// Upstream: `meshopt_decodeFilterExp`. `stride` is a multiple of 4;
/// `buffer` holds `count * stride` bytes.
pub fn decode_filter_exp(buffer: &mut [u8], count: usize, stride: usize) {
    debug_assert!(stride > 0 && stride.is_multiple_of(4));

    decode_filter_exp_u32(buffer, count * (stride / 4));
}

/// Upstream: `meshopt_decodeFilterColor`. `stride` is 4 (8-bit components)
/// or 8 (16-bit components); `buffer` holds `count * stride` bytes.
pub fn decode_filter_color(buffer: &mut [u8], count: usize, stride: usize) {
    debug_assert!(stride == 4 || stride == 8);

    if stride == 4 {
        decode_filter_color_u8(buffer, count);
    } else {
        decode_filter_color_u16(buffer, count);
    }
}

/// Upstream: `meshopt_encodeFilterOct`. `stride` is 4 or 8; `destination`
/// holds `count * stride` bytes; `data` holds `count * 4` floats.
pub fn encode_filter_oct(
    destination: &mut [u8],
    count: usize,
    stride: usize,
    bits: i32,
    data: &[f32],
) {
    debug_assert!(stride == 4 || stride == 8);
    debug_assert!((2..=16).contains(&bits));
    debug_assert!(stride == 8 || bits <= 8);

    let byte_bits = (stride * 2) as i32;

    for i in 0..count {
        // octahedral encoding of a unit vector
        let nx = data[i * 4];
        let ny = data[i * 4 + 1];
        let nz = data[i * 4 + 2];
        let nw = data[i * 4 + 3];

        let nl = nx.abs() + ny.abs() + nz.abs();
        let ns = if nl == 0.0 { 0.0 } else { 1.0 / nl };

        let nx = nx * ns;
        let ny = ny * ns;

        let u = if nz >= 0.0 {
            nx
        } else {
            (1.0 - ny.abs()) * if nx >= 0.0 { 1.0 } else { -1.0 }
        };
        let v = if nz >= 0.0 {
            ny
        } else {
            (1.0 - nx.abs()) * if ny >= 0.0 { 1.0 } else { -1.0 }
        };

        let fu = quantize_snorm(u, bits);
        let fv = quantize_snorm(v, bits);
        let fo = quantize_snorm(1.0, bits);
        let fw = quantize_snorm(nw, byte_bits);

        if stride == 4 {
            destination[i * 4] = fu as i8 as u8;
            destination[i * 4 + 1] = fv as i8 as u8;
            destination[i * 4 + 2] = fo as i8 as u8;
            destination[i * 4 + 3] = fw as i8 as u8;
        } else {
            destination[i * 8..i * 8 + 2].copy_from_slice(&(fu as i16).to_le_bytes());
            destination[i * 8 + 2..i * 8 + 4].copy_from_slice(&(fv as i16).to_le_bytes());
            destination[i * 8 + 4..i * 8 + 6].copy_from_slice(&(fo as i16).to_le_bytes());
            destination[i * 8 + 6..i * 8 + 8].copy_from_slice(&(fw as i16).to_le_bytes());
        }
    }
}

/// Upstream: `meshopt_encodeFilterQuat`. `stride` is 8; `destination` holds
/// `count * stride` bytes; `data` holds `count * 4` floats.
pub fn encode_filter_quat(
    destination: &mut [u8],
    count: usize,
    stride: usize,
    bits: i32,
    data: &[f32],
) {
    debug_assert!(stride == 8);
    debug_assert!((4..=16).contains(&bits));

    let scaler = math::sqrt(2.0);

    for i in 0..count {
        let q = &data[i * 4..i * 4 + 4];

        // establish maximum quaternion component
        let mut qc = 0usize;
        if q[1].abs() > q[qc].abs() {
            qc = 1;
        }
        if q[2].abs() > q[qc].abs() {
            qc = 2;
        }
        if q[3].abs() > q[qc].abs() {
            qc = 3;
        }

        // we use double-cover properties to discard the sign
        let sign = if q[qc] < 0.0 { -1.0 } else { 1.0 };

        // note: we always encode a cyclical swizzle to be able to recover the order via rotation
        let d0 = quantize_snorm(q[(qc + 1) & 3] * scaler * sign, bits);
        let d1 = quantize_snorm(q[(qc + 2) & 3] * scaler * sign, bits);
        let d2 = quantize_snorm(q[(qc + 3) & 3] * scaler * sign, bits);
        let d3 = (quantize_snorm(1.0, bits) & !3) | qc as i32;

        let d = &mut destination[i * 8..i * 8 + 8];
        d[0..2].copy_from_slice(&(d0 as i16).to_le_bytes());
        d[2..4].copy_from_slice(&(d1 as i16).to_le_bytes());
        d[4..6].copy_from_slice(&(d2 as i16).to_le_bytes());
        d[6..8].copy_from_slice(&(d3 as i16).to_le_bytes());
    }
}

// optimized variant of frexp
fn optlog2(v: f32) -> i32 {
    // +1 accounts for implicit 1. in mantissa; zero and denormalized numbers will end up clamped to min_exp by calling code
    // this is important for shared exponent modes, as a zero component should not inflate the shared exponent
    (((v.to_bits() >> 23) & 0xff) as i32) - 127 + 1
}

// optimized variant of ldexp
fn optexp2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// Upstream: `meshopt_encodeFilterExp`. `stride` is a multiple of 4 and at
/// most 256; `destination` holds `count * stride` bytes; `data` holds
/// `count * stride / 4` floats.
pub fn encode_filter_exp(
    destination: &mut [u8],
    count: usize,
    stride: usize,
    bits: i32,
    data: &[f32],
    mode: EncodeExpMode,
) {
    debug_assert!(stride > 0 && stride.is_multiple_of(4) && stride <= 256);
    debug_assert!((1..=24).contains(&bits));

    let stride_float = stride / 4;

    let mut component_exp = [0i32; 64];

    const MIN_EXP: i32 = -100;

    // used to track maximum exponent for EncodeExpSharedComponent and zero exponent reuse for EncodeExpSeparate
    if mode != EncodeExpMode::Separate {
        component_exp[..stride_float].fill(MIN_EXP);
    }

    if mode == EncodeExpMode::SharedComponent {
        for i in 0..count {
            let v = &data[i * stride_float..i * stride_float + stride_float];

            // use maximum exponent to encode values; this guarantees that mantissa is [-1, 1]
            for (j, &value) in v.iter().enumerate() {
                let e = optlog2(value);

                if component_exp[j] < e {
                    component_exp[j] = e;
                }
            }
        }
    }

    for i in 0..count {
        let v = &data[i * stride_float..i * stride_float + stride_float];

        let mut vector_exp = MIN_EXP;

        match mode {
            EncodeExpMode::SharedVector => {
                // use maximum exponent to encode values; this guarantees that mantissa is [-1, 1]
                for &value in v {
                    let e = optlog2(value);

                    if vector_exp < e {
                        vector_exp = e;
                    }
                }
            }
            EncodeExpMode::Separate => {
                for (j, &value) in v.iter().enumerate() {
                    let e = optlog2(value);

                    // zero values inherit the last encoded exponent to keep the encoded data more compressible
                    if value != 0.0 {
                        component_exp[j] = MIN_EXP.max(e);
                    }
                }
            }
            EncodeExpMode::Clamped => {
                for (j, &value) in v.iter().enumerate() {
                    let e = optlog2(value);

                    component_exp[j] = 0.max(e);
                }
            }
            EncodeExpMode::SharedComponent => {}
        }

        for (j, &value) in v.iter().enumerate() {
            let mut exp = if mode == EncodeExpMode::SharedVector {
                vector_exp
            } else {
                component_exp[j]
            };

            // note that we additionally scale the mantissa to make it a K-bit signed integer (K-1 bits for magnitude)
            exp -= bits - 1;

            // compute renormalized rounded mantissa for each component
            const MMASK: i32 = (1 << 24) - 1;
            const MMAX: i32 = MMASK >> 1;

            let mut m = cast_i32(value * optexp2(-exp) + if value >= 0.0 { 0.5 } else { -0.5 });

            // clamp mantissa so that rounding away from zero doesn't overflow
            m = m.min(MMAX);

            let d = (m & MMASK) as u32 | ((exp as u32) << 24);
            destination[i * stride + j * 4..i * stride + j * 4 + 4]
                .copy_from_slice(&d.to_le_bytes());
        }
    }
}

/// Upstream: `meshopt_encodeFilterColor`. `stride` is 4 or 8; `destination`
/// holds `count * stride` bytes; `data` holds `count * 4` floats.
pub fn encode_filter_color(
    destination: &mut [u8],
    count: usize,
    stride: usize,
    bits: i32,
    data: &[f32],
) {
    debug_assert!(stride == 4 || stride == 8);
    debug_assert!((2..=16).contains(&bits));
    debug_assert!(stride == 8 || bits <= 8);

    for i in 0..count {
        let c = &data[i * 4..i * 4 + 4];

        let fr = quantize_unorm(c[0], bits);
        let fg = quantize_unorm(c[1], bits);
        let fb = quantize_unorm(c[2], bits);

        // YCoCg-R encoding with truncated Co/Cg ensures that decoding can be done using integers
        let fco = (fr - fb) / 2;
        let tmp = fb + fco;
        let fcg = (fg - tmp) / 2;
        let fy = tmp + fcg;

        // validate that R/G/B can be reconstructed with K bit integers
        debug_assert!((((fy + fco - fcg) | (fy + fcg) | (fy - fco - fcg)) as u32) < (1u32 << bits));

        // alpha: K-1-bit encoding with high bit set to 1
        let fa = (quantize_unorm(c[3], bits) >> 1) | (1 << (bits - 1));

        if stride == 4 {
            destination[i * 4] = fy as u8;
            destination[i * 4 + 1] = fco as u8;
            destination[i * 4 + 2] = fcg as u8;
            destination[i * 4 + 3] = fa as u8;
        } else {
            destination[i * 8..i * 8 + 2].copy_from_slice(&(fy as u16).to_le_bytes());
            destination[i * 8 + 2..i * 8 + 4].copy_from_slice(&(fco as u16).to_le_bytes());
            destination[i * 8 + 4..i * 8 + 6].copy_from_slice(&(fcg as u16).to_le_bytes());
            destination[i * 8 + 6..i * 8 + 8].copy_from_slice(&(fa as u16).to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_decodeFilterOct,
    // meshopt_decodeFilterQuat, meshopt_decodeFilterExp, meshopt_decodeFilterColor,
    // meshopt_encodeFilterOct, meshopt_encodeFilterQuat, meshopt_encodeFilterExp,
    // meshopt_encodeFilterColor.
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

        fn u32(&mut self) -> usize {
            let mut a = [0u8; 4];
            a.copy_from_slice(self.take(4));
            u32::from_le_bytes(a) as usize
        }

        fn f32(&mut self) -> f32 {
            f32::from_bits(self.u32() as u32)
        }

        fn f32_vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.f32()).collect()
        }
    }

    fn golden_decode(file: &[u8], elem_size: usize, apply: fn(&mut [u8], usize, usize)) {
        let mut c = Cursor::new(file);

        while c.pos < file.len() {
            let count = c.u32();
            let input = c.take(count * elem_size).to_vec();
            let expected = c.take(count * elem_size).to_vec();

            let mut buffer = input;
            apply(&mut buffer, count, elem_size);
            assert_eq!(buffer, expected, "count={count}");
        }
    }

    #[test]
    fn golden_decode_filter_oct8() {
        golden_decode(
            include_bytes!("../tests/data/vf_dec_oct8.bin"),
            4,
            decode_filter_oct,
        );
    }

    #[test]
    fn golden_decode_filter_oct16() {
        golden_decode(
            include_bytes!("../tests/data/vf_dec_oct16.bin"),
            8,
            decode_filter_oct,
        );
    }

    #[test]
    fn golden_decode_filter_quat() {
        golden_decode(
            include_bytes!("../tests/data/vf_dec_quat.bin"),
            8,
            decode_filter_quat,
        );
    }

    #[test]
    fn golden_decode_filter_exp() {
        golden_decode(
            include_bytes!("../tests/data/vf_dec_exp.bin"),
            4,
            decode_filter_exp,
        );
    }

    #[test]
    fn golden_decode_filter_color8() {
        golden_decode(
            include_bytes!("../tests/data/vf_dec_color8.bin"),
            4,
            decode_filter_color,
        );
    }

    #[test]
    fn golden_decode_filter_color16() {
        golden_decode(
            include_bytes!("../tests/data/vf_dec_color16.bin"),
            8,
            decode_filter_color,
        );
    }

    #[test]
    fn golden_encode_filter_oct() {
        let file = include_bytes!("../tests/data/vf_enc_oct.bin");
        let mut c = Cursor::new(file);

        while c.pos < file.len() {
            let stride = c.u32();
            let bits = c.u32() as i32;
            let count = c.u32();
            let data = c.f32_vec(count * 4);
            let expected = c.take(count * stride).to_vec();

            let mut dest = vec![0u8; count * stride];
            encode_filter_oct(&mut dest, count, stride, bits, &data);
            assert_eq!(dest, expected, "stride={stride} bits={bits}");
        }
    }

    #[test]
    fn golden_encode_filter_quat() {
        let file = include_bytes!("../tests/data/vf_enc_quat.bin");
        let mut c = Cursor::new(file);

        while c.pos < file.len() {
            let bits = c.u32() as i32;
            let count = c.u32();
            let data = c.f32_vec(count * 4);
            let expected = c.take(count * 8).to_vec();

            let mut dest = vec![0u8; count * 8];
            encode_filter_quat(&mut dest, count, 8, bits, &data);
            assert_eq!(dest, expected, "bits={bits}");
        }
    }

    #[test]
    fn golden_encode_filter_exp() {
        let file = include_bytes!("../tests/data/vf_enc_exp.bin");
        let mut c = Cursor::new(file);

        while c.pos < file.len() {
            let stride_float = c.u32();
            let bits = c.u32() as i32;
            let mode = c.u32() as i32;
            let count = c.u32();
            let data = c.f32_vec(count * stride_float);
            let expected = c.take(count * stride_float * 4).to_vec();

            let mode = match mode {
                0 => EncodeExpMode::Separate,
                1 => EncodeExpMode::SharedVector,
                2 => EncodeExpMode::SharedComponent,
                _ => EncodeExpMode::Clamped,
            };

            let mut dest = vec![0u8; count * stride_float * 4];
            encode_filter_exp(&mut dest, count, stride_float * 4, bits, &data, mode);
            assert_eq!(
                dest, expected,
                "sf={stride_float} bits={bits} mode={mode:?}"
            );
        }
    }

    #[test]
    fn golden_encode_filter_color() {
        let file = include_bytes!("../tests/data/vf_enc_color.bin");
        let mut c = Cursor::new(file);

        while c.pos < file.len() {
            let stride = c.u32();
            let bits = c.u32() as i32;
            let count = c.u32();
            let data = c.f32_vec(count * 4);
            let expected = c.take(count * stride).to_vec();

            let mut dest = vec![0u8; count * stride];
            encode_filter_color(&mut dest, count, stride, bits, &data);
            assert_eq!(dest, expected, "stride={stride} bits={bits}");
        }
    }
}
