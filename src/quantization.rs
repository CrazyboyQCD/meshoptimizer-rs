//! Floating-point quantization helpers.
//!
//! Upstream: `quantization.cpp` (`meshopt_quantizeHalf`, `meshopt_quantizeFloat`,
//! `meshopt_dequantizeHalf`, `meshopt_computePositionExponent`).

use crate::math;

/// Upstream: `meshopt_quantizeHalf`.
pub fn quantize_half(v: f32) -> u16 {
    let ui = v.to_bits();

    let s = (ui >> 16) as u16 & 0x8000;
    let em = (ui & 0x7fff_ffff) as i32;

    // bias exponent and round to nearest; 112 is relative exponent bias (127-15)
    let mut h = (em - (112 << 23) + (1 << 12)) >> 13;
    // underflow: flush to zero; 113 encodes exponent -14
    h = if em < (113 << 23) { 0 } else { h };
    // overflow: infinity; 143 encodes exponent 16
    h = if em >= (143 << 23) { 0x7c00 } else { h };
    // NaN: convert all types of NaN to qNaN
    h = if em > (255 << 23) { 0x7e00 } else { h };

    s | h as u16
}

/// Upstream: `meshopt_quantizeFloat`. `n` is the number of mantissa bits to keep
/// (`0..=23`); values outside the contract are `debug_assert`ed like upstream's
/// `assert`.
pub fn quantize_float(v: f32, n: i32) -> f32 {
    debug_assert!((0..=23).contains(&n));

    let ui = v.to_bits();

    let mask = (1u32 << (23 - n)) - 1;
    let round = (1u32 << (23 - n)) >> 1;

    let e = ui & 0x7f80_0000;
    // round all numbers except inf/nan; this is important to make sure nan doesn't overflow into -0
    let rui = ui.wrapping_add(round) & !mask;
    let ui = if e == 0x7f80_0000 { ui } else { rui };
    // flush denormals to zero
    let ui = if e == 0 { 0 } else { ui };

    f32::from_bits(ui)
}

/// Upstream: `meshopt_dequantizeHalf`.
pub fn dequantize_half(h: u16) -> f32 {
    let s = u32::from(h & 0x8000) << 16;
    let em = i32::from(h & 0x7fff);

    // bias exponent and pad mantissa with 0; 112 is relative exponent bias (127-15)
    let mut r = (em + (112 << 10)) << 13;
    // denormal: flush to zero
    r = if em < (1 << 10) { 0 } else { r };
    // infinity/NaN; applying the 112 exponent bias fixup twice converts 31 to 255
    r += if em >= (31 << 10) { 112 << 23 } else { 0 };

    f32::from_bits(s | r as u32)
}

// C float->int casts truncate toward zero and produce INT_MIN for NaN,
// +-inf and out-of-range values on x86 (cvttss2si); Rust saturates instead,
// so hostile filter inputs would diverge without this wrapper. 2^31 itself
// is not representable as i32 and casts to INT_MIN on x86 as well.
pub(crate) fn cast_i32(v: f32) -> i32 {
    if v.is_nan() || v >= 2147483648.0 || v < -2147483648.0 {
        i32::MIN
    } else {
        v as i32
    }
}

// upstream header inlines (meshoptimizer.h): meshopt_quantizeSnorm
pub(crate) fn quantize_snorm(v: f32, n: i32) -> i32 {
    let scale = ((1i32 << (n - 1)) - 1) as f32;

    let round = if v >= 0.0 { 0.5 } else { -0.5 };

    let v = if v >= -1.0 { v } else { -1.0 };
    let v = if v <= 1.0 { v } else { 1.0 };

    // the clamp above is total (NaN fails `>= -1.0` and clamps to -1.0), so the
    // saturating `as` is bit-identical to the cvttss2si wrapper for every input
    (v * scale + round) as i32
}

/// Upstream: `meshopt_computePositionExponent`. `minv`/`maxv` hold at least three
/// components.
pub fn compute_position_exponent(minv: &[f32], maxv: &[f32], min_exp: i32, max_bits: i32) -> i32 {
    debug_assert!(min_exp >= -126);
    debug_assert!((2..=24).contains(&max_bits));
    debug_assert!(minv.len() >= 3 && maxv.len() >= 3);

    let mut exp = min_exp;

    // compute max absolute component to ensure that individual endpoints fit on a 24-bit signed grid
    let mut maxc = 0f32;

    for k in 0..3 {
        let a = minv[k].abs();
        let b = maxv[k].abs();
        if maxc < a {
            maxc = a;
        }
        if maxc < b {
            maxc = b;
        }
    }

    let (maxc_fr, maxc_exp) = math::frexp(maxc);

    // maxc is representable as 2^(maxc_exp-24) * 24-bit *unsigned* integer
    // we have to use a 24-bit *signed* grid, so we have to chop off the last bit
    // however, rounding in the corner case (mantissa is 1.111...) may increase the exponent by 1 so we need to offset it
    let maxc_off = 23 - i32::from(maxc_fr >= 1.0 - f32::EPSILON / 2.0);

    exp = if exp < maxc_exp - maxc_off {
        maxc_exp - maxc_off
    } else {
        exp
    };

    // compute effective range with conservative rounding, to allow for some ambiguity in the caller's rounding direction
    let scale = math::ldexp(1.0, -exp);
    let mut range = 0f32;

    for k in 0..3 {
        let a = math::floor(minv[k] * scale);
        let v = math::ceil(maxv[k] * scale);
        if range < v - a {
            range = v - a;
        }
    }

    // range must be representable as max_bits unsigned integer
    let (range_fr, range_exp) = math::frexp(range);

    exp += if range_exp > max_bits {
        range_exp - max_bits
    } else {
        0
    };

    // correct range if we are at the rounding boundary that would push us to overflow max_bits
    exp += i32::from(range_exp > max_bits && range_fr >= 1.0 - 1.0 / ((1 << max_bits) - 1) as f32);

    exp
}

#[cfg(test)]
mod tests {
    // Upstream entry points covered: meshopt_quantizeHalf,
    // meshopt_quantizeFloat, meshopt_dequantizeHalf,
    // meshopt_computePositionExponent.
    use super::*;

    fn le_u16(b: &[u8]) -> u16 {
        let mut a = [0u8; 2];
        a.copy_from_slice(b);
        u16::from_le_bytes(a)
    }

    fn le_u32(b: &[u8]) -> u32 {
        let mut a = [0u8; 4];
        a.copy_from_slice(b);
        u32::from_le_bytes(a)
    }

    fn le_i32(b: &[u8]) -> i32 {
        let mut a = [0u8; 4];
        a.copy_from_slice(b);
        i32::from_le_bytes(a)
    }

    fn le_f32(b: &[u8]) -> f32 {
        f32::from_bits(le_u32(b))
    }

    #[test]
    fn golden_quantize_half() {
        let data = include_bytes!("../tests/data/quantization_half.bin");
        for r in data.as_chunks::<6>().0 {
            let input = f32::from_bits(le_u32(&r[0..4]));
            let expected = le_u16(&r[4..6]);
            assert_eq!(
                quantize_half(input),
                expected,
                "quantize_half(0x{:08x})",
                input.to_bits()
            );
        }
    }

    #[test]
    fn golden_dequantize_half() {
        let data = include_bytes!("../tests/data/quantization_dehalf.bin");
        for r in data.as_chunks::<6>().0 {
            let input = le_u16(&r[0..2]);
            let expected = f32::from_bits(le_u32(&r[2..6]));
            assert_eq!(
                dequantize_half(input).to_bits(),
                expected.to_bits(),
                "dequantize_half(0x{:04x})",
                input
            );
        }
    }

    #[test]
    fn golden_quantize_float() {
        let data = include_bytes!("../tests/data/quantization_float.bin");
        for r in data.as_chunks::<10>().0 {
            let input = f32::from_bits(le_u32(&r[0..4]));
            let n = i32::from(le_u16(&r[4..6]));
            let expected = f32::from_bits(le_u32(&r[6..10]));
            assert_eq!(
                quantize_float(input, n).to_bits(),
                expected.to_bits(),
                "quantize_float(0x{:08x}, {})",
                input.to_bits(),
                n
            );
        }
    }

    #[test]
    fn golden_compute_position_exponent() {
        let data = include_bytes!("../tests/data/quantization_posexp.bin");
        for r in data.as_chunks::<36>().0 {
            let mut minv = [0f32; 3];
            let mut maxv = [0f32; 3];
            for k in 0..3 {
                minv[k] = le_f32(&r[k * 4..k * 4 + 4]);
                maxv[k] = le_f32(&r[12 + k * 4..16 + k * 4]);
            }
            let min_exp = le_i32(&r[24..28]);
            let max_bits = le_i32(&r[28..32]);
            let expected = le_i32(&r[32..36]);
            assert_eq!(
                compute_position_exponent(&minv, &maxv, min_exp, max_bits),
                expected,
                "case {minv:?} {maxv:?} {min_exp} {max_bits}"
            );
        }
    }
}
