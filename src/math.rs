//! Internal numeric wrappers over the math backend.
//!
//! Every transcendental and rounding operation used by the port goes through this
//! module (ADR-0004); call sites never name
//! `libm` or `std` directly. With the default `std` feature the wrappers call the
//! standard library; without it they call `libm`, keeping the crate `no_std`.
//!
//! `f32::abs`, `min`/`max`, and plain arithmetic are `core` and are used directly —
//! no wrappers for those.

#![allow(dead_code)] // wrappers land ahead of their consumers (design.md §9: P1.0 before P1.1+)

#[inline(always)]
pub(crate) fn sqrt(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        f32::sqrt(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrtf(x)
    }
}

#[inline(always)]
pub(crate) fn sqrt_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::sqrt(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(x)
    }
}

#[inline(always)]
pub(crate) fn floor(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        f32::floor(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::floorf(x)
    }
}

#[inline(always)]
pub(crate) fn floor_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::floor(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::floor(x)
    }
}

#[inline(always)]
pub(crate) fn ceil(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        f32::ceil(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::ceilf(x)
    }
}

#[inline(always)]
pub(crate) fn ceil_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::ceil(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::ceil(x)
    }
}

/// Not correctly rounded; differential tests on paths through this function use a
/// documented ≤1 ULP tolerance (TDD §7).
#[inline(always)]
pub(crate) fn acos_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::acos(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::acos(x)
    }
}

/// Not correctly rounded; see `acos_f64`.
#[inline(always)]
pub(crate) fn log2(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        f32::log2(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::log2f(x)
    }
}

/// Not correctly rounded; see `acos_f64`.
#[inline(always)]
pub(crate) fn sin_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::sin(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sin(x)
    }
}

/// Not correctly rounded; see `acos_f64`.
#[inline(always)]
pub(crate) fn cos(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        f32::cos(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::cosf(x)
    }
}

/// Exact function (bit-level result, identical on every platform); `libm` in both
/// configurations because std's `frexp` is unstable.
#[inline(always)]
pub(crate) fn frexp(x: f32) -> (f32, i32) {
    libm::frexpf(x)
}

/// Exact function (see `frexp`).
#[inline(always)]
pub(crate) fn ldexp(x: f32, n: i32) -> f32 {
    libm::ldexpf(x, n)
}

/// Not correctly rounded; see `acos_f64`.
#[inline(always)]
pub(crate) fn cos_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::cos(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::cos(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_values() {
        assert_eq!(sqrt(4.0), 2.0);
        assert_eq!(sqrt_f64(4.0), 2.0);
        assert_eq!(floor(1.5), 1.0);
        assert_eq!(floor_f64(1.5), 1.0);
        assert_eq!(ceil(1.5), 2.0);
        assert_eq!(ceil_f64(1.5), 2.0);
        assert_eq!(acos_f64(1.0), 0.0);
        assert_eq!(log2(8.0), 3.0);
        assert_eq!(sin_f64(0.0), 0.0);
        assert_eq!(cos(0.0), 1.0);
        assert_eq!(cos_f64(0.0), 1.0);
    }

    #[test]
    fn exact_functions() {
        assert_eq!(frexp(8.0), (0.5, 4));
        assert_eq!(frexp(0.0), (0.0, 0));
        assert_eq!(frexp(-0.75), (-0.75, 0));
        assert_eq!(ldexp(0.75, 2), 3.0);
        assert_eq!(ldexp(1.0, -126).to_bits(), 0x0080_0000);
    }
}
