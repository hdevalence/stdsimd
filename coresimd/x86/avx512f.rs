//! AVX-512F, AVX-512F+VL

use coresimd::simd::*;
use coresimd::simd_llvm::*;
use coresimd::x86::*;
use mem;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Broadcast 8-bit integer `a` to all elements of returned vector.
/// This intrinsic may generate the `vpbroadcastb`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set1_epi8)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_set1_epi8(a: i8) -> __m512i {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    _mm512_set_epi8(
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
    )
}

/// Broadcast 16-bit integer `a` to all all elements of returned vector.
/// This intrinsic may generate the `vpbroadcastw`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set1_epi16)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_set1_epi16(a: i16) -> __m512i {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    _mm512_set_epi16(
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,
    )
}

/// Broadcast 32-bit integer `a` to all elements of returned vector.
/// This intrinsic may generate the `vpbroadcastd`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set1_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_set1_epi32(a: i32) -> __m512i {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    _mm512_set_epi32(
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
    )
}

/// Broadcast 64-bit integer `a` to all elements of returned vector.
/// This intrinsic may generate the `vpbroadcastq`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set1_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_set1_epi64(a: i64) -> __m512i {
    _mm512_set_epi64(a, a, a, a, a, a, a, a)
}

/// Return vector of type __m512i with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_setzero_si512)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpxorq))]
pub unsafe fn _mm512_setzero_si512() -> __m512i {
    _mm512_set1_epi8(0)
}

/// Set packed 8-bit integers in returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set_epi8)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_set_epi8(
    e63: i8, e62: i8, e61: i8, e60: i8, e59: i8, e58: i8, e57: i8, e56: i8,
    e55: i8, e54: i8, e53: i8, e52: i8, e51: i8, e50: i8, e49: i8, e48: i8,
    e47: i8, e46: i8, e45: i8, e44: i8, e43: i8, e42: i8, e41: i8, e40: i8,
    e39: i8, e38: i8, e37: i8, e36: i8, e35: i8, e34: i8, e33: i8, e32: i8,
    e31: i8, e30: i8, e29: i8, e28: i8, e27: i8, e26: i8, e25: i8, e24: i8,
    e23: i8, e22: i8, e21: i8, e20: i8, e19: i8, e18: i8, e17: i8, e16: i8,
    e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e09: i8, e08: i8,
    e07: i8, e06: i8, e05: i8, e04: i8, e03: i8, e02: i8, e01: i8, e00: i8,
) -> __m512i {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    mem::transmute(i8x64::new(
        e00, e01, e02, e03, e04, e05, e06, e07,
        e08, e09, e10, e11, e12, e13, e14, e15,
        e16, e17, e18, e19, e20, e21, e22, e23,
        e24, e25, e26, e27, e28, e29, e30, e31,
        e32, e33, e34, e35, e36, e37, e38, e39,
        e40, e41, e42, e43, e44, e45, e46, e47,
        e48, e49, e50, e51, e52, e53, e54, e55,
        e56, e57, e58, e59, e60, e61, e62, e63,
    ))
}

/// Set packed 16-bit integers in returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set_epi16)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_set_epi16(
    e31: i16, e30: i16, e29: i16, e28: i16, e27: i16, e26: i16, e25: i16, e24: i16,
    e23: i16, e22: i16, e21: i16, e20: i16, e19: i16, e18: i16, e17: i16, e16: i16,
    e15: i16, e14: i16, e13: i16, e12: i16, e11: i16, e10: i16, e09: i16, e08: i16,
    e07: i16, e06: i16, e05: i16, e04: i16, e03: i16, e02: i16, e01: i16, e00: i16,
) -> __m512i {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    mem::transmute(i16x32::new(
        e00, e01, e02, e03, e04, e05, e06, e07,
        e08, e09, e10, e11, e12, e13, e14, e15,
        e16, e17, e18, e19, e20, e21, e22, e23,
        e24, e25, e26, e27, e28, e29, e30, e31,
    ))
}

/// Set packed 32-bit integers in returned vector with the supplied values in
/// reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_setr_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_setr_epi32(
    e15: i32, e14: i32, e13: i32, e12: i32, e11: i32, e10: i32, e09: i32, e08: i32,
    e07: i32, e06: i32, e05: i32, e04: i32, e03: i32, e02: i32, e01: i32, e00: i32,
) -> __m512i {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    mem::transmute(i32x16::new(
        e15, e14, e13, e12, e11, e10, e09, e08,
        e07, e06, e05, e04, e03, e02, e01, e00,
    ))
}

/// Set packed 32-bit integers in returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_set_epi32(
    e15: i32, e14: i32, e13: i32, e12: i32, e11: i32, e10: i32, e09: i32, e08: i32,
    e07: i32, e06: i32, e05: i32, e04: i32, e03: i32, e02: i32, e01: i32, e00: i32,
) -> __m512i {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    mem::transmute(i32x16::new(
        e00, e01, e02, e03, e04, e05, e06, e07,
        e08, e09, e10, e11, e12, e13, e14, e15,
    ))
}

/// Set packed 64-bit integers in returned vector with the supplied values in
/// reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_setr_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_setr_epi64(
    e7: i64, e6: i64, e5: i64, e4: i64, e3: i64, e2: i64, e1: i64, e0: i64,
) -> __m512i {
    mem::transmute(i64x8::new(e7, e6, e5, e4, e3, e2, e1, e0))
}

/// Set packed 64-bit integers in returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_set_epi64(
    e7: i64, e6: i64, e5: i64, e4: i64, e3: i64, e2: i64, e1: i64, e0: i64,
) -> __m512i {
    mem::transmute(i64x8::new(e0, e1, e2, e3, e4, e5, e6, e7))
}

/// Add packed 64-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm512_add_epi64(a: __m512i, b: __m512i) -> __m512i {
    mem::transmute(simd_add(a.as_i64x8(), b.as_i64x8()))
}

/// Add packed 64-bit integers in `a` and `b`, and return the results
/// using writemask `k` (elements are copied from `src` when the
/// corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_add_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm512_mask_add_epi64(
    src: __m512i, k: __mmask8, a: __m512i, b: __m512i,
) -> __m512i {
    simd_select(k.0, _mm512_add_epi64(a, b), src)
}

/// Add packed 64-bit integers in a and b, and return the results using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not
/// set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_add_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm512_maskz_add_epi64(
    k: __mmask8, a: __m512i, b: __m512i,
) -> __m512i {
    simd_select(k.0, _mm512_add_epi64(a, b), _mm512_setzero_si512())
}

/// Add packed 32-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_add_epi32(a: __m512i, b: __m512i) -> __m512i {
    mem::transmute(simd_add(a.as_i32x16(), b.as_i32x16()))
}

/// Add packed 16-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_epi16)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub unsafe fn _mm512_add_epi16(a: __m512i, b: __m512i) -> __m512i {
    mem::transmute(simd_add(a.as_i16x32(), b.as_i16x32()))
}

/// Add packed 8-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_epi8)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub unsafe fn _mm512_add_epi8(a: __m512i, b: __m512i) -> __m512i {
    mem::transmute(simd_add(a.as_i8x64(), b.as_i8x64()))
}

#[cfg(test)]
mod tests {
    use std;
    use stdsimd_test::simd_test;

    use coresimd::x86::*;

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_add_epi64() {
        let a = _mm512_set_epi64(
            -10,
            0,
            100,
            1_000_000_000,
            -10,
            0,
            100,
            1_000_000_000,
        );
        let b = _mm512_set_epi64(-1, 0, 1, 2, -1, 0, 1, 2);
        let r = _mm512_add_epi64(a, b);
        let e = _mm512_set_epi64(
            -11,
            0,
            101,
            1_000_000_002,
            -11,
            0,
            101,
            1_000_000_002,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_add_epi32() {
        let a =
            _mm512_set_epi32(-1, 0, 1, 2, 3, 4, 5, 6, -1, 0, 1, 2, 3, 4, 5, 6);
        let b =
            _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_add_epi32(a, b);
        let e = _mm512_set_epi32(
            0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_add_epi16() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        let r = _mm512_add_epi16(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm512_set_epi16(
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30,
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_add_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm512_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm512_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = _mm512_add_epi8(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm512_set_epi8(
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30,
            32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62,
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30,
            32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62,
        );
        assert_eq_m512i(r, e);
    }

}
