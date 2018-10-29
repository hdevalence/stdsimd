//! AVX-512F, AVX-512F+VL

use coresimd::simd::*;
use coresimd::simd_llvm::*;
use coresimd::x86::*;
use mem;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Set packed 32-bit integers in returned vector with the supplied values in
/// reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_setr_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_setr_epi32(
    e0: i32, e1: i32, e2: i32, e3: i32, e4: i32, e5: i32, e6: i32, e7: i32,
    e8: i32, e9: i32, e10: i32, e11: i32, e12: i32, e13: i32, e14: i32, e15: i32,
) -> __m512i {
    mem::transmute(i32x16::new(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0))
}

/// Computes the absolute values of packed 32-bit integers in `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_abs_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsd))]
pub unsafe fn _mm512_abs_epi32(a: __m512i) -> __m512i {
    mem::transmute(pabsd512(a.as_i32x16()))
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.pabs.d.512"]
    fn pabsd512(a: i32x16) -> u32x16;
}

#[cfg(test)]
mod tests {
    use std;
    use stdsimd_test::simd_test;

    use coresimd::x86::*;

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_abs_epi32() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm512_setr_epi32(
            0, 1, -1, std::i32::MAX,
            std::i32::MIN, 100, -100, -32,
            0, 1, -1, std::i32::MAX,
            std::i32::MIN, 100, -100, -32,
        );
        let r = _mm512_abs_epi32(a);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm512_setr_epi32(
            0, 1, 1, std::i32::MAX,
            std::i32::MAX.wrapping_add(1), 100, 100, 32,
            0, 1, 1, std::i32::MAX,
            std::i32::MAX.wrapping_add(1), 100, 100, 32,
        );
        assert_eq_m512i(r, e);
    }
}
