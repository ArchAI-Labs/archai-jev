//! The arithmetic of the converter (spec 018, section 3).
//!
//! Everything here is **deterministic**: the same inputs give the same bits on every OS and CPU.
//! Only exact IEEE operations are used (`+`, `*`, fused multiply-add, rounding), in a fixed order.

/// The `f32` a bfloat16 bit pattern stands for (exact: bfloat16 is the top half of an `f32`).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Round an `f32` to bfloat16, to nearest and ties to even (what `torch` and `gguf-py` do).
///
/// NaN stays a NaN (the converter refuses non-finite values before it gets here); a finite value
/// too large for bfloat16 rounds to infinity, like in `torch`.
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7fff + lsb) >> 16) as u16
}

/// `-exp(x)` as the converter needs it for `A_log`. Computed in `f64` and rounded once to `f32`
/// (correctly rounded in all but astronomically rare cases, whatever the C library).
pub fn neg_exp(x: f32) -> f32 {
    (-f64::from(x).exp()) as f32
}

/// One output row of a LoRA merge: `out[j] = w[j] + scale * (sum over k of b[k] * a[k][j])`.
///
/// `a` holds `b.len()` rows of `w.len()` values. The sum over `k` is **sequential** and uses a
/// fused multiply-add per term, starting from zero, with no rounding in between; then one product
/// by `scale` and one sum with `w`, each rounded to `f32`. `delta` is scratch space of `w.len()`.
/// This is the order a plain `W + (alpha/r) * (B @ A)` in `f32` has, to be checked against the
/// reference (spec 018, AC-12).
pub fn merge_row(w: &[f32], b: &[f32], a: &[f32], scale: f32, delta: &mut [f32], out: &mut [f32]) {
    let n = w.len();
    delta.iter_mut().for_each(|d| *d = 0.0);
    if n > 0 {
        for (bk, a_row) in b.iter().zip(a.chunks_exact(n)) {
            for (d, av) in delta.iter_mut().zip(a_row) {
                *d = bk.mul_add(*av, *d);
            }
        }
    }
    for ((o, wv), d) in out.iter_mut().zip(w).zip(delta.iter()) {
        *o = *wv + scale * *d;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(f32 bits, bfloat16 bits)` computed with PyTorch (`tensor.to(torch.bfloat16)`): random
    /// values, exact ties of both parities and negative sign, and subnormals.
    #[rustfmt::skip]
    const BF16_TABLE: &[(u32, u16)] = &[
        (0xBD653997, 0xBD65), (0x32EC94EC, 0x32ED), (0xBA44736E, 0xBA44), (0xF96D6D77, 0xF96D),
        (0x6D6F5BE1, 0x6D6F), (0x80C66B9C, 0x80C6), (0x8ADBBBB2, 0x8ADC), (0xE8B14A25, 0xE8B1),
        (0x2F63F904, 0x2F64), (0x5474FC72, 0x5475), (0xAB089551, 0xAB09), (0x6DD293D2, 0x6DD3),
        (0x929AA63A, 0x929B), (0x45093E44, 0x4509), (0x8B53A136, 0x8B54), (0xE330482B, 0xE330),
        (0x72CC038A, 0x72CC), (0x3DB10C41, 0x3DB1), (0xDEC7EF4D, 0xDEC8), (0x18A3ACFF, 0x18A4),
        (0xBCFB5C3D, 0xBCFB), (0x2BADCB75, 0x2BAE), (0x35F1E448, 0x35F2), (0x11D4E337, 0x11D5),
        (0x7287B226, 0x7288), (0x9EE90905, 0x9EE9), (0xF071D78C, 0xF072), (0xFE3EC005, 0xFE3F),
        (0x874A2414, 0x874A), (0x2A59E2EB, 0x2A5A), (0x853D60EB, 0x853D), (0xD5548A5A, 0xD555),
        (0xE592B68B, 0xE593), (0xF7EC205E, 0xF7EC), (0xF9DA33D9, 0xF9DA), (0x6CBB6135, 0x6CBB),
        (0x6315828F, 0x6316), (0x24B82375, 0x24B8), (0xA5BEA253, 0xA5BF), (0x09D1FDE2, 0x09D2),
        (0x057032D0, 0x0570), (0x9522EC6E, 0x9523), (0x8D345080, 0x8D34), (0x8B3490E8, 0x8B35),
        (0x5E2D85F6, 0x5E2E), (0xCC7285E9, 0xCC73), (0xB09EA713, 0xB09F), (0xDF37651E, 0xDF37),
        (0x02F12956, 0x02F1), (0xC00A7581, 0xC00A), (0xEEF132B3, 0xEEF1), (0xB04F59EE, 0xB04F),
        (0xE8E96A55, 0xE8E9), (0x876C1365, 0x876C), (0x3D9E2A45, 0x3D9E), (0x810DE778, 0x810E),
        (0x4407EB76, 0x4408), (0x85A59495, 0x85A6), (0x522B7E3F, 0x522B), (0xEA3F9EFE, 0xEA40),
        (0xDEACCD24, 0xDEAD), (0x8F7AB9DD, 0x8F7B), (0x5CC22E56, 0x5CC2), (0xEEAD6BA4, 0xEEAD),
        (0xE5508BB3, 0xE551), (0xD519293C, 0xD519), (0xE9CFAA70, 0xE9D0), (0x5F4996E7, 0x5F4A),
        (0xF1575050, 0xF157), (0xA18E422F, 0xA18E), (0xD78C809F, 0xD78D), (0x168DE46A, 0x168E),
        (0xEEE0C5E8, 0xEEE1), (0x4825E7B7, 0x4826), (0x4D1B4334, 0x4D1B), (0x1777BA0A, 0x1778),
        (0xD5A25D02, 0xD5A2), (0xFEB000B5, 0xFEB0), (0xA055623B, 0xA055), (0xF92E6C19, 0xF92E),
        (0xE0EC6736, 0xE0EC), (0xB48BA0A0, 0xB48C), (0x1AE0DC76, 0x1AE1), (0x7BB6D58F, 0x7BB7),
        (0x848ED9DD, 0x848F), (0xF43E8683, 0xF43F), (0x07E96D55, 0x07E9), (0x81427733, 0x8142),
        (0x5BA30896, 0x5BA3), (0x4A75D6DD, 0x4A76), (0xBBD141E2, 0xBBD1), (0x4D5173A3, 0x4D51),
        (0xD753A9AC, 0xD754), (0x66430362, 0x6643), (0x6C3CBA53, 0x6C3D), (0x897BCE78, 0x897C),
        (0x1AA7B735, 0x1AA8), (0x25795896, 0x2579), (0x5BE08738, 0x5BE1), (0xC5AA87E6, 0xC5AB),
        (0xCED07645, 0xCED0), (0xF0EF0CD9, 0xF0EF), (0xA79D6636, 0xA79D), (0x3CDCD52C, 0x3CDD),
        (0x262B787A, 0x262B), (0xDC97BEFD, 0xDC98), (0xC24520E0, 0xC245), (0xBDEB2AAA, 0xBDEB),
        (0xACBA3CEA, 0xACBA), (0xC8600C9F, 0xC860), (0x3A1CA090, 0x3A1D), (0x91390B73, 0x9139),
        (0xA767936B, 0xA768), (0xD60B087C, 0xD60B), (0x8B59A7E2, 0x8B5A), (0x9B0AE653, 0x9B0B),
        (0xE871E81B, 0xE872), (0x682D8955, 0x682E), (0x01F5546F, 0x01F5), (0x0AF772B8, 0x0AF7),
        (0x00094820, 0x0009), (0x7D7F85D0, 0x7D80), (0x28970924, 0x2897), (0x08D5CEC2, 0x08D6),
        (0xD349578A, 0xD349), (0xDFEC26B2, 0xDFEC), (0xF9A6E1AB, 0xF9A7), (0xFE5D4536, 0xFE5D),
        (0xFAEC736C, 0xFAEC), (0xE83C6336, 0xE83C), (0xD802D9B2, 0xD803), (0x47B09E0F, 0x47B1),
        (0xD76E9B0C, 0xD76F), (0x17A2BB33, 0x17A3), (0x95C49A8C, 0x95C5), (0xD537D7C5, 0xD538),
        (0xDDACD804, 0xDDAD), (0x1B2C6D86, 0x1B2C), (0x05AF801C, 0x05B0), (0x9D2D8E42, 0x9D2E),
        (0x56E56045, 0x56E5), (0xEFC810D0, 0xEFC8), (0xF4E33B79, 0xF4E3), (0xDBC9A7C2, 0xDBCA),
        (0xA0893673, 0xA089), (0x25EB3E55, 0x25EB), (0xA11C2FB4, 0xA11C), (0x5D8108BB, 0x5D81),
        (0x67DCC796, 0x67DD), (0x813D418A, 0x813D), (0x9CBD2EE8, 0x9CBD), (0xBE2FA713, 0xBE30),
        (0xAF370C06, 0xAF37), (0xED3B0002, 0xED3B), (0x83C344D1, 0x83C3), (0x0B5A0A0F, 0x0B5A),
        (0xCE62BB66, 0xCE63), (0x8BC0EE97, 0x8BC1), (0x01398000, 0x013A), (0x5ACE8000, 0x5ACE),
        (0x9E728000, 0x9E72), (0xF1548000, 0xF154), (0x97518000, 0x9752), (0x24198000, 0x241A),
        (0x69AE8000, 0x69AE), (0x189F8000, 0x18A0), (0xFDD48000, 0xFDD4), (0x48898000, 0x488A),
        (0x0B588000, 0x0B58), (0xA0B98000, 0xA0BA), (0xA17F8000, 0xA180), (0xF2EF8000, 0xF2F0),
        (0x7BC98000, 0x7BCA), (0xABFC8000, 0xABFC), (0xB76E8000, 0xB76E), (0x066A8000, 0x066A),
        (0x99018000, 0x9902), (0x213F8000, 0x2140), (0x49A68000, 0x49A6), (0xB7D28000, 0xB7D2),
        (0x7EE58000, 0x7EE6), (0x93748000, 0x9374), (0x24DC8000, 0x24DC), (0xD1E78000, 0xD1E8),
        (0x46298000, 0x462A), (0x3B4F8000, 0x3B50), (0x47DC8000, 0x47DC), (0xD39E8000, 0xD39E),
        (0x61AC8000, 0x61AC), (0xAC818000, 0xAC82), (0xB8D48000, 0xB8D4), (0x1F508000, 0x1F50),
        (0x40C78000, 0x40C8), (0x31118000, 0x3112), (0x63DC8000, 0x63DC), (0xF08D8000, 0xF08E),
        (0xA8A28000, 0xA8A2), (0x98068000, 0x9806), (0x2B058000, 0x2B06), (0xF5A28000, 0xF5A2),
        (0x67AF8000, 0x67B0), (0x2AC58000, 0x2AC6), (0xF76E8000, 0xF76E), (0xC69E8000, 0xC69E),
        (0x6B0E8000, 0x6B0E), (0x18C98000, 0x18CA), (0x00568D2F, 0x0057), (0x00761C0A, 0x0076),
        (0x0043BD9C, 0x0044), (0x00073034, 0x0007), (0x0059C7A0, 0x005A), (0x0064287D, 0x0064),
        (0x00716FB7, 0x0071), (0x003239F8, 0x0032), (0x0050611D, 0x0050), (0x00199338, 0x001A),
        (0x0044D4CC, 0x0045), (0x00790BD9, 0x0079), (0x002F6BF5, 0x002F), (0x0077904C, 0x0078),
        (0x0052B0FA, 0x0053), (0x005EF29E, 0x005F), (0x0074DA5B, 0x0075), (0x0032103C, 0x0032),
        (0x002718A5, 0x0027), (0x0040AC11, 0x0041), (0x006C0C8A, 0x006C), (0x007090AD, 0x0071),
        (0x0059BE06, 0x005A), (0x001397E9, 0x0014), (0x00062500, 0x0006), (0x007FC77A, 0x0080),
        (0x000256E0, 0x0002), (0x0038A0F1, 0x0039), (0x00107B69, 0x0010), (0x005729DE, 0x0057),
        (0x001E9393, 0x001F), (0x004B92C7, 0x004C),
    ];

    #[test]
    fn rounding_agrees_with_pytorch_on_a_table() {
        assert_eq!(BF16_TABLE.len(), 238);
        for (bits, expected) in BF16_TABLE {
            assert_eq!(
                f32_to_bf16(f32::from_bits(*bits)),
                *expected,
                "f32 bits {bits:#010x}"
            );
        }
    }

    #[test]
    fn bf16_widening_is_exact() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC000), -2.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert!(bf16_to_f32(0x8000).is_sign_negative());
        assert!(bf16_to_f32(0x7F80).is_infinite());
        for bits in [0u16, 1, 0x007F, 0x0080, 0x3F81, 0x7F7F, 0xFF7F] {
            assert_eq!(f32_to_bf16(bf16_to_f32(bits)), bits, "{bits:#06x}");
        }
    }

    #[test]
    fn rounding_is_to_nearest_even() {
        // 1 + 2^-8 is exactly between 0x3F80 and 0x3F81: ties go to the even one
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
        // 1 + 3*2^-8 is exactly between 0x3F81 and 0x3F82
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
        // just above and just below the tie
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8001)), 0x3F81);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_7FFF)), 0x3F80);
        // negative values are symmetric
        assert_eq!(f32_to_bf16(f32::from_bits(0xBF80_8000)), 0xBF80);
        assert_eq!(f32_to_bf16(f32::from_bits(0xBF81_8000)), 0xBF82);
        // signed zero, subnormals
        assert_eq!(f32_to_bf16(0.0), 0x0000);
        assert_eq!(f32_to_bf16(-0.0), 0x8000);
        assert_eq!(f32_to_bf16(f32::from_bits(0x0000_8000)), 0x0000); // tie to even (0)
        assert_eq!(f32_to_bf16(f32::from_bits(0x0001_8000)), 0x0002); // tie to even (2)
        // the largest finite bfloat16 survives; the next float up rounds to infinity
        assert_eq!(f32_to_bf16(f32::from_bits(0x7F7F_0000)), 0x7F7F);
        assert_eq!(f32_to_bf16(f32::from_bits(0x7F7F_FFFF)), 0x7F80);
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7F80);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn neg_exp_matches_known_values() {
        assert_eq!(neg_exp(0.0), -1.0);
        assert!((neg_exp(1.0) + std::f32::consts::E).abs() < 1e-6);
        assert!(neg_exp(-1.796_875) < 0.0);
        assert!(neg_exp(-200.0) == 0.0 && neg_exp(-200.0).is_sign_negative());
    }

    #[test]
    fn merge_matches_a_hand_computed_case() {
        // r = 2, n = 3: delta = 1*[1,2,3] + 2*[10,20,30] = [21,42,63]; scale 2 -> [42,84,126]
        let w = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0];
        let a = [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let (mut delta, mut out) = ([0.0f32; 3], [0.0f32; 3]);
        merge_row(&w, &b, &a, 2.0, &mut delta, &mut out);
        assert_eq!(out, [43.0, 86.0, 129.0]);
    }

    #[test]
    fn a_zero_adapter_leaves_the_weights_bit_for_bit() {
        let w = [0.1f32, -3.5e-39, 7.0e8, -0.0];
        let a = [1.0f32; 8];
        let (mut delta, mut out) = ([0.0f32; 4], [0.0f32; 4]);
        merge_row(&w, &[0.0, 0.0], &a, 2.0, &mut delta, &mut out);
        for (o, x) in out.iter().zip(&w) {
            // -0.0 + 0.0 is +0.0 in IEEE: the value is equal, the sign of a zero may flip
            assert!(o.to_bits() == x.to_bits() || (*o == 0.0 && *x == 0.0));
        }
    }

    #[test]
    fn the_sum_is_sequential_and_fused() {
        // Second term: big*big = 1 + 2^-22 + 2^-46 exactly. A plain product rounds the 2^-46
        // away, so adding it to the first term (-(1 + 2^-22)) gives 0; a fused one keeps it and
        // gives 2^-46. The test proves the two kinds of arithmetic differ, and that we fuse.
        let big = 1.0f32 + f32::EPSILON; // 1 + 2^-23
        let b = [1.0f32, big];
        let a = [-(1.0f32 + 2.0 * f32::EPSILON), big]; // one column
        let (mut delta, mut out) = ([0.0f32; 1], [0.0f32; 1]);
        merge_row(&[0.0], &b, &a, 1.0, &mut delta, &mut out);
        let separate = b[0] * a[0] + b[1] * a[1];
        assert_eq!(delta[0], 2.0f32.powi(-46));
        assert_eq!(out[0], 2.0f32.powi(-46));
        assert_eq!(
            separate, 0.0,
            "the case must tell fused from separate arithmetic"
        );
    }

    #[test]
    fn an_empty_adapter_or_row_is_harmless() {
        let (mut delta, mut out): ([f32; 0], [f32; 0]) = ([], []);
        merge_row(&[], &[1.0], &[], 2.0, &mut delta, &mut out);
        let w = [5.0f32];
        let (mut delta, mut out) = ([0.0f32; 1], [0.0f32; 1]);
        merge_row(&w, &[], &[], 2.0, &mut delta, &mut out);
        assert_eq!(out, [5.0]);
    }
}
