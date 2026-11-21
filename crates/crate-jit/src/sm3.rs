use crate::{backend::Cpu, state::CpuState};

fn lanes(v: &[u8; 16]) -> [u32; 4] {
    std::array::from_fn(|i| u32::from_le_bytes(v[i * 4..i * 4 + 4].try_into().unwrap()))
}

fn bytes(v: [u32; 4]) -> [u8; 16] {
    let mut out = [0; 16];
    for (i, x) in v.into_iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
    }
    out
}

fn add(xs: &[u32]) -> u32 {
    xs.iter().fold(0u32, |a, &b| a.wrapping_add(b))
}

const SM4_SBOX: [u8; 256] = [
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
    0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
    0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
    0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
    0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
    0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
    0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
    0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
    0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
    0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
    0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
    0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48,
];

fn sm4_subword(x: u32) -> u32 {
    let mut out = 0;
    for shift in [0, 8, 16, 24] {
        out |= (SM4_SBOX[((x >> shift) & 0xff) as usize] as u32) << shift;
    }
    out
}

pub unsafe extern "C" fn jit_sm3(_cpu: *mut Cpu, word: u64, state: *mut CpuState) -> u64 {
    let state = unsafe { &mut *state };
    let w = word as u32;
    let d = (w & 31) as usize;
    let n = ((w >> 5) & 31) as usize;
    let m = ((w >> 16) & 31) as usize;
    let vd = lanes(&state.v[d]);
    let vn = lanes(&state.v[n]);
    let vm = lanes(&state.v[m]);
    let out = if w & 0xffff_fc00 == 0xcec0_8400 {
        let mut r = vd;
        for i in 0..4 {
            let t = sm4_subword(r[(i + 1) % 4] ^ r[(i + 2) % 4] ^ r[(i + 3) % 4] ^ vn[i]);
            r[i] ^=
                t ^ t.rotate_left(2) ^ t.rotate_left(10) ^ t.rotate_left(18) ^ t.rotate_left(24);
        }
        r
    } else if w & 0xffe0_fc00 == 0xce60_c800 {
        let mut r = vn;
        for i in 0..4 {
            let t = sm4_subword(r[(i + 1) % 4] ^ r[(i + 2) % 4] ^ r[(i + 3) % 4] ^ vm[i]);
            r[i] ^= t ^ t.rotate_left(13) ^ t.rotate_left(23);
        }
        r
    } else if w & 0xffe0_8000 == 0xce40_0000 {
        let va = lanes(&state.v[((w >> 10) & 31) as usize]);
        [
            0,
            0,
            0,
            add(&[vn[3].rotate_left(12), vm[3], va[3]]).rotate_left(7),
        ]
    } else if w & 0xffe0_fc00 == 0xce60_c000 {
        let mut r = [0; 4];
        for i in 0..3 {
            r[i] = vd[i] ^ vn[i] ^ vm[i + 1].rotate_left(15);
            r[i] ^= r[i].rotate_left(15) ^ r[i].rotate_left(23);
        }
        r[3] = vd[3] ^ vn[3] ^ r[0].rotate_left(15);
        r[3] ^= r[3].rotate_left(15) ^ r[3].rotate_left(23);
        r
    } else if w & 0xffe0_fc00 == 0xce60_c400 {
        let t: [u32; 4] = std::array::from_fn(|i| vn[i] ^ vm[i].rotate_left(7));
        let q = t[0].rotate_left(15);
        let p = q ^ q.rotate_left(15) ^ q.rotate_left(23);
        [vd[0] ^ t[0], vd[1] ^ t[1], vd[2] ^ t[2], vd[3] ^ t[3] ^ p]
    } else {
        let i = ((w >> 12) & 3) as usize;
        let op = (w >> 10) & 3;
        if op < 2 {
            let f = if op == 0 {
                vd[1] ^ vd[2] ^ vd[3]
            } else {
                (vd[3] & vd[1]) | (vd[3] & vd[2]) | (vd[1] & vd[2])
            };
            let ss2 = vn[3] ^ vd[3].rotate_left(12);
            [
                vd[1],
                vd[2].rotate_left(9),
                vd[3],
                add(&[f, vd[0], ss2, vm[i]]),
            ]
        } else {
            let f = if op == 2 {
                vd[1] ^ vd[2] ^ vd[3]
            } else {
                (vd[3] & vd[2]) | (!vd[3] & vd[1])
            };
            let t = add(&[f, vd[0], vn[3], vm[i]]);
            [
                vd[1],
                vd[2].rotate_left(19),
                vd[3],
                t ^ t.rotate_left(9) ^ t.rotate_left(17),
            ]
        }
    };
    state.v[d] = bytes(out);
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crash_opcode_is_partw1() {
        assert_eq!(0xce63_c004u32 & 0xffe0_fc00, 0xce60_c000);
    }
    #[test]
    fn lane_roundtrip() {
        let x = [1, 2, 3, 4];
        assert_eq!(lanes(&bytes(x)), x);
    }
    #[test]
    fn sm4_sbox_and_subword() {
        assert_eq!(SM4_SBOX[0], 0xd6);
        assert_eq!(SM4_SBOX[0x80], 0xea);
        assert_eq!(SM4_SBOX[0xff], 0x48);
        assert_eq!(sm4_subword(0x0001_80ff), 0xd690_ea48);
    }
}
