// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Pixel formats of 2D resources (VIRTIO 1.2, 5.7.6.8), which are virgl's own
//! numbers for the same formats, and the conversion to the frames' R, G, B,
//! A byte order.

pub const B8G8R8A8: u32 = 1;
pub const B8G8R8X8: u32 = 2;
pub const A8R8G8B8: u32 = 3;
pub const X8R8G8B8: u32 = 4;
pub const R8G8B8A8: u32 = 67;
pub const X8B8G8R8: u32 = 68;
pub const A8B8G8R8: u32 = 121;
pub const R8G8B8X8: u32 = 134;

/// Whether `format` is one of the 4-byte formats above.
pub fn supported(format: u32) -> bool {
    matches!(
        format,
        B8G8R8A8 | B8G8R8X8 | A8R8G8B8 | X8R8G8B8 | R8G8B8A8 | X8B8G8R8 | A8B8G8R8 | R8G8B8X8
    )
}

/// The same layout with its fourth byte read as alpha. A cursor's resource
/// says X when it holds A: Linux makes every dumb buffer XRGB, whatever the
/// framebuffer on it.
pub fn with_alpha(format: u32) -> u32 {
    match format {
        B8G8R8X8 => B8G8R8A8,
        X8R8G8B8 => A8R8G8B8,
        R8G8B8X8 => R8G8B8A8,
        X8B8G8R8 => A8B8G8R8,
        other => other,
    }
}

/// Rewrites `pixels` of `format` as R, G, B, A in memory; formats without
/// alpha come out opaque.
pub fn to_rgba(format: u32, pixels: &mut [u8]) {
    // Each pixel as a little-endian word, its first byte lowest.
    let each = |pixels: &mut [u8], f: fn(u32) -> u32| {
        for p in pixels.chunks_exact_mut(4) {
            let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
            p.copy_from_slice(&f(v).to_le_bytes());
        }
    };
    match format {
        // B, G, R, A: swap the first and third bytes.
        B8G8R8A8 => each(pixels, |v| {
            (v & 0xff00_ff00) | (v & 0xff) << 16 | (v >> 16) & 0xff
        }),
        B8G8R8X8 => each(pixels, |v| {
            0xff00_0000 | (v & 0xff00) | (v & 0xff) << 16 | (v >> 16) & 0xff
        }),
        // A, R, G, B: one byte down.
        A8R8G8B8 => each(pixels, |v| v.rotate_right(8)),
        X8R8G8B8 => each(pixels, |v| v >> 8 | 0xff00_0000),
        R8G8B8A8 => {}
        R8G8B8X8 => each(pixels, |v| v | 0xff00_0000),
        // A, B, G, R: reversed.
        A8B8G8R8 => each(pixels, u32::swap_bytes),
        X8B8G8R8 => each(pixels, |v| v.swap_bytes() | 0xff00_0000),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_format_comes_out_rgba() {
        // Red 1, green 2, blue 3, alpha 4, in each format's own byte order.
        let cases: [(u32, [u8; 4], [u8; 4]); 8] = [
            (B8G8R8A8, [3, 2, 1, 4], [1, 2, 3, 4]),
            (B8G8R8X8, [3, 2, 1, 9], [1, 2, 3, 255]),
            (A8R8G8B8, [4, 1, 2, 3], [1, 2, 3, 4]),
            (X8R8G8B8, [9, 1, 2, 3], [1, 2, 3, 255]),
            (R8G8B8A8, [1, 2, 3, 4], [1, 2, 3, 4]),
            (R8G8B8X8, [1, 2, 3, 9], [1, 2, 3, 255]),
            (A8B8G8R8, [4, 3, 2, 1], [1, 2, 3, 4]),
            (X8B8G8R8, [9, 3, 2, 1], [1, 2, 3, 255]),
        ];
        for (format, input, output) in cases {
            let mut p = input;
            to_rgba(format, &mut p);
            assert_eq!(p, output, "format {format}");
            assert!(supported(format));
        }
        assert!(!supported(5));
    }
}
