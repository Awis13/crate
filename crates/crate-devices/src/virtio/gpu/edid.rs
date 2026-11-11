// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! An EDID (VESA E-EDID 1.4) offering exactly the modes the display was given:
//! each as a detailed timing descriptor, the first one preferred, with CVT
//! timings at 60 Hz. The base block holds three and a monitor name; the rest
//! go in CTA-861 extension blocks, six to a block.

use crate_core::display::Mode;

const BLOCK: usize = 128;
/// The most extension blocks a virtio-gpu EDID response has room for (1024
/// bytes in all).
const MAX_EXTENSIONS: usize = 7;
const BASE_TIMINGS: usize = 3;
const EXTENSION_TIMINGS: usize = 6;

/// The most modes an EDID can carry.
pub const MAX_MODES: usize = BASE_TIMINGS + MAX_EXTENSIONS * EXTENSION_TIMINGS;

/// A mode's timings: the active area, blanking and sync, in pixels and lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timing {
    /// In kHz.
    pub clock_khz: u32,
    pub h_active: u32,
    pub h_front: u32,
    pub h_sync: u32,
    pub h_blank: u32,
    pub v_active: u32,
    pub v_front: u32,
    pub v_sync: u32,
    pub v_blank: u32,
    /// CVT's reduced blanking, for modes whose CRT timings would need a
    /// faster pixel clock than a descriptor can say (655.35 MHz).
    pub reduced: bool,
}

/// The fastest pixel clock a detailed timing descriptor can carry.
const MAX_CLOCK_KHZ: u32 = 655_350;

/// VESA Coordinated Video Timings (CVT 1.2) for `mode` at 60 Hz, progressive,
/// without margins: CRT blanking, or reduced blanking when that is too fast.
pub fn cvt(mode: Mode) -> Timing {
    let crt = cvt_crt(mode);
    if crt.clock_khz <= MAX_CLOCK_KHZ {
        crt
    } else {
        cvt_reduced(mode, crt.v_sync)
    }
}

const CELL: f64 = 8.0;
const RATE: f64 = 60.0;
const MIN_V_PORCH: f64 = 3.0;
const MIN_V_BPORCH: f64 = 6.0;
const CLOCK_STEP_MHZ: f64 = 0.25;

fn cvt_crt(mode: Mode) -> Timing {
    const MIN_VSYNC_BP_US: f64 = 550.0;
    const C_PRIME: f64 = 30.0; // ((40 - 20) * 128 / 256) + 20
    const M_PRIME: f64 = 300.0; // 128 / 256 * 600
    const H_SYNC_PERCENT: f64 = 8.0;

    let h_pixels = (mode.width as f64 / CELL).floor() * CELL;
    let v_lines = mode.height as f64;
    let (w, h) = (mode.width, mode.height);
    // The sync width says the aspect ratio to a monitor that cares.
    let v_sync = if w * 3 == h * 4 {
        4.0
    } else if w * 9 == h * 16 {
        5.0
    } else if w * 10 == h * 16 {
        6.0
    } else if w * 4 == h * 5 || w * 9 == h * 15 {
        7.0
    } else {
        10.0
    };
    let h_period_us = (1_000_000.0 / RATE - MIN_VSYNC_BP_US) / (v_lines + MIN_V_PORCH);
    let v_sync_bp = ((MIN_VSYNC_BP_US / h_period_us).floor() + 1.0).max(v_sync + MIN_V_BPORCH);
    let v_blank = v_sync_bp + MIN_V_PORCH;
    let duty = (C_PRIME - M_PRIME * h_period_us / 1000.0).max(20.0);
    let h_blank = (h_pixels * duty / (100.0 - duty) / (2.0 * CELL)).floor() * 2.0 * CELL;
    let total_pixels = h_pixels + h_blank;
    let clock_mhz = CLOCK_STEP_MHZ * (total_pixels / h_period_us / CLOCK_STEP_MHZ).floor();
    let h_sync = (H_SYNC_PERCENT / 100.0 * total_pixels / CELL).floor() * CELL;
    let h_back = h_blank / 2.0;
    // Blanking stays what CVT computed for the rounded width, so an exact width
    // that is not a multiple of eight still totals the same pixels a line.
    let extra = mode.width as f64 - h_pixels;
    Timing {
        clock_khz: (clock_mhz * 1000.0) as u32,
        h_active: mode.width,
        h_front: (h_blank - h_sync - h_back) as u32,
        h_sync: h_sync as u32,
        h_blank: (h_blank - extra).max(h_sync + 1.0) as u32,
        v_active: mode.height,
        v_front: MIN_V_PORCH as u32,
        v_sync: v_sync as u32,
        v_blank: v_blank as u32,
        reduced: false,
    }
}

/// CVT with reduced blanking (version 1): a fixed 160-pixel horizontal blank.
fn cvt_reduced(mode: Mode, v_sync: u32) -> Timing {
    const MIN_V_BLANK_US: f64 = 460.0;
    const H_BLANK: f64 = 160.0;
    const H_SYNC: f64 = 32.0;
    const H_FRONT: f64 = 48.0;
    let v_lines = mode.height as f64;
    let h_period_us = (1_000_000.0 / RATE - MIN_V_BLANK_US) / v_lines;
    let vbi = ((MIN_V_BLANK_US / h_period_us).floor() + 1.0)
        .max(MIN_V_PORCH + v_sync as f64 + MIN_V_BPORCH);
    let total_lines = v_lines + vbi;
    let total_pixels = mode.width as f64 + H_BLANK;
    let clock_mhz =
        CLOCK_STEP_MHZ * (RATE * total_lines * total_pixels / 1_000_000.0 / CLOCK_STEP_MHZ).floor();
    Timing {
        clock_khz: ((clock_mhz * 1000.0) as u32).min(MAX_CLOCK_KHZ),
        h_active: mode.width,
        h_front: H_FRONT as u32,
        h_sync: H_SYNC as u32,
        h_blank: H_BLANK as u32,
        v_active: mode.height,
        v_front: MIN_V_PORCH as u32,
        v_sync,
        v_blank: vbi as u32,
        reduced: true,
    }
}

/// A detailed timing descriptor (18 bytes).
fn detailed_timing(mode: Mode) -> [u8; 18] {
    let t = cvt(mode);
    let mut d = [0u8; 18];
    let clock = (t.clock_khz / 10).min(u16::MAX as u32) as u16;
    d[0..2].copy_from_slice(&clock.to_le_bytes());
    d[2] = t.h_active as u8;
    d[3] = t.h_blank as u8;
    d[4] = ((t.h_active >> 8) as u8 & 0xf) << 4 | ((t.h_blank >> 8) as u8 & 0xf);
    d[5] = t.v_active as u8;
    d[6] = t.v_blank as u8;
    d[7] = ((t.v_active >> 8) as u8 & 0xf) << 4 | ((t.v_blank >> 8) as u8 & 0xf);
    d[8] = t.h_front as u8;
    d[9] = t.h_sync as u8;
    d[10] = (t.v_front as u8 & 0xf) << 4 | (t.v_sync as u8 & 0xf);
    d[11] = ((t.h_front >> 8) as u8 & 3) << 6
        | ((t.h_sync >> 8) as u8 & 3) << 4
        | ((t.v_front >> 4) as u8 & 3) << 2
        | ((t.v_sync >> 4) as u8 & 3);
    // The picture's size in millimetres, as at 96 dots an inch.
    let mm = |px: u32| px * 254 / 960;
    let (wmm, hmm) = (mm(mode.width), mm(mode.height));
    d[12] = wmm as u8;
    d[13] = hmm as u8;
    d[14] = ((wmm >> 8) as u8 & 0xf) << 4 | ((hmm >> 8) as u8 & 0xf);
    // Digital separate sync. CVT's polarities are vsync positive and hsync
    // negative; reduced blanking's the other way round.
    d[17] = 0x18 | if t.reduced { 0x02 } else { 0x04 };
    d
}

fn monitor_name(name: &str) -> [u8; 18] {
    let mut d = [0u8; 18];
    d[3] = 0xfc;
    let text = &mut d[5..];
    text.fill(b' ');
    let bytes: Vec<u8> = name.bytes().filter(u8::is_ascii_graphic).take(12).collect();
    text[..bytes.len()].copy_from_slice(&bytes);
    text[bytes.len()] = b'\n';
    d
}

fn checksum(block: &mut [u8]) {
    let sum = block[..BLOCK - 1]
        .iter()
        .fold(0u8, |sum, &b| sum.wrapping_add(b));
    block[BLOCK - 1] = 0u8.wrapping_sub(sum);
}

/// The EDID for `modes`, preferred first; modes past `MAX_MODES` are left out.
pub fn edid(modes: &[Mode]) -> Vec<u8> {
    let modes = &modes[..modes.len().min(MAX_MODES)];
    let extensions = modes
        .len()
        .saturating_sub(BASE_TIMINGS)
        .div_ceil(EXTENSION_TIMINGS);
    let mut out = vec![0u8; BLOCK * (1 + extensions)];

    let base = &mut out[..BLOCK];
    base[0..8].copy_from_slice(&[0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0]);
    // Manufacturer "CRT", three letters of five bits each, big-endian.
    let id: u16 = (3 << 10) | (18 << 5) | 20;
    base[8..10].copy_from_slice(&id.to_be_bytes());
    base[10..12].copy_from_slice(&1u16.to_le_bytes()); // product
    base[12..16].copy_from_slice(&1u32.to_le_bytes()); // serial
    base[16] = 1; // week
    base[17] = (2026 - 1990) as u8;
    base[18] = 1;
    base[19] = 4;
    base[20] = 0xa0; // digital, 8 bits a colour
    if let Some(first) = modes.first() {
        base[21] = (first.width * 254 / 96_00).max(1) as u8; // centimetres
        base[22] = (first.height * 254 / 96_00).max(1) as u8;
    }
    base[23] = 120; // gamma 2.2
    // sRGB the default colour space; the preferred timing is the native mode.
    base[24] = 0x06;
    // sRGB's primaries and white point, ten bits each.
    let c = |v: f64| (v * 1024.0).round() as u16;
    let [rx, ry, gx, gy, bx, by, wx, wy] =
        [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290].map(c);
    let low = |a: u16, b: u16, c: u16, d: u16| {
        ((a & 3) << 6 | (b & 3) << 4 | (c & 3) << 2 | (d & 3)) as u8
    };
    base[25] = low(rx, ry, gx, gy);
    base[26] = low(bx, by, wx, wy);
    for (i, v) in [rx, ry, gx, gy, bx, by, wx, wy].iter().enumerate() {
        base[27 + i] = (v >> 2) as u8;
    }
    // No established timings; standard timings unused.
    for i in 0..8 {
        base[38 + 2 * i] = 1;
        base[39 + 2 * i] = 1;
    }
    for slot in 0..4 {
        let at = 54 + 18 * slot;
        let descriptor = match modes.get(slot) {
            Some(&mode) if slot < BASE_TIMINGS => detailed_timing(mode),
            _ if slot == 3 => monitor_name("crate"),
            // A dummy descriptor.
            _ => {
                let mut d = [0u8; 18];
                d[3] = 0x10;
                d
            }
        };
        base[at..at + 18].copy_from_slice(&descriptor);
    }
    base[126] = extensions as u8;
    checksum(base);

    for (n, chunk) in modes
        .get(BASE_TIMINGS..)
        .unwrap_or(&[])
        .chunks(EXTENSION_TIMINGS)
        .enumerate()
    {
        let block = &mut out[BLOCK * (1 + n)..BLOCK * (2 + n)];
        block[0] = 0x02; // CTA-861
        block[1] = 3;
        block[2] = 4; // the timings start right after this header: no data blocks
        block[3] = 0;
        for (i, &mode) in chunk.iter().enumerate() {
            let at = 4 + 18 * i;
            block[at..at + 18].copy_from_slice(&detailed_timing(mode));
        }
        checksum(block);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads the detailed timings back the way a driver does.
    fn modes_in(edid: &[u8]) -> Vec<(u32, u32, u32)> {
        let mut found = Vec::new();
        let mut parse = |d: &[u8]| {
            let clock = u16::from_le_bytes([d[0], d[1]]) as u32 * 10;
            if clock == 0 {
                return;
            }
            let h = d[2] as u32 | (d[4] as u32 >> 4) << 8;
            let hb = d[3] as u32 | (d[4] as u32 & 0xf) << 8;
            let v = d[5] as u32 | (d[7] as u32 >> 4) << 8;
            let vb = d[6] as u32 | (d[7] as u32 & 0xf) << 8;
            let total = (h + hb) * (v + vb);
            let refresh = (clock * 1000 + total / 2) / total;
            found.push((h, v, refresh));
        };
        for slot in 0..4 {
            parse(&edid[54 + 18 * slot..72 + 18 * slot]);
        }
        for block in edid[BLOCK..].chunks(BLOCK) {
            let mut at = block[2] as usize;
            while at + 18 < BLOCK {
                parse(&block[at..at + 18]);
                at += 18;
            }
        }
        found
    }

    #[test]
    fn cvt_matches_the_published_timing() {
        // VESA's CVT 1024x768 at 60 Hz: 63.50 MHz, 1024 1072 1176 1328,
        // 768 771 775 798.
        let t = cvt(Mode::new(1024, 768));
        assert_eq!(t.clock_khz, 63_500);
        assert_eq!(
            (t.h_front, t.h_sync, t.h_blank),
            (1072 - 1024, 1176 - 1072, 1328 - 1024)
        );
        assert_eq!(
            (t.v_front, t.v_sync, t.v_blank),
            (771 - 768, 775 - 771, 798 - 768)
        );
    }

    #[test]
    fn carries_every_mode_with_valid_checksums() {
        let modes: Vec<Mode> = [
            (800, 600),
            (640, 480),
            (1024, 768),
            (1280, 960),
            (1920, 1080),
            (1366, 768),
            (2560, 1600),
            (3840, 2160),
            (1180, 820),
            (1001, 999),
        ]
        .iter()
        .map(|&(w, h)| Mode::new(w, h))
        .collect();
        let e = edid(&modes);
        assert_eq!(e.len(), 3 * BLOCK);
        assert_eq!(&e[0..8], &[0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0]);
        assert_eq!(e[126], 2);
        for block in e.chunks(BLOCK) {
            assert_eq!(block.iter().fold(0u8, |s, &b| s.wrapping_add(b)), 0);
        }
        let found = modes_in(&e);
        let sizes: Vec<(u32, u32)> = found.iter().map(|&(w, h, _)| (w, h)).collect();
        let wanted: Vec<(u32, u32)> = modes.iter().map(|m| (m.width, m.height)).collect();
        assert_eq!(sizes, wanted, "in order, the preferred one first");
        for (w, h, refresh) in found {
            assert!((59..=60).contains(&refresh), "{w}x{h} at {refresh} Hz");
        }
    }

    #[test]
    fn small_lists_fit_the_base_block() {
        let e = edid(&[Mode::new(1280, 960)]);
        assert_eq!(e.len(), BLOCK);
        assert_eq!(e[126], 0);
        assert_eq!(&e[54 + 3 * 18 + 3..54 + 3 * 18 + 4], &[0xfc]);
        assert_eq!(&e[54 + 3 * 18 + 5..54 + 3 * 18 + 11], b"crate\n");
        assert_eq!(modes_in(&e), vec![(1280, 960, 60)]);
    }
}
