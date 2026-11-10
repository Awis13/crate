// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Linux's arm64 boot protocol (Documentation/arch/arm64/booting.rst): the
//! Image header, and where the kernel, initrd and device tree go in RAM.

use std::io::Read;

use crate_core::{Error, Result};

const MAGIC: u32 = 0x644d_5241; // "ARM\x64"
const HEADER_SIZE: usize = 64;
const SZ_2M: u64 = 2 << 20;
/// The device tree may be at most 2 MB.
pub const DTB_MAX: u64 = SZ_2M;

#[derive(Debug)]
pub struct KernelImage {
    pub data: Vec<u8>,
    pub text_offset: u64,
    /// The memory the kernel needs from where it is loaded, bss included.
    pub image_size: u64,
}

/// An arm64 Image, raw or gzip-compressed.
pub fn parse_image(bytes: Vec<u8>) -> Result<KernelImage> {
    let data = if bytes.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(bytes.as_slice())
            .read_to_end(&mut out)
            .map_err(|e| Error::Guest(format!("decompressing the kernel: {e}")))?;
        out
    } else {
        bytes
    };
    if data.len() < HEADER_SIZE {
        return Err(Error::Guest(
            "the kernel is shorter than an Image header".into(),
        ));
    }
    let u64_at = |at: usize| u64::from_le_bytes(data[at..at + 8].try_into().unwrap());
    let magic = u32::from_le_bytes(data[56..60].try_into().unwrap());
    if magic != MAGIC {
        return Err(Error::Guest(
            "not an arm64 Image (no ARM\\x64 magic at offset 56)".into(),
        ));
    }
    let flags = u64_at(24);
    if flags & 1 != 0 {
        return Err(Error::Guest("a big-endian kernel".into()));
    }
    let mut image_size = u64_at(16);
    let text_offset = if image_size == 0 {
        // Before Linux 3.17 the header gave no size and text_offset was 0x80000.
        image_size = data.len() as u64;
        0x8_0000
    } else {
        u64_at(8)
    };
    Ok(KernelImage {
        image_size: image_size.max(data.len() as u64),
        text_offset,
        data,
    })
}

/// Where everything goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub kernel: u64,
    pub initrd: Option<(u64, u64)>,
    pub dtb: u64,
}

/// Places the kernel at the start of RAM (which must be 2 MB aligned), the
/// device tree in the last 2 MB, and the initrd just below it.
pub fn place(
    ram_base: u64,
    ram_size: u64,
    kernel: &KernelImage,
    initrd_len: Option<u64>,
) -> Result<Placement> {
    if ram_base % SZ_2M != 0 || ram_size % SZ_2M != 0 {
        return Err(Error::InvalidArgument(
            "RAM must start and end on 2 MB boundaries".into(),
        ));
    }
    let load = ram_base + kernel.text_offset;
    let kernel_end = (load + kernel.image_size).div_ceil(SZ_2M) * SZ_2M;
    let dtb = ram_base + ram_size - DTB_MAX;
    let initrd = match initrd_len {
        Some(len) => {
            let start = dtb
                .checked_sub(len)
                .map(|s| s & !0xffff)
                .filter(|&s| s >= kernel_end)
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "{} MiB of RAM cannot hold the kernel and initrd",
                        ram_size >> 20
                    ))
                })?;
            Some((start, start + len))
        }
        None if dtb < kernel_end => {
            return Err(Error::InvalidArgument(format!(
                "{} MiB of RAM cannot hold the kernel",
                ram_size >> 20
            )));
        }
        None => None,
    };
    Ok(Placement {
        kernel: load,
        initrd,
        dtb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(text_offset: u64, image_size: u64) -> Vec<u8> {
        let mut h = vec![0u8; 4096];
        h[8..16].copy_from_slice(&text_offset.to_le_bytes());
        h[16..24].copy_from_slice(&image_size.to_le_bytes());
        h[24..32].copy_from_slice(&0b1010u64.to_le_bytes());
        h[56..60].copy_from_slice(&MAGIC.to_le_bytes());
        h
    }

    #[test]
    fn reads_the_header() {
        let k = parse_image(header(0, 40 << 20)).unwrap();
        assert_eq!((k.text_offset, k.image_size), (0, 40 << 20));
        assert!(parse_image(vec![0; 4096]).is_err());
        assert!(parse_image(vec![0; 10]).is_err());
    }

    #[test]
    fn decompresses_gzip() {
        use std::io::Write;
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        gz.write_all(&header(0, 1 << 20)).unwrap();
        let k = parse_image(gz.finish().unwrap()).unwrap();
        assert_eq!(k.data.len(), 4096);
        assert_eq!(k.image_size, 1 << 20);
    }

    #[test]
    fn places_kernel_initrd_and_dtb() {
        let k = parse_image(header(0, 40 << 20)).unwrap();
        let p = place(0x4000_0000, 2 << 30, &k, Some(36 << 20)).unwrap();
        assert_eq!(p.kernel, 0x4000_0000);
        assert_eq!(p.dtb, 0x4000_0000 + (2 << 30) - (2 << 20));
        let (start, end) = p.initrd.unwrap();
        assert_eq!(end, start + (36 << 20));
        assert!(end <= p.dtb && start >= p.kernel + (40 << 20));
        assert_eq!(start % 0x10000, 0);
        assert!(
            place(0x4000_0000, 64 << 20, &k, Some(36 << 20)).is_err(),
            "too little RAM"
        );
        assert!(
            place(0x4000_1000, 2 << 30, &k, None).is_err(),
            "unaligned RAM"
        );
    }
}
