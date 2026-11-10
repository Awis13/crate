// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Memory-mapped I/O: devices at guest physical addresses outside RAM.
//!
//! A CPU backend decodes a guest load or store that missed RAM into an address,
//! a size of 1, 2, 4 or 8 bytes and, for a store, a value, and hands it to the
//! bus, which passes it to the device as an offset into that device's window.

use std::sync::Arc;

use crate::{Error, Result};

/// A device register window. Accesses arrive from any vCPU thread, so a
/// device keeps its own state behind its own lock.
pub trait MmioDevice: Send + Sync {
    /// A load of `size` bytes at `offset`; the value is zero-extended.
    fn read(&self, offset: u64, size: u8) -> u64;
    /// A store of the low `size` bytes of `value` at `offset`.
    fn write(&self, offset: u64, size: u8, value: u64);
}

struct Window {
    base: u64,
    size: u64,
    device: Arc<dyn MmioDevice>,
}

#[derive(Default)]
pub struct MmioBus {
    windows: Vec<Window>,
}

impl MmioBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, base: u64, size: u64, device: Arc<dyn MmioDevice>) -> Result<()> {
        let end = base
            .checked_add(size)
            .filter(|_| size > 0)
            .ok_or_else(|| Error::InvalidArgument(format!("MMIO window {base:#x}+{size:#x}")))?;
        if let Some(other) = self
            .windows
            .iter()
            .find(|w| base < w.base + w.size && w.base < end)
        {
            return Err(Error::InvalidArgument(format!(
                "MMIO window at {base:#x} overlaps the one at {:#x}",
                other.base
            )));
        }
        let at = self.windows.partition_point(|w| w.base < base);
        self.windows.insert(at, Window { base, size, device });
        Ok(())
    }

    fn find(&self, addr: u64) -> Option<(&Window, u64)> {
        let at = self.windows.partition_point(|w| w.base <= addr);
        let window = self.windows.get(at.checked_sub(1)?)?;
        let offset = addr - window.base;
        (offset < window.size).then_some((window, offset))
    }

    /// `None` when no device answers at `addr`.
    pub fn read(&self, addr: u64, size: u8) -> Option<u64> {
        let (window, offset) = self.find(addr)?;
        Some(window.device.read(offset, size) & size_mask(size))
    }

    /// `false` when no device answers at `addr`.
    pub fn write(&self, addr: u64, size: u8, value: u64) -> bool {
        match self.find(addr) {
            Some((window, offset)) => {
                window.device.write(offset, size, value & size_mask(size));
                true
            }
            None => false,
        }
    }
}

pub fn size_mask(size: u8) -> u64 {
    match size {
        8.. => u64::MAX,
        n => (1u64 << (n as u32 * 8)) - 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Recorder(Mutex<Vec<(u64, u8, u64)>>);

    impl MmioDevice for Recorder {
        fn read(&self, offset: u64, _size: u8) -> u64 {
            0x1122_3344_5566_7788 + offset
        }
        fn write(&self, offset: u64, size: u8, value: u64) {
            self.0.lock().unwrap().push((offset, size, value));
        }
    }

    #[test]
    fn routes_by_address_and_masks_by_size() {
        let a = Arc::new(Recorder::default());
        let b = Arc::new(Recorder::default());
        let mut bus = MmioBus::new();
        bus.insert(0x2000, 0x1000, b.clone()).unwrap();
        bus.insert(0x1000, 0x1000, a.clone()).unwrap();
        assert_eq!(bus.read(0x1000, 4), Some(0x5566_7788));
        assert_eq!(bus.read(0x2010, 1), Some(0x98));
        assert_eq!(bus.read(0x3000, 4), None);
        assert_eq!(bus.read(0xfff, 4), None);
        assert!(bus.write(0x2ffc, 2, 0xabcd_ef01));
        assert_eq!(*b.0.lock().unwrap(), vec![(0xffc, 2, 0xef01)]);
        assert!(a.0.lock().unwrap().is_empty());
    }

    #[test]
    fn refuses_overlaps() {
        let mut bus = MmioBus::new();
        bus.insert(0x1000, 0x1000, Arc::new(Recorder::default()))
            .unwrap();
        assert!(
            bus.insert(0x1800, 0x1000, Arc::new(Recorder::default()))
                .is_err()
        );
        assert!(
            bus.insert(0x0800, 0x1000, Arc::new(Recorder::default()))
                .is_err()
        );
        assert!(
            bus.insert(0x2000, 0, Arc::new(Recorder::default()))
                .is_err()
        );
        bus.insert(0x2000, 0x10, Arc::new(Recorder::default()))
            .unwrap();
    }
}
