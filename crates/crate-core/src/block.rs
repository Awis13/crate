// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Storage behind a block device: a disk image file on most hosts.

use std::io;
use std::sync::{Arc, Mutex, RwLock};

pub trait BlockBackend: Send + Sync {
    /// Size in bytes.
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn read_only(&self) -> bool;
    /// Fills `buf` from `offset`; a short read is an error.
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()>;
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<()>;
    /// Makes completed writes durable.
    fn flush(&self) -> io::Result<()>;
    /// The device on this backend is live; `changed` tells the guest its
    /// size changed. Only a backend that can change needs it.
    fn on_activate(&self, _changed: Arc<dyn Fn() + Send + Sync>) {}
}

/// A disk whose image can be put in and taken out while the machine runs,
/// as a drive takes media: empty (no sectors) until one is set. The guest
/// sees the size change, so it can use a disk it did not boot with (a
/// container started later). It changes only while the guest leaves it
/// alone: nothing mounted on it, no requests in flight.
#[derive(Default)]
pub struct DiskSlot {
    image: RwLock<Option<Arc<dyn BlockBackend>>>,
    changed: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

impl DiskSlot {
    pub fn new() -> Self {
        Self::default()
    }

    /// Puts an image in, or takes it out with `None`.
    pub fn set(&self, image: Option<Arc<dyn BlockBackend>>) {
        *self.image.write().unwrap() = image;
        let changed = self.changed.lock().unwrap().clone();
        if let Some(changed) = changed {
            changed();
        }
    }

    pub fn has_image(&self) -> bool {
        self.image.read().unwrap().is_some()
    }

    fn image(&self) -> io::Result<Arc<dyn BlockBackend>> {
        self.image
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "the disk slot is empty"))
    }
}

impl BlockBackend for DiskSlot {
    fn len(&self) -> u64 {
        self.image.read().unwrap().as_ref().map_or(0, |i| i.len())
    }
    /// Writable: the guest learns that once, and an image put in later may
    /// be either (one opened read-only fails its writes).
    fn read_only(&self) -> bool {
        false
    }
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        self.image()?.read_at(buf, offset)
    }
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<()> {
        self.image()?.write_at(buf, offset)
    }
    fn flush(&self) -> io::Result<()> {
        match self.image.read().unwrap().clone() {
            Some(i) => i.flush(),
            None => Ok(()),
        }
    }
    fn on_activate(&self, changed: Arc<dyn Fn() + Send + Sync>) {
        *self.changed.lock().unwrap() = Some(changed);
    }
}

/// A disk in memory, for tests.
pub struct MemoryDisk {
    pub data: Mutex<Vec<u8>>,
    pub read_only: bool,
}

impl MemoryDisk {
    pub fn new(data: Vec<u8>) -> Self {
        MemoryDisk {
            data: Mutex::new(data),
            read_only: false,
        }
    }
}

impl BlockBackend for MemoryDisk {
    fn len(&self) -> u64 {
        self.data.lock().unwrap().len() as u64
    }
    fn read_only(&self) -> bool {
        self.read_only
    }
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        let data = self.data.lock().unwrap();
        let start = offset as usize;
        let src = data
            .get(start..start + buf.len())
            .ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))?;
        buf.copy_from_slice(src);
        Ok(())
    }
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<()> {
        let mut data = self.data.lock().unwrap();
        let start = offset as usize;
        let dst = data
            .get_mut(start..start + buf.len())
            .ok_or_else(|| io::Error::from(io::ErrorKind::WriteZero))?;
        dst.copy_from_slice(buf);
        Ok(())
    }
    fn flush(&self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn slot_takes_images_and_says_so() {
        let slot = DiskSlot::new();
        assert_eq!(slot.len(), 0);
        assert!(slot.read_at(&mut [0; 4], 0).is_err());
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        slot.on_activate(Arc::new(move || {
            seen.fetch_add(1, Ordering::SeqCst);
        }));
        slot.set(Some(Arc::new(MemoryDisk::new(vec![7; 1024]))));
        assert_eq!(slot.len(), 1024);
        let mut buf = [0; 4];
        slot.read_at(&mut buf, 512).unwrap();
        assert_eq!(buf, [7; 4]);
        slot.set(None);
        assert_eq!(slot.len(), 0);
        assert!(!slot.has_image());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
