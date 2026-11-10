// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Storage behind a block device: a disk image file on most hosts.

use std::io;
use std::sync::Mutex;

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
