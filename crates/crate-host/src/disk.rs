// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A raw disk image file.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;

use crate_core::block::BlockBackend;
use crate_core::{Error, Result};

/// What a guest's flush does to the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flush {
    /// fsync: durable against a crash of this process or the guest, which is
    /// what a flush promises here, at a fraction of the cost of the next.
    Data,
    /// On Apple systems F_FULLFSYNC, which also empties the drive's own cache
    /// and so survives power loss; elsewhere the same as `Data`.
    Full,
    /// Nothing: fastest, for scratch disks, but a host crash can lose recent
    /// writes.
    Ignore,
}

pub struct FileDisk {
    file: File,
    len: u64,
    read_only: bool,
    flush: Flush,
}

impl FileDisk {
    pub fn open(path: &Path, read_only: bool) -> Result<Self> {
        Self::open_with(path, read_only, Flush::Data)
    }

    pub fn open_with(path: &Path, read_only: bool, flush: Flush) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(!read_only)
            .open(path)
            .map_err(|e| Error::io(format!("opening disk {}", path.display()), e))?;
        let len = file
            .metadata()
            .map_err(|e| Error::io(format!("reading the size of {}", path.display()), e))?
            .len();
        Ok(FileDisk {
            file,
            len,
            read_only,
            flush,
        })
    }
}

impl BlockBackend for FileDisk {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_only(&self) -> bool {
        self.read_only
    }
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        self.file.read_exact_at(buf, offset)
    }
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<()> {
        self.file.write_all_at(buf, offset)
    }
    fn flush(&self) -> io::Result<()> {
        match self.flush {
            Flush::Data => self.file.sync_data(),
            #[cfg(target_vendor = "apple")]
            Flush::Full => {
                use std::os::fd::AsRawFd;
                // SAFETY: an fcntl with no argument on our own descriptor.
                match unsafe { libc::fcntl(self.file.as_raw_fd(), libc::F_FULLFSYNC) } {
                    -1 => Err(io::Error::last_os_error()),
                    _ => Ok(()),
                }
            }
            #[cfg(not(target_vendor = "apple"))]
            Flush::Full => self.file.sync_data(),
            Flush::Ignore => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_and_writes_a_file() {
        let path = std::env::temp_dir().join(format!("crate-disk-test-{}", std::process::id()));
        std::fs::write(&path, vec![7u8; 2048]).unwrap();
        let disk = FileDisk::open(&path, false).unwrap();
        assert_eq!(disk.len(), 2048);
        disk.write_at(&[1, 2, 3], 1000).unwrap();
        let mut buf = [0; 5];
        disk.read_at(&mut buf, 999).unwrap();
        assert_eq!(buf, [7, 1, 2, 3, 7]);
        assert!(disk.read_at(&mut buf, 2046).is_err(), "past the end");
        let ro = FileDisk::open(&path, true).unwrap();
        assert!(ro.read_only());
        assert!(ro.write_at(&[0], 0).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
