// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Guest RAM as an anonymous private mapping: pages cost nothing until the
//! guest touches them.

use crate_core::memory::RamBlock;
use crate_core::{Error, Result};

pub struct AnonymousRam {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the mapping belongs to this value and is only reached through raw
// pointers, whose users accept that guest memory is shared.
unsafe impl Send for AnonymousRam {}
// SAFETY: as for Send.
unsafe impl Sync for AnonymousRam {}

impl AnonymousRam {
    /// `len` is rounded up to the host's page size.
    pub fn new(len: usize) -> Result<Self> {
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let len = len.div_ceil(page) * page;
        // SAFETY: a fresh anonymous mapping aliases nothing.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(Error::NoMemory(format!(
                "mapping {} MiB of guest RAM: {}",
                len >> 20,
                std::io::Error::last_os_error()
            )));
        }
        Ok(AnonymousRam {
            ptr: ptr.cast(),
            len,
        })
    }
}

impl Drop for AnonymousRam {
    fn drop(&mut self) {
        // SAFETY: mapped in `new` with this length and not unmapped since.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}

// SAFETY: the mapping never moves, spans `len` bytes and is page-aligned.
unsafe impl RamBlock for AnonymousRam {
    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}
