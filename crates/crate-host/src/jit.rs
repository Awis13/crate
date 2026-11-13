// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Executable memory for the translator.
//!
//! Code is written through one address and run through another. Where the host
//! allows a page to be writable and executable at different times (a Mac's
//! MAP_JIT), the two addresses are the same and the writing thread switches
//! the region between the two states. Where it does not (iOS, Linux with a
//! strict policy), the same pages are mapped twice: once writable, once
//! executable. The translator never assumes which: it writes through
//! `write_base` and branches to addresses inside `exec_base`.
//!
//! On iOS an app may not make memory executable on its own; only a debugger
//! attached to it can, and attaching one and having it prepare the pages is
//! the app's business. So on iOS crate never maps a region for itself: the app
//! passes one in (`from_raw`, which `crate_config_set_jit_region` uses), and
//! `shared` fails. A machine with several CPUs carves its region into slices
//! (`JitRegion::split`), one per CPU.

use std::sync::Arc;

use crate_core::{Error, Result};

/// Where the region came from, which decides how writing is switched on and
/// how it is released.
enum Kind {
    /// One MAP_JIT mapping; writing is switched per thread.
    #[cfg(target_os = "macos")]
    MapJit,
    /// Two mappings of the same pages, owned here.
    Dual,
    /// Memory the app handed over, released by the app.
    Borrowed,
    /// Part of another region, which it keeps alive and whose way of
    /// switching writing it follows.
    Slice(#[cfg_attr(not(target_os = "macos"), allow(dead_code))] Arc<JitRegion>),
}

pub struct JitRegion {
    exec: *mut u8,
    write: *mut u8,
    len: usize,
    kind: Kind,
}

// SAFETY: the region is plain memory owned by this value (or lent to it for
// its whole life); who writes where is the translator's business.
unsafe impl Send for JitRegion {}
// SAFETY: as for Send.
unsafe impl Sync for JitRegion {}

fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

impl JitRegion {
    /// The region a machine's CPUs share when the app has not passed one in:
    /// a fresh pair of mappings (`new_dual`), so no CPU switches its W^X
    /// state. On iOS it fails, since there only a debugger the app attached
    /// can make the pages executable; the app passes the region in instead.
    pub fn shared(len: usize) -> Result<Arc<Self>> {
        #[cfg(target_os = "ios")]
        {
            let _ = len;
            Err(Error::Jit(
                "on iOS the app passes in executable memory that a debugger has \
                 prepared (crate_config_set_jit_region); crate cannot get its own"
                    .into(),
            ))
        }
        #[cfg(not(target_os = "ios"))]
        {
            Ok(Arc::new(Self::new_dual(len)?))
        }
    }

    /// The host's usual kind of executable memory: MAP_JIT on a Mac, two
    /// mappings of the same pages elsewhere. `len` is rounded up to pages.
    /// On iOS these pages are not executable unless a debugger prepares them,
    /// which is the app's to arrange; see `shared`.
    pub fn new(len: usize) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            Self::map_jit(len)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self::new_dual(len)
        }
    }

    #[cfg(target_os = "macos")]
    fn map_jit(len: usize) -> Result<Self> {
        let len = len.div_ceil(page_size()) * page_size();
        // SAFETY: a fresh anonymous mapping aliases nothing.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(Error::Jit(format!(
                "MAP_JIT, {} MiB: {}",
                len >> 20,
                std::io::Error::last_os_error()
            )));
        }
        Ok(JitRegion {
            exec: ptr.cast(),
            write: ptr.cast(),
            len,
            kind: Kind::MapJit,
        })
    }

    /// The same pages mapped twice, writable at one address and executable at
    /// another: the layout iOS needs. On a Mac it is the way to test that the
    /// translator keeps the two apart.
    pub fn new_dual(len: usize) -> Result<Self> {
        let len = len.div_ceil(page_size()) * page_size();
        // SAFETY: dual_map returns two fresh mappings of `len` bytes.
        let (exec, write) = unsafe { dual_map(len)? };
        Ok(JitRegion {
            exec,
            write,
            len,
            kind: Kind::Dual,
        })
    }

    /// Memory the app provides (`crate_config_set_jit_region`).
    ///
    /// # Safety
    ///
    /// `exec` and `write` must be aliases of the same `len` bytes (or the same
    /// address, where the host allows writing and running one mapping), the
    /// first executable and the second writable, valid until this value is
    /// dropped and used by nothing else.
    pub unsafe fn from_raw(exec: *mut u8, write: *mut u8, len: usize) -> Self {
        JitRegion {
            exec,
            write,
            len,
            kind: Kind::Borrowed,
        }
    }

    /// Carves the region into `parts` slices of equal size, each at most
    /// `max_each` bytes and a whole number of host pages, which keep the
    /// region alive and switch writing the way it does. What does not fit in
    /// them is left unused.
    pub fn split(self: Arc<Self>, parts: usize, max_each: usize) -> Result<Vec<JitRegion>> {
        let page = page_size();
        let each = (self.len / parts.max(1)).min(max_each) / page * page;
        if parts == 0 || each == 0 {
            return Err(Error::Jit(format!(
                "{} KiB cannot be split {parts} ways",
                self.len >> 10
            )));
        }
        Ok((0..parts)
            .map(|i| JitRegion {
                // SAFETY: each slice lies inside the region: i * each + each
                // <= parts * each <= len.
                exec: unsafe { self.exec.add(i * each) },
                // SAFETY: as above, in the writable alias.
                write: unsafe { self.write.add(i * each) },
                len: each,
                kind: Kind::Slice(self.clone()),
            })
            .collect())
    }

    /// Whether writing is switched per thread (MAP_JIT).
    #[cfg(target_os = "macos")]
    fn switches_writing(&self) -> bool {
        match &self.kind {
            Kind::MapJit => true,
            Kind::Slice(parent) => parent.switches_writing(),
            _ => false,
        }
    }

    pub fn exec_base(&self) -> *const u8 {
        self.exec
    }

    pub fn write_base(&self) -> *mut u8 {
        self.write
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether code is written and run at different addresses.
    pub fn is_aliased(&self) -> bool {
        self.exec != self.write
    }

    /// Lets the calling thread write the region until the guard is dropped.
    /// With MAP_JIT the thread cannot run the region's code meanwhile.
    pub fn writer(&self) -> JitWriter<'_> {
        #[cfg(target_os = "macos")]
        if self.switches_writing() {
            // SAFETY: switches this thread's view of MAP_JIT memory.
            unsafe { libc::pthread_jit_write_protect_np(0) };
        }
        JitWriter { region: self }
    }

    /// Makes `len` bytes written at `offset` visible to instruction fetch.
    pub fn flush(&self, offset: usize, len: usize) {
        assert!(offset.checked_add(len).is_some_and(|end| end <= self.len));
        // SAFETY: the range lies inside the region.
        unsafe { flush_icache(self.exec.add(offset), len) }
    }

    /// Makes every `(offset, len)` range written visible to instruction
    /// fetch, with one set of barriers for all of them.
    pub fn flush_ranges(&self, ranges: &[(usize, usize)]) {
        for &(offset, len) in ranges {
            assert!(offset.checked_add(len).is_some_and(|end| end <= self.len));
        }
        // SAFETY: every range lies inside the region.
        unsafe { flush_icache_ranges(self.exec, ranges) }
    }
}

impl Drop for JitRegion {
    fn drop(&mut self) {
        match self.kind {
            #[cfg(target_os = "macos")]
            // SAFETY: mapped in map_jit with this length.
            Kind::MapJit => unsafe {
                libc::munmap(self.exec.cast(), self.len);
            },
            // SAFETY: both were mapped by dual_map with this length.
            Kind::Dual => unsafe {
                libc::munmap(self.exec.cast(), self.len);
                libc::munmap(self.write.cast(), self.len);
            },
            Kind::Borrowed | Kind::Slice(_) => {}
        }
    }
}

/// Write access to a region for the calling thread.
pub struct JitWriter<'a> {
    region: &'a JitRegion,
}

impl JitWriter<'_> {
    /// Copies `words` to `offset` through the writable alias.
    pub fn write_words(&mut self, offset: usize, words: &[u32]) {
        let bytes = std::mem::size_of_val(words);
        assert!(offset % 4 == 0);
        assert!(
            offset
                .checked_add(bytes)
                .is_some_and(|end| end <= self.region.len)
        );
        // SAFETY: the range lies inside the writable alias, which this thread
        // may write while the guard lives.
        unsafe {
            std::ptr::copy_nonoverlapping(
                words.as_ptr().cast::<u8>(),
                self.region.write.add(offset),
                bytes,
            )
        }
    }

    /// Replaces one instruction, atomically as instruction fetch sees it.
    pub fn patch_word(&mut self, offset: usize, word: u32) {
        assert!(offset % 4 == 0 && offset + 4 <= self.region.len);
        // SAFETY: aligned, inside the writable alias.
        unsafe {
            std::sync::atomic::AtomicU32::from_ptr(self.region.write.add(offset).cast())
                .store(word, std::sync::atomic::Ordering::Release)
        }
    }
}

impl Drop for JitWriter<'_> {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        if self.region.switches_writing() {
            // SAFETY: switches this thread's view back to executable.
            unsafe { libc::pthread_jit_write_protect_np(1) };
        }
    }
}

#[cfg(target_vendor = "apple")]
mod mach {
    // From the SDK's mach/kern_return.h, mach/vm_statistics.h and
    // mach/vm_inherit.h.
    pub type KernReturn = i32;
    pub const KERN_SUCCESS: KernReturn = 0;
    pub const VM_FLAGS_ANYWHERE: libc::c_int = 1;
    pub const VM_INHERIT_NONE: u32 = 2;

    unsafe extern "C" {
        /// What mach_task_self() returns.
        pub static mach_task_self_: libc::mach_port_t;
        pub fn mach_vm_remap(
            target_task: libc::mach_port_t,
            target_address: *mut u64,
            size: u64,
            mask: u64,
            flags: libc::c_int,
            src_task: libc::mach_port_t,
            src_address: u64,
            copy: libc::boolean_t,
            cur_protection: *mut libc::vm_prot_t,
            max_protection: *mut libc::vm_prot_t,
            inheritance: u32,
        ) -> KernReturn;
        pub fn sys_icache_invalidate(start: *mut libc::c_void, len: usize);
    }
}

/// Two mappings of the same fresh pages: (executable, writable).
///
/// # Safety
///
/// `len` must be a non-zero multiple of the page size.
#[cfg(target_vendor = "apple")]
unsafe fn dual_map(len: usize) -> Result<(*mut u8, *mut u8)> {
    let fail = |what: &str| Error::Jit(format!("{what}: {}", std::io::Error::last_os_error()));
    // SAFETY: a fresh anonymous mapping aliases nothing.
    let write = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_ANON,
            -1,
            0,
        )
    };
    if write == libc::MAP_FAILED {
        return Err(fail("mapping the writable alias"));
    }
    let mut exec = 0u64;
    let (mut cur, mut max) = (0, 0);
    // SAFETY: remaps our own fresh pages into a new range of our own task.
    let kr = unsafe {
        let task = mach::mach_task_self_;
        mach::mach_vm_remap(
            task,
            &mut exec,
            len as u64,
            0,
            mach::VM_FLAGS_ANYWHERE,
            task,
            write as u64,
            0,
            &mut cur,
            &mut max,
            mach::VM_INHERIT_NONE,
        )
    };
    if kr != mach::KERN_SUCCESS {
        // SAFETY: mapped above.
        unsafe { libc::munmap(write, len) };
        return Err(Error::Jit(format!("mach_vm_remap: {kr:#x}")));
    }
    // SAFETY: the alias was just made and is ours.
    if unsafe { libc::mprotect(exec as *mut _, len, libc::PROT_READ | libc::PROT_EXEC) } != 0 {
        let e = fail("making the alias executable");
        // SAFETY: both mapped above.
        unsafe {
            libc::munmap(exec as *mut _, len);
            libc::munmap(write, len);
        }
        return Err(e);
    }
    Ok((exec as *mut u8, write.cast()))
}

#[cfg(target_os = "linux")]
unsafe fn dual_map(len: usize) -> Result<(*mut u8, *mut u8)> {
    let fail = |what: &str| Error::Jit(format!("{what}: {}", std::io::Error::last_os_error()));
    // SAFETY: the name is a valid C string.
    let fd = unsafe { libc::memfd_create(c"crate-jit".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(fail("memfd_create"));
    }
    // SAFETY: fd is ours; each mapping is fresh.
    let result = unsafe {
        if libc::ftruncate(fd, len as libc::off_t) != 0 {
            Err(fail("sizing the code file"))
        } else {
            let write = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            let exec = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_EXEC,
                libc::MAP_SHARED,
                fd,
                0,
            );
            if write == libc::MAP_FAILED || exec == libc::MAP_FAILED {
                let e = fail("mapping the code file");
                for m in [write, exec] {
                    if m != libc::MAP_FAILED {
                        libc::munmap(m, len);
                    }
                }
                Err(e)
            } else {
                Ok((exec.cast(), write.cast()))
            }
        }
    };
    // SAFETY: the mappings keep the file alive.
    unsafe { libc::close(fd) };
    result
}

/// # Safety
///
/// The range must be mapped.
#[cfg(target_vendor = "apple")]
unsafe fn flush_icache(start: *mut u8, len: usize) {
    // SAFETY: the caller's range is mapped.
    unsafe { mach::sys_icache_invalidate(start.cast(), len) }
}

/// The sequence `sys_icache_invalidate` runs for one range (which on these
/// cores cleans no data cache: their data side is coherent with instruction
/// fetch), for every range with one set of barriers: invalidate each 64-byte
/// line of instruction cache to the point of unification, with a DSB every
/// 20 lines as the system's own does on the cores that need one, then wait
/// for all of it and resynchronize this thread's fetch.
///
/// # Safety
///
/// Every range, from `base`, must be mapped.
#[cfg(all(target_vendor = "apple", target_arch = "aarch64"))]
unsafe fn flush_icache_ranges(base: *mut u8, ranges: &[(usize, usize)]) {
    use std::arch::asm;
    if ranges.is_empty() {
        return;
    }
    let mut lines = 0u32;
    // SAFETY: cache maintenance by address over mapped memory, and barriers.
    unsafe {
        asm!("dsb ish");
        for &(offset, len) in ranges {
            let start = base as usize + offset;
            let mut a = start & !63;
            while a < start + len {
                asm!("ic ivau, {}", in(reg) a);
                a += 64;
                lines += 1;
                if lines % 20 == 0 {
                    asm!("dsb ish");
                }
            }
        }
        asm!("dsb ish", "isb");
    }
}

/// Cleans the data cache to the point of unification and invalidates the
/// instruction cache over every range, with one set of barriers.
///
/// # Safety
///
/// Every range, from `base`, must be mapped and readable.
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
unsafe fn flush_icache_ranges(base: *mut u8, ranges: &[(usize, usize)]) {
    use std::arch::asm;
    if ranges.is_empty() {
        return;
    }
    let ctr: u64;
    // SAFETY: reads an ID register Linux exposes to user space.
    unsafe { asm!("mrs {}, ctr_el0", out(reg) ctr) };
    let dline = 4usize << ((ctr >> 16) & 15);
    let iline = 4usize << (ctr & 15);
    // SAFETY: cache maintenance by address over mapped memory, then barriers.
    unsafe {
        for &(offset, len) in ranges {
            let start = base as usize + offset;
            let mut a = start & !(dline - 1);
            while a < start + len {
                asm!("dc cvau, {}", in(reg) a);
                a += dline;
            }
        }
        asm!("dsb ish");
        for &(offset, len) in ranges {
            let start = base as usize + offset;
            let mut a = start & !(iline - 1);
            while a < start + len {
                asm!("ic ivau, {}", in(reg) a);
                a += iline;
            }
        }
        asm!("dsb ish", "isb");
    }
}

/// Cleans the data cache to the point of unification and invalidates the
/// instruction cache over the range, with the line sizes CTR_EL0 gives
/// (which Linux lets user space read).
///
/// # Safety
///
/// The range must be mapped and readable.
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
unsafe fn flush_icache(start: *mut u8, len: usize) {
    use std::arch::asm;
    let ctr: u64;
    // SAFETY: reads an ID register Linux exposes to user space.
    unsafe { asm!("mrs {}, ctr_el0", out(reg) ctr) };
    let dline = 4usize << ((ctr >> 16) & 15);
    let iline = 4usize << (ctr & 15);
    let (begin, end) = (start as usize, start as usize + len);
    // SAFETY: cache maintenance by address over mapped memory, then barriers.
    unsafe {
        let mut a = begin & !(dline - 1);
        while a < end {
            asm!("dc cvau, {}", in(reg) a);
            a += dline;
        }
        asm!("dsb ish");
        let mut a = begin & !(iline - 1);
        while a < end {
            asm!("ic ivau, {}", in(reg) a);
            a += iline;
        }
        asm!("dsb ish", "isb");
    }
}

/// The host's generic timer: the virtual counter and its frequency, which the
/// architecture lets user space read on every host crate runs on.
#[cfg(target_arch = "aarch64")]
pub mod counter {
    use std::arch::asm;

    /// CNTVCT_EL0, read after earlier instructions complete.
    #[inline]
    pub fn now() -> u64 {
        let v: u64;
        // SAFETY: reads the counter, which hosts expose to user space.
        unsafe { asm!("isb", "mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack)) };
        v
    }

    /// CNTFRQ_EL0, in ticks a second.
    pub fn frequency() -> u64 {
        let v: u64;
        // SAFETY: as for `now`.
        unsafe { asm!("mrs {}, cntfrq_el0", out(reg) v, options(nomem, nostack)) };
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes `ret`-returning code that loads a constant and runs it.
    fn runs_code(region: &JitRegion) {
        // movz x0, #0x1234; movk x0, #0x5678, lsl #16; ret
        let code = [0xd282_4680u32, 0xf2aa_cf00, 0xd65f_03c0];
        region.writer().write_words(64, &code);
        region.flush(64, 12);
        // SAFETY: the words above are a complete function under the C ABI.
        let f: extern "C" fn() -> u64 =
            unsafe { std::mem::transmute(region.exec_base().wrapping_add(64)) };
        assert_eq!(f(), 0x5678_1234);
        region.writer().patch_word(64, 0xd282_4600); // movz x0, #0x1230
        region.flush(64, 4);
        assert_eq!(f(), 0x5678_1230);
    }

    #[test]
    fn runs_what_it_wrote() {
        let region = JitRegion::new(1 << 20).unwrap();
        assert!(region.len() >= 1 << 20);
        runs_code(&region);
    }

    #[test]
    fn runs_what_it_wrote_through_another_alias() {
        let region = JitRegion::new_dual(1 << 20).unwrap();
        assert!(region.is_aliased());
        runs_code(&region);
    }

    #[test]
    fn slices_run_code_of_their_own() {
        for region in [JitRegion::new(4 << 20), JitRegion::new_dual(4 << 20)] {
            let slices = Arc::new(region.unwrap()).split(3, 1 << 20).unwrap();
            assert_eq!(slices.len(), 3);
            for pair in slices.windows(2) {
                assert_eq!(pair[0].len(), 1 << 20, "capped at max_each");
                assert!(
                    pair[0].exec_base() as usize + pair[0].len() <= pair[1].exec_base() as usize
                );
            }
            for slice in &slices {
                runs_code(slice);
            }
        }
        let small = Arc::new(JitRegion::new_dual(page_size()).unwrap());
        assert!(small.split(2, usize::MAX).is_err(), "less than a page each");
    }

    #[test]
    fn the_counter_moves() {
        let a = counter::now();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = counter::now();
        assert!(counter::frequency() > 0);
        assert!(b > a);
    }
}
