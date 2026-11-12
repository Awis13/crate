// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The memory slow path: what a load or store does when translated code's
//! inline fast path cannot do it (a software-TLB miss, a device, an access
//! that crosses a page, a misaligned exclusive), or when the fast paths are
//! off. It works on the state's copies of the guest registers, which the
//! shared stub has saved, performs the whole access, and fills the TLB entry
//! so the next access to the page stays inline.
//!
//! Every access to RAM here is one the architecture would make: aligned
//! accesses are single host accesses of their size, and the exclusive,
//! acquire/release, compare-and-swap and atomic forms are the host's own
//! atomics on the host address, so this path is as correct with several CPUs
//! as the inline one.

use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering, fence};

use crate::backend::{Cpu, WATCHPOINT_KIND};
use crate::decode::{self, Mem, MemKind, Op};
use crate::mmu::{self, Access};
use crate::state::{
    CpuState, Exit, MONITOR_CLEAR, MmuRegs, TLB_SHIFT, TlbEntry, TlbKind, tlb_index,
};

/// The helper the memory stub calls: performs the access the site's
/// instruction describes. Returns 0 on success and 1 on a fault (with the
/// exception fields and the exit code set).
///
/// # Safety
///
/// `cpu` is the `*mut Cpu` the state names and `site` a translated call
/// site's literal (word, return offset, offset in the page); both are valid
/// while the block runs.
pub(crate) unsafe extern "C" fn jit_mem(
    cpu: *mut Cpu,
    site: *const u8,
    _state: *mut CpuState,
) -> u64 {
    // SAFETY: the dispatcher owns the Cpu for the block's whole run, and the
    // site literal is 16 bytes of the block's code.
    let (cpu, word, off) = unsafe {
        (
            &mut *cpu,
            std::ptr::read_unaligned(site as *const u32),
            std::ptr::read_unaligned(site.add(8) as *const u64),
        )
    };
    let pc = cpu.state.page_va.wrapping_add(off);
    cpu.state.pc = pc;
    cpu.counters.slow_mem += 1;
    match cpu.memory_access(word, pc) {
        Ok(()) => 0,
        Err(fault) => {
            cpu.counters.slow_fault += 1;
            // A watchpoint has already recorded its own exception kind.
            if cpu.state.exception_kind != WATCHPOINT_KIND {
                cpu.take_data_abort(fault);
            }
            cpu.state.exit = Exit::Exception as u32;
            1
        }
    }
}

/// An alignment fault (DFSC 0b100001).
fn alignment_fault(write: bool) -> mmu::Fault {
    mmu::Fault {
        fsc: mmu::Fault::ALIGNMENT,
        write,
    }
}

fn external_abort(write: bool) -> mmu::Fault {
    mmu::Fault { fsc: 0x10, write }
}

fn mask(size: u32) -> u64 {
    if size >= 3 {
        u64::MAX
    } else {
        (1u64 << (8 << size)) - 1
    }
}

fn sign_extend(value: u64, bits: u32) -> u64 {
    let shift = 64 - bits;
    (((value << shift) as i64) >> shift) as u64
}

/// Whether an `n`-byte access at `addr` runs past the end of its page, so it
/// must be split and each page translated on its own: consecutive guest pages
/// need not map to consecutive physical pages.
pub(crate) fn crosses_page(addr: u64, n: u64, page_shift: u32) -> bool {
    let page = 1u64 << page_shift;
    (addr & (page - 1)) + n > page
}

/// Applies an extend-and-shift (the option field of an extended register) to a
/// 64-bit value, as a register-offset address calculation does.
pub(crate) fn extend(value: u64, option: u32, amount: u32) -> u64 {
    let extended = match option {
        0b000 => value & 0xff,                         // UXTB
        0b001 => value & 0xffff,                       // UXTH
        0b010 => value & 0xffff_ffff,                  // UXTW
        0b011 => value,                                // UXTX / LSL
        0b100 => sign_extend(value, 8),                // SXTB
        0b101 => sign_extend(value, 16),               // SXTH
        0b110 => sign_extend(value & 0xffff_ffff, 32), // SXTW
        _ => value,                                    // SXTX
    };
    extended << amount
}

/// Extends a freshly loaded value to the destination register width.
fn extend_loaded(m: &Mem, raw: u64) -> u64 {
    let mut value = raw;
    if m.signed {
        value = sign_extend(value, 8 << m.size);
    }
    if !m.reg64 {
        value &= 0xffff_ffff;
    }
    value
}

/// The signed maximum or minimum of two values at a given access size, for the
/// LDSMAX/LDSMIN atomics.
fn signed_minmax(a: u64, b: u64, size: u32, max: bool) -> u64 {
    let bits = 8 << size;
    let sa = sign_extend(a, bits) as i64;
    let sb = sign_extend(b, bits) as i64;
    (if max { sa.max(sb) } else { sa.min(sb) }) as u64
}

/// A host atomic of `size` (log2 bytes) at `ptr`, as one operation on u64
/// values.
///
/// # Safety
///
/// `ptr` is aligned to the size and valid for its whole span.
unsafe fn atomic_rmw(ptr: *mut u8, size: u32, f: impl Fn(u64) -> u64) -> u64 {
    macro_rules! rmw {
        ($t:ty, $ut:ty) => {{
            // SAFETY: the caller's contract.
            let a = unsafe { <$t>::from_ptr(ptr.cast()) };
            let mut old = a.load(Ordering::Relaxed);
            loop {
                let new = f(old as u64) as $ut;
                match a.compare_exchange_weak(old, new, Ordering::SeqCst, Ordering::Relaxed) {
                    Ok(_) => return old as u64,
                    Err(cur) => old = cur,
                }
            }
        }};
    }
    match size {
        0 => rmw!(AtomicU8, u8),
        1 => rmw!(AtomicU16, u16),
        2 => rmw!(AtomicU32, u32),
        _ => rmw!(AtomicU64, u64),
    }
}

/// A host compare-and-swap of `size`; returns the old value.
///
/// # Safety
///
/// As for `atomic_rmw`.
unsafe fn atomic_cas(ptr: *mut u8, size: u32, expected: u64, new: u64) -> u64 {
    macro_rules! cas {
        ($t:ty, $ut:ty) => {{
            // SAFETY: the caller's contract.
            let a = unsafe { <$t>::from_ptr(ptr.cast()) };
            match a.compare_exchange(
                expected as $ut,
                new as $ut,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(v) | Err(v) => v as u64,
            }
        }};
    }
    match size {
        0 => cas!(AtomicU8, u8),
        1 => cas!(AtomicU16, u16),
        2 => cas!(AtomicU32, u32),
        _ => cas!(AtomicU64, u64),
    }
}

/// # Safety
///
/// As for `atomic_rmw`.
unsafe fn atomic_load(ptr: *mut u8, size: u32) -> u64 {
    // SAFETY: the caller's contract.
    unsafe {
        match size {
            0 => AtomicU8::from_ptr(ptr.cast()).load(Ordering::SeqCst) as u64,
            1 => AtomicU16::from_ptr(ptr.cast()).load(Ordering::SeqCst) as u64,
            2 => AtomicU32::from_ptr(ptr.cast()).load(Ordering::SeqCst) as u64,
            _ => AtomicU64::from_ptr(ptr.cast()).load(Ordering::SeqCst),
        }
    }
}

/// # Safety
///
/// As for `atomic_rmw`.
unsafe fn atomic_store(ptr: *mut u8, size: u32, value: u64) {
    // SAFETY: the caller's contract.
    unsafe {
        match size {
            0 => AtomicU8::from_ptr(ptr.cast()).store(value as u8, Ordering::SeqCst),
            1 => AtomicU16::from_ptr(ptr.cast()).store(value as u16, Ordering::SeqCst),
            2 => AtomicU32::from_ptr(ptr.cast()).store(value as u32, Ordering::SeqCst),
            _ => AtomicU64::from_ptr(ptr.cast()).store(value, Ordering::SeqCst),
        }
    }
}

/// A 128-bit compare-and-swap (the host's CASPAL); returns the old pair.
///
/// # Safety
///
/// `ptr` is 16-byte aligned and valid for 16 bytes.
// Every host the translator runs on has LSE (it copies the guest's LSE atomics
// as they are), but the iOS target's baseline CPU predates it.
#[target_feature(enable = "lse")]
unsafe fn cas128(ptr: *mut u8, expected: (u64, u64), new: (u64, u64)) -> (u64, u64) {
    let (mut lo, mut hi) = expected;
    // SAFETY: the caller's contract; CASP wants even/odd register pairs, which
    // the explicit registers give it.
    unsafe {
        std::arch::asm!(
            "caspal x4, x5, x6, x7, [{p}]",
            p = in(reg) ptr,
            inout("x4") lo,
            inout("x5") hi,
            in("x6") new.0,
            in("x7") new.1,
            options(nostack),
        );
    }
    (lo, hi)
}

impl Cpu {
    /// The guest register `n`, reading SP (31) from the state's SP.
    pub(crate) fn xread(&self, n: u32) -> u64 {
        if n == 31 {
            self.state.sp
        } else {
            self.state.x[n as usize]
        }
    }

    pub(crate) fn xwrite(&mut self, n: u32, value: u64) {
        if n == 31 {
            self.state.sp = value;
        } else {
            self.state.x[n as usize] = value;
        }
    }

    /// Reads a general register where 31 means the zero register.
    pub(crate) fn xread_z(&self, n: u32) -> u64 {
        if n == 31 { 0 } else { self.state.x[n as usize] }
    }

    pub(crate) fn xwrite_z(&mut self, n: u32, value: u64) {
        if n != 31 {
            self.state.x[n as usize] = value;
        }
    }

    /// Performs the load or store `word` (at guest `pc`) describes, from the
    /// current state.
    pub fn memory_access(&mut self, word: u32, pc: u64) -> Result<(), mmu::Fault> {
        match decode::decode(word, pc) {
            Op::Memory(m) => self.access(&m, None),
            Op::LiteralLoad {
                rt,
                addr,
                size,
                signed,
                vector,
            } => {
                let mut m =
                    match decode::decode(literal_as_offset_load(rt, size, signed, vector), 0) {
                        Op::Memory(m) => m,
                        _ => unreachable!(),
                    };
                m.rn = 31;
                self.access(&m, Some(addr))
            }
            Op::DcZva { rt } => self.dc_zva(rt),
            other => unreachable!("memory helper on {word:#x} ({other:?})"),
        }
    }

    fn access(&mut self, m: &Mem, literal: Option<u64>) -> Result<(), mmu::Fault> {
        let el = self.el();
        let regs = self.state.mmu;
        let acc_el = if m.unprivileged { 0 } else { el };
        let base = literal.unwrap_or_else(|| self.xread(m.rn));
        let addr = match (literal, m.index_reg) {
            (Some(a), _) => a,
            (None, Some((rm, option, amount))) => {
                base.wrapping_add(extend(self.xread_z(rm), option, amount))
            }
            (None, None) if m.pre => base.wrapping_add(m.offset as u64),
            _ => base, // post-index: the access is at the base
        };
        let rmw = matches!(
            m.kind,
            MemKind::Cas | MemKind::CasPair | MemKind::Atomic | MemKind::StoreExclusive
        );
        let write = !m.load || rmw;
        self.state.exception_far = addr;

        // A watchpoint takes precedence over the access, which must not happen.
        if self.debug_active() {
            self.watchpoint(addr, m.total as u64, write)?;
        }
        if m.align > 1 && addr % m.align as u64 != 0 {
            return Err(alignment_fault(write));
        }

        match m.kind {
            MemKind::Gpr => {
                let n = 1u64 << m.size;
                if m.load {
                    let raw = self.load(addr, n, acc_el, &regs)?;
                    self.xwrite_z(m.rt, extend_loaded(m, raw));
                } else {
                    let value = self.xread_z(m.rt);
                    self.store(addr, n, value, acc_el, &regs)?;
                }
            }
            MemKind::Pair => {
                // Both halves are translated before either is done, so a
                // fault on the second leaves the registers as they were.
                let n = 1u64 << m.size;
                let second = addr.wrapping_add(n);
                let acc = if m.load { Access::Read } else { Access::Write };
                self.state.exception_far = addr;
                mmu::translate_cached(&self.mem, &regs, acc_el, addr, acc, &mut self.walk)?;
                self.state.exception_far = second;
                mmu::translate_cached(&self.mem, &regs, acc_el, second, acc, &mut self.walk)?;
                if m.load {
                    let a = self.load(addr, n, acc_el, &regs)?;
                    let b = self.load(second, n, acc_el, &regs)?;
                    self.xwrite_z(m.rt, extend_loaded(m, a));
                    self.xwrite_z(m.rt2, extend_loaded(m, b));
                } else {
                    let (a, b) = (self.xread_z(m.rt), self.xread_z(m.rt2));
                    self.store(addr, n, a, acc_el, &regs)?;
                    self.store(second, n, b, acc_el, &regs)?;
                }
            }
            MemKind::Vec | MemKind::VecPair => {
                let bytes = 1usize << m.size;
                let count = if m.kind == MemKind::VecPair { 2 } else { 1 };
                let total = bytes * count;
                let mut buf = [0u8; 32];
                if m.load {
                    self.gather(addr, &mut buf[..total], acc_el, &regs)?;
                    for (i, rt) in [m.rt, m.rt2].into_iter().take(count).enumerate() {
                        // The upper lanes past `bytes` read as zero.
                        let mut reg = [0u8; 16];
                        reg[..bytes].copy_from_slice(&buf[i * bytes..(i + 1) * bytes]);
                        self.state.v[rt as usize] = reg;
                    }
                } else {
                    for (i, rt) in [m.rt, m.rt2].into_iter().take(count).enumerate() {
                        buf[i * bytes..(i + 1) * bytes]
                            .copy_from_slice(&self.state.v[rt as usize][..bytes]);
                    }
                    self.scatter(addr, &buf[..total], acc_el, &regs)?;
                }
                self.fill_after(addr, total as u64, acc_el, &regs, !m.load);
            }
            MemKind::VecStruct => {
                self.vec_struct(m, addr, acc_el, &regs)?;
                self.fill_after(addr, m.total as u64, acc_el, &regs, !m.load);
            }
            MemKind::LoadExclusive
            | MemKind::StoreExclusive
            | MemKind::LoadAcquire
            | MemKind::StoreRelease
            | MemKind::Cas
            | MemKind::CasPair
            | MemKind::Atomic => self.ordered(m, addr, el, &regs, write)?,
        }

        if m.writeback {
            let new = if m.pre {
                addr
            } else if let Some(rm) = m.wb_reg {
                base.wrapping_add(self.xread_z(rm))
            } else {
                base.wrapping_add(m.offset as u64)
            };
            self.xwrite(m.rn, new);
        }
        Ok(())
    }

    /// A plain load of `n` bytes (1 to 8) at guest `addr`.
    fn load(&mut self, addr: u64, n: u64, el: u32, regs: &MmuRegs) -> Result<u64, mmu::Fault> {
        self.state.exception_far = addr;
        let t = mmu::translate_cached(&self.mem, regs, el, addr, Access::Read, &mut self.walk)?;
        if crosses_page(addr, n, TLB_SHIFT) {
            self.counters.slow_cross += 1;
            let mut buf = [0u8; 8];
            self.gather(addr, &mut buf[..n as usize], el, regs)?;
            return Ok(u64::from_le_bytes(buf));
        }
        if let Ok(ptr) = self.mem.host_range(t.pa, n) {
            // A page it may also write serves its stores from now on too.
            let kind = if t.writable {
                TlbKind::Store
            } else {
                TlbKind::Load
            };
            self.tlb_fill(el, kind, addr, &t);
            // SAFETY: host_range checked the span; an aligned access of its
            // size is one host access.
            return Ok(unsafe { read_host(ptr, n) });
        }
        self.read_device(t.pa, n, false)
    }

    /// A plain store of the low `n` bytes of `value` at guest `addr`.
    fn store(
        &mut self,
        addr: u64,
        n: u64,
        value: u64,
        el: u32,
        regs: &MmuRegs,
    ) -> Result<(), mmu::Fault> {
        self.state.exception_far = addr;
        let t = mmu::translate_cached(&self.mem, regs, el, addr, Access::Write, &mut self.walk)?;
        if crosses_page(addr, n, TLB_SHIFT) {
            self.counters.slow_cross += 1;
            return self.scatter(addr, &value.to_le_bytes()[..n as usize], el, regs);
        }
        if let Ok(ptr) = self.mem.host_range(t.pa, n) {
            self.tlb_fill(el, TlbKind::Store, addr, &t);
            // SAFETY: as in `load`.
            unsafe { write_host(ptr, n, value) };
            return Ok(());
        }
        self.write_device(t.pa, n, value)
    }

    /// Not RAM: a device, if the board has one here. The bus takes a byte
    /// count, not the log2 size the access carries. Where no device answers,
    /// a read is zero and a write is ignored, as on hvf, which runs the same
    /// machine. Without a board (the bare test runner) it is an external
    /// abort.
    fn read_device(&mut self, pa: u64, n: u64, write: bool) -> Result<u64, mmu::Fault> {
        self.counters.slow_device += 1;
        match self.board_opt() {
            Some(board) => Ok(board.mmio.read(pa, n as u8).unwrap_or_else(|| {
                crate_core::debug!("reads of {pa:#x}, where nothing is, read as zero");
                0
            })),
            None => Err(external_abort(write)),
        }
    }

    fn write_device(&mut self, pa: u64, n: u64, value: u64) -> Result<(), mmu::Fault> {
        self.counters.slow_device += 1;
        match self.board_opt() {
            Some(board) => {
                if !board.mmio.write(pa, n as u8, value) {
                    crate_core::debug!("writes to {pa:#x}, where nothing is, ignored");
                }
                Ok(())
            }
            None => Err(external_abort(true)),
        }
    }

    /// Reads `buf.len()` bytes at guest VA `addr`, re-walking the page tables at
    /// every page boundary so an access that crosses into a physically
    /// separate page reads each page's bytes from its own translation.
    pub(crate) fn gather(
        &mut self,
        addr: u64,
        buf: &mut [u8],
        el: u32,
        regs: &MmuRegs,
    ) -> Result<(), mmu::Fault> {
        let mut done = 0usize;
        while done < buf.len() {
            let va = addr.wrapping_add(done as u64);
            self.state.exception_far = va;
            let t = mmu::translate_cached(&self.mem, regs, el, va, Access::Read, &mut self.walk)?;
            let page = 1u64 << TLB_SHIFT;
            let chunk = ((page - (va & (page - 1))) as usize).min(buf.len() - done);
            if self.mem.read(t.pa, &mut buf[done..done + chunk]).is_err() {
                for k in 0..chunk {
                    buf[done + k] = self.read_device(t.pa + k as u64, 1, false)? as u8;
                }
            }
            done += chunk;
        }
        Ok(())
    }

    /// Writes `bytes` at guest VA `addr`, re-walking the page tables at every
    /// page boundary; the write counterpart of `gather`.
    pub(crate) fn scatter(
        &mut self,
        addr: u64,
        bytes: &[u8],
        el: u32,
        regs: &MmuRegs,
    ) -> Result<(), mmu::Fault> {
        // Every page is checked before any byte is written, so a fault on a
        // later page leaves memory unchanged.
        let mut done = 0usize;
        let page = 1u64 << TLB_SHIFT;
        while done < bytes.len() {
            let va = addr.wrapping_add(done as u64);
            self.state.exception_far = va;
            mmu::translate_cached(&self.mem, regs, el, va, Access::Write, &mut self.walk)?;
            done += ((page - (va & (page - 1))) as usize).min(bytes.len() - done);
        }
        let mut done = 0usize;
        while done < bytes.len() {
            let va = addr.wrapping_add(done as u64);
            let t = mmu::translate_cached(&self.mem, regs, el, va, Access::Write, &mut self.walk)?;
            let chunk = ((page - (va & (page - 1))) as usize).min(bytes.len() - done);
            if self.mem.write(t.pa, &bytes[done..done + chunk]).is_err() {
                for k in 0..chunk {
                    self.write_device(t.pa + k as u64, 1, bytes[done + k] as u64)?;
                }
            }
            done += chunk;
        }
        Ok(())
    }

    /// LD1..LD4 and ST1..ST4 (Arm ARM, "LD1 (multiple structures)" and the
    /// single-structure forms), on the state's V registers.
    fn vec_struct(
        &mut self,
        m: &Mem,
        addr: u64,
        el: u32,
        regs: &MmuRegs,
    ) -> Result<(), mmu::Fault> {
        let w = m.word;
        let q = w & (1 << 30) != 0;
        let t = w & 31;
        let datasize = if q { 16 } else { 8 };
        let total = m.total as usize;
        let mut buf = [0u8; 64];
        if m.load {
            self.gather(addr, &mut buf[..total], el, regs)?;
        }
        let mut offs = 0usize;
        if w & (1 << 24) == 0 {
            // Multiple structures.
            let size = (w >> 10) & 3;
            let (regs_n, selem) = match (w >> 12) & 0xf {
                0b0000 => (4, 4),
                0b0010 => (4, 1),
                0b0100 => (3, 3),
                0b0110 => (3, 1),
                0b0111 => (1, 1),
                0b1000 => (2, 2),
                _ => (2, 1),
            };
            let ebytes = 1usize << size;
            let elements = datasize / ebytes;
            let rpt = regs_n / selem;
            if m.load {
                // A load writes whole registers: the upper half clears for Q=0.
                for r in 0..regs_n {
                    self.state.v[((t + r as u32) % 32) as usize] = [0; 16];
                }
            }
            for r in 0..rpt {
                for e in 0..elements {
                    let mut tt = ((t + r as u32) % 32) as usize;
                    for _ in 0..selem {
                        let lane = e * ebytes;
                        if m.load {
                            self.state.v[tt][lane..lane + ebytes]
                                .copy_from_slice(&buf[offs..offs + ebytes]);
                        } else {
                            buf[offs..offs + ebytes]
                                .copy_from_slice(&self.state.v[tt][lane..lane + ebytes]);
                        }
                        offs += ebytes;
                        tt = (tt + 1) % 32;
                    }
                }
            }
        } else {
            // A single structure: one lane, or one element replicated.
            let size = (w >> 10) & 3;
            let s = (w >> 12) & 1;
            let opcode = (w >> 13) & 7;
            let r = (w >> 21) & 1;
            let selem = ((((opcode & 1) << 1) | r) + 1) as usize;
            let mut scale = opcode >> 1;
            let replicate = scale == 3;
            let qv = q as u32;
            let index = match scale {
                3 => {
                    scale = size;
                    0
                }
                0 => (qv << 3) | (s << 2) | size,
                1 => (qv << 2) | (s << 1) | (size >> 1),
                _ => {
                    if size & 1 == 0 {
                        (qv << 1) | s
                    } else {
                        scale = 3;
                        qv
                    }
                }
            } as usize;
            let ebytes = 1usize << scale;
            let mut tt = t as usize;
            for _ in 0..selem {
                if replicate {
                    let mut reg = [0u8; 16];
                    for lane in (0..datasize).step_by(ebytes) {
                        reg[lane..lane + ebytes].copy_from_slice(&buf[offs..offs + ebytes]);
                    }
                    self.state.v[tt] = reg;
                } else {
                    let lane = index * ebytes;
                    if m.load {
                        self.state.v[tt][lane..lane + ebytes]
                            .copy_from_slice(&buf[offs..offs + ebytes]);
                    } else {
                        buf[offs..offs + ebytes]
                            .copy_from_slice(&self.state.v[tt][lane..lane + ebytes]);
                    }
                }
                offs += ebytes;
                tt = (tt + 1) % 32;
            }
        }
        if !m.load {
            self.scatter(addr, &buf[..total], el, regs)?;
        }
        Ok(())
    }

    /// The exclusive, acquire/release, compare-and-swap and atomic forms. On
    /// RAM each is the host's own atomic; on a device (which the architecture
    /// does not promise atomics on) a plain read and write.
    fn ordered(
        &mut self,
        m: &Mem,
        addr: u64,
        el: u32,
        regs: &MmuRegs,
        write: bool,
    ) -> Result<(), mmu::Fault> {
        let acc = if write { Access::Write } else { Access::Read };
        let t = mmu::translate_cached(&self.mem, regs, el, addr, acc, &mut self.walk)?;
        let n = m.total as u64;
        let ram = self.mem.host_range(t.pa, n).ok();
        if ram.is_some() {
            let kind = if write { TlbKind::Store } else { TlbKind::Load };
            self.tlb_fill(el, kind, addr, &t);
        }
        let size = m.size;
        let pair =
            m.rt2 != 31 && matches!(m.kind, MemKind::LoadExclusive | MemKind::StoreExclusive);
        // Reads and writes `size`-sized values at an offset into the access.
        let read_at = |cpu: &mut Cpu, off: u64, sz: u32| -> Result<u64, mmu::Fault> {
            match ram {
                // SAFETY: inside the span host_range checked, aligned (the
                // alignment check passed).
                Some(p) => Ok(unsafe { atomic_load(p.add(off as usize), sz) }),
                None => cpu.read_device(t.pa + off, 1 << sz, write),
            }
        };
        match m.kind {
            MemKind::LoadAcquire => {
                let v = read_at(self, 0, size)?;
                fence(Ordering::Acquire);
                self.xwrite_z(m.rt, v);
            }
            MemKind::StoreRelease => {
                fence(Ordering::Release);
                let v = self.xread_z(m.rt) & mask(size);
                match ram {
                    // SAFETY: as for read_at.
                    Some(p) => unsafe { atomic_store(p, size, v) },
                    None => self.write_device(t.pa, 1 << size, v)?,
                }
            }
            MemKind::LoadExclusive => {
                let a = read_at(self, 0, size)?;
                let b = if pair {
                    read_at(self, 1 << size, size)?
                } else {
                    0
                };
                if m.acquire {
                    fence(Ordering::Acquire);
                }
                self.xwrite_z(m.rt, a);
                if pair {
                    self.xwrite_z(m.rt2, b);
                }
                self.state.monitor = addr;
                self.state.monitor_value = a;
                self.state.monitor_value2 = b;
            }
            MemKind::StoreExclusive => {
                if m.release {
                    fence(Ordering::Release);
                }
                let reserved = self.state.monitor == addr && addr != MONITOR_CLEAR;
                self.state.monitor = MONITOR_CLEAR;
                let (v0, v1) = (
                    self.xread_z(m.rt) & mask(size),
                    self.xread_z(m.rt2) & mask(size),
                );
                let (e0, e1) = (self.state.monitor_value, self.state.monitor_value2);
                let ok = reserved
                    && match ram {
                        // SAFETY: as for read_at; a pair is aligned to its total.
                        Some(p) => unsafe {
                            if !pair {
                                atomic_cas(p, size, e0, v0) == e0 & mask(size)
                            } else if size == 2 {
                                let e = e0 | e1 << 32;
                                atomic_cas(p, 3, e, v0 | v1 << 32) == e
                            } else {
                                cas128(p, (e0, e1), (v0, v1)) == (e0, e1)
                            }
                        },
                        None => {
                            self.write_device(t.pa, 1 << size, v0)?;
                            if pair {
                                self.write_device(t.pa + (1 << size), 1 << size, v1)?;
                            }
                            true
                        }
                    };
                self.xwrite_z(m.rs, (!ok) as u64);
            }
            MemKind::Cas => {
                let (expected, new) = (
                    self.xread_z(m.rs) & mask(size),
                    self.xread_z(m.rt) & mask(size),
                );
                let old = match ram {
                    // SAFETY: as for read_at.
                    Some(p) => unsafe { atomic_cas(p, size, expected, new) },
                    None => {
                        let old = self.read_device(t.pa, 1 << size, true)? & mask(size);
                        if old == expected {
                            self.write_device(t.pa, 1 << size, new)?;
                        }
                        old
                    }
                };
                self.xwrite_z(m.rs, old);
            }
            MemKind::CasPair => {
                let rs = m.rs;
                let rt = m.rt;
                let e = (
                    self.xread_z(rs) & mask(size),
                    self.xread_z(rs + 1) & mask(size),
                );
                let v = (
                    self.xread_z(rt) & mask(size),
                    self.xread_z(rt + 1) & mask(size),
                );
                let old = match ram {
                    // SAFETY: as for read_at.
                    Some(p) => unsafe {
                        if size == 2 {
                            let o = atomic_cas(p, 3, e.0 | e.1 << 32, v.0 | v.1 << 32);
                            (o & 0xffff_ffff, o >> 32)
                        } else {
                            cas128(p, e, v)
                        }
                    },
                    None => return Err(external_abort(true)),
                };
                self.xwrite_z(rs, old.0);
                self.xwrite_z(rs + 1, old.1);
            }
            MemKind::Atomic => {
                let w = m.word;
                let o3 = (w >> 15) & 1;
                let opc = (w >> 12) & 7;
                let msk = mask(size);
                let rs_val = self.xread_z(m.rs) & msk;
                let op = move |old: u64| -> u64 {
                    let old = old & msk;
                    (if o3 == 1 {
                        rs_val // SWP
                    } else {
                        match opc {
                            0b000 => old.wrapping_add(rs_val),                // LDADD
                            0b001 => old & !rs_val,                           // LDCLR
                            0b010 => old ^ rs_val,                            // LDEOR
                            0b011 => old | rs_val,                            // LDSET
                            0b100 => signed_minmax(old, rs_val, size, true),  // LDSMAX
                            0b101 => signed_minmax(old, rs_val, size, false), // LDSMIN
                            0b110 => old.max(rs_val),                         // LDUMAX
                            _ => old.min(rs_val),                             // LDUMIN
                        }
                    }) & msk
                };
                let old = match ram {
                    // SAFETY: as for read_at.
                    Some(p) => unsafe { atomic_rmw(p, size, op) },
                    None => {
                        let old = self.read_device(t.pa, 1 << size, true)? & msk;
                        self.write_device(t.pa, 1 << size, op(old))?;
                        old
                    }
                };
                // Rt 31 is the store form (ST<op>), which returns nothing.
                self.xwrite_z(m.rt, old & msk);
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// DC ZVA: zero the DCZID_EL0-sized block (64 bytes) holding Rt's VA. The
    /// guest zeroes memory this way (clear_page, memset), so a recycled page
    /// it clears must actually read back zero.
    fn dc_zva(&mut self, rt: u32) -> Result<(), mmu::Fault> {
        let va = self.xread_z(rt) & !63;
        let el = self.el();
        let regs = self.state.mmu;
        if self.debug_active() {
            self.watchpoint(va, 64, true)?;
        }
        self.state.exception_far = va;
        let t = mmu::translate_cached(&self.mem, &regs, el, va, Access::Write, &mut self.walk)?;
        if self.mem.fill(t.pa, 64, 0).is_ok() {
            self.tlb_fill(el, TlbKind::Store, va, &t);
            return Ok(());
        }
        for off in (0..64u64).step_by(8) {
            self.write_device(t.pa + off, 8, 0)?;
        }
        Ok(())
    }

    /// Fills the TLB for an access the vector paths did with `gather` or
    /// `scatter`, if it stayed in one page.
    fn fill_after(&mut self, addr: u64, n: u64, el: u32, regs: &MmuRegs, write: bool) {
        if crosses_page(addr, n, TLB_SHIFT) {
            return;
        }
        let acc = if write { Access::Write } else { Access::Read };
        if let Ok(t) = mmu::translate_cached(&self.mem, regs, el, addr, acc, &mut self.walk) {
            let kind = if write { TlbKind::Store } else { TlbKind::Load };
            self.tlb_fill(el, kind, addr, &t);
        }
    }

    /// Records the translation of the guest page holding `va` in the software
    /// TLB for `el` and `kind` (a store's translation also serves loads), if
    /// the page is RAM and the fast paths can use it: every 4 KiB piece of it
    /// at once, so a 16 or 64 KiB page misses once, not once a piece.
    pub(crate) fn tlb_fill(&mut self, el: u32, kind: TlbKind, va: u64, t: &mmu::Translation) {
        if self.debug_active() {
            return; // every access goes through here to meet the watchpoints
        }
        let page = (1u64 << t.page_shift.max(TLB_SHIFT)) - 1;
        let kinds: &[TlbKind] = if kind == TlbKind::Store {
            &[TlbKind::Store, TlbKind::Load]
        } else {
            &[TlbKind::Load]
        };
        for piece in (0..=page).step_by(1 << TLB_SHIFT) {
            let (piece_va, piece_pa) = ((va & !page) + piece, (t.pa & !page) + piece);
            let Ok(host) = self.mem.host_range(piece_pa, 1 << TLB_SHIFT) else {
                return;
            };
            let delta = (host as u64).wrapping_sub(piece_va);
            for &k in kinds {
                let ti = tlb_index(el, k);
                let i = TlbEntry::index(ti, piece_va);
                self.state.tlb[ti].entries[i] = TlbEntry {
                    tag: TlbEntry::tag_for(piece_va),
                    delta,
                };
                self.tlbs.filled(ti, i, piece_va, t.size_shift, t.global);
            }
        }
    }
}

/// The literal form of a load as the equivalent unsigned-offset load, so it
/// decodes to the same `Mem` (with the base then replaced by the address).
fn literal_as_offset_load(rt: u32, size: u32, signed: bool, vector: bool) -> u32 {
    let (size_field, opc) = match (vector, size, signed) {
        (false, 2, false) => (2, 0b01),
        (false, 3, _) => (3, 0b01),
        (false, _, true) => (2, 0b10),
        (true, 2, _) => (2, 0b01),
        (true, 3, _) => (3, 0b01),
        _ => (0, 0b11), // Q
    };
    size_field << 30 | 0x3900_0000 | (vector as u32) << 26 | opc << 22 | rt
}

/// One host load of `n` bytes (1, 2, 4 or 8), as one access when aligned.
///
/// # Safety
///
/// `ptr` is valid for `n` bytes.
unsafe fn read_host(ptr: *mut u8, n: u64) -> u64 {
    // SAFETY: the caller's contract; unaligned spans fall back to bytes.
    unsafe {
        if ptr as u64 % n == 0 {
            match n {
                1 => std::ptr::read_volatile(ptr) as u64,
                2 => std::ptr::read_volatile(ptr.cast::<u16>()) as u64,
                4 => std::ptr::read_volatile(ptr.cast::<u32>()) as u64,
                _ => std::ptr::read_volatile(ptr.cast::<u64>()),
            }
        } else {
            let mut buf = [0u8; 8];
            std::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), n as usize);
            u64::from_le_bytes(buf)
        }
    }
}

/// One host store of the low `n` bytes of `value`.
///
/// # Safety
///
/// `ptr` is valid for `n` bytes.
unsafe fn write_host(ptr: *mut u8, n: u64, value: u64) {
    // SAFETY: the caller's contract.
    unsafe {
        if ptr as u64 % n == 0 {
            match n {
                1 => std::ptr::write_volatile(ptr, value as u8),
                2 => std::ptr::write_volatile(ptr.cast::<u16>(), value as u16),
                4 => std::ptr::write_volatile(ptr.cast::<u32>(), value as u32),
                _ => std::ptr::write_volatile(ptr.cast::<u64>(), value),
            }
        } else {
            std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), ptr, n as usize);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extend_matches_the_architecture() {
        assert_eq!(extend(0x1_0000_00ff, 0b010, 2), 0xff << 2); // UXTW #2
        assert_eq!(
            extend(0xffff_ffff_ffff_ff80, 0b100, 0),
            0xffff_ffff_ffff_ff80
        ); // SXTB
        assert_eq!(extend(0x8000_0000, 0b110, 0), 0xffff_ffff_8000_0000); // SXTW
    }

    #[test]
    fn crosses_page_is_exact() {
        // A whole page's worth that ends exactly on the boundary does not cross.
        assert!(!crosses_page(0x1000, 0x1000, 12));
        assert!(!crosses_page(0x1ff8, 8, 12)); // ends at 0x2000
        assert!(crosses_page(0x1ff9, 8, 12)); // one byte over
        assert!(crosses_page(0x1ffc, 8, 12));
        assert!(!crosses_page(0x1ffc, 4, 12)); // ends at 0x2000
        assert!(crosses_page(0x3ffc, 8, 14)); // 16 KiB granule, still crosses
    }

    #[test]
    fn literal_forms_decode_as_loads() {
        for (size, signed, vector) in [
            (2, false, false),
            (3, false, false),
            (2, true, false),
            (2, false, true),
            (3, false, true),
            (4, false, true),
        ] {
            match decode::decode(literal_as_offset_load(5, size, signed, vector), 0) {
                Op::Memory(m) => {
                    assert!(
                        m.load && m.size == size && m.signed == signed,
                        "{size} {signed} {vector}"
                    );
                    assert_eq!(m.kind == MemKind::Vec, vector);
                }
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn cas128_swaps_only_on_a_match() {
        #[repr(align(16))]
        struct A([u64; 2]);
        let mut a = A([1, 2]);
        let p = a.0.as_mut_ptr().cast::<u8>();
        // SAFETY: a local, aligned 16-byte buffer.
        unsafe {
            assert_eq!(cas128(p, (1, 2), (3, 4)), (1, 2));
            assert_eq!(cas128(p, (1, 2), (5, 6)), (3, 4));
        }
        assert_eq!(a.0, [3, 4]);
    }
}
