// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The guest CPU's state as the translated code sees it, and the register
//! convention between guest and host.
//!
//! While translated guest code runs, the guest's own registers live in the
//! host's registers of the same number: guest X0..X30 in host X0..X30, the
//! guest's NZCV flags in the host's, the guest V0..V31 in the host's. So a
//! guest instruction that only moves data between registers is copied into the
//! host block unchanged.
//!
//! The host's stack pointer does not hold the guest's. It points instead at
//! this block, so translated code can spill a guest register to it with a
//! plain `STP Xn, Xm, [SP, #off]` without first needing a scratch register,
//! and can read or move SP with `ADD Xn, SP, #0`. The guest's own SP lives in
//! the `sp` field here and every guest access to it is rewritten. A real host
//! stack, for the moments translated code calls back into Rust, is kept in
//! `host_sp`.
//!
//! The field offsets are part of the contract with the code the assembler
//! emits; `offsets` names them and the tests check them against the struct.

use std::sync::atomic::{AtomicU32, Ordering};

/// The number of entries in each direct-mapped software TLB. A power of two.
pub const TLB_ENTRIES: usize = 256;

/// One software-TLB entry: a page's translation, kept so the common load or
/// store needs no page-table walk. `tag` is the guest page with the low bit
/// set to mark the entry valid (an all-zero entry is therefore a miss, and a
/// guest VA of 0 does not accidentally hit). `delta` added to the guest
/// virtual address gives the host address of the same byte.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct TlbEntry {
    pub tag: u64,
    pub delta: u64,
}

impl TlbEntry {
    pub const VALID: u64 = 1;

    pub fn index(va: u64, page_shift: u32) -> usize {
        ((va >> page_shift) as usize) & (TLB_ENTRIES - 1)
    }

    pub fn tag_for(va: u64, page_shift: u32) -> u64 {
        ((va >> page_shift) << page_shift) | Self::VALID
    }
}

/// A direct-mapped TLB for one kind of access.
#[repr(C)]
pub struct Tlb {
    pub entries: [TlbEntry; TLB_ENTRIES],
}

impl Default for Tlb {
    fn default() -> Self {
        Tlb {
            entries: [TlbEntry::default(); TLB_ENTRIES],
        }
    }
}

impl Tlb {
    pub fn clear(&mut self) {
        self.entries = [TlbEntry::default(); TLB_ENTRIES];
    }
}

/// The guest's exception levels have their own stack pointers and a few
/// banked registers; EL0 and EL1 are all the translator runs.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Banked {
    pub sp_el0: u64,
    pub sp_el1: u64,
    pub elr_el1: u64,
    pub spsr_el1: u64,
    pub esr_el1: u64,
    pub far_el1: u64,
    pub vbar_el1: u64,
    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
    pub tpidr_el1: u64,
    /// The generic timer's control and compare values (virtual and physical).
    pub cntv_ctl: u64,
    pub cntv_cval: u64,
    pub cntp_ctl: u64,
    pub cntp_cval: u64,
    pub cntkctl_el1: u64,
    /// Registers the guest writes that the engine keeps but does not act on:
    /// CPACR (FP is always on here), the debug monitor control, and ACTLR.
    pub cpacr_el1: u64,
    pub mdscr_el1: u64,
    pub actlr_el1: u64,
}

/// The translation and memory system registers the MMU reads.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct MmuRegs {
    pub sctlr_el1: u64,
    pub ttbr0_el1: u64,
    pub ttbr1_el1: u64,
    pub tcr_el1: u64,
    pub mair_el1: u64,
    pub amair_el1: u64,
    pub contextidr_el1: u64,
}

/// Why a block of translated code returned to the dispatcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Exit {
    /// The next guest PC is in `next_pc`; find or make its block.
    Branch = 0,
    /// A guest exception is pending (its cause is in the exception fields);
    /// the dispatcher enters the guest's handler.
    Exception = 1,
    /// A system register the dispatcher must handle (MRS/MSR not inlined).
    Sysreg = 2,
    /// WFI or WFE: wait, then continue at `next_pc`.
    WaitForInterrupt = 3,
    WaitForEvent = 4,
    /// The guest executed an instruction the translator does not know.
    Undefined = 5,
    /// A service call (SVC/HVC/SMC); the kind is in `exception_kind`.
    Call = 6,
}

impl Exit {
    pub fn from_u32(v: u32) -> Option<Exit> {
        Some(match v {
            0 => Exit::Branch,
            1 => Exit::Exception,
            2 => Exit::Sysreg,
            3 => Exit::WaitForInterrupt,
            4 => Exit::WaitForEvent,
            5 => Exit::Undefined,
            6 => Exit::Call,
            _ => return None,
        })
    }
}

/// The state block. Host SP points here while translated code runs.
///
/// `#[repr(C)]` and the field order are the contract with the emitted code;
/// the `offsets` module names every offset the assembler uses and a test
/// checks them.
#[repr(C, align(16))]
pub struct CpuState {
    /// Guest X0..X30. Index 31 is unused (the encoding's register 31 is the
    /// zero register or SP, never X31).
    pub x: [u64; 32],
    /// Guest SP (the EL the guest runs at selects SP_EL0 or SP_EL1; this is
    /// the one in use).
    pub sp: u64,
    /// Guest PC where a block resumes or faulted.
    pub pc: u64,
    /// Guest NZCV in bits 31:28, as the host's NZCV holds them.
    pub nzcv: u64,
    /// The rest of PSTATE the translator tracks (EL in bits 3:2, SPSel bit 0,
    /// DAIF in bits 9:6).
    pub pstate: u64,
    /// Guest V0..V31, 16 bytes each.
    pub v: [[u8; 16]; 32],
    pub fpcr: u64,
    pub fpsr: u64,

    pub banked: Banked,
    pub mmu: MmuRegs,

    // --- the dispatcher's own fields, below what guest state the block holds ---
    /// Where a block puts its exit code before returning.
    pub exit: u32,
    /// Padding to keep `next_pc` 8-aligned.
    pub _pad0: u32,
    /// The guest PC to resume at on a Branch/WFI/Call exit.
    pub next_pc: u64,
    /// For an Exception exit: the cause the dispatcher turns into ESR_EL1.
    pub exception_kind: u64,
    pub exception_syndrome: u64,
    pub exception_far: u64,
    /// For a Sysreg exit: the packed SysReg and the Rt, and whether a read.
    pub sysreg: u32,
    pub sysreg_rt: u32,
    pub sysreg_read: u32,
    pub _pad1: u32,

    /// The real host stack top, restored into SP before calling back to Rust.
    pub host_sp: u64,
    /// This block's own address, so a helper can be handed it.
    pub self_ptr: u64,
    /// The `Cpu` object (Rust side) a helper acts on.
    pub cpu_ptr: u64,
    /// The host address `enter` branches to: a translated block's first word.
    pub dispatch_target: u64,
    /// The host address the exit stub returns through (`crate_jit_epilogue`).
    pub epilogue: u64,

    /// Scratch the memory fast path spills victim registers to.
    pub scratch: [u64; 4],
    /// The page shift in force (12, 14 or 16), cached from TCR for the fast
    /// path's TLB index.
    pub page_shift: u64,

    /// Read-side TLB (loads) and write-side TLB (stores), each direct-mapped.
    pub load_tlb: Tlb,
    pub store_tlb: Tlb,
}

impl Default for CpuState {
    fn default() -> Self {
        // SAFETY: every field is plain data for which all-zero is valid.
        unsafe { std::mem::zeroed() }
    }
}

impl CpuState {
    pub fn new() -> Box<Self> {
        Box::new(Self::default())
    }

    pub fn clear_tlbs(&mut self) {
        self.load_tlb.clear();
        self.store_tlb.clear();
    }
}

/// Byte offsets into `CpuState`, the contract with emitted code.
pub mod offsets {
    use super::*;
    use std::mem::offset_of;

    pub const X: u32 = offset_of!(CpuState, x) as u32;
    pub const SP: u32 = offset_of!(CpuState, sp) as u32;
    pub const PC: u32 = offset_of!(CpuState, pc) as u32;
    pub const NZCV: u32 = offset_of!(CpuState, nzcv) as u32;
    pub const PSTATE: u32 = offset_of!(CpuState, pstate) as u32;
    pub const V: u32 = offset_of!(CpuState, v) as u32;
    pub const FPCR: u32 = offset_of!(CpuState, fpcr) as u32;
    pub const FPSR: u32 = offset_of!(CpuState, fpsr) as u32;
    pub const EXIT: u32 = offset_of!(CpuState, exit) as u32;
    pub const NEXT_PC: u32 = offset_of!(CpuState, next_pc) as u32;
    pub const EXCEPTION_KIND: u32 = offset_of!(CpuState, exception_kind) as u32;
    pub const EXCEPTION_SYNDROME: u32 = offset_of!(CpuState, exception_syndrome) as u32;
    pub const EXCEPTION_FAR: u32 = offset_of!(CpuState, exception_far) as u32;
    pub const SYSREG: u32 = offset_of!(CpuState, sysreg) as u32;
    pub const SYSREG_RT: u32 = offset_of!(CpuState, sysreg_rt) as u32;
    pub const SYSREG_READ: u32 = offset_of!(CpuState, sysreg_read) as u32;
    pub const HOST_SP: u32 = offset_of!(CpuState, host_sp) as u32;
    pub const SELF_PTR: u32 = offset_of!(CpuState, self_ptr) as u32;
    pub const CPU_PTR: u32 = offset_of!(CpuState, cpu_ptr) as u32;
    pub const DISPATCH_TARGET: u32 = offset_of!(CpuState, dispatch_target) as u32;
    pub const EPILOGUE: u32 = offset_of!(CpuState, epilogue) as u32;
    pub const SCRATCH: u32 = offset_of!(CpuState, scratch) as u32;
    pub const PAGE_SHIFT: u32 = offset_of!(CpuState, page_shift) as u32;
    pub const LOAD_TLB: u32 = offset_of!(CpuState, load_tlb) as u32;
    pub const STORE_TLB: u32 = offset_of!(CpuState, store_tlb) as u32;

    /// The offset of guest register `n`'s spill slot.
    pub fn x(n: u32) -> u32 {
        X + n * 8
    }

    /// The offset of guest V register `n`'s spill slot.
    pub fn v(n: u32) -> u32 {
        V + n * 16
    }
}

/// A count of translated blocks, for the backend's statistics and tests.
#[derive(Default)]
pub struct Stats {
    pub blocks: AtomicU32,
}

impl Stats {
    pub fn translated(&self) {
        self.blocks.fetch_add(1, Ordering::Relaxed);
    }
    pub fn block_count(&self) -> u32 {
        self.blocks.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_reachable_by_unsigned_loads() {
        // Everything emitted code reaches with LDR/STR (unsigned offset) must
        // fit that form: a multiple of 8 under 32 KiB.
        for off in [
            offsets::X,
            offsets::SP,
            offsets::PC,
            offsets::NZCV,
            offsets::FPCR,
            offsets::HOST_SP,
            offsets::SELF_PTR,
            offsets::CPU_PTR,
            offsets::SCRATCH,
            offsets::PAGE_SHIFT,
            offsets::x(30),
        ] {
            assert_eq!(off % 8, 0, "offset {off} not 8-aligned");
            assert!(off < 32 * 1024, "offset {off} out of LDR range");
        }
        assert_eq!(offsets::x(0), offsets::X);
        assert_eq!(offsets::x(1), offsets::X + 8);
        assert_eq!(offsets::v(1), offsets::V + 16);
    }

    #[test]
    fn exit_codes_round_trip() {
        for e in [
            Exit::Branch,
            Exit::Exception,
            Exit::Sysreg,
            Exit::WaitForInterrupt,
            Exit::Undefined,
            Exit::Call,
        ] {
            assert_eq!(Exit::from_u32(e as u32), Some(e));
        }
        assert_eq!(Exit::from_u32(99), None);
    }

    #[test]
    fn tlb_tagging() {
        assert_eq!(TlbEntry::tag_for(0x1234, 12), 0x1000 | 1);
        assert_ne!(TlbEntry::tag_for(0, 12), 0, "a zero VA still tags valid");
        assert_eq!(TlbEntry::index(0x5_0000, 12) & !(TLB_ENTRIES - 1), 0);
    }
}
