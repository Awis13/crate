// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The guest CPU's state as the translated code sees it, and the register
//! convention between guest and host.
//!
//! While translated guest code runs, the guest's own registers live in the
//! host's registers of the same number: guest X0..X30 in host X0..X30, the
//! guest's NZCV flags in the host's, the guest V0..V31 in the host's, and the
//! guest's FPCR and FPSR in the host's. So a guest instruction that only moves
//! data between registers is copied into the host block unchanged. The
//! exceptions are X18, the platform register, which the host OS reserves and
//! clobbers: guest X18 lives in `x[18]` here, like SP; and X15..X17, which
//! live in `x[15..18]` wherever one block of code hands over to another, so
//! the host registers are scratch for the stubs, the jump cache and the
//! memory fast path. A block loads one of them into its host register when
//! an instruction names it and stores it back before it leaves.
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
//! Their order matters too: the registers and the scratch slots sit where a
//! pair store reaches them (within 504 bytes), the vector registers where a
//! pair of Q registers reaches them (within 1008), and every table on a page
//! boundary of the block.

use std::sync::atomic::{AtomicU32, Ordering};

/// The software TLB granule: every TLB entry covers 4 KiB, whatever the
/// guest's page size, so one entry never claims more than one host page and a
/// guest's 16 or 64 KiB page simply takes several entries.
pub const TLB_SHIFT: u32 = 12;

/// Entries in each direct-mapped software TLB (a power of two): 32 MiB of
/// 4 KiB pages each.
pub const TLB_BITS: u32 = 15;
pub const TLB_ENTRIES: usize = 1 << TLB_BITS;

/// Entries in each jump cache (a power of two).
pub const JC_BITS: u32 = 16;

/// The jump cache's index folds higher PC bits into the low ones (shifted
/// down this far), so code a power of two apart does not always collide.
pub const JC_FOLD: u32 = 16;
pub const JC_ENTRIES: usize = 1 << JC_BITS;

/// The tag of an empty TLB entry. A tag is a guest page number, which never
/// reaches all ones (virtual addresses have at most 56 significant bits).
pub const TLB_INVALID: u64 = u64::MAX;

/// The exclusive monitor's address when nothing is reserved. Guest address 0
/// is never RAM in the machines crate builds, so it cannot be a real
/// reservation, and clearing the monitor is a store of the zero register.
pub const MONITOR_CLEAR: u64 = 0;

/// One software-TLB entry: a 4 KiB guest page's translation, kept so the
/// common load or store needs no page-table walk. `tag` is the guest page
/// number (`va >> TLB_SHIFT`); `delta` added to a guest virtual address in the
/// page gives the host address of the same byte.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct TlbEntry {
    pub tag: u64,
    pub delta: u64,
}

impl Default for TlbEntry {
    fn default() -> Self {
        TlbEntry {
            tag: TLB_INVALID,
            delta: 0,
        }
    }
}

impl TlbEntry {
    /// The entry of TLB `tlb` (see `tlb_index`) for `va`. The table that ends
    /// at SP holds its entries in reverse order, which is how code that
    /// subtracts the index from SP reaches them.
    pub fn index(tlb: usize, va: u64) -> usize {
        let i = ((va >> TLB_SHIFT) as usize) & (TLB_ENTRIES - 1);
        if tlb == TLB_NEAR {
            i ^ (TLB_ENTRIES - 1)
        } else {
            i
        }
    }

    pub fn tag_for(va: u64) -> u64 {
        va >> TLB_SHIFT
    }
}

/// A direct-mapped TLB for one kind of access at one exception level. Page
/// aligned, so translated code reaches an entry with a `SUB` of the table's
/// distance below SP in pages and a pair load.
#[repr(C, align(4096))]
pub struct Tlb {
    pub entries: [TlbEntry; TLB_ENTRIES],
}

/// One jump-cache entry: the guest PC a translated block starts at, and the
/// host address translated code branches to for it. An empty entry's target
/// is the jump cache's own miss path, so a stale tag can never send code
/// anywhere but back to the dispatcher.
#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
pub struct JcEntry {
    pub tag: u64,
    pub host: u64,
}

impl JcEntry {
    pub fn index(pc: u64) -> usize {
        (((pc ^ (pc >> JC_FOLD)) >> 2) as usize) & (JC_ENTRIES - 1)
    }
}

#[repr(C, align(4096))]
pub struct JumpCache {
    pub entries: [JcEntry; JC_ENTRIES],
}

/// Which software TLB an access uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TlbKind {
    Load = 0,
    Store = 1,
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
    /// The result of the last address-translation (AT) instruction.
    pub par_el1: u64,
}

/// The self-hosted debug registers: hardware breakpoints and watchpoints, and
/// the OS lock. Linux and the debug tests program these through MSR/MRS.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Debug {
    /// Breakpoint value and control, DBGBVR0..5 / DBGBCR0..5.
    pub bvr: [u64; 6],
    pub bcr: [u64; 6],
    /// Watchpoint value and control, DBGWVR0..3 / DBGWCR0..3.
    pub wvr: [u64; 4],
    pub wcr: [u64; 4],
    /// OSLAR_EL1.OSLK: the OS lock, which Linux unlocks at boot.
    pub oslock: u64,
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
/// checks the ranges the emitted instructions need.
#[repr(C, align(16))]
pub struct CpuState {
    /// The software TLBs, indexed by `tlb_index(el, kind)`. Translated code
    /// has the address of `x` in SP, so they are below it, and the last, the
    /// one for EL0 loads, ends right there: code reaches its entry for an
    /// index with one instruction (`SUB Xd, SP, Xi, LSL #4`) rather than two.
    pub tlb: [Tlb; 4],
    /// Guest X0..X30. Index 31 is unused (the encoding's register 31 is the
    /// zero register or SP, never X31). Translated code has their address in
    /// SP, and every offset in `offsets` is from there.
    pub x: [u64; 32],
    /// Where the memory fast path parks the host registers it borrows.
    pub scratch: [u64; 4],
    /// Guest SP (the EL the guest runs at selects SP_EL0 or SP_EL1; this is
    /// the one in use).
    pub sp: u64,
    /// Guest PC where a block resumes or faulted.
    pub pc: u64,
    /// Guest NZCV in bits 31:28, as the host's NZCV holds them.
    pub nzcv: u64,
    /// The rest of PSTATE the translator tracks (EL in bits 3:2, SPSel bit 0,
    /// DAIF in bits 9:6, SS in bit 21).
    pub pstate: u64,

    /// Set by any thread that needs this CPU back in its dispatcher soon (an
    /// interrupt, a timer, another CPU's TLB or code invalidation, a stop).
    /// Translated code tests it at backward branches, jump-cache lookups and
    /// ISB; the dispatcher clears it.
    pub exit_request: u32,
    /// Set when another CPU invalidated code this CPU may have translated:
    /// it must look at its queue at its next ISB (with `exit_request`, which
    /// is next to it, in one load) or dispatcher entry.
    pub sync_request: u32,
    /// Set when an interrupt may be pending for this CPU; cleared only by the
    /// dispatcher when the interrupt controller says nothing is. Inline code
    /// that unmasks interrupts tests it.
    pub irq_hint: u32,
    /// Bumped by the interrupt controller's waker each time it signals this
    /// CPU, so the dispatcher can tell a new interrupt from one still masked.
    pub irq_wakes: u32,
    /// Where a block puts its exit code before returning.
    pub exit: u32,
    pub _pad0: u32,
    /// The guest PC to resume at on a Branch/WFI/Call exit.
    pub next_pc: u64,
    /// The guest virtual address of the 4 KiB page the running block's code
    /// is in. A translation depends only on the code's physical page; the
    /// guest addresses it computes (PC-relative values, link values, branch
    /// targets) are this plus an offset. Set by whoever enters a block from
    /// outside its page: the dispatcher, and the jump cache.
    pub page_va: u64,
    /// For a Branch exit from a direct branch whose target is in the same
    /// page: the host address of the branch to patch once the target block
    /// exists (block chaining). Zero otherwise.
    pub chain_site: u64,
    /// The memory slow path's call site (its literal: the instruction word
    /// and the guest PC's offset in its page).
    pub slow_site: u64,
    /// The exclusive monitor: the address a load-exclusive reserved
    /// (`MONITOR_CLEAR` for none) and the value(s) it read. A store-exclusive
    /// succeeds only if the reservation is for its address and memory still
    /// holds the value, which it checks and writes in one compare-and-swap.
    pub monitor: u64,
    pub monitor_value: u64,
    pub monitor_value2: u64,

    /// Guest V0..V31, 16 bytes each.
    pub v: [[u8; 16]; 32],
    /// Unused: the guest's FPCR and FPSR stay in the host's registers of the
    /// CPU's thread (reading and writing them stalls the host, so the
    /// trampoline does neither).
    pub fpcr: u64,
    pub fpsr: u64,

    // --- the dispatcher's own fields ---
    /// For an Exception exit: the cause the dispatcher turns into ESR_EL1.
    pub exception_kind: u64,
    pub exception_syndrome: u64,
    pub exception_far: u64,
    /// For a Sysreg exit: the packed SysReg and the Rt, and whether a read.
    pub sysreg: u32,
    pub sysreg_rt: u32,
    pub sysreg_read: u32,
    /// Set when the guest invalidated the instruction cache or changed what
    /// translations depend on; the dispatcher acts on it after the exit.
    pub flush_cache: u32,

    /// How many jump-cache lookups missed, and how many there were (with
    /// CRATE_JIT_STATS=1) (statistics).
    pub jc_misses: u64,
    pub jc_lookups: u64,
    /// XORed into an EL0 jump-cache entry's tag: the context number of the
    /// address space in its top 16 bits (user addresses have none there).
    pub jc_context: u64,
    /// The real host stack top, restored into SP before calling back to Rust.
    pub host_sp: u64,
    /// This block's own address, so a helper can be handed it.
    pub self_ptr: u64,
    /// The `Cpu` object (Rust side) a helper acts on.
    pub cpu_ptr: u64,
    /// The host address `enter` branches to: a translated block's entry.
    pub dispatch_target: u64,
    /// The host address the exit stubs return through (`crate_jit_epilogue`).
    pub epilogue: u64,
    /// Where the memory slow path returns to: the instruction after the
    /// site's call.
    pub slow_ret: u64,

    pub banked: Banked,
    pub mmu: MmuRegs,
    pub debug: Debug,

    /// The jump caches for EL0 and EL1 code.
    pub jc: [JumpCache; 2],
}

/// Which of `CpuState::tlb` an access at `el` of `kind` uses.
pub fn tlb_index(el: u32, kind: TlbKind) -> usize {
    match (el, kind) {
        (0, TlbKind::Load) => TLB_NEAR,
        (0, TlbKind::Store) => 2,
        (_, TlbKind::Load) => 0,
        (_, TlbKind::Store) => 1,
    }
}

/// The TLB that ends where translated code's SP points: EL0's loads, the
/// most frequent accesses of the programs a machine runs.
pub const TLB_NEAR: usize = 3;

impl CpuState {
    pub fn new() -> Box<Self> {
        let layout = std::alloc::Layout::new::<Self>();
        // SAFETY: the layout is the type's own, non-zero in size; every field
        // is plain data for which all-zero is valid, and the TLB tags and the
        // monitor are then set to their empty values.
        let mut s: Box<Self> = unsafe {
            let p = std::alloc::alloc_zeroed(layout).cast::<Self>();
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(p)
        };
        for t in s.tlb.iter_mut() {
            t.entries.fill(TlbEntry::default());
        }
        s.monitor = MONITOR_CLEAR;
        s
    }

    /// The exit-request word as the atomic other threads write.
    pub fn exit_request(&self) -> &AtomicU32 {
        // SAFETY: the field is a properly aligned u32 that lives as long as
        // the state; translated code only reads it.
        unsafe { AtomicU32::from_ptr(&self.exit_request as *const u32 as *mut u32) }
    }

    pub fn sync_request(&self) -> &AtomicU32 {
        // SAFETY: as for exit_request.
        unsafe { AtomicU32::from_ptr(&self.sync_request as *const u32 as *mut u32) }
    }

    pub fn irq_wakes(&self) -> &AtomicU32 {
        // SAFETY: as for exit_request.
        unsafe { AtomicU32::from_ptr(&self.irq_wakes as *const u32 as *mut u32) }
    }

    pub fn irq_hint(&self) -> &AtomicU32 {
        // SAFETY: as for exit_request.
        unsafe { AtomicU32::from_ptr(&self.irq_hint as *const u32 as *mut u32) }
    }

    /// Asks the CPU to come back to its dispatcher.
    pub fn kick(&self) {
        self.exit_request().store(1, Ordering::Release);
    }
}

// Pair loads and stores of Q registers need `v` 16-byte aligned.
const _: () = assert!(offsets::V % 16 == 0);

/// Byte offsets into `CpuState`, the contract with emitted code.
pub mod offsets {
    use super::*;
    use std::mem::offset_of;

    /// Where translated code's SP points in the state: at `x`.
    pub const ANCHOR: usize = offset_of!(CpuState, x);

    /// The offset from SP of a field at `offset` in the state.
    const fn at(offset: usize) -> u32 {
        (offset - ANCHOR) as u32
    }

    pub const X: u32 = at(offset_of!(CpuState, x));
    pub const SCRATCH: u32 = at(offset_of!(CpuState, scratch));
    pub const SP: u32 = at(offset_of!(CpuState, sp));
    pub const PC: u32 = at(offset_of!(CpuState, pc));
    pub const NZCV: u32 = at(offset_of!(CpuState, nzcv));
    pub const PSTATE: u32 = at(offset_of!(CpuState, pstate));
    pub const EXIT_REQUEST: u32 = at(offset_of!(CpuState, exit_request));
    pub const IRQ_HINT: u32 = at(offset_of!(CpuState, irq_hint));
    pub const EXIT: u32 = at(offset_of!(CpuState, exit));
    pub const NEXT_PC: u32 = at(offset_of!(CpuState, next_pc));
    pub const PAGE_VA: u32 = at(offset_of!(CpuState, page_va));
    pub const CHAIN_SITE: u32 = at(offset_of!(CpuState, chain_site));
    pub const SLOW_SITE: u32 = at(offset_of!(CpuState, slow_site));
    pub const SLOW_RET: u32 = at(offset_of!(CpuState, slow_ret));
    pub const MONITOR: u32 = at(offset_of!(CpuState, monitor));
    pub const MONITOR_VALUE: u32 = at(offset_of!(CpuState, monitor_value));
    pub const MONITOR_VALUE2: u32 = at(offset_of!(CpuState, monitor_value2));
    pub const V: u32 = at(offset_of!(CpuState, v));
    pub const FPCR: u32 = at(offset_of!(CpuState, fpcr));
    pub const FPSR: u32 = at(offset_of!(CpuState, fpsr));
    pub const EXCEPTION_KIND: u32 = at(offset_of!(CpuState, exception_kind));
    pub const EXCEPTION_SYNDROME: u32 = at(offset_of!(CpuState, exception_syndrome));
    pub const EXCEPTION_FAR: u32 = at(offset_of!(CpuState, exception_far));
    pub const SYSREG: u32 = at(offset_of!(CpuState, sysreg));
    pub const SYSREG_RT: u32 = at(offset_of!(CpuState, sysreg_rt));
    pub const SYSREG_READ: u32 = at(offset_of!(CpuState, sysreg_read));
    pub const HOST_SP: u32 = at(offset_of!(CpuState, host_sp));
    pub const JC_MISSES: u32 = at(offset_of!(CpuState, jc_misses));
    pub const JC_LOOKUPS: u32 = at(offset_of!(CpuState, jc_lookups));
    pub const JC_CONTEXT: u32 = at(offset_of!(CpuState, jc_context));
    pub const SELF_PTR: u32 = at(offset_of!(CpuState, self_ptr));
    pub const CPU_PTR: u32 = at(offset_of!(CpuState, cpu_ptr));
    pub const DISPATCH_TARGET: u32 = at(offset_of!(CpuState, dispatch_target));
    pub const EPILOGUE: u32 = at(offset_of!(CpuState, epilogue));
    pub const BANKED: u32 = at(offset_of!(CpuState, banked));
    pub const TPIDR_EL0: u32 = BANKED + offset_of!(Banked, tpidr_el0) as u32;
    pub const TPIDRRO_EL0: u32 = BANKED + offset_of!(Banked, tpidrro_el0) as u32;
    pub const TPIDR_EL1: u32 = BANKED + offset_of!(Banked, tpidr_el1) as u32;
    pub const SP_EL0: u32 = BANKED + offset_of!(Banked, sp_el0) as u32;
    pub const MMU: u32 = at(offset_of!(CpuState, mmu));

    /// The offset of a banked register, from its offset in `Banked`.
    pub const fn banked(off: usize) -> u32 {
        BANKED + off as u32
    }
    pub const JC: u32 = at(offset_of!(CpuState, jc));

    /// The offset of guest register `n`'s spill slot.
    pub const fn x(n: u32) -> u32 {
        X + n * 8
    }

    /// The offset of guest V register `n`'s spill slot.
    pub const fn v(n: u32) -> u32 {
        V + n * 16
    }

    /// The offset of scratch slot `n`.
    pub const fn scratch(n: u32) -> u32 {
        SCRATCH + n * 8
    }

    /// How far below SP TLB `index` (see `tlb_index`) starts.
    pub const fn tlb_below(index: usize) -> u32 {
        (ANCHOR - offset_of!(CpuState, tlb) - index * std::mem::size_of::<Tlb>()) as u32
    }

    /// The offset of the first entry of the jump cache for `el`.
    pub const fn jc(el: u32) -> u32 {
        JC + (if el == 0 { 0 } else { 1 } * std::mem::size_of::<JumpCache>()) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_reachable_by_the_emitted_forms() {
        // Plain loads and stores (unsigned offset, scaled by 8) reach 32 KiB.
        for off in [
            offsets::X,
            offsets::SP,
            offsets::PC,
            offsets::NZCV,
            offsets::PSTATE,
            offsets::NEXT_PC,
            offsets::PAGE_VA,
            offsets::CHAIN_SITE,
            offsets::SLOW_SITE,
            offsets::SLOW_RET,
            offsets::MONITOR,
            offsets::MONITOR_VALUE,
            offsets::FPCR,
            offsets::FPSR,
            offsets::HOST_SP,
            offsets::CPU_PTR,
            offsets::EPILOGUE,
            offsets::TPIDR_EL0,
            offsets::TPIDR_EL1,
            offsets::SP_EL0,
            offsets::x(30),
        ] {
            assert_eq!(off % 8, 0, "offset {off} not 8-aligned");
            assert!(off < 32 * 1024, "offset {off} out of LDR range");
        }
        // Word fields other threads write and code tests with LDR W, and the
        // two requests ISB tests with one LDR X.
        assert_eq!(offsets::EXIT_REQUEST % 8, 0);
        assert_eq!(
            (std::mem::offset_of!(CpuState, sync_request) - offsets::ANCHOR) as u32,
            offsets::EXIT_REQUEST + 4
        );
        assert_eq!(offsets::IRQ_HINT % 4, 0);
        // Pairs of X registers and the scratch slots: STP reaches 504.
        assert!(offsets::x(29) <= 504 && offsets::scratch(2) <= 504);
        // Pairs of Q registers reach 1008.
        assert_eq!(offsets::V % 16, 0);
        assert!(offsets::v(30) <= 1008);
        // Every table starts on a page of the block: code subtracts the
        // pages; the near one ends at SP.
        for i in 0..4 {
            assert_eq!(offsets::tlb_below(i) & 0xfff, 0);
        }
        assert_eq!(
            offsets::tlb_below(TLB_NEAR) as usize,
            std::mem::size_of::<Tlb>()
        );
        for el in 0..2 {
            assert_eq!(offsets::jc(el) & 0xfff, 0);
        }
        assert_eq!(std::mem::size_of::<TlbEntry>(), 16);
        assert_eq!(std::mem::size_of::<JcEntry>(), 16);
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
        assert_eq!(TlbEntry::tag_for(0x1234), 1);
        assert_ne!(
            TlbEntry::tag_for(0),
            TLB_INVALID,
            "a zero VA still tags valid"
        );
        assert_eq!(TlbEntry::index(0, 0x5_0000) & !(TLB_ENTRIES - 1), 0);
        assert_eq!(TlbEntry::index(TLB_NEAR, 0), TLB_ENTRIES - 1);
        let s = CpuState::new();
        assert!(
            s.tlb
                .iter()
                .all(|t| t.entries.iter().all(|e| e.tag == TLB_INVALID))
        );
        assert_eq!(s.monitor, MONITOR_CLEAR);
    }
}
