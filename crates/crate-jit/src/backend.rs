// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The dispatcher and the EL1 machine: the trampoline that enters and leaves
//! translated code, finding (or making) the block for the guest PC, acting on
//! why a block returned, and everything translated code leaves to Rust --
//! exceptions, system registers, the generic timer, self-hosted debug, TLB and
//! cache maintenance, PSCI -- and the CPU backend that runs one host thread
//! per guest CPU.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate_core::board::Board;
use crate_core::memory::GuestMemory;
use crate_host::jit::JitRegion;

pub use crate::cache::CodeCache;
use crate::cache::{BlockKey, FastHasher};
use crate::mmu::{self, Access};
use crate::smp::{Remote, Smp};
use crate::state::{
    CpuState, Exit, JC_ENTRIES, JcEntry, MONITOR_CLEAR, MmuRegs, TLB_ENTRIES, TLB_INVALID,
    TlbEntry, offsets,
};
use crate::translate::{Ctx, ERET_MARKER, FETCH_ABORT, SYSTEM_WORD_MARKER};

// The host ABI functions that enter a block and return through its exit.
unsafe extern "C" {
    fn crate_jit_enter(state: *mut CpuState) -> u64;
    /// Only its address is used (stored in `CpuState.epilogue`).
    fn crate_jit_epilogue();
}

core::arch::global_asm!(
    ".p2align 2",
    ".globl _crate_jit_enter",
    "_crate_jit_enter:",
    // Save the host's callee-saved registers and make a frame.
    "stp x29, x30, [sp, #-176]!",
    "stp x19, x20, [sp, #16]",
    "stp x21, x22, [sp, #32]",
    "stp x23, x24, [sp, #48]",
    "stp x25, x26, [sp, #64]",
    "stp x27, x28, [sp, #80]",
    "stp d8,  d9,  [sp, #96]",
    "stp d10, d11, [sp, #112]",
    "stp d12, d13, [sp, #128]",
    "stp d14, d15, [sp, #144]",
    // Remember the host stack, then make SP the address of the state's
    // registers (`offsets::ANCHOR` into it).
    "mov x9, sp",
    "add sp, x0, #{anchor}, lsl #12",
    "str x9, [sp, #{host_sp}]",
    // Load the guest's flags and SIMD registers. The guest's FPCR and FPSR
    // stay in the host's registers between runs (the dispatcher does no
    // floating point, and reading or writing them stalls the pipeline), so
    // they are neither loaded here nor saved on the way out.
    "ldr x9, [sp, #{nzcv}]",
    "msr nzcv, x9",
    "ldp q0,  q1,  [sp, #({v}+0)]",
    "ldp q2,  q3,  [sp, #({v}+32)]",
    "ldp q4,  q5,  [sp, #({v}+64)]",
    "ldp q6,  q7,  [sp, #({v}+96)]",
    "ldp q8,  q9,  [sp, #({v}+128)]",
    "ldp q10, q11, [sp, #({v}+160)]",
    "ldp q12, q13, [sp, #({v}+192)]",
    "ldp q14, q15, [sp, #({v}+224)]",
    "ldp q16, q17, [sp, #({v}+256)]",
    "ldp q18, q19, [sp, #({v}+288)]",
    "ldp q20, q21, [sp, #({v}+320)]",
    "ldp q22, q23, [sp, #({v}+352)]",
    "ldp q24, q25, [sp, #({v}+384)]",
    "ldp q26, q27, [sp, #({v}+416)]",
    "ldp q28, q29, [sp, #({v}+448)]",
    "ldp q30, q31, [sp, #({v}+480)]",
    // Load guest X0..X30 but X15..X17 (a block takes them from their slots
    // when it needs them) and X18 (virtual: it lives in the state).
    "ldp x0,  x1,  [sp, #({x}+0)]",
    "ldp x2,  x3,  [sp, #({x}+16)]",
    "ldp x4,  x5,  [sp, #({x}+32)]",
    "ldp x6,  x7,  [sp, #({x}+48)]",
    "ldp x8,  x9,  [sp, #({x}+64)]",
    "ldp x10, x11, [sp, #({x}+80)]",
    "ldp x12, x13, [sp, #({x}+96)]",
    "ldr x14, [sp, #({x}+112)]",
    "ldr x19, [sp, #({x}+152)]",
    "ldp x20, x21, [sp, #({x}+160)]",
    "ldp x22, x23, [sp, #({x}+176)]",
    "ldp x24, x25, [sp, #({x}+192)]",
    "ldp x26, x27, [sp, #({x}+208)]",
    "ldp x28, x29, [sp, #({x}+224)]",
    "ldr x30, [sp, #({x}+240)]",
    // Jump to the block.
    "ldr x16, [sp, #{dispatch}]",
    "br x16",
    ".globl _crate_jit_epilogue",
    "_crate_jit_epilogue:",
    // Reached from a stub with SP = state block; return the exit code.
    "ldr w0, [sp, #{exit}]",
    "ldr x9, [sp, #{host_sp}]",
    "mov sp, x9",
    "ldp d14, d15, [sp, #144]",
    "ldp d12, d13, [sp, #128]",
    "ldp d10, d11, [sp, #112]",
    "ldp d8,  d9,  [sp, #96]",
    "ldp x27, x28, [sp, #80]",
    "ldp x25, x26, [sp, #64]",
    "ldp x23, x24, [sp, #48]",
    "ldp x21, x22, [sp, #32]",
    "ldp x19, x20, [sp, #16]",
    "ldp x29, x30, [sp], #176",
    "ret",
    anchor = const offsets::ANCHOR >> 12,
    host_sp = const offsets::HOST_SP,
    nzcv = const offsets::NZCV,
    v = const offsets::V,
    x = const offsets::X,
    dispatch = const offsets::DISPATCH_TARGET,
    exit = const offsets::EXIT,
);

// The guest's FPCR and FPSR live in the host's registers of this thread (see
// the trampoline); the dispatcher reads and writes them there.
fn fpcr() -> u64 {
    let v: u64;
    // SAFETY: reads a register.
    unsafe { std::arch::asm!("mrs {}, fpcr", out(reg) v, options(nomem, nostack)) };
    v
}
fn fpsr() -> u64 {
    let v: u64;
    // SAFETY: reads a register.
    unsafe { std::arch::asm!("mrs {}, fpsr", out(reg) v, options(nomem, nostack)) };
    v
}
fn set_fpcr(v: u64) {
    // SAFETY: the guest's FPCR, which the dispatcher's own code does not rely on.
    unsafe { std::arch::asm!("msr fpcr, {}", in(reg) v, options(nomem, nostack)) };
}
fn set_fpsr(v: u64) {
    // SAFETY: as for set_fpcr.
    unsafe { std::arch::asm!("msr fpsr, {}", in(reg) v, options(nomem, nostack)) };
}

fn epilogue_addr() -> u64 {
    let f: unsafe extern "C" fn() = crate_jit_epilogue;
    f as usize as u64
}

impl CodeCache {
    /// A cache of `bytes` of executable memory whose fast paths are on unless
    /// the environment says `CRATE_JIT_SLOW=1`.
    pub fn new(bytes: usize) -> crate_core::Result<Self> {
        Self::with_mode(bytes, !slow_from_env())
    }

    /// A cache with the fast paths on or off.
    pub fn with_mode(bytes: usize, fast: bool) -> crate_core::Result<Self> {
        let helper: unsafe extern "C" fn(*mut Cpu, *const u8, *mut CpuState) -> u64 =
            crate::mem::jit_mem;
        // Two mappings of the same pages, one writable and one executable, so
        // translating and patching never switch a thread's W^X state.
        Ok(CodeCache::with_region(
            JitRegion::new_dual(bytes)?,
            helper as usize as u64,
            fast,
        ))
    }

    /// A cache in `region` (a slice of a machine's shared region, say).
    pub fn in_region(region: JitRegion, fast: bool) -> Self {
        let helper: unsafe extern "C" fn(*mut Cpu, *const u8, *mut CpuState) -> u64 =
            crate::mem::jit_mem;
        CodeCache::with_region(region, helper as usize as u64, fast)
    }

    pub fn epilogue_addr() -> u64 {
        epilogue_addr()
    }
}

/// Whether the environment turns the fast paths off (`CRATE_JIT_SLOW=1`), so
/// a failure can be re-run on the plain translation.
pub fn slow_from_env() -> bool {
    std::env::var("CRATE_JIT_SLOW").is_ok_and(|v| v != "0" && !v.is_empty())
}

/// Why `Cpu::run` returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    Svc(u32),
    Hvc(u32),
    Smc(u32),
    Brk(u32),
    Undefined(u32),
    Sysreg,
    Exception,
    Unknown(u32),
}

/// A list of the entries of a table filled since it was last emptied, so
/// emptying it touches only those, until there are too many to track.
struct Filled {
    idx: Vec<u16>,
    overflow: bool,
}

impl Filled {
    const MAX: usize = 2048;

    fn new() -> Self {
        Filled {
            idx: Vec::with_capacity(Self::MAX),
            overflow: false,
        }
    }

    fn push(&mut self, i: usize) {
        if self.idx.len() < Self::MAX {
            self.idx.push(i as u16);
        } else {
            self.overflow = true;
        }
    }

    /// Visits the entries that may be in use: those listed, or all of them.
    fn each(&self, len: usize, mut f: impl FnMut(usize)) {
        if self.overflow {
            (0..len).for_each(f);
        } else {
            self.idx.iter().for_each(|&i| f(i as usize));
        }
    }

    fn clear(&mut self) {
        self.idx.clear();
        self.overflow = false;
    }
}

/// What the Rust side keeps about each software-TLB entry beyond what the
/// fast path needs: the size of the mapping it came from and whether that
/// mapping was global, and lists of the entries an invalidation must visit,
/// so a TLBI of a user address does not scan what only the kernel uses.
pub(crate) struct TlbBook {
    /// Per TLB and entry: log2 of the mapping size (64 with the MMU off), and
    /// bit 7 set for a non-global mapping.
    meta: Box<[[u8; TLB_ENTRIES]; 4]>,
    /// Every entry filled.
    filled: [Filled; 4],
    /// Entries for the lower half of the address space (TTBR0); for the EL0
    /// TLBs that is every entry.
    low: [Filled; 4],
    /// Entries from mappings larger than a TLB entry, by half of the address
    /// space (0 lower, 1 upper): a TLBI of one address drops every piece.
    large: [[Filled; 2]; 4],
    /// Some upper-half entry is non-global (only with a split kernel map).
    ng_high: [bool; 4],
}

const META_NG: u8 = 1 << 7;

impl TlbBook {
    fn new() -> Self {
        let f = || [Filled::new(), Filled::new(), Filled::new(), Filled::new()];
        TlbBook {
            meta: Box::new([[0; TLB_ENTRIES]; 4]),
            filled: f(),
            low: f(),
            large: [
                [Filled::new(), Filled::new()],
                [Filled::new(), Filled::new()],
                [Filled::new(), Filled::new()],
                [Filled::new(), Filled::new()],
            ],
            ng_high: [false; 4],
        }
    }

    pub(crate) fn filled(&mut self, tlb: usize, i: usize, va: u64, size_shift: u32, global: bool) {
        let shift = size_shift.min(64) as u8;
        self.meta[tlb][i] = shift | if global { 0 } else { META_NG };
        self.filled[tlb].push(i);
        let high = (va >> 55) & 1 == 1;
        if !high {
            self.low[tlb].push(i);
        } else if !global {
            self.ng_high[tlb] = true;
        }
        if shift > crate::state::TLB_SHIFT as u8 {
            self.large[tlb][high as usize].push(i);
        }
    }

    fn clear(&mut self, tlb: usize) {
        self.filled[tlb].clear();
        self.low[tlb].clear();
        self.large[tlb][0].clear();
        self.large[tlb][1].clear();
        self.ng_high[tlb] = false;
    }
}

/// Which of a TLB's entry lists an invalidation visits.
#[derive(Clone, Copy)]
enum TlbList {
    All,
    Low,
    Large(usize),
}

/// A cache of instruction-fetch translations for the dispatcher's block
/// lookups, emptied (by moving to a new generation) by every TLB
/// invalidation.
struct FetchTlb {
    /// Per EL and entry: (page | 1 for valid, physical page, mapping size,
    /// generation).
    entries: Box<[[FetchEntry; 256]; 2]>,
    generation: u32,
}

type FetchEntry = (u64, u64, u32, u32);

impl FetchTlb {
    fn lookup(&self, el: u32, va: u64) -> Option<(u64, u32)> {
        let e = &self.entries[el.min(1) as usize][((va >> 12) & 255) as usize];
        (e.0 == (va >> 12 << 1 | 1) && e.3 == self.generation).then_some((e.1 | (va & 0xfff), e.2))
    }

    fn insert(&mut self, el: u32, va: u64, pa: u64, size_shift: u32) {
        self.entries[el.min(1) as usize][((va >> 12) & 255) as usize] =
            (va >> 12 << 1 | 1, pa & !0xfff, size_shift, self.generation);
    }

    fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Wrapped: forget every entry for real.
            self.entries.iter_mut().for_each(|t| t.fill((0, 0, 0, 0)));
            self.generation = 1;
        }
    }
}

/// The jump caches' bookkeeping: what was filled, and which guest pages of
/// code are in them, so a TLB invalidation of an address with no code in the
/// cache leaves it alone.
struct JcBook {
    filled: [Filled; 2],
    pages: [HashSet<u64, BuildHasherDefault<FastHasher>>; 2],
    /// The EL1 cache holds code from the lower half of the address space
    /// (TTBR0), which a change of TTBR0 makes stale.
    el1_low: bool,
    /// The blocks (mapping size, VA >> size) larger than a page that cached
    /// code came from, so a TLBI inside one drops the cache; `None` once too
    /// many to track, when any TLBI does.
    large: [Option<Vec<(u32, u64)>>; 2],
    /// The miss target empty entries point at, for the cache in use.
    miss: [u64; 2],
    /// EL0 entries are tagged with a context number for the address space
    /// (ASID and TTBR0) they were filled in, so a context switch keeps them;
    /// retiring a context's number drops them all at once.
    contexts: HashMap<(u16, u64), u16, BuildHasherDefault<FastHasher>>,
    next_context: u16,
}

/// Debug switches read once from the environment.
#[derive(Clone, Default)]
struct Trace {
    /// `CRATE_TRACE_PC=1`: log every block the dispatcher enters.
    pc: bool,
    /// `CRATE_TRAP_PC=0x...`: dump registers when a block starts there.
    trap_pc: Option<u64>,
    /// `CRATE_TRAP_MEM=1`: log every data abort.
    mem: bool,
    /// `CRATE_TRACE_IRQ=1`: log interrupts taken and timer lines driven.
    irq: bool,
    /// `CRATE_JIT_STATS=1`: count exits and slow-path calls, printed when
    /// the CPU stops.
    stats: bool,
}

/// Counters `CRATE_JIT_STATS=1` keeps.
#[derive(Default, Debug)]
pub(crate) struct Counters {
    pub branch: u64,
    pub sysreg: u64,
    pub system_word: u64,
    pub call: u64,
    pub exception: u64,
    pub eret: u64,
    pub irq: u64,
    pub wfi: u64,
    pub wfe: u64,
    pub slow_mem: u64,
    pub translated: u64,
    /// Time spent translating, in microseconds.
    pub translate_us: u64,
    pub remote: u64,
    pub jc_flushes: u64,
    pub jc_misses: u64,
    pub jc_lookups: u64,
    pub jc_flush_reason: [u64; 6],
    pub chain_exits: u64,
    pub slow_device: u64,
    pub slow_cross: u64,
    pub slow_fault: u64,
    /// Exits by system register.
    pub sysregs: std::collections::BTreeMap<u16, u64>,
}

impl Trace {
    fn from_env() -> Self {
        Trace {
            pc: std::env::var_os("CRATE_TRACE_PC").is_some(),
            trap_pc: std::env::var("CRATE_TRAP_PC")
                .ok()
                .and_then(|t| u64::from_str_radix(t.trim_start_matches("0x"), 16).ok()),
            mem: std::env::var_os("CRATE_TRAP_MEM").is_some(),
            irq: std::env::var_os("CRATE_TRACE_IRQ").is_some(),
            stats: std::env::var_os("CRATE_JIT_STATS").is_some(),
        }
    }
}

/// A guest CPU the dispatcher drives.
pub struct Cpu {
    pub state: Box<CpuState>,
    pub mem: Arc<GuestMemory>,
    /// The board, for routing MMIO and reaching the interrupt controller,
    /// PSCI and the run state. Null for the bare EL0 test runner. Valid for a
    /// `dispatch` call's whole run.
    board: *const Board,
    /// The machine's CPUs, for kicks and broadcasts. Null when standalone.
    smp: *const Smp,
    /// Which CPU this is (its index into the board), for the GIC and PSCI.
    cpu: usize,
    /// Its MPIDR_EL1.
    mpidr: u64,
    pub(crate) tlbs: TlbBook,
    pub(crate) counters: Counters,
    fetch: FetchTlb,
    jc: JcBook,
    /// The PPI levels the timers last drove, and their conditions then.
    vtimer_line: bool,
    ptimer_line: bool,
    timer_fired: [bool; 2],
    /// An interrupt was pending but masked when last looked at.
    masked_pending: bool,
    /// When that was found: PSTATE's I and F bits, and the interrupt
    /// controller's wake count. Until either changes, nothing new can be
    /// taken, so the dispatcher does not ask the controller again.
    masked_at: Option<(u64, u32)>,
    trace: Trace,
}

// SAFETY: a Cpu is driven by one thread at a time; the board and machine it
// points at are shared, thread-safe, and outlive its run.
unsafe impl Send for Cpu {}

impl Cpu {
    pub fn new(mem: Arc<GuestMemory>) -> Box<Self> {
        let mut cpu = Box::new(Cpu {
            state: CpuState::new(),
            mem,
            board: std::ptr::null(),
            smp: std::ptr::null(),
            cpu: 0,
            mpidr: 0x8000_0000,
            tlbs: TlbBook::new(),
            counters: Counters::default(),
            fetch: FetchTlb {
                entries: Box::new([[(0, 0, 0, 0); 256]; 2]),
                generation: 1,
            },
            jc: JcBook {
                filled: [Filled::new(), Filled::new()],
                pages: [HashSet::default(), HashSet::default()],
                el1_low: false,
                large: [Some(Vec::new()), Some(Vec::new())],
                miss: [0; 2],
                contexts: HashMap::default(),
                next_context: 1,
            },
            vtimer_line: false,
            ptimer_line: false,
            timer_fired: [false; 2],
            masked_pending: false,
            masked_at: None,
            trace: Trace::from_env(),
        });
        cpu.state.epilogue = epilogue_addr();
        let ptr: *mut Cpu = &mut *cpu;
        cpu.state.cpu_ptr = ptr as u64;
        let sp: *mut CpuState = &mut *cpu.state;
        cpu.state.self_ptr = sp as u64;
        cpu
    }

    pub(crate) fn board_opt(&self) -> Option<&Board> {
        // SAFETY: as for `board`; null outside a machine.
        unsafe { self.board.as_ref() }
    }

    fn smp(&self) -> Option<&Smp> {
        // SAFETY: set for the machine's whole run, or null.
        unsafe { self.smp.as_ref() }
    }

    pub fn el(&self) -> u32 {
        ((self.state.pstate >> 2) & 3) as u32
    }

    /// Points the jump caches' empty entries at `cache`'s miss path, emptying
    /// them, if they do not already.
    fn bind_cache(&mut self, cache: &CodeCache) {
        let miss = cache.stubs().jc_miss;
        if self.jc.miss != miss {
            self.jc.miss = miss;
            for (el, jc) in self.state.jc.iter_mut().enumerate() {
                jc.entries.fill(JcEntry {
                    tag: 0,
                    host: miss[el],
                });
                self.jc.filled[el].clear();
                self.jc.pages[el].clear();
                self.jc.large[el] = Some(Vec::new());
            }
            self.jc.el1_low = false;
        }
    }

    /// Runs the guest until a block returns something the translator cannot
    /// handle on its own (a call, an exception, a system register, or an
    /// unknown instruction). Branches and waits are followed here. This is
    /// the bare runner the EL0 tests use, with no board.
    pub fn run(&mut self, cache: &mut CodeCache) -> Stop {
        self.bind_cache(cache);
        let mut chain: Option<(u64, u64)> = None;
        loop {
            let (target, cacheable) = self.next_block(cache, chain.take());
            self.state.dispatch_target = target;
            self.state.page_va = self.state.pc & !0xfff;
            self.state.chain_site = 0;
            let epoch = cache.epoch();
            // SAFETY: the state is a stable heap box; the enter trampoline runs
            // the block at `target` and returns the exit code.
            let code = unsafe { crate_jit_enter(&mut *self.state) } as u32;
            match Exit::from_u32(code) {
                Some(Exit::Branch) | Some(Exit::WaitForInterrupt) | Some(Exit::WaitForEvent) => {
                    self.state.pc = self.state.next_pc;
                    if self.state.chain_site != 0 && cacheable {
                        chain = Some((self.state.chain_site, epoch));
                    }
                }
                Some(Exit::Call) => {
                    let syn = self.state.exception_syndrome;
                    let imm = (syn & 0xffff) as u32;
                    return match (syn >> 16) & 0xff {
                        0 => Stop::Svc(imm),
                        1 => Stop::Hvc(imm),
                        2 => Stop::Smc(imm),
                        _ => Stop::Brk(imm),
                    };
                }
                Some(Exit::Undefined) => {
                    return Stop::Undefined(self.state.exception_syndrome as u32);
                }
                Some(Exit::Sysreg) => {
                    if !self.el0_sysreg() {
                        return Stop::Sysreg;
                    }
                }
                Some(Exit::Exception) => return Stop::Exception,
                None => return Stop::Unknown(code),
            }
        }
    }

    /// The bare runner's system registers: the ones an EL0 program uses with
    /// the fast paths off (NZCV, FPCR, FPSR, TPIDR_EL0). Returns false for
    /// anything else.
    fn el0_sysreg(&mut self) -> bool {
        let reg = self.state.sysreg as u16;
        let rt = self.state.sysreg_rt as usize;
        let read = self.state.sysreg_read != 0;
        let value = if rt == 31 { 0 } else { self.state.x[rt] };
        let got = match reg {
            r if r == sr::NZCV.0 => {
                if !read {
                    self.state.nzcv = value & 0xf000_0000;
                }
                self.state.nzcv
            }
            r if r == sr::FPCR.0 => {
                if !read {
                    set_fpcr(value);
                }
                fpcr()
            }
            r if r == sr::FPSR.0 => {
                if !read {
                    set_fpsr(value);
                }
                fpsr()
            }
            r if r == sr::TPIDR_EL0.0 => {
                if !read {
                    self.state.banked.tpidr_el0 = value;
                }
                self.state.banked.tpidr_el0
            }
            _ => return false,
        };
        if read && rt != 31 {
            self.state.x[rt] = got;
        }
        self.state.pc = self.state.next_pc;
        true
    }

    /// The host entry of the block for the current PC, translating it if
    /// needed, and whether it is a normal (cached) block. A pending chain from
    /// the previous block's exit is patched to it.
    fn next_block(&mut self, cache: &mut CodeCache, chain: Option<(u64, u64)>) -> (u64, bool) {
        let pc = self.state.pc;
        let el = self.el();
        let debug = self.debug_active();
        let ctx = Ctx {
            el,
            fast: cache.fast,
            debug,
            stubs: *cache.stubs(),
            mpidr: self.mpidr,
        };
        let (pa, size_shift) = match self.fetch.lookup(el, pc) {
            Some(hit) => hit,
            None => match mmu::translate(&self.mem, &self.state.mmu, el, pc, Access::Execute) {
                Ok(t) => {
                    self.fetch.insert(el, pc, t.pa, t.size_shift);
                    (t.pa, t.size_shift)
                }
                Err(f) => return (cache.fetch_fault(pc, f.fsc, ctx), false),
            },
        };
        let key = BlockKey {
            pa,
            mode: el | (debug as u32) << 2,
        };
        let entry = match cache.lookup(&key) {
            Some(e) => e,
            None => {
                let page = pa & !0xfff;
                let off = (pc & 0xfff) as usize;
                let Ok(host) = self.mem.host_range(page, 0x1000) else {
                    // The fetch translated but missed RAM: an external abort.
                    return (cache.fetch_fault(pc, 0x10, ctx), false);
                };
                // Tell the other CPUs before reading the code, so one that
                // rewrites it and does not see this sends the invalidation.
                if let Some(smp) = self.smp() {
                    smp.note_code(page);
                }
                let n = ((0x1000 - off) / 4).min(512);
                let mut words = [0u32; 512];
                // SAFETY: inside the page host_range checked; guest code may be
                // written concurrently by another CPU, which is the guest's
                // problem (it must invalidate after writing code).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        host.add(off).cast::<u32>(),
                        words.as_mut_ptr(),
                        n,
                    );
                }
                let epoch = cache.epoch();
                self.counters.translated += 1;
                let started = self.trace.stats.then(std::time::Instant::now);
                let e = cache.translate(key, pc, &words[..n], ctx);
                if let Some(t) = started {
                    self.counters.translate_us += t.elapsed().as_micros() as u64;
                }
                if cache.epoch() != epoch {
                    // The region filled and was emptied: every jump-cache
                    // entry points at code that is gone.
                    self.jc_flush(0);
                    self.jc_flush(1);
                }
                e
            }
        };
        if let Some((site, epoch)) = chain {
            cache.chain(site, epoch, entry);
        }
        if cache.fast && !debug && pc & 3 == 0 {
            let el_i = el.min(1) as usize;
            let tag = if el_i == 0 {
                pc ^ self.state.jc_context
            } else {
                pc
            };
            let i = JcEntry::index(tag);
            self.state.jc[el_i].entries[i] = JcEntry { tag, host: entry };
            self.jc.filled[el_i].push(i);
            self.jc.pages[el_i].insert(pc >> 12);
            if size_shift > 12 {
                let region = (size_shift.min(63), pc >> size_shift.min(63));
                if let Some(list) = &mut self.jc.large[el_i] {
                    if !list.contains(&region) {
                        if list.len() >= 32 {
                            self.jc.large[el_i] = None;
                        } else {
                            list.push(region);
                        }
                    }
                }
            }
            if el_i == 1 && pc >> 55 == 0 {
                self.jc.el1_low = true;
            }
        }
        (entry, true)
    }

    // ---------------------------------------------------------------- TLB and jump cache

    /// Empties the jump cache for `el`.
    fn jc_flush(&mut self, el: usize) {
        self.counters.jc_flushes += 1;
        let miss = self.jc.miss[el];
        let entries = &mut self.state.jc[el].entries;
        self.jc.filled[el].each(JC_ENTRIES, |i| {
            entries[i] = JcEntry { tag: 0, host: miss };
        });
        self.jc.filled[el].clear();
        self.jc.pages[el].clear();
        self.jc.large[el] = Some(Vec::new());
        if el == 1 {
            self.jc.el1_low = false;
        }
    }

    /// Empties software TLB `t` of the entries `drop` picks (given an entry's
    /// tag and its book byte), visiting those `list` holds.
    fn tlb_drop(&mut self, t: usize, list: TlbList, mut drop: impl FnMut(u64, u8) -> bool) {
        let entries = &mut self.state.tlb[t].entries;
        let meta = &self.tlbs.meta[t];
        let filled = match list {
            TlbList::All => &self.tlbs.filled[t],
            TlbList::Low => &self.tlbs.low[t],
            TlbList::Large(half) => &self.tlbs.large[t][half],
        };
        filled.each(TLB_ENTRIES, |i| {
            let e = &mut entries[i];
            if e.tag != TLB_INVALID && drop(e.tag, meta[i]) {
                e.tag = TLB_INVALID;
            }
        });
    }

    fn tlb_empty(&mut self, t: usize) {
        self.tlb_drop(t, TlbList::All, |_, _| true);
        self.tlbs.clear(t);
    }

    /// Drops every translation (TLBI VMALLE1, an MMU or translation-control
    /// change).
    pub(crate) fn flush_all(&mut self) {
        self.counters.jc_flush_reason[0] += 1;
        for t in 0..4 {
            self.tlb_empty(t);
        }
        self.fetch.clear();
        self.jc_flush(0);
        self.jc_flush(1);
        self.jc.contexts.clear();
        self.context_changed();
    }

    /// Drops the data translations of the lower half of the address space (a
    /// change of address space). The EL0 jump cache keeps its entries: they
    /// are tagged with the address space's context number.
    fn flush_low(&mut self) {
        self.counters.jc_flush_reason[1] += 1;
        self.tlb_empty(2);
        self.tlb_empty(3);
        for t in 0..2 {
            self.tlb_drop(t, TlbList::Low, |tag, _| tag >> (55 - 12) & 1 == 0);
            self.tlbs.low[t].clear();
            self.tlbs.large[t][0].clear();
        }
        self.fetch.clear();
        if self.jc.el1_low {
            self.jc_flush(1);
        }
    }

    /// The address space changed (TTBR0, the ASID, TCR or the MMU): publish
    /// its ASID for other CPUs' TLBIs, and tag EL0 jump-cache entries with
    /// its context number from now on.
    fn context_changed(&mut self) {
        let m = &self.state.mmu;
        let ttbr = if m.tcr_el1 & (1 << 22) != 0 {
            m.ttbr1_el1
        } else {
            m.ttbr0_el1
        };
        let width = if m.tcr_el1 & (1 << 36) != 0 {
            0xffff
        } else {
            0xff
        };
        let on = m.sctlr_el1 & 1 != 0;
        let asid = (ttbr >> 48) as u16 & width;
        let key = (asid, m.ttbr0_el1 & 0x0000_ffff_ffff_fffe | !on as u64);
        if let Some(smp) = self.smp() {
            smp.set_asid(self.cpu, on.then_some(asid));
        }
        let ctx = match self.jc.contexts.get(&key) {
            Some(&c) => c,
            None => {
                if self.jc.next_context == u16::MAX {
                    // Out of numbers: forget every context.
                    self.jc.contexts.clear();
                    self.jc.next_context = 1;
                    self.jc_flush(0);
                }
                let c = self.jc.next_context;
                self.jc.next_context += 1;
                self.jc.contexts.insert(key, c);
                c
            }
        };
        self.state.jc_context = (ctx as u64) << 48;
    }

    /// Retires the context numbers of address space `asid` (every one with
    /// `None`), so their EL0 jump-cache entries never match again.
    fn retire_contexts(&mut self, asid: Option<u16>) {
        match asid {
            Some(a) => self.jc.contexts.retain(|&(ctx_asid, _), _| ctx_asid != a),
            None => {
                self.jc.contexts.clear();
                self.jc_flush(0);
            }
        }
        self.context_changed();
    }

    /// Drops every non-global translation of address space `asid` (TLBI
    /// ASIDE1): the data TLBs hold only the running address space's, and its
    /// code in the jump cache goes by retiring its context numbers.
    fn flush_user(&mut self, asid: u16) {
        self.counters.jc_flush_reason[2] += 1;
        self.tlb_empty(2);
        self.tlb_empty(3);
        for t in 0..2 {
            let list = if self.tlbs.ng_high[t] {
                TlbList::All
            } else {
                TlbList::Low
            };
            self.tlb_drop(t, list, |_, meta| meta & META_NG != 0);
        }
        self.fetch.clear();
        self.retire_contexts(Some(asid));
        if self.jc.el1_low {
            self.jc_flush(1);
        }
    }

    /// Drops the translations of `va` (TLBI VAE1 and friends), including every
    /// piece of a block mapping that covers it.
    fn flush_va(&mut self, va: u64, asid: Option<u16>) {
        let page = va >> 12;
        let half = ((va >> 55) & 1) as usize;
        for t in 0..4 {
            let i = TlbEntry::index(t, va);
            if self.state.tlb[t].entries[i].tag == page {
                self.state.tlb[t].entries[i].tag = TLB_INVALID;
            }
            self.tlb_drop(t, TlbList::Large(half), |tag, meta| {
                let shift = (meta & !META_NG) as u32;
                shift >= 64 || (shift > 12 && (tag << 12) >> shift == va >> shift)
            });
        }
        self.fetch.clear();
        for el in 0..2 {
            let in_large = match &self.jc.large[el] {
                Some(list) => list.iter().any(|&(shift, base)| va >> shift == base),
                None => true,
            };
            if in_large || self.jc.pages[el].contains(&page) {
                self.counters.jc_flush_reason[3] += 1;
                if el == 0 && half == 0 {
                    self.retire_contexts(asid);
                } else {
                    self.jc_flush(el);
                }
            }
        }
    }

    /// Applies an operation another CPU (or this one) asked for.
    fn apply(&mut self, op: Remote, cache: &mut CodeCache) {
        match op {
            Remote::FlushAll => self.flush_all(),
            Remote::FlushVa(va, asid) => self.flush_va(va, asid),
            Remote::FlushUser(asid) => self.flush_user(asid),
            Remote::CodePage(pa) => {
                if cache.invalidate_page(pa) {
                    self.counters.jc_flush_reason[4] += 1;
                    self.jc_flush(0);
                    self.jc_flush(1);
                }
            }
            Remote::CodeAll => {
                cache.reset();
                self.jc_flush(0);
                self.jc_flush(1);
            }
        }
    }

    /// Does `op` here and, for the inner-shareable forms, on every other CPU.
    fn maintain(&mut self, op: Remote, broadcast: bool, cache: &mut CodeCache) {
        self.apply(op, cache);
        if broadcast {
            if let Some(smp) = self.smp() {
                if smp.cpus() > 1 {
                    smp.broadcast(self.cpu, op);
                }
            }
        }
    }

    fn apply_remote(&mut self, cache: &mut CodeCache) {
        let ops = match self.smp() {
            Some(smp) => smp.take(self.cpu),
            None => return,
        };
        for op in ops {
            self.counters.remote += 1;
            self.apply(op, cache);
        }
    }
}

// ---------------------------------------------------------------- the EL1 machine

use crate_core::cpu::SysReg;
use crate_core::psci::Effect;
use crate_core::run::StopReason;

/// PSTATE bits the translator tracks in `state.pstate`.
const PSTATE_SPSEL: u64 = 1 << 0;
const PSTATE_EL_SHIFT: u64 = 2;
const PSTATE_F: u64 = 1 << 6;
const PSTATE_I: u64 = 1 << 7;
const PSTATE_A: u64 = 1 << 8;
const PSTATE_D: u64 = 1 << 9;
const PSTATE_DAIF: u64 = PSTATE_D | PSTATE_A | PSTATE_I | PSTATE_F;

/// Exception classes (ESR_EL1.EC) the machine raises.
const EC_UNKNOWN: u64 = 0x00;
const EC_SVC64: u64 = 0x15;
const EC_INSN_ABORT_LOWER: u64 = 0x20;
const EC_INSN_ABORT_SAME: u64 = 0x21;
const EC_PC_ALIGNMENT: u64 = 0x22;
const EC_DATA_ABORT_LOWER: u64 = 0x24;
const EC_DATA_ABORT_SAME: u64 = 0x25;
const EC_BRK: u64 = 0x3c;
const EC_BREAKPOINT_LOWER: u64 = 0x30;
const EC_BREAKPOINT_SAME: u64 = 0x31;
const EC_SOFTWARE_STEP_LOWER: u64 = 0x32;
const EC_SOFTWARE_STEP_SAME: u64 = 0x33;
const EC_WATCHPOINT_LOWER: u64 = 0x34;
const EC_WATCHPOINT_SAME: u64 = 0x35;
/// PSTATE.SS, the software-step bit, in `state.pstate`.
const PSTATE_SS: u64 = 1 << 21;
pub(crate) const WATCHPOINT_KIND: u64 = 2;
const DATA_ABORT_KIND: u64 = 1;

/// SCTLR_EL1 at reset: the MMU, caches and alignment checks off (the RES1
/// bits of Armv8.0 set).
const SCTLR_RESET: u64 = 0x30d0_0800;

/// How a CPU's run ended.
enum Flow {
    Stop,
    CpuOff,
}

impl Cpu {
    /// Builds a CPU for `board`, at its reset state (EL1h, MMU off,
    /// interrupts masked).
    fn for_board(board: &Board, smp: &Smp, cpu: usize) -> Box<Self> {
        let mut c = Cpu::new(board.memory.clone());
        c.board = board as *const Board;
        c.smp = smp as *const Smp;
        c.cpu = cpu;
        c.mpidr = board.mpidrs[cpu];
        c.reset(0, [0; 4]);
        c
    }

    /// Puts the CPU in its reset state, starting at `pc` with X0..X3 set.
    fn reset(&mut self, pc: u64, x: [u64; 4]) {
        let s = &mut self.state;
        s.x = [0; 32];
        s.x[..4].copy_from_slice(&x);
        s.sp = 0;
        s.pc = pc;
        s.nzcv = 0;
        s.pstate = (1 << PSTATE_EL_SHIFT) | PSTATE_SPSEL | PSTATE_DAIF; // EL1h, masked
        s.v = [[0; 16]; 32];
        set_fpcr(0);
        set_fpsr(0);
        s.monitor = MONITOR_CLEAR;
        s.banked = Default::default();
        s.mmu = MmuRegs {
            sctlr_el1: SCTLR_RESET,
            ..Default::default()
        };
        s.debug = Default::default();
        self.vtimer_line = false;
        self.ptimer_line = false;
        self.masked_pending = false;
        self.flush_all();
    }

    fn set_el(&mut self, el: u32) {
        self.state.pstate = (self.state.pstate & !(3 << PSTATE_EL_SHIFT)) | ((el as u64) << 2);
    }

    /// Runs this CPU until the machine stops or the CPU turns itself off.
    fn dispatch(&mut self, cache: &mut CodeCache) -> Flow {
        self.bind_cache(cache);
        let board = self.board;
        // SAFETY: the board outlives the run.
        let board = unsafe { &*board };
        let mut chain: Option<(u64, u64)> = None;
        self.service_timer(board);
        loop {
            if board.run.is_stopped() {
                return Flow::Stop;
            }
            // Whatever another thread asked for: remote maintenance, the timer.
            let kicked = self.state.exit_request().swap(0, Ordering::Acquire) != 0;
            if self.state.sync_request().swap(0, Ordering::Acquire) != 0 && !kicked {
                self.apply_remote(cache);
                chain = None;
            }
            if kicked {
                chain = None;
                self.apply_remote(cache);
                self.service_timer(board);
                if dump_requested() {
                    self.dump();
                }
            }
            if self.interrupt_may_be_taken() && self.take_pending_interrupt(board) {
                chain = None;
                continue;
            }
            let pc = self.state.pc;
            if self.trace.pc {
                crate_core::debug!("cpu{} block pc={pc:#x} el={}", self.cpu, self.el());
            }
            if self.trace.trap_pc == Some(pc) {
                self.dump_regs();
            }
            if pc & 3 != 0 {
                self.take_exception(EC_PC_ALIGNMENT, 0, Some(pc), pc);
                continue;
            }
            // Self-hosted debug: when a breakpoint, watchpoint or step is armed,
            // take any breakpoint that matches this PC before running, and
            // translate one instruction at a time so a step fires after it.
            let debug = self.debug_active();
            if debug && self.breakpoint_at(pc) {
                self.take_breakpoint(pc);
                continue;
            }
            // Software step: with MDSCR.SS set and debug unmasked, PSTATE.SS=0 is
            // the active-pending state -- take the step exception before this
            // instruction. PSTATE.SS=1 is active-not-pending -- run one
            // instruction, then move to pending.
            let step_on =
                debug && self.state.banked.mdscr_el1 & 1 != 0 && self.state.pstate & PSTATE_D == 0;
            if step_on && self.state.pstate & PSTATE_SS == 0 {
                self.take_step(pc);
                continue;
            }
            let stepping_now = step_on && self.state.pstate & PSTATE_SS != 0;
            let (target, cacheable) = self.next_block(cache, chain.take());
            self.state.dispatch_target = target;
            self.state.page_va = pc & !0xfff;
            self.state.chain_site = 0;
            LAST_PC[self.cpu.min(LAST_PC.len() - 1)].store(pc, Ordering::Relaxed);
            let epoch = cache.epoch();
            let smp = self.smp;
            // SAFETY: set for the whole run.
            let smp = unsafe { &*smp };
            if !smp.enter(self.cpu) {
                self.apply_remote(cache);
                continue;
            }
            // SAFETY: the state is a stable heap box for the whole dispatch run.
            let code = unsafe { crate_jit_enter(&mut *self.state) } as u32;
            smp.leave(self.cpu);
            // One instruction ran: leave active-pending unless it faulted.
            if stepping_now
                && !matches!(
                    Exit::from_u32(code),
                    Some(Exit::Exception) | Some(Exit::Undefined)
                )
            {
                self.state.pstate &= !PSTATE_SS;
            }
            if self.trace.stats {
                let c = &mut self.counters;
                match Exit::from_u32(code) {
                    Some(Exit::Branch) => {
                        c.branch += 1;
                        if self.state.chain_site != 0 {
                            c.chain_exits += 1;
                        }
                    }
                    Some(Exit::Sysreg) if self.state.sysreg as u16 == SYSTEM_WORD_MARKER => {
                        c.system_word += 1
                    }
                    Some(Exit::Sysreg) => {
                        c.sysreg += 1;
                        *c.sysregs.entry(self.state.sysreg as u16).or_default() += 1;
                    }
                    Some(Exit::Call) => c.call += 1,
                    Some(Exit::Exception) if self.state.exception_syndrome & ERET_MARKER != 0 => {
                        c.eret += 1
                    }
                    Some(Exit::Exception) => c.exception += 1,
                    Some(Exit::WaitForInterrupt) => c.wfi += 1,
                    Some(Exit::WaitForEvent) => c.wfe += 1,
                    _ => {}
                }
            }
            match Exit::from_u32(code) {
                Some(Exit::Branch) => {
                    self.state.pc = self.state.next_pc;
                    if self.state.chain_site != 0 && cacheable {
                        chain = Some((self.state.chain_site, epoch));
                    }
                }
                Some(Exit::WaitForInterrupt) => {
                    self.state.pc = self.state.next_pc;
                    self.wait_for_interrupt(board, cache);
                }
                Some(Exit::WaitForEvent) => {
                    self.state.pc = self.state.next_pc;
                    std::thread::yield_now();
                }
                Some(Exit::Sysreg) => {
                    self.handle_sysreg(board, cache);
                    if self.state.flush_cache != 0 {
                        self.state.flush_cache = 0;
                        cache.reset();
                        self.jc_flush(0);
                        self.jc_flush(1);
                    }
                }
                Some(Exit::Call) => {
                    if let Some(flow) = self.handle_call(board) {
                        return flow;
                    }
                }
                Some(Exit::Exception) => self.handle_exception_exit(),
                Some(Exit::Undefined) => {
                    let pc = self.state.pc;
                    self.take_exception(EC_UNKNOWN, 0, None, pc);
                }
                None => {
                    board
                        .run
                        .stop(StopReason::Error(format!("unknown exit {code}")));
                    return Flow::Stop;
                }
            }
        }
    }

    fn dump_regs(&self) {
        let x = &self.state.x;
        crate_core::warn!(
            "cpu{} pc={:#x} el={} x0={:#x} x1={:#x} x2={:#x} x3={:#x} x4={:#x} x5={:#x} x30={:#x} sp={:#x} elr={:#x} esr={:#x} far={:#x}",
            self.cpu,
            self.state.pc,
            self.el(),
            x[0],
            x[1],
            x[2],
            x[3],
            x[4],
            x[5],
            x[30],
            self.state.sp,
            self.state.banked.elr_el1,
            self.state.banked.esr_el1,
            self.state.banked.far_el1,
        );
    }

    fn dump(&self) {
        self.dump_regs();
    }

    // ---- interrupts ----

    /// Takes an IRQ or FIQ that is pending and unmasked. Leaves the hint set
    /// while one is pending but masked, so unmasking looks again.
    fn take_pending_interrupt(&mut self, board: &Board) -> bool {
        self.state.irq_hint().store(0, Ordering::Relaxed);
        self.masked_at = None;
        let wakes = self.state.irq_wakes().load(Ordering::Acquire);
        let irq = board.interrupts.irq_pending(self.cpu);
        let fiq = board.interrupts.fiq_pending(self.cpu);
        let pstate = self.state.pstate;
        if self.trace.irq {
            crate_core::warn!(
                "cpu{} irq check: irq={irq} fiq={fiq} pstate={pstate:#x} pc={:#x}",
                self.cpu,
                self.state.pc
            );
        }
        if irq && pstate & PSTATE_I == 0 {
            let pc = self.state.pc;
            self.counters.irq += 1;
            self.masked_pending = false;
            self.take_exception_kind(ExcKind::Irq, Some(0), None, pc);
            return true;
        }
        if fiq && pstate & PSTATE_F == 0 {
            let pc = self.state.pc;
            self.masked_pending = false;
            self.take_exception_kind(ExcKind::Fiq, Some(0), None, pc);
            return true;
        }
        self.masked_pending = irq || fiq;
        if self.masked_pending {
            self.state.irq_hint().store(1, Ordering::Relaxed);
            self.masked_at = Some((pstate & (PSTATE_I | PSTATE_F), wakes));
        }
        false
    }

    /// Whether an interrupt could be taken now that was not when last looked
    /// at: a hint is set, and the mask or the controller's state changed.
    fn interrupt_may_be_taken(&self) -> bool {
        if self.state.irq_hint().load(Ordering::Relaxed) == 0 {
            return false;
        }
        match self.masked_at {
            Some((mask, wakes)) => {
                mask != self.state.pstate & (PSTATE_I | PSTATE_F)
                    || wakes != self.state.irq_wakes().load(Ordering::Acquire)
            }
            None => true,
        }
    }

    /// After anything that may unmask an interrupt or change what is
    /// pending: look again before the next block.
    fn recheck_interrupts(&mut self) {
        self.masked_at = None;
        self.state.irq_hint().store(1, Ordering::Relaxed);
    }

    fn wait_for_interrupt(&mut self, board: &Board, cache: &mut CodeCache) {
        // Sleep until an interrupt could be taken or the timer would fire.
        loop {
            if board.run.is_stopped() {
                return;
            }
            self.state.exit_request().store(0, Ordering::Relaxed);
            self.apply_remote(cache);
            self.service_timer(board);
            // WFI ends on a pending interrupt even if it is masked.
            if board.interrupts.irq_pending(self.cpu) || board.interrupts.fiq_pending(self.cpu) {
                self.recheck_interrupts();
                return;
            }
            self.state.irq_hint().store(0, Ordering::Relaxed);
            let deadline = self.timer_deadline();
            let smp = self.smp;
            // SAFETY: set for the whole run.
            unsafe { &*smp }.sleep(self.cpu, deadline);
        }
    }

    // ---- calls (SVC / HVC / SMC / BRK) ----

    fn handle_call(&mut self, board: &Board) -> Option<Flow> {
        let syn = self.state.exception_syndrome;
        let imm = (syn & 0xffff) as u32;
        let pc = self.state.pc; // the call instruction
        match (syn >> 16) & 0xff {
            0 => self.take_exception(EC_SVC64, imm as u64, None, pc.wrapping_add(4)), // SVC
            1 | 2 => {
                // HVC or SMC: the PSCI conduit.
                let args = [
                    self.state.x[0],
                    self.state.x[1],
                    self.state.x[2],
                    self.state.x[3],
                ];
                let (ret, effect) = board.psci.call(self.cpu, args);
                self.state.x[0] = ret;
                match effect {
                    Effect::Stop => return Some(Flow::Stop),
                    Effect::CpuOff => return Some(Flow::CpuOff),
                    Effect::Continue => self.state.pc = self.state.next_pc,
                }
            }
            _ => self.take_exception(EC_BRK, imm as u64, None, pc), // BRK
        }
        None
    }

    // ---- an exception reported by a block (fetch fault / data abort / ERET) ----

    fn handle_exception_exit(&mut self) {
        let syn = self.state.exception_syndrome;
        let pc = self.state.pc;
        if syn & ERET_MARKER != 0 {
            self.eret();
            return;
        }
        if syn & FETCH_ABORT != 0 {
            let ec = if self.el() == 0 {
                EC_INSN_ABORT_LOWER
            } else {
                EC_INSN_ABORT_SAME
            };
            // The IFSC the fetch recorded: a translation/permission/access fault
            // so the guest can demand-page or signal, or 0x10 (external) when the
            // fetch translated but missed RAM. FAR is the faulting PC.
            let ifsc = syn & 0x3f;
            self.take_exception(ec, ifsc, Some(pc), pc);
            return;
        }
        if self.state.exception_kind == WATCHPOINT_KIND {
            // A watchpoint the memory helper matched: EC 0x34/0x35, FAR set, the
            // access not performed. ISS: the cache-maintenance and write bits.
            let ec = if self.el() == 0 {
                EC_WATCHPOINT_LOWER
            } else {
                EC_WATCHPOINT_SAME
            };
            let write = self.state.exception_syndrome & (1 << 6) != 0;
            let far = self.state.exception_far;
            self.take_exception(ec, 0x22 | ((write as u64) << 6), Some(far), pc);
            self.state.exception_kind = 0;
            return;
        }
        // A data abort the memory helper reported.
        let ec = if self.el() == 0 {
            EC_DATA_ABORT_LOWER
        } else {
            EC_DATA_ABORT_SAME
        };
        let far = self.state.exception_far;
        // ISS: ISV=0, so just the fault status and the write bit the helper set.
        let iss = self.state.exception_syndrome & 0x7f;
        self.state.exception_kind = 0;
        self.take_exception(ec, iss, Some(far), pc);
    }

    pub(crate) fn take_data_abort(&mut self, fault: mmu::Fault) {
        if self.trace.mem {
            let far = self.state.exception_far;
            crate_core::warn!(
                "cpu{} data abort far={far:#x} fsc={:#x} write={} pc={:#x} el={}",
                self.cpu,
                fault.fsc,
                fault.write,
                self.state.pc,
                self.el()
            );
        }
        // ESR for a data abort from the current EL: EC 0x24/0x25. The dispatcher
        // builds the full ESR; here record the fault class and FAR.
        self.state.exception_syndrome = fault.fsc as u64 | ((fault.write as u64) << 6);
        self.state.exception_kind = DATA_ABORT_KIND;
    }

    /// Checks a watchpoint before an access; on a match it records a watchpoint
    /// exception and returns an error so the access is not performed.
    pub(crate) fn watchpoint(
        &mut self,
        addr: u64,
        size: u64,
        write: bool,
    ) -> Result<(), mmu::Fault> {
        if self.watchpoint_at(addr, size, write) {
            self.state.exception_far = addr;
            self.state.exception_kind = WATCHPOINT_KIND;
            self.state.exception_syndrome = (write as u64) << 6;
            return Err(mmu::Fault { fsc: 0x22, write });
        }
        Ok(())
    }

    // ---- exceptions ----

    fn take_exception(&mut self, ec: u64, iss: u64, far: Option<u64>, elr: u64) {
        self.state.banked.esr_el1 = (ec << 26) | (1 << 25) | (iss & 0x1ff_ffff);
        self.take_exception_kind(ExcKind::Sync, None, far, elr);
    }

    fn take_exception_kind(
        &mut self,
        kind: ExcKind,
        _iss: Option<u64>,
        far: Option<u64>,
        elr: u64,
    ) {
        let from_el = self.el();
        // Taking an exception clears the local exclusive monitor.
        self.state.monitor = MONITOR_CLEAR;
        if let Some(f) = far {
            self.state.banked.far_el1 = f;
        }
        self.state.banked.elr_el1 = elr;
        self.state.banked.spsr_el1 = self.pack_spsr();

        // Bank the stack pointer when coming from EL0.
        if from_el == 0 {
            self.state.banked.sp_el0 = self.state.sp;
            self.state.sp = self.state.banked.sp_el1;
        }
        self.set_el(1);
        self.state.pstate |= PSTATE_SPSEL | PSTATE_DAIF; // EL1h, all masked
        self.state.pstate &= !PSTATE_SS;

        let base = self.state.banked.vbar_el1 & !0x7ff;
        let origin = if from_el == 0 { 0x400 } else { 0x200 };
        let category = match kind {
            ExcKind::Sync => 0x000,
            ExcKind::Irq => 0x080,
            ExcKind::Fiq => 0x100,
        };
        self.state.pc = base + origin + category;
    }

    fn eret(&mut self) {
        let spsr = self.state.banked.spsr_el1;
        let target_el = ((spsr >> 2) & 3) as u32;
        let from_el = self.el();
        // Unbank the stack pointer when returning to EL0.
        if target_el == 0 && from_el == 1 {
            self.state.banked.sp_el1 = self.state.sp;
            self.state.sp = self.state.banked.sp_el0;
        }
        self.state.monitor = MONITOR_CLEAR;
        self.state.nzcv = spsr & 0xf000_0000;
        self.state.pstate = (spsr & (0xf << 6)) // DAIF
            | (spsr & PSTATE_SS) // software-step
            | (((spsr >> 2) & 3) << PSTATE_EL_SHIFT)
            | (spsr & PSTATE_SPSEL);
        self.state.pc = self.state.banked.elr_el1;
        if self.masked_pending {
            self.recheck_interrupts();
        }
    }

    fn pack_spsr(&self) -> u64 {
        let m = ((self.el() as u64) << 2) | (self.state.pstate & PSTATE_SPSEL);
        (self.state.nzcv & 0xf000_0000)
            | (self.state.pstate & (0xf << 6))
            | (self.state.pstate & PSTATE_SS)
            | m
    }

    // ---- self-hosted debug ----

    /// Whether an enabled, unmasked breakpoint matches the instruction at `pc`
    /// for the current EL.
    fn breakpoint_at(&self, pc: u64) -> bool {
        if self.state.pstate & PSTATE_D != 0 {
            return false; // debug exceptions are masked
        }
        let el = self.el();
        for n in 0..6 {
            let bcr = self.state.debug.bcr[n];
            if bcr & 1 == 0 {
                continue; // not enabled
            }
            // Address-match breakpoint (BT == 0). PMC (bits 2:1) gives the EL:
            // 0b01 = EL1, 0b10 = EL0, 0b11 = both.
            let pmc = (bcr >> 1) & 3;
            let el_ok = match el {
                0 => pmc & 2 != 0,
                _ => pmc & 1 != 0,
            };
            if el_ok && self.state.debug.bvr[n] & !0x3 == pc & !0x3 {
                return true;
            }
        }
        false
    }

    /// Takes a hardware breakpoint exception: the instruction at `pc` has not
    /// run, and ELR points at it.
    fn take_breakpoint(&mut self, pc: u64) {
        let ec = if self.el() == 0 {
            EC_BREAKPOINT_LOWER
        } else {
            EC_BREAKPOINT_SAME
        };
        self.take_exception(ec, 0x22, None, pc);
    }

    /// Takes a software-step exception in the active-pending state, before the
    /// instruction at `pc` runs; ELR points at that instruction. The dispatcher
    /// manages PSTATE.SS (cleared once a stepped instruction completes).
    fn take_step(&mut self, pc: u64) {
        let ec = if self.el() == 0 {
            EC_SOFTWARE_STEP_LOWER
        } else {
            EC_SOFTWARE_STEP_SAME
        };
        // ISS.ISV = 0, IFSC = 0b100010 (debug exception).
        self.take_exception(ec, 0x22, None, pc);
    }

    /// Whether an enabled, unmasked watchpoint matches a `size`-byte access at
    /// `addr` of the given direction; the helper calls this before the access.
    fn watchpoint_at(&self, addr: u64, size: u64, write: bool) -> bool {
        if self.state.pstate & PSTATE_D != 0 {
            return false;
        }
        for n in 0..4 {
            let wcr = self.state.debug.wcr[n];
            if wcr & 1 == 0 {
                continue;
            }
            // LSC (bits 4:3): bit 0 load, bit 1 store.
            let lsc = (wcr >> 3) & 3;
            if (write && lsc & 2 == 0) || (!write && lsc & 1 == 0) {
                continue;
            }
            // BAS (bits 12:5) selects which bytes of the 8-byte-aligned DBGWVR
            // are watched; MASK (bits 28:24) widens the region.
            let mask_bits = (wcr >> 24) & 0x1f;
            let region = if mask_bits >= 3 {
                !((1u64 << mask_bits) - 1)
            } else {
                !0x7
            };
            let base = self.state.debug.wvr[n] & region;
            let bas = (wcr >> 5) & 0xff;
            // The access overlaps a watched byte of this doubleword.
            let aligned = addr & !0x7;
            if aligned & region == base {
                for b in 0..size {
                    let byte = (addr.wrapping_add(b) & 0x7) as u32;
                    if mask_bits >= 3 || bas & (1 << byte) != 0 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Whether any self-hosted debug exception is armed, so blocks must be
    /// translated one instruction at a time and the dispatcher must check for
    /// breakpoints and steps.
    pub(crate) fn debug_active(&self) -> bool {
        let mdscr = self.state.banked.mdscr_el1;
        let breakpoints = mdscr & (1 << 15) != 0 // MDE
            && self.state.debug.bcr.iter().any(|&c| c & 1 != 0);
        let watchpoints = self.state.debug.wcr.iter().any(|&c| c & 1 != 0);
        let step = mdscr & 1 != 0; // SS
        (breakpoints || watchpoints || step) && self.state.debug.oslock & 2 == 0
    }

    // ---- the generic timer ----

    /// The guest's counter: the host's own (the guest and host share the
    /// counter, so translated code reads CNTVCT_EL0 directly).
    fn guest_counter(&self) -> u64 {
        crate_host::jit::counter::now()
    }

    /// When either architected timer would next fire, in counter ticks, if
    /// either is armed, unmasked and not yet firing.
    fn timer_deadline(&self) -> Option<u64> {
        let mut deadline: Option<u64> = None;
        for (ctl, cval) in [
            (self.state.banked.cntv_ctl, self.state.banked.cntv_cval),
            (self.state.banked.cntp_ctl, self.state.banked.cntp_cval),
        ] {
            if ctl & 1 == 0 || ctl & 2 != 0 {
                continue; // disabled or masked
            }
            deadline = Some(deadline.map_or(cval, |x| x.min(cval)));
        }
        deadline
    }

    /// Raises or lowers each architected timer's interrupt from its compare
    /// value, and tells the alarm when to look next. CNTVCT and CNTPCT share
    /// the one counter (CNTVOFF is 0 for an EL1 guest), so both compare
    /// against it.
    fn service_timer(&mut self, board: &Board) {
        let now = self.guest_counter();
        let vctl = self.state.banked.cntv_ctl;
        let vfired = vctl & 1 != 0 && now >= self.state.banked.cntv_cval;
        let vline = vfired && vctl & 2 == 0;
        self.timer_fired = [vfired, false];
        if vline != self.vtimer_line {
            if self.trace.irq {
                crate_core::warn!("cpu{} vtimer line {vline}", self.cpu);
            }
            self.vtimer_line = vline;
            board.lines.set_ppi(self.cpu, board.vtimer_ppi, vline);
        }
        let pctl = self.state.banked.cntp_ctl;
        let pfired = pctl & 1 != 0 && now >= self.state.banked.cntp_cval;
        let pline = pfired && pctl & 2 == 0;
        self.timer_fired[1] = pfired;
        if pline != self.ptimer_line {
            if self.trace.irq {
                crate_core::warn!("cpu{} ptimer line {pline}", self.cpu);
            }
            self.ptimer_line = pline;
            board.lines.set_ppi(self.cpu, board.ptimer_ppi, pline);
        }
        // The next time a timer that is not firing yet would.
        let mut next: Option<u64> = None;
        for (ctl, cval, fired) in [
            (vctl, self.state.banked.cntv_cval, vfired),
            (pctl, self.state.banked.cntp_cval, pfired),
        ] {
            if ctl & 1 != 0 && ctl & 2 == 0 && !fired {
                next = Some(next.map_or(cval, |n| n.min(cval)));
            }
        }
        if let Some(smp) = self.smp() {
            smp.alarm().set(self.cpu, next);
        }
    }

    /// CNTx_CTL (0 virtual, 1 physical) with ISTATUS (bit 2) as the last
    /// `service_timer` found it, so a guest never sees the condition before
    /// the interrupt line shows it.
    fn timer_ctl(&self, which: usize) -> u64 {
        let ctl = if which == 0 {
            self.state.banked.cntv_ctl
        } else {
            self.state.banked.cntp_ctl
        };
        (ctl & 3) | ((self.timer_fired[which] as u64) << 2)
    }
}

#[derive(Clone, Copy)]
enum ExcKind {
    Sync,
    Irq,
    Fiq,
}

/// The last guest PC each CPU's dispatcher entered, for diagnostics.
static LAST_PC: [std::sync::atomic::AtomicU64; 32] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 32];
static DUMP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static MACHINE: std::sync::atomic::AtomicPtr<Smp> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

fn dump_requested() -> bool {
    DUMP.load(Ordering::Relaxed)
}

/// SIGUSR1: print where each CPU is, and ask every CPU to report its
/// registers. A boot watcher sends it before it gives up on a silent guest.
extern "C" fn on_dump_signal(_sig: libc::c_int) {
    let mut buf = [0u8; 64];
    for (cpu, pc) in LAST_PC.iter().enumerate() {
        let pc = pc.load(Ordering::Relaxed);
        if pc == 0 {
            continue;
        }
        let mut n = 0;
        for &b in b"crate: cpu" {
            buf[n] = b;
            n += 1;
        }
        buf[n] = b'0' + (cpu % 10) as u8;
        n += 1;
        for &b in b" last pc 0x" {
            buf[n] = b;
            n += 1;
        }
        for i in (0..16).rev() {
            let d = ((pc >> (i * 4)) & 0xf) as u8;
            buf[n] = if d < 10 { b'0' + d } else { b'a' + d - 10 };
            n += 1;
        }
        buf[n] = b'\n';
        n += 1;
        // SAFETY: write(2) is async-signal-safe; the buffer is ours.
        unsafe { libc::write(2, buf.as_ptr().cast(), n) };
    }
    DUMP.store(true, Ordering::Relaxed);
    let smp = MACHINE.load(Ordering::Acquire);
    // SAFETY: the machine outlives the handler's registration; kicking only
    // stores to atomics and notifies condition variables... which is not
    // async-signal-safe, so only the exit requests are set here.
    if let Some(smp) = unsafe { smp.as_ref() } {
        smp.kick_requests();
    }
}

fn install_dump_handler() {
    // SAFETY: installs a handler that only does async-signal-safe work.
    unsafe {
        let mut act: libc::sigaction = std::mem::zeroed();
        act.sa_sigaction = on_dump_signal as *const () as usize;
        act.sa_flags = libc::SA_RESTART;
        libc::sigaction(libc::SIGUSR1, &act, std::ptr::null_mut());
    }
}

/// Where the CPUs' translated code goes.
enum CodeMemory {
    /// A region of this many bytes for each CPU, of its own.
    PerCpu(usize),
    /// One region for the machine, a slice of it for each CPU.
    Shared(Arc<JitRegion>),
}

/// The least code region a CPU can work in: the stubs and the largest block.
const MIN_CPU_REGION: usize = 1 << 20;

/// The CPU backend: runs each of a board's CPUs on the translator, one host
/// thread per CPU.
pub struct Jit {
    memory: CodeMemory,
    fast: bool,
}

impl Jit {
    pub fn new() -> Self {
        Jit {
            memory: CodeMemory::PerCpu(crate::cache::MAX_REGION),
            fast: !slow_from_env(),
        }
    }

    /// The backend with the fast paths on or off.
    pub fn with_fast_paths(fast: bool) -> Self {
        Jit {
            fast,
            ..Self::new()
        }
    }

    /// Each CPU's code in a region of `bytes` of its own; a CPU uses at most
    /// `cache::MAX_REGION` of it.
    pub fn with_code_size(self, bytes: usize) -> Self {
        Jit {
            memory: CodeMemory::PerCpu(bytes),
            ..self
        }
    }

    /// Every CPU's code in an equal slice of `region` rather than in a region
    /// of its own: how a host that grants one region a launch (iOS) runs
    /// several CPUs. Each slice is at most `cache::MAX_REGION`.
    pub fn in_region(self, region: Arc<JitRegion>) -> Self {
        Jit {
            memory: CodeMemory::Shared(region),
            ..self
        }
    }

    fn caches(&self, count: usize) -> crate_core::Result<Vec<CodeCache>> {
        match &self.memory {
            CodeMemory::PerCpu(bytes) => (0..count)
                .map(|_| CodeCache::with_mode(*bytes, self.fast))
                .collect(),
            CodeMemory::Shared(region) => {
                let slices = region.clone().split(count, crate::cache::MAX_REGION)?;
                if slices[0].len() < MIN_CPU_REGION {
                    return Err(crate_core::Error::Jit(format!(
                        "{} KiB of code for each of {count} CPUs; each needs {} KiB",
                        slices[0].len() >> 10,
                        MIN_CPU_REGION >> 10
                    )));
                }
                crate_core::debug!(
                    "{count} CPUs in {} MiB of code, {} KiB each",
                    region.len() >> 20,
                    slices[0].len() >> 10
                );
                Ok(slices
                    .into_iter()
                    .map(|slice| CodeCache::in_region(slice, self.fast))
                    .collect())
            }
        }
    }
}

impl Default for Jit {
    fn default() -> Self {
        Self::new()
    }
}

impl crate_core::board::CpuBackend for Jit {
    fn name(&self) -> &'static str {
        "jit"
    }

    fn run(&self, board: &Board) -> crate_core::Result<StopReason> {
        let count = board.cpu_count();
        let smp = Arc::new(Smp::new(count, board.run.clone()));
        let mut cpus: Vec<Box<Cpu>> = (0..count).map(|c| Cpu::for_board(board, &smp, c)).collect();
        let caches = self.caches(count)?;
        for (c, cpu) in cpus.iter_mut().enumerate() {
            smp.attach(c, &mut *cpu.state);
        }
        // The interrupt controller wakes a CPU through the machine.
        board.interrupts.set_waker(smp.clone());
        {
            let smp = Arc::downgrade(&smp);
            board.run.on_stop(move || {
                if let Some(smp) = smp.upgrade() {
                    smp.kick_all();
                    smp.stop_alarm();
                }
            });
        }
        let smp_ref: &Smp = &smp;
        MACHINE.store(smp_ref as *const Smp as *mut Smp, Ordering::Release);
        install_dump_handler();
        std::thread::scope(|scope| {
            scope.spawn(|| smp_ref.run_alarm());
            for (c, (mut cpu, mut cache)) in cpus.into_iter().zip(caches).enumerate() {
                std::thread::Builder::new()
                    .name(format!("crate-jit{c}"))
                    .spawn_scoped(scope, move || cpu.power_cycle(board, &mut cache))
                    .expect("spawning a vCPU thread");
            }
        });
        MACHINE.store(std::ptr::null_mut(), Ordering::Release);
        Ok(board.run.reason().unwrap_or(StopReason::Killed))
    }
}

impl Cpu {
    /// Starts the CPU each time it is turned on, until the machine stops.
    fn power_cycle(&mut self, board: &Board, cache: &mut CodeCache) {
        let mut start = (self.cpu == 0).then_some((board.boot.pc, board.boot.x));
        loop {
            let (pc, x) = match start.take() {
                Some(s) => s,
                None => match board.psci.wait_for_on(self.cpu) {
                    Some(entry) => (entry.pc, [entry.context, 0, 0, 0]),
                    None => return,
                },
            };
            crate_core::debug!("CPU {} starts at {pc:#x}", self.cpu);
            self.reset(pc, x);
            let flow = self.dispatch(cache);
            if self.trace.stats {
                let top: Vec<String> = {
                    let mut v: Vec<_> = self.counters.sysregs.iter().collect();
                    v.sort_by(|a, b| b.1.cmp(a.1));
                    v.iter()
                        .take(12)
                        .map(|(r, n)| format!("{:?}={n}", SysReg(**r)))
                        .collect()
                };
                let mut c = std::mem::take(&mut self.counters);
                c.sysregs.clear();
                c.jc_misses = self.state.jc_misses;
                c.jc_lookups = self.state.jc_lookups;
                eprintln!(
                    "crate: cpu{} {c:?} blocks={} code={}K sysregs: {}",
                    self.cpu,
                    cache.block_count(),
                    cache.used() >> 10,
                    top.join(" ")
                );
            }
            match flow {
                Flow::Stop => return,
                Flow::CpuOff => {
                    if self.vtimer_line {
                        board.lines.set_ppi(self.cpu, board.vtimer_ppi, false);
                    }
                    if self.ptimer_line {
                        board.lines.set_ppi(self.cpu, board.ptimer_ppi, false);
                    }
                    if let Some(smp) = self.smp() {
                        smp.alarm().set(self.cpu, None);
                    }
                    crate_core::debug!("CPU {} off", self.cpu);
                }
            }
        }
    }
}

/// System-register reads and writes the translator traps.
impl Cpu {
    fn handle_sysreg(&mut self, board: &Board, cache: &mut CodeCache) {
        let reg = SysReg(self.state.sysreg as u16);
        let rt = self.state.sysreg_rt;
        let read = self.state.sysreg_read != 0;
        let next = self.state.next_pc;

        if self.state.sysreg as u16 == SYSTEM_WORD_MARKER {
            // An MSR-immediate or a maintenance op, the whole word in syndrome.
            let word = self.state.exception_syndrome as u32;
            self.handle_system_word(word, cache);
            self.state.pc = next;
            return;
        }

        if read {
            match self.read_sysreg(board, reg) {
                Some(v) => {
                    if rt != 31 {
                        self.state.x[rt as usize] = v;
                    }
                    self.state.pc = next;
                }
                None => {
                    let pc = self.state.pc;
                    self.take_exception(EC_UNKNOWN, 0, None, pc); // unimplemented sysreg: undefined
                }
            }
        } else {
            let value = if rt == 31 {
                0
            } else {
                self.state.x[rt as usize]
            };
            if self.write_sysreg(board, reg, value) {
                self.state.pc = next;
            } else {
                let pc = self.state.pc;
                self.take_exception(EC_UNKNOWN, 0, None, pc); // unimplemented sysreg: undefined
            }
        }
    }

    fn handle_system_word(&mut self, word: u32, cache: &mut CodeCache) {
        // MSR (immediate): CRm holds the value, op1:op2 the field.
        if (word >> 19) & 3 == 0 && (word >> 12) & 0xf == 0b0100 {
            let op1 = (word >> 16) & 7;
            let op2 = (word >> 5) & 7;
            let crm = (word >> 8) & 0xf;
            match (op1, op2) {
                (0b000, 0b101) => {
                    // SPSel.
                    self.state.pstate =
                        (self.state.pstate & !PSTATE_SPSEL) | ((crm as u64 & 1) * PSTATE_SPSEL);
                }
                (0b011, 0b110) => self.set_daif(crm, true), // DAIFSet
                (0b011, 0b111) => self.set_daif(crm, false), // DAIFClr
                _ => {}
            }
            return;
        }
        let op1 = (word >> 16) & 7;
        let crn = (word >> 12) & 0xf;
        let crm = (word >> 8) & 0xf;
        let op2 = (word >> 5) & 7;
        let rt = word & 31;
        let xt = if rt == 31 {
            0
        } else {
            self.state.x[rt as usize]
        };
        if crn == 0b0011 && (word >> 19) & 3 == 0 {
            // CLREX: clear the exclusive monitor.
            self.state.monitor = MONITOR_CLEAR;
        } else if crn == 0b1000 && op1 == 0 {
            // TLBI at EL1: CRm 0011 inner shareable (and 0001 outer), 0111 local.
            let broadcast = crm != 0b0111;
            // VA[55:12] is in bits 43:0, sign-extended from VA bit 55; the ASID
            // in bits 63:48.
            let va = (((xt << 20) as i64) >> 8) as u64 & !0xfff;
            let width = if self.state.mmu.tcr_el1 & (1 << 36) != 0 {
                0xffff
            } else {
                0xff
            };
            let asid = (xt >> 48) as u16 & width;
            let op = match (crm, op2) {
                (0b0011 | 0b0111 | 0b0001, 0) => Remote::FlushAll, // VMALLE1
                (0b0011 | 0b0111 | 0b0001, 1 | 5) => Remote::FlushVa(va, Some(asid)), // VA(L)E1
                (0b0011 | 0b0111 | 0b0001, 3 | 7) => Remote::FlushVa(va, None), // VAA(L)E1
                (0b0011 | 0b0111 | 0b0001, 2) => Remote::FlushUser(asid), // ASIDE1
                _ => Remote::FlushAll, // range forms and the rest: everything
            };
            self.maintain(op, broadcast, cache);
        } else if crn == 0b0111 {
            match (op1, crm, op2) {
                // AT (address translation): translate Rt's VA, set PAR_EL1.
                (_, 0b1000 | 0b1001, _) => self.address_translate(word),
                // IC IALLUIS / IC IALLU: drop every translation of code.
                (0, 0b0001, 0) => self.maintain(Remote::CodeAll, true, cache),
                (0, 0b0101, 0) => self.maintain(Remote::CodeAll, false, cache),
                // IC IVAU: the code of the page holding Rt's VA, everywhere.
                (3, 0b0101, 1) => {
                    let el = self.el();
                    let regs = self.state.mmu;
                    if let Ok(t) = mmu::translate(&self.mem, &regs, el, xt, Access::Read) {
                        self.maintain(Remote::CodePage(t.pa & !0xfff), true, cache);
                    }
                }
                _ => {}
            }
        }
    }

    /// AT S1E{0,1}{R,W}: do a stage-1 translation of the VA in Rt and record
    /// the result (a physical address, or a fault) in PAR_EL1.
    fn address_translate(&mut self, word: u32) {
        let rt = (word & 31) as usize;
        let va = if rt == 31 { 0 } else { self.state.x[rt] };
        let op2 = (word >> 5) & 7;
        // op2: 0 S1E1R, 1 S1E1W, 2 S1E0R, 3 S1E0W (4..7 are the EL2/unpriv forms).
        let el = if op2 & 0b010 != 0 { 0 } else { 1 };
        let write = op2 & 1 != 0;
        let regs = self.state.mmu;
        let access = if write { Access::Write } else { Access::Read };
        self.state.banked.par_el1 = match mmu::translate(&self.mem, &regs, el, va, access) {
            // F=0: the physical address in bits [51:12], attributes left zero.
            Ok(t) => t.pa & 0x000f_ffff_ffff_f000,
            // F=1 (bit 0), with the fault status in bits [6:1]; stage-1 (bit 9 = 0).
            Err(f) => 1 | ((f.fsc as u64) << 1),
        };
    }

    fn set_daif(&mut self, crm: u32, set: bool) {
        // DAIF immediate uses bit3=D, bit2=A, bit1=I, bit0=F, each at CRm.
        let mask = (((crm >> 3) & 1) as u64) << 9
            | (((crm >> 2) & 1) as u64) << 8
            | (((crm >> 1) & 1) as u64) << 7
            | ((crm & 1) as u64) << 6;
        if set {
            self.state.pstate |= mask;
        } else {
            self.state.pstate &= !mask;
            self.recheck_interrupts();
        }
    }

    /// Reads a system register, `None` if it is not one the machine implements
    /// (which the caller turns into an undefined-instruction exception).
    fn read_sysreg(&mut self, board: &Board, reg: SysReg) -> Option<u64> {
        // The GIC's CPU interface owns the ICC_* registers.
        if let Some(v) = board.interrupts.read_sysreg(self.cpu, reg) {
            // Acknowledging an interrupt changes what is pending.
            self.recheck_interrupts();
            return Some(v);
        }
        // A guest polling a timer must see its interrupt as soon as it sees
        // the condition (the alarm that would raise it may not have run yet).
        if matches!(
            reg,
            r if r == sr::CNTV_CTL_EL0 || r == sr::CNTP_CTL_EL0 || r == sr::CNTV_TVAL_EL0 || r == sr::CNTP_TVAL_EL0
        ) {
            self.service_timer(board);
        }
        let b = &self.state.banked;
        Some(match reg {
            r if r == sr::SCTLR_EL1 => self.state.mmu.sctlr_el1,
            r if r == sr::TTBR0_EL1 => self.state.mmu.ttbr0_el1,
            r if r == sr::TTBR1_EL1 => self.state.mmu.ttbr1_el1,
            r if r == sr::TCR_EL1 => self.state.mmu.tcr_el1,
            r if r == sr::MAIR_EL1 => self.state.mmu.mair_el1,
            r if r == sr::AMAIR_EL1 => self.state.mmu.amair_el1,
            r if r == sr::CONTEXTIDR_EL1 => self.state.mmu.contextidr_el1,
            r if r == sr::PAR_EL1 => b.par_el1,
            r if r == sr::VBAR_EL1 => b.vbar_el1,
            r if r == sr::ELR_EL1 => b.elr_el1,
            r if r == sr::SPSR_EL1 => b.spsr_el1,
            r if r == sr::ESR_EL1 => b.esr_el1,
            r if r == sr::FAR_EL1 => b.far_el1,
            r if r == sr::SP_EL0 => b.sp_el0,
            r if r == sr::SP_EL1 => b.sp_el1,
            r if r == sr::TPIDR_EL0 => b.tpidr_el0,
            r if r == sr::TPIDRRO_EL0 => b.tpidrro_el0,
            r if r == sr::TPIDR_EL1 => b.tpidr_el1,
            r if r == sr::CPACR_EL1 => b.cpacr_el1,
            r if r == sr::MDSCR_EL1 => b.mdscr_el1,
            r if r == sr::ACTLR_EL1 => b.actlr_el1,
            r if r == sr::CNTKCTL_EL1 => b.cntkctl_el1,
            r if r == sr::CNTFRQ_EL0 => crate_host::jit::counter::frequency(),
            r if r == sr::CNTVCT_EL0 || r == sr::CNTPCT_EL0 => self.guest_counter(),
            r if r == sr::CNTV_CTL_EL0 => self.timer_ctl(0),
            r if r == sr::CNTV_CVAL_EL0 => b.cntv_cval,
            r if r == sr::CNTV_TVAL_EL0 => {
                (b.cntv_cval.wrapping_sub(self.guest_counter())) & 0xffff_ffff
            }
            r if r == sr::CNTP_CTL_EL0 => self.timer_ctl(1),
            r if r == sr::CNTP_CVAL_EL0 => b.cntp_cval,
            r if r == sr::CNTP_TVAL_EL0 => {
                (b.cntp_cval.wrapping_sub(self.guest_counter())) & 0xffff_ffff
            }
            r if r == sr::NZCV => self.state.nzcv,
            r if r == sr::DAIF => self.state.pstate & (0xf << 6),
            r if r == sr::FPCR => fpcr(),
            r if r == sr::FPSR => fpsr(),
            r if r == sr::CURRENT_EL => (self.el() as u64) << 2,
            r if r == sr::SPSEL => self.state.pstate & PSTATE_SPSEL,
            r if r == sr::MPIDR_EL1 => self.mpidr,
            r if r == sr::MIDR_EL1 => MIDR_EL1,
            r if r == sr::CTR_EL0 => CTR_EL0,
            r if r == sr::DCZID_EL0 => DCZID_EL0,
            r if r == sr::CLIDR_EL1 => 0, // no cache levels advertised
            // Implementation defined; nothing to say. Linux 7.x reads it into
            // its CPU information at boot.
            r if r == sr::AIDR_EL1 => 0,
            r if r == sr::ID_AA64MMFR0_EL1 => ID_AA64MMFR0,
            r if r == sr::ID_AA64PFR0_EL1 => ID_AA64PFR0,
            r if r == sr::ID_AA64DFR0_EL1 => ID_AA64DFR0,
            r if r == sr::ID_AA64ISAR0_EL1 => ID_AA64ISAR0,
            r if r == sr::ID_AA64ISAR1_EL1 => ID_AA64ISAR1,
            r if r == sr::OSLSR_EL1 => {
                // OSLM = 0b10 (the v8 model), OSLK (bit 1) from the OS lock.
                0b1000 | (self.state.debug.oslock & 2)
            }
            // The hardware breakpoint and watchpoint value/control registers.
            _ if debug_reg(reg).is_some() => {
                let (which, n) = debug_reg(reg).unwrap();
                match which {
                    DebugReg::Bvr => self.state.debug.bvr[n],
                    DebugReg::Bcr => self.state.debug.bcr[n],
                    DebugReg::Wvr => self.state.debug.wvr[n],
                    DebugReg::Wcr => self.state.debug.wcr[n],
                }
            }
            // The rest of the ID register space reads as zero (RAZ).
            r if is_id_space(r) => 0,
            // The remaining debug and OS-lock registers: quiet.
            r if is_quiet(r) => 0,
            _ => return None,
        })
    }

    /// Writes a system register, `false` if it is not one the machine knows.
    fn write_sysreg(&mut self, board: &Board, reg: SysReg, value: u64) -> bool {
        if board.interrupts.write_sysreg(self.cpu, reg, value) {
            // An EOI, a priority mask or an SGI to this CPU may change what is
            // pending.
            self.recheck_interrupts();
            return true;
        }
        match reg {
            r if r == sr::SCTLR_EL1 => {
                // The MMU, WXN or endianness changing changes every translation.
                let old = self.state.mmu.sctlr_el1;
                self.state.mmu.sctlr_el1 = value;
                if (old ^ value) & (1 | 1 << 19 | 1 << 25) != 0 {
                    self.flush_all();
                }
                self.context_changed();
            }
            r if r == sr::TTBR0_EL1 => {
                if self.state.mmu.ttbr0_el1 != value {
                    self.state.mmu.ttbr0_el1 = value;
                    self.context_changed();
                    self.flush_low();
                }
            }
            r if r == sr::TTBR1_EL1 => {
                let old = self.state.mmu.ttbr1_el1;
                self.state.mmu.ttbr1_el1 = value;
                const BADDR: u64 = 0x0000_ffff_ffff_fffe;
                if (old ^ value) & BADDR != 0 {
                    self.flush_all();
                } else if old != value {
                    // Only the ASID changed (TCR.A1 puts it here): another
                    // address space.
                    self.context_changed();
                    self.flush_low();
                }
            }
            r if r == sr::TCR_EL1 => {
                if self.state.mmu.tcr_el1 != value {
                    self.state.mmu.tcr_el1 = value;
                    self.flush_all();
                }
            }
            r if r == sr::MAIR_EL1 => self.state.mmu.mair_el1 = value,
            r if r == sr::AMAIR_EL1 => self.state.mmu.amair_el1 = value,
            r if r == sr::CONTEXTIDR_EL1 => self.state.mmu.contextidr_el1 = value,
            r if r == sr::PAR_EL1 => self.state.banked.par_el1 = value,
            r if r == sr::VBAR_EL1 => self.state.banked.vbar_el1 = value,
            r if r == sr::ELR_EL1 => self.state.banked.elr_el1 = value,
            r if r == sr::SPSR_EL1 => self.state.banked.spsr_el1 = value,
            r if r == sr::ESR_EL1 => self.state.banked.esr_el1 = value,
            r if r == sr::FAR_EL1 => self.state.banked.far_el1 = value,
            r if r == sr::SP_EL0 => self.state.banked.sp_el0 = value,
            r if r == sr::SP_EL1 => self.state.banked.sp_el1 = value,
            r if r == sr::TPIDR_EL0 => self.state.banked.tpidr_el0 = value,
            r if r == sr::TPIDRRO_EL0 => self.state.banked.tpidrro_el0 = value,
            r if r == sr::TPIDR_EL1 => self.state.banked.tpidr_el1 = value,
            r if r == sr::CPACR_EL1 => self.state.banked.cpacr_el1 = value,
            r if r == sr::ACTLR_EL1 => self.state.banked.actlr_el1 = value,
            r if r == sr::CNTKCTL_EL1 => self.state.banked.cntkctl_el1 = value,
            r if r == sr::CNTV_CTL_EL0 => {
                self.state.banked.cntv_ctl = value & 0x3;
                self.service_timer(board);
            }
            r if r == sr::CNTV_CVAL_EL0 => {
                self.state.banked.cntv_cval = value;
                self.service_timer(board);
            }
            r if r == sr::CNTV_TVAL_EL0 => {
                self.state.banked.cntv_cval = self
                    .guest_counter()
                    .wrapping_add(value as i32 as i64 as u64);
                self.service_timer(board);
            }
            r if r == sr::CNTP_CTL_EL0 => {
                self.state.banked.cntp_ctl = value & 0x3;
                self.service_timer(board);
            }
            r if r == sr::CNTP_CVAL_EL0 => {
                self.state.banked.cntp_cval = value;
                self.service_timer(board);
            }
            r if r == sr::CNTP_TVAL_EL0 => {
                self.state.banked.cntp_cval = self
                    .guest_counter()
                    .wrapping_add(value as i32 as i64 as u64);
                self.service_timer(board);
            }
            r if r == sr::NZCV => self.state.nzcv = value & 0xf000_0000,
            r if r == sr::DAIF => {
                self.state.pstate = (self.state.pstate & !(0xf << 6)) | (value & (0xf << 6));
                self.recheck_interrupts();
            }
            r if r == sr::FPCR => set_fpcr(value),
            r if r == sr::FPSR => set_fpsr(value),
            r if r == sr::SPSEL => {
                self.state.pstate = (self.state.pstate & !PSTATE_SPSEL) | (value & PSTATE_SPSEL)
            }
            r if r == sr::MDSCR_EL1 => {
                self.state.banked.mdscr_el1 = value;
                // MDSCR.SS / MDE change whether blocks must step; re-translate.
                self.state.flush_cache = 1;
            }
            r if r == sr::OSLAR_EL1 => {
                self.state.debug.oslock = (value & 1) << 1;
                self.state.flush_cache = 1;
            }
            // A hardware breakpoint or watchpoint register; a control write can
            // change whether debug is active, so re-translate.
            _ if debug_reg(reg).is_some() => {
                let (which, n) = debug_reg(reg).unwrap();
                match which {
                    DebugReg::Bvr => self.state.debug.bvr[n] = value,
                    DebugReg::Bcr => {
                        self.state.debug.bcr[n] = value;
                        self.state.flush_cache = 1;
                    }
                    DebugReg::Wvr => self.state.debug.wvr[n] = value,
                    DebugReg::Wcr => {
                        self.state.debug.wcr[n] = value;
                        self.state.flush_cache = 1;
                    }
                }
            }
            // Cache-size selection and the remaining debug/OS-lock registers:
            // accepted, no effect here.
            r if r == sr::CSSELR_EL1 || is_quiet(r) => {}
            _ => return false,
        }
        true
    }
}

/// The hardware debug registers, by kind and index.
enum DebugReg {
    Bvr,
    Bcr,
    Wvr,
    Wcr,
}

/// Decodes a debug breakpoint/watchpoint register (op0 2, op1 0, CRn 0, op2
/// 4..7, CRm the index) into its kind and index.
fn debug_reg(r: SysReg) -> Option<(DebugReg, usize)> {
    if r.op0() != 2 || r.op1() != 0 || r.crn() != 0 {
        return None;
    }
    let n = r.crm() as usize;
    match r.op2() {
        4 if n < 6 => Some((DebugReg::Bvr, n)),
        5 if n < 6 => Some((DebugReg::Bcr, n)),
        6 if n < 4 => Some((DebugReg::Wvr, n)),
        7 if n < 4 => Some((DebugReg::Wcr, n)),
        _ => None,
    }
}

/// CTR_EL0: 64-byte I- and D-cache lines (IminLine and DminLine 4), a PIPT
/// instruction cache (L1Ip 0b11), and IDC
/// set so no data-cache cleaning is needed for the instruction path. DIC is
/// left 0 so the guest issues IC after writing code, which the dispatcher
/// honours by dropping the page's translations; without that, rewritten code
/// would run the stale translation.
pub(crate) const CTR_EL0: u64 = (1 << 31) | (1 << 28) | (3 << 24) | (4 << 16) | (3 << 14) | 4;
/// DCZID_EL0: DC ZVA allowed, 64-byte blocks (4 words << 4).
pub(crate) const DCZID_EL0: u64 = 0x4;
/// MIDR_EL1: a generic Cortex-A.
pub(crate) const MIDR_EL1: u64 = 0x410f_d080;
/// ID_AA64MMFR0_EL1: PARange 40-bit (2) and all three granules supported —
/// 4 KiB (TGran4=0), 16 KiB (TGran16=1) and 64 KiB (TGran64=0) — but **without**
/// the 52-bit (LPA2) variants (TGran4=1 / TGran16=2), which the MMU does not
/// model. Advertising LPA2 made Linux build 52-bit page tables and fault.
const ID_AA64MMFR0: u64 = 0x0000_0000_0010_0002;
/// ID_AA64PFR0_EL1: EL0 and EL1 at AArch64 only, FP and AdvSIMD present, no SVE.
const ID_AA64PFR0: u64 = 0x0000_0000_0000_0011;
/// ID_AA64DFR0_EL1: debug architecture v8, 6 breakpoints and 4 watchpoints,
/// PMUVer 0 (so the PMU tests skip rather than fail).
const ID_AA64DFR0: u64 = 0x0000_0000_1035_0006;
/// ID_AA64ISAR0_EL1: the instructions the host runs as they are, copied --
/// AES with PMULL (AES 2), SHA1 (1), SHA256 (SHA2 1), CRC32 (1), RDM (1) and
/// the dot product (DP 1) -- and the LSE atomics (Atomic 2), which the
/// translator turns into the host's own on the host address.
const ID_AA64ISAR0: u64 =
    (1 << 44) | (1 << 28) | (2 << 20) | (1 << 16) | (1 << 12) | (1 << 8) | (2 << 4);
/// ID_AA64ISAR1_EL1: DC CVAP (DPB 1, a no-op here), FJCVTZS (JSCVT 1), the
/// complex-number instructions (FCMA 1) and LDAPR (LRCPC 1).
const ID_AA64ISAR1: u64 = (1 << 20) | (1 << 16) | (1 << 12) | 1;

fn is_id_space(r: SysReg) -> bool {
    r.op0() == 3 && r.op1() == 0 && r.crn() == 0 && (0..8).contains(&r.crm())
}

/// Debug breakpoint/watchpoint registers and the OS lock, kept quiet during
/// bring-up (CRn 0 with op1 0 overlaps the ID space, which is handled first).
fn is_quiet(r: SysReg) -> bool {
    // OSLAR_EL1, OSLSR_EL1, OSDLR_EL1, DBGPRCR, and the DBGB/DBGW banks (CRn 0,
    // op1 0, CRm 0..15, with op2 selecting the register) at the debug op0 2.
    r.op0() == 2
}

/// System-register encodings the machine implements, by their packed form.
#[rustfmt::skip]
mod sr {
    use super::SysReg;
    pub const SCTLR_EL1: SysReg = SysReg::new(3, 0, 1, 0, 0);
    pub const CPACR_EL1: SysReg = SysReg::new(3, 0, 1, 0, 2);
    pub const ACTLR_EL1: SysReg = SysReg::new(3, 0, 1, 0, 1);
    pub const TTBR0_EL1: SysReg = SysReg::new(3, 0, 2, 0, 0);
    pub const TTBR1_EL1: SysReg = SysReg::new(3, 0, 2, 0, 1);
    pub const TCR_EL1: SysReg = SysReg::new(3, 0, 2, 0, 2);
    pub const MAIR_EL1: SysReg = SysReg::new(3, 0, 10, 2, 0);
    pub const AMAIR_EL1: SysReg = SysReg::new(3, 0, 10, 3, 0);
    pub const CONTEXTIDR_EL1: SysReg = SysReg::new(3, 0, 13, 0, 1);
    pub const PAR_EL1: SysReg = SysReg::new(3, 0, 7, 4, 0);
    pub const VBAR_EL1: SysReg = SysReg::new(3, 0, 12, 0, 0);
    pub const ELR_EL1: SysReg = SysReg::new(3, 0, 4, 0, 1);
    pub const SPSR_EL1: SysReg = SysReg::new(3, 0, 4, 0, 0);
    pub const ESR_EL1: SysReg = SysReg::new(3, 0, 5, 2, 0);
    pub const FAR_EL1: SysReg = SysReg::new(3, 0, 6, 0, 0);
    pub const SP_EL0: SysReg = SysReg::new(3, 0, 4, 1, 0);
    pub const SP_EL1: SysReg = SysReg::new(3, 4, 4, 1, 0);
    pub const SPSEL: SysReg = SysReg::new(3, 0, 4, 2, 0);
    pub const DAIF: SysReg = SysReg::new(3, 3, 4, 2, 1);
    pub const NZCV: SysReg = SysReg::new(3, 3, 4, 2, 0);
    pub const FPCR: SysReg = SysReg::new(3, 3, 4, 4, 0);
    pub const FPSR: SysReg = SysReg::new(3, 3, 4, 4, 1);
    pub const CURRENT_EL: SysReg = SysReg::new(3, 0, 4, 2, 2);
    pub const TPIDR_EL0: SysReg = SysReg::new(3, 3, 13, 0, 2);
    pub const TPIDRRO_EL0: SysReg = SysReg::new(3, 3, 13, 0, 3);
    pub const TPIDR_EL1: SysReg = SysReg::new(3, 0, 13, 0, 4);
    pub const MPIDR_EL1: SysReg = SysReg::new(3, 0, 0, 0, 5);
    pub const MIDR_EL1: SysReg = SysReg::new(3, 0, 0, 0, 0);
    pub const CTR_EL0: SysReg = SysReg::new(3, 3, 0, 0, 1);
    pub const DCZID_EL0: SysReg = SysReg::new(3, 3, 0, 0, 7);
    pub const CLIDR_EL1: SysReg = SysReg::new(3, 1, 0, 0, 1);
    pub const AIDR_EL1: SysReg = SysReg::new(3, 1, 0, 0, 7);
    pub const CSSELR_EL1: SysReg = SysReg::new(3, 2, 0, 0, 0);
    pub const ID_AA64PFR0_EL1: SysReg = SysReg::new(3, 0, 0, 4, 0);
    pub const ID_AA64DFR0_EL1: SysReg = SysReg::new(3, 0, 0, 5, 0);
    pub const ID_AA64ISAR0_EL1: SysReg = SysReg::new(3, 0, 0, 6, 0);
    pub const ID_AA64ISAR1_EL1: SysReg = SysReg::new(3, 0, 0, 6, 1);
    pub const ID_AA64MMFR0_EL1: SysReg = SysReg::new(3, 0, 0, 7, 0);
    pub const MDSCR_EL1: SysReg = SysReg::new(2, 0, 0, 2, 2);
    pub const OSLAR_EL1: SysReg = SysReg::new(2, 0, 1, 0, 4);
    pub const OSLSR_EL1: SysReg = SysReg::new(2, 0, 1, 1, 4);
    pub const CNTFRQ_EL0: SysReg = SysReg::new(3, 3, 14, 0, 0);
    pub const CNTPCT_EL0: SysReg = SysReg::new(3, 3, 14, 0, 1);
    pub const CNTVCT_EL0: SysReg = SysReg::new(3, 3, 14, 0, 2);
    pub const CNTKCTL_EL1: SysReg = SysReg::new(3, 0, 14, 1, 0);
    pub const CNTP_TVAL_EL0: SysReg = SysReg::new(3, 3, 14, 2, 0);
    pub const CNTP_CTL_EL0: SysReg = SysReg::new(3, 3, 14, 2, 1);
    pub const CNTP_CVAL_EL0: SysReg = SysReg::new(3, 3, 14, 2, 2);
    pub const CNTV_TVAL_EL0: SysReg = SysReg::new(3, 3, 14, 3, 0);
    pub const CNTV_CTL_EL0: SysReg = SysReg::new(3, 3, 14, 3, 1);
    pub const CNTV_CVAL_EL0: SysReg = SysReg::new(3, 3, 14, 3, 2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epilogue_symbol_resolves() {
        assert_ne!(CodeCache::epilogue_addr(), 0);
    }

    /// Maps two adjacent guest pages to two *non-adjacent* physical frames, so a
    /// misaligned access over the boundary only reads/writes the right bytes if
    /// each page is translated on its own. Returns (cpu, va_base, pa_a, pa_b).
    fn two_page_cpu() -> (Box<Cpu>, u64, u64, u64) {
        use crate_core::memory::{HeapRam, RamBlock};
        const RAM: u64 = 0x4000_0000;
        let block: Arc<dyn RamBlock> = Arc::new(HeapRam::new(0x40_0000).unwrap());
        let mem = Arc::new(GuestMemory::new(vec![(RAM, block)]).unwrap());
        // A four-level 4 KiB walk (T0SZ = 16). All four tables live in one frame
        // each; `va` and `va + 0x1000` share the whole path down to one leaf
        // table, differing only in the level-3 index.
        let (l0, l1, l2, l3) = (
            RAM + 0x10_0000,
            RAM + 0x10_1000,
            RAM + 0x10_2000,
            RAM + 0x10_3000,
        );
        let va = 0x4_0000u64;
        let pa_a = RAM + 0x20_0000; // the frame for `va`
        let pa_b = RAM + 0x30_0000; // the frame for `va + 0x1000`, far from pa_a
        mem.write_u64(l0 + ((va >> 39) & 0x1ff) * 8, l1 | 3)
            .unwrap();
        mem.write_u64(l1 + ((va >> 30) & 0x1ff) * 8, l2 | 3)
            .unwrap();
        mem.write_u64(l2 + ((va >> 21) & 0x1ff) * 8, l3 | 3)
            .unwrap();
        let leaf = |pa: u64| pa | 3 | (1 << 10); // valid page, AF set, AP = EL1 RW
        mem.write_u64(l3 + ((va >> 12) & 0x1ff) * 8, leaf(pa_a))
            .unwrap();
        mem.write_u64(l3 + (((va + 0x1000) >> 12) & 0x1ff) * 8, leaf(pa_b))
            .unwrap();

        let mut cpu = Cpu::new(mem);
        cpu.state.pstate = 0b0101; // EL1h
        cpu.state.mmu = MmuRegs {
            sctlr_el1: 1, // M: MMU on
            ttbr0_el1: l0,
            tcr_el1: 16, // T0SZ = 16, TG0 = 4 KiB
            ..Default::default()
        };
        (cpu, va, pa_a, pa_b)
    }

    #[test]
    fn misaligned_load_crossing_a_page_reads_both_frames() {
        let (mut cpu, va, pa_a, pa_b) = two_page_cpu();
        // Eight bytes at va+0xffc: four in page A, four in page B.
        cpu.mem
            .write(pa_a + 0xffc, &[0x11, 0x22, 0x33, 0x44])
            .unwrap();
        cpu.mem.write(pa_b, &[0x55, 0x66, 0x77, 0x88]).unwrap();
        cpu.state.x[1] = va + 0xffc;
        // LDR x5, [x1]
        cpu.memory_access(0xf940_0025, 0).unwrap();
        assert_eq!(cpu.state.x[5], 0x8877_6655_4433_2211);
    }

    #[test]
    fn detects_unprivileged_accesses() {
        use crate::decode::{self, Op};
        // STTR x0, [x1] and LDTR x2, [x1] are unprivileged; STUR/LDR/register-
        // offset forms are not.
        let unpriv = |w| matches!(decode::decode(w, 0), Op::Memory(m) if m.unprivileged);
        assert!(unpriv(0xf8000820)); // sttr x0, [x1]
        assert!(unpriv(0xf8400822)); // ldtr x2, [x1]
        assert!(!unpriv(0xf8000020)); // stur x0, [x1]
        assert!(!unpriv(0xf9000020)); // str x0, [x1] (unsigned offset)
        assert!(!unpriv(0xf8206820)); // str x0, [x1, x0] (register offset)
    }

    #[test]
    fn unprivileged_store_is_checked_at_el0() {
        // two_page_cpu maps its pages AP=00: EL1 read/write, EL0 no access.
        let (mut cpu, va, _pa_a, _pa_b) = two_page_cpu();
        cpu.state.x[1] = va;
        cpu.state.x[0] = 0x1234;
        // A normal store runs at EL1 and succeeds.
        assert!(cpu.memory_access(0xf9000020, 0).is_ok(), "str at EL1");
        // STTR is unprivileged -- the same page is forbidden at EL0.
        let err = cpu.memory_access(0xf8000820, 0).unwrap_err();
        assert_eq!(err.fsc & 0x3c, 0x0c, "STTR takes a permission fault at EL0");
    }

    #[test]
    fn misaligned_store_crossing_a_page_writes_both_frames() {
        let (mut cpu, va, pa_a, pa_b) = two_page_cpu();
        cpu.state.x[1] = va + 0xffc;
        cpu.state.x[7] = 0xaabb_ccdd_eeff_0011;
        // STR x7, [x1]
        cpu.memory_access(0xf900_0027, 0).unwrap();
        let (mut a, mut b) = ([0u8; 4], [0u8; 4]);
        cpu.mem.read(pa_a + 0xffc, &mut a).unwrap();
        cpu.mem.read(pa_b, &mut b).unwrap();
        assert_eq!(a, [0x11, 0x00, 0xff, 0xee]);
        assert_eq!(b, [0xdd, 0xcc, 0xbb, 0xaa]);
    }

    #[test]
    fn tlb_fill_and_flush_by_address() {
        let (mut cpu, va, pa_a, _) = two_page_cpu();
        cpu.mem.write_u64(pa_a + 8, 0x55).unwrap();
        cpu.state.x[1] = va;
        // LDR x5, [x1, #8] fills the EL1 load TLB for the page.
        cpu.memory_access(0xf940_0425, 0).unwrap();
        let i = TlbEntry::index(0, va);
        assert_eq!(cpu.state.tlb[0].entries[i].tag, va >> 12);
        let host = cpu.state.tlb[0].entries[i].delta.wrapping_add(va + 8);
        // SAFETY: the host address of guest RAM the test owns.
        assert_eq!(unsafe { *(host as *const u64) }, 0x55);
        cpu.flush_va(va + 0x123, None);
        assert_eq!(cpu.state.tlb[0].entries[i].tag, TLB_INVALID);
    }
}
