// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The differential test the plan asks for: random EL0 instruction streams run
//! on the CPU itself and through the translator, with the full register and
//! memory state compared. Guest and host are the same architecture, so the
//! stream runs natively in a child process; a trailing BRK is caught and the
//! child's registers are read from the signal's machine context. Any
//! divergence means the translator got an instruction wrong.

#![cfg(target_arch = "aarch64")]

use std::sync::Arc;

use crate_core::memory::{GuestMemory, RamBlock};
use crate_host::jit::JitRegion;
use crate_jit::asm;
use crate_jit::backend::{CodeCache, Cpu, Stop};

// ---------------------------------------------------------------- shared RAM

/// A page-aligned, readable/writable/executable region shared with the child,
/// so the child's native run and the parent's reads see the same memory. It is
/// the guest's RAM: code at the start, a scratch area for loads and stores
/// higher up.
struct Shared {
    ptr: *mut u8,
    len: usize,
}

impl Shared {
    fn new(len: usize) -> Self {
        // SAFETY: a fresh shared anonymous mapping aliases nothing.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert!(ptr != libc::MAP_FAILED, "mapping shared RWX memory");
        Shared {
            ptr: ptr.cast(),
            len,
        }
    }

    fn addr(&self) -> u64 {
        self.ptr as u64
    }

    fn write(&self, off: usize, bytes: &[u8]) {
        // SAFETY: `off + bytes.len()` stays inside the mapping (callers keep
        // within RAM_SIZE).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(off), bytes.len()) }
    }

    fn read(&self, off: usize, len: usize) -> Vec<u8> {
        let mut v = vec![0u8; len];
        // SAFETY: as above.
        unsafe { std::ptr::copy_nonoverlapping(self.ptr.add(off), v.as_mut_ptr(), len) };
        v
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: mapped in `new` with this length.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}

const RAM_SIZE: usize = 1 << 20;
const CODE_OFF: usize = 0;
// The guest's stack lives in its own region, which the native run's signal
// delivery may scribble and which is therefore not compared. Guest loads and
// stores target the data region, which is.
const STACK_OFF: usize = 0x8000;
const STACK_LEN: usize = 0x4000;
const DATA_OFF: usize = 0xc000;
const DATA_LEN: usize = 0x4000;

// ---------------------------------------------------------------- captured state

/// The guest registers both runs end with, in the shared mapping so the child
/// writes them where the parent reads them.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Captured {
    x: [u64; 31],
    sp: u64,
    pc: u64,
    nzcv: u64,
    done: u64,
}

/// macOS arm64 machine context, from <mach/arm/_structs.h>, enough to read the
/// registers at the BRK.
#[repr(C)]
struct ExcState64 {
    far: u64,
    esr: u32,
    exc: u32,
}
#[repr(C)]
struct ThreadState64 {
    x: [u64; 29],
    fp: u64,
    lr: u64,
    sp: u64,
    pc: u64,
    cpsr: u32,
    pad: u32,
}
#[repr(C)]
struct NeonState64 {
    v: [[u8; 16]; 32],
    fpsr: u32,
    fpcr: u32,
}
#[repr(C)]
struct Mcontext64 {
    es: ExcState64,
    ss: ThreadState64,
    ns: NeonState64,
}

/// Where the child's BRK handler writes the captured registers. Set before the
/// fork, inherited by the child.
static mut RESULT: *mut Captured = std::ptr::null_mut();

extern "C" fn on_brk(_sig: i32, _info: *mut libc::siginfo_t, ucontext: *mut libc::c_void) {
    // SAFETY: the kernel hands a valid ucontext_t; its mcontext matches the
    // layout above. RESULT points at the shared mapping.
    unsafe {
        let uc = ucontext as *mut libc::ucontext_t;
        let mc = (*uc).uc_mcontext as *mut Mcontext64;
        let ss = &(*mc).ss;
        let out = &mut *RESULT;
        out.x[..29].copy_from_slice(&ss.x);
        out.x[29] = ss.fp;
        out.x[30] = ss.lr;
        out.sp = ss.sp;
        out.pc = ss.pc;
        out.nzcv = (ss.cpsr as u64) & 0xf000_0000;
        out.done = 1;
        libc::_exit(0);
    }
}

/// Runs the guest stream natively in a child process and returns its registers.
/// `entry` is an absolute executable host address (the trampoline's start).
fn run_native(entry: u64, init: &Captured) -> Captured {
    // A shared page the child writes its captured state to, and an alternate
    // stack the child's handler runs on. Allocated once; inherited by the fork.
    prepare();
    // SAFETY: statics only written before the fork and read after.
    unsafe {
        (*RESULT_BOX).done = 0;
        RESULT = RESULT_BOX;
    }

    // SAFETY: fork; the child installs its own handlers (no allocation) and
    // runs the trampoline. The parent installs nothing, so a translator bug
    // that faults surfaces as a real crash rather than being swallowed.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0, "fork");
    if pid == 0 {
        // SAFETY: async-signal-safe set-up (pre-allocated stack, static
        // handler), then the trampoline, which ends in a BRK.
        unsafe {
            let ss = libc::stack_t {
                ss_sp: ALT_STACK,
                ss_flags: 0,
                ss_size: ALT_STACK_LEN,
            };
            libc::sigaltstack(&ss, std::ptr::null_mut());
            let mut act: libc::sigaction = std::mem::zeroed();
            act.sa_sigaction = on_brk as *const () as usize;
            act.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
            for sig in [
                libc::SIGTRAP,
                libc::SIGILL,
                libc::SIGBUS,
                libc::SIGSEGV,
                libc::SIGALRM,
            ] {
                libc::sigaction(sig, &act, std::ptr::null_mut());
            }
            libc::alarm(5); // a watchdog, should the stream never reach the BRK
            let tramp: extern "C" fn(*const Captured) = std::mem::transmute(entry);
            tramp(init);
            libc::_exit(1);
        }
    }
    let mut status = 0;
    // SAFETY: valid pid and pointer.
    unsafe { libc::waitpid(pid, &mut status, 0) };
    // SAFETY: the child wrote the shared result before _exit.
    let captured = unsafe { *(RESULT_BOX as *const Captured) };
    assert_eq!(
        captured.done, 1,
        "the child did not reach the BRK (status {status:#x})"
    );
    captured
}

const ALT_STACK_LEN: usize = 256 * 1024;
static mut RESULT_BOX: *mut Captured = std::ptr::null_mut();
static mut ALT_STACK: *mut libc::c_void = std::ptr::null_mut();
static PREPARE: std::sync::Once = std::sync::Once::new();

fn prepare() {
    PREPARE.call_once(|| {
        // SAFETY: one-time allocation of the shared result page and the child's
        // alternate signal stack, both inherited across the fork.
        unsafe {
            let page = libc::mmap(
                std::ptr::null_mut(),
                4096,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_ANON,
                -1,
                0,
            );
            assert!(page != libc::MAP_FAILED, "result page");
            RESULT_BOX = page as *mut Captured;
            ALT_STACK = libc::mmap(
                std::ptr::null_mut(),
                ALT_STACK_LEN,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
        }
    });
}

// ---------------------------------------------------------------- a tiny RNG

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*.
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn reg(&mut self) -> u32 {
        // x0..x28, but never x18: Apple reserves the platform register and the
        // OS may clobber it in the native run, so it is neither used nor
        // compared.
        let r = (self.next() % 28) as u32;
        if r >= 18 { r + 1 } else { r }
    }
    fn bits(&mut self, n: u32) -> u32 {
        (self.next() & ((1 << n) - 1)) as u32
    }
    fn bool(&mut self) -> bool {
        self.next() & 1 != 0
    }
}

// ---------------------------------------------------------------- the stream

/// A generated program: the trampoline, the random body and the BRK, and the
/// guest PC the translated run starts at.
struct Program {
    words: Vec<u32>,
    guest_pc: u64,
    body_start: usize,
}

/// Builds the register-loading trampoline, then `count` random instructions,
/// then a BRK. `scratch` is the guest address of the scratch area, handed to
/// memory instructions.
fn generate(rng: &mut Rng, count: usize, ram_base: u64) -> Program {
    let mut tramp = asm::Asm::new();
    // x0 = &Captured at entry. Load NZCV and SP first (using x1 as scratch),
    // then x2..x30, then x1, then x0.
    use asm::*;
    // Offsets into `Captured`: x[31] at 0, sp at 248, pc at 256, nzcv at 264.
    let nzcv_off = (33 * 8) as u32;
    let sp_off = (31 * 8) as u32;
    tramp.emit(ldr_imm(3, 1, 0, nzcv_off));
    tramp.emit(msr(NZCV, 1));
    tramp.emit(ldr_imm(3, 1, 0, sp_off));
    tramp.emit(mov_sp(SP, 1));
    for r in 2..=30u32 {
        tramp.emit(ldr_imm(3, r, 0, r * 8));
    }
    tramp.emit(ldr_imm(3, 1, 0, 8));
    tramp.emit(ldr_imm(3, 0, 0, 0));
    let mut words = tramp.words.clone();
    let body_start = words.len();

    let scratch = ram_base + DATA_OFF as u64;
    for _ in 0..count {
        emit_random(rng, &mut words, scratch);
    }
    words.push(0xd4200000); // brk #0
    Program {
        guest_pc: ram_base + CODE_OFF as u64 + (body_start * 4) as u64,
        words,
        body_start,
    }
}

/// Appends one random, well-defined EL0 instruction.
fn emit_random(rng: &mut Rng, out: &mut Vec<u32>, scratch: u64) {
    use asm::*;
    let rd = rng.reg();
    let rn = rng.reg();
    let rm = rng.reg();
    let sf = rng.bool();
    // ADD/SUB shifted register allow LSL/LSR/ASR only (shift 0b11 is reserved);
    // the logical ops also allow ROR.
    let arith_shift = rng.next() as u32 % 3;
    let amount = rng.bits(5) & 31;
    // A bitfield position/width valid for both register widths (0..31).
    let immr = rng.bits(5) & 31;
    let imms = rng.bits(5) & 31;
    let cond = rng.bits(4);
    match rng.next() % 40 {
        0 => out.push(add_reg(sf, rd, rn, rm, arith_shift, amount)),
        1 => out.push(adds_reg(sf, rd, rn, rm, arith_shift, amount)),
        2 => out.push(sub_reg(sf, rd, rn, rm, arith_shift, amount)),
        3 => out.push(subs_reg(sf, rd, rn, rm, arith_shift, amount)),
        4 => out.push(and_reg(sf, rd, rn, rm, rng.bits(2), rng.bits(5) & 31)),
        5 => out.push(orr_reg(sf, rd, rn, rm, rng.bits(2), rng.bits(5) & 31)),
        6 => out.push(eor_reg(sf, rd, rn, rm, rng.bits(2), rng.bits(5) & 31)),
        7 => out.push(bic_reg(sf, rd, rn, rm, rng.bits(2), rng.bits(5) & 31)),
        8 => out.push(add_imm(sf, rd, rn, rng.bits(12), rng.bool())),
        9 => out.push(subs_imm(sf, rd, rn, rng.bits(12), false)),
        10 => out.push(movz(sf, rd, rng.bits(16), rng.bits(if sf { 2 } else { 1 }))),
        11 => out.push(movk(sf, rd, rng.bits(16), rng.bits(if sf { 2 } else { 1 }))),
        12 => out.push(movn(sf, rd, rng.bits(16), rng.bits(if sf { 2 } else { 1 }))),
        13 => {
            // Logical immediate (AND/ORR): pick a value that is one.
            let v = pick_logical(rng, sf);
            if rng.bool() {
                out.push(and_imm(sf, rd, rn, v));
            } else {
                out.push(orr_imm(sf, rd, rn, v));
            }
        }
        14 => out.push(lslv(sf, rd, rn, rm)),
        15 => out.push(lsrv(sf, rd, rn, rm)),
        16 => out.push(asrv(sf, rd, rn, rm)),
        17 => out.push(rorv(sf, rd, rn, rm)),
        18 => out.push(madd(sf, rd, rn, rm, rng.reg())),
        19 => out.push(csel(sf, rd, rn, rm, rng.bits(4))),
        20 => out.push(csinc(sf, rd, rn, rm, rng.bits(4))),
        21 => {
            // A store then a dependent load to the same scratch slot, so both
            // runs touch defined memory. Offset within the scratch area.
            let base = rng.reg().max(1); // avoid x0 so the trampoline survives
            let slot = (rng.next() as usize % (DATA_LEN / 8 - 2)) * 8;
            // movz/movk the scratch base into `base`, then str/ldr.
            for w in mov_imm(base, scratch + slot as u64) {
                out.push(w);
            }
            let rt = rng.reg();
            out.push(str_imm(3, rt, base, 0));
            out.push(ldr_imm(3, rng.reg(), base, 0));
        }
        22 => {
            // A byte/half store and load, exercising sizes and zero-extension.
            let base = rng.reg().max(1);
            let slot = (rng.next() as usize % (DATA_LEN - 8)) & !7;
            for w in mov_imm(base, scratch + slot as u64) {
                out.push(w);
            }
            let size = rng.bits(2);
            out.push(str_imm(size, rng.reg(), base, 0));
            out.push(ldr_imm(size, rng.reg(), base, 0));
        }
        23 => {
            // A stack-pointer adjustment and read-back.
            out.push(sub_imm(
                true,
                asm::SP,
                asm::SP,
                (rng.bits(9) & 0x1f0) + 16,
                false,
            ));
            out.push(add_imm(true, rd.max(1), asm::SP, 0, false));
        }
        24 => out.push(ubfm(sf, rd, rn, immr & bw(sf), imms & bw(sf))),
        25 => out.push(sbfm(sf, rd, rn, immr & bw(sf), imms & bw(sf))),
        26 => out.push(bfm(sf, rd, rn, immr & bw(sf), imms & bw(sf))),
        27 => out.push(extr(sf, rd, rn, rm, imms & bw(sf))),
        28 => out.push(ccmp_imm(sf, rn, rng.bits(5), cond, rng.bits(4))),
        29 => out.push(ccmn_reg(sf, rn, rm, cond, rng.bits(4))),
        30 => out.push(csinv(sf, rd, rn, rm, cond)),
        31 => out.push(csneg(sf, rd, rn, rm, cond)),
        32 => out.push(data1(sf, rd, rn, rng.bits(2))), // RBIT/REV16/CLZ/CLS
        33 => out.push(udiv(sf, rd, rn, rm)),
        34 => out.push(sdiv(sf, rd, rn, rm)),
        35 => out.push(mulh(rd, rn, rm, rng.bool())), // UMULH/SMULH
        36 => {
            // ADC/SBC use the carry flag, so a flag-setter first.
            out.push(subs_reg(sf, 31, rn, rm, 0, 0));
            out.push(adc(sf, rd, rn, rm, rng.bool()));
        }
        37 => {
            // A pre/post-index load and store, exercising writeback. The
            // transfer register must differ from the base, or the writeback
            // makes the result UNPREDICTABLE.
            let base = rng.reg().max(1);
            let slot = (rng.next() as usize % (DATA_LEN / 2)) & !7;
            for w in mov_imm(base, scratch + slot as u64) {
                out.push(w);
            }
            out.push(str_imm_wb(3, reg_not(rng, &[base]), base, 8, rng.bool()));
            out.push(ldr_imm_wb(3, reg_not(rng, &[base]), base, -8, rng.bool()));
        }
        38 => {
            // A register pair store and load; the two load targets must differ.
            let base = rng.reg().max(1);
            let slot = (rng.next() as usize % (DATA_LEN / 2)) & !7;
            for w in mov_imm(base, scratch + slot as u64) {
                out.push(w);
            }
            out.push(stp(rng.reg(), rng.reg(), base, 0));
            let t1 = reg_not(rng, &[base]);
            let t2 = reg_not(rng, &[base, t1]);
            out.push(ldp(t1, t2, base, 16));
        }
        _ => {
            // A signed byte/half load and a register-offset access at a zero
            // index, covering sign-extension and the indexed addressing mode.
            let base = rng.reg().max(1);
            let idx = reg_not(rng, &[base, 0]);
            let slot = (rng.next() as usize % (DATA_LEN - 16)) & !7;
            for w in mov_imm(base, scratch + slot as u64) {
                out.push(w);
            }
            out.push(movz(true, idx, 0, 0)); // index 0
            let size = (rng.bits(2)).min(2); // signed loads are byte/half/word
            out.push(str_reg(size, rng.reg(), base, idx));
            out.push(ldrs64(size, rng.reg(), base)); // signed load to 64 bits
        }
    }
}

/// The widest bit index for a bitfield at this register width.
fn bw(sf: bool) -> u32 {
    if sf { 63 } else { 31 }
}

/// A random register not in `avoid` (for writeback bases and paired targets).
fn reg_not(rng: &mut Rng, avoid: &[u32]) -> u32 {
    loop {
        let r = rng.reg();
        if !avoid.contains(&r) {
            return r;
        }
    }
}

/// A value that is a valid logical immediate, built directly: a run of ones in
/// an element of size 2, 4, ... bits, rotated, then replicated across 64 bits.
fn pick_logical(rng: &mut Rng, sf: bool) -> u64 {
    let sizes: &[u32] = if sf {
        &[2, 4, 8, 16, 32, 64]
    } else {
        &[2, 4, 8, 16, 32]
    };
    loop {
        let size = sizes[rng.next() as usize % sizes.len()];
        let ones = 1 + rng.next() as u32 % (size - 1);
        let rot = rng.next() as u32 % size;
        let mask = if size == 64 {
            u64::MAX
        } else {
            (1u64 << size) - 1
        };
        let run = (1u64 << ones) - 1;
        let elem = if rot == 0 {
            run
        } else {
            ((run >> rot) | (run << (size - rot))) & mask
        };
        // Replicate the element to fill 64 bits.
        let mut v = 0u64;
        let mut bit = 0;
        while bit < 64 {
            v |= elem << bit;
            bit += size;
        }
        if asm::logical_imm(v, sf).is_some() {
            return v;
        }
    }
}

// Extra encoders the generator needs that `asm` does not expose.
#[rustfmt::skip]
mod extra {
    use super::asm::Reg;
    pub fn adds_reg(sf: bool, rd: Reg, rn: Reg, rm: Reg, shift: u32, amount: u32) -> u32 {
        (sf as u32) << 31 | 0x2b00_0000 | shift << 22 | rm << 16 | amount << 10 | rn << 5 | rd
    }
    pub fn sub_reg(sf: bool, rd: Reg, rn: Reg, rm: Reg, shift: u32, amount: u32) -> u32 {
        (sf as u32) << 31 | 0x4b00_0000 | shift << 22 | rm << 16 | amount << 10 | rn << 5 | rd
    }
    pub fn subs_reg(sf: bool, rd: Reg, rn: Reg, rm: Reg, shift: u32, amount: u32) -> u32 {
        (sf as u32) << 31 | 0x6b00_0000 | shift << 22 | rm << 16 | amount << 10 | rn << 5 | rd
    }
    pub fn subs_imm(sf: bool, rd: Reg, rn: Reg, imm12: u32, sh: bool) -> u32 {
        (sf as u32) << 31 | 0x7100_0000 | (sh as u32) << 22 | imm12 << 10 | rn << 5 | rd
    }
    pub fn lslv(sf: bool, rd: Reg, rn: Reg, rm: Reg) -> u32 {
        (sf as u32) << 31 | 0x1ac0_2000 | rm << 16 | rn << 5 | rd
    }
    pub fn lsrv(sf: bool, rd: Reg, rn: Reg, rm: Reg) -> u32 {
        (sf as u32) << 31 | 0x1ac0_2400 | rm << 16 | rn << 5 | rd
    }
    pub fn asrv(sf: bool, rd: Reg, rn: Reg, rm: Reg) -> u32 {
        (sf as u32) << 31 | 0x1ac0_2800 | rm << 16 | rn << 5 | rd
    }
    pub fn rorv(sf: bool, rd: Reg, rn: Reg, rm: Reg) -> u32 {
        (sf as u32) << 31 | 0x1ac0_2c00 | rm << 16 | rn << 5 | rd
    }
    pub fn madd(sf: bool, rd: Reg, rn: Reg, rm: Reg, ra: Reg) -> u32 {
        (sf as u32) << 31 | 0x1b00_0000 | rm << 16 | ra << 10 | rn << 5 | rd
    }
    pub fn csel(sf: bool, rd: Reg, rn: Reg, rm: Reg, cond: u32) -> u32 {
        (sf as u32) << 31 | 0x1a80_0000 | rm << 16 | cond << 12 | rn << 5 | rd
    }
    pub fn csinc(sf: bool, rd: Reg, rn: Reg, rm: Reg, cond: u32) -> u32 {
        (sf as u32) << 31 | 0x1a80_0400 | rm << 16 | cond << 12 | rn << 5 | rd
    }
    pub fn csinv(sf: bool, rd: Reg, rn: Reg, rm: Reg, cond: u32) -> u32 {
        (sf as u32) << 31 | 0x5a80_0000 | rm << 16 | cond << 12 | rn << 5 | rd
    }
    pub fn csneg(sf: bool, rd: Reg, rn: Reg, rm: Reg, cond: u32) -> u32 {
        (sf as u32) << 31 | 0x5a80_0400 | rm << 16 | cond << 12 | rn << 5 | rd
    }
    pub fn sbfm(sf: bool, rd: Reg, rn: Reg, immr: u32, imms: u32) -> u32 {
        (sf as u32) << 31 | 0x1300_0000 | (sf as u32) << 22 | immr << 16 | imms << 10 | rn << 5 | rd
    }
    pub fn extr(sf: bool, rd: Reg, rn: Reg, rm: Reg, lsb: u32) -> u32 {
        (sf as u32) << 31 | 0x1380_0000 | (sf as u32) << 22 | rm << 16 | lsb << 10 | rn << 5 | rd
    }
    pub fn ccmp_imm(sf: bool, rn: Reg, imm5: u32, cond: u32, nzcv: u32) -> u32 {
        (sf as u32) << 31 | 0x7a40_0800 | imm5 << 16 | cond << 12 | rn << 5 | nzcv
    }
    pub fn ccmn_reg(sf: bool, rn: Reg, rm: Reg, cond: u32, nzcv: u32) -> u32 {
        (sf as u32) << 31 | 0x3a40_0000 | rm << 16 | cond << 12 | rn << 5 | nzcv
    }
    /// A one-source op: 0 RBIT, 1 REV16, 2 CLZ, 3 CLS (all valid at both widths).
    pub fn data1(sf: bool, rd: Reg, rn: Reg, which: u32) -> u32 {
        let opc = match which {
            0 => 0b000000, // RBIT
            1 => 0b000001, // REV16
            2 => 0b000100, // CLZ
            _ => 0b000101, // CLS
        };
        (sf as u32) << 31 | 0x5ac0_0000 | opc << 10 | rn << 5 | rd
    }
    pub fn udiv(sf: bool, rd: Reg, rn: Reg, rm: Reg) -> u32 {
        (sf as u32) << 31 | 0x1ac0_0800 | rm << 16 | rn << 5 | rd
    }
    pub fn sdiv(sf: bool, rd: Reg, rn: Reg, rm: Reg) -> u32 {
        (sf as u32) << 31 | 0x1ac0_0c00 | rm << 16 | rn << 5 | rd
    }
    /// UMULH / SMULH (64-bit only).
    pub fn mulh(rd: Reg, rn: Reg, rm: Reg, signed: bool) -> u32 {
        let base = if signed { 0x9b40_7c00 } else { 0x9bc0_7c00 };
        base | rm << 16 | rn << 5 | rd
    }
    /// ADC / SBC.
    pub fn adc(sf: bool, rd: Reg, rn: Reg, rm: Reg, subtract: bool) -> u32 {
        let base: u32 = if subtract { 0x5a00_0000 } else { 0x1a00_0000 };
        (sf as u32) << 31 | base | rm << 16 | rn << 5 | rd
    }
    /// STR (pre- or post-index).
    pub fn str_imm_wb(size: u32, rt: Reg, rn: Reg, imm: i32, pre: bool) -> u32 {
        let mode = if pre { 0xc00 } else { 0x400 };
        size << 30 | 0x3800_0000 | mode | ((imm as u32 & 0x1ff) << 12) | rn << 5 | rt
    }
    /// LDR (pre- or post-index).
    pub fn ldr_imm_wb(size: u32, rt: Reg, rn: Reg, imm: i32, pre: bool) -> u32 {
        let mode = if pre { 0xc00 } else { 0x400 };
        size << 30 | 0x3840_0000 | mode | ((imm as u32 & 0x1ff) << 12) | rn << 5 | rt
    }
    /// A signed load to a 64-bit register (unsigned offset 0), size 0..2.
    pub fn ldrs64(size: u32, rt: Reg, rn: Reg) -> u32 {
        size << 30 | 0x3980_0000 | rn << 5 | rt
    }
}
use extra::*;

// ---------------------------------------------------------------- the test

fn run_translated(shared: &Shared, prog: &Program, init: &Captured) -> Captured {
    let block: Arc<dyn RamBlock> = Arc::new(ExternalRam {
        ptr: shared.ptr,
        len: shared.len,
    });
    let mem = Arc::new(GuestMemory::new(vec![(shared.addr(), block)]).unwrap());
    let mut cpu = Cpu::new(mem);
    cpu.state.x[..31].copy_from_slice(&init.x);
    cpu.state.sp = init.sp;
    cpu.state.nzcv = init.nzcv;
    cpu.state.pc = prog.guest_pc;
    let mut cache = CodeCache::new(4 << 20).unwrap();
    let stop = cpu.run(&mut cache);
    if stop != Stop::Brk(0) {
        eprintln!("translated stop {stop:?} at pc {:#x}", cpu.state.pc);
        eprintln!("  esr/far syndrome {:#x}", cpu.state.exception_syndrome);
        let faulting = cpu.mem.read_u32(cpu.state.pc).unwrap_or(0);
        eprintln!("  faulting word {faulting:08x}");
        for i in 0..31 {
            eprintln!("    x{i} = {:#x}", cpu.state.x[i]);
        }
        eprintln!("    sp = {:#x}", cpu.state.sp);
        eprintln!("  guest_pc {:#x}, body:", prog.guest_pc);
        for (i, w) in prog.words[prog.body_start..].iter().enumerate() {
            eprintln!("    +{:x}: {w:08x}", (prog.body_start + i) * 4);
        }
        panic!("translated run ended wrong: {stop:?}");
    }
    let mut out = Captured::default();
    out.x.copy_from_slice(&cpu.state.x[..31]);
    out.sp = cpu.state.sp;
    out.pc = cpu.state.pc;
    out.nzcv = cpu.state.nzcv & 0xf000_0000;
    out.done = 1;
    out
}

/// A RamBlock over the test's shared mapping.
struct ExternalRam {
    ptr: *mut u8,
    len: usize,
}
// SAFETY: the mapping outlives the GuestMemory; guest memory is shared by
// design and reached only through raw pointers.
unsafe impl Send for ExternalRam {}
unsafe impl Sync for ExternalRam {}
// SAFETY: `ptr` is valid for `len` bytes for the block's whole life and page
// aligned (mmap result).
unsafe impl RamBlock for ExternalRam {
    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

#[test]
fn random_streams_match_the_cpu() {
    let shared = Shared::new(RAM_SIZE);
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let iterations: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);

    for iter in 0..iterations {
        // A fresh random program and initial register file.
        let prog = generate(&mut rng, 24, shared.addr());
        let mut init = Captured::default();
        for x in init.x.iter_mut() {
            *x = rng.next();
        }
        init.x[18] = 0;
        // SP: aligned, inside the scratch area so stack adjustments stay sane.
        init.sp = shared.addr() + (STACK_OFF + STACK_LEN * 3 / 4) as u64;
        init.nzcv = (rng.next() & 0xf000_0000) as u64;

        // Lay the code into guest RAM (so the translated run reads it) and into
        // an executable region (so the child runs it natively).
        let code: Vec<u8> = prog.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        shared.write(CODE_OFF, &code);
        let exec = JitRegion::new(code.len() + 4096).unwrap();
        exec.writer().write_words(0, &prog.words);
        exec.flush(0, code.len());

        // Native, then reset the data region and run translated.
        shared.write(DATA_OFF, &vec![0u8; DATA_LEN]);
        let mut native = run_native(exec.exec_base() as u64, &init);
        let native_scratch = shared.read(DATA_OFF, DATA_LEN);

        shared.write(DATA_OFF, &vec![0u8; DATA_LEN]);
        let mut translated = run_translated(&shared, &prog, &init);
        let translated_scratch = shared.read(DATA_OFF, DATA_LEN);

        // The native copy runs at a different address, so ignore PC; and x18
        // (the platform register) is not preserved by the OS natively.
        native.pc = 0;
        translated.pc = 0;
        native.x[18] = 0;
        translated.x[18] = 0;

        if native != translated || native_scratch != translated_scratch {
            report_divergence(iter, &prog, &init, &native, &translated);
            for (off, (a, b)) in native_scratch.iter().zip(&translated_scratch).enumerate() {
                if a != b {
                    eprintln!(
                        "  scratch[{:#x}]: native {:02x} jit {:02x}",
                        DATA_OFF + off,
                        a,
                        b
                    );
                }
            }
            panic!("iteration {iter}: translated run diverged from the CPU");
        }
    }
}

fn report_divergence(
    iter: usize,
    prog: &Program,
    init: &Captured,
    native: &Captured,
    translated: &Captured,
) {
    eprintln!("--- divergence at iteration {iter} ---");
    eprintln!(
        "body ({} instructions):",
        prog.words.len() - prog.body_start - 1
    );
    for w in &prog.words[prog.body_start..] {
        eprintln!("  {w:08x}");
    }
    eprintln!("init sp={:#x} nzcv={:#x}", init.sp, init.nzcv);
    for i in 0..31 {
        if i == 18 {
            continue;
        } // the platform register, not preserved natively
        if native.x[i] != translated.x[i] {
            eprintln!(
                "  x{i}: native {:#018x}  jit {:#018x}",
                native.x[i], translated.x[i]
            );
        }
    }
    if native.sp != translated.sp {
        eprintln!("  sp: native {:#x} jit {:#x}", native.sp, translated.sp);
    }
    if native.nzcv != translated.nzcv {
        eprintln!(
            "  nzcv: native {:#x} jit {:#x}",
            native.nzcv, translated.nzcv
        );
    }
    if native.pc != translated.pc {
        eprintln!("  pc: native {:#x} jit {:#x}", native.pc, translated.pc);
    }
}
