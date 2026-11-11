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
    sig: u64,
    v: [[u8; 16]; 32],
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

extern "C" fn on_brk(sig: i32, _info: *mut libc::siginfo_t, ucontext: *mut libc::c_void) {
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
        out.v = (*mc).ns.v;
        out.done = 1;
        out.sig = sig as u64;
        libc::_exit(0);
    }
}

/// Runs the guest stream natively in a child process and returns its registers.
/// `entry` is an absolute executable host address (the trampoline's start).
fn run_native(entry: u64, init: &Captured) -> Captured {
    // The result page and the alternate stack are shared by every test in
    // the process; one native run at a time.
    static NATIVE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _one = NATIVE.lock().unwrap_or_else(|e| e.into_inner());
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

/// The register-loading trampoline: x0 = &Captured at entry. Loads the V
/// registers, NZCV and SP first (x1 as scratch), then x2..x30, then x1, x0.
fn trampoline() -> Vec<u32> {
    use asm::*;
    let mut tramp = asm::Asm::new();
    // Offsets into `Captured`: x[31] at 0, sp 248, pc 256, nzcv 264, done 272,
    // sig 280, v at 288.
    let nzcv_off = (33 * 8) as u32;
    let sp_off = (31 * 8) as u32;
    let v_off = 288u32;
    for i in 0..32u32 {
        tramp.emit(ldr_q(i, 0, v_off + i * 16));
    }
    tramp.emit(ldr_imm(3, 1, 0, nzcv_off));
    tramp.emit(msr(NZCV, 1));
    tramp.emit(ldr_imm(3, 1, 0, sp_off));
    tramp.emit(mov_sp(SP, 1));
    for r in 2..=30u32 {
        tramp.emit(ldr_imm(3, r, 0, r * 8));
    }
    tramp.emit(ldr_imm(3, 1, 0, 8));
    tramp.emit(ldr_imm(3, 0, 0, 0));
    tramp.words.clone()
}

/// What a generator run builds the program for.
#[derive(Clone, Copy)]
struct Shape {
    /// Units of random instructions in the body.
    count: usize,
    /// Byte offset of the code in RAM (so the body can cross a page).
    code_off: usize,
    /// The guest register logical X18 is: 18 for the translated run, 29 for
    /// the native one, where the OS may clobber x18.
    x18: u32,
    /// Whether streams may name logical X18 at all.
    use_x18: bool,
}

/// Builds the trampoline, then the random body, then a BRK.
fn generate(rng: &mut Rng, shape: Shape, ram_base: u64) -> Program {
    let mut words = trampoline();
    let body_start = words.len();
    let scratch = ram_base + DATA_OFF as u64;
    let mut g = Gen {
        rng,
        scratch,
        x18: shape.x18,
        use_x18: shape.use_x18,
    };
    for _ in 0..shape.count {
        g.structured(&mut words);
    }
    words.push(0xd4200000); // brk #0
    Program {
        guest_pc: ram_base + shape.code_off as u64 + (body_start * 4) as u64,
        words,
        body_start,
    }
}

struct Gen<'a> {
    rng: &'a mut Rng,
    scratch: u64,
    x18: u32,
    use_x18: bool,
}

impl Gen<'_> {
    /// A random general register, as the encoding names it (logical X18 is
    /// mapped to the run's register for it).
    fn reg(&mut self) -> u32 {
        if self.use_x18 && self.rng.next() % 6 == 0 {
            return self.x18;
        }
        self.rng.reg()
    }

    /// A random register other than those in `avoid`.
    fn reg_not(&mut self, avoid: &[u32]) -> u32 {
        loop {
            let r = self.reg();
            if !avoid.contains(&r) {
                return r;
            }
        }
    }

    /// A base register for a memory access: never x0 (so the trampoline's
    /// pointer survives) nor logical X18 when that is x29... any but x0.
    fn base(&mut self) -> u32 {
        loop {
            let r = self.reg();
            if r != 0 {
                return r;
            }
        }
    }

    /// Control flow around straight-line units: forward branches, counted
    /// loops, calls and indirect jumps. x30 is the loop counter and the link,
    /// which straight units never touch.
    fn structured(&mut self, out: &mut Vec<u32>) {
        use asm::*;
        match self.rng.next() % 20 {
            0 | 1 => {
                // A forward conditional branch over 1..3 units.
                let mut body = Vec::new();
                for _ in 0..1 + self.rng.next() % 3 {
                    self.unit(&mut body);
                }
                let off = (body.len() as i64 + 1) * 4;
                let r = self.reg();
                out.push(match self.rng.next() % 4 {
                    0 => b_cond(self.rng.bits(4) & 0xe, off),
                    1 => cbz(self.rng.bool(), r, off),
                    2 => cbnz(self.rng.bool(), r, off),
                    _ => tbz(r, self.rng.bits(6), off),
                });
                out.extend(body);
            }
            2 => {
                // A counted loop of 1..5 trips over 1..2 units.
                out.push(movz(true, 30, 1 + self.rng.bits(2), 0));
                let top = out.len();
                for _ in 0..1 + self.rng.next() % 2 {
                    self.unit(out);
                }
                out.push(subs_imm(true, 30, 30, 1, false));
                out.push(b_cond(NE, (top as i64 - out.len() as i64) * 4));
            }
            3 => {
                // A call to a subroutine placed right after, then a branch
                // over it: bl sub; b over; sub: units; ret; over:
                let mut sub = Vec::new();
                for _ in 0..1 + self.rng.next() % 2 {
                    self.unit(&mut sub);
                }
                sub.push(ret());
                out.push(bl(8));
                out.push(b((sub.len() as i64 + 1) * 4));
                out.extend(sub);
            }
            4 => {
                // An indirect jump over 0..2 units, then x30 cleared so the
                // result does not depend on where the code runs.
                let mut skipped = Vec::new();
                for _ in 0..self.rng.next() % 3 {
                    self.unit(&mut skipped);
                }
                out.push(adr(30, (skipped.len() as i64 + 2) * 4));
                out.push(br(30));
                out.extend(skipped);
                out.push(movz(true, 30, 0, 0));
            }
            _ => self.unit(out),
        }
    }

    /// Appends one random, well-defined, straight-line unit.
    fn unit(&mut self, out: &mut Vec<u32>) {
        use asm::*;
        let rd = self.reg();
        let rn = self.reg();
        let rm = self.reg();
        let rng = &mut *self.rng;
        let sf = rng.bool();
        // ADD/SUB shifted register allow LSL/LSR/ASR only (shift 0b11 is reserved);
        // the logical ops also allow ROR.
        let arith_shift = rng.next() as u32 % 3;
        let amount = rng.bits(5) & 31;
        // A bitfield position/width valid for both register widths (0..31).
        let immr = rng.bits(5) & 31;
        let imms = rng.bits(5) & 31;
        let cond = rng.bits(4);
        let scratch = self.scratch;
        match self.rng.next() % 72 {
            0 => out.push(add_reg(sf, rd, rn, rm, arith_shift, amount)),
            1 => out.push(adds_reg(sf, rd, rn, rm, arith_shift, amount)),
            2 => out.push(sub_reg(sf, rd, rn, rm, arith_shift, amount)),
            3 => out.push(subs_reg(sf, rd, rn, rm, arith_shift, amount)),
            4 => out.push(and_reg(sf, rd, rn, rm, self.rng.bits(2), self.rng.bits(5))),
            5 => out.push(orr_reg(sf, rd, rn, rm, self.rng.bits(2), self.rng.bits(5))),
            6 => out.push(eor_reg(sf, rd, rn, rm, self.rng.bits(2), self.rng.bits(5))),
            7 => out.push(bic_reg(sf, rd, rn, rm, self.rng.bits(2), self.rng.bits(5))),
            8 => out.push(add_imm(sf, rd, rn, self.rng.bits(12), self.rng.bool())),
            9 => out.push(subs_imm(sf, rd, rn, self.rng.bits(12), false)),
            10 => {
                let hw = self.rng.bits(if sf { 2 } else { 1 });
                out.push(movz(sf, rd, self.rng.bits(16), hw))
            }
            11 => {
                let hw = self.rng.bits(if sf { 2 } else { 1 });
                out.push(movk(sf, rd, self.rng.bits(16), hw))
            }
            12 => {
                let hw = self.rng.bits(if sf { 2 } else { 1 });
                out.push(movn(sf, rd, self.rng.bits(16), hw))
            }
            13 => {
                // Logical immediate (AND/ORR): pick a value that is one.
                let v = pick_logical(self.rng, sf);
                if self.rng.bool() {
                    out.push(and_imm(sf, rd, rn, v));
                } else {
                    out.push(orr_imm(sf, rd, rn, v));
                }
            }
            14 => out.push(lslv(sf, rd, rn, rm)),
            15 => out.push(lsrv(sf, rd, rn, rm)),
            16 => out.push(asrv(sf, rd, rn, rm)),
            17 => out.push(rorv(sf, rd, rn, rm)),
            18 => {
                let ra = self.reg();
                out.push(madd(sf, rd, rn, rm, ra))
            }
            19 => out.push(csel(sf, rd, rn, rm, self.rng.bits(4))),
            20 => out.push(csinc(sf, rd, rn, rm, self.rng.bits(4))),
            21 => {
                // A store then a dependent load to the same scratch slot.
                let base = self.base();
                let slot = (self.rng.next() as usize % (DATA_LEN / 8 - 2)) * 8;
                out.extend(mov_imm(base, scratch + slot as u64));
                let (rt, rt2) = (self.reg(), self.reg());
                out.push(str_imm(3, rt, base, 0));
                out.push(ldr_imm(3, rt2, base, 0));
            }
            22 => {
                // A byte/half store and load, exercising sizes and zero-extension.
                let base = self.base();
                let slot = (self.rng.next() as usize % (DATA_LEN - 8)) & !7;
                out.extend(mov_imm(base, scratch + slot as u64));
                let size = self.rng.bits(2);
                let (rt, rt2) = (self.reg(), self.reg());
                out.push(str_imm(size, rt, base, 0));
                out.push(ldr_imm(size, rt2, base, 0));
            }
            23 => {
                // A stack-pointer adjustment, a read-back, and SP put back.
                let n = (self.rng.bits(9) & 0x1f0) + 16;
                out.push(sub_imm(true, SP, SP, n, false));
                out.push(add_imm(true, if rd == 0 { 1 } else { rd }, SP, 0, false));
                out.push(add_imm(true, SP, SP, n, false));
            }
            24 => out.push(ubfm(sf, rd, rn, immr & bw(sf), imms & bw(sf))),
            25 => out.push(sbfm(sf, rd, rn, immr & bw(sf), imms & bw(sf))),
            26 => out.push(bfm(sf, rd, rn, immr & bw(sf), imms & bw(sf))),
            27 => out.push(extr(sf, rd, rn, rm, imms & bw(sf))),
            28 => out.push(ccmp_imm(sf, rn, self.rng.bits(5), cond, self.rng.bits(4))),
            29 => out.push(ccmn_reg(sf, rn, rm, cond, self.rng.bits(4))),
            30 => out.push(csinv(sf, rd, rn, rm, cond)),
            31 => out.push(csneg(sf, rd, rn, rm, cond)),
            32 => out.push(data1(sf, rd, rn, self.rng.bits(2))), // RBIT/REV16/CLZ/CLS
            33 => out.push(udiv(sf, rd, rn, rm)),
            34 => out.push(sdiv(sf, rd, rn, rm)),
            35 => out.push(mulh(rd, rn, rm, self.rng.bool())), // UMULH/SMULH
            36 => {
                // ADC/SBC use the carry flag, so a flag-setter first.
                out.push(subs_reg(sf, 31, rn, rm, 0, 0));
                out.push(adc(sf, rd, rn, rm, self.rng.bool()));
            }
            37 => {
                // A pre/post-index load and store, exercising writeback. The
                // transfer register must differ from the base.
                let base = self.base();
                let slot = (self.rng.next() as usize % (DATA_LEN / 2)) & !7;
                out.extend(mov_imm(base, scratch + slot as u64));
                let (t1, t2) = (self.reg_not(&[base]), self.reg_not(&[base]));
                out.push(str_imm_wb(3, t1, base, 8, self.rng.bool()));
                out.push(ldr_imm_wb(3, t2, base, -8, self.rng.bool()));
            }
            38 => {
                // A register pair store and load; the two load targets must differ.
                let base = self.base();
                let slot = (self.rng.next() as usize % (DATA_LEN / 2)) & !7;
                out.extend(mov_imm(base, scratch + slot as u64));
                let (s1, s2) = (self.reg(), self.reg());
                out.push(stp(s1, s2, base, 0));
                let t1 = self.reg_not(&[base]);
                let t2 = self.reg_not(&[base, t1]);
                out.push(ldp(t1, t2, base, 16));
            }
            39 => {
                // Register-offset accesses with a scaled or sign-extended index.
                let base = self.base();
                let idx = self.reg_not(&[base, 0]);
                let slot = (self.rng.next() as usize % (DATA_LEN / 2)) & !7;
                out.extend(mov_imm(base, scratch + slot as u64));
                out.push(movz(true, idx, self.rng.bits(5), 0));
                let (rt, rt2) = (self.reg(), self.reg());
                let size = self.rng.bits(2).min(2); // signed loads are byte/half/word
                out.push(str_reg_idx(size, rt, base, idx, 0b011, true));
                out.push(ldrs_reg_idx(size, rt2, base, idx, 0b110, true));
            }
            40 => out.push(neon3(
                self.rng.next() as u32 % 6,
                vreg(self.rng),
                vreg(self.rng),
                vreg(self.rng),
            )),
            41 => out.push(fmov_d_x(vreg(self.rng), rd)),
            42 => out.push(fmov_x_d(rd, vreg(self.rng))),
            43 => out.push(dup_2d(vreg(self.rng), rn)),
            44 => out.push(ins_d1(vreg(self.rng), rn)),
            45 => out.push(umov_d0(rd, vreg(self.rng))),
            46 | 47 => {
                // A SIMD load and store, q/d/s.
                let base = self.base();
                let slot = (self.rng.next() as usize % (DATA_LEN / 16 - 1)) * 16;
                out.extend(mov_imm(base, scratch + slot as u64));
                let sz = self.rng.next() as u32 % 3;
                out.push(simd_str(sz, vreg(self.rng), base));
                out.push(simd_ldr(sz, vreg(self.rng), base));
            }
            48 | 49 => {
                // An LSE atomic read-modify-write; Rs and Rt differ from the base.
                let base = self.base();
                let slot = (self.rng.next() as usize % (DATA_LEN / 8 - 1)) * 8;
                out.extend(mov_imm(base, scratch + slot as u64));
                let rs = self.reg_not(&[base]);
                let rt = self.reg_not(&[base]);
                let size = 2 + self.rng.bits(1);
                out.push(lse_atomic(
                    size,
                    self.rng.bits(3),
                    self.rng.bool(),
                    rs,
                    rt,
                    base,
                ));
            }
            50 | 51 => {
                // LD1/ST1 of 1..4 registers, or LD2..LD4 / ST2..ST4, with or
                // without post-index writeback.
                let base = self.base();
                let slot = (self.rng.next() as usize % (DATA_LEN / 64 - 1)) * 64;
                out.extend(mov_imm(base, scratch + slot as u64));
                let load = self.rng.bool();
                let (opcode, regs) = match self.rng.next() % 7 {
                    0 => (0b0111, 1),
                    1 => (0b1010, 2),
                    2 => (0b0110, 3),
                    3 => (0b0010, 4),
                    4 => (0b1000, 2),
                    5 => (0b0100, 3),
                    _ => (0b0000, 4),
                };
                let q = self.rng.bool();
                let size = self.rng.bits(2).min(if q { 3 } else { 2 });
                let post = self.rng.bool();
                let _ = regs;
                out.push(simd_struct(
                    q,
                    load,
                    post,
                    opcode,
                    size,
                    base,
                    vreg(self.rng),
                ));
            }
            52 => {
                // A single lane, or one element replicated.
                let base = self.base();
                let slot = (self.rng.next() as usize % (DATA_LEN / 16 - 1)) * 16;
                out.extend(mov_imm(base, scratch + slot as u64));
                let idx = self.rng.bits(2);
                if self.rng.bool() {
                    out.push(simd_lane_s(self.rng.bool(), idx, base, vreg(self.rng)));
                } else {
                    out.push(ld1r(
                        self.rng.bool(),
                        self.rng.bits(2),
                        base,
                        vreg(self.rng),
                    ));
                }
            }
            53 => {
                // A pair of Q, D or S registers.
                let base = self.base();
                let slot = (self.rng.next() as usize % (DATA_LEN / 64 - 1)) * 64;
                out.extend(mov_imm(base, scratch + slot as u64));
                let opc = self.rng.next() as u32 % 3;
                let (a, b2) = (vreg(self.rng), vreg(self.rng));
                out.push(simd_pair(opc, false, a, b2, base));
                let t1 = vreg(self.rng);
                let t2 = (t1 + 1 + (self.rng.next() % 31) as u32) % 32;
                out.push(simd_pair(opc, true, t1, t2, base));
            }
            54 | 55 => {
                // Unaligned accesses, some of them across a page boundary
                // inside the data region.
                let base = self.base();
                let addr = if self.rng.bool() {
                    scratch + 0x1000 - 1 - (self.rng.next() % 15)
                } else {
                    (scratch + (self.rng.next() % (DATA_LEN as u64 - 32))) | 1
                };
                out.extend(mov_imm(base, addr));
                let (rt, rt2) = (self.reg_not(&[base]), self.reg_not(&[base]));
                let size = self.rng.bits(2);
                out.push(stur(size, rt, base, 0));
                out.push(ldur(size, rt2, base, 0));
                if self.rng.bool() {
                    let v = vreg(self.rng);
                    out.push(simd_ldr(2, v, base));
                }
            }
            56 => {
                // A load-exclusive / store-exclusive retry loop, plain or
                // acquire/release.
                let base = self.base();
                let slot = (self.rng.next() as usize % (DATA_LEN / 8 - 1)) * 8;
                out.extend(mov_imm(base, scratch + slot as u64));
                let rt = self.reg_not(&[base]);
                let ws = self.reg_not(&[base, rt]);
                let ordered = self.rng.bool();
                let size = 2 + self.rng.bits(1);
                let top = out.len();
                out.push(ldxr(size, ordered, rt, base));
                out.push(add_imm(size == 3, rt, rt, 1 + self.rng.bits(8), false));
                out.push(stxr(size, ordered, ws, rt, base));
                out.push(cbnz(false, ws, (top as i64 - out.len() as i64) * 4));
            }
            57 => {
                // Compare-and-swap, which may or may not swap.
                let base = self.base();
                let slot = (self.rng.next() as usize % (DATA_LEN / 8 - 1)) * 8;
                out.extend(mov_imm(base, scratch + slot as u64));
                let rs = self.reg_not(&[base]);
                let rt = self.reg_not(&[base, rs]);
                let size = 2 + self.rng.bits(1);
                out.push(cas(size, self.rng.bool(), self.rng.bool(), rs, rt, base));
            }
            58 => {
                // Store-release then load-acquire.
                let base = self.base();
                let slot = (self.rng.next() as usize % (DATA_LEN / 8 - 1)) * 8;
                out.extend(mov_imm(base, scratch + slot as u64));
                let (rt, rt2) = (self.reg(), self.reg_not(&[base]));
                let size = self.rng.bits(2);
                out.push(stlr(size, rt, base));
                out.push(ldar(size, rt2, base));
            }
            59 => {
                // A literal load: b over the 8-byte literal, then load it.
                let rt = self.reg();
                let v = self.rng.next();
                out.push(b(12));
                out.push(v as u32);
                out.push((v >> 32) as u32);
                out.push(match self.rng.next() % 4 {
                    0 => ldr_literal_x(rt, -8),
                    1 => ldr_literal_w(rt, -8),
                    2 => ldrsw_literal(rt, -8),
                    _ => ldr_literal_d(vreg(self.rng), -8),
                });
            }
            60 => {
                // DC ZVA zeroes a 64-byte block.
                let base = self.base();
                let slot = (self.rng.next() as usize % (DATA_LEN / 64)) * 64;
                out.extend(mov_imm(
                    base,
                    scratch + slot as u64 + self.rng.bits(6) as u64,
                ));
                out.push(dc_zva(base));
            }
            61 => {
                // The flags as a register, and back.
                if self.rng.bool() {
                    out.push(mrs(rd, NZCV));
                } else {
                    out.push(msr(NZCV, rn));
                }
            }
            62 => {
                // A push and pop through SP: a pre-indexed pair store, a load
                // and a store at offsets, then a post-indexed pair load that
                // puts SP back.
                let (a, b2, c2, d2) = (self.reg(), self.reg(), self.reg(), self.reg());
                let e = self.reg_not(&[0]);
                let f = self.reg_not(&[0, e]);
                out.push(stp_pre(a, b2, SP, -16));
                out.push(ldr_imm(3, c2, SP, 8));
                out.push(str_imm(3, d2, SP, 0));
                out.push(ldp_post(e, f, SP, 16));
            }
            63 => {
                // SP and an extended register: SP minus it, SP plus it into a
                // register, and SP put back.
                let k = self.reg_not(&[0]);
                let r = self.reg_not(&[0, k]);
                out.push(movz(true, k, 1 + self.rng.bits(4), 0));
                out.push(sub_ext(SP, SP, k, EXT_UXTX, 4));
                out.push(add_ext(r, SP, k, EXT_UXTX, 4));
                out.push(add_ext(SP, SP, k, EXT_UXTX, 4));
            }
            _ => out.push(add_reg(sf, rd, rn, rm, arith_shift, amount)),
        }
    }
}

/// The widest bit index for a bitfield at this register width.
fn bw(sf: bool) -> u32 {
    if sf { 63 } else { 31 }
}

/// A random SIMD register 0..31.
fn vreg(rng: &mut Rng) -> u32 {
    (rng.next() % 32) as u32
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
    // SIMD loads and stores (unsigned offset 0): q, d or s by `sz` (2, 1, 0).
    pub fn ldr_q(rt: Reg, rn: Reg, off: u32) -> u32 {
        0x3dc0_0000 | (off / 16) << 10 | rn << 5 | rt
    }
    pub fn simd_ldr(sz: u32, rt: Reg, rn: Reg) -> u32 {
        match sz {
            0 => 0xbd40_0000 | rn << 5 | rt, // ldr s
            1 => 0xfd40_0000 | rn << 5 | rt, // ldr d
            _ => 0x3dc0_0000 | rn << 5 | rt, // ldr q
        }
    }
    pub fn simd_str(sz: u32, rt: Reg, rn: Reg) -> u32 {
        match sz {
            0 => 0xbd00_0000 | rn << 5 | rt, // str s
            1 => 0xfd00_0000 | rn << 5 | rt, // str d
            _ => 0x3d80_0000 | rn << 5 | rt, // str q
        }
    }
    /// Integer NEON 3-same on .16b/.4s: ADD, SUB, AND, ORR, EOR, MUL.
    pub fn neon3(op: u32, vd: Reg, vn: Reg, vm: Reg) -> u32 {
        let base = match op {
            0 => 0x4e20_8400, // add .16b
            1 => 0x6e20_8400, // sub .16b
            2 => 0x4e20_1c00, // and .16b
            3 => 0x4ea0_1c00, // orr .16b
            4 => 0x6e20_1c00, // eor .16b
            _ => 0x4eb0_9c00, // mul .4s
        };
        base | vm << 16 | vn << 5 | vd
    }
    pub fn fmov_d_x(vd: Reg, xn: Reg) -> u32 {
        0x9e67_0000 | xn << 5 | vd
    }
    pub fn fmov_x_d(xd: Reg, vn: Reg) -> u32 {
        0x9e66_0000 | vn << 5 | xd
    }
    pub fn dup_2d(vd: Reg, xn: Reg) -> u32 {
        0x4e08_0c00 | xn << 5 | vd
    }
    pub fn ins_d1(vd: Reg, xn: Reg) -> u32 {
        0x4e18_1c00 | xn << 5 | vd
    }
    pub fn umov_d0(xd: Reg, vn: Reg) -> u32 {
        0x4e08_3c00 | vn << 5 | xd
    }
    /// STR/LDRS (register offset) with an extend option and the scale bit.
    pub fn str_reg_idx(size: u32, rt: Reg, rn: Reg, rm: Reg, option: u32, s: bool) -> u32 {
        size << 30 | 0x3820_0800 | rm << 16 | option << 13 | (s as u32) << 12 | rn << 5 | rt
    }
    pub fn ldrs_reg_idx(size: u32, rt: Reg, rn: Reg, rm: Reg, option: u32, s: bool) -> u32 {
        size << 30 | 0x38a0_0800 | rm << 16 | option << 13 | (s as u32) << 12 | rn << 5 | rt
    }
    /// STUR/LDUR (unscaled immediate) of a general register.
    pub fn stur(size: u32, rt: Reg, rn: Reg, imm: i32) -> u32 {
        size << 30 | 0x3800_0000 | ((imm as u32 & 0x1ff) << 12) | rn << 5 | rt
    }
    pub fn ldur(size: u32, rt: Reg, rn: Reg, imm: i32) -> u32 {
        size << 30 | 0x3840_0000 | ((imm as u32 & 0x1ff) << 12) | rn << 5 | rt
    }
    /// LD1..LD4 / ST1..ST4 (multiple structures), optionally post-indexed by
    /// the transfer size.
    pub fn simd_struct(q: bool, load: bool, post: bool, opcode: u32, size: u32, rn: Reg, rt: Reg) -> u32 {
        let mut w = (q as u32) << 30 | 0x0c00_0000 | (load as u32) << 22 | opcode << 12 | size << 10 | rn << 5 | rt;
        if post {
            w |= 1 << 23 | 31 << 16;
        }
        w
    }
    /// LD1/ST1 of one 32-bit lane.
    pub fn simd_lane_s(load: bool, idx: u32, rn: Reg, rt: Reg) -> u32 {
        (idx >> 1) << 30 | 0x0d00_0000 | (load as u32) << 22 | 0b100 << 13 | (idx & 1) << 12 | rn << 5 | rt
    }
    /// LD1R: one element, replicated.
    pub fn ld1r(q: bool, size: u32, rn: Reg, rt: Reg) -> u32 {
        (q as u32) << 30 | 0x0d40_c000 | size << 10 | rn << 5 | rt
    }
    /// LDP/STP of S (opc 0), D (1) or Q (2) registers, signed offset 0.
    /// STP (pre-index) and LDP (post-index) of X registers.
    pub fn stp_pre(rt: Reg, rt2: Reg, rn: Reg, imm: i32) -> u32 {
        0xa980_0000 | ((imm / 8) as u32 & 0x7f) << 15 | rt2 << 10 | rn << 5 | rt
    }
    pub fn ldp_post(rt: Reg, rt2: Reg, rn: Reg, imm: i32) -> u32 {
        0xa8c0_0000 | ((imm / 8) as u32 & 0x7f) << 15 | rt2 << 10 | rn << 5 | rt
    }
    pub fn simd_pair(opc: u32, load: bool, rt: Reg, rt2: Reg, rn: Reg) -> u32 {
        opc << 30 | 0x2d00_0000 | (load as u32) << 22 | rt2 << 10 | rn << 5 | rt
    }
    /// LDXR/LDAXR and STXR/STLXR.
    pub fn ldxr(size: u32, acquire: bool, rt: Reg, rn: Reg) -> u32 {
        size << 30 | 0x085f_7c00 | (acquire as u32) << 15 | rn << 5 | rt
    }
    pub fn stxr(size: u32, release: bool, rs: Reg, rt: Reg, rn: Reg) -> u32 {
        size << 30 | 0x0800_7c00 | rs << 16 | (release as u32) << 15 | rn << 5 | rt
    }
    /// STLR and LDAR.
    pub fn stlr(size: u32, rt: Reg, rn: Reg) -> u32 {
        size << 30 | 0x089f_fc00 | rn << 5 | rt
    }
    pub fn ldar(size: u32, rt: Reg, rn: Reg) -> u32 {
        size << 30 | 0x08df_fc00 | rn << 5 | rt
    }
    /// The literal loads, `offset` bytes from the instruction.
    pub fn ldr_literal_x(rt: Reg, offset: i64) -> u32 {
        0x5800_0000 | ((offset / 4) as u32 & 0x7ffff) << 5 | rt
    }
    pub fn ldr_literal_w(rt: Reg, offset: i64) -> u32 {
        0x1800_0000 | ((offset / 4) as u32 & 0x7ffff) << 5 | rt
    }
    pub fn ldrsw_literal(rt: Reg, offset: i64) -> u32 {
        0x9800_0000 | ((offset / 4) as u32 & 0x7ffff) << 5 | rt
    }
    pub fn ldr_literal_d(rt: Reg, offset: i64) -> u32 {
        0x5c00_0000 | ((offset / 4) as u32 & 0x7ffff) << 5 | rt
    }
    pub fn dc_zva(rt: Reg) -> u32 {
        0xd50b_7420 | rt
    }
    /// An LSE atomic (LDADD/LDCLR/.../SWP). `op` 0..7 picks the operation; `swp`
    /// makes it SWP. Non-acquire, non-release.
    pub fn lse_atomic(size: u32, op: u32, swp: bool, rs: Reg, rt: Reg, rn: Reg) -> u32 {
        if swp {
            // SWP: 00 111000 0 01 1 Rs 1 00000 Rn Rt  (o3=1, opc=000 base 0x20_8000)
            size << 30 | 0x3820_8000 | rs << 16 | rn << 5 | rt
        } else {
            // LD<op>: 00 111000 0 01 1 Rs 0 ooo 00 Rn Rt  (opc in bits 14:12)
            size << 30 | 0x3820_0000 | rs << 16 | (op & 7) << 12 | rn << 5 | rt
        }
    }
}
use extra::*;

// ---------------------------------------------------------------- the test

/// How the translated run is set up.
#[derive(Clone, Copy)]
struct Run {
    /// Fast paths on (the inline TLB, chaining, the jump cache).
    fast: bool,
    /// Run with the MMU on, at this virtual address of the program's first
    /// word (page tables from `page_tables`); `None` runs with the MMU off.
    at: Option<u64>,
}

fn guest_memory(shared: &Shared) -> Arc<GuestMemory> {
    let block: Arc<dyn RamBlock> = Arc::new(ExternalRam {
        ptr: shared.ptr,
        len: shared.len,
    });
    Arc::new(GuestMemory::new(vec![(shared.addr(), block)]).unwrap())
}

/// Runs the program's body translated. With `run.at`, the MMU is on and the
/// code is reached at that virtual address; `cache` may be shared between
/// runs, so a later run can reuse an earlier run's translations.
fn run_translated(
    shared: &Shared,
    prog: &Program,
    init: &Captured,
    run: Run,
    cache: &mut CodeCache,
    tables: Option<u64>,
) -> Captured {
    let mut cpu = Cpu::new(guest_memory(shared));
    cpu.state.x[..31].copy_from_slice(&init.x);
    cpu.state.sp = init.sp;
    cpu.state.nzcv = init.nzcv;
    cpu.state.v = init.v;
    cpu.state.pc = prog.guest_pc;
    if let (Some(at), Some(l0)) = (run.at, tables) {
        cpu.state.mmu = crate_jit::state::MmuRegs {
            sctlr_el1: 1,
            ttbr0_el1: l0,
            tcr_el1: 16, // T0SZ 16 (48-bit), TG0 4 KiB
            ..Default::default()
        };
        cpu.state.pc = at + (prog.body_start * 4) as u64;
    }
    let stop = cpu.run(cache);
    if stop != Stop::Brk(0) {
        eprintln!(
            "translated stop {stop:?} at pc {:#x} (fast {})",
            cpu.state.pc, run.fast
        );
        eprintln!("  esr/far syndrome {:#x}", cpu.state.exception_syndrome);
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
    out.v = cpu.state.v;
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
// SAFETY: as for Send.
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

/// Where the page tables of the MMU test live, above everything the streams
/// touch.
const TABLES_OFF: usize = 0x40000;

/// Builds 4 KiB stage-1 tables in the shared RAM: every page of RAM mapped at
/// its own address, readable, writable and executable at EL0, and the code
/// pages mapped a second time at `alias` (the same physical pages).
fn page_tables(shared: &Shared, alias: u64, code_pages: std::ops::Range<u64>) -> u64 {
    let base = shared.addr();
    let mut next = TABLES_OFF;
    let mut alloc = || {
        let t = next;
        next += 0x1000;
        shared.write(t, &[0u8; 0x1000]);
        base + t as u64
    };
    let l0 = alloc();
    let read = |pa: u64| -> u64 {
        u64::from_le_bytes(shared.read((pa - base) as usize, 8).try_into().unwrap())
    };
    let write = |pa: u64, v: u64| shared.write((pa - base) as usize, &v.to_le_bytes());
    let mut map = |va: u64, pa: u64| {
        let mut table = l0;
        for level in 0..3 {
            let shift = 39 - 9 * level;
            let slot = table + ((va >> shift) & 0x1ff) * 8;
            let d = read(slot);
            table = if d & 1 == 0 {
                let t = alloc();
                write(slot, t | 3);
                t
            } else {
                d & 0x0000_ffff_ffff_f000
            };
        }
        // A page: valid, AF, AP=01 (EL0 and EL1 read/write), executable.
        write(table + ((va >> 12) & 0x1ff) * 8, pa | 3 | 1 << 10 | 1 << 6);
    };
    for off in (0..TABLES_OFF as u64).step_by(0x1000) {
        map(base + off, base + off);
    }
    for page in code_pages {
        map(alias + (page << 12), base + (page << 12));
    }
    l0
}

/// x30 relative to where the code ran when it points into the code (a link
/// value), so runs at different addresses compare equal.
fn normalize(mut c: Captured, code: u64, len: u64) -> Captured {
    if (code..code + len).contains(&c.x[30]) {
        c.x[30] -= code;
    }
    c.pc = 0;
    c.sig = 0;
    c
}

fn random_init(rng: &mut Rng, shared: &Shared) -> Captured {
    let mut init = Captured::default();
    for x in init.x.iter_mut() {
        *x = rng.next();
    }
    for lane in init.v.iter_mut() {
        for b in lane.iter_mut() {
            *b = rng.next() as u8;
        }
    }
    // SP: aligned, inside the scratch area so stack adjustments stay sane.
    init.sp = shared.addr() + (STACK_OFF + STACK_LEN * 3 / 4) as u64;
    init.nzcv = rng.next() & 0xf000_0000;
    init
}

/// One generated stream, run natively and translated (fast and slow, and with
/// `mmu_alias` also at a second virtual address through the MMU, sharing the
/// first run's translations), with everything compared.
fn check_stream(shared: &Shared, rng: &mut Rng, iter: usize, x18: bool, mmu_alias: Option<u64>) {
    let code_off = if rng.bool() {
        0x1000 - 4 * (rng.next() % 200) as usize
    } else {
        4 * (rng.next() % 64) as usize
    };
    let shape = Shape {
        count: 16 + (rng.next() % 16) as usize,
        code_off,
        x18: 18,
        use_x18: x18,
    };
    // The same stream twice: as the translator runs it (logical X18 in x18)
    // and as the CPU runs it (in x29, since the OS may clobber x18).
    let seed = rng.next();
    let prog = generate(&mut Rng(seed | 1), shape, shared.addr());
    let native_prog = generate(
        &mut Rng(seed | 1),
        Shape { x18: 29, ..shape },
        shared.addr(),
    );
    let mut init = random_init(rng, shared);
    init.x[29] = init.x[18];
    let mut native_init = init;
    native_init.x[18] = 0;

    // Lay the translated program into guest RAM, and the native one into an
    // executable region.
    let code: Vec<u8> = prog.words.iter().flat_map(|w| w.to_le_bytes()).collect();
    shared.write(0, &[0u8; 0x4000]);
    shared.write(code_off, &code);
    let exec = JitRegion::new(code.len() + 4096).unwrap();
    exec.writer().write_words(0, &native_prog.words);
    exec.flush(0, code.len());
    let len = code.len() as u64;

    shared.write(DATA_OFF, &vec![0u8; DATA_LEN]);
    let native = run_native(exec.exec_base() as u64, &native_init);
    // Only a clean BRK (SIGTRAP) makes the native run a valid oracle; a fault
    // or the watchdog means an unsafe generated stream, which is a generator
    // edge case, not a translator error, so skip it.
    if native.sig != libc::SIGTRAP as u64 {
        return;
    }
    let native_scratch = shared.read(DATA_OFF, DATA_LEN);
    let mut native = normalize(native, exec.exec_base() as u64, len);
    if x18 {
        // The CPU's x29 is the translated run's x18; neither run's other
        // register of the two is meaningful.
        native.x[18] = native.x[29];
        native.x[29] = init.x[29];
    }
    native.x[18] = if x18 { native.x[18] } else { 0 };

    let mut runs = vec![
        Run {
            fast: true,
            at: None,
        },
        Run {
            fast: false,
            at: None,
        },
    ];
    let tables = mmu_alias.map(|alias| {
        let first = (code_off as u64) >> 12;
        let last = (code_off as u64 + len) >> 12;
        page_tables(shared, alias, first..last + 1)
    });
    let mut shared_cache = CodeCache::with_mode(4 << 20, true).unwrap();
    if let Some(alias) = mmu_alias {
        runs = vec![
            Run {
                fast: true,
                at: Some(shared.addr() + code_off as u64),
            },
            Run {
                fast: true,
                at: Some(alias + code_off as u64),
            },
        ];
    }
    for run in runs {
        shared.write(DATA_OFF, &vec![0u8; DATA_LEN]);
        let mut fresh = CodeCache::with_mode(4 << 20, run.fast).unwrap();
        let cache = if mmu_alias.is_some() {
            &mut shared_cache
        } else {
            &mut fresh
        };
        let translated = run_translated(shared, &prog, &init, run, cache, tables);
        let translated_scratch = shared.read(DATA_OFF, DATA_LEN);
        let base = run.at.unwrap_or(shared.addr() + code_off as u64);
        let mut translated = normalize(translated, base, len);
        if !x18 {
            translated.x[18] = 0;
        }
        if native != translated || native_scratch != translated_scratch {
            eprintln!("run: fast {} at {:x?} x18 {x18}", run.fast, run.at);
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

fn iterations() -> usize {
    std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400)
}

#[test]
fn random_streams_match_the_cpu() {
    let shared = Shared::new(RAM_SIZE);
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for iter in 0..iterations() {
        check_stream(&shared, &mut rng, iter, false, None);
    }
}

/// Guest X18 is virtual (the host's x18 is the platform register): streams
/// that name it run translated with it in x18 and natively with it in x29.
#[test]
fn x18_streams_match_the_cpu() {
    let shared = Shared::new(RAM_SIZE);
    let mut rng = Rng(0x0bad_cafe_dead_beef);
    for iter in 0..iterations() / 2 {
        check_stream(&shared, &mut rng, iter, true, None);
    }
}

/// One translation serves every virtual address its physical page is mapped
/// at: the same code run through the MMU at two addresses, with one code
/// cache, must give the CPU's results both times.
#[test]
fn translations_are_shared_across_addresses() {
    let shared = Shared::new(RAM_SIZE);
    let mut rng = Rng(0x005e_ed0f_5a3e_d0d0);
    let alias = 0x7f00_0000_0000u64;
    for iter in 0..iterations() / 2 {
        check_stream(&shared, &mut rng, iter, false, Some(alias));
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
    for i in 0..32 {
        if native.v[i] != translated.v[i] {
            eprintln!(
                "  v{i}: native {:02x?} jit {:02x?}",
                native.v[i], translated.v[i]
            );
        }
    }
    if native.pc != translated.pc {
        eprintln!("  pc: native {:#x} jit {:#x}", native.pc, translated.pc);
    }
}
