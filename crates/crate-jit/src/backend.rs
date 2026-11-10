// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The translator proper: the trampoline that enters and leaves a block, the
//! block cache, the per-instruction emission, and the helpers the emitted code
//! calls back into. A dispatcher runs a guest CPU by translating the block at
//! its PC, entering it, and acting on why it returned.

use std::collections::HashMap;
use std::sync::Arc;

use crate_core::memory::GuestMemory;
use crate_host::jit::JitRegion;

use crate::asm::*;
use crate::decode::{self, CallKind, CondKind, Mem, MemKind, Op};
use crate::emit::{self, ExitInfo, Next};
use crate::mmu::{self, Access};
use crate::state::{CpuState, Exit, MmuRegs, offsets};

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
    // Remember the host stack, then make SP the state block's address.
    "mov x9, sp",
    "str x9, [x0, #{host_sp}]",
    "mov sp, x0",
    // Load the guest's flags, FP control and SIMD registers.
    "ldr x9, [sp, #{nzcv}]",
    "msr nzcv, x9",
    "ldr x9, [sp, #{fpcr}]",
    "msr fpcr, x9",
    "ldr x9, [sp, #{fpsr}]",
    "msr fpsr, x9",
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
    // Load guest X0..X29; X30 is loaded by the block's dispatch prologue.
    "ldp x0,  x1,  [sp, #({x}+0)]",
    "ldp x2,  x3,  [sp, #({x}+16)]",
    "ldp x4,  x5,  [sp, #({x}+32)]",
    "ldp x6,  x7,  [sp, #({x}+48)]",
    "ldp x8,  x9,  [sp, #({x}+64)]",
    "ldp x10, x11, [sp, #({x}+80)]",
    "ldp x12, x13, [sp, #({x}+96)]",
    "ldp x14, x15, [sp, #({x}+112)]",
    "ldp x16, x17, [sp, #({x}+128)]",
    "ldp x18, x19, [sp, #({x}+144)]",
    "ldp x20, x21, [sp, #({x}+160)]",
    "ldp x22, x23, [sp, #({x}+176)]",
    "ldp x24, x25, [sp, #({x}+192)]",
    "ldp x26, x27, [sp, #({x}+208)]",
    "ldp x28, x29, [sp, #({x}+224)]",
    // Jump to the block, whose first word loads guest X30.
    "ldr x30, [sp, #{dispatch}]",
    "br x30",
    ".globl _crate_jit_epilogue",
    "_crate_jit_epilogue:",
    // Reached from a block's save stub with SP = state block; return the exit.
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
    host_sp = const offsets::HOST_SP,
    nzcv = const offsets::NZCV,
    fpcr = const offsets::FPCR,
    fpsr = const offsets::FPSR,
    v = const offsets::V,
    x = const offsets::X,
    dispatch = const offsets::DISPATCH_TARGET,
    exit = const offsets::EXIT,
);

/// How many guest instructions a block may hold before it is cut off.
const MAX_BLOCK: usize = 1024;

/// Translated host code, with a cache from guest PC to the block's offset.
pub struct CodeCache {
    region: JitRegion,
    /// Next free byte in the region.
    cursor: usize,
    /// Guest PC to host offset of the block's dispatch prologue.
    blocks: HashMap<u64, usize>,
    /// Offset of the shared save stub, emitted once.
    save_stub: usize,
}

/// A fault while translating: the region filled up, or memory was unreadable.
#[derive(Debug)]
pub enum TranslateError {
    OutOfSpace,
}

impl CodeCache {
    pub fn new(bytes: usize) -> crate_core::Result<Self> {
        let region = JitRegion::new(bytes)?;
        let mut cache = CodeCache {
            region,
            cursor: 0,
            blocks: HashMap::new(),
            save_stub: 0,
        };
        cache.emit_save_stub();
        Ok(cache)
    }

    pub fn epilogue_addr() -> u64 {
        let f: unsafe extern "C" fn() = crate_jit_epilogue;
        f as usize as u64
    }

    fn emit_save_stub(&mut self) {
        let mut asm = Asm::new();
        emit::save_stub(&mut asm);
        let base = self.region.exec_base() as u64 + self.cursor as u64;
        let words = asm.finish(base);
        self.save_stub = self.cursor;
        self.put(&words);
    }

    fn put(&mut self, words: &[u32]) {
        let bytes = std::mem::size_of_val(words);
        let mut w = self.region.writer();
        w.write_words(self.cursor, words);
        drop(w);
        self.region.flush(self.cursor, bytes);
        self.cursor += bytes;
    }

    /// The host address of the block for `pc`, translating it if needed.
    pub fn block(&mut self, pc: u64, mem: &GuestMemory, el: u32, el1_regs: &MmuRegs) -> u64 {
        if let Some(&off) = self.blocks.get(&pc) {
            return self.region.exec_base() as u64 + off as u64;
        }
        let off = self.translate(pc, mem, el, el1_regs).unwrap_or_else(|_| {
            // Out of space: start over. Safe because a block is only entered
            // from the dispatcher, never while another block runs.
            self.reset();
            self.translate(pc, mem, el, el1_regs)
                .expect("a block in an empty cache")
        });
        self.region.exec_base() as u64 + off as u64
    }

    fn reset(&mut self) {
        self.cursor = 0;
        self.blocks.clear();
        self.emit_save_stub();
    }

    fn translate(
        &mut self,
        pc: u64,
        mem: &GuestMemory,
        el: u32,
        el1_regs: &MmuRegs,
    ) -> Result<usize, TranslateError> {
        let base = self.region.exec_base() as u64 + self.cursor as u64;
        let save_stub_rel = self.save_stub as i64 - self.cursor as i64;
        let mut t = Block {
            asm: Asm::new(),
            save_stub_rel,
            terminated: false,
        };
        emit::block_dispatch_prologue(&mut t.asm);

        let mut cur = pc;
        for _ in 0..MAX_BLOCK {
            let word = match read_insn(mem, el, el1_regs, cur) {
                Ok(w) => w,
                Err(_) => {
                    // The fetch itself faulted: end the block with an
                    // instruction-abort exit.
                    t.emit_fetch_fault(cur);
                    break;
                }
            };
            let op = decode::decode(word, cur);
            if t.emit(op, word, cur) {
                break; // the instruction ended the block
            }
            cur = cur.wrapping_add(4);
        }
        // A block that ran off the end without a terminator continues at `cur`.
        if !t.terminated {
            t.exit(Exit::Branch as u32, Next::Const(cur), ExitInfo::default());
        }

        let words = t.asm.finish(base);
        let bytes = std::mem::size_of_val(words.as_slice());
        if self.cursor + bytes > self.region.len() {
            return Err(TranslateError::OutOfSpace);
        }
        let off = self.cursor;
        self.put(&words);
        self.blocks.insert(pc, off);
        Ok(off)
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

/// Reads the instruction word at `pc`, honouring the guest MMU for execute.
fn read_insn(mem: &GuestMemory, el: u32, regs: &MmuRegs, pc: u64) -> Result<u32, mmu::Fault> {
    let t = mmu::translate(mem, regs, el, pc, Access::Execute)?;
    mem.read_u32(t.pa).map_err(|_| mmu::Fault {
        fsc: 0x10, // a synchronous external abort on the fetch
        write: false,
    })
}

/// One block being built.
struct Block {
    asm: Asm,
    /// Offset of the save stub relative to the start of this block.
    save_stub_rel: i64,
    terminated: bool,
}

impl Block {
    /// Emits an exit to the save stub.
    fn exit(&mut self, code: u32, next: Next, info: ExitInfo) {
        emit::exit(&mut self.asm, self.save_stub_rel, code, next, info);
        self.terminated = true;
    }

    fn emit_fetch_fault(&mut self, pc: u64) {
        // An instruction abort: EC 0x21 (from a lower EL) or 0x20; the
        // dispatcher builds ESR from the kind. Record the faulting PC.
        self.exit(
            Exit::Exception as u32,
            Next::None,
            ExitInfo {
                pc: Some(pc),
                syndrome: Some(FETCH_ABORT),
                ..Default::default()
            },
        );
    }

    /// Emits one instruction. Returns true if it ended the block.
    fn emit(&mut self, op: Op, word: u32, pc: u64) -> bool {
        match op {
            Op::Copy => {
                self.asm.emit(word);
                false
            }
            Op::PcRel { rd, imm, .. } => {
                if rd != 31 {
                    self.asm.mov_imm(rd, imm as u64);
                }
                false
            }
            Op::AddSubSpImm {
                sub,
                sf,
                set_flags,
                rd,
                rn,
                imm12,
                shift12,
            } => {
                self.emit_sp_imm(sub, sf, set_flags, rd, rn, imm12, shift12);
                false
            }
            Op::AddSubSpExt {
                sub,
                sf,
                set_flags,
                rd,
                rn,
                rm,
                option,
                amount,
            } => {
                self.emit_sp_ext(sub, sf, set_flags, rd, rn, rm, option, amount);
                false
            }
            Op::Memory(m) => {
                self.emit_memory(m, word, pc);
                false
            }
            Op::Branch { link, target } => {
                self.exit(
                    Exit::Branch as u32,
                    Next::Const(target),
                    ExitInfo {
                        link: link.then_some(pc.wrapping_add(4)),
                        ..Default::default()
                    },
                );
                true
            }
            Op::CondBranch { word, kind, taken } => {
                self.emit_cond_branch(word, kind, taken, pc.wrapping_add(4));
                true
            }
            Op::IndBranch { link, rn, .. } => {
                self.exit(
                    Exit::Branch as u32,
                    Next::Reg(rn),
                    ExitInfo {
                        link: link.then_some(pc.wrapping_add(4)),
                        ..Default::default()
                    },
                );
                true
            }
            Op::Sysreg { read, reg, rt } => {
                self.exit(
                    Exit::Sysreg as u32,
                    Next::Const(pc.wrapping_add(4)),
                    ExitInfo {
                        sysreg: Some((reg, rt, read)),
                        pc: Some(pc),
                        ..Default::default()
                    },
                );
                true
            }
            Op::MsrImm { .. } | Op::System { .. } => {
                // Handled by a helper that takes the whole word.
                self.exit(
                    Exit::Sysreg as u32,
                    Next::Const(pc.wrapping_add(4)),
                    ExitInfo {
                        sysreg: Some((SYSTEM_WORD_MARKER, word & 31, false)),
                        pc: Some(pc),
                        syndrome: Some(word as u64),
                        ..Default::default()
                    },
                );
                true
            }
            Op::Call { kind, imm } => {
                self.exit(
                    Exit::Call as u32,
                    Next::Const(pc.wrapping_add(4)),
                    ExitInfo {
                        pc: Some(pc),
                        syndrome: Some(((call_kind_code(kind) as u64) << 16) | imm as u64),
                        ..Default::default()
                    },
                );
                true
            }
            Op::Wait { event } => {
                self.exit(
                    if event {
                        Exit::WaitForEvent as u32
                    } else {
                        Exit::WaitForInterrupt as u32
                    },
                    Next::Const(pc.wrapping_add(4)),
                    ExitInfo::default(),
                );
                true
            }
            Op::Eret => {
                self.exit(
                    Exit::Exception as u32,
                    Next::None,
                    ExitInfo {
                        pc: Some(pc),
                        syndrome: Some(ERET_MARKER),
                        ..Default::default()
                    },
                );
                true
            }
            Op::Undefined => {
                self.exit(
                    Exit::Undefined as u32,
                    Next::None,
                    ExitInfo {
                        pc: Some(pc),
                        syndrome: Some(word as u64),
                        ..Default::default()
                    },
                );
                true
            }
        }
    }

    /// A victim host register to borrow, not equal to `avoid`.
    fn victim(avoid: &[u32]) -> u32 {
        (9..=15).find(|r| !avoid.contains(r)).unwrap()
    }

    fn emit_sp_imm(
        &mut self,
        sub: bool,
        sf: bool,
        set_flags: bool,
        rd: u32,
        rn: u32,
        imm12: u32,
        shift12: bool,
    ) {
        let discard = rd == 31 && set_flags;
        let v = Self::victim(&[rd, rn]);
        self.asm.emit(str_imm(3, v, SP, offsets::SCRATCH));
        // Load the source.
        if rn == 31 {
            self.asm.emit(ldr_imm(3, v, SP, offsets::SP));
        } else {
            self.asm.emit(mov(v, rn));
        }
        // Compute into v, setting host flags when asked.
        let op = match (sub, set_flags) {
            (false, false) => add_imm(sf, v, v, imm12, shift12),
            (true, false) => sub_imm(sf, v, v, imm12, shift12),
            (false, true) => adds_imm(sf, v, v, imm12, shift12),
            (true, true) => subs_imm(sf, v, v, imm12, shift12),
        };
        self.asm.emit(op);
        // Store the result.
        if !discard {
            if rd == 31 {
                self.asm.emit(str_imm(3, v, SP, offsets::SP));
            } else {
                self.asm.emit(mov(rd, v));
            }
        }
        self.asm.emit(ldr_imm(3, v, SP, offsets::SCRATCH));
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_sp_ext(
        &mut self,
        sub: bool,
        sf: bool,
        set_flags: bool,
        rd: u32,
        rn: u32,
        rm: u32,
        option: u32,
        amount: u32,
    ) {
        let discard = rd == 31 && set_flags;
        let v = Self::victim(&[rd, rn, rm]);
        let vm = Self::victim(&[rd, rn, rm, v]);
        self.asm.emit(stp(v, vm, SP, offsets::SCRATCH as i32));
        if rn == 31 {
            self.asm.emit(ldr_imm(3, v, SP, offsets::SP));
        } else {
            self.asm.emit(mov(v, rn));
        }
        // rm is never SP (register 31 there is the zero register).
        if rm == 31 {
            self.asm.emit(movz(true, vm, 0, 0));
        } else {
            self.asm.emit(mov(vm, rm));
        }
        let op = match (sub, set_flags) {
            (false, false) => add_ext(v, v, vm, option, amount),
            (true, false) => sub_ext(v, v, vm, option, amount),
            (false, true) => adds_ext(v, v, vm, option, amount),
            (true, true) => subs_ext(v, v, vm, option, amount),
        };
        let _ = sf; // extended-register add/sub here is always 64-bit (SP form)
        self.asm.emit(op);
        if !discard {
            if rd == 31 {
                self.asm.emit(str_imm(3, v, SP, offsets::SP));
            } else {
                self.asm.emit(mov(rd, v));
            }
        }
        self.asm.emit(ldp(v, vm, SP, offsets::SCRATCH as i32));
    }

    fn emit_memory(&mut self, _m: Mem, word: u32, pc: u64) {
        // The slow, always-correct path: a call into Rust that does the whole
        // access (translation, the access, sign-extension, writeback and any
        // fault). It is the only mid-block call; the inline fast path over a
        // software TLB is a later speed-up that does not change this.
        let helper: unsafe extern "C" fn(*mut Cpu, u32) -> u64 = jit_mem;
        emit::call_mem(&mut self.asm, helper as usize as u64, word);
        // x3 holds the result: 0 ok, non-zero means a fault whose exception
        // fields the helper has set. On a fault, exit at once (the registers
        // are already in the state); otherwise reload and carry on.
        let ok = self.asm.label();
        self.asm.cbz(true, 3, ok);
        self.emit_fault_exit(pc);
        self.asm.bind(ok);
        emit::reload_full(&mut self.asm);
    }

    /// A memory fault reported by the helper, with the register file already in
    /// the state: record the faulting PC, set the exit code, and return through
    /// the epilogue without saving again.
    fn emit_fault_exit(&mut self, pc: u64) {
        self.asm.mov_imm(0, pc);
        self.asm.emit(str_imm(3, 0, SP, offsets::PC));
        self.asm.mov_imm(0, Exit::Exception as u64);
        self.asm.emit(str_imm(2, 0, SP, offsets::EXIT));
        self.asm.emit(ldr_imm(3, 0, SP, offsets::EPILOGUE));
        self.asm.emit(br(0));
    }

    fn emit_cond_branch(&mut self, word: u32, kind: CondKind, taken: u64, fallthrough: u64) {
        let taken_label = self.asm.label();
        match kind {
            CondKind::Cond => {
                let cond = word & 0xf;
                self.asm.b_cond(cond, taken_label);
            }
            CondKind::Cbz { sf, rt } => self.asm.cbz(sf, rt, taken_label),
            CondKind::Cbnz { sf, rt } => self.asm.cbnz(sf, rt, taken_label),
            CondKind::Tbz { rt, bit } => self.asm.tbz(rt, bit, taken_label),
            CondKind::Tbnz { rt, bit } => self.asm.tbnz(rt, bit, taken_label),
        }
        // Fall-through (condition false).
        emit::exit(
            &mut self.asm,
            self.save_stub_rel,
            Exit::Branch as u32,
            Next::Const(fallthrough),
            ExitInfo::default(),
        );
        self.asm.bind(taken_label);
        emit::exit(
            &mut self.asm,
            self.save_stub_rel,
            Exit::Branch as u32,
            Next::Const(taken),
            ExitInfo::default(),
        );
        self.terminated = true;
    }
}

fn adds_imm(sf: bool, rd: Reg, rn: Reg, imm12: u32, shift12: bool) -> u32 {
    (sf as u32) << 31 | 0x3100_0000 | (shift12 as u32) << 22 | imm12 << 10 | rn << 5 | rd
}
fn subs_imm(sf: bool, rd: Reg, rn: Reg, imm12: u32, shift12: bool) -> u32 {
    (sf as u32) << 31 | 0x7100_0000 | (shift12 as u32) << 22 | imm12 << 10 | rn << 5 | rd
}
fn adds_ext(rd: Reg, rn: Reg, rm: Reg, option: u32, amount: u32) -> u32 {
    0xab20_0000 | rm << 16 | option << 13 | amount << 10 | rn << 5 | rd
}
fn subs_ext(rd: Reg, rn: Reg, rm: Reg, option: u32, amount: u32) -> u32 {
    0xeb20_0000 | rm << 16 | option << 13 | amount << 10 | rn << 5 | rd
}

/// Markers packed into an exit's syndrome field the dispatcher recognises.
const FETCH_ABORT: u64 = 1 << 40;
const ERET_MARKER: u64 = 1 << 41;
/// A Sysreg exit whose encoding is this marker carries a whole instruction
/// word in the syndrome, for MSR-immediate and maintenance ops.
const SYSTEM_WORD_MARKER: u16 = 0xffff;

fn call_kind_code(kind: CallKind) -> u32 {
    match kind {
        CallKind::Svc => 0,
        CallKind::Hvc => 1,
        CallKind::Smc => 2,
        CallKind::Brk => 3,
    }
}

/// The memory helper: does the whole load or store the block asked for. Returns
/// 0 on success and 1 on a fault (with the exception fields set).
///
/// # Safety
///
/// `cpu` is a `*mut Cpu` a translated block passed; it is valid for the block's
/// run.
unsafe extern "C" fn jit_mem(cpu: *mut Cpu, word: u32) -> u64 {
    // SAFETY: the dispatcher owns the Cpu for the block's whole run.
    let cpu = unsafe { &mut *cpu };
    match cpu.memory_access(word) {
        Ok(()) => 0,
        Err(fault) => {
            cpu.take_data_abort(fault);
            1
        }
    }
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

/// A guest CPU the dispatcher drives.
pub struct Cpu {
    pub state: Box<CpuState>,
    pub mem: Arc<GuestMemory>,
}

impl Cpu {
    pub fn new(mem: Arc<GuestMemory>) -> Box<Self> {
        let mut cpu = Box::new(Cpu {
            state: CpuState::new(),
            mem,
        });
        cpu.state.epilogue = CodeCache::epilogue_addr();
        let ptr: *mut Cpu = &mut *cpu;
        cpu.state.cpu_ptr = ptr as u64;
        cpu
    }

    pub fn el(&self) -> u32 {
        ((self.state.pstate >> 2) & 3) as u32
    }

    /// Runs the guest until a block returns something the translator cannot
    /// handle on its own (a call, an exception, a system register, or an
    /// unknown instruction). Branches and waits are followed here.
    pub fn run(&mut self, cache: &mut CodeCache) -> Stop {
        let mem = self.mem.clone();
        loop {
            let pc = self.state.pc;
            let el = self.el();
            let regs = self.state.mmu;
            let target = cache.block(pc, &mem, el, &regs);
            self.state.dispatch_target = target;
            // SAFETY: the state is a stable heap box; the enter trampoline runs
            // the block at `target` and returns the exit code.
            let code = unsafe { crate_jit_enter(&mut *self.state) } as u32;
            match Exit::from_u32(code) {
                Some(Exit::Branch) | Some(Exit::WaitForInterrupt) | Some(Exit::WaitForEvent) => {
                    self.state.pc = self.state.next_pc;
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
                Some(Exit::Sysreg) => return Stop::Sysreg,
                Some(Exit::Exception) => return Stop::Exception,
                None => return Stop::Unknown(code),
            }
        }
    }

    fn mmu_regs(&self) -> MmuRegs {
        self.state.mmu
    }

    /// The guest register `n`, reading SP (31) from the state's SP.
    fn xread(&self, n: u32) -> u64 {
        if n == 31 {
            self.state.sp
        } else {
            self.state.x[n as usize]
        }
    }

    fn xwrite(&mut self, n: u32, value: u64) {
        if n == 31 {
            self.state.sp = value;
        } else {
            self.state.x[n as usize] = value;
        }
    }

    /// Performs the load or store `word` describes, from the current state.
    fn memory_access(&mut self, word: u32) -> Result<(), mmu::Fault> {
        let op = decode::decode(word, self.state.pc);
        let Op::Memory(m) = op else {
            unreachable!("jit_mem on a non-memory word {word:#x}");
        };
        let el = self.el();
        let regs = self.mmu_regs();
        let base = self.xread(m.rn);

        match m.kind {
            MemKind::Gpr => {
                let addr = self.effective_address(&m, base);
                let n = 1u64 << m.size;
                let t = mmu::translate(
                    &self.mem,
                    &regs,
                    el,
                    addr,
                    if m.load { Access::Read } else { Access::Write },
                )?;
                if m.load {
                    self.xwrite_z(m.rt, self.load_value(t.pa, &m)?);
                } else {
                    let value = self.xread_z(m.rt);
                    self.write_phys(t.pa, m.size, value)?;
                }
                let _ = n;
                self.apply_writeback(&m, base);
                Ok(())
            }
            MemKind::Pair => {
                let addr = self.pair_address(&m, base);
                let n = 1u64 << m.size;
                for (i, rt) in [m.rt, m.rt2].into_iter().enumerate() {
                    let ea = addr.wrapping_add(i as u64 * n);
                    let t = mmu::translate(
                        &self.mem,
                        &regs,
                        el,
                        ea,
                        if m.load { Access::Read } else { Access::Write },
                    )?;
                    if m.load {
                        self.xwrite_z(rt, self.load_value(t.pa, &m)?);
                    } else {
                        let value = self.xread_z(rt);
                        self.write_phys(t.pa, m.size, value)?;
                    }
                }
                self.apply_writeback(&m, base);
                Ok(())
            }
            MemKind::Vec | MemKind::VecPair => self.vec_access(&m, base, el, &regs),
            MemKind::Ordered => self.ordered_access(word, el, &regs),
        }
    }

    fn effective_address(&self, m: &Mem, base: u64) -> u64 {
        if let Some((rm, option, amount)) = m.index_reg {
            let idx = self.xread(rm);
            let extended = extend(idx, option, amount);
            base.wrapping_add(extended)
        } else if m.pre {
            base.wrapping_add(m.offset as u64)
        } else {
            base // post-index: the access is at the base
        }
    }

    fn pair_address(&self, m: &Mem, base: u64) -> u64 {
        if m.pre {
            base.wrapping_add(m.offset as u64)
        } else {
            base
        }
    }

    fn apply_writeback(&mut self, m: &Mem, base: u64) {
        if m.writeback {
            self.xwrite(m.rn, base.wrapping_add(m.offset as u64));
        }
    }

    /// Reads a general register where 31 means the zero register.
    fn xread_z(&self, n: u32) -> u64 {
        if n == 31 { 0 } else { self.state.x[n as usize] }
    }
    fn xwrite_z(&mut self, n: u32, value: u64) {
        if n != 31 {
            self.state.x[n as usize] = value;
        }
    }

    fn read_phys(&self, pa: u64, size: u32) -> Result<u64, mmu::Fault> {
        let mut buf = [0u8; 8];
        self.mem
            .read(pa, &mut buf[..1 << size])
            .map_err(|_| external_abort(false))?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Reads a load's value and extends it to the destination register width.
    fn load_value(&self, pa: u64, m: &Mem) -> Result<u64, mmu::Fault> {
        let mut value = self.read_phys(pa, m.size)?;
        if m.signed {
            value = sign_extend(value, 8 << m.size);
        }
        if !m.reg64 {
            value &= 0xffff_ffff;
        }
        Ok(value)
    }

    fn write_phys(&self, pa: u64, size: u32, value: u64) -> Result<(), mmu::Fault> {
        self.mem
            .write(pa, &value.to_le_bytes()[..1 << size])
            .map_err(|_| external_abort(true))
    }

    fn vec_access(
        &mut self,
        _m: &Mem,
        _base: u64,
        _el: u32,
        _regs: &MmuRegs,
    ) -> Result<(), mmu::Fault> {
        // SIMD loads and stores are modelled by the slow helper reading and
        // writing the V spill slots. Filled in once the GPR path is proven.
        Err(external_abort(false))
    }

    fn ordered_access(&mut self, _word: u32, _el: u32, _regs: &MmuRegs) -> Result<(), mmu::Fault> {
        Err(external_abort(false))
    }

    fn take_data_abort(&mut self, fault: mmu::Fault) {
        // ESR for a data abort from the current EL: EC 0x24/0x25. The dispatcher
        // builds the full ESR; here record the fault class and FAR.
        self.state.exception_syndrome = fault.fsc as u64 | ((fault.write as u64) << 6);
        self.state.exception_kind = DATA_ABORT_KIND;
    }
}

const DATA_ABORT_KIND: u64 = 1;

fn external_abort(write: bool) -> mmu::Fault {
    mmu::Fault { fsc: 0x10, write }
}

fn sign_extend(value: u64, bits: u32) -> u64 {
    let shift = 64 - bits;
    (((value << shift) as i64) >> shift) as u64
}

/// Applies an extend-and-shift (the option field of an extended register) to a
/// 64-bit value, as a register-offset address calculation does.
fn extend(value: u64, option: u32, amount: u32) -> u64 {
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
    fn epilogue_symbol_resolves() {
        assert_ne!(CodeCache::epilogue_addr(), 0);
    }
}
