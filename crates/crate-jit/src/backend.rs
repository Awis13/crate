// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The translator proper: the trampoline that enters and leaves a block, the
//! block cache, the per-instruction emission, and the helpers the emitted code
//! calls back into. A dispatcher runs a guest CPU by translating the block at
//! its PC, entering it, and acting on why it returned.

use std::collections::HashMap;
use std::sync::Arc;

use crate_core::board::Board;
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

    /// The host address of the block for `pc`, translating it if needed. With
    /// `debug`, the block holds a single instruction so the dispatcher can
    /// check for breakpoints and steps around each one.
    pub fn block(
        &mut self,
        pc: u64,
        mem: &GuestMemory,
        el: u32,
        el1_regs: &MmuRegs,
        debug: bool,
    ) -> u64 {
        if let Some(&off) = self.blocks.get(&pc) {
            return self.region.exec_base() as u64 + off as u64;
        }
        let limit = if debug { 1 } else { MAX_BLOCK };
        let off = self
            .translate(pc, mem, el, el1_regs, limit)
            .unwrap_or_else(|_| {
                // Out of space: start over. Safe because a block is only entered
                // from the dispatcher, never while another block runs.
                self.reset();
                self.translate(pc, mem, el, el1_regs, limit)
                    .expect("a block in an empty cache")
            });
        self.region.exec_base() as u64 + off as u64
    }

    fn reset(&mut self) {
        self.cursor = 0;
        self.blocks.clear();
        self.emit_save_stub();
    }

    /// Discards every translation after the guest invalidated the instruction
    /// cache (it may re-run code it changed). Coarse; fine-grained invalidation
    /// is milestone 3. Safe only between block runs, where the dispatcher calls
    /// it.
    pub fn flush(&mut self) {
        self.reset();
    }

    fn translate(
        &mut self,
        pc: u64,
        mem: &GuestMemory,
        el: u32,
        el1_regs: &MmuRegs,
        limit: usize,
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
        for _ in 0..limit {
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
            // A literal load is resolved here, where guest memory is reachable.
            if let Op::LiteralLoad {
                rt,
                addr,
                size,
                signed,
                vector,
                prefetch,
            } = op
            {
                if !t.emit_literal(mem, el, el1_regs, rt, addr, size, signed, vector, prefetch) {
                    break; // a faulting or unsupported literal ended the block
                }
                cur = cur.wrapping_add(4);
                continue;
            }
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

    /// Materialises a PC-relative literal load. The literal is read now, from
    /// guest memory through the MMU; self-modifying literals are milestone 3.
    /// Returns false (ending the block with an exit) on a fault or an
    /// unsupported SIMD literal.
    #[allow(clippy::too_many_arguments)]
    fn emit_literal(
        &mut self,
        mem: &GuestMemory,
        el: u32,
        regs: &MmuRegs,
        rt: u32,
        addr: u64,
        size: u32,
        signed: bool,
        vector: bool,
        prefetch: bool,
    ) -> bool {
        if prefetch {
            return true; // a hint; nothing to do
        }
        if vector {
            // A SIMD literal load: raise undefined for now (very rare).
            self.exit(
                Exit::Undefined as u32,
                Next::None,
                ExitInfo {
                    pc: Some(addr),
                    ..Default::default()
                },
            );
            return false;
        }
        let Ok(t) = mmu::translate(mem, regs, el, addr, Access::Read) else {
            self.emit_fetch_fault(addr);
            return false;
        };
        let mut buf = [0u8; 8];
        if mem.read(t.pa, &mut buf[..1 << size]).is_err() {
            self.emit_fetch_fault(addr);
            return false;
        }
        let mut value = u64::from_le_bytes(buf);
        if signed {
            value = sign_extend(value, 8 << size);
        } else if size < 3 {
            value &= (1u64 << (8 << size)) - 1;
        }
        if rt != 31 {
            self.asm.mov_imm(rt, value);
        }
        true
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
            Op::LiteralLoad { .. } => unreachable!("literal loads are resolved in translate"),
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

    #[allow(clippy::too_many_arguments)]
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
        // STR (unsigned offset), not STP: SCRATCH is far past STP's ±512 reach.
        self.asm.emit(str_imm(3, v, SP, offsets::SCRATCH));
        self.asm.emit(str_imm(3, vm, SP, offsets::SCRATCH + 8));
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
        self.asm.emit(ldr_imm(3, v, SP, offsets::SCRATCH));
        self.asm.emit(ldr_imm(3, vm, SP, offsets::SCRATCH + 8));
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
            // A watchpoint has already recorded its own exception kind.
            if cpu.state.exception_kind != WATCHPOINT_KIND {
                cpu.take_data_abort(fault);
            }
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
    /// The board, for routing MMIO and reaching the interrupt controller,
    /// PSCI and the run state. Null for the bare EL0 test runner. Valid for a
    /// `dispatch` call's whole run.
    board: *const Board,
    /// Which CPU this is (its index into the board), for the GIC and PSCI.
    cpu: usize,
    /// The host counter value the guest's virtual counter is measured from.
    timer_offset: u64,
}

impl Cpu {
    pub fn new(mem: Arc<GuestMemory>) -> Box<Self> {
        let mut cpu = Box::new(Cpu {
            state: CpuState::new(),
            mem,
            board: std::ptr::null(),
            cpu: 0,
            timer_offset: 0,
        });
        cpu.state.epilogue = CodeCache::epilogue_addr();
        let ptr: *mut Cpu = &mut *cpu;
        cpu.state.cpu_ptr = ptr as u64;
        cpu
    }

    /// The board, while a `dispatch` call holds it.
    fn board(&self) -> &Board {
        // SAFETY: set non-null for the whole `dispatch` run, which outlives
        // every helper call and block entry.
        unsafe { &*self.board }
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
            let target = cache.block(pc, &mem, el, &regs, false);
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

        // A watchpoint takes precedence over the access, which must not happen.
        if self.debug_active() {
            let addr = match m.kind {
                MemKind::Pair | MemKind::VecPair => self.pair_address(&m, base),
                MemKind::Ordered | MemKind::Atomic => base,
                _ => self.effective_address(&m, base),
            };
            let total = (1u64 << m.size)
                * if matches!(m.kind, MemKind::Pair) {
                    2
                } else {
                    1
                };
            self.watchpoint(addr, total, !m.load)?;
        }

        match m.kind {
            MemKind::Gpr => {
                let addr = self.effective_address(&m, base);
                let n = 1u64 << m.size;
                self.state.exception_far = addr;
                let t = mmu::translate(
                    &self.mem,
                    &regs,
                    el,
                    addr,
                    if m.load { Access::Read } else { Access::Write },
                )?;
                let crosses = crosses_page(addr, n, t.page_shift);
                if m.load {
                    let raw = if crosses {
                        let mut buf = [0u8; 8];
                        self.gather(addr, &mut buf[..n as usize], el, &regs)?;
                        u64::from_le_bytes(buf)
                    } else {
                        self.read_phys(t.pa, m.size)?
                    };
                    self.xwrite_z(m.rt, extend_loaded(&m, raw));
                } else {
                    let value = self.xread_z(m.rt);
                    if crosses {
                        self.scatter(addr, &value.to_le_bytes()[..n as usize], el, &regs)?;
                    } else {
                        self.write_phys(t.pa, m.size, value)?;
                    }
                }
                self.apply_writeback(&m, base);
                Ok(())
            }
            MemKind::Pair => {
                let addr = self.pair_address(&m, base);
                let n = 1u64 << m.size;
                for (i, rt) in [m.rt, m.rt2].into_iter().enumerate() {
                    let ea = addr.wrapping_add(i as u64 * n);
                    self.state.exception_far = ea;
                    let t = mmu::translate(
                        &self.mem,
                        &regs,
                        el,
                        ea,
                        if m.load { Access::Read } else { Access::Write },
                    )?;
                    let crosses = crosses_page(ea, n, t.page_shift);
                    if m.load {
                        let raw = if crosses {
                            let mut buf = [0u8; 8];
                            self.gather(ea, &mut buf[..n as usize], el, &regs)?;
                            u64::from_le_bytes(buf)
                        } else {
                            self.read_phys(t.pa, m.size)?
                        };
                        self.xwrite_z(rt, extend_loaded(&m, raw));
                    } else {
                        let value = self.xread_z(rt);
                        if crosses {
                            self.scatter(ea, &value.to_le_bytes()[..n as usize], el, &regs)?;
                        } else {
                            self.write_phys(t.pa, m.size, value)?;
                        }
                    }
                }
                self.apply_writeback(&m, base);
                Ok(())
            }
            MemKind::Vec | MemKind::VecPair => self.vec_access(&m, base, el, &regs),
            MemKind::Ordered => self.ordered_access(word, el, &regs),
            MemKind::Atomic => self.atomic_access(word, el, &regs),
        }
    }

    /// An LSE atomic (LDADD, LDCLR, LDEOR, LDSET, LDSMAX/MIN, LDUMAX/MIN, SWP)
    /// or LDAPR, as a single-CPU read-modify-write: the only observer is this
    /// CPU, so plain accesses with the host's ordering suffice.
    fn atomic_access(&mut self, word: u32, el: u32, regs: &MmuRegs) -> Result<(), mmu::Fault> {
        let size = (word >> 30) & 3;
        let rs = (word >> 16) & 31;
        let rn = (word >> 5) & 31;
        let rt = word & 31;
        let o3 = (word >> 15) & 1;
        let opc = (word >> 12) & 7;
        let addr = self.xread(rn);
        self.state.exception_far = addr;
        let mask = if size >= 3 {
            u64::MAX
        } else {
            (1u64 << (8 << size)) - 1
        };

        // LDAPR (an acquiring load, no modify) is o3==1, opc==0b100.
        if o3 == 1 && opc == 0b100 {
            let t = mmu::translate(&self.mem, regs, el, addr, Access::Read)?;
            let old = self.read_phys(t.pa, size)? & mask;
            self.xwrite_z(rt, old);
            return Ok(());
        }
        let t = mmu::translate(&self.mem, regs, el, addr, Access::Write)?;
        let old = self.read_phys(t.pa, size)?;
        let rs_val = self.xread_z(rs);
        let new = if o3 == 1 {
            rs_val // SWP
        } else {
            match opc {
                0b000 => old.wrapping_add(rs_val),                // LDADD
                0b001 => old & !rs_val,                           // LDCLR
                0b010 => old ^ rs_val,                            // LDEOR
                0b011 => old | rs_val,                            // LDSET
                0b100 => signed_minmax(old, rs_val, size, true),  // LDSMAX
                0b101 => signed_minmax(old, rs_val, size, false), // LDSMIN
                0b110 => (old & mask).max(rs_val & mask),         // LDUMAX
                0b111 => (old & mask).min(rs_val & mask),         // LDUMIN
                _ => unreachable!(),
            }
        };
        self.write_phys(t.pa, size, new & mask)?;
        // The return register gets the old value; Rt 31 is a store-form with no
        // return (ST<op>).
        self.xwrite_z(rt, old & mask);
        Ok(())
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
        if self.mem.read(pa, &mut buf[..1 << size]).is_ok() {
            return Ok(u64::from_le_bytes(buf));
        }
        // Not RAM: a device, if the board has one here. The bus takes a byte
        // count, not the log2 size the access carries.
        if !self.board.is_null() {
            if let Some(v) = self.board().mmio.read(pa, 1u8 << size) {
                return Ok(v);
            }
        }
        Err(external_abort(false))
    }

    /// Reads `buf.len()` bytes at guest VA `addr`, re-walking the page tables at
    /// every page boundary so a misaligned access that crosses into a
    /// physically separate page reads each page's bytes from its own
    /// translation. The fast single-page path stays in `read_phys`; this is the
    /// rare page-crossing case, where contiguous physical bytes would be wrong.
    fn gather(
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
            let t = mmu::translate(&self.mem, regs, el, va, Access::Read)?;
            let page = 1u64 << t.page_shift;
            let chunk = ((page - (va & (page - 1))) as usize).min(buf.len() - done);
            if self.mem.read(t.pa, &mut buf[done..done + chunk]).is_err() {
                for k in 0..chunk {
                    buf[done + k] = self.read_phys(t.pa + k as u64, 0)? as u8;
                }
            }
            done += chunk;
        }
        Ok(())
    }

    /// Writes `bytes` at guest VA `addr`, re-walking the page tables at every
    /// page boundary; the write counterpart of `gather`.
    fn scatter(
        &mut self,
        addr: u64,
        bytes: &[u8],
        el: u32,
        regs: &MmuRegs,
    ) -> Result<(), mmu::Fault> {
        let mut done = 0usize;
        while done < bytes.len() {
            let va = addr.wrapping_add(done as u64);
            self.state.exception_far = va;
            let t = mmu::translate(&self.mem, regs, el, va, Access::Write)?;
            let page = 1u64 << t.page_shift;
            let chunk = ((page - (va & (page - 1))) as usize).min(bytes.len() - done);
            if self.mem.write(t.pa, &bytes[done..done + chunk]).is_err() {
                for k in 0..chunk {
                    self.write_phys(t.pa + k as u64, 0, bytes[done + k] as u64)?;
                }
            }
            done += chunk;
        }
        Ok(())
    }

    fn write_phys(&self, pa: u64, size: u32, value: u64) -> Result<(), mmu::Fault> {
        if self
            .mem
            .write(pa, &value.to_le_bytes()[..1 << size])
            .is_ok()
        {
            return Ok(());
        }
        if !self.board.is_null() && self.board().mmio.write(pa, 1u8 << size, value) {
            return Ok(());
        }
        Err(external_abort(true))
    }

    /// SIMD loads and stores read and write the V spill slots, which the block
    /// has saved before the call and reloads after.
    fn vec_access(
        &mut self,
        m: &Mem,
        base: u64,
        el: u32,
        regs: &MmuRegs,
    ) -> Result<(), mmu::Fault> {
        let bytes = 1usize << m.size;
        let regs_list: &[u32] = if m.kind == MemKind::VecPair {
            &[m.rt, m.rt2]
        } else {
            &[m.rt]
        };
        let start = if m.kind == MemKind::VecPair {
            self.pair_address(m, base)
        } else {
            self.effective_address(m, base)
        };
        for (i, &rt) in regs_list.iter().enumerate() {
            let ea = start.wrapping_add((i * bytes) as u64);
            if m.load {
                // The upper lanes past `bytes` read as zero (SIMD loads clear
                // the rest of the register).
                let mut reg = [0u8; 16];
                self.gather(ea, &mut reg[..bytes], el, regs)?;
                self.state.v[rt as usize] = reg;
            } else {
                let reg = self.state.v[rt as usize];
                self.scatter(ea, &reg[..bytes], el, regs)?;
            }
        }
        self.apply_writeback(m, base);
        Ok(())
    }

    /// Exclusive and ordered accesses (LDXR/STXR, LDAR/STLR, CAS). On one CPU
    /// the ordering is the host's own; the local monitor (set by a
    /// load-exclusive, cleared by a store-exclusive, a CLREX or an exception)
    /// still matters because an interrupt between the LDXR and the STXR must
    /// make the STXR fail so the guest's retry loop runs.
    fn ordered_access(&mut self, word: u32, el: u32, regs: &MmuRegs) -> Result<(), mmu::Fault> {
        let size = (word >> 30) & 3;
        let o2 = word & (1 << 23) != 0;
        let load = word & (1 << 22) != 0;
        let o1 = word & (1 << 21) != 0;
        let rs = (word >> 16) & 31;
        let rn = (word >> 5) & 31;
        let rt = word & 31;
        let addr = self.xread(rn);
        self.state.exception_far = addr;

        if o1 && o2 {
            // CAS: compare Rs with memory, store Rt if equal, Rs gets the old.
            let t = mmu::translate(&self.mem, regs, el, addr, Access::Write)?;
            let old = self.read_phys(t.pa, size)?;
            let mask = if size >= 3 {
                u64::MAX
            } else {
                (1u64 << (8 << size)) - 1
            };
            if old & mask == self.xread_z(rs) & mask {
                self.write_phys(t.pa, size, self.xread_z(rt))?;
            }
            self.xwrite_z(rs, old & mask);
            return Ok(());
        }
        if o1 {
            let rt2 = (word >> 10) & 31;
            let elem = if word & (1 << 30) != 0 { 3 } else { 2 };
            let n = 1u64 << elem;
            let m = if elem >= 3 {
                u64::MAX
            } else {
                (1u64 << (8 << elem)) - 1
            };
            let acc = if load { Access::Read } else { Access::Write };
            let t0 = mmu::translate(&self.mem, regs, el, addr, acc)?;
            self.state.exception_far = addr.wrapping_add(n);
            let t1 = mmu::translate(&self.mem, regs, el, addr.wrapping_add(n), acc)?;
            if word & (1 << 31) != 0 {
                // LDXP / STXP: an exclusive 32- or 64-bit register pair.
                if load {
                    self.xwrite_z(rt, self.read_phys(t0.pa, elem)? & m);
                    self.xwrite_z(rt2, self.read_phys(t1.pa, elem)? & m);
                    self.state.monitor = addr | 1;
                } else {
                    let reserved = self.state.monitor;
                    self.state.monitor = 0;
                    if reserved == addr | 1 {
                        self.write_phys(t0.pa, elem, self.xread_z(rt))?;
                        self.write_phys(t1.pa, elem, self.xread_z(rt2))?;
                        self.xwrite_z(rs, 0); // success
                    } else {
                        self.xwrite_z(rs, 1); // the reservation was lost
                    }
                }
            } else {
                // CASP: a compare-and-swap of a register pair. Rs is the even
                // register of the compare pair, Rt of the swap pair.
                let t0w = mmu::translate(&self.mem, regs, el, addr, Access::Write)?;
                let t1w = mmu::translate(&self.mem, regs, el, addr.wrapping_add(n), Access::Write)?;
                let (old0, old1) = (self.read_phys(t0w.pa, elem)?, self.read_phys(t1w.pa, elem)?);
                if old0 & m == self.xread_z(rs) & m && old1 & m == self.xread_z(rs | 1) & m {
                    self.write_phys(t0w.pa, elem, self.xread_z(rt))?;
                    self.write_phys(t1w.pa, elem, self.xread_z(rt | 1))?;
                }
                self.xwrite_z(rs, old0 & m);
                self.xwrite_z(rs | 1, old1 & m);
            }
            return Ok(());
        }
        let access = if load { Access::Read } else { Access::Write };
        let t = mmu::translate(&self.mem, regs, el, addr, access)?;
        if load {
            let mut value = self.read_phys(t.pa, size)?;
            if size < 3 {
                value &= (1u64 << (8 << size)) - 1;
            }
            self.xwrite_z(rt, value);
            if !o2 {
                // A load-exclusive (LDXR/LDAXR) reserves; LDAR (o2) does not.
                self.state.monitor = addr | 1;
            }
        } else if !o2 {
            // A store-exclusive (STXR/STLXR) succeeds only while the monitor
            // still holds this address, and always clears it.
            let reserved = self.state.monitor;
            self.state.monitor = 0;
            if reserved == addr | 1 {
                self.write_phys(t.pa, size, self.xread_z(rt))?;
                self.xwrite_z(rs, 0);
            } else {
                self.xwrite_z(rs, 1);
            }
        } else {
            // STLR: a plain store-release.
            self.write_phys(t.pa, size, self.xread_z(rt))?;
        }
        Ok(())
    }

    /// Checks a watchpoint before an access; on a match it records a watchpoint
    /// exception and returns an error so the access is not performed.
    fn watchpoint(&mut self, addr: u64, size: u64, write: bool) -> Result<(), mmu::Fault> {
        if self.watchpoint_at(addr, size, write) {
            self.state.exception_far = addr;
            self.state.exception_kind = WATCHPOINT_KIND;
            self.state.exception_syndrome = (write as u64) << 6;
            return Err(mmu::Fault { fsc: 0x22, write });
        }
        Ok(())
    }

    fn take_data_abort(&mut self, fault: mmu::Fault) {
        if std::env::var_os("CRATE_TRAP_MEM").is_some() {
            let far = self.state.exception_far;
            let pa = mmu::translate(&self.mem, &self.state.mmu, self.el(), far, Access::Read)
                .map(|t| t.pa);
            crate_core::debug!(
                "data abort far={far:#x} fsc={:#x} write={} pc={:#x} -> pa={pa:x?}",
                fault.fsc,
                fault.write,
                self.state.pc
            );
        }
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
const WATCHPOINT_KIND: u64 = 2;

impl Cpu {
    /// Builds a CPU for `board`, starting the boot CPU at its reset state
    /// (EL1h, MMU off, interrupts masked). Only the boot CPU runs under this
    /// milestone; the others need the cross-CPU work of milestone 3.
    fn for_board(board: &Board, cpu: usize) -> Box<Self> {
        let mut c = Cpu::new(board.memory.clone());
        c.board = board as *const Board;
        c.cpu = cpu;
        c.state.pstate = (1 << PSTATE_EL_SHIFT) | PSTATE_SPSEL | PSTATE_DAIF; // EL1h, masked
        c
    }

    fn set_el(&mut self, el: u32) {
        self.state.pstate = (self.state.pstate & !(3 << PSTATE_EL_SHIFT)) | ((el as u64) << 2);
    }

    /// Runs the boot CPU until the machine stops.
    fn dispatch(&mut self, board: &Board, cache: &mut CodeCache) {
        self.timer_offset = crate_host::jit::counter::now();
        let mem = self.mem.clone();
        loop {
            if board.run.is_stopped() {
                return;
            }
            self.service_timer(board);
            if self.take_pending_interrupt(board) {
                continue;
            }
            let pc = self.state.pc;
            if std::env::var_os("CRATE_TRACE_PC").is_some() {
                crate_core::debug!("block pc={pc:#x} el={}", self.el());
            }
            if let Ok(t) = std::env::var("CRATE_TRAP_PC") {
                if u64::from_str_radix(t.trim_start_matches("0x"), 16) == Ok(pc) {
                    let x = &self.state.x;
                    crate_core::debug!(
                        "trap {pc:#x}: x0={:#x} x1={:#x} x2={:#x} x3={:#x} x4={:#x} x5={:#x} x6={:#x} sp={:#x}",
                        x[0],
                        x[1],
                        x[2],
                        x[3],
                        x[4],
                        x[5],
                        x[6],
                        self.state.sp
                    );
                }
            }
            let el = self.el();
            let regs = self.state.mmu;
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
            self.state.dispatch_target = cache.block(pc, &mem, el, &regs, debug);
            // SAFETY: the state is a stable heap box for the whole dispatch run.
            let code = unsafe { crate_jit_enter(&mut *self.state) } as u32;
            // One instruction ran: leave active-pending unless it faulted.
            if stepping_now
                && !matches!(
                    Exit::from_u32(code),
                    Some(Exit::Exception) | Some(Exit::Undefined)
                )
            {
                self.state.pstate &= !PSTATE_SS;
            }
            match Exit::from_u32(code) {
                Some(Exit::Branch) => {
                    self.state.pc = self.state.next_pc;
                }
                Some(Exit::WaitForInterrupt) => {
                    self.state.pc = self.state.next_pc;
                    self.wait_for_interrupt(board);
                }
                Some(Exit::WaitForEvent) => {
                    self.state.pc = self.state.next_pc;
                    std::thread::yield_now();
                }
                Some(Exit::Sysreg) => {
                    self.handle_sysreg(board);
                    if self.state.flush_cache != 0 {
                        self.state.flush_cache = 0;
                        cache.flush();
                    }
                }
                Some(Exit::Call) => self.handle_call(board),
                Some(Exit::Exception) => self.handle_exception_exit(),
                Some(Exit::Undefined) => {
                    let pc = self.state.pc;
                    self.take_exception(EC_UNKNOWN, 0, None, pc);
                }
                None => {
                    board
                        .run
                        .stop(StopReason::Error(format!("unknown exit {code}")));
                    return;
                }
            }
        }
    }

    // ---- interrupts ----

    fn take_pending_interrupt(&mut self, board: &Board) -> bool {
        let pstate = self.state.pstate;
        if pstate & PSTATE_I == 0 && board.interrupts.irq_pending(self.cpu) {
            let pc = self.state.pc;
            self.take_exception_kind(ExcKind::Irq, 0, None, pc);
            return true;
        }
        if pstate & PSTATE_F == 0 && board.interrupts.fiq_pending(self.cpu) {
            let pc = self.state.pc;
            self.take_exception_kind(ExcKind::Fiq, 0, None, pc);
            return true;
        }
        false
    }

    fn wait_for_interrupt(&mut self, board: &Board) {
        // Sleep until an interrupt could be taken or the timer would fire.
        loop {
            if board.run.is_stopped() {
                return;
            }
            self.service_timer(board);
            if board.interrupts.irq_pending(self.cpu) || board.interrupts.fiq_pending(self.cpu) {
                return;
            }
            match self.timer_deadline() {
                Some(d) => {
                    let now = crate_host::jit::counter::now();
                    if now >= d {
                        continue;
                    }
                    let ticks = d - now;
                    let nanos = ticks as u128 * 1_000_000_000
                        / crate_host::jit::counter::frequency() as u128;
                    std::thread::sleep(std::time::Duration::from_nanos(
                        (nanos as u64).clamp(1, 1_000_000),
                    ));
                }
                None => std::thread::sleep(std::time::Duration::from_millis(1)),
            }
        }
    }

    // ---- calls (SVC / HVC / SMC / BRK) ----

    fn handle_call(&mut self, board: &Board) {
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
                    Effect::Stop => {}
                    _ => self.state.pc = self.state.next_pc,
                }
            }
            _ => self.take_exception(EC_BRK, imm as u64, None, pc), // BRK
        }
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
            // IFSC 0x10 (external) for a fetch that missed RAM, else the walk's.
            self.take_exception(ec, 0x10, Some(pc), pc);
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
        self.take_exception(ec, iss, Some(far), pc);
    }

    // ---- exceptions ----

    fn take_exception(&mut self, ec: u64, iss: u64, far: Option<u64>, elr: u64) {
        self.state.banked.esr_el1 = (ec << 26) | (1 << 25) | (iss & 0x1ff_ffff);
        self.take_exception_kind(ExcKind::Sync, iss, far, elr);
    }

    fn take_exception_kind(&mut self, kind: ExcKind, _iss: u64, far: Option<u64>, elr: u64) {
        let from_el = self.el();
        // Taking an exception clears the local exclusive monitor.
        self.state.monitor = 0;
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
        self.state.nzcv = spsr & 0xf000_0000;
        self.state.pstate = (spsr & (0xf << 6)) // DAIF
            | (spsr & PSTATE_SS) // software-step
            | (((spsr >> 2) & 3) << PSTATE_EL_SHIFT)
            | (spsr & PSTATE_SPSEL);
        self.state.pc = self.state.banked.elr_el1;
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

    // ---- the generic timer ----

    fn guest_counter(&self) -> u64 {
        crate_host::jit::counter::now().wrapping_sub(self.timer_offset)
    }

    /// When either architected timer would next fire, in host counter ticks,
    /// whichever is sooner, if either is armed and unmasked.
    fn timer_deadline(&self) -> Option<u64> {
        let mut deadline: Option<u64> = None;
        for (ctl, cval) in [
            (self.state.banked.cntv_ctl, self.state.banked.cntv_cval),
            (self.state.banked.cntp_ctl, self.state.banked.cntp_cval),
        ] {
            if ctl & 1 == 0 || ctl & 2 != 0 {
                continue; // disabled or masked
            }
            let d = self.timer_offset.wrapping_add(cval);
            deadline = Some(deadline.map_or(d, |x| x.min(d)));
        }
        deadline
    }

    /// Raises or lowers each architected timer's interrupt from its compare
    /// value. CNTVCT and CNTPCT share the one guest counter (CNTVOFF is 0 for
    /// an EL1 guest), so both compare against it.
    fn service_timer(&mut self, board: &Board) {
        let now = self.guest_counter();
        // ISTATUS (bit 2) reads as the raw condition; the line is gated by IMASK.
        let vctl = self.state.banked.cntv_ctl;
        let vfired = vctl & 1 != 0 && now >= self.state.banked.cntv_cval;
        self.state.banked.cntv_ctl = (vctl & !(1 << 2)) | ((vfired as u64) << 2);
        board
            .lines
            .set_ppi(self.cpu, board.vtimer_ppi, vfired && vctl & 2 == 0);

        let pctl = self.state.banked.cntp_ctl;
        let pfired = pctl & 1 != 0 && now >= self.state.banked.cntp_cval;
        self.state.banked.cntp_ctl = (pctl & !(1 << 2)) | ((pfired as u64) << 2);
        board
            .lines
            .set_ppi(self.cpu, board.ptimer_ppi, pfired && pctl & 2 == 0);
    }
}

#[derive(Clone, Copy)]
enum ExcKind {
    Sync,
    Irq,
    Fiq,
}

/// The CPU backend: runs a board's boot CPU on the translator.
pub struct Jit {
    jit_bytes: usize,
}

impl Jit {
    pub fn new() -> Self {
        Jit {
            jit_bytes: 64 << 20,
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
        let mut cpu = Cpu::for_board(board, 0);
        cpu.state.pc = board.boot.pc;
        cpu.state.x[0] = board.boot.x[0];
        cpu.state.x[1] = board.boot.x[1];
        cpu.state.x[2] = board.boot.x[2];
        cpu.state.x[3] = board.boot.x[3];
        let mut cache = CodeCache::new(self.jit_bytes)?;
        // The interrupt controller wakes a sleeping CPU through this; with one
        // CPU the dispatcher already re-checks, so a no-op waker suffices.
        board.interrupts.set_waker(Arc::new(NoopWaker));
        cpu.dispatch(board, &mut cache);
        Ok(board.run.reason().unwrap_or(StopReason::Killed))
    }
}

struct NoopWaker;
impl crate_core::cpu::CpuWaker for NoopWaker {
    fn wake(&self, _cpu: usize) {}
}

/// System-register reads and writes the translator traps.
impl Cpu {
    fn handle_sysreg(&mut self, board: &Board) {
        let reg = SysReg(self.state.sysreg as u16);
        let rt = self.state.sysreg_rt;
        let read = self.state.sysreg_read != 0;
        let next = self.state.next_pc;

        if self.state.sysreg as u16 == SYSTEM_WORD_MARKER {
            // An MSR-immediate or a maintenance op, the whole word in syndrome.
            let word = self.state.exception_syndrome as u32;
            self.handle_system_word(word);
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

    fn handle_system_word(&mut self, word: u32) {
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
        let crn = (word >> 12) & 0xf;
        let crm = (word >> 8) & 0xf;
        if crn == 0b0011 {
            // CLREX: clear the exclusive monitor.
            self.state.monitor = 0;
        } else if crn == 0b1000 {
            // A TLBI: drop the software TLBs the fast path would keep (none yet).
            self.state.clear_tlbs();
        } else if crn == 0b0111 {
            let op2 = (word >> 5) & 7;
            match (crm, op2) {
                // AT (address translation): translate Rt's VA, set PAR_EL1.
                (0b1000, _) => self.address_translate(word),
                // IC IALLU(IS) / IC IVAU (CRm 1 or 5): the guest changed code it
                // may re-run, so drop every translation. The cache-maintenance
                // DC ops are no-ops since the helper reads memory afresh.
                (0b0001, _) | (0b0101, _) => self.state.flush_cache = 1,
                // DC ZVA: zero the block at Rt's VA. This one is not a cache
                // hint -- it writes zeros, and we advertise it in DCZID_EL0.
                (0b0100, 1) => self.dc_zva(word),
                _ => {}
            }
        }
    }

    /// DC ZVA: zero the DCZID_EL0-sized block (64 bytes) holding Rt's VA. The
    /// guest zeroes memory this way (clear_page, kzalloc), so a recycled page
    /// it clears must actually read back zero -- otherwise a freed structure
    /// keeps stale bytes and, e.g., the slab freelist is corrupted. The block
    /// fits within one page, so one translation covers it; a fault is dropped
    /// (the caller advances past the instruction) as the guest only zeroes
    /// mapped, writable memory.
    fn dc_zva(&mut self, word: u32) {
        let rt = (word & 31) as usize;
        let va = if rt == 31 { 0 } else { self.state.x[rt] };
        let base = va & !63;
        let el = self.el();
        let regs = self.state.mmu;
        if let Ok(t) = mmu::translate(&self.mem, &regs, el, base, Access::Write) {
            for off in (0..64u64).step_by(8) {
                let _ = self.write_phys(t.pa + off, 3, 0);
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
        }
    }

    /// Reads a system register, `None` if it is not one the machine implements
    /// (which the caller turns into an undefined-instruction exception).
    fn read_sysreg(&mut self, board: &Board, reg: SysReg) -> Option<u64> {
        // The GIC's CPU interface owns the ICC_* registers.
        if let Some(v) = board.interrupts.read_sysreg(self.cpu, reg) {
            return Some(v);
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
            r if r == sr::CNTV_CTL_EL0 => self.state.banked.cntv_ctl,
            r if r == sr::CNTV_CVAL_EL0 => self.state.banked.cntv_cval,
            r if r == sr::CNTV_TVAL_EL0 => {
                (self
                    .state
                    .banked
                    .cntv_cval
                    .wrapping_sub(self.guest_counter()))
                    & 0xffff_ffff
            }
            r if r == sr::CNTP_CTL_EL0 => self.state.banked.cntp_ctl,
            r if r == sr::CNTP_CVAL_EL0 => self.state.banked.cntp_cval,
            r if r == sr::CNTP_TVAL_EL0 => {
                (self
                    .state
                    .banked
                    .cntp_cval
                    .wrapping_sub(self.guest_counter()))
                    & 0xffff_ffff
            }
            r if r == sr::NZCV => self.state.nzcv,
            r if r == sr::DAIF => self.state.pstate & (0xf << 6),
            r if r == sr::FPCR => self.state.fpcr,
            r if r == sr::FPSR => self.state.fpsr,
            r if r == sr::CURRENT_EL => (self.el() as u64) << 2,
            r if r == sr::SPSEL => self.state.pstate & PSTATE_SPSEL,
            r if r == sr::MPIDR_EL1 => board.mpidrs[self.cpu],
            r if r == sr::MIDR_EL1 => 0x410f_d080, // a generic Cortex-A
            r if r == sr::CTR_EL0 => CTR_EL0,
            r if r == sr::DCZID_EL0 => 0x4, // 64-byte zero block, DC ZVA allowed
            r if r == sr::CLIDR_EL1 => 0,   // no cache levels advertised
            r if r == sr::ID_AA64MMFR0_EL1 => ID_AA64MMFR0,
            r if r == sr::ID_AA64PFR0_EL1 => ID_AA64PFR0,
            r if r == sr::ID_AA64DFR0_EL1 => ID_AA64DFR0,
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
            return true;
        }
        match reg {
            r if r == sr::SCTLR_EL1 => self.state.mmu.sctlr_el1 = value,
            r if r == sr::TTBR0_EL1 => self.state.mmu.ttbr0_el1 = value,
            r if r == sr::TTBR1_EL1 => self.state.mmu.ttbr1_el1 = value,
            r if r == sr::TCR_EL1 => self.state.mmu.tcr_el1 = value,
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
            r if r == sr::MDSCR_EL1 => self.state.banked.mdscr_el1 = value,
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
            r if r == sr::CNTP_CTL_EL0 => self.state.banked.cntp_ctl = value & 0x3,
            r if r == sr::CNTP_CVAL_EL0 => self.state.banked.cntp_cval = value,
            r if r == sr::CNTP_TVAL_EL0 => {
                self.state.banked.cntp_cval = self
                    .guest_counter()
                    .wrapping_add(value as i32 as i64 as u64)
            }
            r if r == sr::NZCV => self.state.nzcv = value & 0xf000_0000,
            r if r == sr::DAIF => {
                self.state.pstate = (self.state.pstate & !(0xf << 6)) | (value & (0xf << 6))
            }
            r if r == sr::FPCR => self.state.fpcr = value,
            r if r == sr::FPSR => self.state.fpsr = value,
            r if r == sr::SPSEL => {
                self.state.pstate = (self.state.pstate & !PSTATE_SPSEL) | (value & PSTATE_SPSEL)
            }
            r if r == sr::MDSCR_EL1 => {
                self.state.banked.mdscr_el1 = value;
                // MDSCR.SS / MDE change whether blocks must step; re-translate.
                self.state.flush_cache = 1;
            }
            r if r == sr::OSLAR_EL1 => self.state.debug.oslock = (value & 1) << 1,
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

    /// Whether any self-hosted debug exception is armed, so blocks must be
    /// translated one instruction at a time and the dispatcher must check for
    /// breakpoints and steps. Recomputed from the debug registers each time it
    /// is needed; it is cheap.
    fn debug_active(&self) -> bool {
        let mdscr = self.state.banked.mdscr_el1;
        let breakpoints = mdscr & (1 << 15) != 0 // MDE
            && self.state.debug.bcr.iter().any(|&c| c & 1 != 0);
        let watchpoints = self.state.debug.wcr.iter().any(|&c| c & 1 != 0);
        let step = mdscr & 1 != 0; // SS
        (breakpoints || watchpoints || step) && self.state.debug.oslock & 2 == 0
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

/// CTR_EL0: 16-byte I- and D-cache lines, IDC and DIC set so no cache
/// maintenance is needed for the data path (the translator reads memory afresh).
/// DIC is left 0 so the guest issues IC after writing code, which the dispatcher
/// honours by dropping its translations; without that, rewritten code would run
/// the stale translation.
const CTR_EL0: u64 = (1 << 28) | (2 << 16) | 2 | (3 << 24);
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
    pub const CSSELR_EL1: SysReg = SysReg::new(3, 2, 0, 0, 0);
    pub const ID_AA64PFR0_EL1: SysReg = SysReg::new(3, 0, 0, 4, 0);
    pub const ID_AA64DFR0_EL1: SysReg = SysReg::new(3, 0, 0, 5, 0);
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

fn sign_extend(value: u64, bits: u32) -> u64 {
    let shift = 64 - bits;
    (((value << shift) as i64) >> shift) as u64
}

/// Whether an `n`-byte access at `addr` runs past the end of its page, so it
/// must be split and each page translated on its own: consecutive guest pages
/// need not map to consecutive physical pages.
fn crosses_page(addr: u64, n: u64, page_shift: u32) -> bool {
    let page = 1u64 << page_shift;
    (addr & (page - 1)) + n > page
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
        cpu.memory_access(0xf940_0025).unwrap();
        assert_eq!(cpu.state.x[5], 0x8877_6655_4433_2211);
    }

    #[test]
    fn misaligned_store_crossing_a_page_writes_both_frames() {
        let (mut cpu, va, pa_a, pa_b) = two_page_cpu();
        cpu.state.x[1] = va + 0xffc;
        cpu.state.x[7] = 0xaabb_ccdd_eeff_0011;
        // STR x7, [x1]
        cpu.memory_access(0xf900_0027).unwrap();
        let (mut a, mut b) = ([0u8; 4], [0u8; 4]);
        cpu.mem.read(pa_a + 0xffc, &mut a).unwrap();
        cpu.mem.read(pa_b, &mut b).unwrap();
        assert_eq!(a, [0x11, 0x00, 0xff, 0xee]);
        assert_eq!(b, [0xdd, 0xcc, 0xbb, 0xaa]);
    }
}
