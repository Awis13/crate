// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Translating one block: guest instructions from one 4 KiB page, starting at
//! a guest PC, into host code. Most instructions are copied; the rest are
//! rewritten here.
//!
//! **Loads and stores** take an inline fast path: the guest address is
//! computed into a borrowed host register, its page looked up in the software
//! TLB for this exception level and kind of access, and on a hit the host
//! performs the same access at the host address. A miss, a device, an access
//! that crosses a page or one that is misaligned where the architecture
//! requires alignment, goes out of line to the shared slow path, which calls
//! into Rust; the Rust side performs the access and fills the TLB entry.
//!
//! **Branches** to the same page end in a branch the dispatcher patches to
//! jump straight to the target block once it exists (chaining). Branches to
//! another page, and indirect branches, look the target up in the jump cache.
//! Backward chained branches and every jump-cache lookup test the CPU's exit
//! request first, so no loop of translated code can keep the CPU from its
//! dispatcher.
//!
//! A block never crosses a 4 KiB page, so a chained branch never leaves the
//! page its block came from: invalidating a page's code throws away every
//! block that could reach another through a chain.
//!
//! With the fast paths off (`Ctx::fast` false), every load and store calls the
//! slow path and every branch returns to the dispatcher: the slow, simple
//! translation, kept to tell a fast-path bug from any other.

use crate::asm::*;
use crate::decode::{self, CallKind, CondKind, Mem, MemKind, Op};
use crate::emit::{self, ExitInfo, JUMP_REG, Next, Stubs};
use std::mem::offset_of;

use crate::state::{Banked, Exit, MmuRegs, TLB_BITS, TLB_SHIFT, TlbKind, offsets, tlb_index};

/// What every block of one translation context shares.
#[derive(Clone, Copy, Debug)]
pub struct Ctx {
    /// The exception level the block runs at: it picks the TLBs and the jump
    /// cache, and the system registers that may be inlined.
    pub el: u32,
    /// Fast paths on: the inline TLB, chaining, the jump cache, inline system
    /// registers.
    pub fast: bool,
    /// Self-hosted debug is armed: one instruction per block, every load and
    /// store through the slow path (which checks watchpoints), no chaining.
    pub debug: bool,
    pub stubs: Stubs,
    /// The CPU's MPIDR_EL1, which its code reads as a constant.
    pub mpidr: u64,
}

impl Ctx {
    fn fast_paths(&self) -> bool {
        self.fast && !self.debug
    }
}

/// Markers packed into an exit's syndrome field the dispatcher recognises.
pub const FETCH_ABORT: u64 = 1 << 40;
pub const ERET_MARKER: u64 = 1 << 41;
/// A Sysreg exit whose encoding is this marker carries a whole instruction
/// word in the syndrome, for MSR-immediate and maintenance ops.
pub const SYSTEM_WORD_MARKER: u16 = 0xffff;

pub fn call_kind_code(kind: CallKind) -> u32 {
    match kind {
        CallKind::Svc => 0,
        CallKind::Hvc => 1,
        CallKind::Smc => 2,
        CallKind::Brk => 3,
    }
}

/// A load or store's out-of-line call into the slow path.
struct SlowSite {
    label: Label,
    resume: Label,
    word: u32,
    pc: u64,
    victims: [Reg; 4],
    regs: Regs,
}

/// A short list of registers, without allocating.
#[derive(Clone, Copy, Default)]
struct Regs {
    r: [u32; 8],
    n: usize,
}

impl Regs {
    /// Adds a register; the zero register (31) and repeats are dropped.
    fn push(&mut self, r: u32) {
        if r != 31 && !self.as_slice().contains(&r) && self.n < 8 {
            self.r[self.n] = r;
            self.n += 1;
        }
    }

    fn as_slice(&self) -> &[u32] {
        &self.r[..self.n]
    }

    /// The ones among X19..X29, which the slow path's call preserves and so
    /// a call site stores and reloads itself.
    fn callee_saved(&self) -> Regs {
        let mut out = Regs::default();
        for &r in self.as_slice() {
            if (19..=29).contains(&r) {
                out.push(r);
            }
        }
        out
    }
}

/// One block being built.
pub struct Block {
    pub asm: Asm,
    ctx: Ctx,
    /// The guest page (`pc >> 12`) the block's code is in.
    page: u64,
    terminated: bool,
    /// The block ended because an instruction fetch faulted, so it must not be
    /// cached: once the guest handles the abort (demand-paging the code) and
    /// retries, the dispatcher has to re-translate the now-mapped instruction
    /// rather than re-run this abort forever.
    pub fetch_faulted: bool,
    slow: Vec<SlowSite>,
}

/// The host registers a fast path may borrow, best first: the intra-procedure
/// scratch registers and the temporaries are the least likely to be operands.
const VICTIMS: [Reg; 29] = [
    16, 17, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29,
];

fn victims(avoid: &[u32]) -> [Reg; 4] {
    let mut out = [0; 4];
    let mut n = 0;
    for &r in VICTIMS.iter() {
        if !avoid.contains(&r) {
            out[n] = r;
            n += 1;
            if n == 4 {
                break;
            }
        }
    }
    out
}

/// Translates guest code at `pc`, whose words up to the end of its 4 KiB page
/// are `words`, into a block of host code. `limit` caps the instructions.
pub fn translate(ctx: Ctx, pc: u64, words: &[u32], limit: usize) -> Block {
    let mut t = Block {
        asm: Asm::new(),
        ctx,
        page: pc >> 12,
        terminated: false,
        fetch_faulted: false,
        slow: Vec::new(),
    };
    emit::block_entry(&mut t.asm);
    let mut cur = pc;
    for &word in words.iter().take(limit.max(1)) {
        let op = decode::decode(word, cur);
        if t.emit(op, word, cur) {
            break;
        }
        cur = cur.wrapping_add(4);
    }
    if !t.terminated {
        t.goto_const(cur, None, cur.wrapping_sub(4));
    }
    t.emit_slow_sites();
    t
}

/// A block that only raises the instruction abort for a fetch at `pc` that
/// faulted with status `fsc`.
pub fn fetch_fault(ctx: Ctx, pc: u64, fsc: u8) -> Block {
    let mut t = Block {
        asm: Asm::new(),
        ctx,
        page: pc >> 12,
        terminated: false,
        fetch_faulted: true,
        slow: Vec::new(),
    };
    emit::block_entry(&mut t.asm);
    t.exit(
        Exit::Exception as u32,
        Next::None,
        ExitInfo {
            pc: Some(pc),
            syndrome: Some(FETCH_ABORT | (fsc as u64 & 0x3f)),
            ..Default::default()
        },
    );
    t
}

impl Block {
    /// Emits an exit to the save stub.
    fn exit(&mut self, code: u32, next: Next, info: ExitInfo) {
        emit::exit(
            &mut self.asm,
            self.ctx.stubs.save,
            self.page << 12,
            code,
            next,
            info,
        );
        self.terminated = true;
    }

    /// Emits one instruction. Returns true if it ended the block.
    fn emit(&mut self, op: Op, word: u32, pc: u64) -> bool {
        match op {
            Op::Copy => {
                self.emit_copy(word);
                false
            }
            Op::Nop => false,
            Op::PcRel { rd, imm, .. } => {
                self.set_reg_va(rd, imm as u64);
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
            Op::LogicalSpImm { word } => {
                self.emit_logical_sp(word);
                false
            }
            Op::Memory(m) => {
                self.emit_memory(m, None, pc);
                false
            }
            Op::LiteralLoad {
                rt,
                addr,
                size,
                signed,
                vector,
            } => {
                self.emit_literal(word, rt, addr, size, signed, vector, pc);
                false
            }
            Op::DcZva { rt } => {
                self.emit_dc_zva(word, rt, pc);
                false
            }
            Op::Branch { link, target } => {
                self.goto_const(target, link.then_some(pc.wrapping_add(4)), pc);
                true
            }
            Op::CondBranch { word, kind, taken } => {
                self.emit_cond_branch(word, kind, taken, pc);
                true
            }
            Op::IndBranch { link, rn, .. } => {
                self.goto_reg(rn, link.then_some(pc.wrapping_add(4)));
                true
            }
            Op::Sysreg { read, reg, rt } => {
                if self.ctx.fast_paths() && self.emit_inline_sysreg(read, reg, rt, word, pc) {
                    return self.terminated;
                }
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
            Op::MsrImm { op1, op2, crm } => {
                if self.ctx.fast_paths()
                    && self.ctx.el == 1
                    && op1 == 0b011
                    && (op2 == 0b110 || op2 == 0b111)
                {
                    self.emit_daif_imm(op2 == 0b110, crm, pc);
                    return self.terminated;
                }
                self.system_exit(word, pc);
                true
            }
            Op::System { .. } => {
                self.system_exit(word, pc);
                true
            }
            Op::Isb => {
                if self.ctx.fast_paths() {
                    self.exit_if_requested(pc.wrapping_add(4));
                }
                false
            }
            Op::Clrex => {
                self.asm.emit(str_imm(3, XZR, SP, offsets::MONITOR));
                false
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

    fn system_exit(&mut self, word: u32, pc: u64) {
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
    }

    // ------------------------------------------------------------ registers

    /// A victim host register to borrow for a short sequence, not equal to
    /// any in `avoid`.
    fn victim(avoid: &[u32]) -> u32 {
        (9..=15).find(|r| !avoid.contains(r)).unwrap()
    }

    /// Rd = a guest virtual address in or near this block's page (an ADR or
    /// ADRP result), computed from the run-time page; Rd may be virtual X18
    /// or the zero register.
    fn set_reg_va(&mut self, rd: u32, va: u64) {
        let page = self.page << 12;
        let far =
            va.wrapping_sub(page) as i64 >= 1 << 24 || (va.wrapping_sub(page) as i64) <= -(1 << 24);
        match rd {
            31 => {}
            18 => {
                let v = Self::victim(&[]);
                let t = Self::victim(&[v]);
                self.asm.emit(stp(v, t, SP, offsets::scratch(0) as i32));
                emit::guest_va(&mut self.asm, v, va, page, t);
                self.asm.emit(str_imm(3, v, SP, offsets::x(18)));
                self.asm.emit(ldp(v, t, SP, offsets::scratch(0) as i32));
            }
            r if far => {
                let t = Self::victim(&[r]);
                self.asm.emit(str_imm(3, t, SP, offsets::scratch(0)));
                emit::guest_va(&mut self.asm, r, va, page, t);
                self.asm.emit(ldr_imm(3, t, SP, offsets::scratch(0)));
            }
            r => emit::guest_va(&mut self.asm, r, va, page, 31),
        }
    }

    /// Rd = a constant, where Rd may be virtual X18 (or the zero register).
    fn set_reg_const(&mut self, rd: u32, value: u64) {
        if rd == 18 {
            let v = Self::victim(&[]);
            self.asm.emit(str_imm(3, v, SP, offsets::SCRATCH));
            self.asm.mov_imm(v, value);
            self.asm.emit(str_imm(3, v, SP, offsets::x(18)));
            self.asm.emit(ldr_imm(3, v, SP, offsets::SCRATCH));
        } else if rd != 31 {
            self.asm.mov_imm(rd, value);
        }
    }

    /// Copies a word, rewriting it to run against a borrowed register when it
    /// names guest X18 in a general-register field.
    fn emit_copy(&mut self, word: u32) {
        let fields = gpr_field_positions(word);
        if fields.iter().any(|&p| (word >> p) & 0x1f == 18) {
            self.emit_x18_copy(word, fields);
        } else {
            self.asm.emit(word);
        }
    }

    /// Emits a verbatim instruction that names guest X18 in one or more GPR
    /// fields. X18 is the platform register: the host OS reserves it and
    /// clobbers it (macOS zeroes it on return from preemption), so guest X18
    /// cannot live in host x18 -- it lives in `state.x[18]`, like SP. The
    /// instruction runs against a victim register loaded from, and stored back
    /// to, that slot. `cands` are the GPR field bit-positions for this word;
    /// a field at bit 0 is the destination of every form that reaches here
    /// (or a source that the store back leaves unchanged).
    fn emit_x18_copy(&mut self, word: u32, cands: &[u32]) {
        let mut avoid = [0u32; 4];
        for (i, &p) in cands.iter().enumerate() {
            avoid[i] = (word >> p) & 0x1f;
        }
        let v = Self::victim(&avoid[..cands.len()]);
        self.asm.emit(str_imm(3, v, SP, offsets::SCRATCH));
        self.asm.emit(ldr_imm(3, v, SP, offsets::x(18)));
        let mut w = word;
        let mut writes_x18 = false;
        for &p in cands {
            if (word >> p) & 0x1f == 18 {
                w = (w & !(0x1f << p)) | (v << p);
                writes_x18 |= p == 0;
            }
        }
        self.asm.emit(w);
        if writes_x18 {
            self.asm.emit(str_imm(3, v, SP, offsets::x(18)));
        }
        self.asm.emit(ldr_imm(3, v, SP, offsets::SCRATCH));
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
        // Load the source (SP and X18 both live in the state block, not a host
        // register).
        if rn == 31 {
            self.asm.emit(ldr_imm(3, v, SP, offsets::SP));
        } else if rn == 18 {
            self.asm.emit(ldr_imm(3, v, SP, offsets::x(18)));
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
        if !discard {
            if rd == 31 {
                self.asm.emit(str_imm(3, v, SP, offsets::SP));
            } else if rd == 18 {
                self.asm.emit(str_imm(3, v, SP, offsets::x(18)));
            } else {
                self.asm.emit(mov(rd, v));
            }
        }
        self.asm.emit(ldr_imm(3, v, SP, offsets::SCRATCH));
    }

    /// AND, ORR or EOR (immediate) into the guest's SP: the same instruction
    /// into a borrowed register, stored to the state's SP. A 32-bit form's
    /// result is zero-extended, as a write of WSP is.
    fn emit_logical_sp(&mut self, word: u32) {
        let rn = (word >> 5) & 31;
        let v = Self::victim(&[rn]);
        self.asm.emit(str_imm(3, v, SP, offsets::SCRATCH));
        let mut w = (word & !31) | v;
        if rn == 18 {
            self.asm.emit(ldr_imm(3, v, SP, offsets::x(18)));
            w = (w & !(31 << 5)) | (v << 5);
        }
        self.asm.emit(w);
        self.asm.emit(str_imm(3, v, SP, offsets::SP));
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
        self.asm.emit(str_imm(3, v, SP, offsets::SCRATCH));
        self.asm.emit(str_imm(3, vm, SP, offsets::SCRATCH + 8));
        if rn == 31 {
            self.asm.emit(ldr_imm(3, v, SP, offsets::SP));
        } else if rn == 18 {
            self.asm.emit(ldr_imm(3, v, SP, offsets::x(18)));
        } else {
            self.asm.emit(mov(v, rn));
        }
        // rm is never SP (register 31 there is the zero register), but may be X18.
        if rm == 31 {
            self.asm.emit(movz(true, vm, 0, 0));
        } else if rm == 18 {
            self.asm.emit(ldr_imm(3, vm, SP, offsets::x(18)));
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
            } else if rd == 18 {
                self.asm.emit(str_imm(3, v, SP, offsets::x(18)));
            } else {
                self.asm.emit(mov(rd, v));
            }
        }
        self.asm.emit(ldr_imm(3, v, SP, offsets::SCRATCH));
        self.asm.emit(ldr_imm(3, vm, SP, offsets::SCRATCH + 8));
    }

    // ------------------------------------------------------------ memory

    #[allow(clippy::too_many_arguments)]
    fn emit_literal(
        &mut self,
        word: u32,
        rt: u32,
        addr: u64,
        size: u32,
        signed: bool,
        vector: bool,
        pc: u64,
    ) {
        // A load from a constant address: the general path, with the address
        // materialised instead of computed from a base register. The slow
        // path decodes the literal form from the word itself.
        let (size_field, opc) = match (vector, size, signed) {
            (false, 2, false) => (2, 0b01),
            (false, 3, _) => (3, 0b01),
            (false, _, true) => (2, 0b10),
            (true, 2, _) => (2, 0b01),
            (true, 3, _) => (3, 0b01),
            _ => (0, 0b11), // Q
        };
        let host_word = size_field << 30 | (vector as u32) << 26 | opc << 22 | rt;
        let m = Mem {
            word: host_word,
            kind: if vector { MemKind::Vec } else { MemKind::Gpr },
            size,
            signed,
            reg64: size == 3 || signed,
            load: true,
            rn: 31,
            rt,
            rt2: 31,
            rs: 31,
            offset: 0,
            writeback: false,
            pre: true,
            index_reg: None,
            wb_reg: None,
            total: 1 << size,
            align: 1,
            acquire: false,
            release: false,
            unprivileged: false,
        };
        self.emit_memory_with(m, Some(addr), word, pc);
    }

    fn emit_memory(&mut self, m: Mem, literal: Option<u64>, pc: u64) {
        self.emit_memory_with(m, literal, m.word, pc);
    }

    /// The general registers an access names (each role that is a general
    /// register; register 31 dropped).
    fn gpr_operands(m: &Mem) -> Regs {
        let mut v = Regs::default();
        v.push(m.rn);
        if let Some((rm, _, _)) = m.index_reg {
            v.push(rm);
        }
        if let Some(rm) = m.wb_reg {
            v.push(rm);
        }
        let gpr_data = !matches!(m.kind, MemKind::Vec | MemKind::VecPair | MemKind::VecStruct);
        if gpr_data {
            v.push(m.rt);
            v.push(m.rt2);
            if m.kind == MemKind::CasPair {
                v.push(m.rt + 1);
            }
        }
        v.push(m.rs);
        if m.kind == MemKind::CasPair {
            v.push(m.rs + 1);
        }
        v
    }

    /// Whether this access always takes the slow path: the exclusive pairs,
    /// and the ordered and atomic forms whose data registers include X18 (the
    /// fast path rewrites X18 only for plain loads and stores).
    fn needs_slow(m: &Mem) -> bool {
        match m.kind {
            MemKind::Gpr
            | MemKind::Pair
            | MemKind::Vec
            | MemKind::VecPair
            | MemKind::VecStruct
            | MemKind::LoadAcquire
            | MemKind::StoreRelease => false,
            MemKind::LoadExclusive | MemKind::StoreExclusive if m.rt2 != 31 => true,
            MemKind::CasPair => [m.rt, m.rs, m.rt + 1, m.rs + 1].contains(&18),
            _ => m.rt == 18 || m.rs == 18,
        }
    }

    fn emit_memory_with(&mut self, m: Mem, literal: Option<u64>, word: u32, pc: u64) {
        let ops = Self::gpr_operands(&m);
        let regs = ops.callee_saved();
        if !self.ctx.fast_paths() || Self::needs_slow(&m) {
            self.emit_slow_inline(word, pc, regs.as_slice());
            return;
        }
        let mut avoid = ops;
        avoid.push(18);
        let vs = victims(avoid.as_slice());
        let [a, b, c, d] = vs;
        let slow = self.asm.label();
        let resume = self.asm.label();
        self.asm.emit(stp(a, b, SP, offsets::scratch(0) as i32));
        self.asm.emit(stp(c, d, SP, offsets::scratch(2) as i32));

        self.emit_address(&m, literal, a, c);
        let el = if m.unprivileged { 0 } else { self.ctx.el };
        let kind = if m.load
            && !matches!(
                m.kind,
                MemKind::Cas | MemKind::CasPair | MemKind::Atomic | MemKind::StoreExclusive
            ) {
            TlbKind::Load
        } else {
            TlbKind::Store
        };
        self.emit_tlb_check(m.total, tlb_index(el, kind), a, b, c, d, slow);
        if m.align > 1 {
            self.asm.emit(and_imm(true, d, a, (m.align - 1) as u64));
            self.asm.cbnz(true, d, slow);
        }
        self.emit_access(&m, a, b, c, d);
        self.emit_writeback(&m, a, c, d);
        self.asm.emit(ldp(c, d, SP, offsets::scratch(2) as i32));
        self.asm.emit(ldp(a, b, SP, offsets::scratch(0) as i32));
        self.asm.bind(resume);
        self.slow.push(SlowSite {
            label: slow,
            resume,
            word,
            pc,
            victims: vs,
            regs,
        });
    }

    /// Puts the guest virtual address of the access in `a`, using `c` as a
    /// temporary.
    fn emit_address(&mut self, m: &Mem, literal: Option<u64>, a: Reg, c: Reg) {
        if let Some(addr) = literal {
            emit::guest_va(&mut self.asm, a, addr, self.page << 12, c);
            return;
        }
        let base = match m.rn {
            31 => {
                self.asm.emit(ldr_imm(3, a, SP, offsets::SP));
                a
            }
            18 => {
                self.asm.emit(ldr_imm(3, a, SP, offsets::x(18)));
                a
            }
            r => r,
        };
        if let Some((rm, option, amount)) = m.index_reg {
            let idx = if rm == 18 {
                self.asm.emit(ldr_imm(3, c, SP, offsets::x(18)));
                c
            } else {
                rm
            };
            self.asm.emit(add_ext(a, base, idx, option, amount));
        } else if m.pre && m.offset != 0 {
            self.asm.emit_all(add_const(a, base, m.offset, c));
        } else if base != a {
            self.asm.emit(mov(a, base));
        }
    }

    /// The software-TLB lookup for `total` bytes at the address in `a`: on a
    /// hit `b` holds the delta to the host address; a miss, or an access whose
    /// last byte is in another page, branches to `slow`.
    #[allow(clippy::too_many_arguments)]
    fn emit_tlb_check(
        &mut self,
        total: u32,
        tlb: usize,
        a: Reg,
        b: Reg,
        c: Reg,
        d: Reg,
        slow: Label,
    ) {
        if total > 1 {
            self.asm.emit(add_imm(true, c, a, total - 1, false));
            self.asm.emit(emit::page_number(c, c));
        } else {
            self.asm.emit(emit::page_number(c, a));
        }
        self.asm.emit(ubfx(b, a, TLB_SHIFT, TLB_BITS));
        self.asm.emit(add_ext(b, SP, b, EXT_UXTX, 4));
        let off = offsets::tlb(tlb);
        debug_assert_eq!(off & 0xfff, 0);
        self.asm.emit(add_imm(true, b, b, off >> 12, true));
        self.asm.emit(ldp(d, b, b, 0));
        self.asm.emit(emit::sub_reg(true, d, d, c));
        self.asm.cbnz(true, d, slow);
    }

    /// The access itself, with the guest address in `a` and the delta in `b`;
    /// `c` and `d` are free.
    fn emit_access(&mut self, m: &Mem, a: Reg, b: Reg, c: Reg, d: Reg) {
        match m.kind {
            MemKind::Gpr | MemKind::Vec => {
                // The same size, sign and register class, register-offset
                // addressing: [a, b].
                let gpr = m.kind == MemKind::Gpr;
                let rt = if gpr && m.rt == 18 { d } else { m.rt };
                if gpr && m.rt == 18 && !m.load {
                    self.asm.emit(ldr_imm(3, d, SP, offsets::x(18)));
                }
                self.asm
                    .emit((m.word & 0xc4c0_0000) | 0x3820_6800 | b << 16 | a << 5 | rt);
                if gpr && m.rt == 18 && m.load {
                    self.asm.emit(str_imm(3, d, SP, offsets::x(18)));
                }
            }
            MemKind::Pair | MemKind::VecPair => {
                self.asm.emit(add_reg(true, b, a, b, LSL, 0));
                let gpr = m.kind == MemKind::Pair;
                let sub = |r: u32, with: Reg| if gpr && r == 18 { with } else { r };
                let (rt, rt2) = (sub(m.rt, d), sub(m.rt2, c));
                if gpr && !m.load {
                    if m.rt == 18 {
                        self.asm.emit(ldr_imm(3, d, SP, offsets::x(18)));
                    }
                    if m.rt2 == 18 {
                        self.asm.emit(ldr_imm(3, c, SP, offsets::x(18)));
                    }
                }
                self.asm
                    .emit((m.word & 0xc440_0000) | 0x2900_0000 | rt2 << 10 | b << 5 | rt);
                if gpr && m.load {
                    if m.rt == 18 {
                        self.asm.emit(str_imm(3, d, SP, offsets::x(18)));
                    }
                    if m.rt2 == 18 {
                        self.asm.emit(str_imm(3, c, SP, offsets::x(18)));
                    }
                }
            }
            MemKind::VecStruct => {
                self.asm.emit(add_reg(true, b, a, b, LSL, 0));
                self.asm
                    .emit((m.word & !(0x1f << 16) & !(1 << 23) & !(0x1f << 5)) | b << 5);
            }
            MemKind::LoadExclusive => {
                self.asm.emit(add_reg(true, b, a, b, LSL, 0));
                let rt = if m.rt == 18 { d } else { m.rt };
                if m.acquire {
                    self.asm.emit(ldar(m.size, rt, b));
                } else {
                    self.asm.emit(ldr_imm(m.size, rt, b, 0));
                }
                self.asm.emit(str_imm(3, a, SP, offsets::MONITOR));
                self.asm.emit(str_imm(3, rt, SP, offsets::MONITOR_VALUE));
                if m.rt == 18 {
                    self.asm.emit(str_imm(3, d, SP, offsets::x(18)));
                }
            }
            MemKind::StoreExclusive => {
                // Succeeds only if the reservation is for this address and
                // memory still holds what the load-exclusive read; the check
                // and the store are one compare-and-swap.
                let fail = self.asm.label();
                let done = self.asm.label();
                self.asm.emit(add_reg(true, b, a, b, LSL, 0));
                self.asm.emit(ldr_imm(3, d, SP, offsets::MONITOR));
                self.asm.emit(emit::sub_reg(true, d, d, a));
                self.asm.cbnz(true, d, fail);
                self.asm.emit(ldr_imm(3, d, SP, offsets::MONITOR_VALUE));
                self.asm.emit(mov(c, d));
                self.asm.emit(cas(m.size, false, m.release, d, m.rt, b));
                self.asm.emit(emit::sub_reg(true, d, d, c));
                self.asm.cbnz(true, d, fail);
                self.asm.emit(movz(false, m.rs, 0, 0));
                self.asm.b(done);
                self.asm.bind(fail);
                self.asm.emit(movz(false, m.rs, 1, 0));
                self.asm.bind(done);
                self.asm.emit(str_imm(3, XZR, SP, offsets::MONITOR));
            }
            MemKind::LoadAcquire | MemKind::StoreRelease => {
                self.asm.emit(add_reg(true, b, a, b, LSL, 0));
                let rt = if m.rt == 18 { d } else { m.rt };
                if m.load {
                    self.asm.emit(ldar(m.size, rt, b));
                    if m.rt == 18 {
                        self.asm.emit(str_imm(3, d, SP, offsets::x(18)));
                    }
                } else {
                    if m.rt == 18 {
                        self.asm.emit(ldr_imm(3, d, SP, offsets::x(18)));
                    }
                    self.asm.emit(stlr(m.size, rt, b));
                }
            }
            MemKind::Cas | MemKind::CasPair | MemKind::Atomic => {
                // The host's own instruction on the host address.
                self.asm.emit(add_reg(true, b, a, b, LSL, 0));
                self.asm.emit((m.word & !(0x1f << 5)) | b << 5);
            }
        }
    }

    fn emit_writeback(&mut self, m: &Mem, a: Reg, c: Reg, d: Reg) {
        if !m.writeback {
            return;
        }
        let slot = match m.rn {
            31 => Some(offsets::SP),
            18 => Some(offsets::x(18)),
            _ => None,
        };
        if m.pre {
            match slot {
                Some(off) => self.asm.emit(str_imm(3, a, SP, off)),
                None => self.asm.emit(mov(m.rn, a)),
            }
            return;
        }
        // Post-index: the base plus the offset or the register.
        let idx = m.wb_reg.map(|rm| {
            if rm == 18 {
                self.asm.emit(ldr_imm(3, d, SP, offsets::x(18)));
                d
            } else {
                rm
            }
        });
        let target = match slot {
            Some(off) => {
                self.asm.emit(ldr_imm(3, c, SP, off));
                c
            }
            None => m.rn,
        };
        match idx {
            Some(r) => self.asm.emit(add_reg(true, target, target, r, LSL, 0)),
            None => self.asm.emit_all(add_const(
                target,
                target,
                m.offset,
                if target == c { d } else { c },
            )),
        }
        if let Some(off) = slot {
            self.asm.emit(str_imm(3, c, SP, off));
        }
    }

    fn emit_dc_zva(&mut self, word: u32, rt: u32, pc: u64) {
        let mut regs = Regs::default();
        if (19..=29).contains(&rt) {
            regs.push(rt);
        }
        if !self.ctx.fast_paths() {
            self.emit_slow_inline(word, pc, regs.as_slice());
            return;
        }
        let vs = victims(&[rt, 18]);
        let [a, b, c, d] = vs;
        let slow = self.asm.label();
        let resume = self.asm.label();
        self.asm.emit(stp(a, b, SP, offsets::scratch(0) as i32));
        self.asm.emit(stp(c, d, SP, offsets::scratch(2) as i32));
        match rt {
            31 => self.asm.emit(movz(true, a, 0, 0)),
            18 => {
                self.asm.emit(ldr_imm(3, a, SP, offsets::x(18)));
                self.asm.emit(and_imm(true, a, a, !63));
            }
            r => self.asm.emit(and_imm(true, a, r, !63)),
        }
        self.emit_tlb_check(64, tlb_index(self.ctx.el, TlbKind::Store), a, b, c, d, slow);
        self.asm.emit(add_reg(true, b, a, b, LSL, 0));
        for off in [0, 16, 32, 48] {
            self.asm.emit(stp(XZR, XZR, b, off));
        }
        self.asm.emit(ldp(c, d, SP, offsets::scratch(2) as i32));
        self.asm.emit(ldp(a, b, SP, offsets::scratch(0) as i32));
        self.asm.bind(resume);
        self.slow.push(SlowSite {
            label: slow,
            resume,
            word,
            pc,
            victims: vs,
            regs,
        });
    }

    /// The call into the memory slow path: stores the operand registers the
    /// Rust side reads from among X19..X29 (it reads the rest from the state
    /// the stub fills), and X30; then the BL whose return address is the
    /// literal; then the return point, which reloads X16 and those operands.
    fn emit_slow_call(&mut self, word: u32, pc: u64, regs: &[Reg]) {
        for &r in regs {
            self.asm.emit(str_imm(3, r, SP, offsets::x(r)));
        }
        self.asm.emit(str_imm(3, 30, SP, offsets::x(30)));
        self.asm.bl_abs(self.ctx.stubs.mem);
        self.asm.emit(word);
        self.asm.emit(16); // return 16 bytes past the literal's start
        let off = pc.wrapping_sub(self.page << 12);
        self.asm.emit(off as u32);
        self.asm.emit((off >> 32) as u32);
        self.asm
            .emit(ldr_imm(3, JUMP_REG, SP, offsets::x(JUMP_REG)));
        for &r in regs {
            self.asm.emit(ldr_imm(3, r, SP, offsets::x(r)));
        }
    }

    fn emit_slow_inline(&mut self, word: u32, pc: u64, regs: &[Reg]) {
        self.emit_slow_call(word, pc, regs);
    }

    /// The out-of-line slow paths of the block's fast loads and stores, after
    /// its code.
    fn emit_slow_sites(&mut self) {
        for site in std::mem::take(&mut self.slow) {
            self.asm.bind(site.label);
            let [a, b, c, d] = site.victims;
            self.asm.emit(ldp(c, d, SP, offsets::scratch(2) as i32));
            self.asm.emit(ldp(a, b, SP, offsets::scratch(0) as i32));
            self.emit_slow_call(site.word, site.pc, site.regs.as_slice());
            self.asm.b(site.resume);
        }
    }

    // ------------------------------------------------------------ branches

    /// Leaves for the dispatcher at `next` if another thread asked this CPU to
    /// come back (`exit_request`); otherwise carries on. Borrows X16.
    fn exit_if_requested(&mut self, next: u64) {
        let go = self.asm.label();
        let cont = self.asm.label();
        self.asm.emit(str_imm(3, 16, SP, offsets::scratch(0)));
        // Order the requests after the loads before the ISB: a CPU that read
        // another's "code is ready" flag must also see the invalidation that
        // came before it. The exit and sync requests are one doubleword.
        self.asm.emit(DMB_ISHLD);
        self.asm.emit(ldr_imm(3, 16, SP, offsets::EXIT_REQUEST));
        self.asm.cbnz(true, 16, go);
        self.asm.emit(ldr_imm(3, 16, SP, offsets::scratch(0)));
        self.asm.b(cont);
        self.asm.bind(go);
        self.asm.emit(ldr_imm(3, 16, SP, offsets::scratch(0)));
        let terminated = self.terminated;
        self.exit(Exit::Branch as u32, Next::Const(next), ExitInfo::default());
        self.terminated = terminated;
        self.asm.bind(cont);
    }

    /// Continues at a constant guest address, with an optional link value for
    /// X30. `pc` is the branch's own address (to tell backward from forward).
    fn goto_const(&mut self, target: u64, link: Option<u64>, pc: u64) {
        if !self.ctx.fast_paths() {
            self.exit(
                Exit::Branch as u32,
                Next::Const(target),
                ExitInfo {
                    link,
                    ..Default::default()
                },
            );
            return;
        }
        if let Some(ret) = link {
            emit::guest_va(&mut self.asm, 30, ret, self.page << 12, 31);
        }
        if target >> 12 == self.page && target & 3 == 0 {
            // Same page: a branch the dispatcher can patch to the target block.
            let exit = self.asm.label();
            if target <= pc {
                // Backward: a chained loop must still let the CPU leave.
                let kicked = self.asm.label();
                self.asm.emit(str_imm(3, 16, SP, offsets::scratch(0)));
                self.asm.emit(ldr_imm(2, 16, SP, offsets::EXIT_REQUEST));
                self.asm.cbnz(false, 16, kicked);
                self.asm.emit(ldr_imm(3, 16, SP, offsets::scratch(0)));
                let site = self.asm.label();
                self.asm.bind(site);
                self.asm.b(exit);
                self.asm.bind(kicked);
                self.asm.emit(ldr_imm(3, 16, SP, offsets::scratch(0)));
                self.asm.bind(exit);
                self.exit(
                    Exit::Branch as u32,
                    Next::Const(target),
                    ExitInfo {
                        chain_site: Some(site),
                        ..Default::default()
                    },
                );
            } else {
                let site = self.asm.label();
                self.asm.bind(site);
                self.asm.b(exit);
                self.asm.bind(exit);
                self.exit(
                    Exit::Branch as u32,
                    Next::Const(target),
                    ExitInfo {
                        chain_site: Some(site),
                        ..Default::default()
                    },
                );
            }
            return;
        }
        // Another page: through the jump cache.
        self.asm.emit(stp(15, 16, SP, offsets::x(15) as i32));
        self.asm.emit(str_imm(3, 17, SP, offsets::x(17)));
        emit::guest_va(&mut self.asm, JUMP_REG, target, self.page << 12, 17);
        self.asm
            .b_abs(self.ctx.stubs.jc[self.ctx.el.min(1) as usize]);
        self.terminated = true;
    }

    /// Continues at the address in guest register `rn` (BR, BLR, RET).
    fn goto_reg(&mut self, rn: u32, link: Option<u64>) {
        if !self.ctx.fast_paths() {
            self.exit(
                Exit::Branch as u32,
                Next::Reg(rn),
                ExitInfo {
                    link,
                    ..Default::default()
                },
            );
            return;
        }
        self.asm.emit(stp(15, 16, SP, offsets::x(15) as i32));
        self.asm.emit(str_imm(3, 17, SP, offsets::x(17)));
        match rn {
            15..=18 => self.asm.emit(ldr_imm(3, JUMP_REG, SP, offsets::x(rn))),
            31 => self.asm.emit(movz(true, JUMP_REG, 0, 0)),
            r => self.asm.emit(mov(JUMP_REG, r)),
        }
        if let Some(ret) = link {
            emit::guest_va(&mut self.asm, 30, ret, self.page << 12, 31);
        }
        self.asm
            .b_abs(self.ctx.stubs.jc[self.ctx.el.min(1) as usize]);
        self.terminated = true;
    }

    fn emit_cond_branch(&mut self, word: u32, kind: CondKind, taken: u64, pc: u64) {
        let taken_label = self.asm.label();
        // A compare-and-branch on guest X18 reads it from the state block (X18 is
        // virtual) via a victim, restored on both paths.
        let x18_reg = match kind {
            CondKind::Cbz { rt, .. } | CondKind::Cbnz { rt, .. } => rt == 18,
            CondKind::Tbz { rt, .. } | CondKind::Tbnz { rt, .. } => rt == 18,
            CondKind::Cond => false,
        };
        let v = Self::victim(&[]);
        let r = |rt: u32| if rt == 18 { v } else { rt };
        if x18_reg {
            self.asm.emit(str_imm(3, v, SP, offsets::SCRATCH));
            self.asm.emit(ldr_imm(3, v, SP, offsets::x(18)));
        }
        match kind {
            CondKind::Cond => self.asm.b_cond(word & 0xf, taken_label),
            CondKind::Cbz { sf, rt } => self.asm.cbz(sf, r(rt), taken_label),
            CondKind::Cbnz { sf, rt } => self.asm.cbnz(sf, r(rt), taken_label),
            CondKind::Tbz { rt, bit } => self.asm.tbz(r(rt), bit, taken_label),
            CondKind::Tbnz { rt, bit } => self.asm.tbnz(r(rt), bit, taken_label),
        }
        // Fall-through (condition false); restore the victim first.
        if x18_reg {
            self.asm.emit(ldr_imm(3, v, SP, offsets::SCRATCH));
        }
        self.goto_const(pc.wrapping_add(4), None, pc);
        self.asm.bind(taken_label);
        if x18_reg {
            self.asm.emit(ldr_imm(3, v, SP, offsets::SCRATCH));
        }
        self.goto_const(taken, None, pc);
        self.terminated = true;
    }

    // ------------------------------------------------------------ system registers

    /// The system registers that are plain state, or the host's own, inlined.
    /// Returns false to leave the access to the dispatcher.
    fn emit_inline_sysreg(&mut self, read: bool, reg: u16, rt: u32, word: u32, _pc: u64) -> bool {
        use crate::state::offsets as o;
        let el1 = self.ctx.el == 1;
        // Registers that are plain fields of the state: the dispatcher reads
        // them where it needs them (ELR, SPSR and VBAR when it takes an
        // exception or returns from one).
        let state_slot = match reg {
            r if r == sr::TPIDR_EL0 => Some(o::TPIDR_EL0),
            r if r == sr::TPIDRRO_EL0 && (read || el1) => Some(o::TPIDRRO_EL0),
            _ if !el1 => None,
            r if r == sr::TPIDR_EL1 => Some(o::TPIDR_EL1),
            r if r == sr::SP_EL0 => Some(o::SP_EL0),
            r if r == sr::ELR_EL1 => Some(o::banked(offset_of!(Banked, elr_el1))),
            r if r == sr::SPSR_EL1 => Some(o::banked(offset_of!(Banked, spsr_el1))),
            r if r == sr::ESR_EL1 => Some(o::banked(offset_of!(Banked, esr_el1))),
            r if r == sr::FAR_EL1 => Some(o::banked(offset_of!(Banked, far_el1))),
            r if r == sr::VBAR_EL1 => Some(o::banked(offset_of!(Banked, vbar_el1))),
            r if r == sr::PAR_EL1 => Some(o::banked(offset_of!(Banked, par_el1))),
            r if r == sr::CPACR_EL1 => Some(o::banked(offset_of!(Banked, cpacr_el1))),
            r if r == sr::CONTEXTIDR_EL1 => {
                Some(o::MMU + offset_of!(MmuRegs, contextidr_el1) as u32)
            }
            r if r == sr::MDSCR_EL1 && read => Some(o::banked(offset_of!(Banked, mdscr_el1))),
            _ => None,
        };
        if let Some(off) = state_slot {
            if read {
                self.load_state_into(rt, off);
            } else {
                self.store_reg_to_state(rt, off);
            }
            return true;
        }
        let host_reg = reg == sr::NZCV
            || reg == sr::FPCR
            || reg == sr::FPSR
            || (read && (reg == sr::CNTVCT_EL0 || reg == sr::CNTPCT_EL0 || reg == sr::CNTFRQ_EL0));
        if host_reg {
            // The host register holds the guest's value (NZCV, FPCR, FPSR), or
            // the guest's counter is the host's (CNTVCT, CNTPCT, CNTFRQ).
            if rt == 18 {
                self.emit_x18_copy(word, &[0]);
            } else {
                self.asm.emit(word);
            }
            return true;
        }
        // Registers that are constants of the machine or of this CPU.
        let constant = match reg {
            _ if !read => None,
            r if r == sr::CURRENT_EL && el1 => Some(1 << 2),
            r if r == sr::DCZID_EL0 => Some(crate::backend::DCZID_EL0),
            r if r == sr::CTR_EL0 => Some(crate::backend::CTR_EL0),
            r if r == sr::MIDR_EL1 && el1 => Some(crate::backend::MIDR_EL1),
            r if r == sr::MPIDR_EL1 && el1 => Some(self.ctx.mpidr),
            _ => None,
        };
        if let Some(v) = constant {
            self.set_reg_const(rt, v);
            return true;
        }
        // The translation controls read back as their fields (writes change
        // what translations hold, so they go to the dispatcher).
        let mmu = |off: usize| Some(o::MMU + off as u32);
        let mmu_slot = match reg {
            _ if !read || !el1 => None,
            r if r == sr::SCTLR_EL1 => mmu(offset_of!(MmuRegs, sctlr_el1)),
            r if r == sr::TTBR0_EL1 => mmu(offset_of!(MmuRegs, ttbr0_el1)),
            r if r == sr::TTBR1_EL1 => mmu(offset_of!(MmuRegs, ttbr1_el1)),
            r if r == sr::TCR_EL1 => mmu(offset_of!(MmuRegs, tcr_el1)),
            r if r == sr::MAIR_EL1 => mmu(offset_of!(MmuRegs, mair_el1)),
            _ => None,
        };
        if let Some(off) = mmu_slot {
            self.load_state_into(rt, off);
            return true;
        }
        if reg == sr::DAIF && el1 {
            if read {
                self.load_state_into(rt, o::PSTATE);
                if rt != 31 {
                    let r = if rt == 18 { 16 } else { rt };
                    if rt == 18 {
                        self.asm.emit(str_imm(3, 16, SP, o::scratch(0)));
                        self.asm.emit(ldr_imm(3, 16, SP, o::x(18)));
                    }
                    self.asm.emit(and_imm(true, r, r, 0x3c0));
                    if rt == 18 {
                        self.asm.emit(str_imm(3, 16, SP, o::x(18)));
                        self.asm.emit(ldr_imm(3, 16, SP, o::scratch(0)));
                    }
                }
            } else {
                self.emit_daif_write(rt, _pc);
            }
            return true;
        }
        false
    }

    /// Rt = the 64-bit state field at `off` (Rt may be X18 or the zero
    /// register).
    fn load_state_into(&mut self, rt: u32, off: u32) {
        match rt {
            31 => {}
            18 => {
                self.asm.emit(str_imm(3, 16, SP, offsets::scratch(0)));
                self.asm.emit(ldr_imm(3, 16, SP, off));
                self.asm.emit(str_imm(3, 16, SP, offsets::x(18)));
                self.asm.emit(ldr_imm(3, 16, SP, offsets::scratch(0)));
            }
            r => self.asm.emit(ldr_imm(3, r, SP, off)),
        }
    }

    fn store_reg_to_state(&mut self, rt: u32, off: u32) {
        match rt {
            18 => {
                self.asm.emit(str_imm(3, 16, SP, offsets::scratch(0)));
                self.asm.emit(ldr_imm(3, 16, SP, offsets::x(18)));
                self.asm.emit(str_imm(3, 16, SP, off));
                self.asm.emit(ldr_imm(3, 16, SP, offsets::scratch(0)));
            }
            r => self.asm.emit(str_imm(3, r, SP, off)),
        }
    }

    /// MSR DAIF, Xt: replaces PSTATE.DAIF; if that unmasks IRQ or FIQ while an
    /// interrupt may be pending, leaves for the dispatcher to take it.
    fn emit_daif_write(&mut self, rt: u32, pc: u64) {
        self.asm.emit(stp(16, 17, SP, offsets::scratch(0) as i32));
        self.asm.emit(ldr_imm(3, 16, SP, offsets::PSTATE));
        self.asm.emit(and_imm(true, 16, 16, !0x3c0));
        match rt {
            31 => self.asm.emit(movz(true, 17, 0, 0)),
            18 => {
                self.asm.emit(ldr_imm(3, 17, SP, offsets::x(18)));
                self.asm.emit(and_imm(true, 17, 17, 0x3c0));
            }
            r => self.asm.emit(and_imm(true, 17, r, 0x3c0)),
        }
        self.asm.emit(orr_reg(true, 16, 16, 17, LSL, 0));
        self.asm.emit(str_imm(3, 16, SP, offsets::PSTATE));
        // Unmasked I or F (bits 7:6 clear in the new value) and a hint set?
        self.asm.emit(eor_imm(true, 17, 17, 0xc0));
        self.asm.emit(and_imm(true, 17, 17, 0xc0));
        self.unmask_check(pc);
    }

    /// MSR DAIFSet/DAIFClr, #imm.
    fn emit_daif_imm(&mut self, set: bool, crm: u32, pc: u64) {
        let mask = ((crm as u64) & 0xf) << 6;
        if mask == 0 {
            return;
        }
        self.asm.emit(stp(16, 17, SP, offsets::scratch(0) as i32));
        self.asm.emit(ldr_imm(3, 16, SP, offsets::PSTATE));
        if set {
            self.asm.emit(orr_imm(true, 16, 16, mask));
            self.asm.emit(str_imm(3, 16, SP, offsets::PSTATE));
            self.asm.emit(ldp(16, 17, SP, offsets::scratch(0) as i32));
            return;
        }
        self.asm.emit(and_imm(true, 16, 16, !mask));
        self.asm.emit(str_imm(3, 16, SP, offsets::PSTATE));
        if mask & 0xc0 == 0 {
            self.asm.emit(ldp(16, 17, SP, offsets::scratch(0) as i32));
            return;
        }
        self.asm.emit(movz(true, 17, 1, 0));
        self.unmask_check(pc);
    }

    /// With X16/X17 borrowed (saved in scratch 0/1) and X17 non-zero when the
    /// write unmasked IRQ or FIQ: if an interrupt may be pending, exit to the
    /// dispatcher at the next instruction; otherwise restore and carry on.
    fn unmask_check(&mut self, pc: u64) {
        let cont = self.asm.label();
        let go = self.asm.label();
        self.asm.cbz(true, 17, cont);
        self.asm.emit(ldr_imm(2, 16, SP, offsets::IRQ_HINT));
        self.asm.cbnz(false, 16, go);
        self.asm.bind(cont);
        let done = self.asm.label();
        self.asm.emit(ldp(16, 17, SP, offsets::scratch(0) as i32));
        self.asm.b(done);
        self.asm.bind(go);
        self.asm.emit(ldp(16, 17, SP, offsets::scratch(0) as i32));
        let terminated = self.terminated;
        self.exit(
            Exit::Branch as u32,
            Next::Const(pc.wrapping_add(4)),
            ExitInfo::default(),
        );
        self.terminated = terminated;
        self.asm.bind(done);
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
fn eor_imm(sf: bool, rd: Reg, rn: Reg, value: u64) -> u32 {
    let f = logical_imm(value, sf).expect("not a logical immediate");
    (sf as u32) << 31 | 0x5200_0000 | f << 10 | rn << 5 | rd
}
const DMB_ISHLD: u32 = 0xd503_39bf;

/// LDAR of `size` (log2 bytes).
fn ldar(size: u32, rt: Reg, rn: Reg) -> u32 {
    size << 30 | 0x08df_fc00 | rn << 5 | rt
}
/// STLR of `size`.
fn stlr(size: u32, rt: Reg, rn: Reg) -> u32 {
    size << 30 | 0x089f_fc00 | rn << 5 | rt
}

/// System-register encodings the translator inlines, packed as `SysReg`.
mod sr {
    const fn enc(op0: u16, op1: u16, crn: u16, crm: u16, op2: u16) -> u16 {
        (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2
    }
    pub const NZCV: u16 = enc(3, 3, 4, 2, 0);
    pub const DAIF: u16 = enc(3, 3, 4, 2, 1);
    pub const FPCR: u16 = enc(3, 3, 4, 4, 0);
    pub const FPSR: u16 = enc(3, 3, 4, 4, 1);
    pub const CURRENT_EL: u16 = enc(3, 0, 4, 2, 2);
    pub const SP_EL0: u16 = enc(3, 0, 4, 1, 0);
    pub const TPIDR_EL0: u16 = enc(3, 3, 13, 0, 2);
    pub const TPIDRRO_EL0: u16 = enc(3, 3, 13, 0, 3);
    pub const TPIDR_EL1: u16 = enc(3, 0, 13, 0, 4);
    pub const CNTFRQ_EL0: u16 = enc(3, 3, 14, 0, 0);
    pub const CNTPCT_EL0: u16 = enc(3, 3, 14, 0, 1);
    pub const CNTVCT_EL0: u16 = enc(3, 3, 14, 0, 2);
    pub const ELR_EL1: u16 = enc(3, 0, 4, 0, 1);
    pub const SPSR_EL1: u16 = enc(3, 0, 4, 0, 0);
    pub const ESR_EL1: u16 = enc(3, 0, 5, 2, 0);
    pub const FAR_EL1: u16 = enc(3, 0, 6, 0, 0);
    pub const VBAR_EL1: u16 = enc(3, 0, 12, 0, 0);
    pub const PAR_EL1: u16 = enc(3, 0, 7, 4, 0);
    pub const CPACR_EL1: u16 = enc(3, 0, 1, 0, 2);
    pub const CONTEXTIDR_EL1: u16 = enc(3, 0, 13, 0, 1);
    pub const MDSCR_EL1: u16 = enc(2, 0, 0, 2, 2);
    pub const DCZID_EL0: u16 = enc(3, 3, 0, 0, 7);
    pub const CTR_EL0: u16 = enc(3, 3, 0, 0, 1);
    pub const MIDR_EL1: u16 = enc(3, 0, 0, 0, 0);
    pub const MPIDR_EL1: u16 = enc(3, 0, 0, 0, 5);
    pub const SCTLR_EL1: u16 = enc(3, 0, 1, 0, 0);
    pub const TTBR0_EL1: u16 = enc(3, 0, 2, 0, 0);
    pub const TTBR1_EL1: u16 = enc(3, 0, 2, 0, 1);
    pub const TCR_EL1: u16 = enc(3, 0, 2, 0, 2);
    pub const MAIR_EL1: u16 = enc(3, 0, 10, 2, 0);
}

/// The bit-positions of the GPR operand fields a verbatim-copied instruction
/// uses -- Rd at 0, Rn at 5, Rm at 16, Ra at 10, as its class needs them.
/// Empty for anything with no GPR operand. Used only to find and rewrite guest
/// X18, which cannot live in the host's reserved x18.
pub fn gpr_field_positions(word: u32) -> &'static [u32] {
    const RD_RN: &[u32] = &[0, 5];
    const RD: &[u32] = &[0];
    const RN: &[u32] = &[5];
    const RD_RN_RM: &[u32] = &[0, 5, 16];
    const RD_RN_RM_RA: &[u32] = &[0, 5, 16, 10];
    const RN_RM: &[u32] = &[5, 16];
    match (word >> 25) & 0xf {
        // Data processing -- immediate. Add/sub, logical and bitfield take Rd and
        // Rn; move-wide only Rd; extract also Rm. (PC-rel and tagged add/sub are
        // decoded elsewhere, never as Copy.)
        0b1000 | 0b1001 => match (word >> 23) & 7 {
            0b010 | 0b100 | 0b110 => RD_RN,
            0b101 => RD,
            0b111 => RD_RN_RM,
            _ => &[],
        },
        // Data processing -- register.
        0b0101 | 0b1101 => {
            if (word >> 28) & 1 == 0 {
                // Logical, add/sub shifted, add/sub extended: Rd, Rn, Rm.
                RD_RN_RM
            } else {
                match (word >> 21) & 0xf {
                    0b0000 => RD_RN_RM, // add/sub with carry
                    0b0010 => {
                        // conditional compare: no Rd
                        if word & (1 << 11) == 0 { RN_RM } else { RN }
                    }
                    0b0100 => RD_RN_RM, // conditional select
                    0b0110 => {
                        // 1- / 2-source
                        if (word >> 30) & 1 == 1 {
                            RD_RN
                        } else {
                            RD_RN_RM
                        }
                    }
                    o if o & 0b1000 != 0 => RD_RN_RM_RA, // 3-source (Ra)
                    _ => RD_RN_RM,                       // unallocated: conservative
                }
            }
        }
        // Data processing -- SIMD and FP: only the transfers between general
        // and SIMD&FP registers name a general register.
        0b0111 | 0b1111 => simd_gpr_field(word),
        // MRS and MSR (register) keep Rt at bit 0.
        0b1010 | 0b1011 if word >> 22 == 0b11_0101_0100 && (word >> 20) & 1 == 1 => RD,
        _ => &[],
    }
}

/// The general-register field of a SIMD&FP instruction that has one: the
/// conversions between floating-point and integer or fixed-point (Rd for the
/// conversions to an integer and the moves out, Rn for the conversions from
/// an integer and the moves in), and DUP/INS (general) and SMOV/UMOV.
fn simd_gpr_field(word: u32) -> &'static [u32] {
    const RD: &[u32] = &[0];
    const RN: &[u32] = &[5];
    // Floating-point <-> integer / fixed-point: bits 30 = 0, 28:24 = 11110.
    if word & (1 << 30) == 0 && (word >> 24) & 0x1f == 0b11110 {
        let int_conv = word & (1 << 21) != 0;
        if int_conv && (word >> 10) & 0x3f != 0 {
            return &[]; // other FP data processing in this space
        }
        let rmode = (word >> 19) & 3;
        let opcode = (word >> 16) & 7;
        return match opcode {
            0b010 | 0b011 => RN,                             // SCVTF, UCVTF
            0b111 if int_conv => RN,                         // FMOV to FP
            0b110 if int_conv && rmode != 0b11 => RD,        // FMOV to general
            0b110 if int_conv => RD,                         // FJCVTZS
            0b000 | 0b001 | 0b100 | 0b101 if int_conv => RD, // FCVT*S/U
            0b000 | 0b001 => RD,                             // FCVTZS/U (fixed)
            _ => &[],
        };
    }
    // Advanced SIMD copy: 0 Q op 01110000 imm5 0 imm4 1 Rn Rd, op = 0.
    if word & (1 << 31) == 0
        && (word >> 21) & 0x1ff == 0b0_0111_0000
        && word & (1 << 15) == 0
        && word & (1 << 10) != 0
    {
        return match (word >> 11) & 0xf {
            0b0001 | 0b0011 => RN, // DUP (general), INS (general)
            0b0101 | 0b0111 => RD, // SMOV, UMOV
            _ => &[],
        };
    }
    &[]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_x18_operands() {
        let uses = |w: u32| gpr_field_positions(w).iter().any(|&p| (w >> p) & 31 == 18);
        // mov x4, x18 (orr x4, xzr, x18): x18 is Rm, not written.
        assert!(uses(0xaa1203e4));
        assert_eq!(gpr_field_positions(0xaa1203e4), &[0, 5, 16]);
        // add x18, x18, #1: x18 is Rd and Rn.
        assert!(uses(0x91000652));
        // movz x18, #0x1234: x18 is Rd only.
        assert!(uses(0xd2824692));
        assert_eq!(gpr_field_positions(0xd2824692), &[0]);
        // add x5, x6, x18 (shifted register): x18 is Rm.
        assert!(uses(0x8b120ca5));
        // madd x0, x1, x2, x18: x18 is Ra (3-source).
        assert!(uses(0x9b024820));
        assert_eq!(gpr_field_positions(0x9b024820), &[0, 5, 16, 10]);
        // add x0, x0, x1: no x18.
        assert!(!uses(0x8b010000));
        // A SIMD/FP word with no general register is never rewritten here.
        assert_eq!(gpr_field_positions(0x4e218400), &[] as &[u32]);
        // fmov x18, d3 ; fmov d3, x18 ; scvtf d0, x18 ; dup v1.2d, x18 ;
        // umov x18, v2.d[0] ; ins v0.d[1], x18
        assert_eq!(gpr_field_positions(0x9e660072), &[0]);
        assert_eq!(gpr_field_positions(0x9e670243), &[5]);
        assert_eq!(gpr_field_positions(0x9e620240), &[5]);
        assert_eq!(gpr_field_positions(0x4e080e41), &[5]);
        assert_eq!(gpr_field_positions(0x4e083c52), &[0]);
        assert_eq!(gpr_field_positions(0x4e181e40), &[5]);
        // fadd d18, d0, d1 names v18, not x18.
        assert_eq!(gpr_field_positions(0x1e612812), &[] as &[u32]);
        // mrs x18, nzcv ; msr fpcr, x18
        assert_eq!(gpr_field_positions(0xd53b4212), &[0]);
        assert_eq!(gpr_field_positions(0xd51b4412), &[0]);
    }

    #[test]
    fn victims_avoid_operands() {
        let v = victims(&[16, 17, 15, 3, 18]);
        assert_eq!(v, [14, 13, 12, 11]);
        let v = victims(&[]);
        assert_eq!(v, [16, 17, 15, 14]);
    }
}
