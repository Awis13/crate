// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Translating one block: guest instructions from one 4 KiB page, starting at
//! a guest PC, into host code. Most instructions are copied; the rest are
//! rewritten here.
//!
//! **Loads and stores** take an inline fast path: the guest address is
//! computed into a scratch host register, its page looked up in the software
//! TLB for this exception level and kind of access, and on a hit the host
//! performs the guest's own instruction with its base register replaced by the
//! base's host address. A miss, a device, an access that crosses a page or one
//! that is misaligned where the architecture requires alignment, goes out of
//! line to the shared slow path, which calls into Rust; the Rust side performs
//! the access and fills the TLB entry.
//!
//! **Scratch registers.** Guest X15..X17 live in their slots of the state
//! between blocks, so their host registers are the fast path's scratch until
//! an instruction names one (`RegCache`). When the fast path needs others it
//! borrows them, preferring ones the rest of the block overwrites, and puts
//! them back only when an instruction names them or the block leaves.
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

/// A load or store's out-of-line call into the slow path, in the cold
/// section; the fast path branches to it through `tramp`, a branch after the
/// block's code (a conditional branch reaches only 1 MiB).
struct SlowSite {
    tramp: Label,
    label: Label,
    resume: Label,
    word: u32,
    pc: u64,
    /// Where the registers were at the access: the site puts every one back
    /// in its slot or host register before the call.
    cache: RegCache,
    /// The instruction's operands among X19..X29.
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

    fn mask(&self) -> u32 {
        self.as_slice().iter().fold(0, |m, &r| m | 1 << r)
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

/// X15..X17, whose guest values are in their slots of the state wherever
/// one block hands over to another (see `state`).
const HOME: u32 = 0b111 << 15;

/// The other host registers a block may borrow for scratch, best first: the
/// temporaries are the least likely to be operands.
const BORROWABLE: [Reg; 26] = [
    14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
];

/// Where the guest's general registers are, at one point of a block, when
/// they are not simply in the host register of the same number.
///
/// X15..X17 are in their slots at the block's start; `loaded` are the ones the
/// block has since loaded into their host registers, and `dirty` the loaded
/// ones it has changed since. `borrowed` are other registers whose host
/// register the block took for scratch after storing the guest's value in its
/// slot; an instruction that names one reloads it first. Before anything
/// leaves the block (a branch, an exit, a call into the memory slow path)
/// every register goes back where a block boundary has it: dirty ones are
/// stored and borrowed ones reloaded, so the guest's state at a fault or an
/// exit is exact.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct RegCache {
    loaded: u32,
    dirty: u32,
    borrowed: u32,
}

/// Emits what puts every register of `c` back where a block boundary has it:
/// stores of the dirty ones among X15..X17 and reloads of the borrowed ones,
/// pairing neighbours.
fn emit_sync(asm: &mut Asm, c: RegCache) {
    pairs(c.dirty, |r, both| {
        asm.emit(if both {
            stp(r, r + 1, SP, offsets::x(r) as i32)
        } else {
            str_imm(3, r, SP, offsets::x(r))
        })
    });
    pairs(c.borrowed, |r, both| {
        asm.emit(if both {
            ldp(r, r + 1, SP, offsets::x(r) as i32)
        } else {
            ldr_imm(3, r, SP, offsets::x(r))
        })
    });
}

/// Calls `f(r, true)` for each pair of neighbours r, r+1 in `mask` and
/// `f(r, false)` for each register left over, lowest first.
fn pairs(mut mask: u32, mut f: impl FnMut(Reg, bool)) {
    while mask != 0 {
        let r = mask.trailing_zeros();
        let both = r < 31 && mask & (2 << r) != 0;
        f(r, both);
        mask &= !(1 << r);
        if both {
            mask &= !(2 << r);
        }
    }
}

/// The general registers an instruction reads and the ones it writes, as
/// masks (register 31 dropped); one it writes without reading it writes
/// whole. A memory access counts as reading every register it names: if it
/// faults, the guest must find even its destination as it was.
fn uses(op: &Op, word: u32) -> (u32, u32) {
    let bit = |r: u32| if r < 31 { 1u32 << r } else { 0 };
    match *op {
        Op::Copy => copy_uses(word),
        Op::PcRel { rd, .. } => (0, bit(rd)),
        Op::AddSubSpImm { rd, rn, .. } => (bit(rn), bit(rd)),
        Op::AddSubSpExt { rd, rn, rm, .. } => (bit(rn) | bit(rm), bit(rd)),
        Op::LogicalSpImm { word } => (bit((word >> 5) & 31), 0),
        Op::Memory(m) => (Block::gpr_operands(&m).mask(), 0),
        Op::LiteralLoad { rt, vector, .. } => (if vector { 0 } else { bit(rt) }, 0),
        Op::DcZva { rt } => (bit(rt), 0),
        Op::CondBranch { kind, .. } => match kind {
            CondKind::Cbz { rt, .. }
            | CondKind::Cbnz { rt, .. }
            | CondKind::Tbz { rt, .. }
            | CondKind::Tbnz { rt, .. } => (bit(rt), 0),
            CondKind::Cond => (0, 0),
        },
        Op::IndBranch { rn, .. } => (bit(rn), 0),
        Op::Sysreg { read: true, rt, .. } => (0, bit(rt)),
        Op::Sysreg {
            read: false, rt, ..
        } => (bit(rt), 0),
        Op::System { word } => (bit(word & 31), 0),
        Op::Branch { .. }
        | Op::Nop
        | Op::MsrImm { .. }
        | Op::Call { .. }
        | Op::Wait { .. }
        | Op::Isb
        | Op::Clrex
        | Op::Eret
        | Op::Undefined => (0, 0),
    }
}

/// `uses` for a copied instruction: every general-register field is read,
/// except the destination of the forms that write it whole, and the field at
/// bit 0 is taken to be written in any case.
fn copy_uses(word: u32) -> (u32, u32) {
    let write_only = rd_write_only(word);
    let (mut reads, mut writes) = (0, 0);
    for &p in gpr_field_positions(word) {
        let r = (word >> p) & 31;
        if r == 31 {
            continue;
        }
        if p == 0 {
            writes |= 1 << r;
        }
        if p != 0 || !write_only {
            reads |= 1 << r;
        }
    }
    (reads, writes)
}

/// Whether a copied instruction's field at bit 0 is a destination it writes
/// whole without reading (Arm ARM, the A64 encoding index): not MOVK, BFM, the
/// pointer-authentication forms of the one-source group, or a field that is a
/// source (conditional compare, MSR, the moves into SIMD registers).
fn rd_write_only(word: u32) -> bool {
    match (word >> 25) & 0xf {
        0b1000 | 0b1001 => match (word >> 23) & 7 {
            0b010 | 0b100 | 0b111 => true,
            0b101 => (word >> 29) & 3 != 0b11,
            0b110 => (word >> 29) & 3 != 0b01,
            _ => false,
        },
        0b0101 | 0b1101 => {
            if (word >> 28) & 1 == 0 {
                true
            } else {
                match (word >> 21) & 0xf {
                    0b0000 | 0b0100 => true,
                    0b0110 => (word >> 30) & 1 == 0 || (word >> 16) & 0x1f == 0,
                    o => o & 0b1000 != 0,
                }
            }
        }
        0b0111 | 0b1111 => simd_gpr_field(word) == [0],
        0b1010 | 0b1011 => word >> 22 == 0b11_0101_0100 && (word >> 20) & 3 == 3,
        _ => false,
    }
}

/// Whether an instruction always ends the block it is in, so looking further
/// ahead is pointless.
fn ends_block(op: &Op) -> bool {
    matches!(
        op,
        Op::Branch { .. }
            | Op::CondBranch { .. }
            | Op::IndBranch { .. }
            | Op::Call { .. }
            | Op::Wait { .. }
            | Op::Eret
            | Op::Undefined
            | Op::System { .. }
    )
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
    /// The slow-path literals (the instruction word, and its offset in the
    /// page), placed after the code of the section that uses them.
    literals: Vec<(Label, u32, u64, bool)>,
    cache: RegCache,
    /// For each instruction of the block, from what follows it: the registers
    /// named again, and the ones written whole before they are next read.
    ahead: Vec<(u32, u32)>,
    /// The index in `ahead` of the instruction being translated.
    at: usize,
}

/// Translates guest code at `pc`, whose words up to the end of its 4 KiB page
/// are `words`, into a block of host code. `limit` caps the instructions.
pub fn translate(ctx: Ctx, pc: u64, words: &[u32], limit: usize) -> Block {
    let mut t = Block::new(ctx, pc);
    // Decode up to the first instruction that surely ends the block, so the
    // choice of scratch registers can look ahead.
    let mut ops = Vec::with_capacity(16);
    let mut cur = pc;
    for &word in words.iter().take(limit.max(1)) {
        let op = decode::decode(word, cur);
        ops.push((op, word, cur));
        if ends_block(&op) {
            break;
        }
        cur = cur.wrapping_add(4);
    }
    t.ahead = vec![(0, 0); ops.len()];
    let (mut named, mut dead) = (0u32, 0u32);
    for (i, &(op, word, _)) in ops.iter().enumerate().rev() {
        t.ahead[i] = (named, dead);
        let (reads, writes) = uses(&op, word);
        named |= reads | writes;
        dead = (dead & !(reads | writes)) | (writes & !reads);
    }
    let mut next = pc;
    for (i, &(op, word, at)) in ops.iter().enumerate() {
        t.at = i;
        next = at.wrapping_add(4);
        if t.emit(op, word, at) {
            break;
        }
    }
    if !t.terminated {
        t.goto_const(next, None, next.wrapping_sub(4));
    }
    t.emit_slow_sites();
    t
}

/// A block that only raises the instruction abort for a fetch at `pc` that
/// faulted with status `fsc`.
pub fn fetch_fault(ctx: Ctx, pc: u64, fsc: u8) -> Block {
    let mut t = Block::new(ctx, pc);
    t.fetch_faulted = true;
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
    fn new(ctx: Ctx, pc: u64) -> Block {
        Block {
            asm: Asm::new(),
            ctx,
            page: pc >> 12,
            terminated: false,
            fetch_faulted: false,
            slow: Vec::new(),
            literals: Vec::new(),
            cache: RegCache::default(),
            ahead: Vec::new(),
            at: 0,
        }
    }

    /// Emits an exit to the save stub.
    fn exit(&mut self, code: u32, next: Next, info: ExitInfo) {
        self.sync();
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
    /// any in `avoid`. The sequence parks it in a scratch slot and puts it
    /// back, so whatever it held (a guest value or nothing) survives.
    fn victim(avoid: &[u32]) -> u32 {
        (9..=15).find(|r| !avoid.contains(r)).unwrap()
    }

    /// Makes guest register `r` live in its host register for an
    /// instruction that reads it, writes it whole, or both: loads it from its
    /// slot when it is one of X15..X17 not loaded yet, or a borrowed register,
    /// and the instruction reads it.
    fn need(&mut self, r: u32, read: bool, write: bool) {
        if r >= 30 || r == 18 {
            return;
        }
        let bit = 1 << r;
        let c = &mut self.cache;
        if HOME & bit != 0 {
            if c.loaded & bit == 0 {
                if read {
                    self.asm.emit(ldr_imm(3, r, SP, offsets::x(r)));
                }
                c.loaded |= bit;
            }
            if write {
                c.dirty |= bit;
            }
        } else if c.borrowed & bit != 0 {
            if read {
                self.asm.emit(ldr_imm(3, r, SP, offsets::x(r)));
            }
            c.borrowed &= !bit;
        }
    }

    /// Puts every register back where a block boundary has it (see
    /// `RegCache`). X15..X17 stay loaded, now equal to their slots.
    fn sync(&mut self) {
        emit_sync(&mut self.asm, self.cache);
        self.cache.dirty = 0;
        self.cache.borrowed = 0;
    }

    /// One of X15..X17 not in `avoid`, free for a short sequence (stored
    /// first if the block changed it), or None if the instruction names all
    /// three.
    fn temp(&mut self, avoid: u32) -> Option<Reg> {
        let r = [16, 17, 15].into_iter().find(|&r| avoid & 1 << r == 0)?;
        let bit = 1 << r;
        if self.cache.dirty & bit != 0 {
            self.asm.emit(str_imm(3, r, SP, offsets::x(r)));
        }
        self.cache.loaded &= !bit;
        self.cache.dirty &= !bit;
        Some(r)
    }

    /// Three host registers for a fast path to compute with, none of them in
    /// `avoid` (which needs not name X18, X30 or SP): X16, X17 and X15 when
    /// free (a dirty one is stored first), then registers the block already
    /// borrowed, then new ones, chosen by what the rest of the block does with
    /// them: best one it overwrites before reading (it never needs reloading),
    /// then one it does not name again (reloaded once, as the block leaves).
    fn scratch(&mut self, avoid: u32) -> [Reg; 3] {
        let avoid = avoid | 1 << 18;
        let mut out = [0; 3];
        let mut n = 0;
        for r in [16, 17, 15] {
            if avoid & 1 << r == 0 {
                let bit = 1 << r;
                if self.cache.dirty & bit != 0 {
                    self.asm.emit(str_imm(3, r, SP, offsets::x(r)));
                }
                self.cache.loaded &= !bit;
                self.cache.dirty &= !bit;
                out[n] = r;
                n += 1;
            }
        }
        for &r in BORROWABLE.iter() {
            if n < 3 && avoid & 1 << r == 0 && self.cache.borrowed & 1 << r != 0 {
                out[n] = r;
                n += 1;
            }
        }
        if n < 3 {
            let (named, dead) = self.ahead.get(self.at).copied().unwrap_or((!0, 0));
            let mut fresh = 0u32;
            for rank in 0..3 {
                for &r in BORROWABLE.iter() {
                    let bit = 1 << r;
                    let fits = match rank {
                        0 => dead & bit != 0,
                        1 => named & bit == 0,
                        _ => true,
                    };
                    if n < 3 && fits && (avoid | self.cache.borrowed | fresh) & bit == 0 {
                        out[n] = r;
                        n += 1;
                        fresh |= bit;
                    }
                }
            }
            pairs(fresh, |r, both| {
                self.asm.emit(if both {
                    stp(r, r + 1, SP, offsets::x(r) as i32)
                } else {
                    str_imm(3, r, SP, offsets::x(r))
                })
            });
            self.cache.borrowed |= fresh;
        }
        out
    }

    /// Rd = a guest virtual address in or near this block's page (an ADR or
    /// ADRP result), computed from the run-time page; Rd may be virtual X18
    /// or the zero register.
    fn set_reg_va(&mut self, rd: u32, va: u64) {
        self.need(rd, false, true);
        let page = self.page << 12;
        let far =
            va.wrapping_sub(page) as i64 >= 1 << 24 || (va.wrapping_sub(page) as i64) <= -(1 << 24);
        match rd {
            31 => {}
            18 => {
                let v = self.temp(0).unwrap();
                let t = self.temp(1 << v).unwrap();
                emit::guest_va(&mut self.asm, v, va, page, t);
                self.asm.emit(str_imm(3, v, SP, offsets::x(18)));
            }
            r if far => {
                let t = self.temp(1 << r).unwrap();
                emit::guest_va(&mut self.asm, r, va, page, t);
            }
            r => emit::guest_va(&mut self.asm, r, va, page, 31),
        }
    }

    /// Rd = a constant, where Rd may be virtual X18 (or the zero register).
    fn set_reg_const(&mut self, rd: u32, value: u64) {
        self.need(rd, false, true);
        if rd == 18 {
            let v = self.temp(0).unwrap();
            self.asm.mov_imm(v, value);
            self.asm.emit(str_imm(3, v, SP, offsets::x(18)));
        } else if rd != 31 {
            self.asm.mov_imm(rd, value);
        }
    }

    /// Copies a word, rewriting it to run against a borrowed register when it
    /// names guest X18 in a general-register field.
    fn emit_copy(&mut self, word: u32) {
        let fields = gpr_field_positions(word);
        let (reads, writes) = copy_uses(word);
        for &p in fields {
            let r = (word >> p) & 0x1f;
            self.need(r, reads & 1 << r != 0, writes & 1 << r != 0);
        }
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
        let named = avoid[..cands.len()].iter().fold(0u32, |m, &r| m | 1 << r);
        let free = self.temp(named);
        let v = free.unwrap_or_else(|| Self::victim(&avoid[..cands.len()]));
        if free.is_none() {
            self.asm.emit(str_imm(3, v, SP, offsets::SCRATCH));
        }
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
        if free.is_none() {
            self.asm.emit(ldr_imm(3, v, SP, offsets::SCRATCH));
        }
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
        self.need(rn, true, false);
        self.need(rd, false, true);
        // Compute in Rd when it is a general register, else in a free one of
        // X15..X17. SP and X18 live in the state.
        let real = |r: u32| r != 31 && r != 18;
        let t = if real(rd) {
            rd
        } else {
            self.temp(1 << rn).unwrap()
        };
        let src = match rn {
            31 | 18 => {
                let slot = if rn == 31 {
                    offsets::SP
                } else {
                    offsets::x(18)
                };
                self.asm.emit(ldr_imm(3, t, SP, slot));
                t
            }
            r => r,
        };
        if imm12 != 0 || set_flags || src != t {
            self.asm.emit(match (sub, set_flags) {
                (false, false) => add_imm(sf, t, src, imm12, shift12),
                (true, false) => sub_imm(sf, t, src, imm12, shift12),
                (false, true) => adds_imm(sf, t, src, imm12, shift12),
                (true, true) => subs_imm(sf, t, src, imm12, shift12),
            });
        }
        if !discard && !real(rd) {
            let slot = if rd == 31 {
                offsets::SP
            } else {
                offsets::x(18)
            };
            self.asm.emit(str_imm(3, t, SP, slot));
        }
    }

    /// AND, ORR or EOR (immediate) into the guest's SP: the same instruction
    /// into a free register, stored to the state's SP. A 32-bit form's
    /// result is zero-extended, as a write of WSP is.
    fn emit_logical_sp(&mut self, word: u32) {
        let rn = (word >> 5) & 31;
        self.need(rn, true, false);
        let t = self.temp(1 << rn).unwrap();
        let mut w = (word & !31) | t;
        if rn == 18 {
            self.asm.emit(ldr_imm(3, t, SP, offsets::x(18)));
            w = (w & !(31 << 5)) | (t << 5);
        }
        self.asm.emit(w);
        self.asm.emit(str_imm(3, t, SP, offsets::SP));
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
        self.need(rn, true, false);
        self.need(rm, true, false);
        self.need(rd, false, true);
        // One of Rd and Rn is SP, so at least one of X15..X17 is free.
        let mask = |r: u32| if r < 31 { 1u32 << r } else { 0 };
        let t = self.temp(mask(rd) | mask(rn) | mask(rm)).unwrap();
        let src = match rn {
            31 => {
                self.asm.emit(ldr_imm(3, t, SP, offsets::SP));
                t
            }
            18 => {
                self.asm.emit(ldr_imm(3, t, SP, offsets::x(18)));
                t
            }
            r => r,
        };
        // Rm is never SP (register 31 there is the zero register), but may be
        // X18; then a second one of X15..X17 is free, since Rd or Rn is SP.
        let m = if rm == 18 {
            let u = self.temp(mask(rd) | mask(rn) | 1 << t).unwrap();
            self.asm.emit(ldr_imm(3, u, SP, offsets::x(18)));
            u
        } else {
            rm
        };
        let dst = if rd != 31 && rd != 18 { rd } else { t };
        let _ = sf; // extended-register add/sub here is always 64-bit (SP form)
        self.asm.emit(match (sub, set_flags) {
            (false, false) => add_ext(dst, src, m, option, amount),
            (true, false) => sub_ext(dst, src, m, option, amount),
            (false, true) => adds_ext(dst, src, m, option, amount),
            (true, true) => subs_ext(dst, src, m, option, amount),
        });
        if !discard {
            match rd {
                31 => self.asm.emit(str_imm(3, t, SP, offsets::SP)),
                18 => self.asm.emit(str_imm(3, t, SP, offsets::x(18))),
                _ => {}
            }
        }
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
        for &r in ops.as_slice() {
            self.need(r, true, true);
        }
        let regs = ops.callee_saved();
        if !self.ctx.fast_paths() || Self::needs_slow(&m) {
            self.emit_slow_inline(word, pc, regs.as_slice());
            return;
        }
        let [a, b, c] = self.scratch(ops.mask());
        let cache = self.cache;
        let tramp = self.asm.label();
        let slow = self.asm.label();
        let resume = self.asm.label();

        let va = self.emit_va(&m, literal, a, c);
        if m.align > 1 {
            self.asm.emit(and_imm(true, c, va, (m.align - 1) as u64));
            self.asm.cbnz(true, c, tramp);
        }
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
        let crosses = m.total > 1 && m.align < m.total;
        self.emit_tlb_check(m.total, crosses, tlb_index(el, kind), va, b, c, tramp);
        // The host address of the access; the host does it at that address
        // with no offset.
        self.asm.emit(add_reg(true, b, b, va, LSL, 0));
        // `a` is free unless it holds the address an exclusive still needs or
        // a writeback starts from.
        let free = if va == a
            && (matches!(m.kind, MemKind::LoadExclusive | MemKind::StoreExclusive)
                || (m.writeback && (m.rn == 31 || m.rn == 18)))
        {
            None
        } else {
            Some(a)
        };
        self.emit_access(&m, va, b, c, free);
        self.emit_writeback(&m, va, c, free);
        self.asm.bind(resume);
        self.slow.push(SlowSite {
            tramp,
            label: slow,
            resume,
            word,
            pc,
            cache,
            regs,
        });
    }

    /// The guest virtual address of the access's first byte: the base
    /// register itself when the access is at its value, or computed into `a`
    /// (with `c` as a temporary).
    fn emit_va(&mut self, m: &Mem, literal: Option<u64>, a: Reg, c: Reg) -> Reg {
        if let Some(addr) = literal {
            emit::guest_va(&mut self.asm, a, addr, self.page << 12, c);
            return a;
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
            return a;
        }
        if m.pre && m.offset != 0 {
            self.asm.emit_all(add_const(a, base, m.offset, c));
            return a;
        }
        base
    }

    /// The software-TLB lookup for `total` bytes at the address in `va`: on a
    /// hit `b` holds the delta to the host address; a miss branches to `slow`.
    /// An access that may cross a page looks up the page of its last byte and
    /// compares the entry's tag with the page of its first, so one that
    /// crosses never matches. Uses `b` and `c`; `va` is left as it is.
    #[allow(clippy::too_many_arguments)]
    fn emit_tlb_check(
        &mut self,
        total: u32,
        crosses: bool,
        tlb: usize,
        va: Reg,
        b: Reg,
        c: Reg,
        slow: Label,
    ) {
        if crosses {
            self.asm.emit(add_imm(true, c, va, total - 1, false));
            self.asm.emit(ubfx(b, c, TLB_SHIFT, TLB_BITS));
        } else {
            self.asm.emit(ubfx(b, va, TLB_SHIFT, TLB_BITS));
        }
        self.asm.emit(add_ext(b, SP, b, EXT_UXTX, 4));
        let off = offsets::tlb(tlb);
        debug_assert_eq!(off & 0xfff, 0);
        self.asm.emit(add_imm(true, b, b, off >> 12, true));
        self.asm.emit(ldp(c, b, b, 0));
        self.asm
            .emit(emit::sub_reg_shifted(c, c, va, LSR, TLB_SHIFT));
        self.asm.cbnz(true, c, slow);
    }

    /// The access itself, at the host address in `b`: the guest's own
    /// instruction with that as its base and no offset (the writeback is done
    /// apart). `c` is scratch for X18 data, and `free`, when there is one, for
    /// a store-exclusive.
    fn emit_access(&mut self, m: &Mem, va: Reg, b: Reg, c: Reg, free: Option<Reg>) {
        match m.kind {
            MemKind::Gpr | MemKind::Vec | MemKind::Pair | MemKind::VecPair => {
                let gpr = matches!(m.kind, MemKind::Gpr | MemKind::Pair);
                let pair = matches!(m.kind, MemKind::Pair | MemKind::VecPair);
                // Guest X18 as data goes through `c` (both of a pair's
                // registers, if both are X18: they are the same value).
                let x18 = (gpr && m.rt == 18) || (gpr && pair && m.rt2 == 18);
                let sub = |r: u32| if gpr && r == 18 { c } else { r };
                let (rt, rt2) = (sub(m.rt), sub(m.rt2));
                let w = if pair {
                    // LDP/STP (signed offset) of the same kind, at [b].
                    (m.word & 0xc440_0000) | 0x2900_0000 | rt2 << 10 | b << 5 | rt
                } else {
                    // LDUR/STUR of the same size and kind, at [b].
                    (m.word & 0xc4c0_0000) | 0x3800_0000 | b << 5 | rt
                };
                if x18 && !m.load {
                    self.asm.emit(ldr_imm(3, c, SP, offsets::x(18)));
                }
                self.asm.emit(w);
                if x18 && m.load {
                    self.asm.emit(str_imm(3, c, SP, offsets::x(18)));
                }
            }
            MemKind::VecStruct => {
                // The no-offset form at [b]; the writeback is done apart.
                self.asm
                    .emit((m.word & !(0x1f << 16) & !(1 << 23) & !(0x1f << 5)) | b << 5);
            }
            MemKind::LoadExclusive => {
                if m.acquire {
                    self.asm.emit(ldar(m.size, m.rt, b));
                } else {
                    self.asm.emit(ldr_imm(m.size, m.rt, b, 0));
                }
                self.asm.emit(str_imm(3, va, SP, offsets::MONITOR));
                self.asm.emit(str_imm(3, m.rt, SP, offsets::MONITOR_VALUE));
            }
            MemKind::StoreExclusive => {
                // Succeeds only if the reservation is for this address and
                // memory still holds what the load-exclusive read; the check
                // and the store are one compare-and-swap.
                let fail = self.asm.label();
                let done = self.asm.label();
                self.asm.emit(ldr_imm(3, c, SP, offsets::MONITOR));
                self.asm.emit(emit::sub_reg(true, c, c, va));
                self.asm.cbnz(true, c, fail);
                // From here the address is not needed: `va` may be reused.
                let copy = free.unwrap_or(va);
                self.asm.emit(ldr_imm(3, c, SP, offsets::MONITOR_VALUE));
                self.asm.emit(mov(copy, c));
                self.asm.emit(cas(m.size, false, m.release, c, m.rt, b));
                self.asm.emit(emit::sub_reg(true, c, c, copy));
                self.asm.cbnz(true, c, fail);
                self.asm.emit(movz(false, m.rs, 0, 0));
                self.asm.b(done);
                self.asm.bind(fail);
                self.asm.emit(movz(false, m.rs, 1, 0));
                self.asm.bind(done);
                self.asm.emit(str_imm(3, XZR, SP, offsets::MONITOR));
            }
            MemKind::LoadAcquire | MemKind::StoreRelease => {
                let rt = if m.rt == 18 { c } else { m.rt };
                if m.load {
                    self.asm.emit(ldar(m.size, rt, b));
                    if m.rt == 18 {
                        self.asm.emit(str_imm(3, c, SP, offsets::x(18)));
                    }
                } else {
                    if m.rt == 18 {
                        self.asm.emit(ldr_imm(3, c, SP, offsets::x(18)));
                    }
                    self.asm.emit(stlr(m.size, rt, b));
                }
            }
            MemKind::Cas | MemKind::CasPair | MemKind::Atomic => {
                // The host's own instruction on the host address.
                self.asm.emit((m.word & !(0x1f << 5)) | b << 5);
            }
        }
    }

    /// The base register's update of a pre- or post-indexed access: the base
    /// plus the offset or the register (pre-indexed, that is the address). For
    /// SP and X18 it starts from `va`, which `emit_memory_with` keeps for it
    /// (the address, or the base itself when post-indexed); `c` and `free`
    /// are scratch.
    fn emit_writeback(&mut self, m: &Mem, va: Reg, c: Reg, free: Option<Reg>) {
        if !m.writeback {
            return;
        }
        let slot = match m.rn {
            31 => offsets::SP,
            18 => offsets::x(18),
            rn => {
                match m.wb_reg {
                    Some(18) => {
                        self.asm.emit(ldr_imm(3, c, SP, offsets::x(18)));
                        self.asm.emit(add_reg(true, rn, rn, c, LSL, 0));
                    }
                    Some(rm) => self.asm.emit(add_reg(true, rn, rn, rm, LSL, 0)),
                    None => self.asm.emit_all(add_const(rn, rn, m.offset, c)),
                }
                return;
            }
        };
        if m.pre {
            self.asm.emit(str_imm(3, va, SP, slot));
            return;
        }
        let tmp = free.unwrap_or(c);
        match m.wb_reg {
            Some(18) => {
                self.asm.emit(ldr_imm(3, c, SP, offsets::x(18)));
                self.asm.emit(add_reg(true, c, va, c, LSL, 0));
            }
            Some(rm) => self.asm.emit(add_reg(true, c, va, rm, LSL, 0)),
            None => self.asm.emit_all(add_const(c, va, m.offset, tmp)),
        }
        self.asm.emit(str_imm(3, c, SP, slot));
    }

    fn emit_dc_zva(&mut self, word: u32, rt: u32, pc: u64) {
        self.need(rt, true, false);
        let mut regs = Regs::default();
        if (19..=29).contains(&rt) {
            regs.push(rt);
        }
        if !self.ctx.fast_paths() {
            self.emit_slow_inline(word, pc, regs.as_slice());
            return;
        }
        let avoid = if rt < 31 { 1 << rt } else { 0 };
        let [a, b, c] = self.scratch(avoid);
        let cache = self.cache;
        let tramp = self.asm.label();
        let slow = self.asm.label();
        let resume = self.asm.label();
        match rt {
            31 => self.asm.emit(movz(true, a, 0, 0)),
            18 => {
                self.asm.emit(ldr_imm(3, a, SP, offsets::x(18)));
                self.asm.emit(and_imm(true, a, a, !63));
            }
            r => self.asm.emit(and_imm(true, a, r, !63)),
        }
        // A 64-byte aligned block never crosses a page.
        let tlb = tlb_index(self.ctx.el, TlbKind::Store);
        self.emit_tlb_check(64, false, tlb, a, b, c, tramp);
        self.asm.emit(add_reg(true, b, b, a, LSL, 0));
        for off in [0, 16, 32, 48] {
            self.asm.emit(stp(XZR, XZR, b, off));
        }
        self.asm.bind(resume);
        self.slow.push(SlowSite {
            tramp,
            label: slow,
            resume,
            word,
            pc,
            cache,
            regs,
        });
    }

    /// The call into the memory slow path, with every register where a block
    /// boundary has it: stores the operand registers the Rust side reads from
    /// among X19..X29 (it reads the rest from the state the stub fills), and
    /// X30; then the BL with the site's literal in X17; then reloads those
    /// operands. X15..X17 are left in their slots.
    fn emit_slow_call(&mut self, word: u32, pc: u64, regs: &[Reg]) {
        for &r in regs {
            self.asm.emit(str_imm(3, r, SP, offsets::x(r)));
        }
        self.asm.emit(str_imm(3, 30, SP, offsets::x(30)));
        let literal = self.asm.label();
        let off = pc.wrapping_sub(self.page << 12);
        self.literals.push((literal, word, off, self.asm.is_cold()));
        self.asm.adr(17, literal);
        self.asm.bl_abs(self.ctx.stubs.mem);
        for &r in regs {
            self.asm.emit(ldr_imm(3, r, SP, offsets::x(r)));
        }
    }

    fn emit_slow_inline(&mut self, word: u32, pc: u64, regs: &[Reg]) {
        self.sync();
        self.emit_slow_call(word, pc, regs);
        self.cache.loaded = 0;
    }

    /// The out-of-line slow paths of the block's fast loads and stores, in
    /// the cold section, reached through branches after the block's code;
    /// then the literals of every slow-path call, in the section of the call.
    fn emit_slow_sites(&mut self) {
        let sites = std::mem::take(&mut self.slow);
        for site in &sites {
            self.asm.bind(site.tramp);
            self.asm.b(site.label);
        }
        self.asm.set_cold(true);
        for site in sites {
            self.asm.bind(site.label);
            emit_sync(&mut self.asm, site.cache);
            self.emit_slow_call(site.word, site.pc, site.regs.as_slice());
            // The access's operands among X15..X17 are loaded where it
            // resumes (the slow path may have written one).
            pairs(site.cache.loaded, |r, both| {
                self.asm.emit(if both {
                    ldp(r, r + 1, SP, offsets::x(r) as i32)
                } else {
                    ldr_imm(3, r, SP, offsets::x(r))
                })
            });
            self.asm.b(site.resume);
        }
        for cold in [true, false] {
            self.asm.set_cold(cold);
            for &(label, word, off, _) in self.literals.iter().filter(|l| l.3 == cold) {
                self.asm.bind(label);
                self.asm.emit_all([word, 0, off as u32, (off >> 32) as u32]);
            }
        }
        self.literals.clear();
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
        let (terminated, cache) = (self.terminated, self.cache);
        self.exit(Exit::Branch as u32, Next::Const(next), ExitInfo::default());
        (self.terminated, self.cache) = (terminated, cache);
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
        self.sync();
        if let Some(ret) = link {
            emit::guest_va(&mut self.asm, 30, ret, self.page << 12, 31);
        }
        // With every register back in place, X15..X17 are scratch.
        if target >> 12 == self.page && target & 3 == 0 {
            // Same page: a branch the dispatcher can patch to the target block.
            let exit = self.asm.label();
            let site = self.asm.label();
            if target <= pc {
                // Backward: a chained loop must still let the CPU leave.
                let kicked = self.asm.label();
                self.asm.emit(ldr_imm(2, 16, SP, offsets::EXIT_REQUEST));
                self.asm.cbnz(false, 16, kicked);
                self.asm.bind(site);
                self.asm.b(exit);
                self.asm.bind(kicked);
                self.asm.b(exit);
            } else {
                self.asm.bind(site);
                self.asm.b(exit);
            }
            // The exit runs until the site is patched: it goes in the cold
            // section.
            self.asm.set_cold(true);
            self.asm.bind(exit);
            self.exit(
                Exit::Branch as u32,
                Next::Const(target),
                ExitInfo {
                    chain_site: Some(site),
                    ..Default::default()
                },
            );
            self.asm.set_cold(false);
            return;
        }
        // Another page: through the jump cache.
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
        self.need(rn, true, false);
        self.sync();
        match rn {
            JUMP_REG => {}
            15 | 17 => self.asm.emit(mov(JUMP_REG, rn)),
            18 => self.asm.emit(ldr_imm(3, JUMP_REG, SP, offsets::x(rn))),
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
        let rt = match kind {
            CondKind::Cbz { rt, .. }
            | CondKind::Cbnz { rt, .. }
            | CondKind::Tbz { rt, .. }
            | CondKind::Tbnz { rt, .. } => rt,
            CondKind::Cond => 31,
        };
        // Both ways leave the block, so every register goes back first (the
        // stores and loads leave the flags as they are). Then X16 is scratch,
        // for guest X18, which is virtual.
        self.need(rt, true, false);
        self.sync();
        let r = if rt == 18 {
            self.asm.emit(ldr_imm(3, 16, SP, offsets::x(18)));
            self.cache.loaded &= !(1 << 16);
            16
        } else {
            rt
        };
        match kind {
            CondKind::Cond => self.asm.b_cond(word & 0xf, taken_label),
            CondKind::Cbz { sf, .. } => self.asm.cbz(sf, r, taken_label),
            CondKind::Cbnz { sf, .. } => self.asm.cbnz(sf, r, taken_label),
            CondKind::Tbz { bit, .. } => self.asm.tbz(r, bit, taken_label),
            CondKind::Tbnz { bit, .. } => self.asm.tbnz(r, bit, taken_label),
        }
        self.goto_const(pc.wrapping_add(4), None, pc);
        self.asm.bind(taken_label);
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
            self.need(rt, !read, read);
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
        self.need(rt, false, true);
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
        self.need(rt, true, false);
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
        self.need(rt, true, false);
        self.asm.emit(stp(16, 17, SP, offsets::scratch(0) as i32));
        // The new bits first: Rt may be X16 or X17.
        match rt {
            31 => self.asm.emit(movz(true, 17, 0, 0)),
            18 => {
                self.asm.emit(ldr_imm(3, 17, SP, offsets::x(18)));
                self.asm.emit(and_imm(true, 17, 17, 0x3c0));
            }
            r => self.asm.emit(and_imm(true, 17, r, 0x3c0)),
        }
        self.asm.emit(ldr_imm(3, 16, SP, offsets::PSTATE));
        self.asm.emit(and_imm(true, 16, 16, !0x3c0));
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
        let (terminated, cache) = (self.terminated, self.cache);
        self.exit(
            Exit::Branch as u32,
            Next::Const(pc.wrapping_add(4)),
            ExitInfo::default(),
        );
        (self.terminated, self.cache) = (terminated, cache);
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

    fn block() -> Block {
        let ctx = Ctx {
            el: 1,
            fast: true,
            debug: false,
            stubs: Stubs::default(),
            mpidr: 0,
        };
        Block::new(ctx, 0x4000_0000)
    }

    #[test]
    fn scratch_takes_x15_to_x17_then_what_the_block_overwrites() {
        let mut t = block();
        assert_eq!(t.scratch(0), [16, 17, 15]);
        assert_eq!(t.asm.here(), 0, "nothing to store");
        // X16 loaded and changed: stored before it is taken.
        t.need(16, false, true);
        assert_eq!(t.scratch(1 << 17), [16, 15, 14]);
        assert_eq!(t.cache.loaded, 0);
        // With X15..X17 operands: a register the rest of the block writes
        // before reading, then one it does not name, then any; stored in
        // pairs where they neighbour.
        let mut t = block();
        t.ahead = vec![(1 << 14 | 1 << 12, 1 << 12)];
        let start = t.asm.here();
        assert_eq!(t.scratch(HOME | 1 << 3), [12, 13, 11]);
        assert_eq!(t.cache.borrowed, 1 << 11 | 1 << 12 | 1 << 13);
        assert_eq!(t.asm.here() - start, 8, "stp x11, x12 and str x13");
        // Borrowed registers are reused, and reloaded when named.
        assert_eq!(t.scratch(HOME), [13, 12, 11]);
        t.need(12, true, false);
        t.need(11, false, true);
        assert_eq!(t.cache.borrowed, 1 << 13);
        t.sync();
        assert_eq!(
            t.cache,
            RegCache {
                loaded: 0,
                dirty: 0,
                borrowed: 0
            }
        );
    }

    #[test]
    fn copies_read_and_write_what_they_name() {
        let x = |r: u32| 1u32 << r;
        // add x1, x2, x3 ; add x1, x1, #1 ; movk x5, #1 ; bfi x1, x2, #4, #8
        assert_eq!(copy_uses(0x8b030041), (x(2) | x(3), x(1)));
        assert_eq!(copy_uses(0x91000421), (x(1), x(1)));
        assert_eq!(copy_uses(0xf2800025), (x(5), x(5)));
        assert_eq!(copy_uses(0xb37c1c41), (x(1) | x(2), x(1)));
        // csel x0, x1, x2, eq ; ccmp x1, x2, #0, eq ; madd x0, x1, x2, x3
        assert_eq!(copy_uses(0x9a820020), (x(1) | x(2), x(0)));
        assert_eq!(copy_uses(0xfa420020), (x(1) | x(2), 0));
        assert_eq!(copy_uses(0x9b020c20), (x(1) | x(2) | x(3), x(0)));
        // rbit x0, x1 ; pacia x0, x1 (reads its destination)
        assert_eq!(copy_uses(0xdac00020), (x(1), x(0)));
        assert_eq!(copy_uses(0xdac10020), (x(0) | x(1), x(0)));
        // lslv x16, x24, x16
        assert_eq!(copy_uses(0x9ad02310), (x(16) | x(24), x(16)));
        // mrs x18, nzcv ; msr fpcr, x18 (a source, taken as written too) ;
        // fmov x18, d3 ; fmov d3, x18
        assert_eq!(copy_uses(0xd53b4212), (0, x(18)));
        assert_eq!(copy_uses(0xd51b4412), (x(18), x(18)));
        assert_eq!(copy_uses(0x9e660072), (0, x(18)));
        assert_eq!(copy_uses(0x9e670243), (x(18), 0));
        // cmp x1, x2 (subs xzr): no destination.
        assert_eq!(copy_uses(0xeb02003f), (x(1) | x(2), 0));
    }
}
