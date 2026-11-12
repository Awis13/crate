// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The host-code sequences the translator emits that are the same whatever the
//! guest instruction is: a block's entry, leaving a block, the shared stubs
//! every block of a code region branches to (saving the guest registers on the
//! way out, the memory slow path, the jump-cache lookup), and the exit prefix.
//!
//! Throughout a block the host stack pointer is the address of the `CpuState`,
//! so a guest register reaches its spill slot with `[SP, #off]` and SP itself
//! is read or written with `ADD Xn, SP, #0`. See `state` for why.

use crate::asm::*;
use crate::state::{Exit, TLB_SHIFT, offsets};

/// A host register the exit prefix and the save stub use as scratch before
/// the matching guest register is live or after it is saved.
const SCRATCH0: Reg = 0;
const SCRATCH1: Reg = 1;

/// The host addresses of the stubs shared by every block of one code region.
#[derive(Clone, Copy, Default, Debug)]
pub struct Stubs {
    /// Saves the guest registers a block leaves live and returns through the
    /// epilogue (see `save_stub`).
    pub save: u64,
    /// The memory slow path (see `mem_stub`).
    pub mem: u64,
    /// The jump-cache lookup for EL0 and EL1 code (see `jc_stubs`): the
    /// first of its `JC_COPIES` copies, `jc_stride` bytes apart.
    pub jc: [u64; 2],
    pub jc_stride: [u64; 2],
    /// The same for returns, which jump with RET.
    pub jc_ret: [u64; 2],
    /// Where a jump-cache miss goes; also the target of an empty entry.
    pub jc_miss: [u64; 2],
    /// The lookup a chained branch to another page goes to when its guard
    /// fails, with the address of its literal in X15 (see `jc_stubs`).
    pub jc_chain: [u64; 2],
}

/// How many copies of the jump-cache lookup each exception level has. A site
/// branches to the copy its guest address picks, so that the CPU predicts the
/// indirect branch each copy ends with from the few sites that share it,
/// rather than one branch from all of them.
pub const JC_COPIES: usize = 32;

impl Stubs {
    /// The copy of the jump-cache lookup for code at `el` that the site at
    /// guest `pc` branches to.
    pub fn jc_for(&self, el: usize, pc: u64) -> u64 {
        let copy = ((pc >> 2) ^ (pc >> 9)) as usize % JC_COPIES;
        self.jc[el] + copy as u64 * self.jc_stride[el]
    }
}

/// The host register a jump into a block (from the jump cache, or a
/// dispatcher's entry) carries the target in. Guest X15..X17 are in their
/// slots of the state wherever one block hands over to another, so a block's
/// entry has nothing to reload, and its chained entry is its first word.
pub const JUMP_REG: Reg = 16;

/// Emits the shared stub that saves the guest registers a block leaves live
/// (X1..X29 but X15..X18, NZCV and the SIMD registers; FPCR and FPSR stay in
/// the host's) and returns through the epilogue. The prefix each exit emits
/// has already saved X0 and X30 and set the exit code, so this stub is the
/// same for every exit.
pub fn save_stub(asm: &mut Asm) {
    asm.emit(str_imm(3, 1, SP, offsets::x(1)));
    save_x2_to_x29(asm);
    asm.emit(mrs(SCRATCH1, NZCV));
    asm.emit(str_imm(3, SCRATCH1, SP, offsets::NZCV));
    save_vectors(asm);
    asm.emit(ldr_imm(3, SCRATCH1, SP, offsets::EPILOGUE));
    asm.emit(br(SCRATCH1));
}

/// Stores X2..X29 except X15..X18, whose slots already hold them (guest X18
/// is virtual: the host's x18 is reserved by the OS and holds garbage).
fn save_x2_to_x29(asm: &mut Asm) {
    for n in [2, 4, 6, 8, 10, 12] {
        asm.emit(stp(n, n + 1, SP, offsets::x(n) as i32));
    }
    asm.emit(str_imm(3, 14, SP, offsets::x(14)));
    save_x19_to_x29(asm);
}

/// Stores X19..X29.
fn save_x19_to_x29(asm: &mut Asm) {
    for n in [19, 21, 23, 25, 27] {
        asm.emit(stp(n, n + 1, SP, offsets::x(n) as i32));
    }
    asm.emit(str_imm(3, 29, SP, offsets::x(29)));
}

fn save_vectors(asm: &mut Asm) {
    for q in (0..32).step_by(2) {
        asm.emit(stp_q(q, q + 1, SP, offsets::v(q) as i32));
    }
}

fn load_vectors(asm: &mut Asm) {
    for q in (0..32).step_by(2) {
        asm.emit(ldp_q(q, q + 1, SP, offsets::v(q) as i32));
    }
}

/// The memory slow path, shared by every load and store of a region. A site
/// reaches it with `BL` after storing guest X30 (and any operand registers it
/// names among X19..X29) to the state and putting the address of its literal
/// in X17: the instruction word, and the instruction's offset in its page
/// (the guest PC is the block's run-time page plus that).
///
/// The stub saves what a call into Rust may clobber -- by the procedure call
/// standard X0..X14 (X15..X17 are in the state already), the flags, and the
/// SIMD registers (all of V0..V7 and V16..V31, and the upper halves of
/// V8..V15) -- but not X19..X29, which the callee preserves. It calls
/// `helper(cpu, site)`, which performs the access on the state's copies of
/// the registers; on success it reloads the same set and returns to the
/// site with `RET`, so the CPU's return prediction stays paired with the
/// site's `BL`. The site then reloads whichever of X15..X17 and X19..X29 the
/// instruction names. On a fault the helper has recorded the exception, so
/// the stub saves the rest of the registers and leaves through the epilogue.
pub fn mem_stub(asm: &mut Asm, helper: u64) {
    asm.emit(str_imm(3, 17, SP, offsets::SLOW_SITE));
    asm.emit(str_imm(3, 30, SP, offsets::SLOW_RET));
    for n in (0..14).step_by(2) {
        asm.emit(stp(n, n + 1, SP, offsets::x(n) as i32));
    }
    asm.emit(str_imm(3, 14, SP, offsets::x(14)));
    asm.emit(mrs(0, NZCV));
    asm.emit(str_imm(3, 0, SP, offsets::NZCV));
    save_vectors(asm);
    call_rust(asm, helper, true);
    let fault = asm.label();
    asm.cbnz(true, 0, fault);
    load_vectors(asm);
    asm.emit(ldr_imm(3, 0, SP, offsets::NZCV));
    asm.emit(msr(NZCV, 0));
    for n in (0..14).step_by(2) {
        asm.emit(ldp(n, n + 1, SP, offsets::x(n) as i32));
    }
    asm.emit(ldr_imm(3, 14, SP, offsets::x(14)));
    asm.emit(ldr_imm(3, 16, SP, offsets::SLOW_RET));
    asm.emit(ldr_imm(3, 30, SP, offsets::x(30)));
    asm.emit(ret_reg(16));

    asm.bind(fault);
    // X0..X14, X30, the flags and the vectors are in the state already, and
    // X15..X17 always were; X19..X29 still hold the guest's values (the call
    // preserved them).
    save_x19_to_x29(asm);
    asm.emit(ldr_imm(3, 0, SP, offsets::EPILOGUE));
    asm.emit(br(0));
}

/// Calls `helper(cpu, site, state)` on the real host stack, with the site in
/// X1 taken from `SLOW_SITE` when `with_site`, and comes back with SP the
/// state block and the result in X0. Uses X0..X2, X9 and X16.
fn call_rust(asm: &mut Asm, helper: u64, with_site: bool) {
    asm.emit(ldr_imm(3, 0, SP, offsets::CPU_PTR));
    if with_site {
        asm.emit(ldr_imm(3, 1, SP, offsets::SLOW_SITE));
    }
    asm.emit(add_imm(true, 2, SP, 0, false)); // x2 = &state
    asm.emit(ldr_imm(3, 9, SP, offsets::HOST_SP));
    asm.emit(mov_sp(SP, 9)); // SP = host stack
    asm.emit(str_imm_pre(3, 2, SP, -16)); // push &state
    // The helper is in the main binary, too far for a relative BL, so call it
    // through a register.
    asm.mov_imm(16, helper);
    asm.emit(blr(16));
    asm.emit(ldr_imm_post(3, 2, SP, 16)); // x2 = &state
    asm.emit(mov_sp(SP, 2)); // SP = &state
}

/// The jump-cache lookups for code at `el`, shared by every indirect branch
/// and every direct branch to another page. A site reaches one with `B` with
/// guest X15..X17 in their slots and the target guest PC in X16. The cache is
/// direct-mapped by PC; a hit sets the page of the target (`PAGE_VA`) and
/// branches to the target block's entry, and a miss -- or a pending exit
/// request, so a loop through indirect branches cannot keep the CPU from its
/// dispatcher -- leaves with a Branch exit to the target.
///
/// There are `copies` copies of the lookup, `stride` bytes apart, which jump
/// with BR, and one more for returns (`ret`), which jumps with RET, so the
/// CPU predicts it from its own return stack: a guest call is a host BL whose
/// return address is the translation of the instruction after the call (see
/// `translate::Block::emit_call`). And one for chained branches to another
/// page whose guard failed (`chain`): it leaves the address of the branch's
/// literal (in X15) in `pending_site`, so the dispatcher chains the branch to
/// what the lookup finds -- recorded in `pending_host` and `pending_key` on a
/// hit, or the block it translates after a miss.
pub fn jc_stubs(asm: &mut Asm, el: u32, save: u64, stats: bool, copies: usize) -> JcStubs {
    let miss = asm.label();
    let kicked = asm.label();
    let leave = asm.label();
    // A miss, from a lookup or from an empty entry: the key is in X17 (for
    // EL0 code the PC and the address space's context number in the top
    // bits) and, for EL1 code, the PC still in X16.
    asm.bind(miss);
    let miss_at = asm.here();
    if el == 0 {
        asm.emit(ldr_imm(3, JUMP_REG, SP, offsets::jc_key(0)));
        asm.emit(eor_reg(true, JUMP_REG, JUMP_REG, 17, LSL, 0));
    }
    asm.emit(str_imm(3, JUMP_REG, SP, offsets::NEXT_PC));
    asm.emit(ldr_imm(3, 15, SP, offsets::JC_MISSES));
    asm.emit(add_imm(true, 15, 15, 1, false));
    asm.emit(str_imm(3, 15, SP, offsets::JC_MISSES));
    asm.b(leave);
    asm.bind(kicked);
    asm.emit(str_imm(3, JUMP_REG, SP, offsets::NEXT_PC));
    asm.bind(leave);
    // The exit prefix, with the next PC already in place.
    asm.emit(str_imm(3, 0, SP, offsets::x(0)));
    asm.emit(str_imm(3, 30, SP, offsets::x(30)));
    asm.emit(movz(true, 0, Exit::Branch as u32, 0));
    asm.emit(str_imm(2, 0, SP, offsets::EXIT));
    asm.b_abs(save);
    // A chain's lookup first records its site for the dispatcher, and on a
    // hit the entry and key it found.
    let lookup = |asm: &mut Asm, ret: bool, chain: bool| {
        if chain {
            asm.emit(stp(15, XZR, SP, offsets::PENDING_SITE as i32));
        }
        if stats {
            asm.emit(ldr_imm(3, 17, SP, offsets::JC_LOOKUPS));
            asm.emit(add_imm(true, 17, 17, 1, false));
            asm.emit(str_imm(3, 17, SP, offsets::JC_LOOKUPS));
        }
        asm.emit(ldr_imm(2, 17, SP, offsets::EXIT_REQUEST));
        asm.cbnz(false, 17, kicked);
        // The target block computes guest addresses from its page; a miss
        // leaves for the dispatcher, which sets it again.
        asm.emit(and_imm(true, 15, JUMP_REG, !0xfff));
        asm.emit(str_imm(3, 15, SP, offsets::PAGE_VA));
        // The key: the PC, and for EL0 code the address space's context
        // number in the top bits.
        let key = if el == 0 {
            asm.emit(ldr_imm(3, 17, SP, offsets::jc_key(0)));
            asm.emit(eor_reg(true, 17, 17, JUMP_REG, LSL, 0));
            17
        } else {
            JUMP_REG
        };
        asm.emit(eor_reg(true, 15, key, key, LSR, crate::state::JC_FOLD));
        asm.emit(ubfx(15, 15, 2, crate::state::JC_BITS));
        asm.emit(add_ext(15, SP, 15, EXT_UXTX, 4));
        let off = offsets::jc(el);
        debug_assert_eq!(off & 0xfff, 0);
        asm.emit(add_imm(true, 15, 15, off >> 12, true));
        // The tag goes where the key is not.
        let tag = if key == 17 { JUMP_REG } else { 17 };
        asm.emit(ldp(tag, 15, 15, 0));
        asm.emit(sub_reg(true, tag, tag, key));
        asm.cbnz(true, tag, miss);
        if chain {
            asm.emit(stp(15, key, SP, offsets::PENDING_HOST as i32));
        }
        asm.emit(if ret { ret_reg(15) } else { br(15) });
    };
    let align = |asm: &mut Asm, to: u32| {
        while asm.here() < to {
            asm.emit(brk(0));
        }
    };
    align(asm, asm.here().next_multiple_of(16));
    let chain = asm.here();
    lookup(asm, false, true);
    align(asm, asm.here().next_multiple_of(16));
    let ret = asm.here();
    lookup(asm, true, false);
    align(asm, asm.here().next_multiple_of(16));
    let first = asm.here();
    lookup(asm, false, false);
    let stride = (asm.here() - first).next_multiple_of(16);
    for copy in 1..copies as u32 {
        align(asm, first + copy * stride);
        lookup(asm, false, false);
    }
    JcStubs {
        first,
        stride,
        ret,
        miss: miss_at,
        chain,
    }
}

/// Where `jc_stubs` put each part, as offsets from its start.
pub struct JcStubs {
    pub first: u32,
    pub stride: u32,
    pub ret: u32,
    pub miss: u32,
    pub chain: u32,
}

/// How a block exit knows the next guest PC.
#[derive(Clone, Copy)]
pub enum Next {
    /// A constant address known at translation time.
    Const(u64),
    /// The value of a guest register (an indirect branch).
    Reg(u32),
    /// Left as whatever the dispatcher already set.
    None,
}

/// The fields an exit prefix sets besides the exit code and next PC.
#[derive(Clone, Copy, Default)]
pub struct ExitInfo {
    /// A link register value to leave in guest X30 (BL/BLR).
    pub link: Option<u64>,
    /// The faulting or current guest PC to record in `pc` (exceptions, calls).
    pub pc: Option<u64>,
    /// For a Sysreg exit: (encoding, Rt, is-read).
    pub sysreg: Option<(u16, u32, bool)>,
    /// For an exception/call: the syndrome to record.
    pub syndrome: Option<u64>,
    /// For a direct branch to the same page: the branch to patch once the
    /// target is translated.
    pub chain_site: Option<Label>,
}

/// Emits the exit prefix: saves X0 and X30, sets the exit code, the next PC
/// and any extra fields, then branches to the save stub. Uses only X0 and
/// X30 (once saved) as scratch, so the other live guest registers are still
/// in their host registers for the stub to save.
///
/// Guest addresses (the next PC, a link value, the PC) are given as virtual
/// addresses in the page whose translation-time base is `page`; the code
/// computes them from the block's run-time page (`PAGE_VA`), so one
/// translation serves every virtual address its physical page is mapped at.
pub fn exit(asm: &mut Asm, save: u64, page: u64, code: u32, next: Next, info: ExitInfo) {
    // Save guest X0 and X30; from here both are scratch (a link value
    // overwrites the X30 slot below).
    asm.emit(str_imm(3, 0, SP, offsets::x(0)));
    asm.emit(str_imm(3, 30, SP, offsets::x(30)));
    match next {
        Next::Const(pc) => {
            guest_va(asm, SCRATCH0, pc, page, 30);
            asm.emit(str_imm(3, SCRATCH0, SP, offsets::NEXT_PC));
        }
        Next::Reg(rn) => {
            // X0 and X30 are saved, X15..X17 are in their slots, and X18 is
            // virtual (it never lives in the host's reserved x18): those are
            // read from the state block.
            if rn == 0 || (15..=18).contains(&rn) || rn == 30 {
                asm.emit(ldr_imm(3, SCRATCH0, SP, offsets::x(rn)));
            } else if rn == 31 {
                asm.emit(movz(true, SCRATCH0, 0, 0));
            } else {
                asm.emit(mov(SCRATCH0, rn));
            }
            asm.emit(str_imm(3, SCRATCH0, SP, offsets::NEXT_PC));
        }
        Next::None => {}
    }
    if let Some(ret) = info.link {
        guest_va(asm, SCRATCH0, ret, page, 30);
        asm.emit(str_imm(3, SCRATCH0, SP, offsets::x(30)));
    }
    if let Some(pc) = info.pc {
        guest_va(asm, SCRATCH0, pc, page, 30);
        asm.emit(str_imm(3, SCRATCH0, SP, offsets::PC));
    }
    if let Some((reg, rt, read)) = info.sysreg {
        asm.mov_imm(SCRATCH0, reg as u64);
        asm.emit(str_imm(2, SCRATCH0, SP, offsets::SYSREG));
        asm.mov_imm(SCRATCH0, rt as u64);
        asm.emit(str_imm(2, SCRATCH0, SP, offsets::SYSREG_RT));
        asm.mov_imm(SCRATCH0, read as u64);
        asm.emit(str_imm(2, SCRATCH0, SP, offsets::SYSREG_READ));
    }
    if let Some(syndrome) = info.syndrome {
        asm.mov_imm(SCRATCH0, syndrome);
        asm.emit(str_imm(3, SCRATCH0, SP, offsets::EXCEPTION_SYNDROME));
    }
    if let Some(site) = info.chain_site {
        // An address literal: the site may be in the other section.
        let literal = asm.address_literal(site);
        asm.ldr_literal(SCRATCH0, literal);
        asm.emit(str_imm(3, SCRATCH0, SP, offsets::CHAIN_SITE));
    }
    asm.emit(movz(true, SCRATCH0, code, 0));
    asm.emit(str_imm(2, SCRATCH0, SP, offsets::EXIT));
    asm.b_abs(save);
}

/// Rd = the guest virtual address `va`, which translation found at a page
/// whose base was `page`: the block's run-time page base plus the same
/// offset. `tmp` (distinct from Rd) is used only for offsets of 16 MiB or
/// more.
pub fn guest_va(asm: &mut Asm, rd: Reg, va: u64, page: u64, tmp: Reg) {
    asm.emit(ldr_imm(3, rd, SP, offsets::PAGE_VA));
    let delta = va.wrapping_sub(page) as i64;
    if delta != 0 {
        asm.emit_all(add_const(rd, rd, delta, tmp));
    }
}

/// The host instruction that makes `rd` the page number of `rn` (the tag a
/// software-TLB entry holds).
pub fn page_number(rd: Reg, rn: Reg) -> u32 {
    lsr_imm(rd, rn, TLB_SHIFT)
}

// Small addressing-mode helpers the sequences above need and that the public
// `asm` module does not expose as pre/post-index forms.

/// STR (pre-index): `[Rn, #imm]!`.
pub fn str_imm_pre(size: u32, rt: Reg, rn: Reg, imm: i32) -> u32 {
    size << 30 | 0x3800_0c00 | ((imm as u32 & 0x1ff) << 12) | rn << 5 | rt
}

/// LDR (post-index): `[Rn], #imm`.
pub fn ldr_imm_post(size: u32, rt: Reg, rn: Reg, imm: i32) -> u32 {
    size << 30 | 0x3840_0400 | ((imm as u32 & 0x1ff) << 12) | rn << 5 | rt
}

/// SUB (shifted register), no shift.
pub fn sub_reg(sf: bool, rd: Reg, rn: Reg, rm: Reg) -> u32 {
    (sf as u32) << 31 | 0x4b00_0000 | rm << 16 | rn << 5 | rd
}

/// SUB (shifted register), 64-bit: Rd = Rn - (Rm shifted by `amount`).
pub fn sub_reg_shifted(rd: Reg, rn: Reg, rm: Reg, shift: u32, amount: u32) -> u32 {
    0xcb00_0000 | shift << 22 | rm << 16 | amount << 10 | rn << 5 | rd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_encodings() {
        // str x2, [sp, #-16]!  and  ldr x2, [sp], #16
        assert_eq!(str_imm_pre(3, 2, SP, -16), 0xf81f0fe2);
        assert_eq!(ldr_imm_post(3, 2, SP, 16), 0xf84107e2);
        // sub x15, x15, x16
        assert_eq!(sub_reg(true, 15, 15, 16), 0xcb1001ef);
        // sub x17, x17, x16, lsr #12
        assert_eq!(sub_reg_shifted(17, 17, 16, LSR, 12), 0xcb503231);
    }
}
