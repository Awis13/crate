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
    /// The jump-cache lookup for EL0 and EL1 code (see `jc_stub`).
    pub jc: [u64; 2],
    /// Where a jump-cache miss goes; also the target of an empty entry.
    pub jc_miss: [u64; 2],
}

/// The registers an indirect jump into a block borrows, which the block's
/// entry restores from their home slots: X16 holds the jump target and X15 is
/// the lookup's scratch. The jump cache restores X17 itself.
pub const JUMP_REG: Reg = 16;

/// Emits a block's entry, the word every jump into the block from the
/// dispatcher or the jump cache lands on: it reloads the two guest registers
/// the jump borrowed. A block reached by a chained direct branch starts at the
/// next word, with every guest register already live.
pub fn block_entry(asm: &mut Asm) {
    asm.emit(ldp(15, 16, SP, offsets::x(15) as i32));
}

/// The byte offset of a block's chained entry from its start.
pub const CHAIN_ENTRY: u64 = 4;

/// Emits the shared stub that saves the guest registers a block leaves live
/// (X1..X29 but X18, NZCV, FPCR, FPSR and the SIMD registers) and returns
/// through the epilogue. The prefix each exit emits has already saved X0 and
/// X30 and set the exit code, so this stub is the same for every exit.
pub fn save_stub(asm: &mut Asm) {
    asm.emit(str_imm(3, 1, SP, offsets::x(1)));
    save_x2_to_x29(asm);
    save_flags_and_vectors(asm, SCRATCH1);
    asm.emit(ldr_imm(3, SCRATCH1, SP, offsets::EPILOGUE));
    asm.emit(br(SCRATCH1));
}

/// Stores X2..X29 except X18 (guest X18 is virtual: the host's x18 is
/// reserved by the OS and holds garbage, and the slot holds the real value).
fn save_x2_to_x29(asm: &mut Asm) {
    let mut n = 2;
    while n <= 28 {
        if n == 18 {
            asm.emit(str_imm(3, 19, SP, offsets::x(19)));
        } else {
            asm.emit(stp(n, n + 1, SP, offsets::x(n) as i32));
        }
        n += 2;
    }
}

fn save_flags_and_vectors(asm: &mut Asm, tmp: Reg) {
    asm.emit(mrs(tmp, NZCV));
    asm.emit(str_imm(3, tmp, SP, offsets::NZCV));
    asm.emit(mrs(tmp, FPCR));
    asm.emit(str_imm(3, tmp, SP, offsets::FPCR));
    asm.emit(mrs(tmp, FPSR));
    asm.emit(str_imm(3, tmp, SP, offsets::FPSR));
    save_vectors(asm);
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
/// names among X19..X29) to the state; the BL's return address is the site's
/// literal: the instruction word, the offset from the literal to where to
/// return, and the guest PC.
///
/// The stub saves what a call into Rust may clobber -- by the procedure call
/// standard X0..X17, the flags, and the SIMD registers (all of V0..V7 and
/// V16..V31, and the upper halves of V8..V15) -- but not X19..X29, which the
/// callee preserves. It calls `helper(cpu, site)`, which performs the access
/// on the state's copies of the registers; on success it reloads the same set
/// and returns to the site's return point, which reloads guest X16 (the
/// return used it) and whichever of X19..X29 the instruction may have
/// written. On a fault the helper has recorded the exception, so the stub
/// saves the rest of the registers and leaves through the epilogue.
pub fn mem_stub(asm: &mut Asm, helper: u64) {
    asm.emit(str_imm(3, 30, SP, offsets::SLOW_SITE));
    for n in (0..18).step_by(2) {
        asm.emit(stp(n, n + 1, SP, offsets::x(n) as i32));
    }
    asm.emit(mrs(0, NZCV));
    asm.emit(str_imm(3, 0, SP, offsets::NZCV));
    save_vectors(asm);
    call_rust(asm, helper, true);
    let fault = asm.label();
    asm.cbnz(true, 0, fault);
    load_vectors(asm);
    asm.emit(ldr_imm(3, 0, SP, offsets::NZCV));
    asm.emit(msr(NZCV, 0));
    asm.emit(ldr_imm(3, 16, SP, offsets::SLOW_SITE));
    asm.emit(ldrsw_imm(17, 16, 4));
    asm.emit(add_reg(true, 16, 16, 17, LSL, 0));
    for n in (0..16).step_by(2) {
        asm.emit(ldp(n, n + 1, SP, offsets::x(n) as i32));
    }
    asm.emit(ldr_imm(3, 17, SP, offsets::x(17)));
    asm.emit(ldr_imm(3, 30, SP, offsets::x(30)));
    asm.emit(br(16));

    asm.bind(fault);
    // X0..X17, X30, the flags and the vectors are in the state already; X19..
    // X29 still hold the guest's values (the call preserved them).
    let mut n = 19;
    while n <= 27 {
        asm.emit(stp(n, n + 1, SP, offsets::x(n) as i32));
        n += 2;
    }
    asm.emit(str_imm(3, 29, SP, offsets::x(29)));
    asm.emit(mrs(0, FPCR));
    asm.emit(str_imm(3, 0, SP, offsets::FPCR));
    asm.emit(mrs(0, FPSR));
    asm.emit(str_imm(3, 0, SP, offsets::FPSR));
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

/// The jump-cache lookup for code at `el`, shared by every indirect branch and
/// every direct branch to another page. A site reaches it with `B` after
/// storing guest X15..X17 to their home slots and putting the target guest PC
/// in X16. The cache is direct-mapped by PC; a hit branches to the target
/// block's entry (which restores X15 and X16), and a miss -- or a pending exit
/// request, so a loop through indirect branches cannot keep the CPU from its
/// dispatcher -- leaves with a Branch exit to the target.
///
/// Returns (lookup, miss) offsets in the region-relative sense of `asm.here`.
pub fn jc_stub(asm: &mut Asm, el: u32, save: u64) -> (u32, u32) {
    let at = asm.here();
    let miss = asm.label();
    asm.emit(str_imm(3, JUMP_REG, SP, offsets::NEXT_PC));
    asm.emit(ldr_imm(2, 17, SP, offsets::EXIT_REQUEST));
    asm.cbnz(false, 17, miss);
    asm.emit(ubfx(17, 16, 2, crate::state::JC_BITS));
    asm.emit(add_ext(17, SP, 17, EXT_UXTX, 4));
    let off = offsets::jc(el);
    debug_assert_eq!(off & 0xfff, 0);
    asm.emit(add_imm(true, 17, 17, off >> 12, true));
    asm.emit(ldp(15, 17, 17, 0));
    asm.emit(sub_reg(true, 15, 15, 16));
    asm.cbnz(true, 15, miss);
    asm.emit(mov(16, 17));
    asm.emit(ldr_imm(3, 17, SP, offsets::x(17)));
    asm.emit(br(16));
    asm.bind(miss);
    let miss_at = asm.here();
    asm.emit(ldp(15, 16, SP, offsets::x(15) as i32));
    asm.emit(ldr_imm(3, 17, SP, offsets::x(17)));
    // The exit prefix, with the next PC already in place.
    asm.emit(str_imm(3, 0, SP, offsets::x(0)));
    asm.emit(str_imm(3, 30, SP, offsets::x(30)));
    asm.emit(movz(true, 0, Exit::Branch as u32, 0));
    asm.emit(str_imm(2, 0, SP, offsets::EXIT));
    asm.b_abs(save);
    (at, miss_at)
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
/// and any extra fields, then branches to the save stub. Uses only X0 as
/// scratch, so the other live guest registers are still in their host
/// registers for the stub to save.
pub fn exit(asm: &mut Asm, save: u64, code: u32, next: Next, info: ExitInfo) {
    // Save guest X0; from here X0 is scratch.
    asm.emit(str_imm(3, 0, SP, offsets::x(0)));
    // The next PC first: an indirect branch through X30 (BLR X30) reads it
    // before the link overwrites it.
    match next {
        Next::Const(pc) => {
            asm.mov_imm(SCRATCH0, pc);
            asm.emit(str_imm(3, SCRATCH0, SP, offsets::NEXT_PC));
        }
        Next::Reg(rn) => {
            // X0 is already saved; X18 is virtual (it never lives in the
            // host's reserved x18) and is read from the state block.
            if rn == 0 || rn == 18 {
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
    // Guest X30: a link value, or its live value.
    match info.link {
        Some(ret) => {
            asm.mov_imm(SCRATCH0, ret);
            asm.emit(str_imm(3, SCRATCH0, SP, offsets::x(30)));
        }
        None => asm.emit(str_imm(3, 30, SP, offsets::x(30))),
    }
    if let Some(pc) = info.pc {
        asm.mov_imm(SCRATCH0, pc);
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
        asm.adr(SCRATCH0, site);
        asm.emit(str_imm(3, SCRATCH0, SP, offsets::CHAIN_SITE));
    }
    asm.emit(movz(true, SCRATCH0, code, 0));
    asm.emit(str_imm(2, SCRATCH0, SP, offsets::EXIT));
    asm.b_abs(save);
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

/// LDRSW (unsigned offset).
pub fn ldrsw_imm(rt: Reg, rn: Reg, offset: u32) -> u32 {
    assert!(offset % 4 == 0 && offset / 4 < 4096);
    0xb980_0000 | (offset / 4) << 10 | rn << 5 | rt
}

/// SUB (shifted register), no shift.
pub fn sub_reg(sf: bool, rd: Reg, rn: Reg, rm: Reg) -> u32 {
    (sf as u32) << 31 | 0x4b00_0000 | rm << 16 | rn << 5 | rd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_encodings() {
        // str x2, [sp, #-16]!  and  ldr x2, [sp], #16
        assert_eq!(str_imm_pre(3, 2, SP, -16), 0xf81f0fe2);
        assert_eq!(ldr_imm_post(3, 2, SP, 16), 0xf84107e2);
        // ldrsw x17, [x16, #4] ; sub x15, x15, x16
        assert_eq!(ldrsw_imm(17, 16, 4), 0xb9800611);
        assert_eq!(sub_reg(true, 15, 15, 16), 0xcb1001ef);
    }
}
