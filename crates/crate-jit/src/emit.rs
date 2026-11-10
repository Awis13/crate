// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The host-code sequences the translator emits that are the same whatever the
//! guest instruction is: entering and leaving a block, saving and restoring the
//! guest registers, and calling back into Rust.
//!
//! Throughout a block the host stack pointer is the address of the `CpuState`,
//! so a guest register reaches its spill slot with `[SP, #off]` and SP itself
//! is read or written with `ADD Xn, SP, #0`. See `state` for why.

use crate::asm::*;
use crate::state::offsets;

/// A host register the entry prologue and the save stub use as scratch before
/// the matching guest register is live or after it is saved.
const SCRATCH0: Reg = 0;
const SCRATCH1: Reg = 1;

/// Emits a block's dispatch entry word: it loads guest X30, which the shared
/// `enter` trampoline could not without a spare register. Blocks reached by a
/// direct branch from another block skip this and start at the body.
pub fn block_dispatch_prologue(asm: &mut Asm) {
    asm.emit(ldr_imm(3, 30, SP, offsets::x(30)));
}

/// Emits the shared stub that saves the guest registers a block leaves live
/// (X1..X29, NZCV, FPCR, FPSR and the SIMD registers) and returns through the
/// epilogue. The prefix each exit emits has already saved X0 and X30 and set
/// the exit code, so this stub is the same for every exit.
///
/// Returns the stub's offset in the block for the exits to branch to.
pub fn save_stub(asm: &mut Asm) -> u32 {
    let at = asm.here();
    // X1 first, then the even pairs up to X29.
    asm.emit(str_imm(3, 1, SP, offsets::x(1)));
    let mut n = 2;
    while n <= 28 {
        asm.emit(stp(n, n + 1, SP, (offsets::x(n)) as i32));
        n += 2;
    }
    // X29 is the last odd register; it was stored by the pair above (x28,x29).
    asm.emit(mrs(SCRATCH1, NZCV));
    asm.emit(str_imm(3, SCRATCH1, SP, offsets::NZCV));
    asm.emit(mrs(SCRATCH1, FPCR));
    asm.emit(str_imm(3, SCRATCH1, SP, offsets::FPCR));
    asm.emit(mrs(SCRATCH1, FPSR));
    asm.emit(str_imm(3, SCRATCH1, SP, offsets::FPSR));
    for q in (0..32).step_by(2) {
        asm.emit(stp_q(q, q + 1, SP, (offsets::v(q)) as i32));
    }
    asm.emit(ldr_imm(3, SCRATCH1, SP, offsets::EPILOGUE));
    asm.emit(br(SCRATCH1));
    at
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
}

/// Emits the exit prefix: saves X0 and X30, sets the exit code and the next PC
/// and any extra fields, then branches to the save stub, which lives
/// `save_stub_rel` bytes from this block's start (a negative distance, since
/// the stub is emitted before the block). Uses only X0 as scratch, so the
/// other live guest registers are still in their host registers for the stub
/// to save.
pub fn exit(asm: &mut Asm, save_stub_rel: i64, code: u32, next: Next, info: ExitInfo) {
    // Save guest X0; from here X0 is scratch.
    asm.emit(str_imm(3, 0, SP, offsets::x(0)));
    // Guest X30: a link value, or its live value.
    match info.link {
        Some(ret) => {
            asm.mov_imm(SCRATCH0, ret);
            asm.emit(str_imm(3, SCRATCH0, SP, offsets::x(30)));
        }
        None => asm.emit(str_imm(3, 30, SP, offsets::x(30))),
    }
    // Next PC.
    match next {
        Next::Const(pc) => {
            asm.mov_imm(SCRATCH0, pc);
            asm.emit(str_imm(3, SCRATCH0, SP, offsets::NEXT_PC));
        }
        Next::Reg(rn) => {
            if rn == 0 {
                asm.emit(ldr_imm(3, SCRATCH0, SP, offsets::x(0)));
            } else if rn == 30 {
                asm.emit(ldr_imm(3, SCRATCH0, SP, offsets::x(30)));
            } else {
                asm.emit(mov(SCRATCH0, rn));
            }
            asm.emit(str_imm(3, SCRATCH0, SP, offsets::NEXT_PC));
        }
        Next::None => {}
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
    asm.mov_imm(SCRATCH0, code as u64);
    asm.emit(str_imm(2, SCRATCH0, SP, offsets::EXIT));
    // Branch to the save stub, PC-relative from this point.
    let delta = save_stub_rel - asm.here() as i64;
    asm.emit(b(delta));
}

/// Emits a call back into Rust for a memory access: saves the whole guest
/// register file (a C call may clobber the caller-saved SIMD registers, so the
/// SIMD registers are saved too), switches to the real host stack, calls
/// `helper(cpu, word)`, then puts the result in X3 for the caller to test. The
/// caller reloads the register file, or, on a fault, exits with the state
/// already saved.
pub fn call_mem(asm: &mut Asm, helper: u64, word: u32) {
    spill_full(asm);
    // x0 = cpu pointer, x1 = instruction word, save &state on the host stack.
    asm.emit(ldr_imm(3, 0, SP, offsets::CPU_PTR));
    asm.mov_imm(1, word as u64);
    asm.emit(add_imm(true, 2, SP, 0, false)); // x2 = &state
    asm.emit(ldr_imm(3, 9, SP, offsets::HOST_SP));
    asm.emit(mov_sp(SP, 9)); // SP = host stack
    asm.emit(str_imm_pre(3, 2, SP, -16)); // push &state
    asm.bl_abs(helper);
    // x0 = result; recover &state and SP.
    asm.emit(mov(3, 0)); // x3 = result
    asm.emit(ldr_imm_post(3, 2, SP, 16)); // x2 = &state
    asm.emit(mov_sp(SP, 2)); // SP = &state
    // The caller inspects x3 before the reload clobbers it.
}

/// Saves the whole guest state (X0..X30, NZCV, FPCR, FPSR, V0..V31) to the
/// state block. X0 and X1 become free afterwards.
pub fn spill_full(asm: &mut Asm) {
    let mut n = 0;
    while n <= 28 {
        asm.emit(stp(n, n + 1, SP, offsets::x(n) as i32));
        n += 2;
    }
    asm.emit(str_imm(3, 30, SP, offsets::x(30)));
    asm.emit(mrs(SCRATCH0, NZCV));
    asm.emit(str_imm(3, SCRATCH0, SP, offsets::NZCV));
    asm.emit(mrs(SCRATCH0, FPCR));
    asm.emit(str_imm(3, SCRATCH0, SP, offsets::FPCR));
    asm.emit(mrs(SCRATCH0, FPSR));
    asm.emit(str_imm(3, SCRATCH0, SP, offsets::FPSR));
    for q in (0..32).step_by(2) {
        asm.emit(stp_q(q, q + 1, SP, offsets::v(q) as i32));
    }
}

/// Reloads the whole guest state the spill above saved.
pub fn reload_full(asm: &mut Asm) {
    asm.emit(ldr_imm(3, SCRATCH0, SP, offsets::NZCV));
    asm.emit(msr(NZCV, SCRATCH0));
    asm.emit(ldr_imm(3, SCRATCH0, SP, offsets::FPCR));
    asm.emit(msr(FPCR, SCRATCH0));
    asm.emit(ldr_imm(3, SCRATCH0, SP, offsets::FPSR));
    asm.emit(msr(FPSR, SCRATCH0));
    for q in (0..32).step_by(2) {
        asm.emit(ldp_q(q, q + 1, SP, offsets::v(q) as i32));
    }
    let mut n = 0;
    while n <= 28 {
        asm.emit(ldp(n, n + 1, SP, offsets::x(n) as i32));
        n += 2;
    }
    asm.emit(ldr_imm(3, 30, SP, offsets::x(30)));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_post_index_encodings() {
        // str x2, [sp, #-16]!  and  ldr x2, [sp], #16
        assert_eq!(str_imm_pre(3, 2, SP, -16), 0xf81f0fe2);
        assert_eq!(ldr_imm_post(3, 2, SP, 16), 0xf84107e2);
    }
}
