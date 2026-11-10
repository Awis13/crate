// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Sorting a guest instruction into what the translator must do with it (Arm
//! ARM, the A64 encoding index).
//!
//! Most instructions are copied into the host block unchanged, because the
//! guest's registers live in the host's registers of the same number while
//! the block runs. Only the ones that touch the machine are rewritten:
//! anything that reads or writes the program counter (branches and the
//! PC-relative addresses), anything that goes through the guest's memory
//! (loads and stores), the system registers, the barriers and cache and TLB
//! maintenance, and the instructions that change the exception level. The
//! stack pointer is virtualised too, so the few data-processing instructions
//! that name it as an operand are rewritten.

/// What to do with one guest instruction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    /// Copy the word into the host block unchanged.
    Copy,
    /// A PC-relative address (ADR, ADRP): compute it from the guest PC into Rd.
    PcRel { adrp: bool, rd: u32, imm: i64 },
    /// An add or subtract immediate that names SP: `rd = rn +/- imm12<<sh`.
    AddSubSpImm {
        sub: bool,
        sf: bool,
        set_flags: bool,
        rd: u32,
        rn: u32,
        imm12: u32,
        shift12: bool,
    },
    /// An add or subtract of an extended register that names SP.
    AddSubSpExt {
        sub: bool,
        sf: bool,
        set_flags: bool,
        rd: u32,
        rn: u32,
        rm: u32,
        option: u32,
        amount: u32,
    },
    /// A load or store the MMU must translate.
    Memory(Mem),
    /// A direct branch: B or BL to a known target.
    Branch { link: bool, target: u64 },
    /// A conditional branch: B.cond, CBZ/CBNZ, TBZ/TBNZ. `taken` is the guest
    /// target; the fall-through is the next instruction.
    CondBranch {
        word: u32,
        kind: CondKind,
        taken: u64,
    },
    /// An indirect branch: BR, BLR or RET through a register.
    IndBranch { link: bool, ret: bool, rn: u32 },
    /// MRS or MSR of a system register.
    Sysreg { read: bool, reg: u16, rt: u32 },
    /// MSR to a PSTATE field by immediate (DAIFSet/Clr, SPSel, ...).
    MsrImm { op1: u32, op2: u32, crm: u32 },
    /// SVC, HVC or SMC.
    Call { kind: CallKind, imm: u32 },
    /// A hint whose only effect that matters is waiting (WFI, WFE).
    Wait { event: bool },
    /// A barrier or cache/TLB maintenance operation handled as a helper.
    System { word: u32 },
    /// An exception-returning instruction (ERET).
    Eret,
    /// A PC-relative literal load (LDR/LDRSW literal). The value lives at a
    /// known guest address; the translator reads it and materialises it.
    LiteralLoad {
        rt: u32,
        addr: u64,
        /// log2 of the access size in bytes (2 or 3 for general registers).
        size: u32,
        signed: bool,
        /// A SIMD/FP destination (not yet handled).
        vector: bool,
        /// PRFM (a hint); do nothing.
        prefetch: bool,
    },
    /// Not understood; raise an Undefined-instruction exception.
    Undefined,
}

/// The register-level details of a load or store.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Mem {
    pub word: u32,
    pub kind: MemKind,
    /// log2 of the access size in bytes.
    pub size: u32,
    /// Whether the loaded value is sign-extended.
    pub signed: bool,
    /// Whether the destination register is 64-bit (else the top 32 bits are
    /// zeroed).
    pub reg64: bool,
    pub load: bool,
    /// Base register (31 is SP here, not the zero register).
    pub rn: u32,
    /// First transfer register.
    pub rt: u32,
    /// Second transfer register for a pair.
    pub rt2: u32,
    /// The computed byte displacement for the simple addressing modes.
    pub offset: i64,
    /// Write the base back (pre/post-index).
    pub writeback: bool,
    /// Add the offset before the access (pre-index or unsigned offset).
    pub pre: bool,
    /// For a register offset: the offset register, its extend and shift.
    pub index_reg: Option<(u32, u32, u32)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemKind {
    /// A single general register.
    Gpr,
    /// A pair of general registers.
    Pair,
    /// A SIMD/FP register of 1, 2, 4, 8 or 16 bytes.
    Vec,
    /// A pair of SIMD/FP registers.
    VecPair,
    /// Exclusive or acquire/release; handled by the helper conservatively.
    Ordered,
    /// An LSE atomic read-modify-write (LDADD, SWP, ...) or CAS, done by the
    /// helper as a single-CPU read-modify-write.
    Atomic,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CondKind {
    /// B.cond: the 4-bit condition is in the word.
    Cond,
    Cbz {
        sf: bool,
        rt: u32,
    },
    Cbnz {
        sf: bool,
        rt: u32,
    },
    Tbz {
        rt: u32,
        bit: u32,
    },
    Tbnz {
        rt: u32,
        bit: u32,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CallKind {
    Svc,
    Hvc,
    Smc,
    Brk,
}

fn sign_extend(value: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

/// Classifies `word`, the instruction at guest address `pc`.
pub fn decode(word: u32, pc: u64) -> Op {
    let op0 = (word >> 25) & 0xf;
    match op0 {
        // Data processing -- immediate.
        0b1000 | 0b1001 => decode_data_imm(word, pc),
        // Branches, exception generating and system.
        0b1010 | 0b1011 => decode_branch_system(word, pc),
        // Loads and stores.
        0b0100 | 0b0110 | 0b1100 | 0b1110 => decode_load_store(word, pc),
        // Data processing -- register.
        0b0101 | 0b1101 => decode_data_reg(word),
        // Data processing -- SIMD and floating point.
        0b0111 | 0b1111 => Op::Copy,
        // Reserved / unallocated (op0 0b0000 is reserved; SVE is 0b0010).
        _ => Op::Undefined,
    }
}

fn decode_data_imm(word: u32, pc: u64) -> Op {
    let op = (word >> 23) & 0x7;
    match op {
        // PC-relative addressing.
        0b000 | 0b001 => {
            let rd = word & 31;
            let immlo = (word >> 29) & 3;
            let immhi = (word >> 5) & 0x7ffff;
            let adrp = word & (1 << 31) != 0;
            if adrp {
                let imm = sign_extend(((immhi << 2) | immlo) as u64, 21) << 12;
                Op::PcRel {
                    adrp: true,
                    rd,
                    imm: (pc & !0xfff).wrapping_add(imm as u64) as i64,
                }
            } else {
                let imm = sign_extend(((immhi << 2) | immlo) as u64, 21);
                Op::PcRel {
                    adrp: false,
                    rd,
                    imm: pc.wrapping_add(imm as u64) as i64,
                }
            }
        }
        // Add/subtract immediate: 31 means SP in Rn and Rd.
        0b010 => {
            let rd = word & 31;
            let rn = (word >> 5) & 31;
            let set_flags = word & (1 << 29) != 0;
            // Register 31 is SP for Rn always, and for Rd only without flags
            // (ADDS/SUBS with Rd 31 is a compare whose result is discarded, so
            // it copies unchanged).
            if rn == 31 || (rd == 31 && !set_flags) {
                Op::AddSubSpImm {
                    sub: word & (1 << 30) != 0,
                    sf: word & (1 << 31) != 0,
                    set_flags,
                    rd,
                    rn,
                    imm12: (word >> 10) & 0xfff,
                    shift12: word & (1 << 22) != 0,
                }
            } else {
                Op::Copy
            }
        }
        // Add/subtract immediate with tags (ADDG/SUBG): rare, not modelled.
        0b011 => Op::Undefined,
        // Logical, move wide, bitfield, extract: 31 is the zero register.
        _ => Op::Copy,
    }
}

fn decode_data_reg(word: u32) -> Op {
    // Add/subtract extended register: op bits 28:24 == 0b01011, bit 21 set,
    // option field in 15:13. 31 means SP for Rn (and Rd when not setting flags).
    if (word >> 24) & 0x1f == 0b01011 && word & (1 << 21) != 0 && (word >> 22) & 3 == 0 {
        let rd = word & 31;
        let rn = (word >> 5) & 31;
        let set_flags = word & (1 << 29) != 0;
        // Rd is SP only when flags are not set; with flags Rd 31 is XZR.
        if rn == 31 || (rd == 31 && !set_flags) {
            return Op::AddSubSpExt {
                sub: word & (1 << 30) != 0,
                sf: word & (1 << 31) != 0,
                set_flags,
                rd,
                rn,
                rm: (word >> 16) & 31,
                option: (word >> 13) & 7,
                amount: (word >> 10) & 7,
            };
        }
    }
    Op::Copy
}

fn decode_branch_system(word: u32, pc: u64) -> Op {
    let top = (word >> 29) & 0x7;
    // Unconditional branch immediate: B and BL.
    if top & 0x3 == 0 && (word >> 26) & 0x3f & 0b011111 == 0b000101 {
        let link = word & (1 << 31) != 0;
        let imm = sign_extend((word & 0x3ff_ffff) as u64, 26) * 4;
        return Op::Branch {
            link,
            target: pc.wrapping_add(imm as u64),
        };
    }
    let op = word >> 24 & 0xff;
    match op {
        // B.cond and BC.cond.
        0b0101_0100 if word & (1 << 4) == 0 => {
            let imm = sign_extend(((word >> 5) & 0x7ffff) as u64, 19) * 4;
            Op::CondBranch {
                word,
                kind: CondKind::Cond,
                taken: pc.wrapping_add(imm as u64),
            }
        }
        // Compare and branch (CBZ/CBNZ).
        0b0011_0100 | 0b1011_0100 | 0b0011_0101 | 0b1011_0101 => {
            let sf = word & (1 << 31) != 0;
            let rt = word & 31;
            let imm = sign_extend(((word >> 5) & 0x7ffff) as u64, 19) * 4;
            let taken = pc.wrapping_add(imm as u64);
            let kind = if word & (1 << 24) == 0 {
                CondKind::Cbz { sf, rt }
            } else {
                CondKind::Cbnz { sf, rt }
            };
            Op::CondBranch { word, kind, taken }
        }
        // Test and branch (TBZ/TBNZ).
        0b0011_0110 | 0b1011_0110 | 0b0011_0111 | 0b1011_0111 => {
            let rt = word & 31;
            let bit = ((word >> 31) << 5) | ((word >> 19) & 0x1f);
            let imm = sign_extend(((word >> 5) & 0x3fff) as u64, 14) * 4;
            let taken = pc.wrapping_add(imm as u64);
            let kind = if word & (1 << 24) == 0 {
                CondKind::Tbz { rt, bit }
            } else {
                CondKind::Tbnz { rt, bit }
            };
            Op::CondBranch { word, kind, taken }
        }
        // Unconditional branch register: BR, BLR, RET (and ERET, which 10:0
        // distinguishes).
        0b1101_0110 => decode_branch_reg(word),
        // Exception generation: SVC/HVC/SMC/BRK and friends.
        0b1101_0100 => decode_exception(word),
        // System: MRS/MSR, barriers, hints, cache/TLB maintenance.
        0b1101_0101 => decode_system(word),
        _ => Op::Undefined,
    }
}

fn decode_branch_reg(word: u32) -> Op {
    let opc = (word >> 21) & 0xf;
    let rn = (word >> 5) & 31;
    match opc {
        0b0000 => Op::IndBranch {
            link: false,
            ret: false,
            rn,
        }, // BR
        0b0001 => Op::IndBranch {
            link: true,
            ret: false,
            rn,
        }, // BLR
        0b0010 => Op::IndBranch {
            link: false,
            ret: true,
            rn,
        }, // RET
        0b0100 => Op::Eret,
        // BRAA/BRAB/RETAA/... (pointer-authenticated branches) are not modelled.
        _ => Op::Undefined,
    }
}

fn decode_exception(word: u32) -> Op {
    let opc = (word >> 21) & 0x7;
    let ll = word & 0x3;
    let imm = (word >> 5) & 0xffff;
    match (opc, ll) {
        (0b000, 0b01) => Op::Call {
            kind: CallKind::Svc,
            imm,
        },
        (0b000, 0b10) => Op::Call {
            kind: CallKind::Hvc,
            imm,
        },
        (0b000, 0b11) => Op::Call {
            kind: CallKind::Smc,
            imm,
        },
        (0b001, 0b00) => Op::Call {
            kind: CallKind::Brk,
            imm,
        },
        // HLT and the debug exceptions: treat as a breakpoint-like call.
        (0b010, 0b00) => Op::Call {
            kind: CallKind::Brk,
            imm,
        },
        _ => Op::Undefined,
    }
}

fn decode_system(word: u32) -> Op {
    let l = word & (1 << 21) != 0;
    let op0 = (word >> 19) & 0x3;
    // MSR (immediate) and hints and barriers have op0 == 0b00.
    if op0 == 0b00 {
        let op1 = (word >> 16) & 0x7;
        let crn = (word >> 12) & 0xf;
        let crm = (word >> 8) & 0xf;
        let op2 = (word >> 5) & 0x7;
        if crn == 0b0100 {
            // MSR (immediate) to a PSTATE field.
            return Op::MsrImm { op1, op2, crm };
        }
        if crn == 0b0010 {
            // Hints: NOP, YIELD, WFE, WFI, SEV, and the rest.
            return match (op1, op2) {
                (0b011, 0b010) => Op::Wait { event: true },  // WFE
                (0b011, 0b011) => Op::Wait { event: false }, // WFI
                _ => Op::Copy,                               // NOP, YIELD, hints: harmless
            };
        }
        if crn == 0b0011 {
            // Barriers (DSB, DMB, ISB) and CLREX: let them run on the host.
            return Op::Copy;
        }
        return Op::System { word };
    }
    // MRS/MSR of a named system register (op0 0b10 or 0b11).
    let reg = sysreg_encoding(word);
    let rt = word & 31;
    // Cache and TLB maintenance are MSR to CRn 7 or 8; keep them as helpers.
    let crn = (word >> 12) & 0xf;
    if !l && (crn == 0b0111 || crn == 0b1000) {
        return Op::System { word };
    }
    Op::Sysreg { read: l, reg, rt }
}

/// Packs a system register the way crate_core::cpu::SysReg does, from an MRS or
/// MSR encoding: op0 in bits 20:19 (plus the implicit 1), op1 18:16, CRn 15:12,
/// CRm 11:8, op2 7:5.
pub fn sysreg_encoding(word: u32) -> u16 {
    let op0 = 2 + ((word >> 19) & 1);
    let op1 = (word >> 16) & 7;
    let crn = (word >> 12) & 0xf;
    let crm = (word >> 8) & 0xf;
    let op2 = (word >> 5) & 7;
    ((op0 as u16 & 3) << 14)
        | ((op1 as u16 & 7) << 11)
        | ((crn as u16 & 15) << 7)
        | ((crm as u16 & 15) << 3)
        | (op2 as u16 & 7)
}

fn decode_load_store(word: u32, pc: u64) -> Op {
    // The load/store encoding group, split by bits 29:28 (op0's low part is in
    // the dispatcher) and 26 (V, a SIMD transfer) and 24.
    let op0 = (word >> 28) & 0xf;
    let v = word & (1 << 26) != 0;

    // Load register (literal): bits 29:27 == 0b011, 25:24 == 0b00.
    if (word >> 27) & 0x7 == 0b011 && (word >> 24) & 0x3 == 0 {
        let opc = (word >> 30) & 0x3;
        let imm = sign_extend(((word >> 5) & 0x7ffff) as u64, 19) * 4;
        let addr = pc.wrapping_add(imm as u64);
        let (size, signed, prefetch) = if v {
            (2 + opc, false, false) // S/D/Q
        } else {
            match opc {
                0b00 => (2, false, false), // LDR (32-bit)
                0b01 => (3, false, false), // LDR (64-bit)
                0b10 => (2, true, false),  // LDRSW
                _ => (3, false, true),     // PRFM (literal)
            }
        };
        return Op::LiteralLoad {
            rt: word & 31,
            addr,
            size,
            signed,
            vector: v,
            prefetch,
        };
    }

    // Load/store register pair (op0 x0x0 with bit 25 == 0, selected here by the
    // 101x0 shape at bits 29:27 == 0b101).
    if (word >> 27) & 0x7 == 0b101 && (word >> 25) & 1 == 0 {
        return decode_pair(word, v);
    }

    // Load/store register (unscaled, immediate post/pre-index, unprivileged,
    // register offset, PAC) and unsigned immediate: bits 29:27 == 0b111.
    if (word >> 27) & 0x7 == 0b111 {
        return decode_single(word, v);
    }

    // Load/store exclusive and ordered (LDAR/STLR, LDXR/STXR, CAS): bits
    // 29:27 == 0b001. Handle conservatively through the helper.
    if (word >> 27) & 0x7 == 0b001 {
        return decode_ordered(word);
    }

    // Literal loads (LDR literal) read from PC-relative memory.
    if op0 & 0b0011 == 0b0001 && (word >> 27) & 0x7 == 0b011 {
        // Falls through to Undefined for now; handled as PC-relative literal
        // in a dedicated path by the translator's caller.
    }
    Op::Undefined
}

fn decode_pair(word: u32, v: bool) -> Op {
    let opc = (word >> 30) & 0x3;
    let load = word & (1 << 22) != 0;
    let kind_bits = (word >> 23) & 0x3; // 01 post, 10 offset, 11 pre
    let rt = word & 31;
    let rt2 = (word >> 10) & 31;
    let rn = (word >> 5) & 31;
    let imm7 = sign_extend(((word >> 15) & 0x7f) as u64, 7);
    let (size, scale, kind) = if v {
        let size = 2 + opc; // 32/64/128-bit vectors
        (size, size, MemKind::VecPair)
    } else {
        let size = if opc == 0b10 { 3 } else { 2 }; // 64 or 32-bit (opc 01 is signed)
        (size, size, MemKind::Pair)
    };
    let offset = imm7 << scale;
    let (pre, writeback) = match kind_bits {
        0b01 => (false, true), // post-index
        0b11 => (true, true),  // pre-index
        _ => (true, false),    // signed offset
    };
    let signed = !v && opc == 0b01;
    // A pair is 64-bit per element except the 32-bit non-vector form; LDPSW
    // (signed) targets 64-bit registers.
    let reg64 = v || opc != 0b00 || signed;
    Op::Memory(Mem {
        word,
        kind,
        size,
        signed,
        reg64,
        load,
        rn,
        rt,
        rt2,
        offset,
        writeback,
        pre,
        index_reg: None,
    })
}

fn decode_single(word: u32, v: bool) -> Op {
    let size_field = (word >> 30) & 0x3;
    let opc = (word >> 22) & 0x3;
    let rt = word & 31;
    let rn = (word >> 5) & 31;

    // Size and whether this is a load, its sign-extension and register width.
    let (size, load, signed, reg64) = if v {
        // SIMD: the size is (opc<1> : size_field).
        let size = ((opc & 2) << 1) | size_field;
        let load = opc & 1 != 0;
        (size, load, false, false)
    } else {
        let load = opc != 0b00; // 00 STR, 01 LDR, 10/11 LDRS
        let signed = opc == 0b10 || opc == 0b11;
        // opc 10 sign-extends to 64 bits; opc 11 to 32. An unsigned 64-bit load
        // is size 3; everything else targets a 32-bit register.
        let reg64 = (signed && opc == 0b10) || (!signed && size_field == 3);
        (size_field, load, signed, reg64)
    };
    let kind = if v { MemKind::Vec } else { MemKind::Gpr };

    // Atomic memory operations (LSE: LDADD, SWP, ...) and LDAPR share the
    // register group but set bit 21 with bits 11:10 == 0b00. Hand the whole
    // word to the helper, which does a single-CPU read-modify-write.
    if !v && word & (1 << 24) == 0 && word & (1 << 21) != 0 && (word >> 10) & 3 == 0 {
        return Op::Memory(Mem {
            word,
            kind: MemKind::Atomic,
            size,
            signed: false,
            reg64: size == 3,
            load: true,
            rn,
            rt,
            rt2: (word >> 16) & 31, // Rs
            offset: 0,
            writeback: false,
            pre: true,
            index_reg: None,
        });
    }

    // Unsigned immediate: bit 24 set.
    if word & (1 << 24) != 0 {
        let offset = (((word >> 10) & 0xfff) as i64) << size;
        return Op::Memory(Mem {
            word,
            kind,
            size,
            signed,
            reg64,
            load,
            rn,
            rt,
            rt2: 0,
            offset,
            writeback: false,
            pre: true,
            index_reg: None,
        });
    }

    let imm9 = sign_extend(((word >> 12) & 0x1ff) as u64, 9);
    let mode = (word >> 10) & 0x3;
    match mode {
        0b00 => {
            // Unscaled immediate (LDUR/STUR) or, with bit 21, register offset /
            // atomics -- but register offset has bit 21 set and 11:10 == 10.
            Op::Memory(Mem {
                word,
                kind,
                size,
                signed,
                reg64,
                load,
                rn,
                rt,
                rt2: 0,
                offset: imm9,
                writeback: false,
                pre: true,
                index_reg: None,
            })
        }
        0b01 => Op::Memory(Mem {
            word,
            kind,
            size,
            signed,
            reg64,
            load,
            rn,
            rt,
            rt2: 0,
            offset: imm9,
            writeback: true,
            pre: false, // post-index
            index_reg: None,
        }),
        0b11 => Op::Memory(Mem {
            word,
            kind,
            size,
            signed,
            reg64,
            load,
            rn,
            rt,
            rt2: 0,
            offset: imm9,
            writeback: true,
            pre: true, // pre-index
            index_reg: None,
        }),
        // Register offset: bit 21 set, option in 15:13, S in 12.
        0b10 if word & (1 << 21) != 0 => {
            let rm = (word >> 16) & 31;
            let option = (word >> 13) & 7;
            let s = (word >> 12) & 1;
            let amount = if s != 0 { size } else { 0 };
            Op::Memory(Mem {
                word,
                kind,
                size,
                signed,
                reg64,
                load,
                rn,
                rt,
                rt2: 0,
                offset: 0,
                writeback: false,
                pre: true,
                index_reg: Some((rm, option, amount)),
            })
        }
        _ => Op::Undefined,
    }
}

fn decode_ordered(word: u32) -> Op {
    let size = (word >> 30) & 0x3;
    let load = word & (1 << 22) != 0;
    let rt = word & 31;
    let rn = (word >> 5) & 31;
    // LDAR/STLR, LDAXR/STLXR, LDXR/STXR, CAS and the pair forms all go through
    // the helper, which does the access with host ordering and the exclusive
    // monitor. The details it needs beyond these are decoded there.
    Op::Memory(Mem {
        word,
        kind: MemKind::Ordered,
        size,
        signed: false,
        reg64: size == 3,
        load,
        rn,
        rt,
        rt2: (word >> 10) & 31,
        offset: 0,
        writeback: false,
        pre: true,
        index_reg: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_ordinary_data_processing() {
        // add x0, x1, x2 ; orr x3, x4, x5 ; movz x0, #1 ; fmov d0, d1
        for w in [0x8b020020, 0xaa050083, 0xd2800020, 0x1e604020] {
            assert_eq!(decode(w, 0x1000), Op::Copy);
        }
        // add x0, x1, #4 (no SP) copies
        assert_eq!(decode(0x91001020, 0x1000), Op::Copy);
    }

    #[test]
    fn finds_sp_forms() {
        // sub sp, sp, #16
        assert!(matches!(
            decode(0xd10043ff, 0),
            Op::AddSubSpImm { rd: 31, rn: 31, .. }
        ));
        // add x29, sp, #0
        assert!(matches!(
            decode(0x910003fd, 0),
            Op::AddSubSpImm { rd: 29, rn: 31, .. }
        ));
        // add sp, sp, x0 (extended register, uxtx)
        assert!(matches!(decode(0x8b2063ff, 0), Op::AddSubSpExt { .. }));
    }

    #[test]
    fn pc_relative() {
        // adrp x0, #0x1000 ; adr x1, .+8
        assert!(matches!(
            decode(0x90000000, 0x4000),
            Op::PcRel {
                adrp: true,
                rd: 0,
                ..
            }
        ));
        match decode(0x10000041, 0x4000) {
            Op::PcRel {
                adrp: false,
                rd: 1,
                imm,
            } => assert_eq!(imm, 0x4008),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn branches() {
        // b .+8 ; bl .-4 ; ret ; br x5 ; blr x9
        assert_eq!(
            decode(0x14000002, 0x4000),
            Op::Branch {
                link: false,
                target: 0x4008
            }
        );
        assert_eq!(
            decode(0x97ffffff, 0x4000),
            Op::Branch {
                link: true,
                target: 0x3ffc
            }
        );
        assert_eq!(
            decode(0xd65f03c0, 0x4000),
            Op::IndBranch {
                link: false,
                ret: true,
                rn: 30
            }
        );
        assert!(matches!(decode(0xd61f00a0, 0), Op::IndBranch { rn: 5, .. }));
        assert!(matches!(
            decode(0xd63f0120, 0),
            Op::IndBranch {
                rn: 9,
                link: true,
                ..
            }
        ));
    }

    #[test]
    fn conditional_branches() {
        // b.eq .+8
        match decode(0x54000040, 0x4000) {
            Op::CondBranch {
                kind: CondKind::Cond,
                taken,
                ..
            } => assert_eq!(taken, 0x4008),
            other => panic!("{other:?}"),
        }
        // cbz x0, .+8 ; cbnz w1, .-4 ; tbz x2, #3, .+8 ; tbnz w3, #5, .+4
        assert!(matches!(
            decode(0xb4000040, 0),
            Op::CondBranch {
                kind: CondKind::Cbz { sf: true, rt: 0 },
                ..
            }
        ));
        assert!(matches!(
            decode(0x37280003, 0),
            Op::CondBranch {
                kind: CondKind::Tbnz { rt: 3, bit: 5 },
                ..
            }
        ));
    }

    #[test]
    fn system_and_exceptions() {
        // svc #0 ; hvc #0 ; brk #1 ; mrs x0, midr_el1 ; msr daifset, #2
        assert!(matches!(
            decode(0xd4000001, 0),
            Op::Call {
                kind: CallKind::Svc,
                imm: 0
            }
        ));
        assert!(matches!(
            decode(0xd4000002, 0),
            Op::Call {
                kind: CallKind::Hvc,
                ..
            }
        ));
        assert!(matches!(
            decode(0xd4200020, 0),
            Op::Call {
                kind: CallKind::Brk,
                imm: 1
            }
        ));
        match decode(0xd5380000, 0) {
            Op::Sysreg {
                read: true,
                reg,
                rt: 0,
            } => assert_eq!(reg, crate_core::cpu::SysReg::new(3, 0, 0, 0, 0).0),
            other => panic!("{other:?}"),
        }
        assert!(matches!(decode(0xd50342df, 0), Op::MsrImm { .. })); // msr daifset
        // wfi ; wfe ; nop
        assert_eq!(decode(0xd503207f, 0), Op::Wait { event: false });
        assert_eq!(decode(0xd503205f, 0), Op::Wait { event: true });
        assert_eq!(decode(0xd503201f, 0), Op::Copy);
        // dsb ish ; isb
        assert_eq!(decode(0xd5033bbf, 0), Op::Copy);
        assert_eq!(decode(0xd5033fdf, 0), Op::Copy);
        // eret
        assert_eq!(decode(0xd69f03e0, 0), Op::Eret);
    }

    #[test]
    fn loads_and_stores() {
        // ldr x1, [x0, #8]
        match decode(0xf9400401, 0) {
            Op::Memory(m) => {
                assert!(m.load && m.size == 3 && m.offset == 8 && m.rt == 1 && m.rn == 0);
            }
            other => panic!("{other:?}"),
        }
        // str w2, [x3, #-4]! (pre-index)
        match decode(0xb81fcc62, 0) {
            Op::Memory(m) => {
                assert!(!m.load && m.writeback && m.pre && m.offset == -4 && m.size == 2);
            }
            other => panic!("{other:?}"),
        }
        // ldrb w0, [x1], #1 (post-index)
        match decode(0x38401420, 0) {
            Op::Memory(m) => {
                assert!(m.load && m.writeback && !m.pre && m.size == 0 && m.offset == 1);
            }
            other => panic!("{other:?}"),
        }
        // ldrsw x0, [x1, x2, lsl #2] (register offset, sign-extended to 64)
        match decode(0xb8a27820, 0) {
            Op::Memory(m) => {
                assert!(m.load && m.signed && m.size == 2);
                assert_eq!(m.index_reg, Some((2, 0b011, 2)));
            }
            other => panic!("{other:?}"),
        }
        // stp x0, x1, [sp, #16]
        match decode(0xa90107e0, 0) {
            Op::Memory(m) => {
                assert!(m.kind == MemKind::Pair && m.rt == 0 && m.rt2 == 1 && m.rn == 31);
            }
            other => panic!("{other:?}"),
        }
        // ldar x0, [x1]
        match decode(0xc8dffc20, 0) {
            Op::Memory(m) => assert!(m.kind == MemKind::Ordered && m.load),
            other => panic!("{other:?}"),
        }
    }
}
