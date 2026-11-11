// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Encodes the host instructions the translator writes (A64, from the Arm
//! ARM's encoding index), and assembles them into blocks with labels, literal
//! pools and branches to addresses outside the block.
//!
//! Registers are numbers 0 to 31; whether 31 means the zero register or SP is
//! the instruction's business, as in the architecture.

pub type Reg = u32;

pub const XZR: Reg = 31;
pub const SP: Reg = 31;

/// Condition codes, as B.cond encodes them.
pub const EQ: u32 = 0;
pub const NE: u32 = 1;

fn imm(value: i64, bits: u32, what: &str) -> u32 {
    let min = -(1i64 << (bits - 1));
    let max = (1i64 << (bits - 1)) - 1;
    assert!(
        (min..=max).contains(&value),
        "{what}: {value} does not fit in {bits} bits"
    );
    (value as u32) & ((1 << bits) - 1)
}

// ---------------------------------------------------------------- moves

pub fn movz(sf: bool, rd: Reg, imm16: u32, hw: u32) -> u32 {
    (sf as u32) << 31 | 0x5280_0000 | hw << 21 | (imm16 & 0xffff) << 5 | rd
}

pub fn movk(sf: bool, rd: Reg, imm16: u32, hw: u32) -> u32 {
    (sf as u32) << 31 | 0x7280_0000 | hw << 21 | (imm16 & 0xffff) << 5 | rd
}

pub fn movn(sf: bool, rd: Reg, imm16: u32, hw: u32) -> u32 {
    (sf as u32) << 31 | 0x1280_0000 | hw << 21 | (imm16 & 0xffff) << 5 | rd
}

/// MOV between general registers (ORR with the zero register); not for SP.
pub fn mov(rd: Reg, rm: Reg) -> u32 {
    orr_reg(true, rd, XZR, rm, 0, 0)
}

/// MOV to or from SP (ADD immediate of zero).
pub fn mov_sp(rd: Reg, rn: Reg) -> u32 {
    add_imm(true, rd, rn, 0, false)
}

/// The shortest MOVZ/MOVN/MOVK sequence for a 64-bit constant.
pub fn mov_imm(rd: Reg, value: u64) -> Vec<u32> {
    let halves: Vec<u32> = (0..4)
        .map(|i| ((value >> (16 * i)) & 0xffff) as u32)
        .collect();
    let zeros = halves.iter().filter(|&&h| h == 0).count();
    let ones = halves.iter().filter(|&&h| h == 0xffff).count();
    let mut out = Vec::new();
    if zeros < 3 && ones < 3 {
        if let Some(f) = logical_imm(value, true) {
            // ORR Rd, XZR, #value.
            return vec![0xb200_0000 | f << 10 | XZR << 5 | rd];
        }
    }
    if ones > zeros {
        for (i, &h) in halves.iter().enumerate() {
            if h == 0xffff {
                continue;
            }
            if out.is_empty() {
                out.push(movn(true, rd, !h & 0xffff, i as u32));
            } else {
                out.push(movk(true, rd, h, i as u32));
            }
        }
        if out.is_empty() {
            out.push(movn(true, rd, 0, 0));
        }
    } else {
        for (i, &h) in halves.iter().enumerate() {
            if h == 0 {
                continue;
            }
            if out.is_empty() {
                out.push(movz(true, rd, h, i as u32));
            } else {
                out.push(movk(true, rd, h, i as u32));
            }
        }
        if out.is_empty() {
            out.push(movz(true, rd, 0, 0));
        }
    }
    out
}

// ---------------------------------------------------------------- arithmetic

/// ADD (immediate): Rd and Rn may be SP. `shift12` shifts the 12-bit value.
pub fn add_imm(sf: bool, rd: Reg, rn: Reg, imm12: u32, shift12: bool) -> u32 {
    assert!(imm12 < 4096);
    (sf as u32) << 31 | 0x1100_0000 | (shift12 as u32) << 22 | imm12 << 10 | rn << 5 | rd
}

pub fn sub_imm(sf: bool, rd: Reg, rn: Reg, imm12: u32, shift12: bool) -> u32 {
    assert!(imm12 < 4096);
    (sf as u32) << 31 | 0x5100_0000 | (shift12 as u32) << 22 | imm12 << 10 | rn << 5 | rd
}

/// Rd = Rn + value for any 64-bit value, through `tmp` when it takes more
/// than two ADD or SUB immediates. Rd and Rn may be SP; `tmp` may equal Rd
/// only when Rd is not SP and differs from Rn.
pub fn add_const(rd: Reg, rn: Reg, value: i64, tmp: Reg) -> Vec<u32> {
    let mag = value.unsigned_abs();
    let op = if value < 0 { sub_imm } else { add_imm };
    if mag < 4096 {
        return vec![op(true, rd, rn, mag as u32, false)];
    }
    if mag < 1 << 24 {
        let mut out = vec![op(true, rd, rn, (mag >> 12) as u32, true)];
        if mag & 0xfff != 0 {
            out.push(op(true, rd, rd, (mag & 0xfff) as u32, false));
        }
        return out;
    }
    let mut out = mov_imm(tmp, value as u64);
    out.push(add_ext(rd, rn, tmp, EXT_UXTX, 0));
    out
}

/// The extend options of ADD (extended register), which loads and stores
/// with a register offset share.
pub const EXT_UXTW: u32 = 0b010;
pub const EXT_UXTX: u32 = 0b011;

/// ADD (extended register), 64-bit: Rd = Rn + extend(Rm) << amount. Rd and
/// Rn may be SP; Rm 31 is the zero register.
pub fn add_ext(rd: Reg, rn: Reg, rm: Reg, option: u32, amount: u32) -> u32 {
    assert!(amount <= 4);
    0x8b20_0000 | rm << 16 | option << 13 | amount << 10 | rn << 5 | rd
}

pub fn sub_ext(rd: Reg, rn: Reg, rm: Reg, option: u32, amount: u32) -> u32 {
    assert!(amount <= 4);
    0xcb20_0000 | rm << 16 | option << 13 | amount << 10 | rn << 5 | rd
}

pub const LSL: u32 = 0;
pub const LSR: u32 = 1;
pub const ASR: u32 = 2;

/// ADD (shifted register); 31 is the zero register throughout.
pub fn add_reg(sf: bool, rd: Reg, rn: Reg, rm: Reg, shift: u32, amount: u32) -> u32 {
    (sf as u32) << 31 | 0x0b00_0000 | shift << 22 | rm << 16 | amount << 10 | rn << 5 | rd
}

pub fn orr_reg(sf: bool, rd: Reg, rn: Reg, rm: Reg, shift: u32, amount: u32) -> u32 {
    (sf as u32) << 31 | 0x2a00_0000 | shift << 22 | rm << 16 | amount << 10 | rn << 5 | rd
}

pub fn eor_reg(sf: bool, rd: Reg, rn: Reg, rm: Reg, shift: u32, amount: u32) -> u32 {
    (sf as u32) << 31 | 0x4a00_0000 | shift << 22 | rm << 16 | amount << 10 | rn << 5 | rd
}

pub fn and_reg(sf: bool, rd: Reg, rn: Reg, rm: Reg, shift: u32, amount: u32) -> u32 {
    (sf as u32) << 31 | 0x0a00_0000 | shift << 22 | rm << 16 | amount << 10 | rn << 5 | rd
}

/// BIC (shifted register): Rd = Rn & !(Rm shifted).
pub fn bic_reg(sf: bool, rd: Reg, rn: Reg, rm: Reg, shift: u32, amount: u32) -> u32 {
    (sf as u32) << 31 | 0x0a20_0000 | shift << 22 | rm << 16 | amount << 10 | rn << 5 | rd
}

/// The N:immr:imms fields of a logical immediate, if `value` is one.
pub fn logical_imm(value: u64, sf: bool) -> Option<u32> {
    let value = if sf {
        value
    } else {
        let low = value & 0xffff_ffff;
        if value >> 32 != 0 && value >> 32 != low {
            return None;
        }
        low | low << 32
    };
    if value == 0 || value == u64::MAX {
        return None;
    }
    // The smallest element size the value repeats with.
    let mut size = 64u32;
    while size > 2 {
        let half = size / 2;
        let mask = (1u64 << half) - 1;
        if (value & mask) != ((value >> half) & mask) {
            break;
        }
        size = half;
    }
    let mask = if size == 64 {
        u64::MAX
    } else {
        (1u64 << size) - 1
    };
    let element = value & mask;
    let ones = element.count_ones();
    // Rotate the element right until its run of ones starts at bit 0.
    let rotr = |x: u64, r: u32| {
        if r == 0 {
            x
        } else {
            ((x >> r) | (x << (size - r))) & mask
        }
    };
    let run = (1u64 << ones) - 1;
    // A rotated run has exactly one bit that is set while the bit below it
    // (cyclically) is clear: where the run starts.
    let below = ((element << 1) | (element >> (size - 1))) & mask;
    let starts = element & !below;
    if starts.count_ones() != 1 {
        return None;
    }
    let rotation = starts.trailing_zeros();
    if rotr(element, rotation) != run {
        return None;
    }
    // immr rotates the run back into place: right by (size - rotation).
    let immr = (size - rotation) % size;
    let n = (size == 64) as u32;
    let imms = ((!(size * 2 - 1) & 0x3f) | (ones - 1)) & 0x3f;
    if !sf && n == 1 {
        return None;
    }
    Some(n << 12 | immr << 6 | imms)
}

pub fn and_imm(sf: bool, rd: Reg, rn: Reg, value: u64) -> u32 {
    let f = logical_imm(value, sf).expect("not a logical immediate");
    (sf as u32) << 31 | 0x1200_0000 | f << 10 | rn << 5 | rd
}

pub fn orr_imm(sf: bool, rd: Reg, rn: Reg, value: u64) -> u32 {
    let f = logical_imm(value, sf).expect("not a logical immediate");
    (sf as u32) << 31 | 0x3200_0000 | f << 10 | rn << 5 | rd
}

/// UBFM; LSR, LSL and UBFX are aliases of it.
pub fn ubfm(sf: bool, rd: Reg, rn: Reg, immr: u32, imms: u32) -> u32 {
    (sf as u32) << 31 | 0x5300_0000 | (sf as u32) << 22 | immr << 16 | imms << 10 | rn << 5 | rd
}

pub fn ubfx(rd: Reg, rn: Reg, lsb: u32, width: u32) -> u32 {
    ubfm(true, rd, rn, lsb, lsb + width - 1)
}

pub fn lsr_imm(rd: Reg, rn: Reg, shift: u32) -> u32 {
    ubfm(true, rd, rn, shift, 63)
}

pub fn lsl_imm(rd: Reg, rn: Reg, shift: u32) -> u32 {
    ubfm(true, rd, rn, (64 - shift) % 64, 63 - shift)
}

/// BFM; BFI and BFXIL are aliases of it.
pub fn bfm(sf: bool, rd: Reg, rn: Reg, immr: u32, imms: u32) -> u32 {
    (sf as u32) << 31 | 0x3300_0000 | (sf as u32) << 22 | immr << 16 | imms << 10 | rn << 5 | rd
}

/// BFI: inserts the low `width` bits of Rn into Rd at `lsb`.
pub fn bfi(sf: bool, rd: Reg, rn: Reg, lsb: u32, width: u32) -> u32 {
    let size = if sf { 64 } else { 32 };
    bfm(sf, rd, rn, (size - lsb) % size, width - 1)
}

// ---------------------------------------------------------------- loads and stores

/// LDR/STR (unsigned immediate) of a general register: `size` is log2 of the
/// bytes, `offset` in bytes and a multiple of them.
pub fn ldr_imm(size: u32, rt: Reg, rn: Reg, offset: u32) -> u32 {
    ls_imm(size, 0b01, rt, rn, offset)
}

pub fn str_imm(size: u32, rt: Reg, rn: Reg, offset: u32) -> u32 {
    ls_imm(size, 0b00, rt, rn, offset)
}

fn ls_imm(size: u32, opc: u32, rt: Reg, rn: Reg, offset: u32) -> u32 {
    assert!(offset % (1 << size) == 0, "unaligned offset {offset}");
    let scaled = offset >> size;
    assert!(scaled < 4096, "offset {offset} out of range");
    size << 30 | 0x3900_0000 | opc << 22 | scaled << 10 | rn << 5 | rt
}

/// LDR/STR (register offset), 64-bit offset register shifted by `scaled` ×
/// size: the form the translator's final access takes.
pub fn ldr_reg(size: u32, rt: Reg, rn: Reg, rm: Reg) -> u32 {
    size << 30 | 0x3860_6800 | rm << 16 | rn << 5 | rt
}

pub fn str_reg(size: u32, rt: Reg, rn: Reg, rm: Reg) -> u32 {
    size << 30 | 0x3820_6800 | rm << 16 | rn << 5 | rt
}

/// LDP/STP of 64-bit registers, signed offset.
pub fn ldp(rt: Reg, rt2: Reg, rn: Reg, offset: i32) -> u32 {
    assert!(offset % 8 == 0);
    0xa940_0000 | imm(offset as i64 / 8, 7, "ldp") << 15 | rt2 << 10 | rn << 5 | rt
}

pub fn stp(rt: Reg, rt2: Reg, rn: Reg, offset: i32) -> u32 {
    assert!(offset % 8 == 0);
    0xa900_0000 | imm(offset as i64 / 8, 7, "stp") << 15 | rt2 << 10 | rn << 5 | rt
}

/// LDP/STP of Q registers, signed offset.
pub fn ldp_q(rt: Reg, rt2: Reg, rn: Reg, offset: i32) -> u32 {
    assert!(offset % 16 == 0);
    0xad40_0000 | imm(offset as i64 / 16, 7, "ldp q") << 15 | rt2 << 10 | rn << 5 | rt
}

pub fn stp_q(rt: Reg, rt2: Reg, rn: Reg, offset: i32) -> u32 {
    assert!(offset % 16 == 0);
    0xad00_0000 | imm(offset as i64 / 16, 7, "stp q") << 15 | rt2 << 10 | rn << 5 | rt
}

/// STP of D registers, pre-index (with writeback).
pub fn stp_d_pre(rt: Reg, rt2: Reg, rn: Reg, offset: i32) -> u32 {
    assert!(offset % 8 == 0);
    0x6d80_0000 | imm(offset as i64 / 8, 7, "stp d") << 15 | rt2 << 10 | rn << 5 | rt
}

pub fn ldp_d(rt: Reg, rt2: Reg, rn: Reg, offset: i32) -> u32 {
    assert!(offset % 8 == 0);
    0x6d40_0000 | imm(offset as i64 / 8, 7, "ldp d") << 15 | rt2 << 10 | rn << 5 | rt
}

pub fn stp_d(rt: Reg, rt2: Reg, rn: Reg, offset: i32) -> u32 {
    assert!(offset % 8 == 0);
    0x6d00_0000 | imm(offset as i64 / 8, 7, "stp d") << 15 | rt2 << 10 | rn << 5 | rt
}

/// LDR (literal), 64-bit: `offset` bytes from this instruction.
pub fn ldr_literal(rt: Reg, offset: i64) -> u32 {
    assert!(offset % 4 == 0);
    0x5800_0000 | imm(offset / 4, 19, "ldr literal") << 5 | rt
}

pub fn adr(rd: Reg, offset: i64) -> u32 {
    let v = imm(offset, 21, "adr");
    0x1000_0000 | (v & 3) << 29 | (v >> 2) << 5 | rd
}

/// CAS{A}{L} of `size` (log2 bytes): compares Rs with memory at Rn, stores Rt
/// if equal; Rs gets the old value.
pub fn cas(size: u32, acquire: bool, release: bool, rs: Reg, rt: Reg, rn: Reg) -> u32 {
    size << 30
        | 0x08a0_7c00
        | (acquire as u32) << 22
        | rs << 16
        | (release as u32) << 15
        | rn << 5
        | rt
}

// ---------------------------------------------------------------- branches

pub fn b(offset: i64) -> u32 {
    assert!(offset % 4 == 0);
    0x1400_0000 | imm(offset / 4, 26, "b")
}

pub fn bl(offset: i64) -> u32 {
    assert!(offset % 4 == 0);
    0x9400_0000 | imm(offset / 4, 26, "bl")
}

pub fn b_cond(cond: u32, offset: i64) -> u32 {
    assert!(offset % 4 == 0);
    0x5400_0000 | imm(offset / 4, 19, "b.cond") << 5 | cond
}

pub fn cbz(sf: bool, rt: Reg, offset: i64) -> u32 {
    (sf as u32) << 31 | 0x3400_0000 | imm(offset / 4, 19, "cbz") << 5 | rt
}

pub fn cbnz(sf: bool, rt: Reg, offset: i64) -> u32 {
    (sf as u32) << 31 | 0x3500_0000 | imm(offset / 4, 19, "cbnz") << 5 | rt
}

pub fn tbz(rt: Reg, bit: u32, offset: i64) -> u32 {
    (bit >> 5) << 31 | 0x3600_0000 | (bit & 31) << 19 | imm(offset / 4, 14, "tbz") << 5 | rt
}

pub fn tbnz(rt: Reg, bit: u32, offset: i64) -> u32 {
    (bit >> 5) << 31 | 0x3700_0000 | (bit & 31) << 19 | imm(offset / 4, 14, "tbnz") << 5 | rt
}

pub fn br(rn: Reg) -> u32 {
    0xd61f_0000 | rn << 5
}

pub fn blr(rn: Reg) -> u32 {
    0xd63f_0000 | rn << 5
}

pub fn ret() -> u32 {
    0xd65f_03c0
}

/// RET Xn: a branch to Xn the CPU predicts as a subroutine return.
pub fn ret_reg(rn: Reg) -> u32 {
    0xd65f_0000 | rn << 5
}

// ---------------------------------------------------------------- system

/// MRS/MSR with the packed op0:op1:CRn:CRm:op2 encoding crate_core's SysReg uses.
pub fn mrs(rt: Reg, reg: u16) -> u32 {
    0xd530_0000 | (reg as u32) << 5 | rt
}

pub fn msr(reg: u16, rt: Reg) -> u32 {
    0xd510_0000 | (reg as u32) << 5 | rt
}

pub const NZCV: u16 = 0xda10;
pub const FPCR: u16 = 0xda20;
pub const FPSR: u16 = 0xda21;
pub const CNTVCT_EL0: u16 = 0xdf02;

pub fn nop() -> u32 {
    0xd503_201f
}

pub fn brk(imm16: u32) -> u32 {
    0xd420_0000 | (imm16 & 0xffff) << 5
}

pub fn dmb_ish() -> u32 {
    0xd503_3bbf
}

pub fn isb() -> u32 {
    0xd503_3fdf
}

// ---------------------------------------------------------------- assembling

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Label(u32);

#[derive(Clone, Copy)]
enum Fixup {
    /// imm26 of B/BL.
    Branch26,
    /// imm19 at bits 23:5 of B.cond, CBZ, CBNZ and LDR (literal).
    Branch19,
    /// imm14 at bits 18:5 of TBZ and TBNZ.
    Branch14,
    /// The immhi:immlo of ADR.
    Adr,
}

/// A block of host code being put together, in two sections: the code that
/// runs, and cold code (slow paths, exits taken once) that a code cache may
/// place elsewhere, away from the code that runs. Offsets are in bytes from
/// the start of their section; targets outside the block are absolute
/// addresses where the code will run, resolved once that address is known.
/// Only B and BL may cross from one section to the other.
#[derive(Default)]
pub struct Asm {
    pub words: Vec<u32>,
    cold: Vec<u32>,
    in_cold: bool,
    labels: Vec<Option<(bool, u32)>>,
    fixups: Vec<(bool, u32, Fixup, Label)>,
    external: Vec<(bool, u32, Fixup, u64)>,
    /// 64-bit words placed after the code of their section.
    literals: Vec<(bool, Label, Literal)>,
}

enum Literal {
    Values(Vec<u64>),
    /// The absolute address of a label.
    Address(Label),
}

impl Literal {
    fn len(&self) -> usize {
        match self {
            Literal::Values(v) => v.len(),
            Literal::Address(_) => 1,
        }
    }
}

impl Asm {
    pub fn new() -> Self {
        Asm {
            words: Vec::with_capacity(4096),
            cold: Vec::new(),
            in_cold: false,
            labels: Vec::with_capacity(256),
            fixups: Vec::with_capacity(256),
            external: Vec::with_capacity(64),
            literals: Vec::new(),
        }
    }

    fn literal_bytes(&self, cold: bool) -> usize {
        let words: usize = self
            .literals
            .iter()
            .filter(|l| l.0 == cold)
            .map(|l| l.2.len())
            .sum();
        words * 8
    }

    /// The most bytes `finish` can make of the code section: its words, an
    /// alignment word and the literals.
    pub fn size_bound(&self) -> usize {
        (self.words.len() + 1) * 4 + self.literal_bytes(false)
    }

    /// The most bytes `finish_split` can make of the cold section.
    pub fn cold_size(&self) -> usize {
        if self.cold.is_empty() && self.literal_bytes(true) == 0 {
            return 0;
        }
        (self.cold.len() + 1) * 4 + self.literal_bytes(true)
    }

    /// Makes what is emitted next go to the cold section (or back).
    pub fn set_cold(&mut self, cold: bool) {
        self.in_cold = cold;
    }

    pub fn is_cold(&self) -> bool {
        self.in_cold
    }

    fn section(&mut self) -> &mut Vec<u32> {
        if self.in_cold {
            &mut self.cold
        } else {
            &mut self.words
        }
    }

    /// The offset of the next instruction in the current section.
    pub fn here(&self) -> u32 {
        (if self.in_cold {
            &self.cold
        } else {
            &self.words
        }
        .len()
            * 4) as u32
    }

    pub fn emit(&mut self, word: u32) {
        self.section().push(word);
    }

    pub fn emit_all(&mut self, words: impl IntoIterator<Item = u32>) {
        self.section().extend(words);
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() as u32 - 1)
    }

    pub fn bind(&mut self, label: Label) {
        assert!(self.labels[label.0 as usize].is_none(), "label bound twice");
        self.labels[label.0 as usize] = Some((self.in_cold, self.here()));
    }

    pub fn is_bound(&self, label: Label) -> bool {
        self.labels[label.0 as usize].is_some()
    }

    /// A label's offset in the code section.
    pub fn offset_of(&self, label: Label) -> u32 {
        match self.labels[label.0 as usize].expect("label not bound") {
            (false, off) => off,
            (true, _) => panic!("label in the cold section"),
        }
    }

    fn emit_fixup(&mut self, word: u32, kind: Fixup, label: Label) {
        self.fixups.push((self.in_cold, self.here(), kind, label));
        self.emit(word);
    }

    pub fn b(&mut self, label: Label) {
        self.emit_fixup(0x1400_0000, Fixup::Branch26, label);
    }

    pub fn bl(&mut self, label: Label) {
        self.emit_fixup(0x9400_0000, Fixup::Branch26, label);
    }

    pub fn b_cond(&mut self, cond: u32, label: Label) {
        self.emit_fixup(0x5400_0000 | cond, Fixup::Branch19, label);
    }

    pub fn cbz(&mut self, sf: bool, rt: Reg, label: Label) {
        self.emit_fixup((sf as u32) << 31 | 0x3400_0000 | rt, Fixup::Branch19, label);
    }

    pub fn cbnz(&mut self, sf: bool, rt: Reg, label: Label) {
        self.emit_fixup((sf as u32) << 31 | 0x3500_0000 | rt, Fixup::Branch19, label);
    }

    pub fn tbz(&mut self, rt: Reg, bit: u32, label: Label) {
        self.emit_fixup(tbz(rt, bit, 0), Fixup::Branch14, label);
    }

    pub fn tbnz(&mut self, rt: Reg, bit: u32, label: Label) {
        self.emit_fixup(tbnz(rt, bit, 0), Fixup::Branch14, label);
    }

    pub fn adr(&mut self, rd: Reg, label: Label) {
        self.emit_fixup(0x1000_0000 | rd, Fixup::Adr, label);
    }

    pub fn ldr_literal(&mut self, rt: Reg, label: Label) {
        self.emit_fixup(0x5800_0000 | rt, Fixup::Branch19, label);
    }

    /// An instruction whose branch field (as `kind` of the word's class) is
    /// filled with the given label: for copied guest branches.
    pub fn emit_branch19(&mut self, word: u32, label: Label) {
        self.emit_fixup(word & !(0x7ffff << 5), Fixup::Branch19, label);
    }

    pub fn emit_branch14(&mut self, word: u32, label: Label) {
        self.emit_fixup(word & !(0x3fff << 5), Fixup::Branch14, label);
    }

    /// B or BL to an absolute address where code runs.
    pub fn b_abs(&mut self, target: u64) {
        self.external
            .push((self.in_cold, self.here(), Fixup::Branch26, target));
        self.emit(0x1400_0000);
    }

    pub fn bl_abs(&mut self, target: u64) {
        self.external
            .push((self.in_cold, self.here(), Fixup::Branch26, target));
        self.emit(0x9400_0000);
    }

    /// 64-bit words placed after the code of the current section, 8-byte
    /// aligned; the label marks the first.
    pub fn literal(&mut self, values: &[u64]) -> Label {
        let label = self.label();
        self.literals
            .push((self.in_cold, label, Literal::Values(values.to_vec())));
        label
    }

    /// A 64-bit word after the code of the current section holding the
    /// address where `of` will run.
    pub fn address_literal(&mut self, of: Label) -> Label {
        let label = self.label();
        self.literals
            .push((self.in_cold, label, Literal::Address(of)));
        label
    }

    pub fn mov_imm(&mut self, rd: Reg, value: u64) {
        let seq = mov_imm(rd, value);
        if seq.len() <= 2 {
            self.emit_all(seq);
        } else {
            let label = self.literal(&[value]);
            self.ldr_literal(rd, label);
        }
    }

    fn patch(word: u32, kind: Fixup, delta: i64) -> u32 {
        match kind {
            Fixup::Branch26 => word | imm(delta / 4, 26, "branch"),
            Fixup::Branch19 => word | imm(delta / 4, 19, "branch") << 5,
            Fixup::Branch14 => word | imm(delta / 4, 14, "branch") << 5,
            Fixup::Adr => {
                let v = imm(delta, 21, "adr");
                word | (v & 3) << 29 | (v >> 2) << 5
            }
        }
    }

    /// Lays out the literals and resolves every branch for code that will
    /// run at `base`, which has no cold section.
    pub fn finish(self, base: u64) -> Vec<u32> {
        assert!(self.cold.is_empty(), "cold code with nowhere to go");
        self.finish_split(base, 0).0
    }

    /// Lays out the literals and resolves every branch for a code section
    /// that will run at `base` and a cold section at `cold_base`.
    pub fn finish_split(mut self, base: u64, cold_base: u64) -> (Vec<u32>, Vec<u32>) {
        let literals = std::mem::take(&mut self.literals);
        let mut addresses = Vec::new();
        for cold in [false, true] {
            self.in_cold = cold;
            let mut first = true;
            for (_, label, literal) in literals.iter().filter(|l| l.0 == cold) {
                if first && self.here() % 8 != 0 {
                    self.emit(brk(0));
                }
                first = false;
                self.bind(*label);
                match literal {
                    Literal::Values(values) => {
                        for &v in values {
                            self.emit(v as u32);
                            self.emit((v >> 32) as u32);
                        }
                    }
                    Literal::Address(of) => {
                        addresses.push((cold, self.here(), *of));
                        self.emit_all([0, 0]);
                    }
                }
            }
        }
        let addr = |cold: bool, off: u32| if cold { cold_base } else { base } + off as u64;
        for (cold, at, of) in addresses {
            let (to_cold, to) = self.labels[of.0 as usize].expect("label never bound");
            let value = addr(to_cold, to);
            let words = if cold {
                &mut self.cold
            } else {
                &mut self.words
            };
            let i = at as usize / 4;
            words[i] = value as u32;
            words[i + 1] = (value >> 32) as u32;
        }
        for &(cold, at, kind, label) in &self.fixups {
            let (to_cold, to) = self.labels[label.0 as usize].expect("label never bound");
            let delta = addr(to_cold, to).wrapping_sub(addr(cold, at)) as i64;
            let words = if cold {
                &mut self.cold
            } else {
                &mut self.words
            };
            let i = at as usize / 4;
            words[i] = Self::patch(words[i], kind, delta);
        }
        for &(cold, at, kind, target) in &self.external {
            let delta = target.wrapping_sub(addr(cold, at)) as i64;
            let words = if cold {
                &mut self.cold
            } else {
                &mut self.words
            };
            let i = at as usize / 4;
            words[i] = Self::patch(words[i], kind, delta);
        }
        (self.words, self.cold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_like_the_manual() {
        assert_eq!(movz(true, 0, 0x1234, 0), 0xd282_4680);
        assert_eq!(movk(true, 0, 0x5678, 1), 0xf2aa_cf00);
        assert_eq!(mov(1, 2), 0xaa02_03e1);
        assert_eq!(mov_sp(0, SP), 0x9100_03e0);
        assert_eq!(add_imm(true, 16, 2, 8, false), 0x9100_2050);
        assert_eq!(sub_imm(true, 27, 27, 16, false), 0xd100_437b);
        assert_eq!(ldr_imm(3, 1, 28, 0x100), 0xf940_8381);
        assert_eq!(str_imm(2, 16, 28, 0x110), 0xb901_1390);
        assert_eq!(ldr_reg(3, 1, 16, 17), 0xf871_6a01);
        assert_eq!(str_reg(0, 1, 16, 17), 0x3831_6a01);
        assert_eq!(ldp(0, 1, 28, 0), 0xa940_0780);
        assert_eq!(stp(29, 30, SP, -16), 0xa93f_7bfd);
        assert_eq!(stp_q(0, 1, 0, 32), 0xad01_0400);
        assert_eq!(eor_reg(true, 17, 17, 16, LSR, 12), 0xca50_3231);
        assert_eq!(ubfx(17, 16, 12, 8), 0xd34c_4e11);
        assert_eq!(lsr_imm(0, 1, 4), 0xd344_fc20);
        assert_eq!(lsl_imm(0, 1, 4), 0xd37c_ec20);
        assert_eq!(add_ext(16, 2, 3, EXT_UXTX, 3), 0x8b23_6c50);
        assert_eq!(br(17), 0xd61f_0220);
        assert_eq!(blr(26), 0xd63f_0340);
        assert_eq!(mrs(0, NZCV), 0xd53b_4200);
        assert_eq!(msr(NZCV, 0), 0xd51b_4200);
        assert_eq!(mrs(16, CNTVCT_EL0), 0xd53b_e050);
        assert_eq!(cbnz(false, 16, 8), 0x3500_0050);
        assert_eq!(tbnz(17, 12, 8), 0x3760_0051);
        assert_eq!(b(-4), 0x17ff_ffff);
        assert_eq!(adr(17, 12), 0x1000_0071);
        assert_eq!(cas(3, false, false, 17, 1, 16), 0xc8b1_7e01);
        assert_eq!(ldr_literal(16, 8), 0x5800_0050);
    }

    #[test]
    fn logical_immediates() {
        // and x0, x0, #0xfffffffffffff000
        assert_eq!(and_imm(true, 0, 0, 0xffff_ffff_ffff_f000), 0x9274_cc00);
        // and x17, x16, #7
        assert_eq!(and_imm(true, 17, 16, 7), 0x9240_0a11);
        // and w0, w1, #0xff00ff00
        assert_eq!(and_imm(false, 0, 1, 0xff00_ff00), 0x1208_9c20);
        // orr x0, xzr, #0x5555555555555555
        assert_eq!(orr_imm(true, 0, 31, 0x5555_5555_5555_5555), 0xb200_f3e0);
        assert_eq!(logical_imm(0, true), None);
        assert_eq!(logical_imm(0x1234, true), None);
        assert_eq!(logical_imm(1 << 40, false), None);
    }

    #[test]
    fn constants_take_short_sequences() {
        assert_eq!(mov_imm(0, 0), vec![movz(true, 0, 0, 0)]);
        assert_eq!(mov_imm(0, u64::MAX), vec![movn(true, 0, 0, 0)]);
        assert_eq!(mov_imm(0, 0xffff_ffff_ffff_fffe), vec![movn(true, 0, 1, 0)]);
        assert_eq!(
            mov_imm(3, 0x1_0000_0002),
            vec![movz(true, 3, 2, 0), movk(true, 3, 1, 2)]
        );
        assert_eq!(mov_imm(3, 0xffff_8000_0000_0000).len(), 1);
    }

    #[test]
    fn cold_code_and_address_literals() {
        let mut a = Asm::new();
        let slow = a.label();
        let back = a.label();
        a.b(slow);
        a.bind(back);
        a.emit(ret());
        a.set_cold(true);
        a.bind(slow);
        let lit = a.address_literal(back);
        a.ldr_literal(0, lit);
        a.b(back);
        let (hot, cold) = a.finish_split(0x1000, 0x8000);
        assert_eq!(hot, vec![b(0x8000 - 0x1000), ret()]);
        assert_eq!(cold[0], ldr_literal(0, 8));
        assert_eq!(cold[1], b(0x1004 - 0x8004));
        assert_eq!(cold[2] as u64 | (cold[3] as u64) << 32, 0x1004);
    }

    #[test]
    fn resolves_labels_literals_and_external_targets() {
        let mut a = Asm::new();
        let end = a.label();
        a.cbz(true, 0, end);
        let lit = a.literal(&[0x1122_3344_5566_7788]);
        a.ldr_literal(1, lit);
        a.b_abs(0x1000);
        a.bind(end);
        a.emit(ret());
        let words = a.finish(0x2000);
        assert_eq!(words[0], cbz(true, 0, 12));
        assert_eq!(words[1], ldr_literal(1, 12)); // at 4; literal at 16
        assert_eq!(words[2], b(0x1000 - 0x2008));
        assert_eq!(words[3], ret());
        assert_eq!(words[4], 0x5566_7788);
        assert_eq!(words[5], 0x1122_3344);
    }
}
