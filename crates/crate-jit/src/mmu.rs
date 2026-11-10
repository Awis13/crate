// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Stage-1 address translation (Arm ARM, VMSAv8-64): the guest's page-table
//! walk for the 4 KiB, 16 KiB and 64 KiB granules, with the access-permission
//! and access-flag checks. The translator calls this to fill its software TLB
//! and on every TLB miss; the guest's page size comes from its TCR, never from
//! configuration.
//!
//! It covers EL0 and EL1 stage-1 translation with one translation regime
//! (TTBR0/TTBR1 chosen by the top address bit). There is no stage 2 (no nested
//! guest) and no EL2/EL3 regime.

use crate_core::memory::GuestMemory;

use crate::state::MmuRegs;

/// What a translated access is for; it decides which permission bits matter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Access {
    Read,
    Write,
    Execute,
}

/// A translation fault, as its fault-status code and level go into ESR_EL1.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fault {
    /// The 6-bit (I/D)FSC.
    pub fsc: u8,
    /// True for a write (the ESR's WnR bit for a data abort).
    pub write: bool,
}

impl Fault {
    const ADDRESS_SIZE: u8 = 0x00;
    const TRANSLATION: u8 = 0x04;
    const ACCESS_FLAG: u8 = 0x08;
    const PERMISSION: u8 = 0x0c;
    pub const ALIGNMENT: u8 = 0x21;

    fn at(base: u8, level: i32, write: bool) -> Fault {
        Fault {
            fsc: base | (level.clamp(0, 3) as u8),
            write,
        }
    }
}

/// A successful translation: the physical address and whether the page may be
/// written (so a store TLB entry is only filled for writable pages).
#[derive(Clone, Copy, Debug)]
pub struct Translation {
    pub pa: u64,
    pub writable: bool,
    pub page_shift: u32,
}

const SCTLR_M: u64 = 1 << 0;

/// The granule chosen for a translation regime, with the per-level geometry.
#[derive(Clone, Copy)]
struct Granule {
    page_shift: u32,
    /// Bits resolved per level (page_shift - 3).
    bits: u32,
}

impl Granule {
    /// From TCR's two-bit TG field. TTBR0 and TTBR1 encode the granule
    /// differently, so the caller passes the decoded page shift.
    fn new(page_shift: u32) -> Self {
        Granule {
            page_shift,
            bits: page_shift - 3,
        }
    }

    /// The first (highest) level a walk of `ia` input bits starts at.
    fn start_level(&self, ia: u32) -> i32 {
        // Level L resolves bits [bottom .. bottom+bits), bottom = p + (3-L)*bits.
        // The start level is the highest level whose bottom is below `ia`.
        let mut level = 3i32;
        while level > 0 {
            let bottom = self.page_shift + ((3 - (level - 1)) as u32) * self.bits;
            if bottom >= ia {
                break;
            }
            level -= 1;
        }
        level
    }
}

/// TG0 (bits 15:14 of TCR): 0b00=4K, 0b01=64K, 0b10=16K.
fn tg0_page_shift(tcr: u64) -> Option<u32> {
    match (tcr >> 14) & 3 {
        0b00 => Some(12),
        0b01 => Some(16),
        0b10 => Some(14),
        _ => None,
    }
}

/// TG1 (bits 31:30 of TCR): 0b01=16K, 0b10=4K, 0b11=64K (a different map from TG0).
fn tg1_page_shift(tcr: u64) -> Option<u32> {
    match (tcr >> 30) & 3 {
        0b10 => Some(12),
        0b11 => Some(16),
        0b01 => Some(14),
        _ => None,
    }
}

/// The page shift the guest's TTBR0 regime uses, from TCR.TG0.
pub fn ttbr0_page_shift(regs: &MmuRegs) -> u32 {
    tg0_page_shift(regs.tcr_el1).unwrap_or(12)
}

/// Translates guest virtual `va` for an access at `el` (0 or 1).
pub fn translate(
    mem: &GuestMemory,
    regs: &MmuRegs,
    el: u32,
    va: u64,
    access: Access,
) -> Result<Translation, Fault> {
    let write = access == Access::Write;
    if regs.sctlr_el1 & SCTLR_M == 0 {
        // The MMU is off: a flat map, and every access is permitted.
        return Ok(Translation {
            pa: va,
            writable: true,
            page_shift: ttbr0_page_shift(regs),
        });
    }

    let tcr = regs.tcr_el1;
    let t0sz = (tcr & 0x3f) as u32;
    let t1sz = ((tcr >> 16) & 0x3f) as u32;
    let epd0 = tcr & (1 << 7) != 0;
    let epd1 = tcr & (1 << 23) != 0;

    // The top bit picks the translation regime. VA bits above the input size
    // must be all zero (TTBR0) or all one (TTBR1), else a translation fault.
    let top = va >> 55 & 1;
    let (ttbr, tnsz, page_shift, disabled) = if top == 0 {
        (regs.ttbr0_el1, t0sz, tg0_page_shift(tcr), epd0)
    } else {
        (regs.ttbr1_el1, t1sz, tg1_page_shift(tcr), epd1)
    };
    let Some(page_shift) = page_shift else {
        return Err(Fault::at(Fault::TRANSLATION, 0, write));
    };
    if disabled {
        return Err(Fault::at(Fault::TRANSLATION, 0, write));
    }
    let ia = 64 - tnsz;
    // Bits [63:ia] must match the regime's sign.
    let want = if top == 0 { 0 } else { (1u64 << (64 - ia)) - 1 };
    if (va >> ia) != want {
        return Err(Fault::at(Fault::TRANSLATION, 0, write));
    }

    let g = Granule::new(page_shift);
    let mut level = g.start_level(ia);
    let mut table = ttbr & 0xffff_ffff_fffe & !((1u64 << page_shift) - 1);
    // The output-address mask: bits [47:0] (PA size capped at 48 here).
    let oa_mask = ((1u64 << 48) - 1) & !((1u64 << page_shift) - 1);

    loop {
        let bottom = page_shift + ((3 - level) as u32) * g.bits;
        // The index may be narrower than `bits` at the start level.
        let index_bits = (ia - bottom).min(g.bits);
        let index = (va >> bottom) & ((1 << index_bits) - 1);
        let entry_addr = table + index * 8;
        let desc = mem
            .read_u64(entry_addr)
            .map_err(|_| Fault::at(Fault::ADDRESS_SIZE, level, write))?;

        if desc & 1 == 0 {
            // Bit 0 clear: an invalid descriptor.
            return Err(Fault::at(Fault::TRANSLATION, level, write));
        }
        let is_table = desc & 2 != 0 && level < 3;
        let is_block = desc & 2 == 0 && level < 3;
        let is_page = desc & 3 == 3 && level == 3;
        if is_table {
            table = desc & oa_mask;
            level += 1;
            continue;
        }
        if !is_block && !is_page {
            // A block at level 3, or bits 0b01 at level 3: invalid.
            return Err(Fault::at(Fault::TRANSLATION, level, write));
        }

        // A leaf. Assemble the output address from the descriptor and the
        // low bits of the virtual address.
        let block_mask = (1u64 << bottom) - 1;
        let pa = (desc & oa_mask & !block_mask) | (va & block_mask);

        // Access flag (bit 10) and permissions (AP[2:1] in bits 7:6).
        if desc & (1 << 10) == 0 {
            return Err(Fault::at(Fault::ACCESS_FLAG, level, write));
        }
        let ap = ((desc >> 6) & 3) as u32;
        let pxn = desc & (1 << 53) != 0;
        let uxn = desc & (1 << 54) != 0;
        let el0_write = ap == 0b01;
        let (read_ok, write_ok, exec_ok) = permissions(el, ap, pxn, uxn, el0_write, regs.sctlr_el1);

        let ok = match access {
            Access::Read => read_ok,
            Access::Write => write_ok,
            Access::Execute => exec_ok,
        };
        if !ok {
            return Err(Fault::at(Fault::PERMISSION, level, write));
        }
        return Ok(Translation {
            pa,
            writable: write_ok,
            page_shift,
        });
    }
}

/// Read/write/execute permission at the current EL from AP and the execute-
/// never bits. The architecture forces no execute on a region writable at EL0
/// when seen from EL1, and SCTLR.WXN turns any writable region execute-never.
fn permissions(
    el: u32,
    ap: u32,
    pxn: bool,
    uxn: bool,
    el0_write: bool,
    sctlr: u64,
) -> (bool, bool, bool) {
    // AP[2:1]: bit1 (value 0b01/0b11) grants EL0; bit2 (0b10/0b11) is read-only.
    let el0 = ap & 1 != 0;
    let read_only = ap & 2 != 0;
    let wxn = sctlr & (1 << 19) != 0;

    let (read, write) = if el == 0 {
        if el0 {
            (true, !read_only)
        } else {
            (false, false)
        }
    } else {
        // EL1 always reads; it writes unless read-only.
        (true, !read_only)
    };

    let xn = if el == 0 {
        uxn || !el0 // a page EL0 cannot read, it cannot execute
    } else {
        pxn || el0_write // EL0-writable is privileged-execute-never
    };
    let exec = !xn && read && !(wxn && write);
    (read, write, exec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate_core::memory::{HeapRam, RamBlock};

    const RAM: u64 = 0x4000_0000;

    fn memory() -> (Arc<GuestMemory>, u64) {
        let block: Arc<dyn RamBlock> = Arc::new(HeapRam::new(0x20_0000).unwrap());
        let mem = Arc::new(GuestMemory::new(vec![(RAM, block)]).unwrap());
        (mem, RAM)
    }

    /// Builds a flat identity map with one leaf for `va` at the given granule
    /// and returns TCR/TTBR0 for it.
    fn identity_map(mem: &GuestMemory, base: u64, page_shift: u32, va: u64, ap: u32) -> (u64, u64) {
        let bits = page_shift - 3;
        let ia = 48u32;
        let g = Granule::new(page_shift);
        let start = g.start_level(ia);
        // Each level gets its own table page.
        let mut tables = Vec::new();
        for i in 0..=(3 - start) {
            tables.push(base + (i as u64) * (1 << page_shift));
        }
        let leaf_table = *tables.last().unwrap();
        let mut level = start;
        let mut table_phys = tables[0];
        let mut ti = 0;
        while level < 3 {
            let bottom = page_shift + ((3 - level) as u32) * bits;
            let index_bits = (ia - bottom).min(bits);
            let index = (va >> bottom) & ((1 << index_bits) - 1);
            let next = tables[ti + 1];
            mem.write_u64(table_phys + index * 8, next | 3).unwrap();
            table_phys = next;
            ti += 1;
            level += 1;
        }
        // The level-3 page descriptor: AF set, SH, AttrIndx 0, AP.
        let index = (va >> page_shift) & ((1 << bits) - 1);
        let desc = (va & !((1u64 << page_shift) - 1)) | 3 | (1 << 10) | ((ap as u64) << 6);
        mem.write_u64(leaf_table + index * 8, desc).unwrap();

        let tg0 = match page_shift {
            12 => 0,
            14 => 2,
            16 => 1,
            _ => unreachable!(),
        };
        // TCR for the test map: T0SZ = 16 (a 48-bit input) and this granule's
        // TG0; the MMU is turned on through SCTLR in `regs`, not here.
        let tcr = 16 | (tg0 << 14);
        (tcr, tables[0])
    }

    fn regs(tcr: u64, ttbr0: u64) -> MmuRegs {
        MmuRegs {
            sctlr_el1: SCTLR_M,
            ttbr0_el1: ttbr0,
            tcr_el1: tcr,
            ..Default::default()
        }
    }

    #[test]
    fn mmu_off_is_flat() {
        let (mem, _) = memory();
        let r = MmuRegs::default();
        let t = translate(&mem, &r, 1, 0x1234, Access::Write).unwrap();
        assert_eq!(t.pa, 0x1234);
    }

    #[test]
    fn walks_each_granule() {
        for &ps in &[12u32, 14, 16] {
            let (mem, base) = memory();
            let va = 0x4008_1240 & !((1u64 << ps) - 1);
            let (tcr, ttbr0) = identity_map(&mem, base + 0x10_0000, ps, va, 0b01);
            let r = regs(tcr, ttbr0);
            let t = translate(&mem, &r, 1, va + 0x30, Access::Read).unwrap();
            assert_eq!(t.pa, va + 0x30, "granule {ps}");
            assert_eq!(t.page_shift, ps);
            assert!(translate(&mem, &r, 0, va, Access::Read).is_ok());
        }
    }

    #[test]
    fn faults_on_an_unmapped_page() {
        let (mem, base) = memory();
        let va = 0x4008_0000;
        let (tcr, ttbr0) = identity_map(&mem, base + 0x10_0000, 12, va, 0b01);
        let r = regs(tcr, ttbr0);
        let f = translate(&mem, &r, 1, va + 0x1_0000, Access::Read).unwrap_err();
        assert_eq!(f.fsc & 0x3c, Fault::TRANSLATION, "a translation fault");
    }

    #[test]
    fn honours_permissions() {
        let (mem, base) = memory();
        let va = 0x4008_0000;
        // AP 0b10: EL1 read-only, EL0 no access.
        let (tcr, ttbr0) = identity_map(&mem, base + 0x10_0000, 12, va, 0b10);
        let r = regs(tcr, ttbr0);
        assert!(translate(&mem, &r, 1, va, Access::Read).is_ok());
        assert_eq!(
            translate(&mem, &r, 1, va, Access::Write).unwrap_err().fsc & 0x3c,
            Fault::PERMISSION
        );
        assert_eq!(
            translate(&mem, &r, 0, va, Access::Read).unwrap_err().fsc & 0x3c,
            Fault::PERMISSION
        );
    }

    #[test]
    fn access_flag_faults() {
        let (mem, base) = memory();
        let va = 0x4008_0000;
        let (tcr, ttbr0) = identity_map(&mem, base + 0x10_0000, 12, va, 0b01);
        // Clear AF on the leaf.
        let leaf = base + 0x10_0000 + 3 * 0x1000;
        let index = (va >> 12) & 0x1ff;
        let desc = mem.read_u64(leaf + index * 8).unwrap() & !(1 << 10);
        mem.write_u64(leaf + index * 8, desc).unwrap();
        let r = regs(tcr, ttbr0);
        assert_eq!(
            translate(&mem, &r, 1, va, Access::Read).unwrap_err().fsc & 0x3c,
            Fault::ACCESS_FLAG
        );
    }
}
