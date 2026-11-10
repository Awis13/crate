// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! What a CPU backend needs from the rest of the machine besides memory and
//! MMIO: the interrupt controller's view of each CPU, and a way for the
//! machine to wake a CPU that sleeps or runs guest code.

use std::fmt;
use std::sync::Arc;

/// A system register, as MRS and MSR encode it: op0, op1, CRn, CRm and op2,
/// packed in that order into 16 bits (op0 in bits 15:14, op2 in bits 2:0).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SysReg(pub u16);

impl SysReg {
    pub const fn new(op0: u8, op1: u8, crn: u8, crm: u8, op2: u8) -> Self {
        SysReg(
            ((op0 as u16 & 3) << 14)
                | ((op1 as u16 & 7) << 11)
                | ((crn as u16 & 15) << 7)
                | ((crm as u16 & 15) << 3)
                | (op2 as u16 & 7),
        )
    }

    /// From the ISS of an exception syndrome for a trapped MSR or MRS
    /// (exception class 0x18): Op0 in bits 21:20, Op2 19:17, Op1 16:14,
    /// CRn 13:10 and CRm 4:1.
    pub const fn from_iss(iss: u32) -> Self {
        SysReg::new(
            ((iss >> 20) & 3) as u8,
            ((iss >> 14) & 7) as u8,
            ((iss >> 10) & 15) as u8,
            ((iss >> 1) & 15) as u8,
            ((iss >> 17) & 7) as u8,
        )
    }

    pub const fn op0(self) -> u8 {
        (self.0 >> 14) as u8 & 3
    }
    pub const fn op1(self) -> u8 {
        (self.0 >> 11) as u8 & 7
    }
    pub const fn crn(self) -> u8 {
        (self.0 >> 7) as u8 & 15
    }
    pub const fn crm(self) -> u8 {
        (self.0 >> 3) as u8 & 15
    }
    pub const fn op2(self) -> u8 {
        self.0 as u8 & 7
    }
}

impl fmt::Debug for SysReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "S{}_{}_C{}_C{}_{}",
            self.op0(),
            self.op1(),
            self.crn(),
            self.crm(),
            self.op2()
        )
    }
}

impl fmt::Display for SysReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Implemented by a CPU backend: makes `cpu` look at its interrupt state again,
/// leaving guest code if it runs it and waking up if it waits for an interrupt.
pub trait CpuWaker: Send + Sync {
    fn wake(&self, cpu: usize);
}

/// The interrupt controller as the CPUs see it: the IRQ and FIQ signals into
/// each one, and the system registers the controller owns (a GICv3's CPU
/// interface is reached through system registers rather than MMIO).
pub trait CpuInterrupts: Send + Sync {
    fn irq_pending(&self, cpu: usize) -> bool;
    fn fiq_pending(&self, cpu: usize) -> bool;
    /// `None` if the register is not the controller's.
    fn read_sysreg(&self, cpu: usize, reg: SysReg) -> Option<u64>;
    /// `false` if the register is not the controller's.
    fn write_sysreg(&self, cpu: usize, reg: SysReg, value: u64) -> bool;
    /// Called once by the backend before the machine runs.
    fn set_waker(&self, waker: Arc<dyn CpuWaker>);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_like_the_instruction_encoding() {
        // MPIDR_EL1 is S3_0_C0_C0_5; CNTV_CTL_EL0 is S3_3_C14_C3_1.
        assert_eq!(SysReg::new(3, 0, 0, 0, 5).0, 0xc005);
        assert_eq!(SysReg::new(3, 3, 14, 3, 1).0, 0xdf19);
        assert_eq!(format!("{}", SysReg(0xdf19)), "S3_3_C14_C3_1");
    }

    #[test]
    fn decodes_a_trapped_access() {
        // MRS x3, ICC_IAR1_EL1 (S3_0_C12_C12_0): Op0 3, Op2 0, Op1 0, CRn 12,
        // Rt 3, CRm 12, read.
        let iss = (3 << 20) | (12 << 10) | (3 << 5) | (12 << 1) | 1;
        assert_eq!(SysReg::from_iss(iss), SysReg::new(3, 0, 12, 12, 0));
    }
}
