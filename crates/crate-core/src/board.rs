// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A built machine as a CPU backend sees it, and the CPU backend interface.
//!
//! A machine (crate-virt's `virt`, say) assembles the board: RAM, devices on
//! the MMIO bus, an interrupt controller, the CPUs and where the first one
//! starts. A CPU backend runs guest code on it. Neither knows how the other is
//! built, so the same board runs on the translator and on a hypervisor.

use std::sync::Arc;

use crate::Result;
use crate::bus::MmioBus;
use crate::cpu::CpuInterrupts;
use crate::irq::InterruptController;
use crate::memory::GuestMemory;
use crate::psci::Psci;
use crate::run::{RunControl, StopReason};

/// Registers of the boot CPU when the machine starts. It starts at EL1 with
/// the MMU off and interrupts masked, as Linux's boot protocol asks.
#[derive(Clone, Copy, Debug, Default)]
pub struct BootState {
    pub pc: u64,
    pub x: [u64; 4],
}

pub struct Board {
    pub memory: Arc<GuestMemory>,
    pub mmio: MmioBus,
    /// The interrupt controller as the CPUs see it.
    pub interrupts: Arc<dyn CpuInterrupts>,
    /// The same controller's input lines, for what the backend raises itself
    /// (the timers).
    pub lines: Arc<dyn InterruptController>,
    /// Each CPU's MPIDR_EL1, which is how the guest and the interrupt
    /// controller name it.
    pub mpidrs: Vec<u64>,
    pub boot: BootState,
    /// The private interrupt the virtual timer raises (11 is INTID 27).
    pub vtimer_ppi: u32,
    /// The private interrupt the EL1 physical timer raises (14 is INTID 30).
    pub ptimer_ppi: u32,
    pub psci: Arc<Psci>,
    pub run: Arc<RunControl>,
}

impl Board {
    pub fn cpu_count(&self) -> usize {
        self.mpidrs.len()
    }
}

/// Runs guest code on a board.
pub trait CpuBackend {
    fn name(&self) -> &'static str;
    /// Runs the board's CPUs until the machine stops, and says why it did.
    fn run(&self, board: &Board) -> Result<StopReason>;
}
