// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A GICv3 interrupt controller (Arm IHI 0069): the distributor, one
//! redistributor per CPU, and each CPU's interface, which the guest reaches
//! through the ICC_* system registers.
//!
//! The whole model is emulated here rather than borrowed from a hypervisor, so
//! the translator gets the same controller the hvf backend does. The CPU
//! backend asks it whether a CPU's IRQ or FIQ signal is asserted and hands it
//! the ICC_* accesses the guest makes.
//!
//! What it is:
//! - A single security state (GICD_CTLR.DS reads as one), affinity routing
//!   always on, no LPIs and no ITS: interrupts come from wires, not messages.
//! - Five priority bits (0x00 to 0xf8 in steps of 8). Preemption compares all
//!   five, which is what the smallest binary point gives; a larger binary
//!   point written to ICC_BPR* is kept but does not coarsen preemption.
//! - Group 0 interrupts signal FIQ, group 1 interrupts IRQ.
//! - Each interrupt is enabled or not, pending or not, active or not. A
//!   level-sensitive interrupt is pending while its line is high or its
//!   pending latch is set; an edge-triggered one only by its latch, which a
//!   rising edge sets.

use std::sync::{Arc, Mutex, RwLock};

use crate_core::bus::MmioDevice;
use crate_core::cpu::{CpuInterrupts, CpuWaker, SysReg};
use crate_core::irq::InterruptController;

pub const DISTRIBUTOR_SIZE: u64 = 0x1_0000;
/// Each CPU's redistributor: the RD_base frame, then the SGI_base frame.
pub const REDISTRIBUTOR_SIZE: u64 = 0x2_0000;

const PRIVATE: usize = 32;
const SPURIOUS: u64 = 1023;
const PRIORITY_MASK: u8 = 0xf8;
const IDLE_PRIORITY: u16 = 0x100;

const CTLR_ENABLE_GRP0: u32 = 1 << 0;
const CTLR_ENABLE_GRP1: u32 = 1 << 1;
const CTLR_ARE: u32 = 1 << 4;
const CTLR_DS: u32 = 1 << 6;

/// GICv3 in PIDR2's ArchRev field, which Linux checks in both frames.
const PIDR2_GICV3: u64 = 0x30;
/// IIDR: implementer 0 and product 0x63 ('c'), so no erratum workaround in a
/// guest matches it.
const IIDR: u64 = 0x6300_0000;

const IROUTER_IRM: u64 = 1 << 31;
/// Aff3 in bits 39:32, IRM in bit 31, Aff2 to Aff0 in bits 23:0.
const IROUTER_MASK: u64 = 0xff_0000_0000 | IROUTER_IRM | 0xff_ffff;

// The CPU interface registers, by encoding.
const ICC_PMR_EL1: SysReg = SysReg::new(3, 0, 4, 6, 0);
const ICC_IAR0_EL1: SysReg = SysReg::new(3, 0, 12, 8, 0);
const ICC_EOIR0_EL1: SysReg = SysReg::new(3, 0, 12, 8, 1);
const ICC_HPPIR0_EL1: SysReg = SysReg::new(3, 0, 12, 8, 2);
const ICC_BPR0_EL1: SysReg = SysReg::new(3, 0, 12, 8, 3);
const ICC_AP0R0_EL1: SysReg = SysReg::new(3, 0, 12, 8, 4);
const ICC_AP0R3_EL1: SysReg = SysReg::new(3, 0, 12, 8, 7);
const ICC_AP1R0_EL1: SysReg = SysReg::new(3, 0, 12, 9, 0);
const ICC_AP1R3_EL1: SysReg = SysReg::new(3, 0, 12, 9, 3);
const ICC_DIR_EL1: SysReg = SysReg::new(3, 0, 12, 11, 1);
const ICC_RPR_EL1: SysReg = SysReg::new(3, 0, 12, 11, 3);
const ICC_SGI1R_EL1: SysReg = SysReg::new(3, 0, 12, 11, 5);
const ICC_ASGI1R_EL1: SysReg = SysReg::new(3, 0, 12, 11, 6);
const ICC_SGI0R_EL1: SysReg = SysReg::new(3, 0, 12, 11, 7);
const ICC_IAR1_EL1: SysReg = SysReg::new(3, 0, 12, 12, 0);
const ICC_EOIR1_EL1: SysReg = SysReg::new(3, 0, 12, 12, 1);
const ICC_HPPIR1_EL1: SysReg = SysReg::new(3, 0, 12, 12, 2);
const ICC_BPR1_EL1: SysReg = SysReg::new(3, 0, 12, 12, 3);
const ICC_CTLR_EL1: SysReg = SysReg::new(3, 0, 12, 12, 4);
const ICC_SRE_EL1: SysReg = SysReg::new(3, 0, 12, 12, 5);
const ICC_IGRPEN0_EL1: SysReg = SysReg::new(3, 0, 12, 12, 6);
const ICC_IGRPEN1_EL1: SysReg = SysReg::new(3, 0, 12, 12, 7);

/// ICC_CTLR_EL1.PRIbits: five priority bits, encoded as one less.
const CTLR_PRIBITS: u64 = 4 << 8;
const CTLR_EOIMODE: u64 = 1 << 1;
const CTLR_CBPR: u64 = 1 << 0;
/// ICC_SRE_EL1: system register access on (SRE), FIQ and IRQ bypass disabled
/// (DFB, DIB). Fixed, as the architecture allows.
const SRE_VALUE: u64 = 0b111;
const BPR0_MIN: u8 = 2;
const BPR1_MIN: u8 = 3;

#[derive(Clone, Copy, Default)]
struct Irq {
    enabled: bool,
    /// Set by a rising edge, a write to ISPENDR or an SGI; cleared when the
    /// interrupt is acknowledged or by ICPENDR.
    latch: bool,
    /// The input line.
    level: bool,
    active: bool,
    edge: bool,
    group1: bool,
    priority: u8,
}

impl Irq {
    fn pending(&self) -> bool {
        self.latch || (!self.edge && self.level)
    }

    fn set_level(&mut self, level: bool) {
        if self.edge && level && !self.level {
            self.latch = true;
        }
        self.level = level;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Group {
    G0,
    G1,
}

struct Cpu {
    mpidr: u64,
    private: [Irq; PRIVATE],
    /// GICR_WAKER.ProcessorSleep.
    asleep: bool,
    pmr: u8,
    bpr0: u8,
    bpr1: u8,
    igrpen0: bool,
    igrpen1: bool,
    eoi_mode: bool,
    cbpr: bool,
    /// Active priorities per group, as ICC_AP0R0 and ICC_AP1R0 hold them: bit
    /// n stands for priority n << 3.
    ap: [u32; 2],
}

impl Cpu {
    fn new(mpidr: u64) -> Self {
        let mut private = [Irq::default(); PRIVATE];
        for sgi in &mut private[..16] {
            sgi.edge = true;
        }
        Cpu {
            mpidr,
            private,
            asleep: true,
            pmr: 0,
            bpr0: BPR0_MIN,
            bpr1: BPR1_MIN,
            igrpen0: false,
            igrpen1: false,
            eoi_mode: false,
            cbpr: false,
            ap: [0; 2],
        }
    }

    /// The running priority: the highest active priority, or idle.
    fn running_priority(&self) -> u16 {
        let active = self.ap[0] | self.ap[1];
        if active == 0 {
            IDLE_PRIORITY
        } else {
            (active.trailing_zeros() as u16) << 3
        }
    }

    /// Affinity in the 32-bit packing GICR_TYPER and the guest use:
    /// Aff3.Aff2.Aff1.Aff0.
    fn affinity(&self) -> u64 {
        ((self.mpidr >> 8) & 0xff00_0000) | (self.mpidr & 0xff_ffff)
    }
}

struct State {
    ctlr: u32,
    spis: Vec<Irq>,
    routes: Vec<u64>,
    cpus: Vec<Cpu>,
}

impl State {
    fn irq(&self, cpu: usize, intid: usize) -> Option<&Irq> {
        if intid < PRIVATE {
            self.cpus.get(cpu)?.private.get(intid)
        } else {
            self.spis.get(intid - PRIVATE)
        }
    }

    fn irq_mut(&mut self, cpu: usize, intid: usize) -> Option<&mut Irq> {
        if intid < PRIVATE {
            self.cpus.get_mut(cpu)?.private.get_mut(intid)
        } else {
            self.spis.get_mut(intid - PRIVATE)
        }
    }

    fn routed_to(&self, spi: usize, cpu: usize) -> bool {
        let route = self.routes[spi];
        if route & IROUTER_IRM != 0 {
            return true;
        }
        let affinity = ((route >> 8) & 0xff00_0000) | (route & 0xff_ffff);
        self.cpus[cpu].affinity() == affinity
    }

    fn group_enabled(&self, cpu: usize, group: Group) -> bool {
        let c = &self.cpus[cpu];
        match group {
            Group::G0 => self.ctlr & CTLR_ENABLE_GRP0 != 0 && c.igrpen0,
            Group::G1 => self.ctlr & CTLR_ENABLE_GRP1 != 0 && c.igrpen1,
        }
    }

    /// The highest-priority interrupt of `group` that could be signalled to
    /// `cpu`, ignoring its priority mask and running priority. Ties go to the
    /// lower INTID.
    fn highest_pending(&self, cpu: usize, group: Group) -> Option<(usize, u8)> {
        if !self.group_enabled(cpu, group) {
            return None;
        }
        let wanted = group == Group::G1;
        let candidate =
            |irq: &Irq| irq.enabled && irq.pending() && !irq.active && irq.group1 == wanted;
        let mut best: Option<(usize, u8)> = None;
        let mut consider = |intid: usize, irq: &Irq| {
            if candidate(irq) && best.is_none_or(|(_, p)| irq.priority < p) {
                best = Some((intid, irq.priority));
            }
        };
        for (intid, irq) in self.cpus[cpu].private.iter().enumerate() {
            consider(intid, irq);
        }
        for (spi, irq) in self.spis.iter().enumerate() {
            if candidate(irq) && self.routed_to(spi, cpu) {
                consider(PRIVATE + spi, irq);
            }
        }
        best
    }

    /// Whether `group`'s signal into `cpu` is asserted: something pending of
    /// higher priority than both the mask and the running priority.
    fn signalled(&self, cpu: usize, group: Group) -> bool {
        match self.highest_pending(cpu, group) {
            Some((_, priority)) => {
                let c = &self.cpus[cpu];
                priority < c.pmr && (priority as u16) < c.running_priority()
            }
            None => false,
        }
    }

    fn acknowledge(&mut self, cpu: usize, group: Group) -> u64 {
        if !self.signalled(cpu, group) {
            return SPURIOUS;
        }
        let Some((intid, priority)) = self.highest_pending(cpu, group) else {
            return SPURIOUS;
        };
        let irq = self
            .irq_mut(cpu, intid)
            .expect("a pending interrupt exists");
        irq.active = true;
        irq.latch = false;
        self.cpus[cpu].ap[group as usize] |= 1 << (priority >> 3);
        intid as u64
    }

    fn end_of_interrupt(&mut self, cpu: usize, group: Group, value: u64) {
        let intid = (value & 0xff_ffff) as usize;
        if (1020..1024).contains(&intid) {
            return;
        }
        let ap = &mut self.cpus[cpu].ap[group as usize];
        if *ap != 0 {
            *ap &= *ap - 1; // the priority drop: clear the highest active priority
        }
        if !self.cpus[cpu].eoi_mode {
            self.deactivate(cpu, intid);
        }
    }

    fn deactivate(&mut self, cpu: usize, intid: usize) {
        if let Some(irq) = self.irq_mut(cpu, intid) {
            irq.active = false;
        }
    }

    /// Pends an SGI on the CPUs ICC_SGI*R names. Returns the CPUs it reached.
    fn generate_sgi(&mut self, cpu: usize, value: u64, group: Group) -> Vec<usize> {
        let intid = ((value >> 24) & 0xf) as usize;
        let targets: Vec<usize> = if value & (1 << 40) != 0 {
            (0..self.cpus.len()).filter(|&c| c != cpu).collect()
        } else {
            let aff3 = (value >> 48) & 0xff;
            let aff2 = (value >> 32) & 0xff;
            let aff1 = (value >> 16) & 0xff;
            let range = (value >> 44) & 0xf;
            let list = value & 0xffff;
            (0..self.cpus.len())
                .filter(|&c| {
                    let a = self.cpus[c].affinity();
                    let aff0 = a & 0xff;
                    a >> 8 == (aff3 << 16) | (aff2 << 8) | aff1
                        && aff0 >> 4 == range
                        && list & (1 << (aff0 & 0xf)) != 0
                })
                .collect()
        };
        let wanted = group == Group::G1;
        targets
            .into_iter()
            .filter(|&c| {
                let sgi = &mut self.cpus[c].private[intid];
                if sgi.group1 == wanted {
                    sgi.latch = true;
                }
                sgi.group1 == wanted
            })
            .collect()
    }
}

pub struct GicV3 {
    state: Mutex<State>,
    waker: RwLock<Option<Arc<dyn CpuWaker>>>,
}

impl GicV3 {
    /// `spis` is rounded up to a multiple of 32, from 32 to 960.
    pub fn new(mpidrs: &[u64], spis: usize) -> Arc<Self> {
        let spis = (spis.div_ceil(32) * 32).clamp(32, 960);
        Arc::new(GicV3 {
            state: Mutex::new(State {
                ctlr: 0,
                spis: vec![Irq::default(); spis],
                routes: vec![0; spis],
                cpus: mpidrs.iter().map(|&m| Cpu::new(m)).collect(),
            }),
            waker: RwLock::new(None),
        })
    }

    pub fn distributor(self: &Arc<Self>) -> Arc<dyn MmioDevice> {
        Arc::new(Distributor(self.clone()))
    }

    /// The redistributors of every CPU, one after another.
    pub fn redistributors(self: &Arc<Self>) -> Arc<dyn MmioDevice> {
        Arc::new(Redistributors(self.clone()))
    }

    pub fn spi_count(&self) -> usize {
        self.state.lock().unwrap().spis.len()
    }

    /// Runs `f` on the state, then wakes every CPU whose IRQ or FIQ signal is
    /// asserted afterwards among `cpus` (all CPUs if `None`). Waking a CPU
    /// that has nothing new costs it one look at its interrupts.
    fn update<R>(&self, cpus: Option<&[usize]>, f: impl FnOnce(&mut State) -> R) -> R {
        let (result, wake) = {
            let mut state = self.state.lock().unwrap();
            let result = f(&mut state);
            let all: Vec<usize>;
            let cpus = match cpus {
                Some(c) => c,
                None => {
                    all = (0..state.cpus.len()).collect();
                    &all
                }
            };
            let wake: Vec<usize> = cpus
                .iter()
                .copied()
                .filter(|&c| state.signalled(c, Group::G1) || state.signalled(c, Group::G0))
                .collect();
            (result, wake)
        };
        if !wake.is_empty() {
            if let Some(waker) = &*self.waker.read().unwrap() {
                for cpu in wake {
                    waker.wake(cpu);
                }
            }
        }
        result
    }

    fn dist_read(&self, offset: u64, size: u8) -> u64 {
        let state = self.state.lock().unwrap();
        let total = PRIVATE + state.spis.len();
        match offset {
            0x0000 => (state.ctlr | CTLR_ARE | CTLR_DS) as u64,
            0x0004 => {
                let it_lines = (total / 32 - 1) as u64;
                // ITLinesNumber, IDbits = 16 bits, No1N.
                it_lines | (15 << 19) | (1 << 25)
            }
            0x0008 => IIDR,
            0x0080..0x0d00 => read_banked(&state, 0, offset, size, true),
            0x6100..0x8000 => {
                let intid = ((offset - 0x6000) / 8) as usize;
                match intid.checked_sub(PRIVATE).and_then(|s| state.routes.get(s)) {
                    Some(&route) => {
                        let shift = (offset % 8) * 8;
                        (route >> shift) & crate_core::bus::size_mask(size)
                    }
                    None => 0,
                }
            }
            0xffe8 => PIDR2_GICV3,
            _ => 0,
        }
    }

    fn dist_write(&self, offset: u64, size: u8, value: u64) {
        self.update(None, |state| match offset {
            0x0000 => state.ctlr = value as u32 & (CTLR_ENABLE_GRP0 | CTLR_ENABLE_GRP1),
            0x0040 | 0x0048 => {
                // SETSPI_NSR and CLRSPI_NSR.
                let intid = (value & 0x3ff) as usize;
                if let Some(irq) = intid
                    .checked_sub(PRIVATE)
                    .and_then(|s| state.spis.get_mut(s))
                {
                    irq.latch = offset == 0x0040;
                }
            }
            0x0080..0x0d00 => write_banked(state, 0, offset, size, value, true),
            0x6100..0x8000 => {
                let intid = ((offset - 0x6000) / 8) as usize;
                if let Some(route) = intid
                    .checked_sub(PRIVATE)
                    .and_then(|s| state.routes.get_mut(s))
                {
                    let shift = (offset % 8) * 8;
                    let mask = crate_core::bus::size_mask(size) << shift;
                    *route = ((*route & !mask) | ((value << shift) & mask)) & IROUTER_MASK;
                }
            }
            _ => {}
        });
    }

    fn redist_read(&self, offset: u64, size: u8) -> u64 {
        let cpu = (offset / REDISTRIBUTOR_SIZE) as usize;
        let offset = offset % REDISTRIBUTOR_SIZE;
        let state = self.state.lock().unwrap();
        let Some(c) = state.cpus.get(cpu) else {
            return 0;
        };
        let value = match offset {
            0x0004 => IIDR,
            0x0008..0x0010 => {
                let last = (cpu + 1 == state.cpus.len()) as u64;
                let typer = (c.affinity() << 32) | ((cpu as u64) << 8) | (last << 4);
                return (typer >> ((offset - 8) * 8)) & crate_core::bus::size_mask(size);
            }
            0x0014 => ((c.asleep as u64) << 1) | ((c.asleep as u64) << 2),
            0xffe8 => PIDR2_GICV3,
            0x1_0080..0x1_0d00 => return read_banked(&state, cpu, offset - 0x1_0000, size, false),
            _ => 0,
        };
        value & crate_core::bus::size_mask(size)
    }

    fn redist_write(&self, offset: u64, size: u8, value: u64) {
        let cpu = (offset / REDISTRIBUTOR_SIZE) as usize;
        let offset = offset % REDISTRIBUTOR_SIZE;
        self.update(Some(&[cpu]), |state| {
            if cpu >= state.cpus.len() {
                return;
            }
            match offset {
                0x0014 => state.cpus[cpu].asleep = value & 2 != 0,
                0x1_0080..0x1_0d00 => {
                    write_banked(state, cpu, offset - 0x1_0000, size, value, false)
                }
                _ => {}
            }
        });
    }
}

/// Reads one of the per-interrupt registers the distributor (for SPIs) and the
/// redistributors' SGI frames (for SGIs and PPIs) share the layout of, at
/// `offset` from the start of that layout.
fn read_banked(state: &State, cpu: usize, offset: u64, size: u8, distributor: bool) -> u64 {
    let in_range = |intid: usize| {
        if distributor {
            intid >= PRIVATE
        } else {
            intid < PRIVATE
        }
    };
    match offset {
        0x0400..0x0800 => {
            // IPRIORITYR: a byte per interrupt.
            let first = (offset - 0x400) as usize;
            (0..size as usize).fold(0, |acc, i| {
                let intid = first + i;
                let p = if in_range(intid) {
                    state.irq(cpu, intid).map_or(0, |irq| irq.priority)
                } else {
                    0
                };
                acc | (p as u64) << (8 * i)
            })
        }
        0x0c00..0x0d00 => {
            // ICFGR: two bits per interrupt, the upper one meaning edge.
            let first = ((offset - 0xc00) / 4 * 16) as usize;
            (0..16).fold(0, |acc, i| {
                let edge =
                    in_range(first + i) && state.irq(cpu, first + i).is_some_and(|irq| irq.edge);
                acc | ((edge as u64) << (2 * i + 1))
            })
        }
        _ => {
            let bits: fn(&Irq) -> bool = match offset & !0x7f {
                0x0080 => |irq| irq.group1,
                0x0100 | 0x0180 => |irq| irq.enabled,
                0x0200 | 0x0280 => |irq| irq.pending(),
                0x0300 | 0x0380 => |irq| irq.active,
                _ => return 0,
            };
            let first = ((offset & 0x7f) / 4 * 32) as usize;
            (0..32).fold(0, |acc, i| {
                let set = in_range(first + i) && state.irq(cpu, first + i).is_some_and(bits);
                acc | ((set as u64) << i)
            })
        }
    }
}

fn write_banked(
    state: &mut State,
    cpu: usize,
    offset: u64,
    size: u8,
    value: u64,
    distributor: bool,
) {
    let in_range = |intid: usize| {
        if distributor {
            intid >= PRIVATE
        } else {
            intid < PRIVATE
        }
    };
    match offset {
        0x0400..0x0800 => {
            let first = (offset - 0x400) as usize;
            for i in 0..size as usize {
                if in_range(first + i) {
                    if let Some(irq) = state.irq_mut(cpu, first + i) {
                        irq.priority = (value >> (8 * i)) as u8 & PRIORITY_MASK;
                    }
                }
            }
        }
        0x0c00..0x0d00 => {
            let first = ((offset - 0xc00) / 4 * 16) as usize;
            for i in 0..16 {
                let intid = first + i;
                // SGIs are always edge-triggered.
                if in_range(intid) && intid >= 16 {
                    if let Some(irq) = state.irq_mut(cpu, intid) {
                        irq.edge = value & (1 << (2 * i + 1)) != 0;
                    }
                }
            }
        }
        _ => {
            if size < 4 {
                return;
            }
            let register = offset & !0x7f;
            let first = ((offset & 0x7f) / 4 * 32) as usize;
            for i in 0..32 {
                let intid = first + i;
                let bit = value & (1 << i) != 0;
                if !in_range(intid) || (register != 0x0080 && !bit) {
                    continue;
                }
                let Some(irq) = state.irq_mut(cpu, intid) else {
                    continue;
                };
                match register {
                    0x0080 => irq.group1 = bit,
                    0x0100 => irq.enabled = true,
                    0x0180 => irq.enabled = false,
                    0x0200 => irq.latch = true,
                    0x0280 => irq.latch = false,
                    0x0300 => irq.active = true,
                    0x0380 => irq.active = false,
                    _ => {}
                }
            }
        }
    }
}

fn is_ap_register(reg: SysReg) -> bool {
    (ICC_AP0R0_EL1..=ICC_AP0R3_EL1).contains(&reg) || (ICC_AP1R0_EL1..=ICC_AP1R3_EL1).contains(&reg)
}

struct Distributor(Arc<GicV3>);

impl MmioDevice for Distributor {
    fn read(&self, offset: u64, size: u8) -> u64 {
        if size == 8 && !(0x6000..0x8000).contains(&offset) {
            return self.0.dist_read(offset, 4) | (self.0.dist_read(offset + 4, 4) << 32);
        }
        self.0.dist_read(offset, size)
    }
    fn write(&self, offset: u64, size: u8, value: u64) {
        self.0.dist_write(offset, size, value)
    }
}

struct Redistributors(Arc<GicV3>);

impl MmioDevice for Redistributors {
    fn read(&self, offset: u64, size: u8) -> u64 {
        self.0.redist_read(offset, size)
    }
    fn write(&self, offset: u64, size: u8, value: u64) {
        self.0.redist_write(offset, size, value)
    }
}

impl InterruptController for GicV3 {
    fn set_spi(&self, spi: u32, level: bool) {
        let targets: Vec<usize> = {
            let mut state = self.state.lock().unwrap();
            let spi = spi as usize;
            let Some(irq) = state.spis.get_mut(spi) else {
                return;
            };
            if irq.level == level {
                return;
            }
            irq.set_level(level);
            if !level {
                return;
            }
            (0..state.cpus.len())
                .filter(|&c| state.routed_to(spi, c))
                .collect()
        };
        self.update(Some(&targets), |_| ());
    }

    fn set_ppi(&self, cpu: usize, ppi: u32, level: bool) {
        self.update(Some(&[cpu]), |state| {
            if let Some(irq) = state
                .cpus
                .get_mut(cpu)
                .and_then(|c| c.private.get_mut(16 + ppi as usize))
            {
                irq.set_level(level);
            }
        });
    }
}

impl CpuInterrupts for GicV3 {
    fn irq_pending(&self, cpu: usize) -> bool {
        self.state.lock().unwrap().signalled(cpu, Group::G1)
    }

    fn fiq_pending(&self, cpu: usize) -> bool {
        self.state.lock().unwrap().signalled(cpu, Group::G0)
    }

    fn read_sysreg(&self, cpu: usize, reg: SysReg) -> Option<u64> {
        let mut state = self.state.lock().unwrap();
        if cpu >= state.cpus.len() {
            return None;
        }
        match reg {
            ICC_IAR1_EL1 => return Some(state.acknowledge(cpu, Group::G1)),
            ICC_IAR0_EL1 => return Some(state.acknowledge(cpu, Group::G0)),
            _ => {}
        }
        let c = &state.cpus[cpu];
        Some(match reg {
            ICC_HPPIR1_EL1 => state
                .highest_pending(cpu, Group::G1)
                .map_or(SPURIOUS, |(i, _)| i as u64),
            ICC_HPPIR0_EL1 => state
                .highest_pending(cpu, Group::G0)
                .map_or(SPURIOUS, |(i, _)| i as u64),
            ICC_PMR_EL1 => c.pmr as u64,
            ICC_RPR_EL1 => c.running_priority().min(0xff) as u64,
            ICC_BPR0_EL1 => c.bpr0 as u64,
            ICC_BPR1_EL1 if c.cbpr => (c.bpr0 + 1).min(7) as u64,
            ICC_BPR1_EL1 => c.bpr1 as u64,
            ICC_CTLR_EL1 => {
                CTLR_PRIBITS | if c.eoi_mode { CTLR_EOIMODE } else { 0 } | c.cbpr as u64
            }
            ICC_SRE_EL1 => SRE_VALUE,
            ICC_IGRPEN0_EL1 => c.igrpen0 as u64,
            ICC_IGRPEN1_EL1 => c.igrpen1 as u64,
            ICC_AP0R0_EL1 => c.ap[0] as u64,
            ICC_AP1R0_EL1 => c.ap[1] as u64,
            r if is_ap_register(r) => 0,
            ICC_EOIR0_EL1 | ICC_EOIR1_EL1 | ICC_DIR_EL1 | ICC_SGI0R_EL1 | ICC_SGI1R_EL1
            | ICC_ASGI1R_EL1 => 0,
            _ => return None,
        })
    }

    fn write_sysreg(&self, cpu: usize, reg: SysReg, value: u64) -> bool {
        let mut state = self.state.lock().unwrap();
        if cpu >= state.cpus.len() {
            return false;
        }
        let sgi_targets = match reg {
            ICC_EOIR1_EL1 => {
                state.end_of_interrupt(cpu, Group::G1, value);
                return true;
            }
            ICC_EOIR0_EL1 => {
                state.end_of_interrupt(cpu, Group::G0, value);
                return true;
            }
            ICC_DIR_EL1 => {
                state.deactivate(cpu, (value & 0xff_ffff) as usize);
                return true;
            }
            ICC_SGI1R_EL1 | ICC_ASGI1R_EL1 => state.generate_sgi(cpu, value, Group::G1),
            ICC_SGI0R_EL1 => state.generate_sgi(cpu, value, Group::G0),
            _ => {
                let c = &mut state.cpus[cpu];
                match reg {
                    ICC_PMR_EL1 => c.pmr = value as u8 & PRIORITY_MASK,
                    ICC_BPR0_EL1 => c.bpr0 = (value as u8 & 7).max(BPR0_MIN),
                    ICC_BPR1_EL1 if !c.cbpr => c.bpr1 = (value as u8 & 7).max(BPR1_MIN),
                    ICC_BPR1_EL1 => {}
                    ICC_CTLR_EL1 => {
                        c.eoi_mode = value & CTLR_EOIMODE != 0;
                        c.cbpr = value & CTLR_CBPR != 0;
                    }
                    ICC_IGRPEN0_EL1 => c.igrpen0 = value & 1 != 0,
                    ICC_IGRPEN1_EL1 => c.igrpen1 = value & 1 != 0,
                    ICC_AP0R0_EL1 => c.ap[0] = value as u32,
                    ICC_AP1R0_EL1 => c.ap[1] = value as u32,
                    r if is_ap_register(r) || r == ICC_SRE_EL1 => {}
                    _ => return false,
                }
                return true;
            }
        };
        drop(state);
        if !sgi_targets.is_empty() {
            self.update(Some(&sgi_targets), |_| ());
        }
        true
    }

    fn set_waker(&self, waker: Arc<dyn CpuWaker>) {
        *self.waker.write().unwrap() = Some(waker);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    #[derive(Default)]
    struct Woken(StdMutex<Vec<usize>>);
    impl CpuWaker for Woken {
        fn wake(&self, cpu: usize) {
            self.0.lock().unwrap().push(cpu);
        }
    }

    fn gic(cpus: usize) -> (Arc<GicV3>, Arc<Woken>) {
        let mpidrs: Vec<u64> = (0..cpus as u64).map(|c| 0x8000_0000 | c).collect();
        let gic = GicV3::new(&mpidrs, 64);
        let woken = Arc::new(Woken::default());
        gic.set_waker(woken.clone());
        (gic, woken)
    }

    /// What Linux does at boot: enable both groups and affinity routing, make
    /// everything group 1, wake the redistributors, open the CPU interfaces.
    fn linux_init(gic: &Arc<GicV3>) {
        let dist = gic.distributor();
        let redist = gic.redistributors();
        dist.write(
            0x0000,
            4,
            (CTLR_ARE | CTLR_ENABLE_GRP0 | CTLR_ENABLE_GRP1) as u64,
        );
        for n in 1..3 {
            dist.write(0x0080 + 4 * n, 4, 0xffff_ffff);
        }
        let cpus = gic.state.lock().unwrap().cpus.len();
        for cpu in 0..cpus {
            let base = cpu as u64 * REDISTRIBUTOR_SIZE;
            redist.write(base + 0x14, 4, 0);
            redist.write(base + 0x1_0080, 4, 0xffff_ffff);
            gic.write_sysreg(cpu, ICC_PMR_EL1, 0xf0);
            gic.write_sysreg(cpu, ICC_IGRPEN1_EL1, 1);
        }
    }

    #[test]
    fn identifies_as_a_gicv3_without_lpis() {
        let (gic, _) = gic(2);
        let dist = gic.distributor();
        assert_eq!(dist.read(0xffe8, 4) & 0xf0, 0x30);
        let typer = dist.read(0x0004, 4);
        assert_eq!(typer & 0x1f, 2, "96 INTIDs in all");
        assert_eq!(typer & (1 << 17), 0, "no LPIs");
        assert_eq!(
            dist.read(0x0000, 4) & (CTLR_DS | CTLR_ARE) as u64,
            (CTLR_DS | CTLR_ARE) as u64
        );
        let redist = gic.redistributors();
        assert_eq!(redist.read(0xffe8, 4) & 0xf0, 0x30);
        assert_eq!(redist.read(REDISTRIBUTOR_SIZE + 0xffe8, 4) & 0xf0, 0x30);
        let typer0 = redist.read(0x8, 8);
        let typer1 = redist.read(REDISTRIBUTOR_SIZE + 0x8, 8);
        assert_eq!(typer0 >> 32, 0, "affinity of CPU 0");
        assert_eq!(typer1 >> 32, 1, "affinity of CPU 1");
        assert_eq!(typer0 & (1 << 4), 0, "CPU 0 is not last");
        assert_ne!(typer1 & (1 << 4), 0, "CPU 1 is last");
        assert_eq!(redist.read(0xc, 4), 0, "upper half of TYPER read alone");
        assert_eq!(redist.read(REDISTRIBUTOR_SIZE + 0xc, 4), 1);
    }

    #[test]
    fn waker_handshake() {
        let (gic, _) = gic(1);
        let redist = gic.redistributors();
        assert_eq!(redist.read(0x14, 4), 0b110, "asleep at reset");
        redist.write(0x14, 4, 0);
        assert_eq!(
            redist.read(0x14, 4),
            0,
            "ChildrenAsleep follows ProcessorSleep"
        );
    }

    #[test]
    fn registers_read_back() {
        let (gic, _) = gic(1);
        let dist = gic.distributor();
        dist.write(0x0100 + 4, 4, 0x8000_0001);
        assert_eq!(dist.read(0x0100 + 4, 4), 0x8000_0001);
        dist.write(0x0180 + 4, 4, 1);
        assert_eq!(dist.read(0x0100 + 4, 4), 0x8000_0000);
        assert_eq!(
            dist.read(0x0100, 4),
            0,
            "INTIDs 0-31 belong to the redistributors"
        );
        dist.write(0x0420, 1, 0xa7);
        assert_eq!(dist.read(0x0420, 1), 0xa0, "five priority bits");
        dist.write(0x0424, 4, 0x1122_3344);
        assert_eq!(dist.read(0x0424, 4), 0x1020_3040);
        dist.write(0x0c08, 4, 0x2);
        assert_eq!(dist.read(0x0c08, 4), 0x2, "INTID 32 edge-triggered");
        dist.write(0x6100, 8, 0xff_0000_0000 | 0x1_0203);
        assert_eq!(dist.read(0x6100, 8), 0xff_0000_0000 | 0x1_0203);
        assert_eq!(dist.read(0x6104, 4), 0xff);
        let redist = gic.redistributors();
        assert_eq!(
            redist.read(0x1_0c00, 4),
            0xaaaa_aaaa,
            "SGIs are edge-triggered"
        );
        redist.write(0x1_0c00, 4, 0);
        assert_eq!(redist.read(0x1_0c00, 4), 0xaaaa_aaaa, "and stay so");
    }

    #[test]
    fn delivers_an_spi_through_acknowledge_and_eoi() {
        let (gic, woken) = gic(2);
        linux_init(&gic);
        let dist = gic.distributor();
        // SPI 5 (INTID 37) to CPU 1, priority 0xa0, enabled, level-sensitive.
        dist.write(0x6000 + 8 * 37, 8, 1);
        dist.write(0x0400 + 37, 1, 0xa0);
        dist.write(0x0104, 4, 1 << 5);
        woken.0.lock().unwrap().clear();

        gic.set_spi(5, true);
        assert_eq!(*woken.0.lock().unwrap(), vec![1]);
        assert!(!gic.irq_pending(0));
        assert!(gic.irq_pending(1));
        assert_eq!(gic.read_sysreg(1, ICC_HPPIR1_EL1), Some(37));
        assert_eq!(gic.read_sysreg(1, ICC_IAR1_EL1), Some(37));
        assert!(!gic.irq_pending(1), "active, so not signalled again");
        assert_eq!(gic.read_sysreg(1, ICC_RPR_EL1), Some(0xa0));
        assert_eq!(dist.read(0x0304, 4), 1 << 5, "active");
        assert_eq!(
            dist.read(0x0204, 4),
            1 << 5,
            "and still pending: the line is high"
        );

        gic.set_spi(5, false);
        gic.write_sysreg(1, ICC_EOIR1_EL1, 37);
        assert_eq!(gic.read_sysreg(1, ICC_RPR_EL1), Some(0xff));
        assert_eq!(dist.read(0x0304, 4), 0);
        assert!(!gic.irq_pending(1));
        assert_eq!(gic.read_sysreg(1, ICC_IAR1_EL1), Some(SPURIOUS));
    }

    #[test]
    fn a_level_interrupt_still_high_after_eoi_is_taken_again() {
        let (gic, _) = gic(1);
        linux_init(&gic);
        let dist = gic.distributor();
        dist.write(0x0104, 4, 1);
        gic.set_spi(0, true);
        assert_eq!(gic.read_sysreg(0, ICC_IAR1_EL1), Some(32));
        gic.write_sysreg(0, ICC_EOIR1_EL1, 32);
        assert!(gic.irq_pending(0));
        assert_eq!(gic.read_sysreg(0, ICC_IAR1_EL1), Some(32));
    }

    #[test]
    fn an_edge_interrupt_is_latched_once() {
        let (gic, _) = gic(1);
        linux_init(&gic);
        let dist = gic.distributor();
        dist.write(0x0c08, 4, 0x2);
        dist.write(0x0104, 4, 1);
        gic.set_spi(0, true);
        gic.set_spi(0, false);
        assert_eq!(gic.read_sysreg(0, ICC_IAR1_EL1), Some(32));
        gic.write_sysreg(0, ICC_EOIR1_EL1, 32);
        assert!(!gic.irq_pending(0));
    }

    #[test]
    fn priority_mask_and_preemption() {
        let (gic, _) = gic(1);
        linux_init(&gic);
        let dist = gic.distributor();
        dist.write(0x0104, 4, 0b11);
        dist.write(0x0420, 1, 0x80); // INTID 32
        dist.write(0x0421, 1, 0x40); // INTID 33

        gic.write_sysreg(0, ICC_PMR_EL1, 0x80);
        gic.set_spi(0, true);
        assert!(!gic.irq_pending(0), "0x80 is not above a mask of 0x80");
        gic.write_sysreg(0, ICC_PMR_EL1, 0xf0);
        assert!(gic.irq_pending(0));
        assert_eq!(gic.read_sysreg(0, ICC_IAR1_EL1), Some(32));

        gic.set_spi(1, true);
        assert!(gic.irq_pending(0), "0x40 preempts a running 0x80");
        assert_eq!(gic.read_sysreg(0, ICC_IAR1_EL1), Some(33));
        assert_eq!(
            gic.read_sysreg(0, ICC_AP1R0_EL1),
            Some((1 << 8) | (1 << 16))
        );
        gic.write_sysreg(0, ICC_EOIR1_EL1, 33);
        assert_eq!(gic.read_sysreg(0, ICC_RPR_EL1), Some(0x80));
        gic.write_sysreg(0, ICC_EOIR1_EL1, 32);
        assert_eq!(gic.read_sysreg(0, ICC_RPR_EL1), Some(0xff));
    }

    #[test]
    fn split_eoi_mode_needs_dir() {
        let (gic, _) = gic(1);
        linux_init(&gic);
        gic.write_sysreg(0, ICC_CTLR_EL1, CTLR_EOIMODE);
        assert_eq!(
            gic.read_sysreg(0, ICC_CTLR_EL1).unwrap() & 0x702,
            CTLR_PRIBITS | CTLR_EOIMODE
        );
        gic.distributor().write(0x0104, 4, 1);
        gic.set_spi(0, true);
        assert_eq!(gic.read_sysreg(0, ICC_IAR1_EL1), Some(32));
        gic.set_spi(0, false);
        gic.write_sysreg(0, ICC_EOIR1_EL1, 32);
        assert_eq!(
            gic.read_sysreg(0, ICC_RPR_EL1),
            Some(0xff),
            "priority dropped"
        );
        assert_eq!(gic.distributor().read(0x0304, 4), 1, "but still active");
        gic.write_sysreg(0, ICC_DIR_EL1, 32);
        assert_eq!(gic.distributor().read(0x0304, 4), 0);
    }

    #[test]
    fn sgis_reach_the_cpus_named() {
        let (gic, woken) = gic(4);
        linux_init(&gic);
        let redist = gic.redistributors();
        for cpu in 0..4 {
            redist.write(cpu * REDISTRIBUTOR_SIZE + 0x1_0100, 4, 0xffff);
        }
        woken.0.lock().unwrap().clear();
        // SGI 3 to CPUs 1 and 2 by target list.
        gic.write_sysreg(0, ICC_SGI1R_EL1, (3 << 24) | 0b110);
        assert_eq!(*woken.0.lock().unwrap(), vec![1, 2]);
        for cpu in [1, 2] {
            assert_eq!(gic.read_sysreg(cpu, ICC_IAR1_EL1), Some(3));
            gic.write_sysreg(cpu, ICC_EOIR1_EL1, 3);
        }
        assert_eq!(gic.read_sysreg(3, ICC_IAR1_EL1), Some(SPURIOUS));
        // SGI 1 to everyone but CPU 3.
        gic.write_sysreg(3, ICC_SGI1R_EL1, (1 << 24) | (1 << 40));
        for cpu in 0..3 {
            assert!(gic.irq_pending(cpu));
        }
        assert!(!gic.irq_pending(3));
        // A target list in another cluster reaches nobody.
        gic.write_sysreg(0, ICC_SGI1R_EL1, (2 << 24) | (1 << 16) | 0xf);
        assert_eq!(gic.read_sysreg(3, ICC_IAR1_EL1), Some(SPURIOUS));
    }

    #[test]
    fn ppis_are_per_cpu_and_need_the_group_enabled() {
        let (gic, _) = gic(2);
        let redist = gic.redistributors();
        redist.write(REDISTRIBUTOR_SIZE + 0x1_0080, 4, 1 << 27);
        redist.write(REDISTRIBUTOR_SIZE + 0x1_0100, 4, 1 << 27);
        gic.write_sysreg(1, ICC_PMR_EL1, 0xf0);
        gic.set_ppi(1, 11, true);
        assert!(!gic.irq_pending(1), "group 1 not enabled yet");
        gic.write_sysreg(1, ICC_IGRPEN1_EL1, 1);
        assert!(!gic.irq_pending(1), "nor in the distributor");
        gic.distributor().write(0, 4, CTLR_ENABLE_GRP1 as u64);
        assert!(gic.irq_pending(1));
        assert!(!gic.irq_pending(0));
        assert_eq!(gic.read_sysreg(1, ICC_IAR1_EL1), Some(27));
    }

    #[test]
    fn group0_signals_fiq() {
        let (gic, _) = gic(1);
        let dist = gic.distributor();
        dist.write(0, 4, CTLR_ENABLE_GRP0 as u64);
        dist.write(0x0104, 4, 1);
        gic.write_sysreg(0, ICC_PMR_EL1, 0xf0);
        gic.write_sysreg(0, ICC_IGRPEN0_EL1, 1);
        gic.set_spi(0, true);
        assert!(gic.fiq_pending(0));
        assert!(!gic.irq_pending(0));
        assert_eq!(gic.read_sysreg(0, ICC_IAR1_EL1), Some(SPURIOUS));
        assert_eq!(gic.read_sysreg(0, ICC_IAR0_EL1), Some(32));
    }

    #[test]
    fn cpu_interface_identification() {
        let (gic, _) = gic(1);
        assert_eq!(gic.read_sysreg(0, ICC_SRE_EL1), Some(0b111));
        assert_eq!(gic.read_sysreg(0, ICC_CTLR_EL1).unwrap() & 0x700, 4 << 8);
        gic.write_sysreg(0, ICC_BPR1_EL1, 0);
        assert_eq!(gic.read_sysreg(0, ICC_BPR1_EL1), Some(BPR1_MIN as u64));
        assert_eq!(gic.read_sysreg(0, SysReg::new(3, 0, 1, 0, 0)), None);
        assert!(!gic.write_sysreg(0, SysReg::new(3, 0, 1, 0, 0), 0));
    }
}
