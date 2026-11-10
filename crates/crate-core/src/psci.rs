// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The Power State Coordination Interface (Arm DEN 0022): how the guest turns
//! CPUs on and off and the machine off, by HVC or SMC with a function ID in x0
//! and arguments in x1 to x3. Every CPU backend calls into this one model.
//!
//! Version 1.0, with the calls Linux needs. CPU_SUSPEND is not offered, so the
//! guest idles with WFI, which the backends already wait on.

use std::sync::{Arc, Condvar, Mutex};

use crate::run::{RunControl, StopReason};

pub const PSCI_VERSION: u32 = 0x8400_0000;
pub const CPU_SUSPEND_32: u32 = 0x8400_0001;
pub const CPU_SUSPEND_64: u32 = 0xc400_0001;
pub const CPU_OFF: u32 = 0x8400_0002;
pub const CPU_ON_32: u32 = 0x8400_0003;
pub const CPU_ON_64: u32 = 0xc400_0003;
pub const AFFINITY_INFO_32: u32 = 0x8400_0004;
pub const AFFINITY_INFO_64: u32 = 0xc400_0004;
pub const MIGRATE_INFO_TYPE: u32 = 0x8400_0006;
pub const SYSTEM_OFF: u32 = 0x8400_0008;
pub const SYSTEM_RESET: u32 = 0x8400_0009;
pub const PSCI_FEATURES: u32 = 0x8400_000a;

pub const SUCCESS: i64 = 0;
pub const NOT_SUPPORTED: i64 = -1;
pub const INVALID_PARAMETERS: i64 = -2;
pub const ALREADY_ON: i64 = -4;
pub const ON_PENDING: i64 = -5;

/// PSCI 1.0 in x0's encoding: major in bits 31:16, minor in 15:0.
const VERSION_1_0: i64 = 0x1_0000;
/// MIGRATE_INFO_TYPE: no trusted OS that would need migrating.
const NO_TRUSTED_OS: i64 = 2;

/// Where a CPU starts when CPU_ON turns it on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuEntry {
    pub pc: u64,
    pub context: u64,
}

/// What the calling CPU does once the call returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    Continue,
    /// CPU_OFF: the CPU stops until another CPU_ON.
    CpuOff,
    /// SYSTEM_OFF or SYSTEM_RESET: the machine has been told to stop.
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Power {
    Off,
    Pending(CpuEntry),
    On,
}

pub struct Psci {
    mpidrs: Vec<u64>,
    power: Mutex<Vec<Power>>,
    cond: Condvar,
    run: Arc<RunControl>,
}

/// The affinity fields of an MPIDR (Aff3 in bits 39:32, Aff2 to Aff0 in 23:0).
pub const MPIDR_AFFINITY_MASK: u64 = 0xff_00ff_ffff;

impl Psci {
    /// CPU 0 starts on, the others off.
    pub fn new(mpidrs: Vec<u64>, run: Arc<RunControl>) -> Arc<Self> {
        let mut power = vec![Power::Off; mpidrs.len()];
        if let Some(first) = power.first_mut() {
            *first = Power::On;
        }
        let psci = Arc::new(Psci {
            mpidrs,
            power: Mutex::new(power),
            cond: Condvar::new(),
            run,
        });
        let weak = Arc::downgrade(&psci);
        psci.run.on_stop(move || {
            if let Some(psci) = weak.upgrade() {
                let _guard = psci.power.lock().unwrap();
                psci.cond.notify_all();
            }
        });
        psci
    }

    fn cpu_of(&self, mpidr: u64) -> Option<usize> {
        self.mpidrs
            .iter()
            .position(|m| m & MPIDR_AFFINITY_MASK == mpidr & MPIDR_AFFINITY_MASK)
    }

    /// Handles a call from `cpu` with x0 to x3 in `args`; returns the value for
    /// x0 and what the CPU does next.
    pub fn call(&self, cpu: usize, args: [u64; 4]) -> (u64, Effect) {
        let function = args[0] as u32;
        let (ret, effect) = match function {
            PSCI_VERSION => (VERSION_1_0, Effect::Continue),
            CPU_OFF => {
                self.power.lock().unwrap()[cpu] = Power::Off;
                (SUCCESS, Effect::CpuOff)
            }
            CPU_ON_32 | CPU_ON_64 => (self.cpu_on(args[1], args[2], args[3]), Effect::Continue),
            AFFINITY_INFO_32 | AFFINITY_INFO_64 => {
                (self.affinity_info(args[1], args[2]), Effect::Continue)
            }
            MIGRATE_INFO_TYPE => (NO_TRUSTED_OS, Effect::Continue),
            SYSTEM_OFF => {
                self.run.stop(StopReason::PowerOff);
                (SUCCESS, Effect::Stop)
            }
            SYSTEM_RESET => {
                self.run.stop(StopReason::Reset);
                (SUCCESS, Effect::Stop)
            }
            PSCI_FEATURES => (Self::features(args[1] as u32), Effect::Continue),
            _ => (NOT_SUPPORTED, Effect::Continue),
        };
        (ret as u64, effect)
    }

    fn features(function: u32) -> i64 {
        match function {
            PSCI_VERSION | CPU_OFF | CPU_ON_32 | CPU_ON_64 | AFFINITY_INFO_32
            | AFFINITY_INFO_64 | MIGRATE_INFO_TYPE | SYSTEM_OFF | SYSTEM_RESET | PSCI_FEATURES => {
                SUCCESS
            }
            _ => NOT_SUPPORTED,
        }
    }

    fn cpu_on(&self, target: u64, pc: u64, context: u64) -> i64 {
        let Some(cpu) = self.cpu_of(target) else {
            return INVALID_PARAMETERS;
        };
        let mut power = self.power.lock().unwrap();
        match power[cpu] {
            Power::On => ALREADY_ON,
            Power::Pending(_) => ON_PENDING,
            Power::Off => {
                power[cpu] = Power::Pending(CpuEntry { pc, context });
                self.cond.notify_all();
                SUCCESS
            }
        }
    }

    fn affinity_info(&self, target: u64, lowest_level: u64) -> i64 {
        if lowest_level != 0 {
            return INVALID_PARAMETERS;
        }
        let Some(cpu) = self.cpu_of(target) else {
            return INVALID_PARAMETERS;
        };
        match self.power.lock().unwrap()[cpu] {
            Power::On => 0,
            Power::Off => 1,
            Power::Pending(_) => 2,
        }
    }

    /// Blocks the thread of an off CPU until CPU_ON names it; `None` once the
    /// machine stops.
    pub fn wait_for_on(&self, cpu: usize) -> Option<CpuEntry> {
        let mut power = self.power.lock().unwrap();
        loop {
            if self.run.is_stopped() {
                return None;
            }
            if let Power::Pending(entry) = power[cpu] {
                power[cpu] = Power::On;
                return Some(entry);
            }
            power = self.cond.wait(power).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn psci() -> Arc<Psci> {
        Psci::new(vec![0x8000_0000, 0x8000_0001], Arc::new(RunControl::new()))
    }

    #[test]
    fn reports_version_and_features() {
        let p = psci();
        assert_eq!(p.call(0, [PSCI_VERSION as u64, 0, 0, 0]).0, 0x1_0000);
        assert_eq!(
            p.call(0, [PSCI_FEATURES as u64, CPU_ON_64 as u64, 0, 0]).0,
            0
        );
        assert_eq!(
            p.call(0, [PSCI_FEATURES as u64, CPU_SUSPEND_64 as u64, 0, 0])
                .0 as i64,
            -1
        );
        assert_eq!(p.call(0, [0xc400_0042, 0, 0, 0]).0 as i64, NOT_SUPPORTED);
    }

    #[test]
    fn turns_a_cpu_on_once() {
        let p = psci();
        assert_eq!(p.call(0, [AFFINITY_INFO_64 as u64, 1, 0, 0]).0, 1);
        assert_eq!(p.call(0, [CPU_ON_64 as u64, 1, 0x4008_0000, 7]).0, 0);
        assert_eq!(p.call(0, [AFFINITY_INFO_64 as u64, 1, 0, 0]).0, 2);
        assert_eq!(p.call(0, [CPU_ON_64 as u64, 1, 0, 0]).0 as i64, ON_PENDING);
        assert_eq!(
            p.wait_for_on(1),
            Some(CpuEntry {
                pc: 0x4008_0000,
                context: 7
            })
        );
        assert_eq!(p.call(0, [CPU_ON_64 as u64, 1, 0, 0]).0 as i64, ALREADY_ON);
        assert_eq!(
            p.call(0, [CPU_ON_64 as u64, 9, 0, 0]).0 as i64,
            INVALID_PARAMETERS
        );
        assert_eq!(p.call(1, [CPU_OFF as u64, 0, 0, 0]), (0, Effect::CpuOff));
        assert_eq!(p.call(0, [AFFINITY_INFO_64 as u64, 1, 0, 0]).0, 1);
    }

    #[test]
    fn system_off_stops_the_machine_and_releases_waiters() {
        let run = Arc::new(RunControl::new());
        let p = Psci::new(vec![0, 1], run.clone());
        let waiter = {
            let p = p.clone();
            std::thread::spawn(move || p.wait_for_on(1))
        };
        assert_eq!(p.call(0, [SYSTEM_OFF as u64, 0, 0, 0]).1, Effect::Stop);
        assert_eq!(run.reason(), Some(StopReason::PowerOff));
        assert_eq!(waiter.join().unwrap(), None);
    }
}
