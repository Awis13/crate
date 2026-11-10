// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Interrupt lines from devices to the interrupt controller.
//!
//! Lines are levels: a device holds its line high while it wants attention and
//! drops it when the guest has dealt with it. An edge-triggered input is the
//! controller's business; it sees the rising edge.

use std::sync::Arc;

/// What devices see of an interrupt controller.
pub trait InterruptController: Send + Sync {
    /// Drives shared peripheral interrupt `spi` (counted from 0; the GIC's
    /// INTID 32 + spi).
    fn set_spi(&self, spi: u32, level: bool);
    /// Drives private peripheral interrupt `ppi` of one CPU (0 to 15; INTID
    /// 16 + ppi), such as that CPU's timer.
    fn set_ppi(&self, cpu: usize, ppi: u32, level: bool);
}

/// One device's line into the controller.
#[derive(Clone)]
pub struct IrqLine {
    chip: Arc<dyn InterruptController>,
    spi: u32,
}

impl IrqLine {
    pub fn new(chip: Arc<dyn InterruptController>, spi: u32) -> Self {
        IrqLine { chip, spi }
    }

    pub fn spi(&self) -> u32 {
        self.spi
    }

    pub fn set(&self, level: bool) {
        self.chip.set_spi(self.spi, level);
    }
}

/// A controller that remembers line levels, for device tests.
#[derive(Default)]
pub struct LevelRecorder {
    levels: std::sync::Mutex<std::collections::HashMap<u32, bool>>,
}

impl LevelRecorder {
    pub fn level(&self, spi: u32) -> bool {
        self.levels
            .lock()
            .unwrap()
            .get(&spi)
            .copied()
            .unwrap_or(false)
    }
}

impl InterruptController for LevelRecorder {
    fn set_spi(&self, spi: u32, level: bool) {
        self.levels.lock().unwrap().insert(spi, level);
    }
    fn set_ppi(&self, _cpu: usize, _ppi: u32, _level: bool) {}
}
