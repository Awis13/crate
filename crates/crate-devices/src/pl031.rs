// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! An Arm PrimeCell PL031 real-time clock: seconds since the epoch, from the
//! host's clock plus whatever offset the guest sets by loading a new time.
//!
//! The match register is kept but never fires: no alarm interrupt yet.

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate_core::bus::MmioDevice;
use crate_core::irq::IrqLine;

pub const SIZE: u64 = 0x1000;

const DR: u64 = 0x00;
const MR: u64 = 0x04;
const LR: u64 = 0x08;
const CR: u64 = 0x0c;
const IMSC: u64 = 0x10;
const RIS: u64 = 0x14;
const MIS: u64 = 0x18;
const ICR: u64 = 0x1c;

/// PeriphID0-3 (part 0x031, designer 0x41 Arm) then CellID0-3.
const ID: [u8; 8] = [0x31, 0x10, 0x04, 0x00, 0x0d, 0xf0, 0x05, 0xb1];

struct State {
    offset: i64,
    mr: u32,
    lr: u32,
    imsc: u32,
    ris: u32,
}

pub struct Pl031 {
    state: Mutex<State>,
    irq: IrqLine,
    clock: fn() -> i64,
}

fn host_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

impl Pl031 {
    pub fn new(irq: IrqLine) -> Arc<Self> {
        Self::with_clock(irq, host_seconds)
    }

    fn with_clock(irq: IrqLine, clock: fn() -> i64) -> Arc<Self> {
        Arc::new(Pl031 {
            state: Mutex::new(State {
                offset: 0,
                mr: 0,
                lr: 0,
                imsc: 0,
                ris: 0,
            }),
            irq,
            clock,
        })
    }
}

impl MmioDevice for Pl031 {
    fn read(&self, offset: u64, _size: u8) -> u64 {
        let state = self.state.lock().unwrap();
        let value = match offset {
            DR => ((self.clock)() + state.offset) as u32,
            MR => state.mr,
            LR => state.lr,
            CR => 1,
            IMSC => state.imsc,
            RIS => state.ris,
            MIS => state.ris & state.imsc,
            0xfe0..=0xffc if offset % 4 == 0 => ID[((offset - 0xfe0) / 4) as usize] as u32,
            _ => 0,
        };
        value as u64
    }

    fn write(&self, offset: u64, _size: u8, value: u64) {
        let value = value as u32;
        let mut state = self.state.lock().unwrap();
        match offset {
            MR => state.mr = value,
            LR => {
                state.lr = value;
                state.offset = value as i64 - (self.clock)();
            }
            IMSC => state.imsc = value & 1,
            ICR => state.ris &= !value,
            _ => {}
        }
        self.irq.set(state.ris & state.imsc != 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate_core::irq::LevelRecorder;

    #[test]
    fn counts_host_seconds_from_what_was_loaded() {
        let rtc = Pl031::with_clock(IrqLine::new(Arc::new(LevelRecorder::default()), 2), || {
            1_000_000
        });
        assert_eq!(rtc.read(DR, 4), 1_000_000);
        assert_eq!(rtc.read(CR, 4), 1, "always running");
        rtc.write(LR, 4, 5);
        assert_eq!(rtc.read(DR, 4), 5);
        assert_eq!(rtc.read(LR, 4), 5);
        let id: Vec<u64> = (0..8).map(|i| rtc.read(0xfe0 + 4 * i, 4)).collect();
        assert_eq!(id, [0x31, 0x10, 0x04, 0x00, 0x0d, 0xf0, 0x05, 0xb1]);
    }
}
