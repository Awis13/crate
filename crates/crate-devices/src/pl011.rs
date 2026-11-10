// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! An Arm PrimeCell PL011 UART (Arm DDI 0183), revision r1p5: Linux's
//! ttyAMA0, and the console from the kernel's first line.
//!
//! Bytes the guest writes leave at once, so the transmit FIFO is always empty:
//! TXFE always reads one, TXFF and BUSY never do, and the transmit interrupt
//! is raised for as long as the guest unmasks it, which is how a FIFO that
//! drains instantly behaves. Received bytes wait in a 32-byte FIFO; the
//! receive interrupt follows the FIFO's trigger level and the receive timeout
//! interrupt stands while anything waits, so a lone keystroke is delivered.
//! Line settings (baud rate, framing) are stored and otherwise ignored.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate_core::bus::MmioDevice;
use crate_core::irq::IrqLine;

pub const SIZE: u64 = 0x1000;
const FIFO_DEPTH: usize = 32;

const DR: u64 = 0x000;
const RSR: u64 = 0x004;
const FR: u64 = 0x018;
const ILPR: u64 = 0x020;
const IBRD: u64 = 0x024;
const FBRD: u64 = 0x028;
const LCR_H: u64 = 0x02c;
const CR: u64 = 0x030;
const IFLS: u64 = 0x034;
const IMSC: u64 = 0x038;
const RIS: u64 = 0x03c;
const MIS: u64 = 0x040;
const ICR: u64 = 0x044;
const DMACR: u64 = 0x048;

const FR_RXFE: u32 = 1 << 4;
const FR_RXFF: u32 = 1 << 6;
const FR_TXFE: u32 = 1 << 7;

const INT_RX: u32 = 1 << 4;
const INT_TX: u32 = 1 << 5;
const INT_RT: u32 = 1 << 6;
const INT_ALL: u32 = 0x7ff;

const CR_LBE: u32 = 1 << 7;

/// PeriphID0-3 (part 0x011, designer 0x41 Arm, revision 3) then CellID0-3.
const ID: [u8; 8] = [0x11, 0x10, 0x34, 0x00, 0x0d, 0xf0, 0x05, 0xb1];

pub type Output = Arc<dyn Fn(&[u8]) + Send + Sync>;

struct State {
    rx: VecDeque<u8>,
    /// Interrupts other than RX, TX and RT that ICR clears (errors, modem).
    latched: u32,
    imsc: u32,
    cr: u32,
    lcr_h: u32,
    ibrd: u32,
    fbrd: u32,
    ifls: u32,
    ilpr: u32,
    dmacr: u32,
    line: bool,
}

impl State {
    fn ris(&self) -> u32 {
        let mut ris = self.latched | INT_TX;
        // IFLS bits 5:3 pick the receive trigger: 1/8, 1/4, 1/2, 3/4 or 7/8 full.
        let trigger = [4, 8, 16, 24, 28][((self.ifls >> 3) & 7).min(4) as usize];
        if self.rx.len() >= trigger {
            ris |= INT_RX;
        }
        if !self.rx.is_empty() {
            ris |= INT_RT;
        }
        ris
    }
}

pub struct Pl011 {
    state: Mutex<State>,
    irq: IrqLine,
    output: Output,
}

impl Pl011 {
    pub fn new(irq: IrqLine, output: Output) -> Arc<Self> {
        Arc::new(Pl011 {
            state: Mutex::new(State {
                rx: VecDeque::with_capacity(FIFO_DEPTH),
                latched: 0,
                imsc: 0,
                cr: 0x300,
                lcr_h: 0,
                ibrd: 0,
                fbrd: 0,
                ifls: 0x12,
                ilpr: 0,
                dmacr: 0,
                line: false,
            }),
            irq,
            output,
        })
    }

    /// Bytes for the guest to read. Takes what fits in the receive FIFO and
    /// returns how many that was.
    pub fn receive(&self, bytes: &[u8]) -> usize {
        let mut state = self.state.lock().unwrap();
        let room = FIFO_DEPTH - state.rx.len();
        let taken = bytes.len().min(room);
        state.rx.extend(&bytes[..taken]);
        self.update_irq(&mut state);
        taken
    }

    fn update_irq(&self, state: &mut State) {
        let level = state.ris() & state.imsc != 0;
        if level != state.line {
            state.line = level;
            self.irq.set(level);
        }
    }
}

impl MmioDevice for Pl011 {
    fn read(&self, offset: u64, _size: u8) -> u64 {
        let mut state = self.state.lock().unwrap();
        let value = match offset {
            DR => {
                let byte = state.rx.pop_front().unwrap_or(0);
                self.update_irq(&mut state);
                byte as u32
            }
            RSR => 0,
            FR => {
                let mut fr = FR_TXFE;
                if state.rx.is_empty() {
                    fr |= FR_RXFE;
                }
                if state.rx.len() == FIFO_DEPTH {
                    fr |= FR_RXFF;
                }
                fr
            }
            ILPR => state.ilpr,
            IBRD => state.ibrd,
            FBRD => state.fbrd,
            LCR_H => state.lcr_h,
            CR => state.cr,
            IFLS => state.ifls,
            IMSC => state.imsc,
            RIS => state.ris(),
            MIS => state.ris() & state.imsc,
            DMACR => state.dmacr,
            0xfe0..=0xffc if offset % 4 == 0 => ID[((offset - 0xfe0) / 4) as usize] as u32,
            _ => 0,
        };
        value as u64
    }

    fn write(&self, offset: u64, _size: u8, value: u64) {
        let value = value as u32;
        let mut state = self.state.lock().unwrap();
        match offset {
            DR => {
                let byte = value as u8;
                if state.cr & CR_LBE != 0 {
                    // Loopback: what is sent is received.
                    if state.rx.len() < FIFO_DEPTH {
                        state.rx.push_back(byte);
                    }
                } else {
                    // Out of the lock: the app's callback may take its time.
                    drop(state);
                    (self.output)(&[byte]);
                    state = self.state.lock().unwrap();
                }
            }
            RSR => {}
            ILPR => state.ilpr = value & 0xff,
            IBRD => state.ibrd = value & 0xffff,
            FBRD => state.fbrd = value & 0x3f,
            LCR_H => state.lcr_h = value & 0xff,
            CR => state.cr = value & 0xff87,
            IFLS => state.ifls = value & 0x3f,
            IMSC => state.imsc = value & INT_ALL,
            ICR => state.latched &= !value,
            DMACR => state.dmacr = value & 7,
            _ => {}
        }
        self.update_irq(&mut state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate_core::irq::LevelRecorder;

    fn uart() -> (Arc<Pl011>, Arc<LevelRecorder>, Arc<Mutex<Vec<u8>>>) {
        let chip = Arc::new(LevelRecorder::default());
        let sent = Arc::new(Mutex::new(Vec::new()));
        let out = sent.clone();
        let uart = Pl011::new(
            IrqLine::new(chip.clone(), 1),
            Arc::new(move |b| out.lock().unwrap().extend(b)),
        );
        (uart, chip, sent)
    }

    #[test]
    fn identifies_as_a_pl011_primecell() {
        let (uart, _, _) = uart();
        let id: Vec<u64> = (0..8).map(|i| uart.read(0xfe0 + 4 * i, 4)).collect();
        assert_eq!(id, [0x11, 0x10, 0x34, 0x00, 0x0d, 0xf0, 0x05, 0xb1]);
    }

    #[test]
    fn transmits_at_once() {
        let (uart, _, sent) = uart();
        for b in b"hi\n" {
            uart.write(DR, 4, *b as u64);
        }
        assert_eq!(&*sent.lock().unwrap(), b"hi\n");
        let fr = uart.read(FR, 4) as u32;
        assert_ne!(fr & FR_TXFE, 0);
        assert_eq!(fr & (1 << 5 | 1 << 3), 0, "never full, never busy");
    }

    #[test]
    fn loopback_does_not_leave() {
        let (uart, _, sent) = uart();
        uart.write(CR, 4, (CR_LBE | 0x301) as u64);
        uart.write(DR, 4, 0x41);
        assert!(sent.lock().unwrap().is_empty());
        assert_eq!(uart.read(DR, 4), 0x41);
    }

    #[test]
    fn receive_fifo_and_its_interrupts() {
        let (uart, chip, _) = uart();
        assert_ne!(uart.read(FR, 4) as u32 & FR_RXFE, 0);
        uart.write(IMSC, 4, (INT_RX | INT_RT) as u64);
        assert!(!chip.level(1));
        assert_eq!(uart.receive(b"a"), 1);
        assert!(chip.level(1), "the timeout interrupt delivers a lone byte");
        assert_eq!(uart.read(MIS, 4) as u32, INT_RT);
        assert_eq!(uart.read(DR, 4), b'a' as u64);
        assert!(!chip.level(1));
        assert_eq!(uart.receive(&[b'x'; 40]), 32, "the FIFO holds 32");
        assert_ne!(uart.read(FR, 4) as u32 & FR_RXFF, 0);
        assert_eq!(
            uart.read(RIS, 4) as u32 & INT_RX,
            INT_RX,
            "past the trigger level"
        );
        assert_eq!(uart.receive(b"y"), 0);
        for _ in 0..32 {
            uart.read(DR, 4);
        }
        assert!(!chip.level(1));
    }

    #[test]
    fn transmit_interrupt_stands_while_unmasked() {
        let (uart, chip, _) = uart();
        uart.write(IMSC, 4, INT_TX as u64);
        assert!(chip.level(1));
        uart.write(ICR, 4, INT_ALL as u64);
        assert!(chip.level(1), "the FIFO is still empty");
        uart.write(IMSC, 4, 0);
        assert!(!chip.level(1));
    }
}
