// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A virtio console (VIRTIO 1.2, section 5.3) with one port and no
//! multiport: Linux's hvc0 (hvc1 for a second device, and so on).
//!
//! Output costs one notification per batch of bytes where the PL011 costs an
//! exit per byte, which is why it exists beside the UART.

use std::sync::{Arc, Mutex};

use crate_core::warn;

use super::{
    Activation, F_VERSION_1, ID_CONSOLE, Interrupt, Queue, VirtioDevice, read_config_bytes,
};

const QUEUE_SIZE: u16 = 64;
const MAX_WRITE: usize = 1 << 20;

pub type Output = Arc<dyn Fn(&[u8]) + Send + Sync>;

#[derive(Default)]
struct Receiver {
    queue: Option<(Queue, Arc<Interrupt>)>,
}

/// The app's side of the console's input.
#[derive(Clone)]
pub struct ConsoleInput(Arc<Mutex<Receiver>>);

impl ConsoleInput {
    /// Bytes for the guest. Takes what fits in the buffers the guest has
    /// posted, possibly nothing, and returns how many that was.
    pub fn write(&self, bytes: &[u8]) -> usize {
        let mut rx = self.0.lock().unwrap();
        let Some((queue, interrupt)) = &mut rx.queue else {
            return 0;
        };
        let mut taken = 0;
        while taken < bytes.len() {
            let chain = match queue.pop() {
                Ok(Some(chain)) => chain,
                Ok(None) => break,
                Err(e) => {
                    warn!("virtio-console: {e}");
                    break;
                }
            };
            let n = chain
                .write_all(queue.memory(), &bytes[taken..])
                .unwrap_or(0);
            if queue.add_used(chain.head, n as u32).is_err() {
                break;
            }
            taken += n;
        }
        if taken > 0 && queue.needs_interrupt() {
            interrupt.signal_used();
        }
        taken
    }
}

pub struct Console {
    output: Output,
    rx: ConsoleInput,
    tx: Option<(Queue, Arc<Interrupt>)>,
}

impl Console {
    pub fn new(output: Output) -> Self {
        Console {
            output,
            rx: ConsoleInput(Arc::default()),
            tx: None,
        }
    }

    pub fn input(&self) -> ConsoleInput {
        self.rx.clone()
    }
}

impl VirtioDevice for Console {
    fn device_id(&self) -> u32 {
        ID_CONSOLE
    }

    fn features(&self) -> u64 {
        F_VERSION_1
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &[QUEUE_SIZE, QUEUE_SIZE]
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        read_config_bytes(&[], offset, data);
    }

    fn activate(&mut self, activation: Activation) {
        let mut queues = activation.queues.into_iter();
        let (Some(rx), Some(tx)) = (queues.next(), queues.next()) else {
            return;
        };
        self.rx.0.lock().unwrap().queue = Some((rx, activation.interrupt.clone()));
        self.tx = Some((tx, activation.interrupt));
    }

    fn notify(&mut self, index: u16) {
        if index != 1 {
            return;
        }
        let Some((queue, interrupt)) = &mut self.tx else {
            return;
        };
        let mut used = false;
        loop {
            let chain = match queue.pop() {
                Ok(Some(chain)) => chain,
                Ok(None) => break,
                Err(e) => {
                    warn!("virtio-console: {e}");
                    break;
                }
            };
            match chain.read_all(queue.memory(), MAX_WRITE) {
                Ok(bytes) if !bytes.is_empty() => (self.output)(&bytes),
                Ok(_) => {}
                Err(e) => warn!("virtio-console: {e}"),
            }
            if queue.add_used(chain.head, 0).is_err() {
                break;
            }
            used = true;
        }
        if used && queue.needs_interrupt() {
            interrupt.signal_used();
        }
    }

    fn reset(&mut self) {
        self.rx.0.lock().unwrap().queue = None;
        self.tx = None;
    }
}

#[cfg(test)]
mod tests {
    use super::super::queue::testing::*;
    use super::*;
    use crate_core::irq::{IrqLine, LevelRecorder};

    #[test]
    fn output_and_input() {
        let memory = memory();
        let printed = Arc::new(Mutex::new(Vec::new()));
        let out = printed.clone();
        let mut dev = Console::new(Arc::new(move |b: &[u8]| {
            out.lock().unwrap().extend_from_slice(b)
        }));
        let input = dev.input();
        assert_eq!(input.write(b"x"), 0, "nothing before the driver is ready");
        // Both queues share the test rings; only one is used at a time here.
        dev.activate(Activation {
            memory: memory.clone(),
            queues: vec![
                Queue::new(memory.clone(), config(8)),
                Queue::new(memory.clone(), config(8)),
            ],
            interrupt: Interrupt::new(IrqLine::new(Arc::new(LevelRecorder::default()), 0)),
            features: F_VERSION_1,
        });
        let mut driver = Driver::new(memory.clone(), 8);
        memory.write(DATA, b"login: ").unwrap();
        driver.offer(&[(DATA, 7, false)]);
        dev.notify(1);
        assert_eq!(&*printed.lock().unwrap(), b"login: ");
        assert_eq!(driver.take_used(), Some((0, 0)));

        driver.offer(&[(DATA + 0x100, 4, true)]);
        assert_eq!(input.write(b"root\n"), 4, "one 4-byte buffer posted");
        let mut got = [0; 4];
        memory.read(DATA + 0x100, &mut got).unwrap();
        assert_eq!(&got, b"root");
        assert_eq!(driver.take_used(), Some((1, 4)));
    }
}
