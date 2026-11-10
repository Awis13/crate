// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A virtio network device (VIRTIO 1.2, section 5.1) on a [`NetBackend`]:
//! one receive and one transmit queue, a MAC address, no offloads.
//!
//! Frames from the guest go to the backend on the vCPU that notified the
//! transmit queue. Frames for the guest arrive on the backend's thread and go
//! straight into receive buffers; when the guest has posted none, up to
//! BACKLOG frames wait for it and anything more is dropped, which TCP
//! recovers from.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate_core::net::{FrameSink, NetBackend};
use crate_core::warn;

use super::{
    Activation, F_INDIRECT_DESC, F_VERSION_1, ID_NET, Interrupt, Queue, VirtioDevice,
    read_config_bytes,
};

const F_MAC: u64 = 1 << 5;
const F_STATUS: u64 = 1 << 16;
const STATUS_LINK_UP: u16 = 1;

/// virtio_net_hdr as VERSION_1 lays it out, num_buffers included.
const HEADER: usize = 12;
const QUEUE_SIZE: u16 = 256;
const BACKLOG: usize = 256;
const MAX_FRAME: usize = 65536;

struct Receiver {
    queue: Option<(Queue, Arc<Interrupt>)>,
    backlog: VecDeque<Vec<u8>>,
}

impl Receiver {
    /// Puts as much of the backlog as fits into the guest's buffers.
    fn flush(&mut self) {
        let Some((queue, interrupt)) = &mut self.queue else {
            self.backlog.clear();
            return;
        };
        let mut used = false;
        while !self.backlog.is_empty() {
            let chain = match queue.pop() {
                Ok(Some(chain)) => chain,
                Ok(None) => break,
                Err(e) => {
                    warn!("virtio-net: receive queue: {e}");
                    break;
                }
            };
            let frame = self.backlog.pop_front().unwrap_or_default();
            let mut packet = Vec::with_capacity(HEADER + frame.len());
            packet.extend_from_slice(&[0; HEADER - 2]);
            packet.extend_from_slice(&1u16.to_le_bytes()); // num_buffers
            packet.extend_from_slice(&frame);
            let written = if chain.writable_len() < packet.len() as u64 {
                warn!(
                    "virtio-net: a {} byte frame does not fit a receive buffer",
                    frame.len()
                );
                0
            } else {
                chain.write_all(queue.memory(), &packet).unwrap_or(0)
            };
            if queue.add_used(chain.head, written as u32).is_err() {
                break;
            }
            used = true;
        }
        if used && queue.needs_interrupt() {
            interrupt.signal_used();
        }
    }
}

struct Sink(Arc<Mutex<Receiver>>);

impl FrameSink for Sink {
    fn deliver(&self, frame: &[u8]) {
        let mut rx = self.0.lock().unwrap();
        if rx.queue.is_none() {
            return;
        }
        if rx.backlog.len() >= BACKLOG {
            rx.backlog.pop_front();
        }
        rx.backlog.push_back(frame.to_vec());
        rx.flush();
    }
}

pub struct Net {
    mac: [u8; 6],
    backend: Arc<dyn NetBackend>,
    rx: Arc<Mutex<Receiver>>,
    tx: Option<(Queue, Arc<Interrupt>)>,
}

impl Net {
    pub fn new(backend: Arc<dyn NetBackend>, mac: [u8; 6]) -> Self {
        let rx = Arc::new(Mutex::new(Receiver {
            queue: None,
            backlog: VecDeque::new(),
        }));
        backend.start(Arc::new(Sink(rx.clone())));
        Net {
            mac,
            backend,
            rx,
            tx: None,
        }
    }

    fn transmit(&mut self) {
        let Some((queue, interrupt)) = &mut self.tx else {
            return;
        };
        let mut used = false;
        loop {
            let chain = match queue.pop() {
                Ok(Some(chain)) => chain,
                Ok(None) => break,
                Err(e) => {
                    warn!("virtio-net: transmit queue: {e}");
                    break;
                }
            };
            match chain.read_all(queue.memory(), HEADER + MAX_FRAME) {
                Ok(packet) if packet.len() > HEADER => self.backend.send(&packet[HEADER..]),
                Ok(_) => {}
                Err(e) => warn!("virtio-net: {e}"),
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
}

impl VirtioDevice for Net {
    fn device_id(&self) -> u32 {
        ID_NET
    }

    fn features(&self) -> u64 {
        F_VERSION_1 | F_INDIRECT_DESC | F_MAC | F_STATUS
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &[QUEUE_SIZE, QUEUE_SIZE]
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        // mac, status.
        let mut config = [0u8; 8];
        config[..6].copy_from_slice(&self.mac);
        config[6..8].copy_from_slice(&STATUS_LINK_UP.to_le_bytes());
        read_config_bytes(&config, offset, data);
    }

    fn activate(&mut self, activation: Activation) {
        let mut queues = activation.queues.into_iter();
        let (Some(rx), Some(tx)) = (queues.next(), queues.next()) else {
            return;
        };
        self.rx.lock().unwrap().queue = Some((rx, activation.interrupt.clone()));
        self.tx = Some((tx, activation.interrupt));
    }

    fn notify(&mut self, index: u16) {
        match index {
            0 => self.rx.lock().unwrap().flush(),
            1 => self.transmit(),
            _ => {}
        }
    }

    fn reset(&mut self) {
        let mut rx = self.rx.lock().unwrap();
        rx.queue = None;
        rx.backlog.clear();
        self.tx = None;
    }
}

#[cfg(test)]
mod tests {
    use super::super::queue::QueueConfig;
    use super::super::queue::testing::*;
    use super::*;
    use crate_core::irq::{IrqLine, LevelRecorder};
    use std::sync::OnceLock;

    #[derive(Default)]
    struct Loop {
        sent: Mutex<Vec<Vec<u8>>>,
        sink: OnceLock<Arc<dyn FrameSink>>,
    }

    impl NetBackend for Loop {
        fn start(&self, sink: Arc<dyn FrameSink>) {
            let _ = self.sink.set(sink);
        }
        fn send(&self, frame: &[u8]) {
            self.sent.lock().unwrap().push(frame.to_vec());
        }
    }

    use crate_core::memory::GuestMemory;

    fn rx_config() -> QueueConfig {
        let base = DATA + 0x8_0000;
        QueueConfig {
            size: 8,
            ready: true,
            desc: base,
            driver: base + 0x1000,
            device: base + 0x2000,
        }
    }

    fn net() -> (Net, Arc<Loop>, Arc<GuestMemory>) {
        let memory = memory();
        let backend = Arc::new(Loop::default());
        let mut dev = Net::new(backend.clone(), [2, 0x63, 0x72, 0x61, 0x74, 0x65]);
        dev.activate(Activation {
            memory: memory.clone(),
            queues: vec![
                Queue::new(memory.clone(), rx_config()),
                Queue::new(memory.clone(), config(8)),
            ],
            interrupt: Interrupt::new(IrqLine::new(Arc::new(LevelRecorder::default()), 0)),
            features: dev.features(),
        });
        (dev, backend, memory)
    }

    #[test]
    fn reports_its_mac_and_link() {
        let (dev, ..) = net();
        let mut config = [0; 8];
        dev.read_config(0, &mut config);
        assert_eq!(config, [2, 0x63, 0x72, 0x61, 0x74, 0x65, 1, 0]);
    }

    #[test]
    fn transmits_frames_without_their_header() {
        let (mut dev, backend, memory) = net();
        let mut driver = Driver::new(memory.clone(), 8);
        memory.write(DATA, &[0; HEADER]).unwrap();
        memory.write(DATA + 0x100, b"frame").unwrap();
        driver.offer(&[(DATA, HEADER as u32, false), (DATA + 0x100, 5, false)]);
        dev.notify(1);
        assert_eq!(*backend.sent.lock().unwrap(), vec![b"frame".to_vec()]);
        assert_eq!(driver.take_used().map(|u| u.1), Some(0));
    }

    #[test]
    fn holds_frames_until_buffers_are_posted() {
        let (mut dev, backend, memory) = net();
        let sink = backend.sink.get().unwrap().clone();
        sink.deliver(b"early");
        let mut driver = Driver::at(memory.clone(), rx_config());
        assert_eq!(driver.take_used(), None);
        let head = driver.offer(&[(DATA + 0x4000, 2048, true)]);
        dev.notify(0);
        assert_eq!(driver.take_used(), Some((head, (HEADER + 5) as u32)));
        let mut got = [0; HEADER + 5];
        memory.read(DATA + 0x4000, &mut got).unwrap();
        assert_eq!(&got[HEADER - 2..HEADER], &[1, 0], "num_buffers");
        assert_eq!(&got[HEADER..], b"early");

        // With a buffer already posted, a frame goes straight in.
        let head = driver.offer(&[(DATA + 0x5000, 2048, true)]);
        sink.deliver(b"second");
        assert_eq!(driver.take_used(), Some((head, (HEADER + 6) as u32)));
    }
}
