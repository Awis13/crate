// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Split virtqueues (VIRTIO 1.2, section 2.7), seen from the device side.
//!
//! Everything in a queue is written by the guest and may be malformed or
//! change underfoot, so every index and length is checked before use and a
//! chain may not be longer than the queue (or its indirect table) holds.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{Ordering, fence};

use crate_core::memory::{GuestMemory, MemoryError};

const DESC_F_NEXT: u16 = 1;
const DESC_F_WRITE: u16 = 2;
const DESC_F_INDIRECT: u16 = 4;
const AVAIL_F_NO_INTERRUPT: u16 = 1;
const DESCRIPTOR_SIZE: u64 = 16;

/// Where the driver put a queue, as the transport learned it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueueConfig {
    pub size: u16,
    pub ready: bool,
    pub desc: u64,
    pub driver: u64,
    pub device: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum QueueError {
    Memory(MemoryError),
    /// The driver broke the ring's rules; the message says how.
    Malformed(&'static str),
}

impl From<MemoryError> for QueueError {
    fn from(e: MemoryError) -> Self {
        QueueError::Memory(e)
    }
}

impl fmt::Display for QueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueueError::Memory(e) => write!(f, "{e}"),
            QueueError::Malformed(m) => write!(f, "{m}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Descriptor {
    pub addr: u64,
    pub len: u32,
    /// Device-writable (a buffer for the device to fill), else device-readable.
    pub writable: bool,
}

/// One request: the head index the used ring returns, and its buffers with
/// any indirect table flattened in.
#[derive(Debug)]
pub struct Chain {
    pub head: u16,
    pub descriptors: Vec<Descriptor>,
}

impl Chain {
    pub fn readable(&self) -> impl Iterator<Item = &Descriptor> {
        self.descriptors.iter().filter(|d| !d.writable)
    }

    pub fn writable(&self) -> impl Iterator<Item = &Descriptor> {
        self.descriptors.iter().filter(|d| d.writable)
    }

    pub fn readable_len(&self) -> u64 {
        self.readable().map(|d| d.len as u64).sum()
    }

    pub fn writable_len(&self) -> u64 {
        self.writable().map(|d| d.len as u64).sum()
    }

    /// The device-readable buffers concatenated, up to `limit` bytes.
    pub fn read_all(&self, memory: &GuestMemory, limit: usize) -> Result<Vec<u8>, MemoryError> {
        let mut out = Vec::with_capacity((self.readable_len() as usize).min(limit));
        for d in self.readable() {
            let take = (d.len as usize).min(limit - out.len());
            let start = out.len();
            out.resize(start + take, 0);
            memory.read(d.addr, &mut out[start..])?;
            if out.len() == limit {
                break;
            }
        }
        Ok(out)
    }

    /// Writes `data` across the device-writable buffers in order; returns how
    /// much fitted.
    pub fn write_all(&self, memory: &GuestMemory, data: &[u8]) -> Result<usize, MemoryError> {
        let mut done = 0;
        for d in self.writable() {
            if done == data.len() {
                break;
            }
            let take = (d.len as usize).min(data.len() - done);
            memory.write(d.addr, &data[done..done + take])?;
            done += take;
        }
        Ok(done)
    }
}

pub struct Queue {
    memory: Arc<GuestMemory>,
    config: QueueConfig,
    next_avail: u16,
    next_used: u16,
}

impl Queue {
    pub fn new(memory: Arc<GuestMemory>, config: QueueConfig) -> Self {
        Queue {
            memory,
            config,
            next_avail: 0,
            next_used: 0,
        }
    }

    pub fn ready(&self) -> bool {
        self.config.ready && self.config.size > 0
    }

    pub fn memory(&self) -> &Arc<GuestMemory> {
        &self.memory
    }

    /// The next request the driver made available, if any.
    pub fn pop(&mut self) -> Result<Option<Chain>, QueueError> {
        if !self.ready() {
            return Ok(None);
        }
        let size = self.config.size;
        let avail = self.config.driver;
        let avail_idx = self.memory.load_u16_acquire(avail + 2)?;
        if avail_idx == self.next_avail {
            return Ok(None);
        }
        if avail_idx.wrapping_sub(self.next_avail) > size {
            return Err(QueueError::Malformed(
                "more requests available than the queue holds",
            ));
        }
        let slot = (self.next_avail % size) as u64;
        let head = self.memory.read_u16(avail + 4 + 2 * slot)?;
        self.next_avail = self.next_avail.wrapping_add(1);
        let descriptors = self.walk(head)?;
        Ok(Some(Chain { head, descriptors }))
    }

    fn read_descriptor(
        &self,
        table: u64,
        index: u16,
    ) -> Result<(Descriptor, u16, u16), QueueError> {
        let at = table + DESCRIPTOR_SIZE * index as u64;
        let addr = self.memory.read_u64(at)?;
        let len = self.memory.read_u32(at + 8)?;
        let flags = self.memory.read_u16(at + 12)?;
        let next = self.memory.read_u16(at + 14)?;
        Ok((
            Descriptor {
                addr,
                len,
                writable: flags & DESC_F_WRITE != 0,
            },
            flags,
            next,
        ))
    }

    fn walk(&self, head: u16) -> Result<Vec<Descriptor>, QueueError> {
        let size = self.config.size;
        let mut out = Vec::new();
        let mut index = head;
        let mut seen = 0u32;
        loop {
            if index >= size {
                return Err(QueueError::Malformed("descriptor index out of range"));
            }
            seen += 1;
            if seen > size as u32 {
                return Err(QueueError::Malformed("descriptor chain loops"));
            }
            let (d, flags, next) = self.read_descriptor(self.config.desc, index)?;
            if flags & DESC_F_INDIRECT != 0 {
                if flags & DESC_F_NEXT != 0 {
                    return Err(QueueError::Malformed("indirect descriptor with NEXT set"));
                }
                self.walk_indirect(d.addr, d.len, &mut out)?;
                return Ok(out);
            }
            out.push(d);
            if flags & DESC_F_NEXT == 0 {
                return Ok(out);
            }
            index = next;
        }
    }

    fn walk_indirect(
        &self,
        table: u64,
        len: u32,
        out: &mut Vec<Descriptor>,
    ) -> Result<(), QueueError> {
        if len == 0 || len as u64 % DESCRIPTOR_SIZE != 0 {
            return Err(QueueError::Malformed("indirect table length"));
        }
        let count = (len as u64 / DESCRIPTOR_SIZE).min(u16::MAX as u64) as u16;
        let mut index = 0u16;
        for _ in 0..count {
            let (d, flags, next) = self.read_descriptor(table, index)?;
            if flags & DESC_F_INDIRECT != 0 {
                return Err(QueueError::Malformed("nested indirect table"));
            }
            out.push(d);
            if flags & DESC_F_NEXT == 0 {
                return Ok(());
            }
            if next >= count {
                return Err(QueueError::Malformed("indirect index out of range"));
            }
            index = next;
        }
        Err(QueueError::Malformed("indirect chain loops"))
    }

    /// Returns a finished request to the driver, `len` being how many bytes
    /// the device wrote into its buffers.
    pub fn add_used(&mut self, head: u16, len: u32) -> Result<(), QueueError> {
        let used = self.config.device;
        let slot = (self.next_used % self.config.size) as u64;
        self.memory.write_u32(used + 4 + 8 * slot, head as u32)?;
        self.memory.write_u32(used + 8 + 8 * slot, len)?;
        self.next_used = self.next_used.wrapping_add(1);
        self.memory.store_u16_release(used + 2, self.next_used)?;
        Ok(())
    }

    /// Whether the driver wants an interrupt for the used buffers just added.
    pub fn needs_interrupt(&self) -> bool {
        // Order the used index store before reading the driver's flags.
        fence(Ordering::SeqCst);
        match self.memory.read_u16(self.config.driver) {
            Ok(flags) => flags & AVAIL_F_NO_INTERRUPT == 0,
            Err(_) => true,
        }
    }
}

/// Test helpers: a driver's side of a queue in guest memory.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate_core::memory::{HeapRam, RamBlock};

    pub const RAM_BASE: u64 = 0x4000_0000;
    pub const DESC: u64 = RAM_BASE;
    pub const AVAIL: u64 = RAM_BASE + 0x1000;
    pub const USED: u64 = RAM_BASE + 0x2000;
    pub const DATA: u64 = RAM_BASE + 0x10000;

    pub fn memory() -> Arc<GuestMemory> {
        let ram: Arc<dyn RamBlock> = Arc::new(HeapRam::new(0x10_0000).unwrap());
        Arc::new(GuestMemory::new(vec![(RAM_BASE, ram)]).unwrap())
    }

    pub fn config(size: u16) -> QueueConfig {
        QueueConfig {
            size,
            ready: true,
            desc: DESC,
            driver: AVAIL,
            device: USED,
        }
    }

    /// The driver's bookkeeping for one queue at DESC/AVAIL/USED.
    pub struct Driver {
        pub memory: Arc<GuestMemory>,
        pub size: u16,
        desc: u64,
        avail: u64,
        used: u64,
        next_desc: u16,
        avail_idx: u16,
        pub used_seen: u16,
    }

    impl Driver {
        pub fn new(memory: Arc<GuestMemory>, size: u16) -> Self {
            Self::at(memory, config(size))
        }

        /// A driver for the rings `config` places.
        pub fn at(memory: Arc<GuestMemory>, config: QueueConfig) -> Self {
            Driver {
                memory,
                size: config.size,
                desc: config.desc,
                avail: config.driver,
                used: config.device,
                next_desc: 0,
                avail_idx: 0,
                used_seen: 0,
            }
        }

        /// Makes a chain of (address, length, writable) buffers available;
        /// returns its head.
        pub fn offer(&mut self, buffers: &[(u64, u32, bool)]) -> u16 {
            let head = self.next_desc;
            for (i, &(addr, len, writable)) in buffers.iter().enumerate() {
                let index = self.next_desc;
                self.next_desc = (self.next_desc + 1) % self.size;
                let at = self.desc + 16 * index as u64;
                let mut flags = if writable { DESC_F_WRITE } else { 0 };
                if i + 1 < buffers.len() {
                    flags |= DESC_F_NEXT;
                }
                self.memory.write_u64(at, addr).unwrap();
                self.memory.write_u32(at + 8, len).unwrap();
                self.memory.write_u16(at + 12, flags).unwrap();
                self.memory.write_u16(at + 14, self.next_desc).unwrap();
            }
            let slot = (self.avail_idx % self.size) as u64;
            self.memory
                .write_u16(self.avail + 4 + 2 * slot, head)
                .unwrap();
            self.avail_idx = self.avail_idx.wrapping_add(1);
            self.memory
                .write_u16(self.avail + 2, self.avail_idx)
                .unwrap();
            head
        }

        /// The next (head, len) the device returned, if any.
        pub fn take_used(&mut self) -> Option<(u16, u32)> {
            let idx = self.memory.read_u16(self.used + 2).unwrap();
            if idx == self.used_seen {
                return None;
            }
            let slot = (self.used_seen % self.size) as u64;
            self.used_seen = self.used_seen.wrapping_add(1);
            let head = self.memory.read_u32(self.used + 4 + 8 * slot).unwrap() as u16;
            let len = self.memory.read_u32(self.used + 8 + 8 * slot).unwrap();
            Some((head, len))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    #[test]
    fn pops_chains_and_returns_them() {
        let memory = memory();
        let mut driver = Driver::new(memory.clone(), 8);
        let mut queue = Queue::new(memory.clone(), config(8));
        assert!(queue.pop().unwrap().is_none());
        let head = driver.offer(&[
            (DATA, 16, false),
            (DATA + 0x100, 512, true),
            (DATA + 0x400, 1, true),
        ]);
        let chain = queue.pop().unwrap().unwrap();
        assert_eq!(chain.head, head);
        assert_eq!(chain.descriptors.len(), 3);
        assert_eq!(chain.readable_len(), 16);
        assert_eq!(chain.writable_len(), 513);
        assert!(queue.pop().unwrap().is_none());
        queue.add_used(chain.head, 513).unwrap();
        assert_eq!(driver.take_used(), Some((head, 513)));
        assert!(queue.needs_interrupt());
        memory.write_u16(AVAIL, AVAIL_F_NO_INTERRUPT).unwrap();
        assert!(!queue.needs_interrupt());
    }

    #[test]
    fn follows_an_indirect_table() {
        let memory = memory();
        let table = DATA + 0x8000;
        for (i, (addr, len, flags)) in [(DATA, 4u32, DESC_F_NEXT), (DATA + 0x10, 8, DESC_F_WRITE)]
            .iter()
            .enumerate()
        {
            let at = table + 16 * i as u64;
            memory.write_u64(at, *addr).unwrap();
            memory.write_u32(at + 8, *len).unwrap();
            memory.write_u16(at + 12, *flags).unwrap();
            memory.write_u16(at + 14, 1).unwrap();
        }
        memory.write_u64(DESC, table).unwrap();
        memory.write_u32(DESC + 8, 32).unwrap();
        memory.write_u16(DESC + 12, DESC_F_INDIRECT).unwrap();
        memory.write_u16(AVAIL + 4, 0).unwrap();
        memory.write_u16(AVAIL + 2, 1).unwrap();
        let mut queue = Queue::new(memory, config(8));
        let chain = queue.pop().unwrap().unwrap();
        assert_eq!(
            chain.descriptors,
            vec![
                Descriptor {
                    addr: DATA,
                    len: 4,
                    writable: false
                },
                Descriptor {
                    addr: DATA + 0x10,
                    len: 8,
                    writable: true
                },
            ]
        );
    }

    #[test]
    fn refuses_malformed_rings() {
        let memory = memory();
        // A chain that points back at itself.
        memory.write_u16(DESC + 12, DESC_F_NEXT).unwrap();
        memory.write_u16(DESC + 14, 0).unwrap();
        memory.write_u16(AVAIL + 2, 1).unwrap();
        let mut queue = Queue::new(memory.clone(), config(4));
        assert_eq!(
            queue.pop().unwrap_err(),
            QueueError::Malformed("descriptor chain loops")
        );

        // A head beyond the table.
        memory.write_u16(AVAIL + 4 + 2, 9).unwrap();
        memory.write_u16(AVAIL + 2, 2).unwrap();
        assert_eq!(
            queue.pop().unwrap_err(),
            QueueError::Malformed("descriptor index out of range")
        );

        // An available index far ahead of the device.
        let mut queue = Queue::new(memory.clone(), config(4));
        memory.write_u16(AVAIL + 2, 100).unwrap();
        assert!(queue.pop().is_err());

        // A buffer outside RAM is caught when it is read.
        let chain = Chain {
            head: 0,
            descriptors: vec![Descriptor {
                addr: 0x10,
                len: 4,
                writable: false,
            }],
        };
        assert!(chain.read_all(&memory, 64).is_err());
    }

    #[test]
    fn reads_and_writes_across_buffers() {
        let memory = memory();
        memory.write(DATA, b"abc").unwrap();
        memory.write(DATA + 0x10, b"defg").unwrap();
        let chain = Chain {
            head: 0,
            descriptors: vec![
                Descriptor {
                    addr: DATA,
                    len: 3,
                    writable: false,
                },
                Descriptor {
                    addr: DATA + 0x10,
                    len: 4,
                    writable: false,
                },
                Descriptor {
                    addr: DATA + 0x20,
                    len: 2,
                    writable: true,
                },
                Descriptor {
                    addr: DATA + 0x30,
                    len: 8,
                    writable: true,
                },
            ],
        };
        assert_eq!(chain.read_all(&memory, 100).unwrap(), b"abcdefg");
        assert_eq!(chain.read_all(&memory, 5).unwrap(), b"abcde");
        assert_eq!(chain.write_all(&memory, b"hello").unwrap(), 5);
        let mut a = [0; 2];
        let mut b = [0; 3];
        memory.read(DATA + 0x20, &mut a).unwrap();
        memory.read(DATA + 0x30, &mut b).unwrap();
        assert_eq!((&a, &b), (b"he", b"llo"));
    }
}
