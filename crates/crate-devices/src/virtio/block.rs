// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A virtio block device (VIRTIO 1.2, section 5.2) on a [`BlockBackend`]: one
//! request queue, 512-byte sectors, reads, writes, flushes and the device ID.
//!
//! Requests run on the vCPU that notifies the queue, reading and writing guest
//! memory in place. A thread of its own per disk is the obvious next step once
//! anything measures it.

use std::sync::Arc;

use crate_core::block::BlockBackend;
use crate_core::memory::GuestMemory;
use crate_core::warn;

use super::queue::{Chain, Descriptor};
use super::{
    Activation, F_INDIRECT_DESC, F_VERSION_1, ID_BLOCK, Interrupt, Queue, VirtioDevice,
    read_config_bytes,
};

const F_SEG_MAX: u64 = 1 << 2;
const F_RO: u64 = 1 << 5;
const F_FLUSH: u64 = 1 << 9;

const T_IN: u32 = 0;
const T_OUT: u32 = 1;
const T_FLUSH: u32 = 4;
const T_GET_ID: u32 = 8;

const S_OK: u8 = 0;
const S_IOERR: u8 = 1;
const S_UNSUPP: u8 = 2;

const SECTOR: u64 = 512;
const QUEUE_SIZE: u16 = 256;
const HEADER: u64 = 16;
const ID_LEN: usize = 20;

pub struct Block {
    disk: Arc<dyn BlockBackend>,
    id: [u8; ID_LEN],
    active: Option<(Queue, Arc<Interrupt>)>,
}

impl Block {
    /// `id` is what the guest reads as the device's serial number
    /// (/sys/block/vda/serial), up to 20 bytes.
    pub fn new(disk: Arc<dyn BlockBackend>, id: &str) -> Self {
        let mut bytes = [0; ID_LEN];
        let n = id.len().min(ID_LEN);
        bytes[..n].copy_from_slice(&id.as_bytes()[..n]);
        Block {
            disk,
            id: bytes,
            active: None,
        }
    }

    fn sectors(&self) -> u64 {
        self.disk.len() / SECTOR
    }

    /// Runs one request; returns how many bytes it wrote into the chain.
    fn process(&self, memory: &GuestMemory, chain: &Chain) -> u32 {
        let readable: Vec<Descriptor> = chain.readable().copied().collect();
        let writable: Vec<Descriptor> = chain.writable().copied().collect();
        let Some(status_at) = last_byte(&writable) else {
            warn!("virtio-blk: a request with nowhere to put its status");
            return 0;
        };
        let mut header = [0u8; HEADER as usize];
        let header_ok = chain.read_all(memory, HEADER as usize).is_ok_and(|h| {
            let ok = h.len() == HEADER as usize;
            header[..h.len()].copy_from_slice(&h);
            ok
        });
        let (status, written) = if header_ok {
            let kind = u32::from_le_bytes(header[0..4].try_into().unwrap());
            let sector = u64::from_le_bytes(header[8..16].try_into().unwrap());
            self.run(memory, kind, sector, &readable, &writable)
        } else {
            (S_IOERR, 0)
        };
        if memory.write(status_at, &[status]).is_err() {
            return 0;
        }
        written + 1
    }

    fn run(
        &self,
        memory: &GuestMemory,
        kind: u32,
        sector: u64,
        readable: &[Descriptor],
        writable: &[Descriptor],
    ) -> (u8, u32) {
        match kind {
            T_IN => {
                let data = trim(writable, 0, 1);
                match self.transfer(memory, sector, &data, false) {
                    Ok(n) => (S_OK, n as u32),
                    Err(()) => (S_IOERR, 0),
                }
            }
            T_OUT if self.disk.read_only() => (S_IOERR, 0),
            T_OUT => {
                let data = trim(readable, HEADER, 0);
                match self.transfer(memory, sector, &data, true) {
                    Ok(_) => (S_OK, 0),
                    Err(()) => (S_IOERR, 0),
                }
            }
            T_FLUSH => match self.disk.flush() {
                Ok(()) => (S_OK, 0),
                Err(e) => {
                    warn!("virtio-blk: flush failed: {e}");
                    (S_IOERR, 0)
                }
            },
            T_GET_ID => {
                let data = trim(writable, 0, 1);
                let chain = Chain {
                    head: 0,
                    descriptors: data,
                };
                match chain.write_all(memory, &self.id) {
                    Ok(n) => (S_OK, n as u32),
                    Err(_) => (S_IOERR, 0),
                }
            }
            _ => (S_UNSUPP, 0),
        }
    }

    /// Moves the bytes of `buffers` to or from the disk from `sector`.
    fn transfer(
        &self,
        memory: &GuestMemory,
        sector: u64,
        buffers: &[Descriptor],
        write: bool,
    ) -> Result<u64, ()> {
        let total: u64 = buffers.iter().map(|d| d.len as u64).sum();
        let start = sector.checked_mul(SECTOR).ok_or(())?;
        if start
            .checked_add(total)
            .is_none_or(|end| end > self.disk.len())
        {
            warn!("virtio-blk: request past the end of the disk (sector {sector}, {total} bytes)");
            return Err(());
        }
        let mut offset = start;
        for d in buffers {
            let len = d.len as u64;
            let host = memory.host_range(d.addr, len).map_err(|_| ())?;
            // SAFETY: host_range checked that the span lies in guest RAM, which
            // outlives this call. The guest may touch these bytes meanwhile, as
            // it could during a real device's DMA; they are only copied.
            let buf = unsafe { std::slice::from_raw_parts_mut(host, len as usize) };
            let result = if write {
                self.disk.write_at(buf, offset)
            } else {
                self.disk.read_at(buf, offset)
            };
            if let Err(e) = result {
                warn!(
                    "virtio-blk: {} at {offset:#x} failed: {e}",
                    if write { "write" } else { "read" }
                );
                return Err(());
            }
            offset += len;
        }
        Ok(total)
    }
}

/// The guest address of the last byte of `buffers`.
fn last_byte(buffers: &[Descriptor]) -> Option<u64> {
    let last = buffers.iter().rev().find(|d| d.len > 0)?;
    Some(last.addr + last.len as u64 - 1)
}

/// `buffers` without the first `front` and last `back` bytes.
fn trim(buffers: &[Descriptor], front: u64, back: u64) -> Vec<Descriptor> {
    let total: u64 = buffers.iter().map(|d| d.len as u64).sum();
    let end = total.saturating_sub(back);
    let mut out = Vec::new();
    let mut at = 0u64;
    for d in buffers {
        let (lo, hi) = (at, at + d.len as u64);
        at = hi;
        let start = lo.max(front);
        let stop = hi.min(end);
        if start < stop {
            out.push(Descriptor {
                addr: d.addr + (start - lo),
                len: (stop - start) as u32,
                writable: d.writable,
            });
        }
    }
    out
}

impl VirtioDevice for Block {
    fn device_id(&self) -> u32 {
        ID_BLOCK
    }

    fn features(&self) -> u64 {
        let ro = if self.disk.read_only() { F_RO } else { 0 };
        F_VERSION_1 | F_INDIRECT_DESC | F_SEG_MAX | F_FLUSH | ro
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &[QUEUE_SIZE]
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        // capacity, size_max, seg_max.
        let mut config = [0u8; 16];
        config[0..8].copy_from_slice(&self.sectors().to_le_bytes());
        config[12..16].copy_from_slice(&(QUEUE_SIZE as u32 - 2).to_le_bytes());
        read_config_bytes(&config, offset, data);
    }

    fn activate(&mut self, mut activation: Activation) {
        let queue = activation.queues.remove(0);
        let interrupt = activation.interrupt.clone();
        self.disk.on_activate(Arc::new(move || interrupt.signal_config()));
        self.active = Some((queue, activation.interrupt));
    }

    fn notify(&mut self, index: u16) {
        if index != 0 {
            return;
        }
        let Some((mut queue, interrupt)) = self.active.take() else {
            return;
        };
        let memory = queue.memory().clone();
        let mut used = false;
        loop {
            match queue.pop() {
                Ok(Some(chain)) => {
                    let written = self.process(&memory, &chain);
                    if queue.add_used(chain.head, written).is_err() {
                        break;
                    }
                    used = true;
                }
                Ok(None) => break,
                Err(e) => {
                    warn!("virtio-blk: {e}");
                    break;
                }
            }
        }
        if used && queue.needs_interrupt() {
            interrupt.signal_used();
        }
        self.active = Some((queue, interrupt));
    }

    fn reset(&mut self) {
        self.active = None;
    }
}

#[cfg(test)]
mod tests {
    use super::super::queue::testing::*;
    use super::*;
    use crate_core::block::MemoryDisk;
    use crate_core::irq::{IrqLine, LevelRecorder};

    fn block(disk: Vec<u8>) -> (Block, Driver, Arc<GuestMemory>, Arc<LevelRecorder>) {
        let memory = memory();
        let chip = Arc::new(LevelRecorder::default());
        let mut dev = Block::new(Arc::new(MemoryDisk::new(disk)), "crate-test-disk");
        dev.activate(Activation {
            memory: memory.clone(),
            queues: vec![Queue::new(memory.clone(), config(16))],
            interrupt: Interrupt::new(IrqLine::new(chip.clone(), 0)),
            features: dev.features(),
        });
        (dev, Driver::new(memory.clone(), 16), memory, chip)
    }

    fn header(memory: &GuestMemory, at: u64, kind: u32, sector: u64) {
        memory.write_u32(at, kind).unwrap();
        memory.write_u32(at + 4, 0).unwrap();
        memory.write_u64(at + 8, sector).unwrap();
    }

    fn status(memory: &GuestMemory, at: u64) -> u8 {
        let mut s = [0xff];
        memory.read(at, &mut s).unwrap();
        s[0]
    }

    #[test]
    fn reports_capacity_and_segments() {
        let (dev, ..) = block(vec![0; 4096]);
        let mut cap = [0; 8];
        dev.read_config(0, &mut cap);
        assert_eq!(u64::from_le_bytes(cap), 8);
        let mut seg = [0; 4];
        dev.read_config(12, &mut seg);
        assert_eq!(u32::from_le_bytes(seg), 254);
        assert_eq!(dev.features() & F_RO, 0);
    }

    #[test]
    fn reads_sectors_across_buffers() {
        let disk: Vec<u8> = (0..4096u32).map(|i| (i / 512) as u8 + 1).collect();
        let (mut dev, mut driver, memory, chip) = block(disk);
        header(&memory, DATA, T_IN, 2);
        // Header and status in one chain with the data split in two buffers.
        let head = driver.offer(&[
            (DATA, 16, false),
            (DATA + 0x1000, 512, true),
            (DATA + 0x2000, 512, true),
            (DATA + 0x3000, 1, true),
        ]);
        dev.notify(0);
        assert_eq!(driver.take_used(), Some((head, 1025)));
        assert_eq!(status(&memory, DATA + 0x3000), S_OK);
        let mut a = [0; 512];
        memory.read(DATA + 0x1000, &mut a).unwrap();
        assert!(a.iter().all(|&b| b == 3));
        memory.read(DATA + 0x2000, &mut a).unwrap();
        assert!(a.iter().all(|&b| b == 4));
        assert!(chip.level(0));
    }

    #[test]
    fn writes_sectors_and_flushes() {
        let (mut dev, mut driver, memory, _) = block(vec![0; 4096]);
        header(&memory, DATA, T_OUT, 1);
        memory.write(DATA + 16, &[0xab; 512]).unwrap();
        // Header and data in one buffer, as some drivers lay it out.
        driver.offer(&[(DATA, 16 + 512, false), (DATA + 0x3000, 1, true)]);
        header(&memory, DATA + 0x4000, T_FLUSH, 0);
        driver.offer(&[(DATA + 0x4000, 16, false), (DATA + 0x3001, 1, true)]);
        dev.notify(0);
        assert_eq!(driver.take_used().unwrap().1, 1);
        assert_eq!(driver.take_used().unwrap().1, 1);
        assert_eq!(status(&memory, DATA + 0x3000), S_OK);
        assert_eq!(status(&memory, DATA + 0x3001), S_OK);
        let mut back = [0; 4];
        dev.disk.read_at(&mut back, 510).unwrap();
        assert_eq!(back, [0, 0, 0xab, 0xab]);
    }

    #[test]
    fn refuses_requests_past_the_end_or_unknown() {
        let (mut dev, mut driver, memory, _) = block(vec![0; 4096]);
        header(&memory, DATA, T_IN, 8);
        driver.offer(&[
            (DATA, 16, false),
            (DATA + 0x1000, 512, true),
            (DATA + 0x3000, 1, true),
        ]);
        header(&memory, DATA + 0x100, 99, 0);
        driver.offer(&[(DATA + 0x100, 16, false), (DATA + 0x3001, 1, true)]);
        dev.notify(0);
        assert_eq!(status(&memory, DATA + 0x3000), S_IOERR);
        assert_eq!(status(&memory, DATA + 0x3001), S_UNSUPP);
    }

    #[test]
    fn read_only_disks_refuse_writes() {
        let memory = memory();
        let disk = MemoryDisk {
            data: std::sync::Mutex::new(vec![0; 1024]),
            read_only: true,
        };
        let mut dev = Block::new(Arc::new(disk), "ro");
        assert_ne!(dev.features() & F_RO, 0);
        dev.activate(Activation {
            memory: memory.clone(),
            queues: vec![Queue::new(memory.clone(), config(16))],
            interrupt: Interrupt::new(IrqLine::new(Arc::new(LevelRecorder::default()), 0)),
            features: 0,
        });
        let mut driver = Driver::new(memory.clone(), 16);
        header(&memory, DATA, T_OUT, 0);
        driver.offer(&[
            (DATA, 16, false),
            (DATA + 0x1000, 512, false),
            (DATA + 0x3000, 1, true),
        ]);
        dev.notify(0);
        assert_eq!(status(&memory, DATA + 0x3000), S_IOERR);
    }

    #[test]
    fn returns_its_id() {
        let (mut dev, mut driver, memory, _) = block(vec![0; 512]);
        header(&memory, DATA, T_GET_ID, 0);
        driver.offer(&[
            (DATA, 16, false),
            (DATA + 0x1000, 20, true),
            (DATA + 0x3000, 1, true),
        ]);
        dev.notify(0);
        assert_eq!(driver.take_used().unwrap().1, 21);
        let mut id = [0; 20];
        memory.read(DATA + 0x1000, &mut id).unwrap();
        assert_eq!(&id[..15], b"crate-test-disk");
    }

    #[test]
    fn trims_buffers() {
        let d = |addr, len| Descriptor {
            addr,
            len,
            writable: true,
        };
        assert_eq!(
            trim(&[d(0, 10), d(100, 10)], 4, 3),
            vec![d(4, 6), d(100, 7)]
        );
        assert_eq!(
            trim(&[d(0, 16), d(100, 512), d(700, 1)], 16, 1),
            vec![d(100, 512)]
        );
        assert!(trim(&[d(0, 4)], 4, 0).is_empty());
    }
}
