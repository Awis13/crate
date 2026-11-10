// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The virt machine's memory map and interrupt assignments.
//!
//! ```text
//! 0x0800_0000  GICv3 distributor, 64 KiB
//! 0x0810_0000  GICv3 redistributors, 128 KiB per CPU, up to 32 CPUs
//! 0x0900_0000  PL011 UART                       SPI 1
//! 0x0901_0000  PL031 RTC                        SPI 2
//! 0x0a00_0000  virtio-mmio, 0x200 apart, 32     SPI 16 onwards
//! 0x4000_0000  RAM
//! ```
//!
//! Everything below 1 GiB is devices, so RAM can grow without moving them.

pub const GICD_BASE: u64 = 0x0800_0000;
pub const GICR_BASE: u64 = 0x0810_0000;
pub const UART_BASE: u64 = 0x0900_0000;
pub const RTC_BASE: u64 = 0x0901_0000;
pub const VIRTIO_BASE: u64 = 0x0a00_0000;
pub const RAM_BASE: u64 = 0x4000_0000;

/// What the redistributor window up to the UART holds.
pub const MAX_CPUS: usize = 32;
pub const VIRTIO_SLOTS: usize = 32;
pub const MIN_MEMORY: u64 = 64 << 20;
/// What a 36-bit guest physical address space leaves above RAM_BASE.
pub const MAX_MEMORY: u64 = (64 << 30) - RAM_BASE;

pub const UART_SPI: u32 = 1;
pub const RTC_SPI: u32 = 2;
pub const VIRTIO_SPI_BASE: u32 = 16;
pub const SPIS: usize = 64;

/// The architected timers' private interrupts, in the order the
/// arm,armv8-timer binding lists them.
pub const SECURE_PTIMER_PPI: u32 = 13;
pub const PTIMER_PPI: u32 = 14;
pub const VTIMER_PPI: u32 = 11;
pub const HYP_TIMER_PPI: u32 = 10;

pub fn virtio_base(slot: usize) -> u64 {
    VIRTIO_BASE + 0x200 * slot as u64
}

/// CPU n is Aff0 n, with MPIDR_EL1's RES1 bit 31 set.
pub fn mpidr(cpu: u64) -> u64 {
    (1 << 31) | ((cpu / 16) << 8) | (cpu % 16)
}

const _: () = assert!(GICR_BASE + 0x2_0000 * MAX_CPUS as u64 <= UART_BASE);
const _: () = assert!(VIRTIO_SPI_BASE as usize + VIRTIO_SLOTS <= SPIS);
