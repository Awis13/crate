// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Device models: the GICv3 interrupt controller, the PL011 UART, the PL031
//! real-time clock, and virtio devices on the virtio-mmio transport.
//!
//! Each device is an [`MmioDevice`](crate_core::bus::MmioDevice) at whatever
//! address a machine gives it, raising interrupts through an
//! [`IrqLine`](crate_core::irq::IrqLine). Devices know nothing of the machine
//! they are in or of the CPU backend that runs it.

pub mod gicv3;
pub mod pl011;
pub mod pl031;
pub mod virtio;
