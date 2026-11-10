// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A CPU backend on Apple's Hypervisor framework, for macOS: the same machines
//! and devices at native speed, and the reference the translator is tested
//! against.
//!
//! Guest code runs on the CPU itself; the framework hands back exits for what
//! the machine must do: loads and stores outside RAM (MMIO), system registers
//! it traps (the GIC's CPU interface among them), HVC and SMC (PSCI), WFI,
//! and the virtual timer firing. The framework's own GIC is not used: the
//! board's interrupt controller is crate's own, as the translator needs.
//!
//! A process using it must carry the `com.apple.security.hypervisor`
//! entitlement.

#[cfg(target_os = "macos")]
mod sys;
#[cfg(target_os = "macos")]
mod vcpu;

#[cfg(target_os = "macos")]
pub use vcpu::Hvf;
