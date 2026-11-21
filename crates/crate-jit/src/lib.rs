// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The translator, the CPU backend for hosts without a hypervisor. Guest code
//! is translated a block at a time: most instructions are copied into host
//! code as they are, and those that touch the machine (loads and stores,
//! branches, system registers, exceptions) are rewritten.

pub mod asm;
pub mod backend;
pub mod cache;
pub mod decode;
pub mod emit;
mod mem;
pub mod mmu;
mod sm3;
pub mod smp;
pub mod state;
pub mod translate;
