// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The parts of Hypervisor.framework's arm64 interface crate uses, declared
//! from the SDK's headers (hv_vm.h, hv_vcpu.h, hv_vcpu_types.h).

use std::ffi::c_void;

pub type HvReturn = i32;
pub type HvVcpu = u64;

pub const HV_SUCCESS: HvReturn = 0;

pub const HV_MEMORY_READ: u64 = 1 << 0;
pub const HV_MEMORY_WRITE: u64 = 1 << 1;
pub const HV_MEMORY_EXEC: u64 = 1 << 2;

pub const HV_EXIT_REASON_CANCELED: u32 = 0;
pub const HV_EXIT_REASON_EXCEPTION: u32 = 1;
pub const HV_EXIT_REASON_VTIMER_ACTIVATED: u32 = 2;

pub const HV_INTERRUPT_TYPE_IRQ: u32 = 0;
pub const HV_INTERRUPT_TYPE_FIQ: u32 = 1;

/// hv_reg_t: X0 to X30 are 0 to 30.
pub const HV_REG_PC: u32 = 31;
pub const HV_REG_CPSR: u32 = 34;

/// hv_sys_reg_t uses the same packing as crate_core::cpu::SysReg.
pub const HV_SYS_REG_MPIDR_EL1: u16 = 0xc005;
pub const HV_SYS_REG_SCTLR_EL1: u16 = 0xc080;
pub const HV_SYS_REG_SPSR_EL1: u16 = 0xc200;
pub const HV_SYS_REG_ELR_EL1: u16 = 0xc201;
pub const HV_SYS_REG_ESR_EL1: u16 = 0xc290;
pub const HV_SYS_REG_FAR_EL1: u16 = 0xc300;
pub const HV_SYS_REG_VBAR_EL1: u16 = 0xc600;
pub const HV_SYS_REG_CNTV_CTL_EL0: u16 = 0xdf19;
pub const HV_SYS_REG_CNTV_CVAL_EL0: u16 = 0xdf1a;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ExitException {
    pub syndrome: u64,
    pub virtual_address: u64,
    pub physical_address: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VcpuExit {
    pub reason: u32,
    pub exception: ExitException,
}

#[link(name = "Hypervisor", kind = "framework")]
unsafe extern "C" {
    pub fn hv_vm_create(config: *mut c_void) -> HvReturn;
    pub fn hv_vm_destroy() -> HvReturn;
    pub fn hv_vm_map(addr: *mut c_void, ipa: u64, size: usize, flags: u64) -> HvReturn;
    pub fn hv_vm_unmap(ipa: u64, size: usize) -> HvReturn;
    pub fn hv_vm_get_max_vcpu_count(count: *mut u32) -> HvReturn;

    pub fn hv_vcpu_create(
        vcpu: *mut HvVcpu,
        exit: *mut *const VcpuExit,
        config: *mut c_void,
    ) -> HvReturn;
    pub fn hv_vcpu_destroy(vcpu: HvVcpu) -> HvReturn;
    pub fn hv_vcpu_get_reg(vcpu: HvVcpu, reg: u32, value: *mut u64) -> HvReturn;
    pub fn hv_vcpu_set_reg(vcpu: HvVcpu, reg: u32, value: u64) -> HvReturn;
    pub fn hv_vcpu_get_sys_reg(vcpu: HvVcpu, reg: u16, value: *mut u64) -> HvReturn;
    pub fn hv_vcpu_set_sys_reg(vcpu: HvVcpu, reg: u16, value: u64) -> HvReturn;
    pub fn hv_vcpu_set_pending_interrupt(vcpu: HvVcpu, kind: u32, pending: bool) -> HvReturn;
    pub fn hv_vcpu_run(vcpu: HvVcpu) -> HvReturn;
    pub fn hv_vcpus_exit(vcpus: *mut HvVcpu, count: u32) -> HvReturn;
    pub fn hv_vcpu_set_vtimer_mask(vcpu: HvVcpu, masked: bool) -> HvReturn;
    pub fn hv_vcpu_set_vtimer_offset(vcpu: HvVcpu, offset: u64) -> HvReturn;
}

#[repr(C)]
#[derive(Default)]
pub struct MachTimebaseInfo {
    pub numer: u32,
    pub denom: u32,
}

unsafe extern "C" {
    pub fn mach_absolute_time() -> u64;
    pub fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
}

/// A readable name for a failure, from hv_error.h.
pub fn describe(ret: HvReturn) -> String {
    let name = match ret as u32 {
        0xfae9_4001 => "HV_ERROR",
        0xfae9_4002 => "HV_BUSY",
        0xfae9_4003 => "HV_BAD_ARGUMENT",
        0xfae9_4004 => "HV_ILLEGAL_GUEST_STATE",
        0xfae9_4005 => "HV_NO_RESOURCES",
        0xfae9_4006 => "HV_NO_DEVICE",
        0xfae9_4007 => "HV_DENIED (is the binary signed with com.apple.security.hypervisor?)",
        0xfae9_4008 => "HV_EXISTS",
        0xfae9_400f => "HV_UNSUPPORTED",
        _ => return format!("error {ret:#x}"),
    };
    name.to_string()
}
