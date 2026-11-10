// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The backend: one host thread per vCPU (the framework ties a vCPU to the
//! thread that made it), each running guest code until an exit and handling
//! the exit itself.

use std::cell::Cell;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate_core::board::{Board, CpuBackend};
use crate_core::cpu::{CpuWaker, SysReg};
use crate_core::psci::Effect;
use crate_core::run::StopReason;
use crate_core::{Error, Result, debug, warn};

use crate::sys::*;

/// PSTATE at entry: EL1 using SP_EL1, with debug, SError, IRQ and FIQ masked.
const PSTATE_EL1H_MASKED: u64 = 0x3c5;

const EC_WFX: u64 = 0x01;
const EC_HVC64: u64 = 0x16;
const EC_SMC64: u64 = 0x17;
const EC_SYSREG: u64 = 0x18;
const EC_IABT_LOWER: u64 = 0x20;
const EC_DATA_ABORT_LOWER: u64 = 0x24;

// The EL1 physical timer, which the framework traps to us.
const CNTPCT_EL0: SysReg = SysReg::new(3, 3, 14, 0, 1);
const CNTP_TVAL_EL0: SysReg = SysReg::new(3, 3, 14, 2, 0);
const CNTP_CTL_EL0: SysReg = SysReg::new(3, 3, 14, 2, 1);
const CNTP_CVAL_EL0: SysReg = SysReg::new(3, 3, 14, 2, 2);

const NO_VCPU: u64 = u64::MAX;
const STATE_OTHER: u8 = 0;
const STATE_RUNNING: u8 = 1;
const STATE_WAITING: u8 = 2;

/// OSLAR_EL1, OSLSR_EL1 and OSDLR_EL1: the OS lock and OS double lock, which
/// Linux unlocks at boot. There is no external debugger to lock out, so they
/// read as zero and ignore writes without a warning.
const QUIET_SYSREGS: [SysReg; 3] = [
    SysReg::new(2, 0, 1, 0, 4),
    SysReg::new(2, 0, 1, 1, 4),
    SysReg::new(2, 0, 1, 3, 4),
];

/// The longest a WFI sleeps on a timer deadline before looking again.
const LONGEST_SLEEP: Duration = Duration::from_secs(60);

thread_local! {
    /// The CPU this thread runs, so a CPU never kicks itself.
    static CURRENT: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Runs boards on the Hypervisor framework. One machine per process: the
/// framework allows one VM per process.
pub struct Hvf {
    _private: (),
}

impl Hvf {
    pub fn new() -> Self {
        Hvf { _private: () }
    }
}

impl Default for Hvf {
    fn default() -> Self {
        Self::new()
    }
}

fn check(ret: HvReturn, what: &str) -> Result<()> {
    if ret == HV_SUCCESS {
        Ok(())
    } else {
        Err(Error::Backend(format!("{what}: {}", describe(ret))))
    }
}

struct Slot {
    id: AtomicU64,
    state: AtomicU8,
    lock: Mutex<()>,
    cond: Condvar,
}

/// Every vCPU's handle and sleep state, shared with whoever needs to wake one.
struct Cpus {
    slots: Vec<Slot>,
}

impl CpuWaker for Cpus {
    fn wake(&self, cpu: usize) {
        if CURRENT.with(|c| c.get()) == Some(cpu) {
            return; // it looks at its interrupts before it next runs anyway
        }
        let Some(slot) = self.slots.get(cpu) else {
            return;
        };
        {
            // Taking the lock orders this wake after the sleeper's last look.
            let _guard = slot.lock.lock().unwrap();
            slot.cond.notify_all();
        }
        if slot.state.load(Ordering::Acquire) != STATE_WAITING {
            let mut id = slot.id.load(Ordering::Acquire);
            if id != NO_VCPU {
                // Leaves guest code at once, or makes the next run return at
                // once if the vCPU is outside it.
                // SAFETY: hv_vcpus_exit may be called from any thread.
                unsafe { hv_vcpus_exit(&mut id, 1) };
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Timebase {
    numer: u64,
    denom: u64,
}

impl CpuBackend for Hvf {
    fn name(&self) -> &'static str {
        "hvf"
    }

    fn run(&self, board: &Board) -> Result<StopReason> {
        let count = board.cpu_count();
        let mut max = 0u32;
        // SAFETY: writes one u32.
        if unsafe { hv_vm_get_max_vcpu_count(&mut max) } == HV_SUCCESS && count > max as usize {
            return Err(Error::InvalidArgument(format!(
                "{count} CPUs; this host allows {max}"
            )));
        }
        // SAFETY: a NULL configuration asks for the default one.
        let ret = unsafe { hv_vm_create(std::ptr::null_mut()) };
        check(ret, "creating the VM")?;
        let result = run_vm(board, count);
        for region in board.memory.regions() {
            // SAFETY: unmapping what run_vm mapped; harmless if it did not.
            unsafe { hv_vm_unmap(region.guest_base(), region.size() as usize) };
        }
        // SAFETY: every vCPU was destroyed by its thread before run_vm returned.
        unsafe { hv_vm_destroy() };
        result
    }
}

fn run_vm(board: &Board, count: usize) -> Result<StopReason> {
    for region in board.memory.regions() {
        // SAFETY: the region's host memory outlives the VM, which is destroyed
        // before `run` returns.
        let ret = unsafe {
            hv_vm_map(
                region.host_ptr().cast(),
                region.guest_base(),
                region.size() as usize,
                HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC,
            )
        };
        check(ret, &format!("mapping RAM at {:#x}", region.guest_base()))?;
    }

    let cpus = Arc::new(Cpus {
        slots: (0..count)
            .map(|_| Slot {
                id: AtomicU64::new(NO_VCPU),
                state: AtomicU8::new(STATE_OTHER),
                lock: Mutex::new(()),
                cond: Condvar::new(),
            })
            .collect(),
    });
    board.interrupts.set_waker(cpus.clone());
    {
        let cpus = cpus.clone();
        board
            .run
            .on_stop(move || (0..count).for_each(|cpu| cpus.wake(cpu)));
    }

    let mut info = MachTimebaseInfo::default();
    // SAFETY: fills in the struct.
    unsafe { mach_timebase_info(&mut info) };
    let timebase = Timebase {
        numer: info.numer.max(1) as u64,
        denom: info.denom.max(1) as u64,
    };
    // Every CPU's virtual counter starts from zero now.
    // SAFETY: no preconditions.
    let offset = unsafe { mach_absolute_time() };

    std::thread::scope(|scope| {
        for cpu in 0..count {
            let cpus = &cpus;
            std::thread::Builder::new()
                .name(format!("crate-vcpu{cpu}"))
                .spawn_scoped(scope, move || {
                    if let Err(e) = vcpu_thread(board, cpus, cpu, offset, timebase) {
                        board.run.stop(StopReason::Error(format!("CPU {cpu}: {e}")));
                    }
                })
                .expect("spawning a vCPU thread");
        }
    });
    Ok(board.run.reason().unwrap_or(StopReason::Killed))
}

fn vcpu_thread(
    board: &Board,
    cpus: &Cpus,
    cpu: usize,
    offset: u64,
    timebase: Timebase,
) -> Result<()> {
    CURRENT.with(|c| c.set(Some(cpu)));
    let mut vcpu = Vcpu::create(board, cpus, cpu, offset, timebase)?;
    let result = vcpu.power_cycle();
    cpus.slots[cpu].id.store(NO_VCPU, Ordering::Release);
    drop(vcpu);
    result
}

enum Flow {
    Continue,
    CpuOff,
    Stop,
}

struct Vcpu<'a> {
    id: HvVcpu,
    exit: *const VcpuExit,
    board: &'a Board,
    cpus: &'a Cpus,
    cpu: usize,
    offset: u64,
    timebase: Timebase,
    reset_sctlr: u64,
    /// The framework masked the virtual timer after it fired; its interrupt is
    /// asserted until the guest has dealt with it.
    vtimer_masked: bool,
    /// The EL1 physical timer, which the framework traps to us rather than
    /// delivering itself: CNTP_CTL_EL0 (ENABLE/IMASK) and CNTP_CVAL_EL0.
    cntp_ctl: u64,
    cntp_cval: u64,
    noted: HashSet<u64>,
}

impl Drop for Vcpu<'_> {
    fn drop(&mut self) {
        // SAFETY: made on this thread by create.
        unsafe { hv_vcpu_destroy(self.id) };
    }
}

impl<'a> Vcpu<'a> {
    fn create(
        board: &'a Board,
        cpus: &'a Cpus,
        cpu: usize,
        offset: u64,
        timebase: Timebase,
    ) -> Result<Self> {
        let mut id = 0;
        let mut exit = std::ptr::null();
        // SAFETY: writes the vCPU's ID and the address of its exit record,
        // which stays valid until the vCPU is destroyed.
        let ret = unsafe { hv_vcpu_create(&mut id, &mut exit, std::ptr::null_mut()) };
        check(ret, "creating a vCPU")?;
        cpus.slots[cpu].id.store(id, Ordering::Release);
        let mut vcpu = Vcpu {
            id,
            exit,
            board,
            cpus,
            cpu,
            offset,
            timebase,
            reset_sctlr: 0,
            vtimer_masked: false,
            cntp_ctl: 0,
            cntp_cval: 0,
            noted: HashSet::new(),
        };
        vcpu.set_sys(HV_SYS_REG_MPIDR_EL1, board.mpidrs[cpu])?;
        // SAFETY: this thread's vCPU.
        let ret = unsafe { hv_vcpu_set_vtimer_offset(id, offset) };
        check(ret, "setting the timer offset")?;
        vcpu.reset_sctlr = vcpu.sys(HV_SYS_REG_SCTLR_EL1)?;
        Ok(vcpu)
    }

    fn reg(&self, reg: u32) -> Result<u64> {
        let mut value = 0;
        // SAFETY: this thread's vCPU; writes one u64.
        let ret = unsafe { hv_vcpu_get_reg(self.id, reg, &mut value) };
        check(ret, "reading a register")?;
        Ok(value)
    }

    fn set_reg(&self, reg: u32, value: u64) -> Result<()> {
        // SAFETY: this thread's vCPU.
        check(
            unsafe { hv_vcpu_set_reg(self.id, reg, value) },
            "writing a register",
        )
    }

    fn sys(&self, reg: u16) -> Result<u64> {
        let mut value = 0;
        // SAFETY: this thread's vCPU; writes one u64.
        let ret = unsafe { hv_vcpu_get_sys_reg(self.id, reg, &mut value) };
        check(ret, "reading a system register")?;
        Ok(value)
    }

    fn set_sys(&self, reg: u16, value: u64) -> Result<()> {
        // SAFETY: this thread's vCPU.
        check(
            unsafe { hv_vcpu_set_sys_reg(self.id, reg, value) },
            "writing a system register",
        )
    }

    fn advance_pc(&self) -> Result<()> {
        self.set_reg(HV_REG_PC, self.reg(HV_REG_PC)? + 4)
    }

    /// Starts the CPU each time it is turned on, until the machine stops.
    fn power_cycle(&mut self) -> Result<()> {
        let mut start = (self.cpu == 0).then_some((self.board.boot.pc, self.board.boot.x));
        loop {
            let (pc, x) = match start.take() {
                Some(s) => s,
                None => match self.board.psci.wait_for_on(self.cpu) {
                    Some(entry) => (entry.pc, [entry.context, 0, 0, 0]),
                    None => return Ok(()),
                },
            };
            debug!("CPU {} starts at {pc:#x}", self.cpu);
            self.set_sys(HV_SYS_REG_SCTLR_EL1, self.reset_sctlr)?;
            for (n, value) in x.iter().enumerate() {
                self.set_reg(n as u32, *value)?;
            }
            self.set_reg(HV_REG_PC, pc)?;
            self.set_reg(HV_REG_CPSR, PSTATE_EL1H_MASKED)?;
            match self.run()? {
                Flow::CpuOff => {
                    if self.vtimer_masked {
                        self.release_vtimer()?;
                    }
                    debug!("CPU {} off", self.cpu);
                }
                Flow::Stop | Flow::Continue => return Ok(()),
            }
        }
    }

    fn run(&mut self) -> Result<Flow> {
        let slot = &self.cpus.slots[self.cpu];
        loop {
            if self.board.run.is_stopped() {
                return Ok(Flow::Stop);
            }
            if self.vtimer_masked && !self.vtimer_asserted()? {
                self.release_vtimer()?;
            }
            // Drive the physical timer's PPI before deciding what is pending, so
            // a WFI that the timer should end delivers its interrupt.
            self.service_ptimer();
            let interrupts = &self.board.interrupts;
            let (irq, fiq) = (
                interrupts.irq_pending(self.cpu),
                interrupts.fiq_pending(self.cpu),
            );
            // SAFETY: this thread's vCPU; the levels apply to the next run only.
            unsafe {
                hv_vcpu_set_pending_interrupt(self.id, HV_INTERRUPT_TYPE_IRQ, irq);
                hv_vcpu_set_pending_interrupt(self.id, HV_INTERRUPT_TYPE_FIQ, fiq);
            }
            slot.state.store(STATE_RUNNING, Ordering::Release);
            // SAFETY: this thread's vCPU.
            let ret = unsafe { hv_vcpu_run(self.id) };
            slot.state.store(STATE_OTHER, Ordering::Release);
            check(ret, "running the vCPU")?;
            // SAFETY: the exit record is valid while the vCPU exists and
            // describes the run that just returned.
            let exit = unsafe { std::ptr::read(self.exit) };
            // Refresh the physical timer after running, so a GIC read in the
            // exit that follows sees its line at the current count.
            self.service_ptimer();
            let flow = match exit.reason {
                HV_EXIT_REASON_CANCELED => Flow::Continue,
                HV_EXIT_REASON_VTIMER_ACTIVATED => {
                    self.board
                        .lines
                        .set_ppi(self.cpu, self.board.vtimer_ppi, true);
                    self.vtimer_masked = true;
                    Flow::Continue
                }
                HV_EXIT_REASON_EXCEPTION => self.exception(&exit.exception)?,
                other => return Err(Error::Backend(format!("unknown exit reason {other}"))),
            };
            match flow {
                Flow::Continue => {}
                other => return Ok(other),
            }
        }
    }

    fn exception(&mut self, e: &ExitException) -> Result<Flow> {
        let esr = e.syndrome;
        let iss = (esr & 0x1ff_ffff) as u32;
        match esr >> 26 & 0x3f {
            EC_WFX => {
                self.advance_pc()?;
                if iss & 1 == 0 {
                    self.wait_for_interrupt()?;
                } else {
                    std::thread::yield_now(); // WFE
                }
                Ok(Flow::Continue)
            }
            // HVC returns past itself; a trapped SMC returns to itself.
            EC_HVC64 => self.psci(iss & 0xffff),
            EC_SMC64 => {
                self.advance_pc()?;
                self.psci(iss & 0xffff)
            }
            EC_SYSREG => {
                self.sysreg(iss)?;
                self.advance_pc()?;
                Ok(Flow::Continue)
            }
            EC_DATA_ABORT_LOWER => {
                self.data_abort(iss, e.physical_address)?;
                self.advance_pc()?;
                Ok(Flow::Continue)
            }
            // An instruction fetch that reached no stage-2 mapping: the guest
            // branched into unbacked memory. Hand it an external abort, as
            // hardware would for a fetch from an address nothing answers.
            EC_IABT_LOWER => {
                self.inject_instruction_abort(e.virtual_address)?;
                Ok(Flow::Continue)
            }
            ec => Err(Error::Backend(format!(
                "unhandled exception class {ec:#x} (ESR {esr:#x}) at PC {:#x}, address {:#x}, IPA {:#x}",
                self.reg(HV_REG_PC)?,
                e.virtual_address,
                e.physical_address
            ))),
        }
    }

    fn psci(&mut self, immediate: u32) -> Result<Flow> {
        if immediate != 0 {
            self.set_reg(0, crate_core::psci::NOT_SUPPORTED as u64)?;
            return Ok(Flow::Continue);
        }
        let args = [self.reg(0)?, self.reg(1)?, self.reg(2)?, self.reg(3)?];
        let (ret, effect) = self.board.psci.call(self.cpu, args);
        self.set_reg(0, ret)?;
        Ok(match effect {
            Effect::Continue => Flow::Continue,
            Effect::CpuOff => Flow::CpuOff,
            Effect::Stop => Flow::Stop,
        })
    }

    /// Emulates the trapped EL1 physical timer, returning whether `reg` was one
    /// of its registers. CNTPCT reads the guest counter; CNTP_CTL exposes a live
    /// ISTATUS; TVAL is the signed `CVAL - now` down-counter.
    fn physical_timer_reg(&mut self, reg: SysReg, rt: u32, is_read: bool) -> Result<bool> {
        let now = self.guest_ticks();
        let read = |this: &mut Self, v: u64| -> Result<()> {
            if rt != 31 {
                this.set_reg(rt, v)?;
            }
            Ok(())
        };
        match reg {
            CNTPCT_EL0 if is_read => read(self, now)?,
            CNTP_CTL_EL0 if is_read => {
                let fired = self.cntp_ctl & 1 != 0 && now >= self.cntp_cval;
                read(self, (self.cntp_ctl & 0b11) | ((fired as u64) << 2))?;
            }
            CNTP_CTL_EL0 => {
                self.cntp_ctl = (if rt == 31 { 0 } else { self.reg(rt)? }) & 0b11;
                self.service_ptimer();
            }
            CNTP_CVAL_EL0 if is_read => read(self, self.cntp_cval)?,
            CNTP_CVAL_EL0 => {
                self.cntp_cval = if rt == 31 { 0 } else { self.reg(rt)? };
                self.service_ptimer();
            }
            CNTP_TVAL_EL0 if is_read => read(self, self.cntp_cval.wrapping_sub(now) & 0xffff_ffff)?,
            CNTP_TVAL_EL0 => {
                let tval = if rt == 31 { 0 } else { self.reg(rt)? };
                self.cntp_cval = now.wrapping_add(tval as i32 as i64 as u64);
                self.service_ptimer();
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn sysreg(&mut self, iss: u32) -> Result<()> {
        let reg = SysReg::from_iss(iss);
        let rt = (iss >> 5) & 31;
        if self.physical_timer_reg(reg, rt, iss & 1 != 0)? {
            return Ok(());
        }
        let interrupts = &self.board.interrupts;
        if iss & 1 != 0 {
            let value = match interrupts.read_sysreg(self.cpu, reg) {
                Some(v) => v,
                None if QUIET_SYSREGS.contains(&reg) => 0,
                None => {
                    self.note(reg.0 as u64, || format!("reads of {reg} read as zero"));
                    0
                }
            };
            if rt != 31 {
                self.set_reg(rt, value)?;
            }
        } else {
            let value = if rt == 31 { 0 } else { self.reg(rt)? };
            if !interrupts.write_sysreg(self.cpu, reg, value) && !QUIET_SYSREGS.contains(&reg) {
                self.note(reg.0 as u64 | 1 << 16, || {
                    format!("writes to {reg} ignored ({value:#x})")
                });
            }
        }
        Ok(())
    }

    fn data_abort(&mut self, iss: u32, ipa: u64) -> Result<()> {
        if iss & (1 << 24) == 0 {
            return Err(Error::Backend(format!(
                "an access to {ipa:#x} the CPU could not describe (ISS {iss:#x}) at PC {:#x}",
                self.reg(HV_REG_PC)?
            )));
        }
        let size = 1u8 << ((iss >> 22) & 3);
        let sign_extend = iss & (1 << 21) != 0;
        let rt = (iss >> 16) & 31;
        let sixty_four = iss & (1 << 15) != 0;
        if iss & (1 << 6) != 0 {
            let value = if rt == 31 { 0 } else { self.reg(rt)? };
            if !self.board.mmio.write(ipa, size, value) {
                self.note(ipa & !0xfff, || {
                    format!("writes to {ipa:#x}, where nothing is, ignored")
                });
            }
        } else {
            let mut value = match self.board.mmio.read(ipa, size) {
                Some(v) => v,
                None => {
                    self.note(ipa & !0xfff, || {
                        format!("reads of {ipa:#x}, where nothing is, read as zero")
                    });
                    0
                }
            };
            if sign_extend && size < 8 {
                let shift = 64 - 8 * size as u32;
                value = (((value << shift) as i64) >> shift) as u64;
            }
            if !sixty_four {
                value &= 0xffff_ffff;
            }
            if rt != 31 {
                self.set_reg(rt, value)?;
            }
        }
        Ok(())
    }

    /// Redirects the guest to its own EL1 vector for a synchronous instruction
    /// abort, as the architecture would, since the framework reports the fault
    /// to us rather than delivering it. `far` is the faulting virtual address.
    fn inject_instruction_abort(&mut self, far: u64) -> Result<()> {
        let cpsr = self.reg(HV_REG_CPSR)?;
        let pc = self.reg(HV_REG_PC)?;
        let lower = (cpsr >> 2) & 3 == 0;
        // EC: instruction abort, from a lower EL (0x20) or the current one.
        let ec: u64 = if lower { 0x20 } else { 0x21 };
        // IFSC 0x10: synchronous external abort, not on a translation walk.
        self.set_sys(HV_SYS_REG_ESR_EL1, (ec << 26) | (1 << 25) | 0x10)?;
        self.set_sys(HV_SYS_REG_FAR_EL1, far)?;
        self.set_sys(HV_SYS_REG_ELR_EL1, pc)?;
        self.set_sys(HV_SYS_REG_SPSR_EL1, cpsr)?;
        let vbar = self.sys(HV_SYS_REG_VBAR_EL1)? & !0x7ff;
        // Vector: lower EL (0x400), or same EL with SP_EL0 (0x000) or SP_ELx.
        let offset = if lower {
            0x400
        } else if cpsr & 1 != 0 {
            0x200
        } else {
            0x000
        };
        self.set_reg(HV_REG_PC, vbar + offset)?;
        self.set_reg(HV_REG_CPSR, (cpsr & 0xf000_0000) | PSTATE_EL1H_MASKED)?;
        Ok(())
    }

    /// Logs something odd the guest did, once per `key`.
    fn note(&mut self, key: u64, message: impl FnOnce() -> String) {
        if self.noted.insert(key) {
            warn!("CPU {}: {}", self.cpu, message());
        }
    }

    fn guest_ticks(&self) -> u64 {
        // SAFETY: no preconditions.
        unsafe { mach_absolute_time() }.wrapping_sub(self.offset)
    }

    /// Whether the virtual timer's condition holds: enabled, not masked by
    /// the guest, and its compare value reached.
    fn vtimer_asserted(&self) -> Result<bool> {
        let ctl = self.sys(HV_SYS_REG_CNTV_CTL_EL0)?;
        if ctl & 1 == 0 || ctl & 2 != 0 {
            return Ok(false);
        }
        Ok(self.guest_ticks() >= self.sys(HV_SYS_REG_CNTV_CVAL_EL0)?)
    }

    fn release_vtimer(&mut self) -> Result<()> {
        self.board
            .lines
            .set_ppi(self.cpu, self.board.vtimer_ppi, false);
        self.vtimer_masked = false;
        // SAFETY: this thread's vCPU.
        check(
            unsafe { hv_vcpu_set_vtimer_mask(self.id, false) },
            "unmasking the timer",
        )
    }

    /// Whether the EL1 physical timer's condition holds: enabled, not masked by
    /// the guest, and its compare value reached. The framework traps CNTP_* to
    /// us instead of delivering the timer, so crate keeps the registers and
    /// drives the modelled PPI itself; CNTPCT shares the one guest counter.
    fn ptimer_asserted(&self) -> bool {
        self.cntp_ctl & 1 != 0 && self.cntp_ctl & 2 == 0 && self.guest_ticks() >= self.cntp_cval
    }

    /// Raises or lowers the physical timer's PPI from its current condition.
    fn service_ptimer(&self) {
        let asserted = self.ptimer_asserted();
        self.board
            .lines
            .set_ppi(self.cpu, self.board.ptimer_ppi, asserted);
    }

    /// `ticks` of the guest counter from now as a host deadline, capped so a
    /// WFI sleep still wakes to look again.
    fn deadline_in(&self, ticks: u64) -> Instant {
        let nanos = ticks as u128 * self.timebase.numer as u128 / self.timebase.denom as u128;
        Instant::now() + Duration::from_nanos(nanos.min(LONGEST_SLEEP.as_nanos()) as u64)
    }

    /// When either architected timer would next fire, whichever is sooner, for
    /// bounding a WFI sleep.
    fn timer_deadline(&self) -> Result<Option<Instant>> {
        let mut deadline: Option<Instant> = None;
        // The virtual timer, unless Apple has already raised and masked it.
        if !self.vtimer_masked {
            let ctl = self.sys(HV_SYS_REG_CNTV_CTL_EL0)?;
            if ctl & 1 != 0 && ctl & 2 == 0 {
                let cval = self.sys(HV_SYS_REG_CNTV_CVAL_EL0)?;
                deadline = Some(self.deadline_in(cval.saturating_sub(self.guest_ticks())));
            }
        }
        // The physical timer, which crate keeps itself.
        if self.cntp_ctl & 1 != 0 && self.cntp_ctl & 2 == 0 {
            let d = self.deadline_in(self.cntp_cval.saturating_sub(self.guest_ticks()));
            deadline = Some(deadline.map_or(d, |e| e.min(d)));
        }
        Ok(deadline)
    }

    /// WFI: sleeps until an interrupt is signalled, the timer would fire or
    /// the machine stops.
    fn wait_for_interrupt(&mut self) -> Result<()> {
        let deadline = self.timer_deadline()?;
        let slot = &self.cpus.slots[self.cpu];
        let interrupts = &self.board.interrupts;
        let mut guard = slot.lock.lock().unwrap();
        slot.state.store(STATE_WAITING, Ordering::Release);
        loop {
            if self.board.run.is_stopped()
                || interrupts.irq_pending(self.cpu)
                || interrupts.fiq_pending(self.cpu)
            {
                break;
            }
            match deadline {
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        break;
                    }
                    guard = slot.cond.wait_timeout(guard, d - now).unwrap().0;
                }
                None => guard = slot.cond.wait(guard).unwrap(),
            }
        }
        slot.state.store(STATE_OTHER, Ordering::Release);
        Ok(())
    }
}
