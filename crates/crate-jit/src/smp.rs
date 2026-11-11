// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! What the CPUs of one machine share: how one asks another to come back to
//! its dispatcher, how one sleeps until an interrupt, the timer alarm that
//! wakes a CPU when its timer fires, and the TLB and code invalidations one
//! CPU broadcasts to the others.
//!
//! Each CPU runs on its own host thread, with its own translations, software
//! TLBs and jump caches, which only that thread touches. Another CPU never
//! changes them directly: it posts the operation to the CPU's queue and kicks
//! it (sets its exit request, and wakes it if it sleeps), and the CPU applies
//! the queue in its dispatcher before it runs guest code again.
//!
//! A TLB invalidation must be complete everywhere before the CPU that issued
//! it goes on -- the guest frees or reuses the page next. Each CPU counts its
//! entries into and exits from guest code in a generation word, odd while it
//! runs guest code. The issuer posts the operation, then reads every other
//! CPU's generation: one that is even is outside guest code and will apply
//! the queue before it enters again; one that is odd may be using the old
//! translation, so the issuer kicks it and waits for its generation to move.
//! The issuer's post and its read, and a CPU's increment and its look at its
//! queue, are each ordered by a sequentially consistent fence, so between the
//! two one of them always sees the other.
//!
//! Instruction-cache invalidations neither wait nor kick: the architecture
//! only promises another CPU the new code after its own context
//! synchronization (an ISB or an exception), so the CPU is only asked to look
//! at its queue at its next ISB (`sync_request`) or dispatcher entry. And a
//! page no CPU ever translated code from (a shared bitmap says) needs no
//! message at all, which is the common case of a JIT writing new code.
//!
//! A TLB invalidation of one address space (by ASID) skips the CPUs running
//! another: a CPU drops its lower-half translations whenever it changes
//! TTBR0, so it only ever holds those of the address space it runs, which it
//! publishes here before it fills any.

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering, fence};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use crate_core::cpu::CpuWaker;
use crate_core::run::RunControl;

use crate::state::CpuState;

/// An operation one CPU asks the others to apply to their own caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Remote {
    /// TLBI VMALLE1(IS): every translation.
    FlushAll,
    /// TLBI VAE1/VAAE1/VALE1/VAALE1(IS): translations of this address, of
    /// one address space (the ASID) or of all.
    FlushVa(u64, Option<u16>),
    /// TLBI ASIDE1(IS): every non-global translation of an address space.
    FlushUser(u16),
    /// IC IVAU: code translated from this 4 KiB physical page.
    CodePage(u64),
    /// IC IALLUIS: every translation of code.
    CodeAll,
}

impl Remote {
    /// The address space the operation is limited to, if it is.
    fn asid(&self) -> Option<u16> {
        match *self {
            Remote::FlushVa(va, asid) if va >> 55 & 1 == 0 => asid,
            Remote::FlushUser(asid) => Some(asid),
            _ => None,
        }
    }
}

/// What other CPUs asked one CPU to do.
#[derive(Default)]
struct Queue {
    tlb: Vec<Remote>,
    tlb_all: bool,
    /// Code pages to drop, without repeats.
    code: Vec<u64>,
    code_all: bool,
}

/// The CPU's address space is not known (it has not published one).
const NO_ASID: u64 = u64::MAX;

/// Pages of the bitmap of physical pages any CPU translated code from.
const CODE_BITS: usize = 1 << 16;

/// One CPU's shared slot.
pub struct Slot {
    state: AtomicPtr<CpuState>,
    /// Odd while the CPU runs guest code.
    generation: AtomicU64,
    sleep: Mutex<()>,
    wake: Condvar,
    queue: Mutex<Queue>,
    pending: AtomicBool,
    /// The ASID of the address space the CPU runs, or NO_ASID.
    asid: AtomicU64,
}

/// The CPUs of one machine.
pub struct Smp {
    slots: Vec<Slot>,
    alarm: Alarm,
    run: std::sync::Arc<RunControl>,
    /// One bit per hash of a 4 KiB physical page: some CPU translated code
    /// from a page with this hash. Never cleared (a false "maybe" only costs
    /// a message).
    code_pages: Vec<AtomicU64>,
}

impl Smp {
    pub fn new(cpus: usize, run: std::sync::Arc<RunControl>) -> Self {
        Smp {
            slots: (0..cpus)
                .map(|_| Slot {
                    state: AtomicPtr::new(std::ptr::null_mut()),
                    generation: AtomicU64::new(0),
                    sleep: Mutex::new(()),
                    wake: Condvar::new(),
                    queue: Mutex::new(Queue::default()),
                    pending: AtomicBool::new(false),
                    asid: AtomicU64::new(NO_ASID),
                })
                .collect(),
            alarm: Alarm::new(cpus),
            run,
            code_pages: (0..CODE_BITS / 64).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    fn code_bit(pa: u64) -> (usize, u64) {
        let h = ((pa >> 12).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 48) as usize % CODE_BITS;
        (h / 64, 1 << (h % 64))
    }

    /// Notes that some CPU translated code from the page holding `pa`.
    pub fn note_code(&self, pa: u64) {
        let (w, b) = Self::code_bit(pa);
        if self.code_pages[w].load(Ordering::Relaxed) & b == 0 {
            self.code_pages[w].fetch_or(b, Ordering::SeqCst);
        }
    }

    /// Whether any CPU may hold code translated from the page holding `pa`.
    pub fn maybe_code(&self, pa: u64) -> bool {
        let (w, b) = Self::code_bit(pa);
        self.code_pages[w].load(Ordering::SeqCst) & b != 0
    }

    /// Publishes the ASID `cpu` now runs (before it fills any translation of
    /// that address space).
    pub fn set_asid(&self, cpu: usize, asid: Option<u16>) {
        self.slots[cpu]
            .asid
            .store(asid.map_or(NO_ASID, |a| a as u64), Ordering::SeqCst);
    }

    pub fn cpus(&self) -> usize {
        self.slots.len()
    }

    /// Registers CPU `cpu`'s state block, which must stay put while the
    /// machine runs.
    pub fn attach(&self, cpu: usize, state: *mut CpuState) {
        self.slots[cpu].state.store(state, Ordering::Release);
    }

    fn state(&self, cpu: usize) -> Option<&CpuState> {
        let p = self.slots.get(cpu)?.state.load(Ordering::Acquire);
        // SAFETY: attached states outlive the machine's run, and the fields
        // other threads touch are atomics.
        unsafe { p.as_ref() }
    }

    /// Asks `cpu` back to its dispatcher, waking it if it sleeps.
    pub fn kick(&self, cpu: usize) {
        if let Some(s) = self.state(cpu) {
            s.kick();
        }
        let slot = &self.slots[cpu];
        let _guard = slot.sleep.lock().unwrap();
        slot.wake.notify_all();
    }

    /// Sets every CPU's exit request and nothing else (safe from a signal
    /// handler, unlike `kick`, which takes a lock).
    pub fn kick_requests(&self) {
        for cpu in 0..self.slots.len() {
            if let Some(s) = self.state(cpu) {
                s.kick();
            }
        }
    }

    pub fn kick_all(&self) {
        for cpu in 0..self.slots.len() {
            self.kick(cpu);
        }
    }

    /// Sleeps until kicked, or until `deadline` (in counter ticks), or a while.
    /// Returns at once if a kick is already pending.
    pub fn sleep(&self, cpu: usize, deadline: Option<u64>) {
        let slot = &self.slots[cpu];
        let guard = slot.sleep.lock().unwrap();
        if let Some(s) = self.state(cpu) {
            if s.exit_request().load(Ordering::Acquire) != 0
                || s.irq_hint().load(Ordering::Acquire) != 0
                || self.run.is_stopped()
            {
                return;
            }
        }
        let wait = match deadline {
            Some(d) => ticks_until(d).min(Duration::from_millis(100)),
            None => Duration::from_millis(100),
        };
        if !wait.is_zero() {
            let _unused = slot.wake.wait_timeout(guard, wait).unwrap();
        }
    }

    /// The CPU is about to run guest code. Returns false if remote work
    /// arrived that it must apply first.
    pub fn enter(&self, cpu: usize) -> bool {
        let slot = &self.slots[cpu];
        slot.generation.fetch_add(1, Ordering::Relaxed);
        fence(Ordering::SeqCst);
        if slot.pending.load(Ordering::Relaxed) {
            slot.generation.fetch_add(1, Ordering::Release);
            return false;
        }
        true
    }

    /// The CPU has left guest code.
    pub fn leave(&self, cpu: usize) {
        self.slots[cpu].generation.fetch_add(1, Ordering::Release);
    }

    /// The operations posted to `cpu`: TLB ones first, then code ones.
    pub fn take(&self, cpu: usize) -> Vec<Remote> {
        let slot = &self.slots[cpu];
        if !slot.pending.load(Ordering::Acquire) {
            return Vec::new();
        }
        let mut q = slot.queue.lock().unwrap();
        slot.pending.store(false, Ordering::Release);
        let mut out = if q.tlb_all {
            vec![Remote::FlushAll]
        } else {
            std::mem::take(&mut q.tlb)
        };
        if q.code_all {
            out.push(Remote::CodeAll);
        } else {
            out.extend(q.code.drain(..).map(Remote::CodePage));
        }
        q.tlb.clear();
        q.code.clear();
        q.tlb_all = false;
        q.code_all = false;
        out
    }

    /// Posts `op` to every CPU but `from` that may hold what it invalidates.
    /// A TLB operation returns once none of them can still be using the old
    /// translation; a code one only asks them to look at their next ISB.
    pub fn broadcast(&self, from: usize, op: Remote) {
        let code = matches!(op, Remote::CodePage(_) | Remote::CodeAll);
        if let Remote::CodePage(pa) = op {
            if !self.maybe_code(pa) {
                return;
            }
        }
        let asid = op.asid();
        let mut posted = [false; 64];
        for (cpu, slot) in self.slots.iter().enumerate() {
            if cpu == from {
                continue;
            }
            if let Some(a) = asid {
                let theirs = slot.asid.load(Ordering::SeqCst);
                if theirs != NO_ASID && theirs != a as u64 {
                    continue;
                }
            }
            let mut q = slot.queue.lock().unwrap();
            match op {
                Remote::CodePage(pa) => {
                    if !q.code_all && !q.code.contains(&pa) {
                        if q.code.len() >= 1024 {
                            q.code_all = true;
                            q.code.clear();
                        } else {
                            q.code.push(pa);
                        }
                    }
                }
                Remote::CodeAll => q.code_all = true,
                _ if q.tlb_all => {}
                Remote::FlushAll => q.tlb_all = true,
                _ => {
                    if q.tlb.len() >= 64 {
                        q.tlb_all = true;
                        q.tlb.clear();
                    } else if !q.tlb.contains(&op) {
                        q.tlb.push(op);
                    }
                }
            }
            drop(q);
            slot.pending.store(true, Ordering::Release);
            posted[cpu.min(63)] = true;
        }
        fence(Ordering::SeqCst);
        if code {
            for (cpu, _) in self.slots.iter().enumerate() {
                if posted[cpu.min(63)] {
                    if let Some(s) = self.state(cpu) {
                        s.sync_request().store(1, Ordering::Release);
                    }
                }
            }
            return;
        }
        let mut running = Vec::new();
        for (cpu, slot) in self.slots.iter().enumerate() {
            if !posted[cpu.min(63)] {
                continue;
            }
            let g = slot.generation.load(Ordering::SeqCst);
            if g & 1 == 1 {
                running.push((cpu, g));
                self.kick(cpu);
            }
        }
        for (cpu, g) in running {
            let slot = &self.slots[cpu];
            let mut spins = 0u32;
            while slot.generation.load(Ordering::Acquire) == g && !self.run.is_stopped() {
                spins += 1;
                if spins < 64 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
        }
    }

    pub fn alarm(&self) -> &Alarm {
        &self.alarm
    }

    /// The alarm thread's loop: kicks each CPU when its timer deadline
    /// passes, until the machine stops.
    pub fn run_alarm(&self) {
        let mut state = self.alarm.deadlines.lock().unwrap();
        while !self.run.is_stopped() {
            let now = crate_host::jit::counter::now();
            let mut next: Option<u64> = None;
            for cpu in 0..state.len() {
                if let Some(d) = state[cpu] {
                    if d <= now {
                        state[cpu] = None;
                        if let Some(s) = self.state(cpu) {
                            s.kick();
                        }
                        let slot = &self.slots[cpu];
                        let _g = slot.sleep.lock().unwrap();
                        slot.wake.notify_all();
                    } else {
                        next = Some(next.map_or(d, |n: u64| n.min(d)));
                    }
                }
            }
            let wait = next.map_or(Duration::from_millis(50), |d| {
                ticks_until(d).clamp(Duration::from_micros(20), Duration::from_millis(50))
            });
            state = self.alarm.cond.wait_timeout(state, wait).unwrap().0;
        }
    }

    /// Wakes the alarm thread (when the machine stops).
    pub fn stop_alarm(&self) {
        let _g = self.alarm.deadlines.lock().unwrap();
        self.alarm.cond.notify_all();
    }
}

impl CpuWaker for Smp {
    fn wake(&self, cpu: usize) {
        if let Some(s) = self.state(cpu) {
            s.irq_hint().store(1, Ordering::Release);
        }
        self.kick(cpu);
    }
}

/// When each CPU's next timer interrupt is due, in counter ticks.
pub struct Alarm {
    deadlines: Mutex<Vec<Option<u64>>>,
    cond: Condvar,
}

impl Alarm {
    fn new(cpus: usize) -> Self {
        Alarm {
            deadlines: Mutex::new(vec![None; cpus]),
            cond: Condvar::new(),
        }
    }

    /// Sets (or clears) `cpu`'s deadline.
    pub fn set(&self, cpu: usize, deadline: Option<u64>) {
        let mut d = self.deadlines.lock().unwrap();
        if d[cpu] != deadline {
            let sooner = match (d[cpu], deadline) {
                (_, None) => false,
                (None, Some(_)) => true,
                (Some(a), Some(b)) => b < a,
            };
            d[cpu] = deadline;
            if sooner {
                self.cond.notify_all();
            }
        }
    }
}

/// How long until the counter reaches `deadline`.
pub fn ticks_until(deadline: u64) -> Duration {
    let now = crate_host::jit::counter::now();
    if deadline <= now {
        return Duration::ZERO;
    }
    let freq = crate_host::jit::counter::frequency().max(1) as u128;
    let nanos = (deadline - now) as u128 * 1_000_000_000 / freq;
    Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
}

/// When the counter reaches `deadline`, as an `Instant`.
pub fn instant_of(deadline: u64) -> Instant {
    Instant::now() + ticks_until(deadline)
}
