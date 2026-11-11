# Design

How the pieces in [PLAN.md](PLAN.md)'s diagram meet in code, as of
milestone 3.

## The board and the CPU backend

A machine builds a `Board` (crate-core): guest RAM, an MMIO bus, the
interrupt controller, each CPU's MPIDR, where the boot CPU starts, PSCI, and
the run state. A CPU backend implements `CpuBackend::run(&Board)` and returns
when the machine stops. Neither side knows how the other is built: crate-virt
never names hvf, and crate-hvf never names a device.

What a backend needs from the interrupt controller is `CpuInterrupts`: whether
each CPU's IRQ and FIQ signals are asserted, and reads and writes of the
system registers the controller owns. The controller calls back through
`CpuWaker` when a CPU should look again; the backend decides what that means
(hvf: leave guest code with `hv_vcpus_exit`, or wake a CPU sleeping in WFI).

PSCI is one model in crate-core that every backend calls with x0 to x3. CPU_ON
hands an entry point to the CPU's thread, which waits in `Psci::wait_for_on`;
SYSTEM_OFF and SYSTEM_RESET stop the machine.

## Devices

A device is an `MmioDevice` (a register window: reads and writes of 1, 2, 4
or 8 bytes at an offset) put on the bus at an address, plus an `IrqLine` into
the interrupt controller. Devices hold their own locks, since any vCPU thread
may reach them. Adding one is: implement `MmioDevice`, insert it on the bus,
and describe it to the guest (for virt, a device-tree node). Host resources
come through traits too: `BlockBackend` for storage, `NetBackend` and
`FrameSink` for Ethernet frames, plain callbacks for console output.

Lines are levels. A device holds its line high while it wants attention;
edge-triggered inputs are the interrupt controller's concern.

## The GICv3

crate emulates the whole GICv3 itself rather than using the Hypervisor
framework's GIC (`hv_gic_create`), because the translator needs the same
model and hvf is the reference the translator is tested against.

- A single security state (GICD_CTLR.DS = 1), affinity routing always on, no
  LPIs and no ITS. virtio-mmio interrupts are wires, so nothing needs MSIs yet.
- The CPU interface is the ICC_* system registers. On hvf they trap to the
  host (exception class 0x18) and go to `CpuInterrupts::read_sysreg` and
  `write_sysreg`; the translator will call the same functions.
- Five priority bits and preemption on all of them, which is what the
  smallest binary point gives. A larger binary point is stored and otherwise
  ignored; Linux never sets one.
- One lock over the whole controller. Before every guest entry hvf asks it
  for the CPU's IRQ level, which scans the private interrupts and the SPIs
  routed to that CPU; with 96 interrupts that is cheap next to an exit, and a
  cached per-CPU summary is the obvious step when it is not.

## Timers on hvf

The virtual timer runs in hardware. When it fires the framework exits with
`VTIMER_ACTIVATED` and masks it; crate raises PPI 11 (INTID 27) and, before
each later entry, checks the timer's condition from CNTV_CTL_EL0 and
CNTV_CVAL_EL0 against the counter. Once the guest has re-armed or masked it,
the line drops and the timer is unmasked. A WFI sleeps on a condition variable
until an interrupt is signalled or the timer's deadline, so an idle guest
costs nothing on the host.

## virtio

Devices (`VirtioDevice`: block, net, console) are split from their transport.
The transport is virtio-mmio, version 2: no PCI configuration space, BARs or
MSI to emulate, and Linux finds each device from one device-tree node. Debian's
kernel has virtio_mmio as a module, which its initramfs loads; PCIe can come
later as a second transport over the same devices, when a GPU's BARs or a
guest that only knows PCI needs it.

Queues run where the work starts: on the vCPU that notified the queue (block
requests, transmitted frames, console output), or on a backend's thread for
traffic from outside (received frames, console input). Everything in a queue
is guest-controlled and checked: indices, lengths, chain loops, indirect
tables.

Block requests read and write guest memory in place with `pread`/`pwrite` on
the vCPU that submitted them. Network frames go to libslirp, which runs on its
own thread because it is not thread-safe; frames the guest has posted no
buffers for wait in a small backlog. No offloads are offered yet, so TCP
moves about 20 MB/s.

## The virt memory map

    0x0800_0000  GICv3 distributor, 64 KiB
    0x0810_0000  GICv3 redistributors, 128 KiB per CPU, up to 32 CPUs
    0x0900_0000  PL011 UART                       SPI 1
    0x0901_0000  PL031 RTC                        SPI 2
    0x0a00_0000  virtio-mmio, 0x200 apart, 32     SPI 16 onwards
    0x4000_0000  RAM, up to the end of a 36-bit address space

The kernel goes at the start of RAM, the device tree in the last 2 MiB and the
initrd just below it. The boot CPU starts at EL1 with the MMU off, interrupts
masked and the device tree's address in x0.

## The translator (jit backend)

The `jit` backend (crate-jit) runs guest code on the same board and devices as
`hvf`, for hosts with no hypervisor. It translates a block at a time: most
instructions are copied into the host block unchanged, and only the ones that
touch the machine are rewritten.

**The register convention.** While translated code runs, the guest's registers
live in the host's registers of the same number — guest X0..X30 in host
X0..X30, the guest's NZCV, FPCR and FPSR in the host's, the guest V0..V31 in the
host's — so a guest instruction that only moves data between registers *is*
the same host instruction. The exception is X18, the platform register: the
host OS reserves it and clobbers it (macOS zeroes it on preemption), so guest
X18 lives in the state and every instruction that names it, data processing,
SIMD transfers and memory alike, runs against a borrowed register. The host
stack pointer does not hold the guest's; it points at the `CpuState` block, so
translated code spills a register with a plain `STP Xn, Xm, [SP, #off]` and
reaches every field with an immediate offset. The guest's own SP lives in the
state too. FPCR and FPSR stay in the host's registers between runs as well as
during them: the dispatcher does no floating point, and reading or writing
them stalls the host.

**Loads and stores** take an inline fast path. The guest address is computed
into a borrowed host register (four are parked in scratch slots of the state),
the page of its last byte is compared with the software-TLB entry for its
first (direct-mapped, 8K entries of 4 KiB pages per exception level and kind of
access), and on a hit the host does the same access — same size, sign,
ordering, exclusivity — at the host address. A miss, a device, an access that
crosses a page or a misaligned exclusive goes out of line to a stub shared by
the code region, which saves what a call into Rust may clobber (X0..X17, the
flags and the vectors, plus the instruction's own operands among X19..X29) and
calls the slow path. The slow path performs the access on the state's copies of
the registers and fills the TLB entry. Exclusives are a monitor (address and
value) set by the load and a compare-and-swap at the store, CAS and the LSE
atomics are the host's own instructions on the host address, and the slow
path uses host atomics for all of them, so both paths are correct across
CPUs. An access to a physical address no device claims reads as zero and
ignores writes, as on hvf.

**One translation per physical page.** A block never crosses a 4 KiB page and
is found by the physical address of its first instruction and its mode (EL,
and single-instruction for debug). Every guest address it computes — ADR and
ADRP, literal addresses, link values, branch targets, the PC an exit records —
is the block's run-time page (`page_va`, set by whoever enters it from
outside its page) plus an offset. So a library is translated once however each
process's layout places it; spawning a process does not retranslate the C
library. The code cache keeps, per physical page, the blocks made from it, so
IC IVAU drops exactly those.

**Branches.** A direct branch to the same page ends in a branch the dispatcher
patches to the target block (chaining); since chains never leave a page, the
blocks of an invalidated page take every chain into them along. Other
branches — indirect ones, returns, and direct ones to another page — look the
target up in a jump cache (64K entries per EL, the PC's high bits folded into
the index), a shared stub per EL; a hit branches straight into the block. EL0
entries carry a per-CPU context number for the address space (ASID and TTBR0)
they were filled in, so a context switch keeps them, and a TLBI of an address
space retires its numbers instead of emptying the cache. Backward chained
branches, jump-cache lookups and ISB test the CPU's exit request, so no loop of
translated code keeps a CPU from its dispatcher.

**System registers** that are plain state (TPIDR, SP_EL0, ELR, SPSR, ESR, FAR,
VBAR, PAR, CPACR, CONTEXTIDR), the host's own (NZCV, FPCR, FPSR, and the
counter, which guest and host share), constants (CTR, DCZID, MIDR, MPIDR,
CurrentEL) and reads of the translation controls (SCTLR, TTBRx, TCR, MAIR) are
inline; so are DAIF reads and writes, which leave for the dispatcher only when
they unmask an interrupt that may be pending. The kernel reads TCR_EL1 about a
million times per CPU in a compile. Everything else exits to the dispatcher,
which emulates it as before (delegating the GIC's CPU interface to the
controller), takes exceptions to VBAR with SP banking, and returns through
ERET.

**CPUs.** Each guest CPU runs on its own host thread with its own code cache,
software TLBs and jump caches, which only that thread touches. PSCI CPU_ON
hands a secondary its entry; CPU_OFF returns it to waiting. Another thread asks
a CPU back to its dispatcher with its exit request (a kick), and wakes it if it
sleeps in WFI on its condition variable: the interrupt controller's waker does
this for interrupts and SGIs, a per-machine alarm thread for timer deadlines,
and the stop hook for the machine stopping. WFE yields.

**Maintenance across CPUs.** A CPU that invalidates something for the inner
shareable domain applies it to itself and posts it to the others' queues,
which each applies in its dispatcher before it runs guest code again. A TLB
invalidation must be finished everywhere before the issuer goes on, so every
CPU counts its entries to and exits from guest code in a generation word, odd
while inside; the issuer posts, then reads each generation, and kicks and
waits for any CPU that is inside (a sequentially consistent fence on each
side makes one of them see the other). It waits only for CPUs running the
address space a TLBI by ASID names, but tells them all, since any may hold its
code. An instruction-cache invalidation neither kicks nor waits: another CPU is
only promised the new code after its own context synchronization, so it is
asked to look at its queue at its next ISB; and a page no CPU ever translated
code from (a shared bitmap) needs no message at all, which is the usual case
of a JIT writing new code.

**Entering and leaving.** A trampoline (`crate_jit_enter`) saves the host's
callee-saved registers, loads the guest state, makes SP the state block, and
jumps to the block. A block ends by storing the guest registers back and
returning an exit code that says why (a branch the jump cache missed, a system
register, a call, an exception, a wait).

**Executable memory** comes from `crate-host`: two mappings of the same pages,
one writable and one executable, so translating and patching never switch a
thread's W^X state. This is also the shape iOS needs. Nothing in crate-jit is
macOS-only.

**The fast paths can be turned off** (`CRATE_JIT_SLOW=1`): every load and store
then calls the slow path, every branch returns to the dispatcher, and system
registers all exit — the plain translation, to tell a fast-path bug from any
other. `CRATE_JIT_STATS=1` counts exits by kind and by system register,
jump-cache lookups and misses, slow-path calls by reason and translations,
printed when a CPU stops.

**The CPU it offers.** An Armv8.0 CPU with the LSE atomics (which the kernel
patches its atomics to at boot), AES/PMULL, SHA1, SHA256, CRC32, RDM, the dot
product, LRCPC, JSCVT and FCMA (all of which the host runs as copies), 4, 16 and
64 KiB granules without LPA2, self-hosted debug, and no PAN, PAC, BTI, MTE or
SVE.

**Testing.** Random EL0 instruction streams — with forward branches, loops,
calls and indirect jumps, across a page boundary, and every load/store form
the fast path rewrites — run natively (in a child process, caught at a
trailing BRK through the signal machine context) and translated with the fast
paths on and off, and the full register file and memory are compared
(`crates/crate-jit/tests/el0_random.rs`). Two variants name X18 (natively it is
X29) and run one translation through the MMU at two virtual addresses. The
arm64 kvm-unit-tests run on jit at 4 and 16 KiB and on hvf, single- and
multi-CPU (`tools/kvm-unit-tests.py`); `tools/el1-diff.sh` compares small EL1
programs on jit and hvf.

## Not yet

- The C API (`include/crate.h`) is not implemented; crate-run uses the Rust
  API in crate-virt directly.
- The translator keeps no translations across launches and shares none
  between CPUs; each CPU has a 256 MiB code region. The guest's 16 KiB pages
  still take four 4 KiB TLB entries.
- PL031 alarms never fire.
- No block I/O thread, no virtio event-index or offloads, no PCIe.
- Loads and stores to MMIO that the CPU cannot describe (ISV clear) stop the
  machine; Linux's accessors never make them.
