# Design

How the pieces in [PLAN.md](PLAN.md)'s diagram meet in code, as of
milestone 1.

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
X0..X30, the guest's NZCV in the host's, the guest V0..V31 in the host's — so a
guest instruction that only moves data between registers *is* the same host
instruction. The host stack pointer does not hold the guest's; it points at the
`CpuState` block. That lets translated code spill a guest register with a plain
`STP Xn, Xm, [SP, #off]` without first needing a scratch register, and read or
set SP with `ADD Xn, SP, #0`. The guest's own SP lives in the state and every
guest access to it is rewritten; a real host stack, for the moments the engine
calls back into Rust, is kept in the state too. Keeping all 31 guest registers
resident is what lets the common instruction copy verbatim, flags included — no
reserved register, no per-instruction register allocation.

**What is rewritten.** Loads and stores (translated through the guest MMU),
branches (which end a block), the stack-pointer forms of add and subtract,
PC-relative addresses and literal loads (materialised at translation time),
system registers, barriers and cache/TLB maintenance, waits, and the
instructions that change the exception level. Everything else is copied.

**Entering and leaving.** A trampoline (`crate_jit_enter`) saves the host's
callee-saved registers, loads the guest state, makes SP the state block, and
branches to the block. A block ends by storing the guest registers back and
returning an exit code that says why (a branch, a system register, a call, an
exception, a wait). The dispatcher acts on it: it follows branches and waits
itself, emulates system registers (delegating the GIC's CPU interface to the
controller, as `hvf` does), takes exceptions to VBAR with SP banking, returns
through ERET, and drives the virtual timer from the host's counter. Memory
accesses call a Rust helper that walks the guest page tables (4 KiB, 16 KiB and
64 KiB granules; the page size comes from the guest's TCR), does the access on
RAM or routes it to a device, and raises a fault where the architecture would.

**Executable memory** comes from `crate-host`: a MAP_JIT region on a Mac, or two
mappings of the same pages (writable and executable aliases) elsewhere, which is
the shape iOS needs. The translator writes through the writable alias and
branches into the executable one, so nothing in crate-jit is macOS-only.

**Testing.** Random EL0 instruction streams run natively (in a child process,
caught at a trailing BRK through the signal machine context) and translated, and
the full register file and memory are compared, so any instruction the
translator gets wrong shows up as a divergence from the CPU itself
(`crates/crate-jit/tests/el0_random.rs`). EL1 behaviour — the MMU, exceptions,
system registers, the timer and the GIC — is checked against `hvf` as the
oracle: `tools/el1-diff.sh` runs the same bare-metal programs on both backends
and compares their output.

The inline software-TLB fast path for loads and stores, block chaining, and
running several CPUs are milestone 3; the helper-per-access memory path and the
single boot CPU here are correct but not yet fast.

## Not yet

- The C API (`include/crate.h`) is not implemented; crate-run uses the Rust
  API in crate-virt directly.
- PL031 alarms never fire.
- No block I/O thread, no virtio event-index or offloads, no PCIe.
- Loads and stores to MMIO that the CPU cannot describe (ISV clear) stop the
  machine; Linux's accessors never make them.
