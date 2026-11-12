# Design

How the pieces in [PLAN.md](PLAN.md)'s diagram meet in code, as of
milestones 3 to 5 and the translator work of milestone 6.

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
moves about 20 MB/s. Every host socket libslirp makes is set SO_NOSIGPIPE on
Apple systems as it is registered, so writing to a forwarded connection whose
other end has gone fails with EPIPE rather than raising SIGPIPE, which there
goes to the whole process: an app embedding crate need not ignore it.

## The GPU, the display and input

**virtio-gpu** (crate-devices, `virtio::gpu`) has one scanout. Every command
runs on the device's own thread, `crate-gpu`: a vCPU's notify only wakes it,
and the thread owns the 3D renderer, since a GL context is current on one
thread at a time. 2D resources are the device's: their pixels live in host
memory, filled from the guest's backing by TRANSFER_TO_HOST_2D. 3D resources
are the renderer's (`crate_core::gpu::Renderer`, virglrenderer in practice):
capability sets, contexts, 3D resources, transfers and command submission go
to it. A command with a fence is answered when the renderer retires its fence;
with a renderer every fence goes through it, because Linux takes an answered
fence as every earlier one done. Fences are polled every half millisecond while
some are outstanding, and not at all otherwise. No blob resources yet.

**Modes** are a list the app gives, preferred first: GET_DISPLAY_INFO reports
the first, and the EDID (VESA E-EDID 1.4) carries each as a detailed timing
with CVT timings at 60 Hz (reduced blanking when the CRT timing's clock is too
fast for a descriptor), three in the base block beside a monitor name and six
in each CTA-861 extension block. Linux lists exactly those modes, and the guest
changes between them itself; the app can offer new ones while it runs, which
raises a display event.

**Frames** go to a `DisplaySink` as R, G, B, A, with the cursor drawn in (from
UPDATE_CURSOR's resource; the fourth byte is read as alpha even when the
resource says X, because Linux makes every dumb buffer XRGB). A
RESOURCE_FLUSH updates the device's copy of the screen in the flushed
rectangle only, so a blinking cursor costs a few pixels; for a 3D resource that
is a GL readback. Nothing is read back while the sink is not active: the
device remembers the screen is stale and reads it whole at the next flush or
when a viewer asks (`DisplayControl::refresh`).

**virglrenderer** (crate-host, `virgl`) is loaded with dlopen when a machine
asks for 3D, as is the EGL it runs on, so crate builds and runs without them
and a GPU that cannot start is 2D. The contexts are crate's own, through
virglrenderer's callbacks: ANGLE's display on Vulkan unless told otherwise
(OpenGL ES 3.1 for the guest; Metal stops at 3.0 and the guest at 2.0),
surfaceless contexts, every one sharing with a primary context. A scanout is
read back by attaching the resource's texture to a framebuffer of the primary
context and `glReadPixels` as RGBA (OpenGL ES cannot read BGRA). Two things
virglrenderer needs from its host that its header does not say: whatever
context was current goes back after a readback and after creating a guest
context, because virglrenderer makes the new context current to set it up yet
believes the earlier one still is; and its `virgl_box` is taken to be the
virtio box (six 32-bit fields). `CRATE_GL_DEBUG=1` makes debug contexts and
logs ANGLE's own messages, which name the failing call. Once started,
virglrenderer stays for the life of the process, as the iOS code region does:
`virgl_renderer_cleanup` waits for the thread of its render server, which the
build crate uses starts even without venus and never finishes. A freed machine
resets the renderer and releases its context, and the next one takes it up on
its own GPU thread.

**virtio-input** devices are a keyboard (every key code from KEY_ESC to
KEY_MICMUTE, with EV_REP so the guest repeats held keys) and a tablet (ABS_X
and ABS_Y from 0 to 32767, BTN_LEFT, BTN_RIGHT, BTN_MIDDLE, REL_WHEEL and
REL_HWHEEL), each its own device on virtio-mmio. Events the guest has posted
no buffers for wait in a backlog of 1024.

**The VNC server** (crate-vnc) is a `DisplaySink` that keeps a copy of the
screen and, per client, the part that changed since it was last sent. RFB 3.8
(3.3 and 3.7 too), security None, Raw in any true-colour format the client
sets, DesktopSize when the screen changes size. It is active only while a
client is connected; the first client's answers wait up to half a second for
the whole frame the server asks for. Each client has a thread reading its
messages and one sending updates. Keysyms become the Linux key codes of a US
keyboard, with Shift pressed around a symbol that needs it when the client
holds none (aTerm sends capitals without Shift); RFB's wheel buttons scroll.

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
the same host instruction. The exceptions are X18, the platform register: the
host OS reserves it and clobbers it (macOS zeroes it on preemption), so guest
X18 lives in the state and every instruction that names it, data processing,
SIMD transfers and memory alike, runs against a borrowed register; and X15..X17,
whose guest values live in the state wherever one block hands over to another
(the dispatcher, a chained branch, the jump cache, the memory stub), so their
host registers are free scratch there. A block loads one into its host register
when an instruction names it and stores it back before it leaves, if it
changed it. The host
stack pointer does not hold the guest's; it points into the `CpuState` block,
at its register slots, so translated code spills a register with a plain
`STP Xn, Xm, [SP, #off]` and reaches every field with an immediate offset. The guest's own SP lives in the
state too. FPCR and FPSR stay in the host's registers between runs as well as
during them: the dispatcher does no floating point, and reading or writing
them stalls the host.

**Loads and stores** take an inline fast path on three scratch registers. The
guest address is computed into one, the software-TLB entry (direct-mapped, 32K
entries of 4 KiB pages per exception level and kind of access) for the page of
its last byte is loaded, and its tag compared with the page of its first, so an
access that crosses a page never matches; on a hit the host runs the guest's
own instruction — same size, sign, ordering, exclusivity — at the host
address: a single register's load or store adds the entry's delta to the
guest address as its register offset (`LDR Wt, [Xva, Xdelta]`), the others
run at the host address with no offset (LDP at offset zero, the no-offset
forms), and the host does any writeback itself. The TLBs lie below the
address SP holds, the one for EL0 loads right below it, so code reaches that
table's entry for an index in one instruction (`SUB Xd, SP, Xi, LSL #4`) and
the others' in two. A run of accesses off one base register with nothing
between them that changes it, branches or leaves (a prologue's stores, a
structure's fields) shares one check of the page that holds them all: the
first finds the host address of the base and keeps it in one of X15..X17, the
others use it with their own offsets, and if the check fails each goes its own
way, out of line. The scratch registers are X15..X17 when the access does
not name them; otherwise the block borrows others, preferring one the rest of
the block overwrites before reading, then one it does not name again, and puts
a borrowed register back only when an instruction names it or the block
leaves, so a block of several accesses borrows once. Everything is back in
place before anything leaves the block, so the guest's registers at a fault are
exact. A miss, a device, an access that crosses a page or a misaligned
exclusive goes out of line to a stub shared by the code region, which saves
what a call into Rust may clobber (X0..X14, the flags and the vectors, plus the
instruction's own operands among X19..X29) and calls the slow path, then
returns with RET to keep the CPU's return prediction paired with the call. The
slow path performs the access on the state's copies of the registers and fills
the TLB entry (a load of a page it may also write fills the store TLB too).
Refills were 15% of a compile's time, mostly page walks: a walk cache keeps
the descriptor one level above the pages for the 64 most recent 2 MiB
regions, so most walks read one descriptor instead of four. It holds only
valid descriptors, which the guest changes only by breaking them and
invalidating their TLB entries, so it is emptied with the software TLBs.
Exclusives are a monitor (address and value) set by the load and a
compare-and-swap at the store, CAS and the LSE atomics are the host's own
instructions on the host address, and the slow path uses host atomics for all
of them, so both paths are correct across CPUs. An access to a physical address no device claims reads as zero and
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
the index), in a stub per EL; a hit branches straight into the block. Each EL
has 32 copies of the stub, and a site branches to the one its guest address
picks, so the CPU predicts each copy's indirect branch from the few sites
that share it instead of one branch from all of them (a cold start of node
took 8% fewer cycles).
Calls (BL, BLR) do not end a block: a call is a host BL to a tail that sets the
link and goes to the target, and the code the call returns to is translated
right after the BL, as an entry of the block of its own (the block for that
PC from then on). A return's lookup, in a second stub per EL, jumps to the
entry it finds with RET, and the CPU's return stack, filled by the host BL,
predicts it; a return was 38% of a compile's lookups, and through one shared
indirect branch it was mostly mispredicted. EL0
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
code. A CPU's EL0 TLBs hold only the address space it runs, so a TLBI of
another's leaves them alone (a process exiting used to empty every CPU's), and
a TLBI of one address looks for the pieces of a block among the entries the
largest block it could be in may have, not every entry, once there are too many
to list: a boot's 140 thousand of them had been most of its work. An
instruction-cache invalidation neither kicks nor waits: another CPU is
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
macOS-only. A machine can run all its CPUs in one region, an equal slice each
(`JitRegion::split`, `Jit::in_region`), which is how the C API runs them. A CPU
uses at most 128 MiB of its region, the reach of the B and BL that take its
blocks to the stubs at the start, and empties its cache when that is full; a
block's room is checked before its branches are resolved. Code that runs once
or never — the memory slow paths and their literals, and the exits a chained
branch takes until it is patched — goes in a second, cold section of the
block. The region is used a MiB at a time: blocks fill a segment upwards from
its start and their cold sections fill it downwards from its end, so the code
that runs is packed together for the instruction cache while every
conditional branch (which reaches a MiB) goes straight to its block's cold
code. Footprint is most of what translated code costs: a profile of a compile
had 40% of its samples on the first instruction of a block, and 32 bytes of
padding after each block, never run, made it 17-23% slower.

**On iOS** an app cannot make memory executable on its own; a debugger
attached to it can, and StikDebug's scripts handle one `brk #0x69` stop a
launch. `JitRegion::shared` maps the region executable (private, anonymous),
checks that a debugger is tracing the process (a stop with none would kill
it), stops with the region's address in x0 and its length in x1, and once the
debugger has prepared every page and resumed past the instruction, remaps the
same pages (copy = false) as the writable alias. The order matters: made the
other way round, a writable mapping with an executable alias that the debugger
prepares, the pages fault on execution. The region is kept for the life of the
process, so a later machine gets it back. Every page is made resident when it
is prepared, so it counts against the app's memory limit from the start.

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
direct and indirect calls and indirect jumps, across a page boundary, every
load/store form the fast path rewrites, and pushes, pops and arithmetic on SP
— run natively (in a child process, caught at a
trailing BRK through the signal machine context) and translated with the fast
paths on and off, and the full register file and memory are compared
(`crates/crate-jit/tests/el0_random.rs`). Two variants name X18 (natively it is
X29) and run one translation through the MMU at two virtual addresses. The
arm64 kvm-unit-tests run on jit at 4 and 16 KiB and on hvf, single- and
multi-CPU (`tools/kvm-unit-tests.py`); `tools/el1-diff.sh` compares small EL1
programs on jit and hvf.

## The C API

`crate-ffi` implements `include/crate.h` over crate-virt's Rust API and builds
as a static library; `tools/xcframework.sh` packages it with the header and a
module map (Swift's `import Crate`) as an XCFramework for iOS and macOS on
arm64. The app links libslirp itself: on iOS the framework it ships (aTerm's
`slirp.0.framework`; Homebrew's libslirp is a macOS library), on a Mac
Homebrew's, and the Hypervisor framework there.

A config is checked as it is built, each call failing on its own mistake. A
machine runs on a thread of its own (`crate-machine`), which reports RUNNING
and then STOPPED with the reason through the state callback; `wait` returns
once that report has returned. Console output reaches the app on the vCPU
thread that wrote it; input goes in through `crate_console_write`, which takes
what fits (the UART's 32-byte FIFO, the buffers a virtio console's driver
posted), and `input_ready` says when there is room again: when the UART's
FIFO empties, when the guest posts buffers. Pointers and enumerations from C
are checked, and a panic never crosses into C: a panic hook stops the machine
with CRATE_STOP_ERROR, so the other CPUs leave guest code rather than run on
without one. `CRATE_CPU_AUTO` probes hvf by making and dropping a VM.

A display's frames reach the app's callback while it says the display is
active, and its VNC servers (`crate_config_add_vnc`, a Unix socket or a TCP
port) while they have clients; the GPU reads frames back only while one of
them wants them. Input calls go straight to the keyboard and tablet devices.
virgl's libraries are the app's to ship: an iOS app embeds the frameworks,
which crate finds by their names.

`tools/crate-boot` is a C program that boots a guest through the API the way
an app does, optionally on executable memory it maps itself.

## Not yet

- Pause and resume, a power button and disk discard are in crate.h but return
  CRATE_ERR_UNSUPPORTED.
- No blob resources, so no zero-copy scanout: every frame shown is a GL
  readback and a copy to the app.
- The translator keeps no translations across launches and shares none
  between CPUs (a second compile translates almost nothing, so sharing would
  not speed it up); each CPU has a code region of at most 128 MiB (a slice of
  one region through the C API). The guest's 16 KiB pages still take four
  4 KiB TLB entries. Direct branches to another page still go through the jump
  cache, and every access's TLB lookup is a load on the path to its data.
- PL031 alarms never fire.
- No block I/O thread, no virtio event-index or offloads, no PCIe.
- Loads and stores to MMIO that the CPU cannot describe (ISV clear) stop the
  machine; Linux's accessors never make them.
