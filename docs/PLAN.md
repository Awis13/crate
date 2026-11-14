# Plan

## What crate is

An engine that runs arm64 virtual machines inside an app on an arm64 host
without hypervisor access. Apps link it the way UTM links QEMU: the app owns
the interface, crate owns the machine. Linux guests come first; the design
leaves room for other guests, macOS among them, as machines of their own.

The longer aim: other emulators for these devices should be able to replace
their own CPU core with crate's, and app developers should be able to build a
VM app on it without touching the engine.

## Shape

    app (crate-run, anyone's)
      |  C API (include/crate.h), Rust API, Swift package
    machine     the board: memory map, firmware tables, boot protocol    crate-virt
      |  device interface (MMIO, PCI functions, interrupts, reset, state)
    devices     GICv3, timer, PL011, PL031, PCIe, virtio-*               crate-devices
      |  CPU backend interface
    CPU         jit (translator)  |  hvf (macOS)  |  later: interpreter   crate-jit, crate-hvf
      |  host interface
    host        executable memory, guest RAM, disks, network, GPU        crate-host

Each line is meant to be an interface a third party can implement: a new
machine, a device, a CPU backend or a host backend, without changing the
others.

**CPU backends.** `jit` is the point of crate: the translator, for hosts where
the app may create executable memory (with a debugger attached on iOS). `hvf`
runs the same machine and devices at native speed on a Mac, through Apple's
Hypervisor framework. It is the reference the translator is tested against, it
proves the devices before the translator exists, and it gives Mac apps a fast
path. An interpreter for hosts without executable memory may come later.

**Guest-specific behaviour lives in machines, never in the engine.** The
translator detects the guest's page size from its translation control
register and generates code for it; tuning that only one guest needs is a
machine's setting. Since the translator writes code, settings like these become
constants in the generated code rather than run-time checks.

**The app's side.** Console bytes, screen updates, input and port forwards
cross the API as calls and callbacks, not sockets: an app embeds crate, it does
not talk to it over a network. A VNC server can still be offered as an
optional piece for apps that want one.


## Milestones

| # | Milestone | Status |
|---|---|---|
| 0 | Repository, plan, C API draft, Rust workspace | Done |
| 1 | `virt` machine and devices on the `hvf` backend: Debian boots to a prompt on a Mac, with disk, network, console and port forwards | Reached 2026-11-10 |
| 2 | Translator on its own: instruction copying, MMU with 4 KB and 16 KB granules, exceptions, system registers, timers; random instruction streams give the same results as the CPU itself, and the arm64 kvm-unit-tests pass | Reached 2026-11-10 |
| 3 | Full system on `jit`: several CPUs, TLB maintenance across CPUs, exclusives, self-modifying code; Debian on 4 CPUs, benchmark outputs identical to native, faster than QEMU's TCG | Reached 2026-11-10 |
| 4 | GPU, display, input: virtio-gpu 2D and virgl (virglrenderer over ANGLE), keyboard and tablet; Linux desktops and Chrome on a Mac | Reached 2026-11-11 |
| 5 | iOS: an XCFramework, executable memory from an app with a debugger attached; Linux with a desktop on an iPhone | Reached 2026-11-11 |
| 6 | Speed: the translator's own costs first | Reached 2026-11-11; more under [Speed](#speed) |
| 7 | Firming up the API: the C API frozen as 1.0, documentation, a sample app | Open; see [below](#milestone-7-firming-up-the-api) |
| later | Other guests as machines of their own | Not started |

## Where it stands

Measured on an M3 MacBook Air, and on an iPhone with an A20 on iOS 27.1, whose
executable memory came from StikDebug's `legacy.js` script.

**On a Mac.** Debian 13 boots to a root prompt in about 1.2 s on `hvf` and
about 3 s on `jit`, with 4 CPUs, its disk, user networking, SSH through a port
forward, and serial and virtio consoles. Every arm64 kvm-unit-test, single- and
multi-CPU, passes or skips on `jit` at 4 and 16 KiB pages and on `hvf` (the
gicv2 and migration tests do not apply). A benchmark run inside the guest (4
CPUs, 4 GB) prints the same output hashes on `jit` as on `hvf`; on `jit`
(medians of 3):

| | crate `jit` | QEMU's TCG | native |
|---|---|---|---|
| the whole pass, one task at a time | 24.5 s | 130.5 s | 9.1 s |
| gcc -O2 | 6.9 s | 35 s | 1.6 s |
| 1000 process spawns | 0.52 s | 4.4 s | 0.13 s |
| four compiles at once | 9.2 s | 56-77 s | 1.8 s |

QEMU's and native's figures are from 2026-11-07, on the same machine and
guest. Every line of the benchmark is in the README's
[Benchmarks](../README.md#benchmarks) section.

A virtio-gpu with virgl (virglrenderer over ANGLE on Vulkan, OpenGL ES 3.1 in
the guest), a keyboard and a tablet run Xfce, labwc and Hyprland desktops on the
GPU on both backends. Chrome composites, rasterizes and runs WebGL and WebGPU
on the GPU: on `jit` it shows a page 3.1-4.3 s after starting (0.8 s on `hvf`),
a WebGL canvas runs at 60 frames a second, and an idle Xfce desktop costs
1.2-1.5% of a host core.

**On an iPhone.** With 3 CPUs and 1950 MiB of guest memory, Debian reaches a
root prompt about 5.3 s after the app launches and a drawn desktop at about
6.3 s. Desktops run on the GPU, Chrome composites on it, the desktop follows the
phone as it rotates or folds, and taps, long presses, two-finger scrolls and a
keyboard reach the guest. In the guest, sha256 of 100 MB takes 84 ms, and
headless Chrome runs a V8 loop of 10 million in 2.4 s, its start included. A
second distribution, on Linux 7.3, needed only one generic fix (AIDR_EL1,
which Linux 7 reads at boot).

**Where the speed came from.** A profile of a compile showed most of the time
was the translator's own. Loads and stores now have scratch registers of their
own and run at their host address with no offset. Code that runs rarely sits
apart from code that runs, so the instruction cache holds what matters. A call
is a host call whose return the CPU predicts, and each indirect branch is
predicted from few sites. TLB refills go through a page-walk cache, and
invalidations no longer scan whole tables. Direct branches to another page are
chained behind a guard that stays right whatever the guest's TLB discipline. A
group of accesses off one base runs each as a single host instruction, with a
cold copy for when a check fails. Each change was judged by the process's
retired cycles over repeated compiles, since wall time was too noisy, and kept
only if it won. Measured and dropped: a code cache shared by the CPUs (a second
compile translates almost nothing), loading a TLB entry's tag and delta apart,
and a virtio-blk thread.

## Milestone 7: firming up the API

crate's C API (`include/crate.h`) is a draft. Until 1.0, its minor version
changes with every incompatible change (0.1 today), so an app pins the release
it builds against. Before 1.0:

- **Unfinished calls.** Pausing and resuming a machine, the power button and
  disk discard are declared in `crate.h` but return `CRATE_ERR_UNSUPPORTED`.
  Each gets implemented or removed.
- **Calls no app relies on yet.** The apps built on crate so far use the
  configuration, console, display and input calls, and start and kill a
  machine. Port forwards, VNC, a fixed network MAC address, asking the guest to
  shut down, waiting on a machine and reading its state are used by the
  development tools at most. Each gets a test through the C API before it is
  frozen.
- **What VM apps will want next.** Faster networking (virtio-net has no
  offloads yet, so TCP moves about 20 MB/s), shared folders, a shared
  clipboard, sound, and saving and restoring a running machine. Which of these
  1.0 needs is open. Structs passed in begin with their own size and everything
  else is set through functions, so later versions can add them without
  breaking apps.
- **Which interfaces are stable.** 1.0 freezes the C API. Whether the Rust
  crates' interfaces (machines, devices, CPU and host backends) become stable
  too, or stay internal with the C API as the one stable surface, is open.
- **A Swift package** over the XCFramework each release publishes.
- **Documentation and a sample.** A guide for app developers, and a minimal
  sample app beside `tools/crate-boot`, a C program that boots a guest through
  the API.
- **Fuzzing** of everything that parses guest-controlled data (virtio queues,
  device registers), since an app may run guests it does not trust.

## Remaining work

Everything else known to be left, as of 2026-11-13. Gains are time saved,
measured or estimated on a Mac.

### Larger pieces

- **An interpreter backend**, for hosts where the app may not create
  executable memory (on iOS, a launch without a debugger). Not started. It is
  the one piece that would let an app use crate on every launch.
- **Other guests as machines of their own.** Not started.
  - Windows 11 on Arm needs UEFI firmware (EDK2, a separate binary under
    its own license) and ACPI tables, since it reads no device tree; PCIe
    with MSIs (the GIC's ITS and LPIs); NVMe and an xHCI controller with a
    USB keyboard and tablet, which it has drivers for; the firmware's
    framebuffer for a display; and virtio-net over PCI. The translator will
    also meet instructions and registers Linux never used. Estimated at about
    the size of milestones 1 to 5 together. A first step would boot the
    firmware on `hvf` and run Windows' boot loader from the existing
    virtio-mmio disk, proving firmware, ACPI, CPU, interrupts and timers
    before any PCIe.
  - macOS, later.
- **PCIe** as a second virtio transport, which both of those need.

### Speed

- **16 KB-page guest kernels: done.** A 16 KiB page fills its four 4 KiB TLB
  pieces at once and a TLBI drops them together. Against the same kernel at
  4 KB: compute ties, boot about 5% faster, 1000 spawns 15-20% faster; it uses
  more memory (about 250 MB more in use with a desktop at 1950 MiB, and
  Chrome's anonymous memory 55-64% more). Which page size a guest uses is the
  machine's or the app's choice.
- **Paravirtual TLB flushes and idle.** Not started. Would be a machine's
  setting, since only Linux has the hooks.
- **Guest memory mapped by the host MMU.** Shelved. On the Mac it would cut
  compute lines by 20-40% but make spawning 2-3x slower, unless used per
  address space. A guest's 47-bit address space does not fit even in macOS's;
  on an iPhone the app gets 7.5 GiB of address space (6 GiB in its largest
  range), and every alias of guest RAM it touches costs a page of the app's
  memory. Only for guests with small address spaces.
- **Translations kept across launches.** Not planned for now: every launch
  starts cold, so a reboot reproduces a test.

Found by profiling since (estimates, together about 10-20%), not started:

1. Guest SP and its page address kept in registers within a block: gcc and
   spawns 3-6%.
2. A return-address stack for RET: 1-4% on most lines.
3. A code cache shared by the CPUs: boot 5-8%, elsewhere nothing.
4. A faster translator (decode, emit, install): boot 4-7%, `node -e 0` 3-5%.
5. Indirect branches looked up at the site: awk 8-12%.
6. IC IVAU without an exit (CTR_EL0.DIC set, stores to translated pages
   caught in the store TLB): `node -e 0` about 6%.
7. Flushes handed off the vCPU in virtio-blk: boot 2-5%.
8. Cheaper exits: spawns 3-5%.
9. A cheaper check for exit requests at backward branches: loops 0-5%.

Beyond those, translated code itself runs at about 1.8x native in a compile;
an optimizing second tier for hot code might give about 1.5x on compute
(a guess, not measured).

### Gaps and known bugs

- **No virtio-net offloads or event index**, so TCP moves about 20 MB/s. The
  gap a user notices most (package installs, clones).
- Pause and resume, the power button and disk discard are in `crate.h` but
  return CRATE_ERR_UNSUPPORTED (see milestone 7).
- No virtio-gpu blob resources: every frame shown is a GL readback and a copy.
- PL031 alarms never fire.
- An MMIO access the CPU cannot describe (ISV clear) stops the machine. Linux
  never makes one; other guests may.
- vCPU threads run translated code with SP just above the CPU state's TLB and
  no alternate signal stack, so a signal delivered to one (the SIGUSR1 dump
  handler is installed without SA_ONSTACK) would write its frame over the
  TLB. Nothing sends one today.
- With breakpoints or single-step active, the dispatcher keeps a pending
  same-page chain across a breakpoint or step, so a chain site could be
  patched to the debug exception's block.
- **Xfce on Wayland does not follow new modes.** `crate_display_set_modes`
  unplugs the display and plugs it back in, and the compositor takes the new
  preferred mode (labwc, with its bar and background, follows a rotation), but
  xfce4-panel disappears and xfdesktop stays at the old size: neither comes
  back when its only output goes away. Possible fixes, none started: connect
  the new modes as a second output before removing the old one, so the guest
  always has one; or restart the two in the guest when outputs change.

## Testing

- **Against the CPU itself.** Guest and host are the same architecture, so any
  instruction sequence can run natively and translated, and the register and
  memory state compared. Random sequences find what hand-written tests miss.
- **Against hvf.** The same machine on both backends must behave the same.
- **kvm-unit-tests' arm64 suite** for exceptions, timers, interrupts, PSCI.
- **Whole systems.** Linux boot, a benchmark whose outputs must match native
  byte for byte, and the programs that break emulators: JITs in the guest (V8,
  Bun, Chrome), compilers, many processes at once.
- **Fuzzing** of everything that parses guest-controlled data (virtio queues,
  device registers).

## Risks

- Correctness across CPUs: TLB maintenance, memory ordering, exclusives.
- Guests that write their own code (every JavaScript engine).
- iOS's rules for executable memory, which have changed before.
- The translator's hot paths are unsafe Rust by nature; safety comes from
  keeping them small and testing them hard.

## Rules

- Written from public specifications only; see [PROVENANCE.md](PROVENANCE.md).
  Contributions follow the same rule; see
  [CONTRIBUTING.md](../CONTRIBUTING.md).
