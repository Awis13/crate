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

    app (aTerm, crate-run, anyone's)
      |  C API (include/crate.h), Rust API, Swift package
    machine     the board: memory map, firmware tables, boot protocol    crate-virt
      |  device interface (MMIO, PCI functions, interrupts, reset, state)
    devices     GICv3, timer, PL011, PL031, PCIe, virtio-*               crate-devices
      |  CPU backend interface
    CPU         jit (translator)  |  hvf (macOS)  |  later: interpreter   crate-jit, crate-hvf
      |  host interface
    host        executable memory, guest RAM, disks, network, GPU        crate-host

Each line is an interface a third party can implement: a new machine, a device,
a CPU backend or a host backend, without changing the others.

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

Each ends in something that runs.

| # | Milestone | Done when |
|---|---|---|
| 0 | Repo, plan, C API draft, workspace | `crate.h` v0 reviewed |
| 1 | `virt` machine and devices on the `hvf` backend | aTerm's Debian image boots to a prompt on a Mac: disk, network, console, port forwards |
| 2 | Translator on its own: instruction copying, MMU with 4 KB and 16 KB granules, exceptions, system registers, timers | random instruction streams give the same results as the CPU itself; the arm64 kvm-unit-tests pass |
| 3 | Full system on `jit`: several CPUs, TLB maintenance across CPUs, exclusives, self-modifying code | Debian boots on 4 CPUs; benchmark outputs byte-identical to native; faster than QEMU's TCG |
| 4 | GPU, display, input: virtio-gpu 2D and virgl (virglrenderer over ANGLE), keyboard and tablet | aTerm's desktops, Chrome and Claude Code run under `crate-run` on a Mac |
| 5 | iOS: XCFramework, iOS executable-memory provider, aTerm adopts crate beside its current engines | Debian and Omarchy on a device pass aTerm's checklist: console, SSH tabs, forwards, desktop, resolution change, Claude Code |
| 6 | Speed: the translator's own costs first (scratch registers for loads and stores, cold code apart, predicted returns), then 16 KB-page guest kernels, Linux paravirtual hooks (TLB flushes, idle), guest memory mapped by the host MMU, translations kept across launches | on an M3, jit, 4 CPUs: crate-bench's single pass under 50 s, gcc -O2 under 15 s, four compiles at once under 30 s, every output hash unchanged; on the iPhone, Debian still at its prompt in about 7 s and its desktop in about 8.4 s |
| 7 | Ready to publish: device and machine interfaces frozen as v1, documentation, a minimal sample app | the owner decides to open the repository |
| later | Other guests (macOS) as machines of their own | |

Milestone 1 was reached on 2026-11-10: aTerm's Debian 13 image boots to a root
prompt under `crate-run` on an M3 in about 1.2 s, with 4 CPUs, its disk, user
networking, SSH through a port forward, and serial and virtio consoles.

Milestone 2 was reached on 2026-11-10 (random instruction streams match the
CPU; the single-CPU kvm-unit-tests pass), and milestone 3 on 2026-11-10: on
`jit`, Debian boots to a root prompt on one CPU in about 13 s and on four in
about 10 s, with SSH through a port forward. Every arm64 kvm-unit-test, single-
and multi-CPU, passes or skips on `jit` at 4 and 16 KiB pages and on `hvf`
(gicv2 and migration tests are not applicable). The engine benchmark
(crate-bench, 4 CPUs, 4 GB) prints the same output hashes as `hvf` on every
line, and every line is faster than QEMU's TCG: the single pass takes 73-81 s
against TCG's 130.5 (native 9), gcc -O2 22-26 s against 35, 1000 spawns 1.6-2.1
s against 4.4; on all four CPUs four compiles take 48-49 s against 56-77. The
measurements, and where the remaining time goes (ordinary loads and stores,
jump-cache lookups), are the input to milestone 6.

The first part of milestone 5 was reached on 2026-11-11, everything but what
needs milestone 4's GPU: the C API is implemented, crate builds as an
XCFramework for iOS and macOS, and aTerm runs Debian on it as a third engine
beside QEMU and Henge. On an iPhone (A20, iOS 27.1, JIT from StikDebug's
legacy.js), Debian on three CPUs reaches a root prompt 6.3-6.6 s after the app
launches, the debugger's authorization of the 384 MiB code region included
(Henge: 7 s); a second tab connects over SSH, port forwards work, Claude Code
runs, and poweroff and relaunch work. The desktop, resolution changes and the
Omarchy guest wait for milestone 4.

Milestone 4 was reached on 2026-11-11: a virtio-gpu with virgl (virglrenderer
over ANGLE on Vulkan, OpenGL ES 3.1 in the guest), a keyboard and a tablet,
the display and input calls of the C API, and an optional VNC server. Under
`crate-run` on an M3, on `jit` and on `hvf`, aTerm's Debian runs all three
desktop styles (Xfce, simple, Hyprland) on the GPU, with clicks, right clicks,
the wheel and the keyboard working and `aterm-resolution` changing modes;
Chrome composites, rasterizes and runs WebGL and WebGPU on the GPU, and Claude
Code runs in a terminal on the desktop. On `jit` Chrome starts and shows a page
in 3.1-4.3 s (hvf 0.8 s) and loads a 2000-row page with script in 1.4-1.5 s
(hvf 0.12 s); a WebGL canvas runs at 60 frames a second on both. An idle Xfce
desktop costs 1.2-1.5% of a host core on `jit` and 0.4% on `hvf`.

Milestone 5 was reached on 2026-11-11. On the same iPhone, aTerm runs crate
with its display (virgl over the ANGLE and virglrenderer frameworks the app
embeds, served to the app by crate's VNC server) and keyboard and tablet,
under the 1950 MiB the app's provisioning profile leaves a guest. Debian
reaches a root prompt 6.6-7.0 s after launch and a drawn desktop 8.2-8.4 s
after; all three desktop styles run on the GPU, Chrome composites on it,
Claude Code runs in a terminal on the desktop, resolution changes work, and
taps, long presses, two-finger scrolls and the key bar reach the guest.
Omarchy (Linux 7.3, Hyprland) needed only a generic fix (AIDR_EL1, which
Linux 7 reads at boot, was undefined); it reaches a prompt in 10-13 s and its
desktop in 43-56 s (18 s and 77 s on its first boot), with SSH tabs, port
forwards, Claude Code and resolution changes working. Its first boot once
stopped responding after two desktop restarts in quick succession, with the
guest short of memory; it did not happen again on the phone or on a Mac at
the same memory. Forwarded sockets no longer raise SIGPIPE when a connection's
other end has gone.

Milestone 6's targets were met on the Mac on 2026-11-11. The speed came from
the translator's own costs, which a profile of a compile showed were most of
its time: loads and stores spilled and reloaded four registers each, every
block entered through the jump cache stalled on its first instruction, and
half of the code was cold. Guest X15..X17 now live in the state between
blocks, so the memory fast path has three scratch registers whenever an access
does not name them, borrows others only otherwise and puts them back lazily,
and runs every access at its host address with no offset; code that runs once
or never goes to the far end of a CPU's code region, so the code that runs is
packed for the instruction cache; and a call is a host BL whose return address
is the translation of the instruction after it, so a return jumps there with
RET and the CPU predicts it. On jit with 4 CPUs and 4 GB, crate-bench's single
pass takes 34.8-36.4 s (milestone 3: 73-81 s; QEMU's TCG 130.5; native 9),
gcc -O2 10.0-10.6 s (22-26), four compiles at once 15.0-16.8 s (48-49), xz -T4
16.6 s (34-38) and 1000 spawns 0.7-0.8 s (1.6-2.1), with every output hash
unchanged and the kvm-unit-tests passing or skipping as before. Debian reaches
a root prompt in 7.0 s on the Mac (4 CPUs), Chrome's V8 gives the same results
faster, and Omarchy boots. PLAN's own items for this milestone remain: 16
KB-page guest kernels, paravirtual TLB flushes and idle, guest memory mapped
by the host MMU, translations kept across launches; so do a code cache shared
by the CPUs and chained branches to another page. The largest cost left in a
compile is the TLB lookup, a load on the path to every access's data.

## How aTerm adopts it

aTerm already chooses an engine at launch. crate becomes another choice beside
the current ones, and replaces them only once it passes aTerm's checklist on a
device. Until then nothing in aTerm depends on crate.

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
- Private until the owner decides otherwise.
