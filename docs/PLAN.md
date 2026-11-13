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
| 6 | Speed: the translator's own costs first (scratch registers for loads and stores, cold code apart, predicted returns), then 16 KB-page guest kernels, Linux paravirtual hooks (TLB flushes, idle), guest memory mapped by the host MMU, translations kept across launches | on an M3, jit, 4 CPUs: every line of crate-bench ties (medians of 3, within about 5%) or beats the other translator its README compares, every output hash unchanged, kvm-unit-tests as before; on the iPhone, Debian still at its prompt in about 7 s and its desktop in about 8.4 s |
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
beside its other two. On an iPhone (A20, iOS 27.1, JIT from StikDebug's
legacy.js), Debian on three CPUs reaches a root prompt 6.3-6.6 s after the app
launches, the debugger's authorization of the 384 MiB code region included
(the other translator: 7 s); a second tab connects over SSH, port forwards
work, Claude Code runs, and poweroff and relaunch work. The desktop,
resolution changes and the Omarchy guest wait for milestone 4.

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
faster, and Omarchy boots. On the iPhone (aTerm crate with this code, Debian,
jit, 3 CPUs, 1950 MiB, two launches) the root prompt came 5.4-5.7 s after
launch (milestone 5: 6.6-7.0) and the desktop 6.5 s (8.2-8.4); in the guest,
every output matching the Mac's, sha256 of 100 MB took 85-90 ms, a perl loop
of 3 million 410-421 ms, sorting 300 thousand lines 87-101 ms, gzip and gunzip
of 50 MB 416-426 ms, three sha256 of 50 MB at once 95-108 ms and `claude
--version` 111-164 ms. PLAN's own items for this milestone remain: 16
KB-page guest kernels, paravirtual TLB flushes and idle, guest memory mapped
by the host MMU, translations kept across launches; so do a code cache shared
by the CPUs and chained branches to another page. The largest cost left in a
compile is the TLB lookup, a load on the path to every access's data.

The milestone's goal then became to tie or beat the other translator
crate-bench compares on every line, which it does on the Mac since
2026-11-11 (medians of 3, M3 MacBook Air under the user's other work): boot
2.9 s (5.27), gcc -O2 8.0 s (9.34), gcc again 7.6-8.3 (8.16), node -e 0 0.42-0.43
(0.42), node's integer loop 0.52-0.55 (0.59), its FP loop 0.92-0.95 (1.03),
python 1.45-1.56 (1.96), awk 0.65-0.73 (0.72), xz 1.41-1.43 (1.79), sha256
0.05 (0.07), find 0.49-0.53 (0.59), 1000 spawns 0.54 (3.39), the whole pass
27-31 s (36.1); four compiles at once 12.4-15.8 s (21.4), xz -T4 12.7-16.0
(18.2), 4 x 250 spawns 0.20 (1.88); an idle shell 0.65% of a core (2.1%).
Wall time on that machine is too noisy for a few percent, so changes were
judged by the process's retired cycles over repeated compiles. What did it:
footprint first (32 bytes of never-run padding per block made gcc 17-23%
slower), so slow paths moved into the same MiB segment as their block and the
trampolines left the code that runs, single loads and stores take the TLB
delta as a register offset, and the EL0 load TLB sits right below SP; 32
copies of the jump-cache stub per EL, so each indirect branch is predicted
from few sites; and the TLB refills, 15% of a compile, which a page-walk cache
and a larger TLB cut, while TLB invalidations stopped scanning whole tables
(a boot's work fell by two thirds). A code cache shared by the CPUs was
measured not to matter: a second compile translates almost nothing. On the
iPhone (aTerm crate with this code, Debian, jit, 3 CPUs, 1950 MiB, launched
through StikDebug) the root prompt came 5.3 s after launch and the desktop
6.3 s, with labwc and xfce4-panel running and no faults; in the guest, every
output matching the Mac's, sha256 of 100 MB took 84 ms, the perl loop 312 ms
(410-421 after the first three steps), sorting 72 ms, gzip and gunzip of 50 MB
363 ms, three sha256 at once 81 ms, `claude --version` 165 ms, and headless
Chrome ran a V8 loop of 10 million to the expected result in 2.4 s, Chrome's
start included.

A second round of speed work followed (2026-11-11), each change measured and
kept only if it won. Direct branches and calls to another page, half of a
compile's jump-cache lookups, are chained behind a guard: a key of the target
address with the jump cache's own context number or generation in its top
bits, which changes exactly when a lookup would stop finding the block, so the
guard is right on any OS's TLB discipline, not Linux's. Instruction-cache
maintenance after writing code is done once per dispatcher visit, with one
set of barriers. A group of accesses off one base runs each access as a single
host instruction, and a failed check runs a cold copy of the whole stretch.
On the way, a TLBI for another address space was found to skip global and
upper-half EL0 translations, which the architecture says it reaches. Medians
of 3 against milestone 6 in the same session (M3 Air, jit, 4 CPUs, 4 GB, a
minute's rest before each pass, load 2-6): gcc -O2 6.93 s (8.07), gcc again
6.67 (7.86), node -e 0 0.41 (0.43), python 1.36 (1.40), xz 1.39 (1.42), the
whole pass 24.5 s (26.9); four compiles at once 9.20 s (10.47), xz -T4 11.40
(11.47); boot 2.97 s (2.89); every output hash unchanged and the
kvm-unit-tests as before. Measured and dropped: loading a TLB entry's tag and
delta apart (the CPU predicts a plain load's value, but the extra instruction
cost more), indexing the other TLBs down from SP like the near one (neutral),
and a virtio-blk thread (the host serves the reads from its cache in
microseconds; small reads got slower).

## Remaining work

Everything known to be left, as of 2026-11-12. Gains are time saved, measured
or estimated on the Mac; "on hold" means the owner has decided to wait.

### Milestones

- **Milestone 7, ready to publish.** On hold.
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

Milestone 6's own list:

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
- **Translations kept across launches.** On hold: every launch starts cold,
  so a reboot reproduces a test.

Found since, by profiling (estimates, together about 10-20%). On hold:

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
  return CRATE_ERR_UNSUPPORTED.
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
