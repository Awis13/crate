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
| 6 | Speed: 16 KB-page guest kernels, Linux paravirtual hooks (TLB flushes, idle), guest memory mapped by the host MMU, translations kept across launches | targets set from the milestone 3 measurements |
| 7 | Ready to publish: device and machine interfaces frozen as v1, documentation, a minimal sample app | the owner decides to open the repository |
| later | Other guests (macOS) as machines of their own | |

Milestone 1 was reached on 2026-11-10: aTerm's Debian 13 image boots to a root
prompt under `crate-run` on an M3 in about 1.2 s, with 4 CPUs, its disk, user
networking, SSH through a port forward, and serial and virtio consoles.

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
