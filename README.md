<p align="center">
  <img src="docs/icon.png" alt="crate's icon: a wooden crate holding a terminal and two other windows" width="128" height="128">
</p>

# crate

**Compact Runtime for Arm64 Translation and Emulation.**

An embeddable engine that runs arm64 virtual machines on arm64 hosts that offer
no hypervisor to apps, such as iPhones and iPads. Guest code runs through an
arm64-to-arm64 translator: most instructions are copied into host code as they
are, and only what touches the machine (memory, branches, system registers,
exceptions) is rewritten. On a Mac the same machines can also run on Apple's
Hypervisor framework.

crate is a library, not an app. An app links it, describes a machine through
its API, and gets the console, the screen and input back as callbacks; the
engine never assumes what the app looks like.

Status: early. The `virt` machine boots Debian to a shell on a Mac through
Apple's Hypervisor framework and through the translator, on one to four CPUs,
several times faster than QEMU's TCG. A virtio-gpu with virgl (virglrenderer
over ANGLE), a keyboard and a tablet run Linux desktops and Chrome with
hardware compositing, on a Mac and on an iPhone, where the app gets executable
memory from an attached debugger. **The C API is a draft**: until 1.0 it
changes incompatibly as needed (see
[milestone 7](docs/PLAN.md#milestone-7-firming-up-the-api)). See
[docs/PLAN.md](docs/PLAN.md) for what is done and what is left, and
[docs/DESIGN.md](docs/DESIGN.md) for how the pieces fit.

## Goals

- **Linux first.** A `virt` machine that boots stock arm64 Linux kernels with a
  device tree, virtio devices and a GPU for desktops.
- **A real library.** A versioned C API (`include/crate.h`) for apps in any
  language, a Rust API for Rust code, and a Swift package for Apple platforms.
- **Extensible.** Machines, devices, CPU backends and host backends are separate
  pieces with interfaces between them, so a project can add its own device or
  board without forking the engine.
- **Not tied to one guest.** Nothing in the engine assumes Linux or macOS;
  what a guest needs lives in its machine.
- **Fast.** Close to native for ordinary code, with the remaining overhead
  (memory translation, the kernel's own work) attacked directly.

## Layout

    include/crate.h     the C API
    crates/
      crate-core        machine model: guest memory, CPUs, buses, interrupts, traits
      crate-jit         the translator (CPU backend for hosts without a hypervisor)
      crate-hvf         CPU backend on Apple's Hypervisor framework (macOS)
      crate-devices     GICv3, UART, RTC, virtio (block, network, console)
      crate-virt        the virt machine: memory map, device tree, Linux boot
      crate-host        executable memory, guest RAM, disks, network, GPU
      crate-vnc         an optional VNC server for a display
      crate-ffi         the C API's implementation (libcrate)
    tools/crate-run     command-line runner for development
    tools/vnc.py        screenshots and input through a VNC server
    docs/               plan, design, provenance

## Building

Needs Rust 1.85 or later, and libslirp for user-mode networking
(`brew install libslirp`; another location can be given in `SLIRP_LIB_DIR`).

    cargo build
    cargo test

## The C API and iOS

`include/crate.h` is the API, a draft until 1.0: its minor version changes with
every incompatible change, so pin a release. `tools/xcframework.sh` builds it
(release) as
`target/xcframework/Crate.xcframework`, a static library for iPhones, Simulators
on Apple silicon and Macs on arm64, with the header and a module map for Swift
(`import CrateEngine`); it needs the `aarch64-apple-ios` and
`aarch64-apple-ios-sim` Rust targets. Pushing a tag `vX.Y.Z`, the workspace
version, publishes it as a release (`.github/workflows/release.yml`): the zip and
its SHA-256, which an app pins. The app links libslirp itself. On iOS the
app passes in executable memory that a debugger it attached has prepared (see
`crate_config_set_jit_region`); crate cannot get its own there.

`tools/crate-boot/build.sh` builds a C program that boots a guest through the
API, as an app does:

    target/release/crate-boot --backend jit --cpus 4 --memory 2048 \
        --kernel vmlinuz --initrd initrd.img --disk rootfs.img \
        --cmdline "console=ttyAMA0 root=/dev/vda rw" --forward 2230:22

## Running a guest on a Mac

`crate-run` needs Apple's `com.apple.security.hypervisor` entitlement, so build
it with the script that signs it (ad hoc), which prints the binary's path:

    tools/crate-run/build.sh

Then, for example, a Debian image with four CPUs, its disk, the network and
SSH forwarded from port 2230:

    target/debug/crate-run --kernel vmlinuz --initrd initrd.img \
        --cmdline "console=ttyAMA0 root=/dev/vda rw" \
        --disk rootfs.img --cpus 4 --memory 2048 --forward 2230:22

The terminal is the guest's serial console; Ctrl-A x stops the machine.
`crate-run --help` lists the rest.

A screen, keyboard and tablet come with `--display 2d` or `--display virgl`,
seen through a VNC server (`--vnc vnc.sock`, `--vnc-tcp 127.0.0.1:5905`).
virgl needs virglrenderer and ANGLE, which crate loads when the machine starts,
found through dyld's paths. Set them in a script that runs crate-run itself:
macOS drops `DYLD_*` variables on the way through system binaries such as
`nohup` or `perl`.

    export DYLD_FRAMEWORK_PATH=$GPU/angle          # ANGLE's EGL.framework
    export DYLD_LIBRARY_PATH=$GPU/lib:/opt/homebrew/opt/molten-vk/lib
    target/release/crate-run ... --display virgl --vnc vnc.sock
    tools/vnc.py vnc.sock shot screen.png

## Benchmarks

A Debian 13 guest runs a fixed set of workloads once it has booted, and prints
each one's time and a hash of what it printed; every engine must print the same
hashes as native. Seconds, medians of 3, on an M3 MacBook Air with 4 CPUs and
4 GB for the guest. crate is its `jit` backend (2026-11-11). Native is QEMU on
Apple's Hypervisor framework (`-accel hvf -cpu host`), and QEMU's TCG is QEMU
10.0.2's translator (`-accel tcg,tb-size=512,split-wx=on -cpu max,sve=off`),
both measured 2026-11-07 on the same machine and guest.

| one at a time | crate `jit` | QEMU's TCG | native |
|---|---|---|---|
| boot to the benchmark | 2.97 | 13.80 | 1.82 |
| gcc -O2, one large file | 6.93 | 35.30 | 1.59 |
| gcc again | 6.67 | 34.17 | 1.48 |
| `node -e 0` | 0.41 | 1.44 | 0.13 |
| node, integer loop | 0.51 | 1.98 | 0.16 |
| node, floating-point loop | 0.91 | 5.61 | 0.32 |
| python loop | 1.36 | 12.97 | 0.26 |
| awk, sqrt loop | 0.58 | 7.94 | 0.27 |
| xz -6, 4 MB | 1.39 | 4.93 | 0.65 |
| sha256, 50 MB | 0.06 | 0.30 | 0.04 |
| 1000 process spawns | 0.52 | 4.42 | 0.13 |
| find /usr | 0.50 | 2.50 | 0.45 |
| **the whole pass** | **24.5** | **130.5** | **9.1** |

| all four CPUs at once | crate `jit` | QEMU's TCG | native |
|---|---|---|---|
| 4 x gcc -O2 | 9.20 | 56-77 | 1.83 |
| xz -T4, 50 MB | 11.40 | 43-61 | 6.09 |
| 4 x 250 process spawns | 0.13 | 2.31-2.52 | 0.03 |

QEMU's TCG had two runs of these, so its column gives both. An idle shell costs
crate about 0.65% of a host core, and QEMU's TCG 1.0%. On a fanless laptop wall
times move 5-20% between identical runs, so smaller changes in crate are judged
by the process's retired cycles instead.

**On an iPhone.** An iPhone Duo (A20, iOS 27.1), `jit` with 3 CPUs and 1950 MiB,
executable memory from StikDebug's `legacy.js`, measured 2026-11-11 (before the
last round of speed work above). Every output matched the Mac's.

| | crate `jit` |
|---|---|
| root prompt, from the app's launch | 5.3 s |
| desktop drawn, from the app's launch | 6.3 s |
| sha256, 100 MB | 84 ms |
| perl loop of 3 million | 312 ms |
| sort, 300,000 lines | 72 ms |
| gzip and gunzip, 50 MB | 363 ms |
| three sha256 of 50 MB at once | 81 ms |
| `claude --version` (Claude Code) | 165 ms |
| headless Chrome: start, then a V8 loop of 10 million | 2.4 s |

The benchmark's guest image (3 GB) is not part of this repository.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): crate is written from public
specifications only, and contributions are too. Security issues go privately,
as [SECURITY.md](SECURITY.md) says.

## License

[MPL-2.0](LICENSE). Changes to crate's own files stay open; apps of any
license, open or closed, can use it. MPL-2.0 also allows combining crate with
GPL-2.0-or-later code.
