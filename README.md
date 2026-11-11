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
engine never assumes what the app looks like. The first app built on it is
[aTerm](https://github.com/oneamendez/aterm-ios), a terminal for a local Linux
machine.

Status: early. The `virt` machine boots Debian to a shell on a Mac through
Apple's Hypervisor framework (milestone 1) and through the translator, on one
to four CPUs, faster than QEMU's TCG (milestone 3). The C API is implemented
and aTerm runs Debian on crate on an iPhone, with executable memory from an
attached debugger (the first part of milestone 5). A virtio-gpu with virgl
(virglrenderer over ANGLE) and a keyboard and tablet run aTerm's desktops and
Chrome with hardware compositing on a Mac, on both backends (milestone 4). See
[docs/PLAN.md](docs/PLAN.md) and, for how the pieces fit,
[docs/DESIGN.md](docs/DESIGN.md).

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

`include/crate.h` is the API. `tools/xcframework.sh` builds it (release) as
`target/xcframework/Crate.xcframework`, a static library for iOS and macOS on
arm64 with the header and a module map for Swift; it needs the
`aarch64-apple-ios` Rust target. The app links libslirp itself. On iOS crate
gets executable memory from an attached debugger (see `crate_config_set_jit_region`).

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

## License

[MPL-2.0](LICENSE). Changes to crate's own files stay open; apps of any
license, open or closed, can use it. MPL-2.0 also allows combining crate with
GPL-2.0-or-later code.
