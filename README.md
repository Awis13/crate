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
Apple's Hypervisor framework (milestone 1); the translator does not exist
yet. See [docs/PLAN.md](docs/PLAN.md) and, for how the pieces fit,
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
      crate-ffi         the C API's implementation (libcrate)
    tools/crate-run     command-line runner for development
    docs/               plan, design, provenance

## Building

Needs Rust 1.85 or later, and libslirp for user-mode networking
(`brew install libslirp`; another location can be given in `SLIRP_LIB_DIR`).

    cargo build
    cargo test

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

## License

[MPL-2.0](LICENSE). Changes to crate's own files stay open; apps of any
license, open or closed, can use it. MPL-2.0 also allows combining crate with
GPL-2.0-or-later code.
