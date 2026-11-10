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

Status: early. Nothing runs yet; see [docs/PLAN.md](docs/PLAN.md).

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
      crate-devices     interrupt controller, timer, UART, RTC, PCIe, virtio
      crate-virt        the virt machine: memory map, device tree, Linux boot
      crate-host        executable memory, guest RAM, disks, network, GPU
      crate-ffi         the C API's implementation (libcrate)
    tools/crate-run     command-line runner for development
    docs/               plan, provenance

## Building

Needs Rust 1.85 or later.

    cargo build

## License

[MPL-2.0](LICENSE). Changes to crate's own files stay open; apps of any
license, open or closed, can use it. MPL-2.0 also allows combining crate with
GPL-2.0-or-later code.
