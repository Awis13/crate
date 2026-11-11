# Provenance

crate is written from public specifications. It contains no code from other
emulators or hypervisors (QEMU, Henge, UTM or any other), and their source is
not consulted while working on crate.

## Sources

- Arm Architecture Reference Manual for A-profile architecture (DDI 0487):
  instructions, exceptions, system registers, the MMU, the generic timer.
- Arm Generic Interrupt Controller Architecture Specification, GICv3 and GICv4
  (IHI 0069).
- Arm Power State Coordination Interface (DEN 0022).
- Arm PrimeCell UART (PL011) Technical Reference Manual (DDI 0183).
- Arm PrimeCell Real Time Clock (PL031) Technical Reference Manual.
- OASIS Virtual I/O Device (VIRTIO) specification.
- PCI Express Base Specification (configuration space, ECAM).
- Devicetree Specification (devicetree.org).
- Linux's own documentation: `Documentation/arch/arm64/booting.rst` and the
  device-tree bindings under `Documentation/devicetree/bindings`.
- VESA E-EDID (1.4) and VESA Coordinated Video Timings (CVT 1.2), for the
  display's mode list and its timings; CTA-861 for the EDID's extension
  blocks.
- RFC 6143, The Remote Framebuffer Protocol, for the optional VNC server, and
  the X Window System's keysym definitions (keysymdef) for its key events.
- Linux's input event codes (`include/uapi/linux/input-event-codes.h`, a
  public interface), for virtio-input's events.
- The Khronos EGL 1.5 and OpenGL ES 3.x specifications and the EGL and GL
  extension specifications crate uses (KHR_surfaceless_context, KHR_debug,
  EXT_platform_base), and ANGLE's public EGL extension documents
  (EGL_ANGLE_platform_angle and its Vulkan and Metal variants).
- The public headers and documentation of libraries crate calls
  (virglrenderer's `virglrenderer.h`, libslirp, Apple's Hypervisor
  framework). virglrenderer's own source is not consulted; what its header
  leaves unsaid (the layout of `virgl_box`, which context it expects current)
  was found by experiment.
- JJTech's write-up on JIT with a debugger on iOS
  (https://gist.github.com/JJTech0130/142aee0f7bda9c61a421140d17afbdeb), for
  the order in which iOS executable memory and its writable alias are mapped
  and handed to the debugger.

A guest's own source code may be read to learn what that guest expects of a
device (how Linux drives a PL011, say). It is never copied.

## Dependencies

Libraries are used as dependencies, through their published interfaces, and
listed with their licenses in the workspace `Cargo.toml`. Only licenses that
combine with MPL-2.0 and with GPL-2.0-or-later are accepted (MIT, BSD,
Apache-2.0 OR BSD-3-Clause dual licenses, MPL-2.0, ISC, zlib).

Today:

| Dependency | License | For |
|---|---|---|
| libc | MIT OR Apache-2.0 | system calls |
| flate2, miniz_oxide, adler2, crc32fast, simd-adler32, cfg-if | MIT, or MIT among their choices | gzip-compressed kernels |
| libslirp (system library, linked dynamically) | BSD-3-Clause | user-mode networking |
| virglrenderer (loaded at run time if present) | MIT | 3D for virtio-gpu |
| an EGL, ANGLE's on Apple systems (loaded at run time if present) | BSD-3-Clause | the contexts virglrenderer renders on |

libslirp in turn links GLib (LGPL-2.1-or-later). Both stay separate dynamic
libraries, which the LGPL allows; crate's own code does not call GLib.

virglrenderer and EGL are not linked: crate opens them with dlopen when a
machine asks for 3D, and the app ships them. What they load in turn is the
app's choice too: virglrenderer uses libepoxy (MIT), and ANGLE on Vulkan
uses MoltenVK (Apache-2.0) on Apple systems.

## Checking

Before a release, crate's sources are compared against the projects named
above for shared lines, and anything found is rewritten or explained here.
