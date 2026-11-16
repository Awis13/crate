// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `crate_config`: a machine's description, kept as the app gives it and
//! checked as each piece arrives, so a mistake fails at the call that made it.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::net::Ipv4Addr;
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;

use crate_host::slirp::PortForward;
use crate_virt::ConsoleKind;
use crate_virt::layout::{MAX_CPUS, MAX_MEMORY, MIN_MEMORY};

use crate::{Context, Status, fail, guarded};

/// `crate_cpu_backend`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Backend {
    Auto,
    Jit,
    Hvf,
}

pub(crate) const DISK_READ_ONLY: u32 = 1 << 0;
pub(crate) const DISK_FULL_FLUSH: u32 = 1 << 1;
pub(crate) const DISK_NO_FLUSH: u32 = 1 << 2;
pub(crate) const DISK_DISCARD: u32 = 1 << 3;

pub(crate) struct Disk {
    pub path: PathBuf,
    pub flags: u32,
}

#[derive(Default)]
pub(crate) struct Network {
    pub mac: Option<[u8; 6]>,
    pub ipv6: bool,
    pub forwards: Vec<PortForward>,
}

pub type OutputFn = unsafe extern "C" fn(*mut c_void, u32, *const u8, usize);
pub type InputReadyFn = unsafe extern "C" fn(*mut c_void, u32);

/// `crate_console_callbacks`, as laid out in crate.h.
#[repr(C)]
pub struct ConsoleCallbacks {
    size: usize,
    context: *mut c_void,
    output: Option<OutputFn>,
    input_ready: Option<InputReadyFn>,
}

/// An app's console callbacks, copied out of its struct.
#[derive(Clone, Copy)]
pub(crate) struct Console {
    pub kind: ConsoleKind,
    pub context: Context,
    pub output: Option<OutputFn>,
    pub input_ready: Option<InputReadyFn>,
}

pub type StateFn = unsafe extern "C" fn(*mut c_void, c_int, c_int);

#[derive(Clone, Copy)]
pub(crate) struct StateCallback {
    pub call: StateFn,
    pub context: Context,
}

/// Executable memory the app supplies.
#[derive(Clone, Copy)]
pub(crate) struct AppRegion {
    pub exec: usize,
    pub write: usize,
    pub size: usize,
}

pub struct Config {
    pub(crate) backend: Backend,
    pub(crate) cpus: usize,
    pub(crate) memory: u64,
    pub(crate) jit_region: Option<AppRegion>,
    pub(crate) jit_size: Option<usize>,
    pub(crate) kernel: Option<PathBuf>,
    pub(crate) initrd: Option<PathBuf>,
    pub(crate) cmdline: String,
    pub(crate) disks: Vec<Disk>,
    /// Empty disks after `disks`, for images put in while the machine runs.
    pub(crate) disk_slots: u32,
    pub(crate) networks: Vec<Network>,
    pub(crate) consoles: Vec<Console>,
    pub(crate) displays: Vec<crate::display::Display>,
    /// CRATE_INPUT_* bits.
    pub(crate) input: u32,
    pub(crate) state: Option<StateCallback>,
    /// The guest context ID of a vsock device.
    pub(crate) vsock: Option<u64>,
}

/// The config behind a pointer, or INVALID_ARGUMENT.
///
/// # Safety
///
/// `config` is NULL or came from `crate_config_new` and was not freed.
unsafe fn config_mut<'a>(config: *mut Config) -> Result<&'a mut Config, Status> {
    // SAFETY: the caller's contract.
    unsafe { config.as_mut() }.ok_or_else(|| fail(Status::InvalidArgument, "the config is NULL"))
}

/// A C string the app passed, or INVALID_ARGUMENT naming `what`.
///
/// # Safety
///
/// `s` is NULL or a NUL-terminated string.
pub(crate) unsafe fn c_str<'a>(s: *const c_char, what: &str) -> Result<&'a CStr, Status> {
    if s.is_null() {
        return Err(fail(Status::InvalidArgument, format!("the {what} is NULL")));
    }
    // SAFETY: the caller's contract.
    Ok(unsafe { CStr::from_ptr(s) })
}

/// # Safety
///
/// As for `c_str`.
unsafe fn path(s: *const c_char, what: &str) -> Result<PathBuf, Status> {
    // SAFETY: the caller's contract.
    let s = unsafe { c_str(s, what)? };
    if s.is_empty() {
        return Err(fail(
            Status::InvalidArgument,
            format!("the {what} is empty"),
        ));
    }
    Ok(PathBuf::from(std::ffi::OsStr::from_bytes(s.to_bytes())))
}

/// Runs `body` on the config, for the entry points below.
///
/// # Safety
///
/// As for `config_mut`.
pub(crate) unsafe fn with(config: *mut Config, body: impl FnOnce(&mut Config) -> Status) -> Status {
    guarded(|| {
        // SAFETY: the caller's contract.
        match unsafe { config_mut(config) } {
            Ok(config) => body(config),
            Err(status) => status,
        }
    })
}

/// `crate_config_new`.
///
/// # Safety
///
/// `machine` is NULL or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_new(machine: *const c_char) -> *mut Config {
    // SAFETY: the caller's contract.
    let name = match unsafe { c_str(machine, "machine type") } {
        Ok(name) => name,
        Err(_) => return std::ptr::null_mut(),
    };
    if name.to_bytes() != b"virt" {
        fail(
            Status::InvalidArgument,
            format!(
                "no machine type {:?}; \"virt\" is the only one",
                name.to_string_lossy()
            ),
        );
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(Config {
        backend: Backend::Auto,
        cpus: 1,
        memory: 1 << 30,
        jit_region: None,
        jit_size: None,
        kernel: None,
        initrd: None,
        cmdline: String::new(),
        disks: Vec::new(),
        disk_slots: 0,
        networks: Vec::new(),
        consoles: Vec::new(),
        displays: Vec::new(),
        input: 0,
        state: None,
        vsock: None,
    }))
}

/// `crate_config_free`.
///
/// # Safety
///
/// `config` is NULL or came from `crate_config_new` and was not freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_free(config: *mut Config) {
    if !config.is_null() {
        // SAFETY: the caller's contract; made by Box::into_raw.
        drop(unsafe { Box::from_raw(config) });
    }
}

/// `crate_config_set_cpu_backend`.
///
/// # Safety
///
/// As for `crate_config_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_set_cpu_backend(
    config: *mut Config,
    backend: c_int,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            c.backend = match backend {
                0 => Backend::Auto,
                1 => Backend::Jit,
                2 => Backend::Hvf,
                _ => return fail(Status::InvalidArgument, format!("no CPU backend {backend}")),
            };
            Status::Ok
        })
    }
}

/// `crate_config_set_cpus`.
///
/// # Safety
///
/// As for `crate_config_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_set_cpus(config: *mut Config, count: u32) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            if count == 0 || count as usize > MAX_CPUS {
                return fail(
                    Status::InvalidArgument,
                    format!("{count} CPUs; virt takes 1 to {MAX_CPUS}"),
                );
            }
            c.cpus = count as usize;
            Status::Ok
        })
    }
}

/// `crate_config_set_memory`.
///
/// # Safety
///
/// As for `crate_config_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_set_memory(config: *mut Config, bytes: u64) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            if !(MIN_MEMORY..=MAX_MEMORY).contains(&bytes) || bytes % (2 << 20) != 0 {
                return fail(
                    Status::InvalidArgument,
                    format!(
                        "{bytes} bytes of RAM; virt takes a multiple of 2 MiB from {} MiB to {} GiB",
                        MIN_MEMORY >> 20,
                        MAX_MEMORY >> 30
                    ),
                );
            }
            c.memory = bytes;
            Status::Ok
        })
    }
}

/// `crate_config_set_jit_region`.
///
/// # Safety
///
/// As for `crate_config_free`; the region is the app's to keep valid, as
/// crate.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_set_jit_region(
    config: *mut Config,
    executable: *mut c_void,
    writable: *mut c_void,
    size: usize,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            if executable.is_null() || writable.is_null() || size == 0 {
                return fail(
                    Status::InvalidArgument,
                    "the executable region needs both addresses and a size",
                );
            }
            if executable as usize % 4 != 0 || writable as usize % 4 != 0 {
                return fail(
                    Status::InvalidArgument,
                    "the executable region is not aligned",
                );
            }
            c.jit_region = Some(AppRegion {
                exec: executable as usize,
                write: writable as usize,
                size,
            });
            Status::Ok
        })
    }
}

/// `crate_config_set_jit_size`.
///
/// # Safety
///
/// As for `crate_config_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_set_jit_size(config: *mut Config, bytes: usize) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            if bytes == 0 {
                return fail(Status::InvalidArgument, "no executable memory asked for");
            }
            c.jit_size = Some(bytes);
            Status::Ok
        })
    }
}

/// `crate_config_set_kernel`.
///
/// # Safety
///
/// As for `crate_config_free`; `path` is NULL or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_set_kernel(
    config: *mut Config,
    path: *const c_char,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| match path_arg(path, "kernel path") {
            Ok(p) => {
                c.kernel = Some(p);
                Status::Ok
            }
            Err(status) => status,
        })
    }
}

/// `crate_config_set_initrd`.
///
/// # Safety
///
/// As for `crate_config_set_kernel`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_set_initrd(
    config: *mut Config,
    path: *const c_char,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| match path_arg(path, "initrd path") {
            Ok(p) => {
                c.initrd = Some(p);
                Status::Ok
            }
            Err(status) => status,
        })
    }
}

fn path_arg(s: *const c_char, what: &str) -> Result<PathBuf, Status> {
    // SAFETY: every caller passes the app's pointer under the same contract.
    unsafe { path(s, what) }
}

/// `crate_config_set_cmdline`.
///
/// # Safety
///
/// As for `crate_config_set_kernel`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_set_cmdline(
    config: *mut Config,
    cmdline: *const c_char,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| match c_str(cmdline, "command line") {
            Ok(s) => match s.to_str() {
                Ok(s) => {
                    c.cmdline = s.to_string();
                    Status::Ok
                }
                Err(_) => fail(Status::InvalidArgument, "the command line is not UTF-8"),
            },
            Err(status) => status,
        })
    }
}

/// `crate_config_add_disk`.
///
/// # Safety
///
/// As for `crate_config_set_kernel`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_add_disk(
    config: *mut Config,
    path: *const c_char,
    flags: u32,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            let known = DISK_READ_ONLY | DISK_FULL_FLUSH | DISK_NO_FLUSH | DISK_DISCARD;
            if flags & !known != 0 {
                return fail(
                    Status::InvalidArgument,
                    format!("unknown disk flags {flags:#x}"),
                );
            }
            if flags & DISK_FULL_FLUSH != 0 && flags & DISK_NO_FLUSH != 0 {
                return fail(Status::InvalidArgument, "FULL_FLUSH and NO_FLUSH together");
            }
            if flags & DISK_DISCARD != 0 {
                return fail(Status::Unsupported, "disks cannot discard yet");
            }
            match path_arg(path, "disk path") {
                Ok(path) => {
                    c.disks.push(Disk { path, flags });
                    Status::Ok
                }
                Err(status) => status,
            }
        })
    }
}

/// `crate_config_add_disk_slots`.
///
/// # Safety
///
/// As for `crate_config_set_kernel`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_add_disk_slots(config: *mut Config, count: u32) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            if c.disk_slots.saturating_add(count) > 16 {
                return fail(Status::InvalidArgument, "at most 16 disk slots");
            }
            c.disk_slots += count;
            Status::Ok
        })
    }
}

/// The network numbered `index`, or INVALID_ARGUMENT.
fn network(c: &mut Config, index: u32) -> Result<&mut Network, Status> {
    c.networks
        .get_mut(index as usize)
        .ok_or_else(|| fail(Status::InvalidArgument, format!("no network {index}")))
}

/// `crate_config_add_network`.
///
/// # Safety
///
/// As for `crate_config_free`; `index` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_add_network(config: *mut Config, index: *mut u32) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            c.networks.push(Network::default());
            if let Some(index) = index.as_mut() {
                *index = (c.networks.len() - 1) as u32;
            }
            Status::Ok
        })
    }
}

/// `crate_config_set_network_mac`.
///
/// # Safety
///
/// As for `crate_config_free`; `mac` points at six bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_set_network_mac(
    config: *mut Config,
    index: u32,
    mac: *const u8,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            if mac.is_null() {
                return fail(Status::InvalidArgument, "the MAC address is NULL");
            }
            let mut bytes = [0; 6];
            std::ptr::copy_nonoverlapping(mac, bytes.as_mut_ptr(), 6);
            if bytes[0] & 1 != 0 {
                return fail(Status::InvalidArgument, "a multicast MAC address");
            }
            match network(c, index) {
                Ok(n) => {
                    n.mac = Some(bytes);
                    Status::Ok
                }
                Err(status) => status,
            }
        })
    }
}

/// `crate_config_set_network_ipv6`.
///
/// # Safety
///
/// As for `crate_config_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_set_network_ipv6(
    config: *mut Config,
    index: u32,
    enabled: bool,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| match network(c, index) {
            Ok(n) => {
                n.ipv6 = enabled;
                Status::Ok
            }
            Err(status) => status,
        })
    }
}

/// `crate_config_add_port_forward`.
///
/// # Safety
///
/// As for `crate_config_free`; `host_address` is NULL or a NUL-terminated
/// string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_add_port_forward(
    config: *mut Config,
    index: u32,
    protocol: c_int,
    host_address: *const c_char,
    host_port: u16,
    guest_port: u16,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            let udp = match protocol {
                0 => false,
                1 => true,
                _ => return fail(Status::InvalidArgument, format!("no protocol {protocol}")),
            };
            let host_addr = if host_address.is_null() {
                Ipv4Addr::UNSPECIFIED
            } else {
                let s = CStr::from_ptr(host_address).to_string_lossy();
                if s.is_empty() {
                    Ipv4Addr::UNSPECIFIED
                } else {
                    match s.parse::<Ipv4Addr>() {
                        Ok(a) => a,
                        Err(_) => {
                            return fail(
                                Status::InvalidArgument,
                                format!("{s:?} is not an IPv4 address"),
                            );
                        }
                    }
                }
            };
            if host_port == 0 || guest_port == 0 {
                return fail(Status::InvalidArgument, "port 0");
            }
            match network(c, index) {
                Ok(n) => {
                    n.forwards.push(PortForward {
                        udp,
                        host_addr,
                        host_port,
                        guest_port,
                    });
                    Status::Ok
                }
                Err(status) => status,
            }
        })
    }
}

/// `crate_config_add_console`.
///
/// # Safety
///
/// As for `crate_config_free`; `callbacks` is NULL or a struct whose `size`
/// says how much of it there is; `index` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_add_console(
    config: *mut Config,
    kind: c_int,
    callbacks: *const ConsoleCallbacks,
    index: *mut u32,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            let kind = match kind {
                0 => ConsoleKind::Serial,
                1 => ConsoleKind::Virtio,
                _ => return fail(Status::InvalidArgument, format!("no console kind {kind}")),
            };
            if kind == ConsoleKind::Serial && c.consoles.iter().any(|c| c.kind == kind) {
                return fail(Status::InvalidArgument, "virt has one serial console");
            }
            let mut console = Console {
                kind,
                context: Context(std::ptr::null_mut()),
                output: None,
                input_ready: None,
            };
            if !callbacks.is_null() {
                // Only the fields the app's struct has: `size` says which.
                let size = (*callbacks).size;
                let has = |field_end: usize| size >= field_end;
                use std::mem::{offset_of, size_of};
                if has(offset_of!(ConsoleCallbacks, context) + size_of::<*mut c_void>()) {
                    console.context = Context((*callbacks).context);
                }
                if has(offset_of!(ConsoleCallbacks, output) + size_of::<usize>()) {
                    console.output = (*callbacks).output;
                }
                if has(offset_of!(ConsoleCallbacks, input_ready) + size_of::<usize>()) {
                    console.input_ready = (*callbacks).input_ready;
                }
            }
            c.consoles.push(console);
            if let Some(index) = index.as_mut() {
                *index = (c.consoles.len() - 1) as u32;
            }
            Status::Ok
        })
    }
}

/// `crate_config_set_state_callback`.
///
/// # Safety
///
/// As for `crate_config_free`; `call`, if given, may be called from any
/// thread with `context` while the machine lives.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_set_state_callback(
    config: *mut Config,
    call: Option<StateFn>,
    context: *mut c_void,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            c.state = call.map(|call| StateCallback {
                call,
                context: Context(context),
            });
            Status::Ok
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new() -> *mut Config {
        // SAFETY: a literal C string.
        unsafe { crate_config_new(c"virt".as_ptr()) }
    }

    #[test]
    fn checks_each_piece() {
        // SAFETY: a config made here, freed at the end; literal strings.
        unsafe {
            assert!(crate_config_new(c"pc".as_ptr()).is_null());
            let c = new();
            assert_eq!(crate_config_set_cpus(c, 0), Status::InvalidArgument);
            assert_eq!(crate_config_set_cpus(c, 4), Status::Ok);
            assert_eq!(
                crate_config_set_memory(c, (1 << 30) + 4096),
                Status::InvalidArgument
            );
            assert_eq!(crate_config_set_cpu_backend(c, 7), Status::InvalidArgument);
            assert_eq!(
                crate_config_add_disk(c, c"d.img".as_ptr(), 1 << 9),
                Status::InvalidArgument
            );
            assert_eq!(
                crate_config_add_disk(c, c"d.img".as_ptr(), DISK_FULL_FLUSH | DISK_NO_FLUSH),
                Status::InvalidArgument
            );
            assert_eq!(
                crate_config_add_disk(c, c"d.img".as_ptr(), DISK_READ_ONLY),
                Status::Ok
            );
            let mut net = 9;
            assert_eq!(crate_config_add_network(c, &mut net), Status::Ok);
            assert_eq!(net, 0);
            assert_eq!(
                crate_config_add_port_forward(c, 0, 0, std::ptr::null(), 2222, 22),
                Status::Ok
            );
            assert_eq!(
                crate_config_add_port_forward(c, 0, 0, c"::1".as_ptr(), 2222, 22),
                Status::InvalidArgument
            );
            assert_eq!(
                crate_config_add_port_forward(c, 1, 0, std::ptr::null(), 2222, 22),
                Status::InvalidArgument,
                "no network 1"
            );
            let config = &*c;
            assert_eq!(config.cpus, 4);
            assert_eq!(
                config.networks[0].forwards[0].host_addr,
                Ipv4Addr::UNSPECIFIED
            );
            let mut index = 9;
            assert_eq!(
                crate_config_add_console(c, 0, std::ptr::null(), &mut index),
                Status::Ok
            );
            assert_eq!(index, 0);
            assert_eq!(
                crate_config_add_console(c, 0, std::ptr::null(), &mut index),
                Status::InvalidArgument,
                "a second serial console"
            );
            assert_eq!(
                crate_config_add_console(c, 1, std::ptr::null(), &mut index),
                Status::Ok
            );
            assert_eq!(index, 1);
            assert_eq!(
                crate_config_set_kernel(std::ptr::null_mut(), c"k".as_ptr()),
                Status::InvalidArgument
            );
            crate_config_free(c);
        }
    }

    #[test]
    fn reads_only_the_callbacks_an_app_has() {
        unsafe extern "C" fn output(_: *mut c_void, _: u32, _: *const u8, _: usize) {}
        unsafe extern "C" fn never(_: *mut c_void, _: u32) {
            unreachable!()
        }
        // An app built when the struct ended after `output`, with whatever
        // follows it in memory: input_ready must not be read.
        let old = ConsoleCallbacks {
            size: std::mem::offset_of!(ConsoleCallbacks, input_ready),
            context: std::ptr::null_mut(),
            output: Some(output),
            input_ready: Some(never),
        };
        // SAFETY: a config made here and freed at the end.
        unsafe {
            let c = new();
            assert_eq!(
                crate_config_add_console(c, 1, &old, std::ptr::null_mut()),
                Status::Ok
            );
            let console = (&*c).consoles[0];
            assert!(console.output.is_some());
            assert!(console.input_ready.is_none());
            crate_config_free(c);
        }
    }
}
