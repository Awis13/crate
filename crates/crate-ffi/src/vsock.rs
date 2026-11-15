// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The vsock entry points: a virtio socket device, and streams to and from
//! the guest as the app's file descriptors.

use std::ffi::{c_int, c_void};
use std::os::fd::IntoRawFd;
use std::sync::Arc;
use std::time::Duration;

use crate::config::{Config, with};
use crate::machine::Machine;
use crate::{Context, Status, fail, guarded};

pub type AcceptFn = unsafe extern "C" fn(*mut c_void, u32, c_int);

/// `crate_config_add_vsock`.
///
/// # Safety
///
/// `config` is NULL or a live config.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_add_vsock(config: *mut Config, guest_cid: u64) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            if guest_cid < 3 || guest_cid >= u32::MAX as u64 {
                return fail(
                    Status::InvalidArgument,
                    format!("vsock: guest context ID {guest_cid} is reserved"),
                );
            }
            if c.vsock.is_some() {
                return fail(
                    Status::InvalidArgument,
                    "vsock: the machine has one already",
                );
            }
            c.vsock = Some(guest_cid);
            Status::Ok
        })
    }
}

/// The machine's vsock, or a failure saying why not.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
unsafe fn handle<'a>(
    machine: *mut Machine,
) -> Result<&'a crate_devices::virtio::vsock::VsockHandle, Status> {
    // SAFETY: the caller's contract.
    let m = unsafe { machine.as_ref() }
        .ok_or_else(|| fail(Status::InvalidArgument, "the machine is NULL"))?;
    m.virt
        .vsock()
        .ok_or_else(|| fail(Status::State, "the machine has no vsock"))
}

/// `crate_vsock_connect`.
///
/// # Safety
///
/// `machine` is NULL or a live machine; `fd` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_vsock_connect(
    machine: *mut Machine,
    port: u32,
    timeout_ms: u32,
    fd: *mut c_int,
) -> Status {
    guarded(|| {
        if fd.is_null() {
            return fail(Status::InvalidArgument, "fd is NULL");
        }
        // SAFETY: the caller's contract.
        let vsock = match unsafe { handle(machine) } {
            Ok(v) => v,
            Err(s) => return s,
        };
        match vsock.connect(port, Duration::from_millis(timeout_ms as u64)) {
            Ok(stream) => {
                // SAFETY: the caller's contract.
                unsafe { *fd = stream.into_raw_fd() };
                Status::Ok
            }
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                fail(Status::Timeout, e.to_string())
            }
            Err(e) => fail(Status::Io, e.to_string()),
        }
    })
}

/// `crate_vsock_listen`.
///
/// # Safety
///
/// `machine` is NULL or a live machine; `accept` and `context` are the app's
/// and stay valid until the port is unlistened or the machine freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_vsock_listen(
    machine: *mut Machine,
    port: u32,
    accept: Option<AcceptFn>,
    context: *mut c_void,
) -> Status {
    guarded(|| {
        let Some(accept) = accept else {
            return fail(Status::InvalidArgument, "the accept callback is NULL");
        };
        // SAFETY: the caller's contract.
        let vsock = match unsafe { handle(machine) } {
            Ok(v) => v,
            Err(s) => return s,
        };
        let context = Context(context);
        let acceptor = Arc::new(move |port: u32, stream: std::os::fd::OwnedFd| {
            // SAFETY: the app's callback, with its own context; it owns the fd.
            unsafe { accept(context.get(), port, stream.into_raw_fd()) };
        });
        match vsock.listen(port, acceptor) {
            Ok(()) => Status::Ok,
            Err(e) => fail(Status::State, e.to_string()),
        }
    })
}

/// `crate_vsock_unlisten`.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_vsock_unlisten(machine: *mut Machine, port: u32) -> Status {
    guarded(|| {
        // SAFETY: the caller's contract.
        match unsafe { handle(machine) } {
            Ok(v) => {
                v.unlisten(port);
                Status::Ok
            }
            Err(s) => s,
        }
    })
}
