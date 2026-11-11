// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! crate's C API, declared in `include/crate.h` and built as a static library
//! for apps to link (`tools/xcframework.sh` packages it as `libcrate.a`).
//!
//! Every function checks the pointers it is given, turns the engine's errors
//! into a `crate_status` and leaves the message for `crate_last_error`, and
//! keeps a panic from crossing into C. Enumerations arrive as plain integers
//! and are checked, since C may pass any value.

use std::cell::RefCell;
use std::ffi::{CString, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate_core::Error;
use crate_core::log::Level;

mod config;
mod display;
mod machine;

/// Major version of the C API this library implements.
pub const API_VERSION_MAJOR: u32 = 0;
/// Minor version of the C API this library implements.
pub const API_VERSION_MINOR: u32 = 1;

/// The version of the library actually linked: `(major << 16) | minor`.
#[unsafe(no_mangle)]
pub extern "C" fn crate_api_version() -> u32 {
    (API_VERSION_MAJOR << 16) | API_VERSION_MINOR
}

/// `crate_status`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok = 0,
    InvalidArgument,
    Unsupported,
    NoMemory,
    Io,
    Jit,
    Guest,
    State,
    Timeout,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Records `message` as this thread's last error and returns `status`.
pub(crate) fn fail(status: Status, message: impl Into<String>) -> Status {
    let mut message = message.into();
    message.retain(|c| c != '\0');
    let message = CString::new(message).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = message);
    status
}

/// The status an engine error stands for, its message recorded.
pub(crate) fn fail_with(error: Error) -> Status {
    let status = match &error {
        Error::InvalidArgument(_) => Status::InvalidArgument,
        Error::Unsupported(_) | Error::Backend(_) => Status::Unsupported,
        Error::NoMemory(_) => Status::NoMemory,
        Error::Io { .. } => Status::Io,
        Error::Guest(_) => Status::Guest,
        Error::Jit(_) => Status::Jit,
    };
    fail(status, error.to_string())
}

/// What a caught panic said.
pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic".into())
}

/// Runs an entry point's body, turning a panic into a failure.
pub(crate) fn guarded(body: impl FnOnce() -> Status) -> Status {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|payload| {
        fail(
            Status::State,
            format!("crate failed inside: {}", panic_message(&*payload)),
        )
    })
}

/// What the calling thread's last failed call went wrong on.
#[unsafe(no_mangle)]
pub extern "C" fn crate_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// An app's context pointer, handed back to its callbacks on the engine's
/// threads. The app promised, by passing it, that this is allowed.
#[derive(Clone, Copy)]
pub(crate) struct Context(pub *mut c_void);

impl Context {
    /// The pointer. A method, so a closure captures the whole (Send) value
    /// rather than the bare pointer.
    pub(crate) fn get(self) -> *mut c_void {
        self.0
    }
}

// SAFETY: crate.h tells apps their callbacks run on the engine's threads; the
// pointer is only ever passed back to them.
unsafe impl Send for Context {}
// SAFETY: as for Send.
unsafe impl Sync for Context {}

pub type LogFn = unsafe extern "C" fn(*mut c_void, c_int, *const c_char);

/// `crate_set_log_callback`.
///
/// # Safety
///
/// `log`, if given, must be callable from any thread with `context` until it
/// is replaced.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_set_log_callback(
    log: Option<LogFn>,
    context: *mut c_void,
    max: c_int,
) {
    let max = match max {
        i32::MIN..=0 => Level::Error,
        1 => Level::Warn,
        2 => Level::Info,
        _ => Level::Debug,
    };
    let sink = log.map(|log| {
        let context = Context(context);
        Box::new(move |level: Level, message: &str| {
            let mut message = message.to_string();
            message.retain(|c| c != '\0');
            let message = CString::new(message).unwrap_or_default();
            // SAFETY: the app's callback, with its own context.
            unsafe { log(context.get(), level as c_int, message.as_ptr()) };
        }) as crate_core::log::Sink
    });
    crate_core::log::set_sink(sink, max);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn last_error_is_per_thread() {
        assert_eq!(fail(Status::Io, "on this thread"), Status::Io);
        // SAFETY: the string is valid until this thread's next failure.
        let here = unsafe { CStr::from_ptr(crate_last_error()) };
        assert_eq!(here.to_str().unwrap(), "on this thread");
        std::thread::spawn(|| {
            // SAFETY: as above.
            let there = unsafe { CStr::from_ptr(crate_last_error()) };
            assert!(there.to_bytes().is_empty());
        })
        .join()
        .unwrap();
    }

    #[test]
    fn statuses_follow_the_errors() {
        assert_eq!(fail_with(Error::Jit("x".into())), Status::Jit);
        assert_eq!(
            fail_with(Error::io("x", std::io::Error::other("y"))),
            Status::Io
        );
        assert_eq!(guarded(|| panic!("boom")), Status::State);
        // SAFETY: as above.
        let message = unsafe { CStr::from_ptr(crate_last_error()) };
        assert!(message.to_str().unwrap().contains("boom"));
        assert_eq!(crate_api_version(), 1);
    }
}
