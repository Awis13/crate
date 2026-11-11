// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The engine's error type. Its kinds follow `crate_status` in the C API, so
//! the FFI layer can map one to the other without guessing.

use std::fmt;
use std::io;

#[derive(Debug)]
pub enum Error {
    InvalidArgument(String),
    /// Not on this host, or not in this build.
    Unsupported(String),
    NoMemory(String),
    /// A file could not be opened, read or written.
    Io {
        context: String,
        source: io::Error,
    },
    /// The kernel or another guest file is unusable.
    Guest(String),
    /// No executable memory could be had for the translator.
    Jit(String),
    /// The CPU backend failed: a hypervisor call, say.
    Backend(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn io(context: impl Into<String>, source: io::Error) -> Self {
        Error::Io {
            context: context.into(),
            source,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::NoMemory(m) => write!(f, "out of memory: {m}"),
            Error::Io { context, source } => write!(f, "{context}: {source}"),
            Error::Guest(m) => write!(f, "guest: {m}"),
            Error::Jit(m) => write!(f, "no executable memory: {m}"),
            Error::Backend(m) => write!(f, "CPU backend: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
