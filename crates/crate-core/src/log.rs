// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Logging for the whole process, shaped like `crate_set_log_callback`: one
//! sink and a level below which messages are not made at all.

use std::fmt;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Level {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
}

pub type Sink = Box<dyn Fn(Level, &str) + Send + Sync>;

static MAX: AtomicU8 = AtomicU8::new(Level::Info as u8);
static SINK: RwLock<Option<Sink>> = RwLock::new(None);

/// Replaces the sink. `None` restores the default, which writes to stderr.
pub fn set_sink(sink: Option<Sink>, max: Level) {
    *SINK.write().unwrap_or_else(|e| e.into_inner()) = sink;
    MAX.store(max as u8, Ordering::Relaxed);
}

pub fn enabled(level: Level) -> bool {
    level as u8 <= MAX.load(Ordering::Relaxed)
}

pub fn write(level: Level, args: fmt::Arguments<'_>) {
    let message = args.to_string();
    match &*SINK.read().unwrap_or_else(|e| e.into_inner()) {
        Some(sink) => sink(level, &message),
        None => eprintln!("crate: {}: {message}", level.name()),
    }
}

impl Level {
    pub fn name(self) -> &'static str {
        match self {
            Level::Error => "error",
            Level::Warn => "warning",
            Level::Info => "info",
            Level::Debug => "debug",
        }
    }
}

#[macro_export]
macro_rules! log_at {
    ($level:expr, $($arg:tt)*) => {
        if $crate::log::enabled($level) {
            $crate::log::write($level, format_args!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => { $crate::log_at!($crate::log::Level::Error, $($arg)*) };
}

#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => { $crate::log_at!($crate::log::Level::Warn, $($arg)*) };
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => { $crate::log_at!($crate::log::Level::Info, $($arg)*) };
}

#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => { $crate::log_at!($crate::log::Level::Debug, $($arg)*) };
}
