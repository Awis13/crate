// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Whether the machine is still running, and why it stopped.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// Mirrors `crate_stop_reason`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The guest turned itself off.
    PowerOff,
    /// The guest asked to restart; starting again is the app's choice.
    Reset,
    /// The app stopped it.
    Killed,
    /// The engine could not go on.
    Error(String),
}

type Hook = Box<dyn Fn() + Send + Sync>;

pub struct RunControl {
    stopped: AtomicBool,
    reason: Mutex<Option<StopReason>>,
    cond: Condvar,
    hooks: Mutex<Vec<Hook>>,
}

impl Default for RunControl {
    fn default() -> Self {
        Self::new()
    }
}

impl RunControl {
    pub fn new() -> Self {
        RunControl {
            stopped: AtomicBool::new(false),
            reason: Mutex::new(None),
            cond: Condvar::new(),
            hooks: Mutex::new(Vec::new()),
        }
    }

    /// Stops the machine. The first reason given is the one that stands.
    pub fn stop(&self, reason: StopReason) {
        {
            let mut current = self.reason.lock().unwrap();
            if current.is_some() {
                return;
            }
            *current = Some(reason);
            self.stopped.store(true, Ordering::Release);
        }
        self.cond.notify_all();
        for hook in self.hooks.lock().unwrap().iter() {
            hook();
        }
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    pub fn reason(&self) -> Option<StopReason> {
        self.reason.lock().unwrap().clone()
    }

    /// Waits for the machine to stop, for at most `timeout` if given.
    pub fn wait(&self, timeout: Option<Duration>) -> Option<StopReason> {
        let guard = self.reason.lock().unwrap();
        let guard = match timeout {
            Some(t) => {
                self.cond
                    .wait_timeout_while(guard, t, |r| r.is_none())
                    .unwrap()
                    .0
            }
            None => self.cond.wait_while(guard, |r| r.is_none()).unwrap(),
        };
        guard.clone()
    }

    /// Runs `hook` when the machine stops: how sleeping threads hear of it.
    pub fn on_stop(&self, hook: impl Fn() + Send + Sync + 'static) {
        self.hooks.lock().unwrap().push(Box::new(hook));
        if self.is_stopped() {
            if let Some(hook) = self.hooks.lock().unwrap().last() {
                hook();
            }
        }
    }
}
