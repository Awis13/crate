// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The network behind a network device: Ethernet frames in and out.

use std::sync::Arc;

/// Where a backend delivers frames for the guest. Called on the backend's own
/// thread; a device that has nowhere to put a frame drops it, as a NIC with a
/// full ring does.
pub trait FrameSink: Send + Sync {
    fn deliver(&self, frame: &[u8]);
}

pub trait NetBackend: Send + Sync {
    /// Starts delivering frames to `sink`. Called once, when the device is
    /// built.
    fn start(&self, sink: Arc<dyn FrameSink>);
    /// A frame from the guest. Must not block for long: it runs on a vCPU.
    fn send(&self, frame: &[u8]);
}
