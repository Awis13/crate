// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! What the engine needs from its host: guest RAM, disk files, user-mode
//! networking, executable memory for the translator and the host's counter,
//! and the GPU (virglrenderer over EGL, loaded at run time).

pub mod disk;
#[cfg(target_arch = "aarch64")]
pub mod jit;
pub mod ram;
#[cfg(feature = "slirp")]
pub mod slirp;
pub mod virgl;
