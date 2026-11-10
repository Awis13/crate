// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! What the engine needs from its host: guest RAM, disk files and user-mode
//! networking so far; executable memory and the GPU later.

pub mod disk;
pub mod ram;
#[cfg(feature = "slirp")]
pub mod slirp;
