// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The machine model every part of crate shares: guest memory, CPUs, buses and
//! interrupts, and the traits that machines, devices, CPU backends and host
//! backends implement. Nothing here knows which guest runs.
