// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! crate's C API, declared in `include/crate.h` and built as a static library
//! for apps to link.

/// Major version of the C API this library implements.
pub const API_VERSION_MAJOR: u32 = 0;
/// Minor version of the C API this library implements.
pub const API_VERSION_MINOR: u32 = 1;

/// The version of the library actually linked: `(major << 16) | minor`.
#[unsafe(no_mangle)]
pub extern "C" fn crate_api_version() -> u32 {
    (API_VERSION_MAJOR << 16) | API_VERSION_MINOR
}
