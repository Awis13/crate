// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Finds libslirp for the `slirp` feature: pkg-config's answer, else
//! Homebrew's prefix.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SLIRP_LIB_DIR");
    if std::env::var_os("CARGO_FEATURE_SLIRP").is_none() {
        return;
    }
    let dir = std::env::var("SLIRP_LIB_DIR").ok().or_else(|| {
        let out = Command::new("pkg-config")
            .args(["--variable=libdir", "slirp"])
            .output()
            .ok()?;
        let dir = String::from_utf8(out.stdout).ok()?.trim().to_string();
        (out.status.success() && !dir.is_empty()).then_some(dir)
    });
    let dir = dir.unwrap_or_else(|| "/opt/homebrew/lib".to_string());
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=slirp");
}
