#!/bin/sh
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Builds crate-boot, a C program that boots a guest through crate's C API, and
# links it against the static library (release, as debug is far too slow to
# boot anything). Signs it ad hoc with the hypervisor entitlement so it can
# compare with hvf. Prints the path of the binary.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
cargo build --manifest-path "$root/Cargo.toml" -p crate-ffi --release >&2
slirp=${SLIRP_LIB_DIR:-$(pkg-config --variable=libdir slirp 2>/dev/null || echo /opt/homebrew/lib)}
binary="$root/target/release/crate-boot"
cc -O2 -Wall -Wextra -I "$root/include" -o "$binary" "$here/boot.c" \
    "$root/target/release/libcrate_ffi.a" -L "$slirp" -lslirp -framework Hypervisor -lpthread
codesign --sign - --force --entitlements "$here/../crate-run/crate-run.entitlements" "$binary" 2>/dev/null
echo "$binary"
