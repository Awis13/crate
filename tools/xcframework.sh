#!/bin/sh
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Builds crate's C API (crate-ffi) as a static library for iPhones, Simulators
# on Apple silicon and Macs on arm64, and packages them as Crate.xcframework,
# with include/crate.h and a module map, so Swift can `import CrateEngine`.
# The releases (.github/workflows/release.yml) are this, zipped.
#
#     tools/xcframework.sh [OUTPUT]     # default target/xcframework/Crate.xcframework
#
# The library leaves libslirp to the app: link slirp.0.framework on iOS (the
# app ships it; Homebrew's is a macOS library) and libslirp on macOS. On macOS
# it also needs the Hypervisor framework.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
out=${1:-$root/target/xcframework/Crate.xcframework}
stage=$root/target/xcframework/stage
# The oldest systems the app supports, so its linker does not warn.
export IPHONEOS_DEPLOYMENT_TARGET=${IPHONEOS_DEPLOYMENT_TARGET:-17.0}
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-13.0}

rm -rf "$stage" "$out"
# In a folder named for the module: Xcode copies every library's headers into
# one include/, where module maps at its top would collide.
mkdir -p "$stage/headers/CrateEngine"
cp "$root/include/crate.h" "$stage/headers/CrateEngine/"
cat > "$stage/headers/CrateEngine/module.modulemap" <<'MAP'
module CrateEngine {
    header "crate.h"
    export *
}
MAP
args=""
for target in aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin; do
    cargo build --manifest-path "$root/Cargo.toml" -p crate-ffi --release --target "$target" >&2
    mkdir -p "$stage/$target"
    cp "$root/target/$target/release/libcrate_ffi.a" "$stage/$target/libcrate.a"
    args="$args -library $stage/$target/libcrate.a -headers $stage/headers"
done
mkdir -p "$(dirname "$out")"
# shellcheck disable=SC2086
xcodebuild -create-xcframework $args -output "$out" >&2
echo "$out"
