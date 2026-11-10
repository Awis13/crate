#!/bin/sh
# Builds crate-run and signs it, ad hoc, with the entitlement Apple's
# Hypervisor framework requires. Arguments go to cargo build (--release, say).
# Prints the path of the signed binary.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
profile=debug
for arg in "$@"; do
    [ "$arg" = "--release" ] && profile=release
done
cargo build --manifest-path "$root/Cargo.toml" -p crate-run "$@" >&2
binary="$root/target/$profile/crate-run"
codesign --sign - --force --entitlements "$here/crate-run.entitlements" "$binary" 2>/dev/null
echo "$binary"
