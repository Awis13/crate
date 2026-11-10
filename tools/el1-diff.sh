#!/bin/sh
# Differential test of the jit backend against hvf (the oracle) on small EL1
# bare-metal programs: assembles each `.s` under crates/crate-jit/tests/el1,
# runs the same image on both backends through the signed crate-run, and checks
# that the console output matches. The programs exercise MMIO, the MMU at 4 KiB
# and 16 KiB, exceptions and ERET, system registers, the generic timer and the
# GIC. See docs/DESIGN.md ("Testing the translator").
#
# Needs an aarch64 bare-metal toolchain (default prefix aarch64-elf-, override
# with CROSS=...) and a Mac (hvf). Run from anywhere.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
cross=${CROSS:-aarch64-elf-}
tests="$root/crates/crate-jit/tests/el1"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

run="$("$root/tools/crate-run/build.sh")"

fail=0
for src in "$tests"/*.s; do
    name=$(basename "$src" .s)
    "${cross}gcc" -c -nostdlib "$src" -o "$work/$name.o"
    "${cross}ld" -Ttext=0x40200000 "$work/$name.o" -o "$work/$name.elf"
    "${cross}objcopy" -O binary "$work/$name.elf" "$work/$name.flat"

    for backend in jit hvf; do
        "$run" --raw --backend "$backend" --kernel "$work/$name.flat" \
            --memory 256 --log warn </dev/null >"$work/$name.$backend" 2>&1 &
        pid=$!
        # The programs power off on their own; kill a stuck one after 6 s. The
        # guard reaps a dead pid silently, so it leaves no job-control noise.
        ( sleep 6; kill -9 "$pid" >/dev/null 2>&1 ) >/dev/null 2>&1 &
        wait "$pid" 2>/dev/null || true
    done

    if diff -q "$work/$name.jit" "$work/$name.hvf" >/dev/null; then
        printf 'ok    %-8s %s\n' "$name" "$(tr -d '\n\r' <"$work/$name.jit")"
    else
        printf 'FAIL  %s\n' "$name"
        echo "  jit: $(tr -d '\n\r' <"$work/$name.jit")"
        echo "  hvf: $(tr -d '\n\r' <"$work/$name.hvf")"
        fail=1
    fi
done

exit "$fail"
