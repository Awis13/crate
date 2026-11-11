#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Runs the arm64 kvm-unit-tests under crate-run and scores them.

kvm-unit-tests is a separate GPL project; build it yourself (its `.flat` images
and `arm/unittests.cfg` are the inputs here) and point this script at the build
directories. This drives our own `crate-run` on each image -- it does not run
any kvm-unit-tests script -- and reads the `EXIT: STATUS=` line it prints:
1 = PASS, 3 = FAIL, 77 = SKIP, 127 = abort.

Every test runs on the jit backend at 4 KiB and 16 KiB pages and on hvf (the
reference) at 4 KiB, with the number of CPUs its configuration asks for
($MAX_SMP is the host's CPU count, or --max-smp). gicv2-* (the machine has
only a GICv3) and the *-migration tests (they block waiting to be migrated)
are scored "n/a".

Example:

    tools/crate-run/build.sh                      # a signed crate-run, for hvf
    tools/kvm-unit-tests.py --kut ~/src/kvm-unit-tests
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time

STATUS = {1: "PASS", 3: "FAIL", 77: "SKIP"}


def parse_cfg(path):
    tests, cur = [], None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"\[(.+)\]", line)
        if m:
            cur = {"name": m.group(1)}
            tests.append(cur)
        elif cur is not None and "=" in line:
            k, v = line.split("=", 1)
            cur[k.strip()] = v.strip().strip("'\"")
    return tests


def cpu_count(t, max_smp):
    """The CPUs a test asks for: a number, $MAX_SMP, or the configuration's
    $((($MAX_SMP < N)?$MAX_SMP:N)) form."""
    smp = t.get("smp", "1").replace("$MAX_SMP", str(max_smp))
    m = re.fullmatch(r"\$\(\(\(?\s*(\d+)\s*<\s*(\d+)\s*\)?\s*\?\s*(\d+)\s*:\s*(\d+)\s*\)\)", smp)
    if m:
        a, b, x, y = map(int, m.groups())
        return x if a < b else y
    return int(smp)


def not_applicable(t):
    # Only a GICv3 is modelled, and the migration tests (the "migration"
    # group) block for a migration that never comes.
    name = t["name"]
    return name.startswith("gicv2-") or "migration" in t.get("groups", "").split()


def run(crate_run, flat, args, backend, env, timeout, cpus):
    cmd = [
        crate_run, "--raw", "--backend", backend, "--kernel", flat,
        "--initrd", env, "--cmdline", args, "--memory", "256", "--log", "warn",
        "--cpus", str(cpus),
    ]
    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError as e:
        return "ERR", str(e)
    start = time.time()
    while time.time() - start < timeout and p.poll() is None:
        time.sleep(0.1)
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except OSError:
        pass
    text = (p.stdout.read() or b"").decode(errors="replace")
    m = re.search(r"EXIT: STATUS=(\d+)", text)
    if m:
        return STATUS.get(int(m.group(1)), f"STATUS{m.group(1)}"), text
    return "HANG", text


def main():
    ap = argparse.ArgumentParser(description="Run kvm-unit-tests under crate-run.")
    ap.add_argument("--kut", help="kvm-unit-tests checkout (has arm/unittests.cfg)")
    ap.add_argument("--build-4k", help="4 KiB build dir (default <kut>/build-4k)")
    ap.add_argument("--build-16k", help="16 KiB build dir (default <kut>/build-16k)")
    ap.add_argument("--cfg", help="unittests.cfg (default <kut>/arm/unittests.cfg)")
    ap.add_argument("--crate-run", default=os.path.expanduser(
        "~/Repos/crate/target/debug/crate-run"), help="the signed crate-run binary")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--only", help="run only tests whose name contains this")
    ap.add_argument("--max-smp", type=int, default=os.cpu_count(),
                    help="$MAX_SMP (default: the host's CPU count)")
    args = ap.parse_args()

    build4 = args.build_4k or (args.kut and os.path.join(args.kut, "build-4k"))
    build16 = args.build_16k or (args.kut and os.path.join(args.kut, "build-16k"))
    cfg = args.cfg or (args.kut and os.path.join(args.kut, "arm/unittests.cfg"))
    if not (build4 and build16 and cfg):
        ap.error("give --kut, or --build-4k/--build-16k/--cfg")

    # An initrd with the environment kvm-unit-tests reads (ERRATA_FORCE=y turns
    # on every erratum workaround, as the suite does for tcg).
    import tempfile
    env = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
    env.write("ERRATA_FORCE=y\n")
    env.close()

    tests = parse_cfg(cfg)
    print(f"# {len(tests)} tests; $MAX_SMP = {args.max_smp}\n")
    hdr = f"{'test':34} {'cpus':>4}  {'jit-4k':10} {'jit-16k':10} {'hvf':10}"
    print(hdr)
    print("-" * len(hdr))

    tallies = {"jit-4k": {}, "jit-16k": {}, "hvf": {}}
    for t in tests:
        name = t["name"]
        if args.only and args.only not in name:
            continue
        if not_applicable(t):
            print(f"{name:34} {'':>4}  n/a (only GICv3 / migration)")
            continue
        cpus = cpu_count(t, args.max_smp)
        flat, cmdline = t["file"], t.get("test_args", "")
        res = {}
        for label, backend, bdir in [("jit-4k", "jit", build4),
                                     ("jit-16k", "jit", build16),
                                     ("hvf", "hvf", build4)]:
            path = os.path.join(bdir, "arm", flat)
            if not os.path.exists(path):
                res[label] = "no-build"
            else:
                res[label], _ = run(args.crate_run, path, cmdline, backend, env.name,
                                    args.timeout, cpus)
            tallies[label][res[label]] = tallies[label].get(res[label], 0) + 1
        print(f"{name:34} {cpus:>4}  {res['jit-4k']:10} {res['jit-16k']:10} {res['hvf']:10}")

    os.unlink(env.name)
    print()
    for label in ("jit-4k", "jit-16k", "hvf"):
        counts = ", ".join(f"{k}={v}" for k, v in sorted(tallies[label].items()))
        print(f"# {label}: {counts}")


if __name__ == "__main__":
    main()
