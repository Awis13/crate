# Contributing

crate is early: the C API is a draft until 1.0 (see
[docs/PLAN.md](docs/PLAN.md#milestone-7-firming-up-the-api)), and what is left
is listed in the same file. Issues and pull requests are welcome.

## Public specifications only

crate is written from public specifications, listed in
[docs/PROVENANCE.md](docs/PROVENANCE.md), and contributions are held to the
same rule:

- Don't copy code from other emulators or hypervisors (QEMU, Henge, UTM or any
  other), and don't read their source while working on crate.
- A guest's own source may be read to learn what that guest expects of a
  device. It is never copied.
- If a change rests on a source PROVENANCE.md doesn't list yet, name it in the
  pull request so it can be added.
- A new dependency needs a license that combines with MPL-2.0 and with
  GPL-2.0-or-later (PROVENANCE.md lists them), noted beside it in the
  workspace `Cargo.toml`.

Contributions are under the [MPL-2.0](LICENSE), and each new file begins with
its header.

## Before a pull request

On an Apple-silicon Mac, with libslirp installed (`brew install libslirp`):

    cargo fmt --all
    cargo clippy --workspace --all-targets -- -D warnings
    cargo clippy --workspace --all-targets --target aarch64-apple-ios -- -D warnings
    cargo test

CI runs the formatting and both lints. A change to the translator should also
pass, where you can run them:

- the random instruction streams in `crates/crate-jit/tests/el0_random.rs`,
  more of them with `ITERS`;
- `tools/el1-diff.sh`, which compares `jit` with `hvf` on small bare-metal
  programs;
- the arm64 kvm-unit-tests, through `tools/kvm-unit-tests.py`;
- and a guest booted on `jit` with `tools/crate-run`.

[docs/DESIGN.md](docs/DESIGN.md) describes how the pieces fit. Commit messages
start with the part they change, as in `crate-jit: ...` or `docs: ...`.
