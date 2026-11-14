# Security

An app may run guests it does not trust, so crate treats everything a guest
does as untrusted. A guest that can do any of these has found a security bug:

- crash the host process;
- read or write host memory outside its own guest RAM;
- run code of its choice on the host;
- reach what the app did not give it (other files, other network endpoints).

It may be through the translator, a device, or how the C API hands the
guest's output to the app.

What the app passes through the C API is trusted: a bad configuration is the
app's bug. A guest that hangs or stops only its own machine is not a security
issue either.

## Reporting

Report it privately through GitHub's private vulnerability reporting (the
repository's Security tab, "Report a vulnerability"), not in a public issue.
Say which guest, which host (Mac or iPhone, and its OS version) and which
backend (`jit` or `hvf`), and how to reproduce it.

crate has no stable releases before 1.0: fixes land on `main` and in the next
release.
