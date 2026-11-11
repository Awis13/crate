// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Runs a crate machine from the command line, for development.
//!
//! The serial console is the terminal: what the guest prints comes out on
//! stdout, and stdin (in raw mode when it is a terminal) goes to the guest.
//! Ctrl-A then x stops the machine; Ctrl-A twice sends one Ctrl-A.

use std::io::{Read, Write};
use std::net::Ipv4Addr;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate_core::log::Level;
use crate_core::run::StopReason;
use crate_core::{Error, Result};
use crate_host::disk::FileDisk;
use crate_host::slirp::{PortForward, UserNet};
use crate_virt::{ConsoleConfig, ConsoleKind, NetworkConfig, VirtConfig, VirtMachine};

const USAGE: &str = "\
usage: crate-run --kernel IMAGE [options]

  --kernel PATH            arm64 Linux Image, raw or gzip-compressed
  --initrd PATH            initial ramdisk
  --cmdline STRING         kernel command line (default: console=ttyAMA0)
  --disk PATH[,ro]         a virtio disk; repeat for vdb, vdc, ...
  --cpus N                 default 1
  --memory MIB             default 1024
  --net                    user-mode networking (10.0.2.15, host 10.0.2.2)
  --forward [udp:|tcp:][ADDR:]HOSTPORT:GUESTPORT
                           forward a host port into the guest (implies --net;
                           ADDR defaults to 127.0.0.1)
  --virtio-console PATH    a virtio console (hvc0) on a Unix socket at PATH
  --dump-dtb PATH          write the device tree to PATH and exit (as text if
                           PATH ends in .dts)
  --log LEVEL              error, warn, info (default) or debug

The serial console (ttyAMA0) is this terminal. Ctrl-A x quits.";

struct Args {
    kernel: PathBuf,
    initrd: Option<PathBuf>,
    cmdline: String,
    disks: Vec<(PathBuf, bool)>,
    cpus: usize,
    memory_mib: u64,
    net: bool,
    forwards: Vec<PortForward>,
    virtio_console: Option<PathBuf>,
    dump_dtb: Option<PathBuf>,
    log: Level,
    raw: bool,
    backend: Backend,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// Apple's Hypervisor framework (macOS, the default there).
    Hvf,
    /// The translator.
    Jit,
}

fn parse_forward(spec: &str) -> std::result::Result<PortForward, String> {
    let (udp, rest) = match spec.split_once(':') {
        Some(("udp", rest)) => (true, rest),
        Some(("tcp", rest)) => (false, rest),
        _ => (false, spec),
    };
    let parts: Vec<&str> = rest.split(':').collect();
    let (addr, host, guest) = match parts.as_slice() {
        [host, guest] => ("127.0.0.1", *host, *guest),
        [addr, host, guest] => (*addr, *host, *guest),
        _ => {
            return Err(format!(
                "--forward {spec}: expected [ADDR:]HOSTPORT:GUESTPORT"
            ));
        }
    };
    let port = |p: &str| {
        p.parse::<u16>()
            .map_err(|_| format!("--forward {spec}: bad port {p}"))
    };
    Ok(PortForward {
        udp,
        host_addr: addr
            .parse::<Ipv4Addr>()
            .map_err(|_| format!("--forward {spec}: bad address {addr}"))?,
        host_port: port(host)?,
        guest_port: port(guest)?,
    })
}

fn parse(mut it: impl Iterator<Item = String>) -> std::result::Result<Args, String> {
    let mut args = Args {
        kernel: PathBuf::new(),
        initrd: None,
        cmdline: "console=ttyAMA0".into(),
        disks: Vec::new(),
        cpus: 1,
        memory_mib: 1024,
        net: false,
        forwards: Vec::new(),
        virtio_console: None,
        dump_dtb: None,
        log: Level::Info,
        raw: false,
        backend: if cfg!(target_os = "macos") {
            Backend::Hvf
        } else {
            Backend::Jit
        },
    };
    let mut kernel = None;
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--kernel" => kernel = Some(PathBuf::from(value()?)),
            "--initrd" => args.initrd = Some(value()?.into()),
            "--cmdline" => args.cmdline = value()?,
            "--disk" => {
                let v = value()?;
                match v.strip_suffix(",ro") {
                    Some(path) => args.disks.push((path.into(), true)),
                    None => args.disks.push((v.into(), false)),
                }
            }
            "--cpus" => args.cpus = value()?.parse().map_err(|_| "--cpus needs a number")?,
            "--memory" => {
                args.memory_mib = value()?
                    .parse()
                    .map_err(|_| "--memory needs a number of MiB")?
            }
            "--net" => args.net = true,
            "--raw" => args.raw = true,
            "--backend" => {
                args.backend = match value()?.as_str() {
                    "hvf" => Backend::Hvf,
                    "jit" => Backend::Jit,
                    other => return Err(format!("--backend {other}: expected hvf or jit")),
                }
            }
            "--forward" => args.forwards.push(parse_forward(&value()?)?),
            "--virtio-console" => args.virtio_console = Some(value()?.into()),
            "--dump-dtb" => args.dump_dtb = Some(value()?.into()),
            "--log" => {
                args.log = match value()?.as_str() {
                    "error" => Level::Error,
                    "warn" => Level::Warn,
                    "info" => Level::Info,
                    "debug" => Level::Debug,
                    other => {
                        return Err(format!(
                            "--log {other}: expected error, warn, info or debug"
                        ));
                    }
                }
            }
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown option {other}")),
        }
    }
    args.kernel = kernel.ok_or("--kernel is required")?;
    args.net |= !args.forwards.is_empty();
    Ok(args)
}

/// The terminal in raw mode for as long as this lives.
struct RawTerminal(Option<libc::termios>);

impl RawTerminal {
    fn enter() -> Self {
        // SAFETY: isatty, tcgetattr and tcsetattr on stdin with a valid struct.
        unsafe {
            if libc::isatty(0) == 0 {
                return RawTerminal(None);
            }
            let mut saved: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(0, &mut saved) != 0 {
                return RawTerminal(None);
            }
            let mut raw = saved;
            libc::cfmakeraw(&mut raw);
            libc::tcsetattr(0, libc::TCSANOW, &raw);
            RawTerminal(Some(saved))
        }
    }

    fn is_raw(&self) -> bool {
        self.0.is_some()
    }
}

impl Drop for RawTerminal {
    fn drop(&mut self) {
        if let Some(saved) = &self.0 {
            // SAFETY: restores what tcgetattr returned.
            unsafe { libc::tcsetattr(0, libc::TCSANOW, saved) };
        }
    }
}

/// Offers `bytes` to a console until it has taken them all or the machine
/// has stopped.
fn feed(machine: &VirtMachine, console: usize, mut bytes: &[u8]) {
    while !bytes.is_empty() && !machine.board().run.is_stopped() {
        let n = machine.console_write(console, bytes);
        bytes = &bytes[n..];
        if n == 0 {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

fn stdin_thread(machine: Arc<VirtMachine>) {
    let mut stdin = std::io::stdin();
    let mut buf = [0u8; 256];
    let mut escape = false;
    loop {
        let n = match stdin.read(&mut buf) {
            Ok(0) | Err(_) => return,
            Ok(n) => n,
        };
        let mut out = Vec::with_capacity(n);
        for &b in &buf[..n] {
            if escape {
                escape = false;
                match b {
                    b'x' | b'X' => {
                        machine.kill();
                        return;
                    }
                    0x01 => out.push(0x01),
                    _ => {}
                }
            } else if b == 0x01 {
                escape = true;
            } else {
                out.push(b);
            }
        }
        feed(&machine, 0, &out);
    }
}

/// Serves a virtio console on a Unix socket, one client at a time.
fn serve_console(
    listener: UnixListener,
    client: Arc<Mutex<Option<UnixStream>>>,
    machine: Arc<VirtMachine>,
    index: usize,
) {
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let Ok(writer) = stream.try_clone() else {
            continue;
        };
        *client.lock().unwrap() = Some(writer);
        let mut buf = [0u8; 1024];
        while let Ok(n) = stream.read(&mut buf) {
            if n == 0 {
                break;
            }
            feed(&machine, index, &buf[..n]);
        }
        *client.lock().unwrap() = None;
    }
}

fn run(args: Args) -> Result<StopReason> {
    let mut disks: Vec<Arc<dyn crate_core::block::BlockBackend>> = Vec::new();
    for (path, read_only) in &args.disks {
        disks.push(Arc::new(FileDisk::open(path, *read_only)?));
    }
    let mut networks = Vec::new();
    if args.net {
        networks.push(NetworkConfig {
            backend: UserNet::new(args.forwards.clone())?,
            mac: None,
        });
    }

    let stdout: crate_virt::Output = Arc::new(|bytes: &[u8]| {
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(bytes);
        let _ = out.flush();
    });
    let mut consoles = vec![ConsoleConfig {
        kind: ConsoleKind::Serial,
        output: stdout,
        input_ready: None,
    }];
    let hvc_client: Arc<Mutex<Option<UnixStream>>> = Arc::default();
    let hvc_listener = match &args.virtio_console {
        Some(path) => {
            let _ = std::fs::remove_file(path);
            let listener = UnixListener::bind(path)
                .map_err(|e| Error::io(format!("listening on {}", path.display()), e))?;
            let client = hvc_client.clone();
            consoles.push(ConsoleConfig {
                kind: ConsoleKind::Virtio,
                output: Arc::new(move |bytes: &[u8]| {
                    if let Some(stream) = client.lock().unwrap().as_mut() {
                        let _ = stream.write_all(bytes);
                    }
                }),
                input_ready: None,
            });
            Some(listener)
        }
        None => None,
    };

    let machine = Arc::new(VirtMachine::new(VirtConfig {
        cpus: args.cpus,
        memory: args.memory_mib << 20,
        kernel: args.kernel.clone(),
        raw_kernel: args.raw,
        initrd: args.initrd.clone(),
        cmdline: args.cmdline.clone(),
        disks,
        networks,
        consoles,
    })?);

    if let Some(path) = &args.dump_dtb {
        let blob = machine.device_tree();
        // A .dts name gets the tree as text, for hosts without dtc.
        let contents = if path.extension().is_some_and(|e| e == "dts") {
            crate_virt::fdt::parse(blob)
                .map_err(Error::Guest)?
                .to_dts()
                .into_bytes()
        } else {
            blob.to_vec()
        };
        std::fs::write(path, contents)
            .map_err(|e| Error::io(format!("writing {}", path.display()), e))?;
        return Ok(StopReason::PowerOff);
    }

    if let Some(listener) = hvc_listener {
        let machine = machine.clone();
        std::thread::spawn(move || serve_console(listener, hvc_client, machine, 1));
    }
    {
        let machine = machine.clone();
        std::thread::spawn(move || stdin_thread(machine));
    }

    use crate_core::board::CpuBackend;
    let backend: Box<dyn CpuBackend> = match args.backend {
        #[cfg(target_os = "macos")]
        Backend::Hvf => Box::new(crate_hvf::Hvf::new()),
        #[cfg(not(target_os = "macos"))]
        Backend::Hvf => {
            drop(machine);
            return Err(Error::Unsupported("hvf is macOS only".into()));
        }
        Backend::Jit => Box::new(crate_jit::backend::Jit::new()),
    };
    crate_core::info!(
        "{} CPUs, {} MiB, on {}",
        args.cpus,
        args.memory_mib,
        backend.name()
    );
    machine.run(backend.as_ref())
}

fn main() {
    let args = match parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(e) if e.is_empty() => {
            println!("{USAGE}");
            return;
        }
        Err(e) => {
            eprintln!("crate-run: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let terminal = RawTerminal::enter();
    // A raw terminal does not turn \n into \r\n.
    let newline = if terminal.is_raw() { "\r\n" } else { "\n" };
    crate_core::log::set_sink(
        Some(Box::new(move |level, message| {
            eprint!("crate-run: {}: {message}{newline}", level.name())
        })),
        args.log,
    );
    let result = run(args);
    drop(terminal);
    let code = match result {
        Ok(StopReason::PowerOff) => 0,
        Ok(StopReason::Reset) => {
            eprintln!("crate-run: the guest asked to restart");
            0
        }
        Ok(StopReason::Killed) => 130,
        Ok(StopReason::Error(e)) => {
            eprintln!("crate-run: stopped: {e}");
            1
        }
        Err(e) => {
            eprintln!("crate-run: {e}");
            1
        }
    };
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_forwards() {
        let f = parse_forward("2223:22").unwrap();
        assert_eq!(
            (f.udp, f.host_addr, f.host_port, f.guest_port),
            (false, Ipv4Addr::LOCALHOST, 2223, 22)
        );
        let f = parse_forward("udp:0.0.0.0:5353:53").unwrap();
        assert_eq!(
            (f.udp, f.host_addr, f.host_port, f.guest_port),
            (true, Ipv4Addr::UNSPECIFIED, 5353, 53)
        );
        assert!(parse_forward("tcp:22").is_err());
        assert!(parse_forward("99999:22").is_err());
    }

    #[test]
    fn parses_arguments() {
        let args = [
            "--kernel",
            "k",
            "--disk",
            "d.img,ro",
            "--disk",
            "e.img",
            "--forward",
            "2223:22",
            "--cpus",
            "4",
        ];
        let a = parse(args.iter().map(|s| s.to_string())).unwrap();
        assert_eq!(
            a.disks,
            vec![("d.img".into(), true), ("e.img".into(), false)]
        );
        assert!(a.net, "a forward implies the network");
        assert_eq!(a.cpus, 4);
        assert!(
            parse(["--cpus", "4"].iter().map(|s| s.to_string())).is_err(),
            "no kernel"
        );
    }
}
