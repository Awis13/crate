// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! User-mode networking through libslirp: the guest's frames become the
//! host's own sockets, so no privileges are needed. The guest is 10.0.2.15,
//! the host 10.0.2.2 and the DNS forwarder 10.0.2.3, with DHCP handing out
//! the guest's address.
//!
//! libslirp is not thread-safe, so one thread owns it: it polls slirp's
//! sockets, runs its timers, feeds it the guest's frames and delivers the
//! frames slirp sends. Frames from the guest reach that thread through a
//! queue and a pipe that wakes its poll.

use std::collections::VecDeque;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::net::{Ipv4Addr, Ipv6Addr};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Instant;

use crate_core::net::{FrameSink, NetBackend};
use crate_core::{Error, Result, debug, info, warn};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortForward {
    pub udp: bool,
    pub host_addr: Ipv4Addr,
    pub host_port: u16,
    pub guest_port: u16,
}

const GUEST: Ipv4Addr = Ipv4Addr::new(10, 0, 2, 15);
/// Frames waiting for the slirp thread beyond this are dropped.
const TX_LIMIT: usize = 1024;

struct Shared {
    tx: Mutex<VecDeque<Vec<u8>>>,
    wake_read: OwnedFd,
    wake_write: OwnedFd,
    stop: AtomicBool,
    sink: OnceLock<Arc<dyn FrameSink>>,
}

impl Shared {
    fn wake(&self) {
        let byte = 1u8;
        // SAFETY: writes one byte from a valid buffer to our own pipe; a full
        // pipe (EAGAIN) already means a wakeup is pending.
        unsafe { libc::write(self.wake_write.as_raw_fd(), (&raw const byte).cast(), 1) };
    }
}

pub struct UserNet {
    shared: Arc<Shared>,
    forwards: Vec<PortForward>,
    ipv6: bool,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl UserNet {
    pub fn new(forwards: Vec<PortForward>) -> Result<Arc<Self>> {
        Self::with_ipv6(forwards, false)
    }

    /// With `ipv6`, the guest also gets IPv6 beside IPv4: the network
    /// fec0::/64, the host fec0::2 and the DNS forwarder fec0::3, the guest
    /// configuring itself from router advertisements.
    pub fn with_ipv6(forwards: Vec<PortForward>, ipv6: bool) -> Result<Arc<Self>> {
        let mut fds = [0; 2];
        // SAFETY: pipe writes two descriptors into the array.
        if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
            return Err(Error::io(
                "making slirp's wakeup pipe",
                std::io::Error::last_os_error(),
            ));
        }
        for fd in fds {
            // SAFETY: fd is one of the pipe's descriptors.
            unsafe {
                libc::fcntl(fd, libc::F_SETFL, libc::O_NONBLOCK);
                libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
            }
        }
        // SAFETY: the descriptors are new and owned by nothing else.
        let (wake_read, wake_write) =
            unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) };
        Ok(Arc::new(UserNet {
            shared: Arc::new(Shared {
                tx: Mutex::new(VecDeque::new()),
                wake_read,
                wake_write,
                stop: AtomicBool::new(false),
                sink: OnceLock::new(),
            }),
            forwards,
            ipv6,
            thread: Mutex::new(None),
        }))
    }
}

impl Drop for UserNet {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.wake();
        if let Some(thread) = self.thread.lock().unwrap().take() {
            let _ = thread.join();
        }
    }
}

impl NetBackend for UserNet {
    fn start(&self, sink: Arc<dyn FrameSink>) {
        if self.shared.sink.set(sink).is_err() {
            return;
        }
        let shared = self.shared.clone();
        let forwards = self.forwards.clone();
        let ipv6 = self.ipv6;
        let thread = std::thread::Builder::new()
            .name("crate-slirp".into())
            .spawn(move || run(shared, forwards, ipv6))
            .expect("spawning the slirp thread");
        *self.thread.lock().unwrap() = Some(thread);
    }

    fn send(&self, frame: &[u8]) {
        let mut tx = self.shared.tx.lock().unwrap();
        if tx.len() >= TX_LIMIT {
            return;
        }
        let was_empty = tx.is_empty();
        tx.push_back(frame.to_vec());
        drop(tx);
        if was_empty {
            self.shared.wake();
        }
    }
}

// ---------------------------------------------------------------- libslirp

#[repr(C)]
#[derive(Clone, Copy)]
struct InAddr(u32);

#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct In6Addr([u8; 16]);

/// SlirpConfig, as libslirp.h (4.9) lays it out; version 4 of its fields.
#[repr(C)]
struct SlirpConfig {
    version: u32,
    restricted: c_int,
    in_enabled: bool,
    vnetwork: InAddr,
    vnetmask: InAddr,
    vhost: InAddr,
    in6_enabled: bool,
    vprefix_addr6: In6Addr,
    vprefix_len: u8,
    vhost6: In6Addr,
    vhostname: *const c_char,
    tftp_server_name: *const c_char,
    tftp_path: *const c_char,
    bootfile: *const c_char,
    vdhcp_start: InAddr,
    vnameserver: InAddr,
    vnameserver6: In6Addr,
    vdnssearch: *const *const c_char,
    vdomainname: *const c_char,
    if_mtu: usize,
    if_mru: usize,
    disable_host_loopback: bool,
    enable_emu: bool,
    outbound_addr: *const c_void,
    outbound_addr6: *const c_void,
    disable_dns: bool,
    disable_dhcp: bool,
    mfr_id: u32,
    oob_eth_addr: [u8; 6],
}

type TimerCb = unsafe extern "C" fn(*mut c_void);

#[repr(C)]
struct SlirpCb {
    send_packet: unsafe extern "C" fn(*const c_void, usize, *mut c_void) -> isize,
    guest_error: unsafe extern "C" fn(*const c_char, *mut c_void),
    clock_get_ns: unsafe extern "C" fn(*mut c_void) -> i64,
    timer_new: unsafe extern "C" fn(TimerCb, *mut c_void, *mut c_void) -> *mut c_void,
    timer_free: unsafe extern "C" fn(*mut c_void, *mut c_void),
    timer_mod: unsafe extern "C" fn(*mut c_void, i64, *mut c_void),
    register_poll_fd: unsafe extern "C" fn(c_int, *mut c_void),
    unregister_poll_fd: unsafe extern "C" fn(c_int, *mut c_void),
    notify: unsafe extern "C" fn(*mut c_void),
    init_completed: Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>,
    timer_new_opaque: Option<unsafe extern "C" fn(c_int, *mut c_void, *mut c_void) -> *mut c_void>,
    register_poll_socket: Option<unsafe extern "C" fn(c_int, *mut c_void)>,
    unregister_poll_socket: Option<unsafe extern "C" fn(c_int, *mut c_void)>,
}

const POLL_IN: c_int = 1 << 0;
const POLL_OUT: c_int = 1 << 1;
const POLL_PRI: c_int = 1 << 2;
const POLL_ERR: c_int = 1 << 3;
const POLL_HUP: c_int = 1 << 4;

type AddPoll = unsafe extern "C" fn(c_int, c_int, *mut c_void) -> c_int;
type GetRevents = unsafe extern "C" fn(c_int, *mut c_void) -> c_int;

#[allow(non_camel_case_types)]
enum Slirp {}

unsafe extern "C" {
    fn slirp_new(
        cfg: *const SlirpConfig,
        callbacks: *const SlirpCb,
        opaque: *mut c_void,
    ) -> *mut Slirp;
    fn slirp_cleanup(slirp: *mut Slirp);
    fn slirp_pollfds_fill_socket(
        slirp: *mut Slirp,
        timeout: *mut u32,
        add_poll: AddPoll,
        opaque: *mut c_void,
    );
    fn slirp_pollfds_poll(
        slirp: *mut Slirp,
        select_error: c_int,
        get_revents: GetRevents,
        opaque: *mut c_void,
    );
    fn slirp_input(slirp: *mut Slirp, pkt: *const u8, pkt_len: c_int);
    fn slirp_add_hostfwd(
        slirp: *mut Slirp,
        is_udp: c_int,
        host_addr: InAddr,
        host_port: c_int,
        guest_addr: InAddr,
        guest_port: c_int,
    ) -> c_int;
    fn slirp_version_string() -> *const c_char;
}

struct Timer {
    cb: TimerCb,
    cb_opaque: *mut c_void,
    /// Milliseconds on the slirp clock, if armed.
    expire: Option<i64>,
}

/// The slirp thread's state, which every callback gets as its opaque pointer.
struct Context {
    shared: Arc<Shared>,
    start: Instant,
    timers: Vec<*mut Timer>,
    pollfds: Vec<libc::pollfd>,
}

impl Context {
    fn now_ms(&self) -> i64 {
        self.start.elapsed().as_millis() as i64
    }
}

/// # Safety
/// `opaque` must be the Context the slirp thread passed to libslirp.
unsafe fn context<'a>(opaque: *mut c_void) -> &'a mut Context {
    // SAFETY: the caller's promise; libslirp calls back only on that thread.
    unsafe { &mut *opaque.cast::<Context>() }
}

unsafe extern "C" fn send_packet(buf: *const c_void, len: usize, opaque: *mut c_void) -> isize {
    // SAFETY: libslirp passes our context and a frame of `len` bytes.
    let (ctx, frame) = unsafe {
        (
            context(opaque),
            std::slice::from_raw_parts(buf.cast::<u8>(), len),
        )
    };
    if let Some(sink) = ctx.shared.sink.get() {
        sink.deliver(frame);
    }
    len as isize
}

unsafe extern "C" fn guest_error(msg: *const c_char, _opaque: *mut c_void) {
    // SAFETY: libslirp passes a NUL-terminated message.
    let msg = unsafe { CStr::from_ptr(msg) };
    warn!("slirp: guest error: {}", msg.to_string_lossy());
}

unsafe extern "C" fn clock_get_ns(opaque: *mut c_void) -> i64 {
    // SAFETY: libslirp passes our context.
    unsafe { context(opaque) }.start.elapsed().as_nanos() as i64
}

unsafe extern "C" fn timer_new(
    cb: TimerCb,
    cb_opaque: *mut c_void,
    opaque: *mut c_void,
) -> *mut c_void {
    // SAFETY: libslirp passes our context.
    let ctx = unsafe { context(opaque) };
    let timer = Box::into_raw(Box::new(Timer {
        cb,
        cb_opaque,
        expire: None,
    }));
    ctx.timers.push(timer);
    timer.cast()
}

unsafe extern "C" fn timer_free(timer: *mut c_void, opaque: *mut c_void) {
    // SAFETY: libslirp passes our context.
    let ctx = unsafe { context(opaque) };
    let timer = timer.cast::<Timer>();
    if let Some(at) = ctx.timers.iter().position(|&t| t == timer) {
        ctx.timers.swap_remove(at);
        // SAFETY: made by Box::into_raw in timer_new and freed only here.
        drop(unsafe { Box::from_raw(timer) });
    }
}

unsafe extern "C" fn timer_mod(timer: *mut c_void, expire_ms: i64, _opaque: *mut c_void) {
    // SAFETY: a timer from timer_new that libslirp has not freed.
    unsafe { (*timer.cast::<Timer>()).expire = Some(expire_ms) };
}

/// Every socket libslirp makes for the host side of a connection, listening
/// ones and those it accepts or connects included. It writes to them, and a
/// write to one whose other end has gone raises SIGPIPE, which ends a process
/// that has not chosen to ignore it: Rust programs ignore it, C and Swift apps
/// do not. With SO_NOSIGPIPE the write fails with EPIPE instead and slirp
/// closes that connection. (On Apple systems the signal goes to the process,
/// so blocking it on this thread alone would not do.)
unsafe extern "C" fn register_poll_fd(fd: c_int, _opaque: *mut c_void) {
    #[cfg(target_vendor = "apple")]
    {
        let on: c_int = 1;
        // SAFETY: an int option on a socket libslirp has just made.
        unsafe {
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_NOSIGPIPE,
                (&raw const on).cast(),
                size_of::<c_int>() as libc::socklen_t,
            )
        };
    }
    #[cfg(not(target_vendor = "apple"))]
    let _ = fd;
}

unsafe extern "C" fn no_poll_fd(_fd: c_int, _opaque: *mut c_void) {}

unsafe extern "C" fn notify(_opaque: *mut c_void) {}

unsafe extern "C" fn add_poll(fd: c_int, events: c_int, opaque: *mut c_void) -> c_int {
    // SAFETY: libslirp passes our context.
    let ctx = unsafe { context(opaque) };
    let mut ev = 0;
    if events & POLL_IN != 0 {
        ev |= libc::POLLIN;
    }
    if events & POLL_OUT != 0 {
        ev |= libc::POLLOUT;
    }
    if events & POLL_PRI != 0 {
        ev |= libc::POLLPRI;
    }
    ctx.pollfds.push(libc::pollfd {
        fd,
        events: ev,
        revents: 0,
    });
    (ctx.pollfds.len() - 1) as c_int
}

unsafe extern "C" fn get_revents(index: c_int, opaque: *mut c_void) -> c_int {
    // SAFETY: libslirp passes our context.
    let ctx = unsafe { context(opaque) };
    let Some(p) = ctx.pollfds.get(index as usize) else {
        return 0;
    };
    let mut out = 0;
    if p.revents & libc::POLLIN != 0 {
        out |= POLL_IN;
    }
    if p.revents & libc::POLLOUT != 0 {
        out |= POLL_OUT;
    }
    if p.revents & libc::POLLPRI != 0 {
        out |= POLL_PRI;
    }
    if p.revents & libc::POLLERR != 0 {
        out |= POLL_ERR;
    }
    if p.revents & libc::POLLHUP != 0 {
        out |= POLL_HUP;
    }
    out
}

static CALLBACKS: SlirpCb = SlirpCb {
    send_packet,
    guest_error,
    clock_get_ns,
    timer_new,
    timer_free,
    timer_mod,
    register_poll_fd,
    unregister_poll_fd: no_poll_fd,
    notify,
    init_completed: None,
    timer_new_opaque: None,
    register_poll_socket: None,
    unregister_poll_socket: None,
};

fn in_addr(a: Ipv4Addr) -> InAddr {
    InAddr(u32::from_ne_bytes(a.octets()))
}

fn in6_addr(a: Ipv6Addr) -> In6Addr {
    In6Addr(a.octets())
}

fn run(shared: Arc<Shared>, forwards: Vec<PortForward>, ipv6: bool) {
    let prefix6 = Ipv6Addr::new(0xfec0, 0, 0, 0, 0, 0, 0, 0);
    let config = SlirpConfig {
        version: 4,
        restricted: 0,
        in_enabled: true,
        vnetwork: in_addr(Ipv4Addr::new(10, 0, 2, 0)),
        vnetmask: in_addr(Ipv4Addr::new(255, 255, 255, 0)),
        vhost: in_addr(Ipv4Addr::new(10, 0, 2, 2)),
        in6_enabled: ipv6,
        vprefix_addr6: in6_addr(prefix6),
        vprefix_len: 64,
        vhost6: in6_addr(Ipv6Addr::new(0xfec0, 0, 0, 0, 0, 0, 0, 2)),
        vhostname: std::ptr::null(),
        tftp_server_name: std::ptr::null(),
        tftp_path: std::ptr::null(),
        bootfile: std::ptr::null(),
        vdhcp_start: in_addr(GUEST),
        vnameserver: in_addr(Ipv4Addr::new(10, 0, 2, 3)),
        vnameserver6: in6_addr(Ipv6Addr::new(0xfec0, 0, 0, 0, 0, 0, 0, 3)),
        vdnssearch: std::ptr::null(),
        vdomainname: std::ptr::null(),
        if_mtu: 0,
        if_mru: 0,
        disable_host_loopback: false,
        enable_emu: false,
        outbound_addr: std::ptr::null(),
        outbound_addr6: std::ptr::null(),
        disable_dns: false,
        disable_dhcp: false,
        mfr_id: 0,
        oob_eth_addr: [0; 6],
    };
    let mut ctx = Box::new(Context {
        shared,
        start: Instant::now(),
        timers: Vec::new(),
        pollfds: Vec::new(),
    });
    let opaque: *mut c_void = (&raw mut *ctx).cast();
    // SAFETY: config and the callbacks outlive the instance, which is cleaned
    // up below before ctx is dropped.
    let slirp = unsafe { slirp_new(&config, &CALLBACKS, opaque) };
    if slirp.is_null() {
        crate_core::error!("slirp: could not start user-mode networking");
        return;
    }
    // SAFETY: returns a static string.
    let version = unsafe { CStr::from_ptr(slirp_version_string()) };
    debug!("slirp {} started", version.to_string_lossy());

    for f in &forwards {
        // SAFETY: slirp is live and this is its thread.
        let r = unsafe {
            slirp_add_hostfwd(
                slirp,
                f.udp as c_int,
                in_addr(f.host_addr),
                f.host_port as c_int,
                in_addr(GUEST),
                f.guest_port as c_int,
            )
        };
        let proto = if f.udp { "udp" } else { "tcp" };
        if r < 0 {
            warn!(
                "port forward {proto} {}:{} -> {}: could not bind; skipped",
                f.host_addr, f.host_port, f.guest_port
            );
        } else {
            info!(
                "port forward {proto} {}:{} -> guest {}",
                f.host_addr, f.host_port, f.guest_port
            );
        }
    }

    let wake_fd = ctx.shared.wake_read.as_raw_fd();
    while !ctx.shared.stop.load(Ordering::Acquire) {
        ctx.pollfds.clear();
        ctx.pollfds.push(libc::pollfd {
            fd: wake_fd,
            events: libc::POLLIN,
            revents: 0,
        });
        let now = ctx.now_ms();
        let mut timeout: u32 = ctx
            .timers
            .iter()
            // SAFETY: every pointer in timers is a live Timer.
            .filter_map(|&t| unsafe { (*t).expire })
            .map(|e| (e - now).clamp(0, 1000) as u32)
            .min()
            .unwrap_or(1000);
        // SAFETY: slirp is live, this is its thread, and add_poll gets ctx.
        unsafe { slirp_pollfds_fill_socket(slirp, &mut timeout, add_poll, opaque) };
        // SAFETY: pollfds is a valid array of its length.
        let ready = unsafe {
            libc::poll(
                ctx.pollfds.as_mut_ptr(),
                ctx.pollfds.len() as libc::nfds_t,
                timeout as c_int,
            )
        };

        if ctx.pollfds[0].revents & libc::POLLIN != 0 {
            let mut buf = [0u8; 64];
            // SAFETY: reads into a valid buffer from our own non-blocking pipe.
            while unsafe { libc::read(wake_fd, buf.as_mut_ptr().cast(), buf.len()) } > 0 {}
        }
        let frames: Vec<Vec<u8>> = ctx.shared.tx.lock().unwrap().drain(..).collect();
        for frame in frames {
            // SAFETY: slirp is live and the frame is a valid buffer.
            unsafe { slirp_input(slirp, frame.as_ptr(), frame.len() as c_int) };
        }
        // SAFETY: slirp is live and get_revents reads the pollfds just filled.
        unsafe { slirp_pollfds_poll(slirp, (ready < 0) as c_int, get_revents, opaque) };

        let now = ctx.now_ms();
        let due: Vec<*mut Timer> = ctx
            .timers
            .iter()
            .copied()
            // SAFETY: every pointer in timers is a live Timer.
            .filter(|&t| unsafe { (*t).expire }.is_some_and(|e| e <= now))
            .collect();
        for timer in due {
            // A callback may have freed a timer that was also due.
            if !ctx.timers.contains(&timer) {
                continue;
            }
            // SAFETY: still in timers, so still live.
            let (cb, cb_opaque) = unsafe {
                (*timer).expire = None;
                ((*timer).cb, (*timer).cb_opaque)
            };
            // SAFETY: the callback and its argument as libslirp gave them.
            unsafe { cb(cb_opaque) };
        }
    }
    // SAFETY: slirp is live; nothing uses it after this.
    unsafe { slirp_cleanup(slirp) };
    for timer in ctx.timers.drain(..) {
        // SAFETY: made by Box::into_raw in timer_new and not yet freed.
        drop(unsafe { Box::from_raw(timer) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    /// The offsets a C compiler gives libslirp.h's structures on 64-bit Apple
    /// and Linux targets.
    #[test]
    fn structures_match_libslirp_h() {
        assert_eq!(offset_of!(SlirpConfig, in_enabled), 8);
        assert_eq!(offset_of!(SlirpConfig, vprefix_addr6), 28);
        assert_eq!(offset_of!(SlirpConfig, vprefix_len), 44);
        assert_eq!(offset_of!(SlirpConfig, vhost6), 48);
        assert_eq!(offset_of!(SlirpConfig, vhostname), 64);
        assert_eq!(offset_of!(SlirpConfig, vdhcp_start), 96);
        assert_eq!(offset_of!(SlirpConfig, vnameserver6), 104);
        assert_eq!(offset_of!(SlirpConfig, vdnssearch), 120);
        assert_eq!(offset_of!(SlirpConfig, if_mtu), 136);
        assert_eq!(offset_of!(SlirpConfig, disable_host_loopback), 152);
        assert_eq!(offset_of!(SlirpConfig, outbound_addr), 160);
        assert_eq!(offset_of!(SlirpConfig, disable_dns), 176);
        assert_eq!(offset_of!(SlirpConfig, disable_dhcp), 177);
        assert_eq!(offset_of!(SlirpConfig, mfr_id), 180);
        assert_eq!(offset_of!(SlirpConfig, oob_eth_addr), 184);
        assert_eq!(std::mem::size_of::<SlirpConfig>(), 192);
        assert_eq!(offset_of!(SlirpCb, notify), 64);
        assert_eq!(offset_of!(SlirpCb, timer_new_opaque), 80);
        assert_eq!(std::mem::size_of::<SlirpCb>(), 104);
    }

    /// A socket libslirp registers says EPIPE rather than raising SIGPIPE.
    #[cfg(target_vendor = "apple")]
    #[test]
    fn registered_sockets_do_not_raise_sigpipe() {
        // SAFETY: a socket of our own, closed at the end.
        unsafe {
            let fd = libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
            assert!(fd >= 0);
            register_poll_fd(fd, std::ptr::null_mut());
            let mut on: c_int = 0;
            let mut len = size_of::<c_int>() as libc::socklen_t;
            let r = libc::getsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_NOSIGPIPE,
                (&raw mut on).cast(),
                &mut len,
            );
            libc::close(fd);
            assert_eq!(r, 0);
            assert_ne!(on, 0);
        }
    }

    struct Frames(Mutex<Vec<Vec<u8>>>);

    impl FrameSink for Frames {
        fn deliver(&self, frame: &[u8]) {
            self.0.lock().unwrap().push(frame.to_vec());
        }
    }

    /// An ARP request for the host's address gets slirp's reply.
    #[test]
    fn answers_arp_for_the_host() {
        let net = UserNet::new(Vec::new()).unwrap();
        let frames = Arc::new(Frames(Mutex::new(Vec::new())));
        net.start(frames.clone());
        let mac = [2, 0x63, 0x72, 0x61, 0x74, 0x65];
        let mut arp = Vec::new();
        arp.extend_from_slice(&[0xff; 6]);
        arp.extend_from_slice(&mac);
        arp.extend_from_slice(&[0x08, 0x06, 0, 1, 0x08, 0, 6, 4, 0, 1]);
        arp.extend_from_slice(&mac);
        arp.extend_from_slice(&[10, 0, 2, 15]);
        arp.extend_from_slice(&[0; 6]);
        arp.extend_from_slice(&[10, 0, 2, 2]);
        net.send(&arp);
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        while frames.0.lock().unwrap().is_empty() && Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let got = frames.0.lock().unwrap();
        let reply = got.first().expect("an ARP reply");
        assert_eq!(&reply[0..6], &mac, "addressed to the guest");
        assert_eq!(&reply[12..14], &[0x08, 0x06]);
        assert_eq!(&reply[20..22], &[0, 2], "a reply");
        assert_eq!(&reply[28..32], &[10, 0, 2, 2], "for the host's address");
    }
}
