// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A virtio socket device (VIRTIO 1.2, section 5.10): stream sockets between
//! the guest and the host, without a network.
//!
//! The host's end of every connection is a Unix socket pair: the app gets one
//! end as an ordinary file descriptor, the device keeps the other and moves
//! bytes between it and the guest's queues. The app either connects to a port
//! the guest listens on ([`VsockHandle::connect`]) or listens on a host port
//! for the guest's connections ([`VsockHandle::listen`]).
//!
//! Packets from the guest are handled on the vCPU that notified the transmit
//! queue. A thread of the device's own polls the host's sockets, reads what
//! the guest has credit for and queues it for the guest's receive buffers.
//! Flow control is the virtio one: each side tells the other how much buffer
//! it has (`buf_alloc`) and how much it has consumed (`fwd_cnt`), and the
//! sender never has more in flight than that allows.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate_core::warn;

use super::{
    Activation, F_INDIRECT_DESC, F_VERSION_1, Interrupt, Queue, VirtioDevice, read_config_bytes,
};

pub const ID_VSOCK: u32 = 19;
pub const HOST_CID: u64 = 2;

const HEADER: usize = 44;
const QUEUE_SIZE: u16 = 256;

const TYPE_STREAM: u16 = 1;

const OP_REQUEST: u16 = 1;
const OP_RESPONSE: u16 = 2;
const OP_RST: u16 = 3;
const OP_SHUTDOWN: u16 = 4;
const OP_RW: u16 = 5;
const OP_CREDIT_UPDATE: u16 = 6;
const OP_CREDIT_REQUEST: u16 = 7;

const SHUTDOWN_RCV: u32 = 1;
const SHUTDOWN_SEND: u32 = 2;

/// The buffer the host offers each connection, in bytes.
const BUF_ALLOC: u32 = 256 * 1024;
/// The most read from a host socket at a time.
const READ_CHUNK: usize = 64 * 1024;
/// The most bytes from a guest packet taken at once.
const MAX_PACKET: usize = HEADER + 64 * 1024;
/// While this many packets wait for the guest's buffers, the host's sockets
/// are not read.
const PENDING_LIMIT: usize = 256;
/// The host ports connections from the host come from.
const EPHEMERAL_FIRST: u32 = 49152;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Header {
    pub src_cid: u64,
    pub dst_cid: u64,
    pub src_port: u32,
    pub dst_port: u32,
    pub len: u32,
    pub kind: u16,
    pub op: u16,
    pub flags: u32,
    pub buf_alloc: u32,
    pub fwd_cnt: u32,
}

impl Header {
    pub fn encode(&self) -> [u8; HEADER] {
        let mut b = [0u8; HEADER];
        b[0..8].copy_from_slice(&self.src_cid.to_le_bytes());
        b[8..16].copy_from_slice(&self.dst_cid.to_le_bytes());
        b[16..20].copy_from_slice(&self.src_port.to_le_bytes());
        b[20..24].copy_from_slice(&self.dst_port.to_le_bytes());
        b[24..28].copy_from_slice(&self.len.to_le_bytes());
        b[28..30].copy_from_slice(&self.kind.to_le_bytes());
        b[30..32].copy_from_slice(&self.op.to_le_bytes());
        b[32..36].copy_from_slice(&self.flags.to_le_bytes());
        b[36..40].copy_from_slice(&self.buf_alloc.to_le_bytes());
        b[40..44].copy_from_slice(&self.fwd_cnt.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> Option<Header> {
        if b.len() < HEADER {
            return None;
        }
        let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let u16_at = |i: usize| u16::from_le_bytes(b[i..i + 2].try_into().unwrap());
        Some(Header {
            src_cid: u64_at(0),
            dst_cid: u64_at(8),
            src_port: u32_at(16),
            dst_port: u32_at(20),
            len: u32_at(24),
            kind: u16_at(28),
            op: u16_at(30),
            flags: u32_at(32),
            buf_alloc: u32_at(36),
            fwd_cnt: u32_at(40),
        })
    }
}

/// Called with the host's end of a connection the guest made to a port the
/// app listens on. The app owns the descriptor.
pub type Acceptor = Arc<dyn Fn(u32, OwnedFd) + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// The host asked; the guest has not answered.
    Connecting,
    Established,
}

/// Wakes a host thread waiting in `connect`.
type Waiter = Arc<(Mutex<Option<bool>>, Condvar)>;

struct Conn {
    /// The device's end of the socket pair, non-blocking.
    fd: OwnedFd,
    state: State,
    waiter: Option<Waiter>,
    /// What the guest last said about its buffer.
    peer_buf_alloc: u32,
    peer_fwd_cnt: u32,
    /// Bytes sent to the guest.
    tx_cnt: u32,
    /// Bytes from the guest written to the host's socket.
    fwd_cnt: u32,
    /// `fwd_cnt` as last told to the guest.
    fwd_cnt_told: u32,
    /// Bytes from the guest the host's socket has not taken yet.
    out: VecDeque<u8>,
    /// The host's side has nothing more to send.
    host_eof: bool,
    /// The guest will send nothing more.
    guest_done_sending: bool,
    /// The guest will take nothing more.
    guest_done_receiving: bool,
}

impl Conn {
    fn new(fd: OwnedFd, state: State) -> Self {
        Conn {
            fd,
            state,
            waiter: None,
            peer_buf_alloc: 0,
            peer_fwd_cnt: 0,
            tx_cnt: 0,
            fwd_cnt: 0,
            fwd_cnt_told: 0,
            out: VecDeque::new(),
            host_eof: false,
            guest_done_sending: false,
            guest_done_receiving: false,
        }
    }

    /// How many more bytes the guest has room for.
    fn credit(&self) -> u32 {
        let in_flight = self.tx_cnt.wrapping_sub(self.peer_fwd_cnt);
        self.peer_buf_alloc.saturating_sub(in_flight)
    }

    fn wants_read(&self) -> bool {
        self.state == State::Established
            && !self.host_eof
            && !self.guest_done_receiving
            && self.credit() > 0
    }

    fn wake(&mut self, ok: bool) {
        if let Some(w) = self.waiter.take() {
            *w.0.lock().unwrap() = Some(ok);
            w.1.notify_all();
        }
    }
}

/// (host port, guest port).
type Key = (u32, u32);

struct Inner {
    guest_cid: u64,
    conns: HashMap<Key, Conn>,
    listeners: HashMap<u32, Acceptor>,
    /// Packets for the guest, oldest first.
    pending: VecDeque<(Header, Vec<u8>)>,
    rx: Option<(Queue, Arc<Interrupt>)>,
    tx: Option<(Queue, Arc<Interrupt>)>,
    _event: Option<Queue>,
    next_port: u32,
    /// Accepted connections to hand to the app once the lock is let go.
    accepted: Vec<(Acceptor, u32, OwnedFd)>,
}

struct Shared {
    inner: Mutex<Inner>,
    stop: AtomicBool,
    wake_read: OwnedFd,
    wake_write: OwnedFd,
}

impl Shared {
    fn wake(&self) {
        let byte = 1u8;
        // SAFETY: writing one byte from a live buffer to our own pipe.
        unsafe { libc::write(self.wake_write.as_raw_fd(), (&raw const byte).cast(), 1) };
    }
}

fn set_nonblocking(fd: RawFd) -> io::Result<()> {
    // SAFETY: fcntl on a descriptor we own.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: as above.
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn set_cloexec(fd: RawFd) {
    // SAFETY: fcntl on a descriptor we own.
    unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
}

/// A connected pair of stream sockets: (the device's, the app's).
fn socket_pair() -> io::Result<(OwnedFd, OwnedFd)> {
    let mut fds = [0; 2];
    // SAFETY: socketpair writes two descriptors into a live array.
    if unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) } < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: both were just made and are ours.
    let (ours, theirs) = unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) };
    set_cloexec(ours.as_raw_fd());
    set_cloexec(theirs.as_raw_fd());
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    for fd in [&ours, &theirs] {
        let one: libc::c_int = 1;
        // SAFETY: setsockopt with a live int; writes to a closed peer then
        // fail with EPIPE instead of killing the process.
        unsafe {
            libc::setsockopt(
                fd.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_NOSIGPIPE,
                (&raw const one).cast(),
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            )
        };
    }
    set_nonblocking(ours.as_raw_fd())?;
    let mut size: libc::c_int = BUF_ALLOC as libc::c_int;
    for fd in [&ours, &theirs] {
        for opt in [libc::SO_SNDBUF, libc::SO_RCVBUF] {
            // SAFETY: as above.
            unsafe {
                libc::setsockopt(
                    fd.as_raw_fd(),
                    libc::SOL_SOCKET,
                    opt,
                    (&raw mut size).cast(),
                    std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                )
            };
        }
    }
    Ok((ours, theirs))
}

impl Inner {
    fn header(&self, key: Key, op: u16) -> Header {
        let fwd_cnt = self.conns.get(&key).map(|c| c.fwd_cnt).unwrap_or(0);
        Header {
            src_cid: HOST_CID,
            dst_cid: self.guest_cid,
            src_port: key.0,
            dst_port: key.1,
            len: 0,
            kind: TYPE_STREAM,
            op,
            flags: 0,
            buf_alloc: BUF_ALLOC,
            fwd_cnt,
        }
    }

    fn send(&mut self, key: Key, op: u16, flags: u32) {
        let mut h = self.header(key, op);
        h.flags = flags;
        if let Some(c) = self.conns.get_mut(&key) {
            c.fwd_cnt_told = c.fwd_cnt;
        }
        self.pending.push_back((h, Vec::new()));
    }

    /// Answers a packet for no connection with a reset.
    fn reset_reply(&mut self, h: &Header) {
        if h.op == OP_RST {
            return;
        }
        let mut r = self.header((h.dst_port, h.src_port), OP_RST);
        r.dst_cid = h.src_cid;
        r.src_cid = h.dst_cid;
        self.pending.push_back((r, Vec::new()));
    }

    fn close(&mut self, key: Key, tell_guest: bool) {
        if let Some(mut c) = self.conns.remove(&key) {
            c.wake(false);
            if tell_guest {
                let mut h = self.header(key, OP_RST);
                h.fwd_cnt = c.fwd_cnt;
                self.pending.push_back((h, Vec::new()));
            }
        }
    }

    /// Writes what the guest sent to the host's socket, as much as it takes.
    fn drain_out(&mut self, key: Key) {
        let Some(c) = self.conns.get_mut(&key) else {
            return;
        };
        let mut broken = false;
        while !c.out.is_empty() {
            let (a, _) = c.out.as_slices();
            // SAFETY: writing from a live slice to a descriptor we own.
            let n = unsafe { libc::write(c.fd.as_raw_fd(), a.as_ptr().cast(), a.len()) };
            if n < 0 {
                let e = io::Error::last_os_error();
                if e.kind() != io::ErrorKind::WouldBlock && e.kind() != io::ErrorKind::Interrupted {
                    broken = true;
                }
                break;
            }
            c.out.drain(..n as usize);
            c.fwd_cnt = c.fwd_cnt.wrapping_add(n as u32);
        }
        if broken {
            self.close(key, true);
            return;
        }
        if c.out.is_empty() && c.guest_done_sending {
            // SAFETY: shutdown on a descriptor we own.
            unsafe { libc::shutdown(c.fd.as_raw_fd(), libc::SHUT_WR) };
        }
        if c.fwd_cnt.wrapping_sub(c.fwd_cnt_told) >= BUF_ALLOC / 4 {
            self.send(key, OP_CREDIT_UPDATE, 0);
        }
    }

    /// Both directions are finished: the connection can go.
    fn finish_if_done(&mut self, key: Key) {
        let Some(c) = self.conns.get(&key) else {
            return;
        };
        if c.host_eof && c.guest_done_sending && c.out.is_empty() {
            self.close(key, true);
        }
    }

    /// One packet from the guest.
    fn receive(&mut self, h: Header, payload: &[u8]) {
        if h.dst_cid != HOST_CID || h.src_cid != self.guest_cid || h.kind != TYPE_STREAM {
            self.reset_reply(&h);
            return;
        }
        let key = (h.dst_port, h.src_port);
        if h.op == OP_REQUEST {
            if self.conns.contains_key(&key) {
                self.reset_reply(&h);
                return;
            }
            let Some(acceptor) = self.listeners.get(&h.dst_port).cloned() else {
                self.reset_reply(&h);
                return;
            };
            match socket_pair() {
                Ok((ours, theirs)) => {
                    let mut c = Conn::new(ours, State::Established);
                    c.peer_buf_alloc = h.buf_alloc;
                    c.peer_fwd_cnt = h.fwd_cnt;
                    self.conns.insert(key, c);
                    self.send(key, OP_RESPONSE, 0);
                    self.accepted.push((acceptor, h.dst_port, theirs));
                }
                Err(e) => {
                    warn!("virtio-vsock: socketpair: {e}");
                    self.reset_reply(&h);
                }
            }
            return;
        }
        let Some(c) = self.conns.get_mut(&key) else {
            self.reset_reply(&h);
            return;
        };
        c.peer_buf_alloc = h.buf_alloc;
        c.peer_fwd_cnt = h.fwd_cnt;
        match h.op {
            OP_RESPONSE => {
                if c.state == State::Connecting {
                    c.state = State::Established;
                    c.wake(true);
                } else {
                    self.close(key, true);
                }
            }
            OP_RST => self.close(key, false),
            OP_SHUTDOWN => {
                if h.flags & SHUTDOWN_SEND != 0 {
                    c.guest_done_sending = true;
                }
                if h.flags & SHUTDOWN_RCV != 0 {
                    c.guest_done_receiving = true;
                }
                if c.guest_done_sending && c.guest_done_receiving {
                    self.close(key, true);
                } else {
                    self.drain_out(key);
                    self.finish_if_done(key);
                }
            }
            OP_RW => {
                if c.state != State::Established {
                    self.close(key, true);
                    return;
                }
                c.out.extend(payload);
                self.drain_out(key);
            }
            OP_CREDIT_UPDATE => {}
            OP_CREDIT_REQUEST => self.send(key, OP_CREDIT_UPDATE, 0),
            _ => {}
        }
    }

    /// Takes the guest's packets off the transmit queue.
    fn transmit(&mut self) {
        let Some((mut queue, interrupt)) = self.tx.take() else {
            return;
        };
        let mut used = false;
        loop {
            let chain = match queue.pop() {
                Ok(Some(chain)) => chain,
                Ok(None) => break,
                Err(e) => {
                    warn!("virtio-vsock: transmit queue: {e}");
                    break;
                }
            };
            match chain.read_all(queue.memory(), MAX_PACKET) {
                Ok(packet) => {
                    if let Some(h) = Header::decode(&packet) {
                        let end = (HEADER + h.len as usize).min(packet.len());
                        self.receive(h, &packet[HEADER..end]);
                    }
                }
                Err(e) => warn!("virtio-vsock: {e}"),
            }
            if queue.add_used(chain.head, 0).is_err() {
                break;
            }
            used = true;
        }
        if used && queue.needs_interrupt() {
            interrupt.signal_used();
        }
        self.tx = Some((queue, interrupt));
    }

    /// Puts as many waiting packets as fit into the guest's receive buffers.
    fn flush(&mut self) {
        let Some((queue, interrupt)) = &mut self.rx else {
            return;
        };
        let mut used = false;
        while let Some((mut h, mut payload)) = self.pending.pop_front() {
            let chain = match queue.pop() {
                Ok(Some(chain)) => chain,
                Ok(None) => {
                    self.pending.push_front((h, payload));
                    break;
                }
                Err(e) => {
                    warn!("virtio-vsock: receive queue: {e}");
                    self.pending.push_front((h, payload));
                    break;
                }
            };
            let room = chain.writable_len() as usize;
            if room < HEADER {
                warn!("virtio-vsock: a receive buffer has no room for a header");
                let _ = queue.add_used(chain.head, 0);
                self.pending.push_front((h, payload));
                used = true;
                continue;
            }
            // A payload too big for this buffer is split across buffers.
            let rest = if payload.len() > room - HEADER {
                let rest = payload.split_off(room - HEADER);
                let mut rh = h;
                rh.len = rest.len() as u32;
                Some((rh, rest))
            } else {
                None
            };
            h.len = payload.len() as u32;
            let mut packet = Vec::with_capacity(HEADER + payload.len());
            packet.extend_from_slice(&h.encode());
            packet.extend_from_slice(&payload);
            let written = chain.write_all(queue.memory(), &packet).unwrap_or(0);
            if let Some(rest) = rest {
                self.pending.push_front(rest);
            }
            if queue.add_used(chain.head, written as u32).is_err() {
                break;
            }
            used = true;
        }
        if used && queue.needs_interrupt() {
            interrupt.signal_used();
        }
    }

    /// Reads what the guest has credit for from one host socket.
    fn read_host(&mut self, key: Key) {
        let Some(c) = self.conns.get_mut(&key) else {
            return;
        };
        if !c.wants_read() {
            return;
        }
        let want = (c.credit() as usize).min(READ_CHUNK);
        let mut buf = vec![0u8; want];
        // SAFETY: reading into a live buffer from a descriptor we own.
        let n = unsafe { libc::read(c.fd.as_raw_fd(), buf.as_mut_ptr().cast(), want) };
        if n < 0 {
            let e = io::Error::last_os_error();
            if e.kind() != io::ErrorKind::WouldBlock && e.kind() != io::ErrorKind::Interrupted {
                self.close(key, true);
            }
            return;
        }
        if n == 0 {
            c.host_eof = true;
            self.send(key, OP_SHUTDOWN, SHUTDOWN_SEND);
            self.finish_if_done(key);
            return;
        }
        buf.truncate(n as usize);
        c.tx_cnt = c.tx_cnt.wrapping_add(n as u32);
        c.fwd_cnt_told = c.fwd_cnt;
        let mut h = self.header(key, OP_RW);
        h.len = n as u32;
        self.pending.push_back((h, buf));
    }

    fn take_accepted(&mut self) -> Vec<(Acceptor, u32, OwnedFd)> {
        std::mem::take(&mut self.accepted)
    }
}

fn hand_over(accepted: Vec<(Acceptor, u32, OwnedFd)>) {
    for (acceptor, port, fd) in accepted {
        acceptor(port, fd);
    }
}

/// The host's side of the device: what the app calls.
#[derive(Clone)]
pub struct VsockHandle {
    shared: Arc<Shared>,
}

impl VsockHandle {
    /// Connects to `port` in the guest; returns the host's end of the stream.
    pub fn connect(&self, port: u32, timeout: Duration) -> io::Result<OwnedFd> {
        let waiter: Waiter = Arc::new((Mutex::new(None), Condvar::new()));
        let (key, theirs) = {
            let mut inner = self.shared.inner.lock().unwrap();
            if inner.rx.is_none() || inner.tx.is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionRefused,
                    "the guest's vsock driver is not up",
                ));
            }
            let mut local = inner.next_port;
            while inner.conns.keys().any(|k| k.0 == local) || inner.listeners.contains_key(&local) {
                local = local.checked_add(1).unwrap_or(EPHEMERAL_FIRST);
            }
            inner.next_port = local.checked_add(1).unwrap_or(EPHEMERAL_FIRST);
            let key = (local, port);
            let (ours, theirs) = socket_pair()?;
            let mut c = Conn::new(ours, State::Connecting);
            c.waiter = Some(waiter.clone());
            inner.conns.insert(key, c);
            inner.send(key, OP_REQUEST, 0);
            inner.flush();
            (key, theirs)
        };
        let deadline = Instant::now() + timeout;
        let (lock, cvar) = &*waiter;
        let mut result = lock.lock().unwrap();
        while result.is_none() {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            result = cvar.wait_timeout(result, deadline - now).unwrap().0;
        }
        let outcome = *result;
        drop(result);
        match outcome {
            Some(true) => {
                self.shared.wake();
                Ok(theirs)
            }
            Some(false) => Err(io::Error::new(
                io::ErrorKind::ConnectionRefused,
                format!("the guest refused vsock port {port}"),
            )),
            None => {
                let mut inner = self.shared.inner.lock().unwrap();
                inner.close(key, true);
                inner.flush();
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("vsock port {port}: the guest did not answer"),
                ))
            }
        }
    }

    /// Hands every connection the guest makes to host `port` to `acceptor`.
    pub fn listen(&self, port: u32, acceptor: Acceptor) -> io::Result<()> {
        let mut inner = self.shared.inner.lock().unwrap();
        if inner.listeners.contains_key(&port) {
            return Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                format!("vsock port {port} already has a listener"),
            ));
        }
        inner.listeners.insert(port, acceptor);
        Ok(())
    }

    pub fn unlisten(&self, port: u32) {
        self.shared.inner.lock().unwrap().listeners.remove(&port);
    }
}

pub struct Vsock {
    guest_cid: u64,
    shared: Arc<Shared>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Vsock {
    pub fn new(guest_cid: u64) -> io::Result<Self> {
        let mut fds = [0; 2];
        // SAFETY: pipe writes two descriptors into a live array.
        if unsafe { libc::pipe(fds.as_mut_ptr()) } < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: both were just made and are ours.
        let (r, w) = unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) };
        for fd in [&r, &w] {
            set_cloexec(fd.as_raw_fd());
            set_nonblocking(fd.as_raw_fd())?;
        }
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner {
                guest_cid,
                conns: HashMap::new(),
                listeners: HashMap::new(),
                pending: VecDeque::new(),
                rx: None,
                tx: None,
                _event: None,
                next_port: EPHEMERAL_FIRST,
                accepted: Vec::new(),
            }),
            stop: AtomicBool::new(false),
            wake_read: r,
            wake_write: w,
        });
        let worker_shared = shared.clone();
        let worker = thread::Builder::new()
            .name("crate-vsock".into())
            .spawn(move || run(&worker_shared))?;
        Ok(Vsock {
            guest_cid,
            shared,
            worker: Some(worker),
        })
    }

    pub fn handle(&self) -> VsockHandle {
        VsockHandle {
            shared: self.shared.clone(),
        }
    }
}

impl Drop for Vsock {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.wake();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// The device's thread: polls the host's sockets until the device goes.
fn run(shared: &Shared) {
    let mut fds: Vec<libc::pollfd> = Vec::new();
    let mut keys: Vec<Key> = Vec::new();
    while !shared.stop.load(Ordering::Acquire) {
        fds.clear();
        keys.clear();
        fds.push(libc::pollfd {
            fd: shared.wake_read.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        });
        {
            let inner = shared.inner.lock().unwrap();
            let reading = inner.pending.len() < PENDING_LIMIT && inner.rx.is_some();
            for (key, c) in &inner.conns {
                let mut events = 0;
                if reading && c.wants_read() {
                    events |= libc::POLLIN;
                }
                if !c.out.is_empty() {
                    events |= libc::POLLOUT;
                }
                if events != 0 {
                    fds.push(libc::pollfd {
                        fd: c.fd.as_raw_fd(),
                        events,
                        revents: 0,
                    });
                    keys.push(*key);
                }
            }
        }
        // SAFETY: poll over a live array of pollfds.
        let n = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, 1000) };
        if n < 0 {
            continue;
        }
        if fds[0].revents != 0 {
            let mut buf = [0u8; 64];
            // SAFETY: draining our own pipe into a live buffer.
            while unsafe { libc::read(fds[0].fd, buf.as_mut_ptr().cast(), buf.len()) } > 0 {}
        }
        let accepted = {
            let mut inner = shared.inner.lock().unwrap();
            for (pfd, key) in fds[1..].iter().zip(&keys) {
                if pfd.revents == 0 {
                    continue;
                }
                if pfd.revents & libc::POLLOUT != 0 {
                    inner.drain_out(*key);
                }
                if pfd.revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) != 0 {
                    inner.read_host(*key);
                }
                inner.finish_if_done(*key);
            }
            inner.flush();
            inner.take_accepted()
        };
        hand_over(accepted);
    }
}

impl VirtioDevice for Vsock {
    fn device_id(&self) -> u32 {
        ID_VSOCK
    }

    fn features(&self) -> u64 {
        F_VERSION_1 | F_INDIRECT_DESC
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &[QUEUE_SIZE, QUEUE_SIZE, QUEUE_SIZE]
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        read_config_bytes(&self.guest_cid.to_le_bytes(), offset, data);
    }

    fn activate(&mut self, activation: Activation) {
        let mut queues = activation.queues.into_iter();
        let (Some(rx), Some(tx), event) = (queues.next(), queues.next(), queues.next()) else {
            return;
        };
        let mut inner = self.shared.inner.lock().unwrap();
        inner.rx = Some((rx, activation.interrupt.clone()));
        inner.tx = Some((tx, activation.interrupt));
        inner._event = event;
        drop(inner);
        self.shared.wake();
    }

    fn notify(&mut self, index: u16) {
        let accepted = {
            let mut inner = self.shared.inner.lock().unwrap();
            match index {
                0 => {}
                1 => inner.transmit(),
                _ => return,
            }
            inner.flush();
            inner.take_accepted()
        };
        hand_over(accepted);
        self.shared.wake();
    }

    fn reset(&mut self) {
        let mut inner = self.shared.inner.lock().unwrap();
        for (_, mut c) in inner.conns.drain() {
            c.wake(false);
        }
        inner.pending.clear();
        inner.rx = None;
        inner.tx = None;
        inner._event = None;
    }
}

#[cfg(test)]
mod tests {
    use super::super::queue::QueueConfig;
    use super::super::queue::testing::*;
    use super::*;
    use crate_core::irq::{IrqLine, LevelRecorder};
    use crate_core::memory::GuestMemory;

    const GUEST: u64 = 3;

    fn qconfig(n: u64) -> QueueConfig {
        let base = DATA + 0x8_0000 + n * 0x1_0000;
        QueueConfig {
            size: 8,
            ready: true,
            desc: base,
            driver: base + 0x1000,
            device: base + 0x2000,
        }
    }

    struct Rig {
        dev: Vsock,
        memory: Arc<GuestMemory>,
        rx: Driver,
        tx: Driver,
        next_buf: u64,
    }

    impl Rig {
        fn new() -> Rig {
            let memory = memory();
            let mut dev = Vsock::new(GUEST).unwrap();
            dev.activate(Activation {
                memory: memory.clone(),
                queues: vec![
                    Queue::new(memory.clone(), qconfig(0)),
                    Queue::new(memory.clone(), qconfig(1)),
                    Queue::new(memory.clone(), qconfig(2)),
                ],
                interrupt: Interrupt::new(IrqLine::new(Arc::new(LevelRecorder::default()), 0)),
                features: dev.features(),
            });
            Rig {
                rx: Driver::at(memory.clone(), qconfig(0)),
                tx: Driver::at(memory.clone(), qconfig(1)),
                dev,
                memory,
                next_buf: DATA,
            }
        }

        fn buf(&mut self, len: u64) -> u64 {
            let a = self.next_buf;
            self.next_buf += len.next_multiple_of(0x100);
            a
        }

        /// The guest sends a packet.
        fn guest_send(&mut self, h: Header, payload: &[u8]) {
            let a = self.buf((HEADER + payload.len()) as u64);
            let mut bytes = h.encode().to_vec();
            bytes.extend_from_slice(payload);
            self.memory.write(a, &bytes).unwrap();
            self.tx.offer(&[(a, bytes.len() as u32, false)]);
            self.dev.notify(1);
        }

        /// The guest posts a receive buffer and takes what comes, waiting a
        /// little for the device's thread.
        fn guest_recv(&mut self) -> Option<(Header, Vec<u8>)> {
            let a = self.buf(4096);
            self.rx.offer(&[(a, 4096, true)]);
            self.dev.notify(0);
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                if let Some((_, len)) = self.rx.take_used() {
                    let mut bytes = vec![0; len as usize];
                    self.memory.read(a, &mut bytes).unwrap();
                    let h = Header::decode(&bytes).unwrap();
                    return Some((h, bytes[HEADER..].to_vec()));
                }
                if Instant::now() > deadline {
                    return None;
                }
                thread::sleep(Duration::from_millis(5));
            }
        }
    }

    fn guest_header(src_port: u32, dst_port: u32, op: u16) -> Header {
        Header {
            src_cid: GUEST,
            dst_cid: HOST_CID,
            src_port,
            dst_port,
            len: 0,
            kind: TYPE_STREAM,
            op,
            flags: 0,
            buf_alloc: 64 * 1024,
            fwd_cnt: 0,
        }
    }

    fn write_fd(fd: &OwnedFd, data: &[u8]) {
        // SAFETY: test write from a live slice.
        let n = unsafe { libc::write(fd.as_raw_fd(), data.as_ptr().cast(), data.len()) };
        assert_eq!(n as usize, data.len());
    }

    fn read_fd(fd: &OwnedFd, len: usize) -> Vec<u8> {
        let mut out = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(2);
        set_nonblocking(fd.as_raw_fd()).unwrap();
        while out.len() < len && Instant::now() < deadline {
            let mut buf = vec![0u8; len - out.len()];
            // SAFETY: test read into a live buffer.
            let n = unsafe { libc::read(fd.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) };
            if n > 0 {
                out.extend_from_slice(&buf[..n as usize]);
            } else {
                thread::sleep(Duration::from_millis(5));
            }
        }
        out
    }

    #[test]
    fn header_round_trips() {
        let h = guest_header(1, 2, OP_RW);
        assert_eq!(Header::decode(&h.encode()), Some(h));
    }

    #[test]
    fn reports_the_guest_cid() {
        let dev = Vsock::new(42).unwrap();
        let mut b = [0; 8];
        dev.read_config(0, &mut b);
        assert_eq!(u64::from_le_bytes(b), 42);
    }

    #[test]
    fn refuses_a_port_nobody_listens_on() {
        let mut rig = Rig::new();
        rig.guest_send(guest_header(1000, 80, OP_REQUEST), &[]);
        let (h, _) = rig.guest_recv().unwrap();
        assert_eq!(h.op, OP_RST);
        assert_eq!((h.src_port, h.dst_port, h.dst_cid), (80, 1000, GUEST));
    }

    #[test]
    fn guest_connects_and_both_sides_talk() {
        let mut rig = Rig::new();
        let got: Arc<Mutex<Vec<OwnedFd>>> = Arc::default();
        let sink = got.clone();
        rig.dev
            .handle()
            .listen(1024, Arc::new(move |_, fd| sink.lock().unwrap().push(fd)))
            .unwrap();
        rig.guest_send(guest_header(5000, 1024, OP_REQUEST), &[]);
        let (h, _) = rig.guest_recv().unwrap();
        assert_eq!(h.op, OP_RESPONSE);
        let fd = got
            .lock()
            .unwrap()
            .pop()
            .expect("the app got the connection");

        let mut data = guest_header(5000, 1024, OP_RW);
        data.len = 5;
        rig.guest_send(data, b"hello");
        assert_eq!(read_fd(&fd, 5), b"hello");

        write_fd(&fd, b"world!");
        let (h, payload) = rig.guest_recv().unwrap();
        assert_eq!(h.op, OP_RW);
        assert_eq!(payload, b"world!");
        assert_eq!((h.src_port, h.dst_port), (1024, 5000));

        drop(fd);
        let (h, _) = rig.guest_recv().unwrap();
        assert_eq!(h.op, OP_SHUTDOWN);
    }

    #[test]
    fn host_connects_to_a_guest_port() {
        let rig = Arc::new(Mutex::new(Rig::new()));
        let handle = rig.lock().unwrap().dev.handle();
        let connecting = thread::spawn(move || handle.connect(1024, Duration::from_secs(2)));
        // The guest sees the request and accepts it.
        let req = loop {
            let r = rig.lock().unwrap().guest_recv();
            if let Some(r) = r {
                break r.0;
            }
        };
        assert_eq!(req.op, OP_REQUEST);
        assert_eq!(req.dst_port, 1024);
        let mut resp = guest_header(1024, req.src_port, OP_RESPONSE);
        resp.src_port = 1024;
        rig.lock().unwrap().guest_send(resp, &[]);
        let fd = connecting.join().unwrap().expect("connected");
        write_fd(&fd, b"ping");
        let (h, payload) = rig.lock().unwrap().guest_recv().unwrap();
        assert_eq!((h.op, payload.as_slice()), (OP_RW, &b"ping"[..]));
    }

    #[test]
    fn host_connect_is_refused_by_a_reset() {
        let rig = Arc::new(Mutex::new(Rig::new()));
        let handle = rig.lock().unwrap().dev.handle();
        let connecting = thread::spawn(move || handle.connect(7, Duration::from_secs(2)));
        let req = loop {
            let r = rig.lock().unwrap().guest_recv();
            if let Some(r) = r {
                break r.0;
            }
        };
        rig.lock()
            .unwrap()
            .guest_send(guest_header(7, req.src_port, OP_RST), &[]);
        let err = connecting.join().unwrap().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionRefused);
    }

    #[test]
    fn respects_the_guests_credit() {
        let mut rig = Rig::new();
        let got: Arc<Mutex<Vec<OwnedFd>>> = Arc::default();
        let sink = got.clone();
        rig.dev
            .handle()
            .listen(9, Arc::new(move |_, fd| sink.lock().unwrap().push(fd)))
            .unwrap();
        let mut req = guest_header(77, 9, OP_REQUEST);
        req.buf_alloc = 3;
        rig.guest_send(req, &[]);
        assert_eq!(rig.guest_recv().unwrap().0.op, OP_RESPONSE);
        let fd = got.lock().unwrap().pop().unwrap();
        write_fd(&fd, b"abcdef");
        let (_, first) = rig.guest_recv().unwrap();
        assert_eq!(first, b"abc");
        // No more until the guest says it consumed them.
        let a = rig.buf(4096);
        rig.rx.offer(&[(a, 4096, true)]);
        rig.dev.notify(0);
        thread::sleep(Duration::from_millis(100));
        assert_eq!(rig.rx.take_used(), None);
        let mut update = guest_header(77, 9, OP_CREDIT_UPDATE);
        update.buf_alloc = 3;
        update.fwd_cnt = 3;
        rig.guest_send(update, &[]);
        let deadline = Instant::now() + Duration::from_secs(2);
        let len = loop {
            if let Some((_, len)) = rig.rx.take_used() {
                break len;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(5));
        };
        let mut bytes = vec![0; len as usize];
        rig.memory.read(a, &mut bytes).unwrap();
        assert_eq!(&bytes[HEADER..], b"def");
    }
}
