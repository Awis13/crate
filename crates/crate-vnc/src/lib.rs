// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! An optional VNC server for a crate display (RFC 6143, the Remote
//! Framebuffer protocol, version 3.8; 3.3 and 3.7 clients too): no
//! authentication, Raw encoding, any true-colour pixel format the client sets,
//! and DesktopSize, so a guest's mode change reaches the client.
//!
//! The server is a `DisplaySink`: it keeps a copy of the screen, and each
//! client the part that changed since it was last sent. It is active, so the
//! GPU makes frames, only while a client is connected. Key events become
//! Linux key codes of a US keyboard and pointer events tablet positions, for
//! the machine's `InputSink`.
//!
//! Every client has two threads: one reads its messages, the other waits for
//! an update request and something to send. It listens on a Unix socket or a
//! TCP port; it is meant for the app's own use, and has no authentication, so
//! TCP should stay on the loopback address.

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate_core::display::{
    BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT, DisplayControl, DisplaySink, Frame, InputSink,
    POINTER_MAX, Rect,
};
use crate_core::{Error, Result, debug, info};

pub mod keysym;

const ENCODING_RAW: i32 = 0;
const ENCODING_DESKTOP_SIZE: i32 = -223;

/// A client's pixel format (RFC 6143, 7.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelFormat {
    pub bits_per_pixel: u8,
    pub depth: u8,
    pub big_endian: bool,
    pub true_colour: bool,
    pub max: [u16; 3],
    pub shift: [u8; 3],
}

impl PixelFormat {
    /// The screen's own: R, G, B, unused in memory.
    pub const NATIVE: PixelFormat = PixelFormat {
        bits_per_pixel: 32,
        depth: 24,
        big_endian: false,
        true_colour: true,
        max: [255, 255, 255],
        shift: [0, 8, 16],
    };

    fn bytes(&self) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[0] = self.bits_per_pixel;
        b[1] = self.depth;
        b[2] = self.big_endian as u8;
        b[3] = self.true_colour as u8;
        for i in 0..3 {
            b[4 + 2 * i..6 + 2 * i].copy_from_slice(&self.max[i].to_be_bytes());
            b[10 + i] = self.shift[i];
        }
        b
    }

    fn parse(b: &[u8; 16]) -> Option<PixelFormat> {
        let format = PixelFormat {
            bits_per_pixel: b[0],
            depth: b[1],
            big_endian: b[2] != 0,
            true_colour: b[3] != 0,
            max: [0, 1, 2].map(|i| u16::from_be_bytes([b[4 + 2 * i], b[5 + 2 * i]])),
            shift: [b[10], b[11], b[12]],
        };
        let fits = |i: usize| {
            let bits = 16 - format.max[i].leading_zeros();
            format.shift[i] as u32 + bits <= format.bits_per_pixel as u32
        };
        // Colour maps are not offered.
        (matches!(format.bits_per_pixel, 8 | 16 | 32) && format.true_colour && (0..3).all(fits))
            .then_some(format)
    }

    fn pixel_bytes(&self) -> usize {
        self.bits_per_pixel as usize / 8
    }

    /// Appends `pixels` (R, G, B, unused) in this format.
    fn convert(&self, pixels: &[u8], out: &mut Vec<u8>) {
        if *self == Self::NATIVE {
            out.extend_from_slice(pixels);
            return;
        }
        let n = self.pixel_bytes();
        let scale =
            |v: u8, i: usize| ((v as u32 * self.max[i] as u32 + 127) / 255) << self.shift[i];
        for p in pixels.chunks_exact(4) {
            let v = scale(p[0], 0) | scale(p[1], 1) | scale(p[2], 2);
            let bytes = if self.big_endian {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            };
            if self.big_endian {
                out.extend_from_slice(&bytes[4 - n..]);
            } else {
                out.extend_from_slice(&bytes[..n]);
            }
        }
    }
}

/// What a client is owed.
#[derive(Default)]
struct Owed {
    /// Changed since the client was last sent it.
    dirty: Rect,
    /// The screen changed size since the client last heard.
    resized: bool,
    /// The client's outstanding request: incremental, and the area.
    request: Option<(bool, Rect)>,
    closed: bool,
}

struct Screen {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    /// The copy is up to date: false from when the first client arrives (the
    /// GPU made no frames for nobody) until the frame that asks for brings it.
    fresh: bool,
    clients: Vec<(u64, Owed)>,
    next_client: u64,
}

impl Screen {
    fn client(&mut self, id: u64) -> Option<&mut Owed> {
        self.clients
            .iter_mut()
            .find(|(i, _)| *i == id)
            .map(|(_, o)| o)
    }
}

/// The parts the machine supplies once it exists.
#[derive(Default)]
struct Machine {
    control: Option<Arc<dyn DisplayControl>>,
    input: Option<Arc<dyn InputSink>>,
}

struct Shared {
    name: String,
    screen: Mutex<Screen>,
    changed: Condvar,
    connected: AtomicUsize,
    machine: Mutex<Machine>,
}

pub struct VncServer {
    shared: Arc<Shared>,
}

impl VncServer {
    /// A server for a screen of `width` x `height` until the first frame says
    /// otherwise; clients see `name` as the desktop's.
    pub fn new(name: &str, width: u32, height: u32) -> Arc<Self> {
        Arc::new(VncServer {
            shared: Arc::new(Shared {
                name: name.into(),
                screen: Mutex::new(Screen {
                    width,
                    height,
                    pixels: vec![0; width as usize * height as usize * 4],
                    fresh: false,
                    clients: Vec::new(),
                    next_client: 0,
                }),
                changed: Condvar::new(),
                connected: AtomicUsize::new(0),
                machine: Mutex::default(),
            }),
        })
    }

    /// Where input goes and whom to ask for whole frames, once the machine
    /// exists.
    pub fn attach(&self, control: Arc<dyn DisplayControl>, input: Arc<dyn InputSink>) {
        let mut m = self.shared.machine.lock().unwrap();
        m.control = Some(control.clone());
        m.input = Some(input);
        drop(m);
        if self.shared.connected.load(Ordering::Acquire) > 0 {
            control.refresh();
        }
    }

    /// Serves clients on a Unix socket at `path`, replacing whatever is there;
    /// only this user may connect.
    pub fn listen_unix(&self, path: &Path) -> Result<()> {
        let _ = std::fs::remove_file(path);
        let listener = UnixListener::bind(path)
            .map_err(|e| Error::io(format!("listening on {}", path.display()), e))?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| Error::io(format!("restricting {}", path.display()), e))?;
        info!("VNC on {}", path.display());
        let shared = self.shared.clone();
        spawn("crate-vnc-listen", move || {
            for stream in listener.incoming().flatten() {
                if let Ok(writer) = stream.try_clone() {
                    serve(&shared, Box::new(stream), Box::new(writer));
                }
            }
        })
    }

    /// Serves clients on a TCP address, such as 127.0.0.1:5905.
    pub fn listen_tcp(&self, address: SocketAddr) -> Result<()> {
        let listener = TcpListener::bind(address)
            .map_err(|e| Error::io(format!("listening on {address}"), e))?;
        info!("VNC on {address}");
        let shared = self.shared.clone();
        spawn("crate-vnc-listen", move || {
            for stream in listener.incoming().flatten() {
                let _ = stream.set_nodelay(true);
                if let Ok(writer) = stream.try_clone() {
                    serve(&shared, Box::new(stream), Box::new(writer));
                }
            }
        })
    }

    /// How many clients are connected.
    pub fn clients(&self) -> usize {
        self.shared.connected.load(Ordering::Acquire)
    }
}

impl DisplaySink for VncServer {
    fn active(&self) -> bool {
        self.clients() > 0
    }

    fn frame(&self, frame: &Frame) {
        let mut screen = self.shared.screen.lock().unwrap();
        let resized = (screen.width, screen.height) != (frame.width, frame.height);
        let dirty = if resized {
            screen.width = frame.width;
            screen.height = frame.height;
            screen.pixels = vec![0; frame.width as usize * frame.height as usize * 4];
            Rect::new(0, 0, frame.width, frame.height)
        } else {
            frame.dirty.clip(frame.width, frame.height)
        };
        let row = dirty.width as usize * 4;
        let stride = frame.width as usize * 4;
        for line in 0..dirty.height as usize {
            let y = dirty.y as usize + line;
            let from = y * frame.stride as usize + dirty.x as usize * 4;
            let to = y * stride + dirty.x as usize * 4;
            if let Some(src) = frame.pixels.get(from..from + row) {
                screen.pixels[to..to + row].copy_from_slice(src);
            }
        }
        screen.fresh = true;
        for (_, owed) in &mut screen.clients {
            owed.dirty = owed.dirty.union(&dirty);
            owed.resized |= resized;
        }
        self.shared.changed.notify_all();
    }
}

fn spawn(name: &str, body: impl FnOnce() + Send + 'static) -> Result<()> {
    std::thread::Builder::new()
        .name(name.into())
        .spawn(body)
        .map(|_| ())
        .map_err(|e| Error::NoMemory(format!("a VNC thread: {e}")))
}

trait Stream: Read + Write + Send {
    fn shut(&self);
}

impl Stream for UnixStream {
    fn shut(&self) {
        let _ = self.shutdown(std::net::Shutdown::Both);
    }
}

impl Stream for TcpStream {
    fn shut(&self) {
        let _ = self.shutdown(std::net::Shutdown::Both);
    }
}

/// What one client's two threads share.
struct Client {
    id: u64,
    format: Mutex<PixelFormat>,
    desktop_size: std::sync::atomic::AtomicBool,
}

fn serve(shared: &Arc<Shared>, reader: Box<dyn Stream>, writer: Box<dyn Stream>) {
    let shared = shared.clone();
    let _ = spawn("crate-vnc-client", move || {
        let mut reader = reader;
        let mut writer = writer;
        // Active from now, so the frame asked for arrives before the client
        // is told the screen's size.
        if shared.connected.fetch_add(1, Ordering::AcqRel) == 0 {
            shared.screen.lock().unwrap().fresh = false;
            let control = shared.machine.lock().unwrap().control.clone();
            if let Some(control) = control {
                control.refresh();
            }
        }
        let Ok((width, height)) = handshake(&shared, &mut reader, &mut writer) else {
            reader.shut();
            shared.connected.fetch_sub(1, Ordering::AcqRel);
            return;
        };
        let id = {
            let mut screen = shared.screen.lock().unwrap();
            let id = screen.next_client;
            screen.next_client += 1;
            let resized = (screen.width, screen.height) != (width, height);
            screen.clients.push((
                id,
                Owed {
                    resized,
                    ..Owed::default()
                },
            ));
            id
        };
        let client = Arc::new(Client {
            id,
            format: Mutex::new(PixelFormat::NATIVE),
            desktop_size: false.into(),
        });
        debug!("VNC client {id} connected");
        let sender = {
            let (shared, client) = (shared.clone(), client.clone());
            std::thread::Builder::new()
                .name("crate-vnc-send".into())
                .spawn(move || send_updates(&shared, &client, writer))
        };
        let mut input = InputState::default();
        let result = read_messages(&shared, &client, &mut reader, &mut input);
        if let Err(e) = result {
            debug!("VNC client {id}: {e}");
        }
        input.release(&shared);
        {
            let mut screen = shared.screen.lock().unwrap();
            if let Some(owed) = screen.client(id) {
                owed.closed = true;
            }
            shared.changed.notify_all();
        }
        reader.shut();
        if let Ok(sender) = sender {
            let _ = sender.join();
        }
        shared
            .screen
            .lock()
            .unwrap()
            .clients
            .retain(|(i, _)| *i != id);
        shared.connected.fetch_sub(1, Ordering::AcqRel);
        debug!("VNC client {id} gone");
    });
}

fn read_exact<const N: usize>(r: &mut dyn Read) -> io::Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}

/// The version and security exchange and the init messages; returns the size
/// the client was told.
fn handshake(
    shared: &Shared,
    r: &mut Box<dyn Stream>,
    w: &mut Box<dyn Stream>,
) -> io::Result<(u32, u32)> {
    w.write_all(b"RFB 003.008\n")?;
    let version: [u8; 12] = read_exact(r.as_mut())?;
    let minor = match &version {
        b"RFB 003.003\n" => 3,
        b"RFB 003.007\n" => 7,
        v if v.starts_with(b"RFB 003.") => 8,
        _ => return Err(io::Error::other("not an RFB client")),
    };
    if minor == 3 {
        // The server picks: None.
        w.write_all(&1u32.to_be_bytes())?;
    } else {
        w.write_all(&[1, 1])?;
        let [chosen] = read_exact::<1>(r.as_mut())?;
        if chosen != 1 {
            return Err(io::Error::other(
                "the client wants a security type not offered",
            ));
        }
        if minor == 8 {
            w.write_all(&0u32.to_be_bytes())?;
        }
    }
    let _shared_flag: [u8; 1] = read_exact(r.as_mut())?;
    let (width, height) = {
        let screen = wait_fresh(shared);
        (screen.width, screen.height)
    };
    let mut init = Vec::new();
    init.extend_from_slice(&(width as u16).to_be_bytes());
    init.extend_from_slice(&(height as u16).to_be_bytes());
    init.extend_from_slice(&PixelFormat::NATIVE.bytes());
    init.extend_from_slice(&(shared.name.len() as u32).to_be_bytes());
    init.extend_from_slice(shared.name.as_bytes());
    w.write_all(&init)?;
    Ok((width, height))
}

/// Keys and buttons a client holds, to let go of when it leaves.
#[derive(Default)]
struct InputState {
    buttons: u8,
    /// In screen pixels, as sent.
    position: Option<(u16, u16)>,
    /// As passed on, 0 to POINTER_MAX.
    scaled: (u32, u32),
    /// Keys down, and whether Shift was pressed for each.
    keys: Vec<(u32, u16, bool)>,
    shift: usize,
}

impl InputState {
    fn release(&mut self, shared: &Shared) {
        let Some(input) = shared.machine.lock().unwrap().input.clone() else {
            return;
        };
        for (_, code, synthetic) in self.keys.drain(..).rev() {
            input.key(code, false);
            if synthetic {
                input.key(keysym::KEY_LEFTSHIFT, false);
            }
        }
        if self.buttons & 0b111 != 0 {
            input.pointer(self.scaled.0, self.scaled.1, 0);
        }
    }
}

fn read_messages(
    shared: &Shared,
    client: &Client,
    r: &mut Box<dyn Stream>,
    input: &mut InputState,
) -> io::Result<()> {
    loop {
        let [kind] = read_exact::<1>(r.as_mut())?;
        match kind {
            0 => {
                let b: [u8; 19] = read_exact(r.as_mut())?;
                let format = PixelFormat::parse(b[3..19].try_into().unwrap())
                    .ok_or_else(|| io::Error::other("a pixel format that is not offered"))?;
                *client.format.lock().unwrap() = format;
            }
            2 => {
                let b: [u8; 3] = read_exact(r.as_mut())?;
                let count = u16::from_be_bytes([b[1], b[2]]);
                let mut desktop_size = false;
                for _ in 0..count {
                    let e = i32::from_be_bytes(read_exact(r.as_mut())?);
                    desktop_size |= e == ENCODING_DESKTOP_SIZE;
                }
                client.desktop_size.store(desktop_size, Ordering::Release);
            }
            3 => {
                let b: [u8; 9] = read_exact(r.as_mut())?;
                let be = |i: usize| u16::from_be_bytes([b[i], b[i + 1]]) as u32;
                let area = Rect::new(be(1), be(3), be(5), be(7));
                let mut screen = shared.screen.lock().unwrap();
                if let Some(owed) = screen.client(client.id) {
                    owed.request = Some((b[0] != 0, area));
                }
                shared.changed.notify_all();
            }
            4 => {
                let b: [u8; 7] = read_exact(r.as_mut())?;
                let keysym = u32::from_be_bytes([b[3], b[4], b[5], b[6]]);
                key_event(shared, input, keysym, b[0] != 0);
            }
            5 => {
                let b: [u8; 5] = read_exact(r.as_mut())?;
                let (x, y) = (
                    u16::from_be_bytes([b[1], b[2]]),
                    u16::from_be_bytes([b[3], b[4]]),
                );
                pointer_event(shared, input, b[0], x, y);
            }
            6 => {
                let b: [u8; 7] = read_exact(r.as_mut())?;
                let length = u32::from_be_bytes([b[3], b[4], b[5], b[6]]);
                io::copy(&mut r.as_mut().take(length as u64), &mut io::sink())?;
            }
            other => return Err(io::Error::other(format!("unknown message {other}"))),
        }
    }
}

fn key_event(shared: &Shared, state: &mut InputState, keysym: u32, down: bool) {
    let Some(input) = shared.machine.lock().unwrap().input.clone() else {
        return;
    };
    let Some((code, needs_shift)) = keysym::key(keysym) else {
        debug!("VNC: no key for keysym {keysym:#x}");
        return;
    };
    if keysym == keysym::SHIFT_L || keysym == keysym::SHIFT_R {
        if down {
            state.shift += 1;
        } else {
            state.shift = state.shift.saturating_sub(1);
        }
    }
    if down {
        // A symbol that takes Shift, sent without it (a capital letter, say):
        // press Shift around it.
        let synthetic = needs_shift && state.shift == 0;
        if synthetic {
            input.key(keysym::KEY_LEFTSHIFT, true);
        }
        input.key(code, true);
        state.keys.push((keysym, code, synthetic));
    } else {
        let found = state.keys.iter().rposition(|&(k, _, _)| k == keysym);
        let synthetic = found.is_some_and(|i| state.keys.remove(i).2);
        input.key(code, false);
        if synthetic {
            input.key(keysym::KEY_LEFTSHIFT, false);
        }
    }
}

fn pointer_event(shared: &Shared, state: &mut InputState, mask: u8, x: u16, y: u16) {
    let Some(input) = shared.machine.lock().unwrap().input.clone() else {
        return;
    };
    let (width, height) = {
        let screen = shared.screen.lock().unwrap();
        (screen.width, screen.height)
    };
    let scale = |v: u16, size: u32| {
        (v as u32).min(size.saturating_sub(1)) * POINTER_MAX / size.saturating_sub(1).max(1)
    };
    // RFB: 1 left, 2 middle, 4 right, then the wheel as buttons 4 to 7.
    let mut buttons = 0;
    if mask & 1 != 0 {
        buttons |= BUTTON_LEFT;
    }
    if mask & 2 != 0 {
        buttons |= BUTTON_MIDDLE;
    }
    if mask & 4 != 0 {
        buttons |= BUTTON_RIGHT;
    }
    if state.position != Some((x, y)) || (state.buttons ^ mask) & 0b111 != 0 {
        state.scaled = (scale(x, width), scale(y, height));
        input.pointer(state.scaled.0, state.scaled.1, buttons);
    }
    let pressed = mask & !state.buttons;
    let (mut dx, mut dy) = (0, 0);
    if pressed & 8 != 0 {
        dy -= 1;
    }
    if pressed & 16 != 0 {
        dy += 1;
    }
    if pressed & 32 != 0 {
        dx -= 1;
    }
    if pressed & 64 != 0 {
        dx += 1;
    }
    if dx != 0 || dy != 0 {
        input.scroll(dx, dy);
    }
    state.buttons = mask;
    state.position = Some((x, y));
}

/// Waits for a request and something to answer it with, and sends it.
/// The screen, once the frame asked for when the first client came has
/// arrived, or a while has passed: it never comes while the guest shows
/// nothing.
fn wait_fresh(shared: &Shared) -> std::sync::MutexGuard<'_, Screen> {
    let patience = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut screen = shared.screen.lock().unwrap();
    while !screen.fresh {
        let now = std::time::Instant::now();
        if now >= patience {
            break;
        }
        screen = shared
            .changed
            .wait_timeout(screen, patience - now)
            .unwrap()
            .0;
    }
    screen
}

fn send_updates(shared: &Shared, client: &Client, mut w: Box<dyn Stream>) {
    let mut message = Vec::new();
    loop {
        message.clear();
        {
            let mut screen = wait_fresh(shared);
            let desktop_size = client.desktop_size.load(Ordering::Acquire);
            loop {
                let Some(owed) = screen.client(client.id) else {
                    return;
                };
                if owed.closed {
                    return;
                }
                if let Some((incremental, _)) = owed.request {
                    if (owed.resized && desktop_size) || !incremental || !owed.dirty.is_empty() {
                        break;
                    }
                }
                screen = shared.changed.wait(screen).unwrap();
            }
            let (width, height) = (screen.width, screen.height);
            let owed = screen.client(client.id).unwrap();
            let (incremental, area) = owed.request.take().unwrap();
            if owed.resized && desktop_size {
                owed.resized = false;
                // The new size as a rectangle of its own; the client asks for
                // the whole screen again.
                owed.dirty = Rect::default();
                message.extend_from_slice(&[0, 0]);
                message.extend_from_slice(&1u16.to_be_bytes());
                rect_header(
                    &mut message,
                    Rect::new(0, 0, width, height),
                    ENCODING_DESKTOP_SIZE,
                );
            } else {
                owed.resized = false;
                let wanted = if incremental { owed.dirty } else { area };
                let send = wanted.intersect(&area).clip(width, height);
                owed.dirty = Rect::default();
                message.extend_from_slice(&[0, 0]);
                if send.is_empty() {
                    // Nothing in the area asked for: an empty update, so a
                    // client that waits for one is not left waiting.
                    message.extend_from_slice(&0u16.to_be_bytes());
                } else {
                    message.extend_from_slice(&1u16.to_be_bytes());
                    rect_header(&mut message, send, ENCODING_RAW);
                    let format = *client.format.lock().unwrap();
                    let stride = width as usize * 4;
                    let row = send.width as usize * 4;
                    message
                        .reserve(send.width as usize * send.height as usize * format.pixel_bytes());
                    for line in 0..send.height as usize {
                        let at = (send.y as usize + line) * stride + send.x as usize * 4;
                        format.convert(&screen.pixels[at..at + row], &mut message);
                    }
                }
            }
        }
        if w.write_all(&message).is_err() {
            w.shut();
            return;
        }
    }
}

fn rect_header(out: &mut Vec<u8>, r: Rect, encoding: i32) {
    for v in [r.x, r.y, r.width, r.height] {
        out.extend_from_slice(&(v as u16).to_be_bytes());
    }
    out.extend_from_slice(&encoding.to_be_bytes());
}

#[cfg(test)]
mod tests;
