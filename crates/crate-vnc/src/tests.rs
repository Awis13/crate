// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::path::PathBuf;
use std::time::Duration;

use super::*;

#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<String>>,
    refreshes: AtomicUsize,
}

impl InputSink for Recorder {
    fn key(&self, code: u16, down: bool) {
        self.events
            .lock()
            .unwrap()
            .push(format!("key {code} {down}"));
    }
    fn pointer(&self, x: u32, y: u32, buttons: u32) {
        self.events
            .lock()
            .unwrap()
            .push(format!("pointer {x} {y} {buttons}"));
    }
    fn scroll(&self, dx: i32, dy: i32) {
        self.events
            .lock()
            .unwrap()
            .push(format!("scroll {dx} {dy}"));
    }
}

impl DisplayControl for Recorder {
    fn refresh(&self) {
        self.refreshes.fetch_add(1, Ordering::Relaxed);
    }
}

fn socket_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("crate-vnc-{}-{name}.sock", std::process::id()))
}

/// A client that speaks enough RFB to test the server.
struct TestClient {
    s: UnixStream,
}

impl TestClient {
    fn connect(path: &Path) -> (TestClient, u16, u16) {
        let mut s = UnixStream::connect(path).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut version = [0u8; 12];
        s.read_exact(&mut version).unwrap();
        assert_eq!(&version, b"RFB 003.008\n");
        s.write_all(b"RFB 003.008\n").unwrap();
        let mut types = [0u8; 2];
        s.read_exact(&mut types).unwrap();
        assert_eq!(types, [1, 1], "one type: None");
        s.write_all(&[1]).unwrap();
        let mut result = [0u8; 4];
        s.read_exact(&mut result).unwrap();
        assert_eq!(result, [0; 4]);
        s.write_all(&[1]).unwrap();
        let mut init = [0u8; 24];
        s.read_exact(&mut init).unwrap();
        let w = u16::from_be_bytes([init[0], init[1]]);
        let h = u16::from_be_bytes([init[2], init[3]]);
        let name_len = u32::from_be_bytes(init[20..24].try_into().unwrap());
        let mut name = vec![0u8; name_len as usize];
        s.read_exact(&mut name).unwrap();
        assert_eq!(name, b"crate test");
        (TestClient { s }, w, h)
    }

    /// aTerm's: 32 bits, little-endian, red at 0, green 8, blue 16.
    fn set_format(&mut self, red: u8, blue: u8, bpp: u8) {
        let mut m = vec![0u8, 0, 0, 0, bpp, 24, 0, 1];
        let max: u16 = if bpp == 16 { 31 } else { 255 };
        for _ in 0..3 {
            m.extend_from_slice(&max.to_be_bytes());
        }
        let green = if bpp == 16 { 5 } else { 8 };
        m.extend_from_slice(&[red, green, blue, 0, 0, 0]);
        self.s.write_all(&m).unwrap();
    }

    fn set_encodings(&mut self, encodings: &[i32]) {
        let mut m = vec![2u8, 0];
        m.extend_from_slice(&(encodings.len() as u16).to_be_bytes());
        for e in encodings {
            m.extend_from_slice(&e.to_be_bytes());
        }
        self.s.write_all(&m).unwrap();
    }

    fn request(&mut self, incremental: bool, w: u16, h: u16) {
        let mut m = vec![3u8, incremental as u8, 0, 0, 0, 0];
        m.extend_from_slice(&w.to_be_bytes());
        m.extend_from_slice(&h.to_be_bytes());
        self.s.write_all(&m).unwrap();
    }

    /// Reads one update: its rectangles as (x, y, w, h, encoding, data).
    fn update(&mut self, bpp: usize) -> Vec<(u16, u16, u16, u16, i32, Vec<u8>)> {
        let mut head = [0u8; 4];
        self.s.read_exact(&mut head).unwrap();
        assert_eq!(head[0], 0, "a FramebufferUpdate");
        let count = u16::from_be_bytes([head[2], head[3]]);
        (0..count)
            .map(|_| {
                let mut r = [0u8; 12];
                self.s.read_exact(&mut r).unwrap();
                let be = |i: usize| u16::from_be_bytes([r[i], r[i + 1]]);
                let encoding = i32::from_be_bytes(r[8..12].try_into().unwrap());
                let mut data = Vec::new();
                if encoding == 0 {
                    data = vec![0u8; be(4) as usize * be(6) as usize * bpp];
                    self.s.read_exact(&mut data).unwrap();
                }
                (be(0), be(2), be(4), be(6), encoding, data)
            })
            .collect()
    }
}

fn frame(server: &VncServer, width: u32, height: u32, dirty: Rect, pixel: [u8; 4]) {
    let pixels: Vec<u8> = (0..width * height).flat_map(|_| pixel).collect();
    server.frame(&Frame {
        width,
        height,
        stride: width * 4,
        pixels: &pixels,
        dirty,
    });
}

#[test]
fn serves_frames_in_the_clients_format() {
    let server = VncServer::new("crate test", 4, 2);
    let machine = Arc::new(Recorder::default());
    server.attach(machine.clone(), machine.clone());
    let path = socket_path("frames");
    server.listen_unix(&path).unwrap();
    assert!(!server.active());

    let (mut c, w, h) = TestClient::connect(&path);
    assert_eq!((w, h), (4, 2));
    c.set_format(0, 16, 32);
    c.set_encodings(&[0, 1, -223]);
    c.request(false, 4, 2);
    let rects = c.update(4);
    assert_eq!(rects.len(), 1);
    assert_eq!(rects[0].5, vec![0u8; 32], "black to start with");
    assert!(server.active());
    assert_eq!(
        machine.refreshes.load(Ordering::Relaxed),
        1,
        "asked for a frame"
    );

    // An incremental request waits for a change, and gets only that.
    c.request(true, 4, 2);
    frame(&server, 4, 2, Rect::new(1, 1, 2, 1), [10, 20, 30, 0]);
    let rects = c.update(4);
    let r = &rects[0];
    assert_eq!((r.0, r.1, r.2, r.3, r.4), (1, 1, 2, 1, 0));
    assert_eq!(rects[0].5, [10, 20, 30, 0, 10, 20, 30, 0]);

    // Blue at bit 0 and red at 16: B, G, R in memory.
    c.set_format(16, 0, 32);
    c.request(false, 1, 1);
    let rects = c.update(4);
    assert_eq!(
        rects[0].5,
        [0, 0, 0, 0],
        "pixel (0, 0) was not in the frame"
    );
    frame(&server, 4, 2, Rect::new(0, 0, 4, 2), [10, 20, 30, 0]);
    c.request(false, 1, 1);
    assert_eq!(c.update(4)[0].5, [30, 20, 10, 0]);
    // 16 bits, 5-6-5.
    c.set_format(11, 0, 16);
    c.request(false, 1, 1);
    let five = |v: u32| (v * 31 + 127) / 255;
    let v = five(10) << 11 | five(20) << 5 | five(30);
    assert_eq!(c.update(2)[0].5, (v as u16).to_le_bytes());

    // A new size goes out as DesktopSize first.
    c.request(true, 4, 2);
    frame(&server, 8, 6, Rect::new(0, 0, 1, 1), [1, 2, 3, 0]);
    let rects = c.update(2);
    let r = &rects[0];
    assert_eq!((r.0, r.1, r.2, r.3, r.4), (0, 0, 8, 6, -223));

    drop(c);
    std::thread::sleep(Duration::from_millis(100));
    assert!(!server.active(), "inactive once the client has gone");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn passes_keys_and_the_pointer_on() {
    let server = VncServer::new("crate test", 101, 51);
    let machine = Arc::new(Recorder::default());
    server.attach(machine.clone(), machine.clone());
    let path = socket_path("input");
    server.listen_unix(&path).unwrap();
    let (mut c, _, _) = TestClient::connect(&path);
    let key = |c: &mut TestClient, keysym: u32, down: bool| {
        let mut m = vec![4u8, down as u8, 0, 0];
        m.extend_from_slice(&keysym.to_be_bytes());
        c.s.write_all(&m).unwrap();
    };
    let pointer = |c: &mut TestClient, mask: u8, x: u16, y: u16| {
        let mut m = vec![5u8, mask];
        m.extend_from_slice(&x.to_be_bytes());
        m.extend_from_slice(&y.to_be_bytes());
        c.s.write_all(&m).unwrap();
    };
    // A capital letter alone gets Shift; with Shift held it does not.
    key(&mut c, 'A' as u32, true);
    key(&mut c, 'A' as u32, false);
    key(&mut c, 0xffe1, true);
    key(&mut c, ':' as u32, true);
    key(&mut c, ':' as u32, false);
    key(&mut c, 0xffe1, false);
    // The far corner, a right click, a notch down.
    pointer(&mut c, 0, 100, 50);
    pointer(&mut c, 4, 100, 50);
    pointer(&mut c, 0, 100, 50);
    pointer(&mut c, 16, 100, 50);
    pointer(&mut c, 0, 100, 50);
    // Left held when the client goes: it is let go.
    pointer(&mut c, 1, 50, 25);
    drop(c);
    std::thread::sleep(Duration::from_millis(200));
    let events = machine.events.lock().unwrap().clone();
    assert_eq!(
        events,
        [
            "key 42 true",
            "key 30 true",
            "key 30 false",
            "key 42 false",
            "key 42 true",
            "key 39 true",
            "key 39 false",
            "key 42 false",
            "pointer 32767 32767 0",
            "pointer 32767 32767 2",
            "pointer 32767 32767 0",
            "scroll 0 1",
            "pointer 16383 16383 1",
            "pointer 16383 16383 0",
        ]
    );
    let _ = std::fs::remove_file(&path);
}
