#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""A small RFB client for a crate display: screenshots and input.

    tools/vnc.py SERVER shot out.png        whole screen as PNG
    tools/vnc.py SERVER size                the screen's size
    tools/vnc.py SERVER click X Y           left click at a screen pixel
    tools/vnc.py SERVER rclick X Y          right click
    tools/vnc.py SERVER drag X0 Y0 X1 Y1    left button held from one to the other
    tools/vnc.py SERVER scroll X Y N        N wheel notches, positive down
    tools/vnc.py SERVER type TEXT           types TEXT (US keyboard)
    tools/vnc.py SERVER key KEYSYM...       presses and releases keysyms (0xff0d
                                            or a name: Return, Escape, Tab, ...);
                                            a+b holds a while pressing b
    tools/vnc.py SERVER watch SECONDS       counts updates for that long

SERVER is a Unix socket path or HOST:PORT. It asks for the pixel format aTerm's
client asks for (red at bit 0, green 8, blue 16) and the encodings it offers
(Raw, CopyRect, DesktopSize), so it sees what aTerm will.
"""
import os
import socket
import struct
import sys
import time
import zlib

NAMES = {
    "Return": 0xFF0D, "Escape": 0xFF1B, "Tab": 0xFF09, "BackSpace": 0xFF08,
    "Delete": 0xFFFF, "Left": 0xFF51, "Up": 0xFF52, "Right": 0xFF53, "Down": 0xFF54,
    "Home": 0xFF50, "End": 0xFF57, "Shift": 0xFFE1, "Control": 0xFFE3,
    "Alt": 0xFFE9, "Super": 0xFFEB, "space": 0x20,
}
for i in range(12):
    NAMES[f"F{i + 1}"] = 0xFFBE + i


def recv_exact(s, n):
    data = bytearray()
    while len(data) < n:
        chunk = s.recv(min(n - len(data), 1 << 20))
        if not chunk:
            raise ConnectionError("the server closed the connection")
        data += chunk
    return bytes(data)


class Client:
    def __init__(self, server):
        if os.path.exists(server) or "/" in server:
            self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.s.connect(server)
        else:
            host, port = server.rsplit(":", 1)
            self.s = socket.create_connection((host, int(port)))
        self.s.settimeout(30)
        version = recv_exact(self.s, 12)
        if not version.startswith(b"RFB 003."):
            raise ConnectionError("not an RFB server")
        self.s.sendall(b"RFB 003.008\n")
        count = recv_exact(self.s, 1)[0]
        if 1 not in recv_exact(self.s, count):
            raise ConnectionError("no 'None' security type")
        self.s.sendall(b"\x01")
        if struct.unpack(">I", recv_exact(self.s, 4))[0] != 0:
            raise ConnectionError("security handshake failed")
        self.s.sendall(b"\x01")
        self.width, self.height = struct.unpack(">HH", recv_exact(self.s, 4))
        recv_exact(self.s, 16)
        recv_exact(self.s, struct.unpack(">I", recv_exact(self.s, 4))[0])
        # 32 bits, little-endian, true colour, red at 0, green 8, blue 16.
        self.s.sendall(struct.pack(">BxxxBBBBHHHBBBxxx", 0, 32, 24, 0, 1,
                                   255, 255, 255, 0, 8, 16))
        self.s.sendall(struct.pack(">BxHiii", 2, 3, 0, 1, -223))
        self.pixels = bytearray(self.width * self.height * 4)

    def request(self, incremental):
        self.s.sendall(struct.pack(">BBHHHH", 3, 1 if incremental else 0,
                                   0, 0, self.width, self.height))

    def read_update(self):
        """Reads one update; returns how many pixels it painted."""
        kind = recv_exact(self.s, 1)[0]
        if kind != 0:
            raise ConnectionError(f"unexpected message {kind}")
        recv_exact(self.s, 1)
        (count,) = struct.unpack(">H", recv_exact(self.s, 2))
        painted = 0
        for _ in range(count):
            x, y, w, h, enc = struct.unpack(">HHHHi", recv_exact(self.s, 12))
            if enc == 0:
                data = recv_exact(self.s, w * h * 4)
                for row in range(h):
                    at = ((y + row) * self.width + x) * 4
                    self.pixels[at:at + w * 4] = data[row * w * 4:(row + 1) * w * 4]
                painted += w * h
            elif enc == 1:
                sx, sy = struct.unpack(">HH", recv_exact(self.s, 4))
                rows = [bytes(self.pixels[((sy + r) * self.width + sx) * 4:
                                          ((sy + r) * self.width + sx + w) * 4])
                        for r in range(h)]
                for r, data in enumerate(rows):
                    at = ((y + r) * self.width + x) * 4
                    self.pixels[at:at + w * 4] = data
            elif enc == -223:
                self.width, self.height = w, h
                self.pixels = bytearray(w * h * 4)
                return -1
            else:
                raise ConnectionError(f"encoding {enc} was not offered")
        return painted

    def screen(self):
        self.request(False)
        while self.read_update() < 0:
            self.request(False)

    def pointer(self, x, y, buttons):
        self.s.sendall(struct.pack(">BBHH", 5, buttons, x, y))

    def key(self, keysym, down):
        self.s.sendall(struct.pack(">BBxxI", 4, 1 if down else 0, keysym))


def png(path, width, height, rgbx):
    rows = b"".join(b"\x00" + bytes(rgbx[y * width * 4:(y + 1) * width * 4])
                    for y in range(height))
    # Colour type 6 (RGBA) with the unused byte forced opaque.
    raw = bytearray(rows)
    for y in range(height):
        base = y * (width * 4 + 1) + 1
        raw[base + 3:base + width * 4:4] = b"\xff" * width

    def chunk(kind, data):
        c = struct.pack(">I", len(data)) + kind + data
        return c + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw), 6)))
        f.write(chunk(b"IEND", b""))


def keysym_of(name):
    if name in NAMES:
        return NAMES[name]
    if name.startswith("0x"):
        return int(name, 16)
    if len(name) == 1:
        return ord(name)
    raise SystemExit(f"unknown key {name}")


SHIFTED = set('~!@#$%^&*()_+{}|:"<>?')


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    c = Client(argv[0])
    command, args = argv[1], argv[2:]
    if command == "size":
        print(f"{c.width}x{c.height}")
    elif command == "shot":
        c.screen()
        png(args[0], c.width, c.height, c.pixels)
        samples = range(0, len(c.pixels), 4 * 97)
        lit = sum(1 for i in samples if any(c.pixels[i:i + 3]))
        print(f"{c.width}x{c.height} -> {args[0]} ({100 * lit // max(1, len(samples))}% lit)")
    elif command in ("click", "rclick"):
        x, y = int(args[0]), int(args[1])
        button = 1 if command == "click" else 4
        c.pointer(x, y, 0)
        time.sleep(0.05)
        c.pointer(x, y, button)
        time.sleep(0.08)
        c.pointer(x, y, 0)
    elif command == "drag":
        x0, y0, x1, y1 = map(int, args[:4])
        c.pointer(x0, y0, 0)
        time.sleep(0.05)
        c.pointer(x0, y0, 1)
        for i in range(1, 11):
            c.pointer(x0 + (x1 - x0) * i // 10, y0 + (y1 - y0) * i // 10, 1)
            time.sleep(0.03)
        c.pointer(x1, y1, 0)
    elif command == "scroll":
        x, y, n = int(args[0]), int(args[1]), int(args[2])
        c.pointer(x, y, 0)
        mask = 16 if n > 0 else 8
        for _ in range(abs(n)):
            c.pointer(x, y, mask)
            c.pointer(x, y, 0)
            time.sleep(0.03)
    elif command == "type":
        # As aTerm sends text: Shift around shifted symbols, capitals alone.
        for ch in " ".join(args):
            if ch == "\n":
                keysym = 0xFF0D
            else:
                keysym = ord(ch)
            if ch in SHIFTED:
                c.key(0xFFE1, True)
            c.key(keysym, True)
            c.key(keysym, False)
            if ch in SHIFTED:
                c.key(0xFFE1, False)
            time.sleep(0.01)
    elif command == "key":
        for combo in args:
            keys = [keysym_of(k) for k in combo.split("+")]
            for k in keys:
                c.key(k, True)
            for k in reversed(keys):
                c.key(k, False)
            time.sleep(0.05)
    elif command == "watch":
        seconds = float(args[0])
        c.screen()
        end = time.time() + seconds
        updates = pixels = 0
        c.s.settimeout(0.5)
        c.request(True)
        while time.time() < end:
            try:
                n = c.read_update()
            except socket.timeout:
                continue
            if n < 0:
                c.request(False)
                continue
            updates += 1
            pixels += n
            c.request(True)
        print(f"{updates} updates, {pixels} pixels in {seconds} s")
    else:
        print(__doc__.strip())
        return 2
    time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
