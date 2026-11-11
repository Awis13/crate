// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! X11 keysyms, which RFB's key events carry, as the Linux key codes of a US
//! keyboard: the key that makes the symbol, and whether it takes Shift.

pub const SHIFT_L: u32 = 0xffe1;
pub const SHIFT_R: u32 = 0xffe2;

pub const KEY_LEFTSHIFT: u16 = 42;

/// The key for `keysym` and whether the symbol needs Shift, if a US keyboard
/// has one.
pub fn key(keysym: u32) -> Option<(u16, bool)> {
    if let Some(found) = printable(keysym) {
        return Some(found);
    }
    let code = match keysym {
        0xff08 => 14,  // BackSpace
        0xff09 => 15,  // Tab
        0xff0a => 101, // Linefeed
        0xff0d => 28,  // Return
        0xff13 => 119, // Pause
        0xff14 => 70,  // Scroll_Lock
        0xff15 => 99,  // Sys_Req
        0xff1b => 1,   // Escape
        0xffff => 111, // Delete
        0xff50 => 102, // Home
        0xff51 => 105, // Left
        0xff52 => 103, // Up
        0xff53 => 106, // Right
        0xff54 => 108, // Down
        0xff55 => 104, // Page_Up
        0xff56 => 109, // Page_Down
        0xff57 => 107, // End
        0xff61 => 99,  // Print
        0xff63 => 110, // Insert
        0xff65 => 131, // Undo
        0xff67 => 139, // Menu
        0xff68 => 136, // Find
        0xff6a => 138, // Help
        0xff6b => 119, // Break
        0xff7f => 69,  // Num_Lock
        0xff8d => 96,  // KP_Enter
        0xff95 => 71,  // KP_Home
        0xff96 => 75,  // KP_Left
        0xff97 => 72,  // KP_Up
        0xff98 => 77,  // KP_Right
        0xff99 => 80,  // KP_Down
        0xff9a => 73,  // KP_Page_Up
        0xff9b => 81,  // KP_Page_Down
        0xff9c => 79,  // KP_End
        0xff9d => 76,  // KP_Begin
        0xff9e => 82,  // KP_Insert
        0xff9f => 83,  // KP_Delete
        0xffaa => 55,  // KP_Multiply
        0xffab => 78,  // KP_Add
        0xffac => 121, // KP_Separator
        0xffad => 74,  // KP_Subtract
        0xffae => 83,  // KP_Decimal
        0xffaf => 98,  // KP_Divide
        0xffb0 => 82,  // KP_0
        0xffb1 => 79,
        0xffb2 => 80,
        0xffb3 => 81,
        0xffb4 => 75,
        0xffb5 => 76,
        0xffb6 => 77,
        0xffb7 => 71,
        0xffb8 => 72,
        0xffb9 => 73,  // KP_9
        0xffbd => 117, // KP_Equal
        // F1 to F10, then F11 and F12 apart from them, then F13 to F24.
        0xffbe..=0xffc7 => 59 + (keysym - 0xffbe) as u16,
        0xffc8 => 87,
        0xffc9 => 88,
        0xffca..=0xffd5 => 183 + (keysym - 0xffca) as u16,
        SHIFT_L => KEY_LEFTSHIFT,
        SHIFT_R => 54,
        0xffe3 => 29,  // Control_L
        0xffe4 => 97,  // Control_R
        0xffe5 => 58,  // Caps_Lock
        0xffe7 => 125, // Meta_L
        0xffe8 => 126, // Meta_R
        0xffe9 => 56,  // Alt_L
        0xffea => 100, // Alt_R
        0xffeb => 125, // Super_L
        0xffec => 126, // Super_R
        0xfe03 => 100, // ISO_Level3_Shift (AltGr)
        _ => return None,
    };
    Some((code, false))
}

/// Latin-1's printable ASCII range, where keysyms are the characters.
fn printable(keysym: u32) -> Option<(u16, bool)> {
    let c = char::from_u32(keysym).filter(char::is_ascii)?;
    const LETTERS: &[u8; 26] = &[
        30, 48, 46, 32, 18, 33, 34, 35, 23, 36, 37, 38, 50, 49, 24, 25, 16, 19, 31, 20, 22, 47, 17,
        45, 21, 44,
    ];
    let found = match c {
        'a'..='z' => (LETTERS[c as usize - 'a' as usize] as u16, false),
        'A'..='Z' => (LETTERS[c as usize - 'A' as usize] as u16, true),
        '1'..='9' => (2 + (c as u16 - '1' as u16), false),
        '0' => (11, false),
        ' ' => (57, false),
        '-' => (12, false),
        '=' => (13, false),
        '[' => (26, false),
        ']' => (27, false),
        ';' => (39, false),
        '\'' => (40, false),
        '`' => (41, false),
        '\\' => (43, false),
        ',' => (51, false),
        '.' => (52, false),
        '/' => (53, false),
        '!' => (2, true),
        '@' => (3, true),
        '#' => (4, true),
        '$' => (5, true),
        '%' => (6, true),
        '^' => (7, true),
        '&' => (8, true),
        '*' => (9, true),
        '(' => (10, true),
        ')' => (11, true),
        '_' => (12, true),
        '+' => (13, true),
        '{' => (26, true),
        '}' => (27, true),
        ':' => (39, true),
        '"' => (40, true),
        '~' => (41, true),
        '|' => (43, true),
        '<' => (51, true),
        '>' => (52, true),
        '?' => (53, true),
        _ => return None,
    };
    Some(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_a_us_keyboard() {
        assert_eq!(key('a' as u32), Some((30, false)));
        assert_eq!(key('Q' as u32), Some((16, true)));
        assert_eq!(key('m' as u32), Some((50, false)));
        assert_eq!(key('0' as u32), Some((11, false)));
        assert_eq!(key('9' as u32), Some((10, false)));
        assert_eq!(key(':' as u32), Some((39, true)));
        assert_eq!(key('|' as u32), Some((43, true)));
        assert_eq!(key(0xff0d), Some((28, false)), "Return");
        assert_eq!(key(0xffbe), Some((59, false)), "F1");
        assert_eq!(key(0xffc7), Some((68, false)), "F10");
        assert_eq!(key(0xffc9), Some((88, false)), "F12");
        assert_eq!(key(0xffeb), Some((125, false)), "Super_L");
        assert_eq!(key(0x0100_20ac), None, "the euro sign has no key");
        assert_eq!(key(0xe9), None, "nor has é");
    }
}
