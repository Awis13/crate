// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A smoke test of the whole translate-and-run pipeline on a hand-built block,
//! without the native comparison: copied data-processing, the register
//! convention, a store and a load through the memory helper, and a BRK exit.

use std::sync::Arc;

use crate_core::memory::{GuestMemory, HeapRam, RamBlock};
use crate_jit::backend::{CodeCache, Cpu, Stop};

const RAM: u64 = 0x4000_0000;

fn setup(words: &[u32]) -> (Box<Cpu>, CodeCache) {
    let block: Arc<dyn RamBlock> = Arc::new(HeapRam::new(0x10000).unwrap());
    let mem = Arc::new(GuestMemory::new(vec![(RAM, block)]).unwrap());
    for (i, w) in words.iter().enumerate() {
        mem.write_u32(RAM + 4 * i as u64, *w).unwrap();
    }
    let cpu = Cpu::new(mem);
    let cache = CodeCache::new(1 << 20).unwrap();
    (cpu, cache)
}

#[test]
fn copies_arithmetic_and_keeps_registers() {
    // add x0, x0, x1 ; add x2, x0, x0 ; sub x3, x2, x1 ; brk #7
    let (mut cpu, mut cache) = setup(&[0x8b010000, 0x8b000002, 0xcb010043, 0xd42000e0]);
    cpu.state.x[0] = 5;
    cpu.state.x[1] = 3;
    cpu.state.pc = RAM;
    assert_eq!(cpu.run(&mut cache), Stop::Brk(7));
    assert_eq!(cpu.state.x[0], 8);
    assert_eq!(cpu.state.x[2], 16);
    assert_eq!(cpu.state.x[3], 13);
    assert_eq!(cpu.state.pc, RAM + 12, "PC at the BRK");
}

#[test]
fn condition_flags_survive() {
    // subs xzr, x0, x1 (cmp) ; cset x2, eq (csinc) ; brk #0
    // cmp x0, x1 => flags; cset x2, eq
    let (mut cpu, mut cache) = setup(&[0xeb01001f, 0x9a9f17e2, 0xd4200000]);
    cpu.state.x[0] = 42;
    cpu.state.x[1] = 42;
    cpu.state.pc = RAM;
    assert_eq!(cpu.run(&mut cache), Stop::Brk(0));
    assert_eq!(cpu.state.x[2], 1, "equal, so cset gives 1");
}

#[test]
fn stack_pointer_is_virtual() {
    // mov x5, sp (add x5, sp, #0) ; sub sp, sp, #32 ; add x6, sp, #0 ; brk #0
    let (mut cpu, mut cache) = setup(&[0x910003e5, 0xd10083ff, 0x910003e6, 0xd4200000]);
    cpu.state.sp = 0x8000;
    cpu.state.pc = RAM;
    assert_eq!(cpu.run(&mut cache), Stop::Brk(0));
    assert_eq!(cpu.state.x[5], 0x8000, "read the guest SP");
    assert_eq!(cpu.state.sp, 0x8000 - 32, "SP was adjusted");
    assert_eq!(cpu.state.x[6], 0x8000 - 32);
}

#[test]
fn store_then_load_through_the_helper() {
    // The scratch word lives at RAM+0x800; x1 points at it.
    // str x0, [x1] ; ldr x2, [x1] ; ldrb w3, [x1] ; brk #0
    let (mut cpu, mut cache) = setup(&[0xf9000020, 0xf9400022, 0x39400023, 0xd4200000]);
    cpu.state.x[0] = 0x1122_3344_5566_7788;
    cpu.state.x[1] = RAM + 0x800;
    cpu.state.pc = RAM;
    assert_eq!(cpu.run(&mut cache), Stop::Brk(0));
    assert_eq!(cpu.state.x[2], 0x1122_3344_5566_7788);
    assert_eq!(cpu.state.x[3], 0x88, "a byte load zero-extends");
    assert_eq!(
        cpu.mem.read_u64(RAM + 0x800).unwrap(),
        0x1122_3344_5566_7788
    );
}

#[test]
fn a_branch_is_followed() {
    // b .+8 ; brk #99 (skipped) ; brk #5
    let (mut cpu, mut cache) = setup(&[0x14000002, 0xd4200c60, 0xd42000a0]);
    cpu.state.pc = RAM;
    assert_eq!(cpu.run(&mut cache), Stop::Brk(5));
}
