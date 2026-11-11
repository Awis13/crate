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

#[test]
fn x18_is_a_virtual_register() {
    // The host reserves x18 (the OS clobbers it), so guest X18 must live in the
    // state block, not host x18. Exercise a read, a write, arithmetic, X18 as the
    // Rm of a shifted-register add, and a store through the memory helper:
    //   movz x18,#0x1234 ; mov x5,x18 ; add x18,x18,#1 ; add x6,x7,x18 ;
    //   str x18,[x1] ; brk #0
    let (mut cpu, mut cache) = setup(&[
        0xd2824692, 0xaa1203e5, 0x91000652, 0x8b1200e6, 0xf9000032, 0xd4200000,
    ]);
    cpu.state.x[7] = 0x100;
    cpu.state.x[1] = RAM + 0x800;
    cpu.state.pc = RAM;
    assert_eq!(cpu.run(&mut cache), Stop::Brk(0));
    assert_eq!(cpu.state.x[18], 0x1235, "X18 written then incremented");
    assert_eq!(cpu.state.x[5], 0x1234, "mov x5,x18 read X18 before the add");
    assert_eq!(cpu.state.x[6], 0x1335, "X18 as the Rm of add x6,x7,x18");
    assert_eq!(
        cpu.mem.read_u64(RAM + 0x800).unwrap(),
        0x1235,
        "str x18 used X18"
    );
}

#[test]
fn unprivileged_load_store_round_trips() {
    // With the MMU off (flat map), STTR/LDTR behave as ordinary accesses -- the
    // point is that they are decoded and run, not rejected as undefined:
    //   sttr x0,[x1] ; ldtr x2,[x1] ; brk #0
    let (mut cpu, mut cache) = setup(&[0xf8000820, 0xf8400822, 0xd4200000]);
    cpu.state.x[0] = 0xdead_beef_0bad_f00d;
    cpu.state.x[1] = RAM + 0x800;
    cpu.state.pc = RAM;
    assert_eq!(cpu.run(&mut cache), Stop::Brk(0));
    assert_eq!(
        cpu.state.x[2], 0xdead_beef_0bad_f00d,
        "LDTR read what STTR wrote"
    );
    assert_eq!(
        cpu.mem.read_u64(RAM + 0x800).unwrap(),
        0xdead_beef_0bad_f00d
    );
}

#[test]
fn x18_is_read_from_the_state() {
    // mov x5, x18 ; add x18, sp, #8 ; brk #0 -- X18 as a plain source, and as the
    // destination of an SP-form add (which also keeps X18 in the state).
    let (mut cpu, mut cache) = setup(&[0xaa1203e5, 0x910023f2, 0xd4200000]);
    cpu.state.x[18] = 0xcafe;
    cpu.state.sp = 0x9000;
    cpu.state.pc = RAM;
    assert_eq!(cpu.run(&mut cache), Stop::Brk(0));
    assert_eq!(cpu.state.x[5], 0xcafe, "read the guest X18");
    assert_eq!(
        cpu.state.x[18], 0x9008,
        "add x18, sp, #8 wrote X18 in the state"
    );
}

#[test]
fn shifted_value_then_store() {
    // mov w2,#1 ; lsl w2,w2,#27 ; str w2,[x1] ; ldr x3,[x1] ; brk #0
    // reproduces the GIC-setup pattern that lost its value.
    let (mut cpu, mut cache) = setup(&[0x52800022, 0x53051042, 0xb9000022, 0xf9400023, 0xd4200000]);
    cpu.state.x[1] = RAM + 0x800;
    cpu.state.pc = RAM;
    assert_eq!(cpu.run(&mut cache), Stop::Brk(0));
    assert_eq!(cpu.state.x[2], 0x0800_0000, "w2 = 1<<27");
    assert_eq!(cpu.state.x[3], 0x0800_0000, "stored and read back");
}

/// PC-relative values come from where the code runs, not where it was first
/// translated: the same physical code run at a second address (through the
/// MMU, with the same code cache) gives the second address's ADR, ADRP and
/// link values.
#[test]
fn pc_relative_values_follow_the_address_the_code_runs_at() {
    use crate_jit::state::MmuRegs;
    let ram = 0x4000_0000u64;
    // adr x1, .+8 ; adrp x2, .+0x3000 ; bl .+8 ; brk #1 ; mov x4, x30 ; brk #0
    let code = [
        0x10000041, 0xf0000002, 0x94000002, 0xd4200020, 0xaa1e03e4, 0xd4200000,
    ];
    let (mut cpu, mut cache) = setup(&code);
    // Identity-map RAM (4 KiB pages, EL1) and map the code page again at a
    // second address.
    let alias = 0x10_0000_0000u64 + 0x7000;
    let tables = ram + 0x8000;
    let l = |n: u64| tables + n * 0x1000;
    let mem = cpu.mem.clone();
    let map = |va: u64, pa: u64, next: &mut u64| {
        let mut table = l(0);
        for level in 0..3 {
            let slot = table + ((va >> (39 - 9 * level)) & 0x1ff) * 8;
            let d = mem.read_u64(slot).unwrap();
            table = if d & 1 == 0 {
                let t = l(*next);
                *next += 1;
                mem.write_u64(slot, t | 3).unwrap();
                t
            } else {
                d & 0xffff_ffff_f000
            };
        }
        mem.write_u64(table + ((va >> 12) & 0x1ff) * 8, pa | 3 | 1 << 10)
            .unwrap();
    };
    let mut next = 1;
    for off in (0..0x10000).step_by(0x1000) {
        map(ram + off, ram + off, &mut next);
    }
    map(alias, ram, &mut next);
    cpu.state.pstate = 0b0101; // EL1h
    cpu.state.mmu = MmuRegs {
        sctlr_el1: 1,
        ttbr0_el1: l(0),
        tcr_el1: 16,
        ..Default::default()
    };
    for base in [ram, alias] {
        cpu.state.pc = base;
        assert_eq!(cpu.run(&mut cache), Stop::Brk(0), "at {base:#x}");
        assert_eq!(cpu.state.x[1], base + 8, "adr at {base:#x}");
        assert_eq!(
            cpu.state.x[2],
            (base & !0xfff) + 0x3000,
            "adrp at {base:#x}"
        );
        assert_eq!(cpu.state.x[4], base + 12, "bl's link at {base:#x}");
    }
}
