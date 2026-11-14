// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

.text
.globl _start
_start:
    adr  x0, vectors
    msr  vbar_el1, x0
    mov  x0, #0xff
    lsl  x0, x0, #8
    msr  mair_el1, x0
    ldr  x0, =0x270590B510      // 16K TG0, T0SZ=16, IPS=40bit
    msr  tcr_el1, x0
    // L0 @0x40100000, L1 @0x40104000, L2 @0x40108000 (16K-aligned)
    ldr  x1, =0x40100000
    ldr  x2, =0x40104000
    ldr  x8, =0x40108000
    orr  x3, x2, #3
    str  x3, [x1]              // L0[0] -> L1
    orr  x3, x8, #3
    str  x3, [x2]              // L1[0] -> L2
    // L2[4] device block (0x08000000..0x0A000000, covers UART 0x09000000)
    ldr  x4, =0x08000000
    mov  x3, #(1<<10)
    orr  x3, x3, #(3<<8)
    orr  x3, x3, #1
    orr  x3, x3, x4
    str  x3, [x8, #0x20]
    // L2[32] normal block (0x40000000..0x42000000, covers code+tables+data)
    ldr  x4, =0x40000000
    mov  x3, #(1<<10)
    orr  x3, x3, #(3<<8)
    orr  x3, x3, #(1<<2)
    orr  x3, x3, #1
    orr  x3, x3, x4
    str  x3, [x8, #0x100]
    ldr  x0, =0x40100000
    msr  ttbr0_el1, x0
    dsb  ishst
    tlbi vmalle1
    dsb  ish
    isb
    mrs  x0, sctlr_el1
    orr  x0, x0, #1
    msr  sctlr_el1, x0
    isb
    ldr  x1, =0x09000000
    mov  w2, #'m'
    str  w2, [x1]
    ldr  x5, =0x40300000
    movz x6, #0x1234
    movk x6, #0xabcd, lsl #16
    str  x6, [x5]
    ldr  x7, [x5]
    cmp  x6, x7
    b.ne 2f
    mov  w2, #'y'
    str  w2, [x1]
2:
    svc  #7
    mov  w2, #'z'
    str  w2, [x1]
    mov  w2, #'\n'
    str  w2, [x1]
    ldr  x0, =0x84000008
    hvc  #0
1:  wfi
    b    1b
.balign 0x800
vectors:
    b .
.balign 0x80
    b .
.balign 0x80
    b .
.balign 0x80
    b .
.balign 0x200
sync_spx:
    ldr  x11, =0x09000000
    mov  w12, #'s'
    str  w12, [x11]
    eret
.balign 0x80
    b .
