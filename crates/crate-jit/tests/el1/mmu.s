.text
.globl _start
_start:
    adr  x0, vectors
    msr  vbar_el1, x0
    // MAIR: idx0 = device(0x00), idx1 = normal WB(0xff)
    mov  x0, #0xff
    lsl  x0, x0, #8
    msr  mair_el1, x0
    ldr  x0, =0x200803510
    msr  tcr_el1, x0
    // L0 at 0x40100000, L1 at 0x40101000
    ldr  x1, =0x40100000
    ldr  x2, =0x40101000
    orr  x3, x2, #3            // L0[0] = L1 | table
    str  x3, [x1]
    // L1[0]: VA 0..1G -> PA 0..1G, device, block
    mov  x3, #(1<<10)
    orr  x3, x3, #(3<<8)
    orr  x3, x3, #1
    str  x3, [x2]
    // L1[1]: VA 1..2G -> PA 0x40000000, normal(attr1), block
    ldr  x4, =0x40000000
    mov  x3, #(1<<10)
    orr  x3, x3, #(3<<8)
    orr  x3, x3, #(1<<2)
    orr  x3, x3, #1
    orr  x3, x3, x4
    str  x3, [x2, #8]
    ldr  x0, =0x40100000
    msr  ttbr0_el1, x0
    dsb  ishst
    tlbi vmalle1
    dsb  ish
    isb
    mrs  x0, sctlr_el1
    orr  x0, x0, #1           // M=1
    msr  sctlr_el1, x0
    isb
    // MMU on. 'M' to UART.
    ldr  x1, =0x09000000
    mov  w2, #'M'
    str  w2, [x1]
    // store/load through MMU to RAM
    ldr  x5, =0x40300000
    movz x6, #0xbeef
    movk x6, #0xdead, lsl #16
    str  x6, [x5]
    ldr  x7, [x5]
    cmp  x6, x7
    b.ne 2f
    mov  w2, #'Y'
    str  w2, [x1]
2:
    svc  #0x42
    mov  w2, #'Z'
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
    // offset 0x200: current EL with SPx, synchronous
.balign 0x200
sync_spx:
    mrs  x9, esr_el1
    lsr  x10, x9, #26
    ldr  x11, =0x09000000
    mov  w12, #'S'
    str  w12, [x11]
    add  w12, w10, #'0'       // not printable EC value, but deterministic
    // print EC as two hex nibbles
    and  w13, w10, #0xf
    cmp  w13, #10
    b.lt 3f
    add  w13, w13, #('a'-10)
    b    4f
3:  add  w13, w13, #'0'
4:  str  w13, [x11]
    eret
.balign 0x80
    b .
