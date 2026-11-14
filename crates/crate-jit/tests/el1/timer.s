// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

.text
.globl _start
_start:
    adr  x0, vectors
    msr  vbar_el1, x0
    ldr  x20, =0x09000000        // UART
    // wake redistributor
    ldr  x1, =0x08100014         // GICR_WAKER
    ldr  w2, [x1]
    bic  w2, w2, #2              // clear ProcessorSleep
    str  w2, [x1]
1:  ldr  w2, [x1]
    tbnz w2, #2, 1b              // wait ChildrenAsleep=0
    // GICD_CTLR = ARE | EnableGrp1
    ldr  x1, =0x08000000
    mov  w2, #0x12
    str  w2, [x1]
    // PPI 27 to group 1 (IGROUPR0 bit27), so it signals IRQ
    ldr  x1, =0x08110080
    mov  w2, #1
    lsl  w2, w2, #27
    str  w2, [x1]
    // enable PPI 27 (ISENABLER0 bit27)
    ldr  x1, =0x08110100
    mov  w2, #1
    lsl  w2, w2, #27
    str  w2, [x1]
    // priority 0 for INTID 27
    ldr  x1, =0x0811041b
    mov  w2, #0
    strb w2, [x1]
    // CPU interface: PMR, IGRPEN1
    mov  x0, #0xff
    msr  ICC_PMR_EL1, x0
    mov  x0, #1
    msr  ICC_IGRPEN1_EL1, x0
    // arm virtual timer
    movz x0, #0x40, lsl #16
    msr  CNTV_TVAL_EL0, x0
    mov  x0, #1
    msr  CNTV_CTL_EL0, x0        // enable, unmasked
    msr  daifclr, #2            // unmask IRQ
    mov  w2, #'W'
    str  w2, [x20]
    wfi
    mov  w2, #'D'
    str  w2, [x20]
    mov  w2, #'\n'
    str  w2, [x20]
    ldr  x0, =0x84000008
    hvc  #0
2:  wfi
    b    2b

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
    b .                          // 0x200 sync SPx (unused)
.balign 0x80
irq_spx:                         // 0x280 IRQ SPx
    mrs  x0, ICC_IAR1_EL1
    mov  x3, #0
    msr  CNTV_CTL_EL0, x3        // disable timer (drops the line)
    msr  ICC_EOIR1_EL1, x0
    mov  w2, #'T'
    str  w2, [x20]
    eret
.balign 0x80
    b .
