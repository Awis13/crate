.text
.globl _start
_start:
    ldr  x1, =0x09000000     // PL011 base
    mov  w2, #'O'
    str  w2, [x1]
    mov  w2, #'K'
    str  w2, [x1]
    mov  w2, #'\n'
    str  w2, [x1]
    // read CurrentEL, print '0'+EL
    mrs  x3, CurrentEL
    lsr  x3, x3, #2
    and  x3, x3, #3
    add  w3, w3, #'0'
    str  w3, [x1]
    mov  w2, #'\n'
    str  w2, [x1]
    // PSCI SYSTEM_OFF
    ldr  x0, =0x84000008
    hvc  #0
1:  wfi
    b    1b
