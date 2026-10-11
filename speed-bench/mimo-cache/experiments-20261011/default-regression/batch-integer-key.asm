
ds4_metal.o:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000000019760 <_ds4_gpu_stream_expert_cache_take_reusable_batch>:
   19760: 6db82beb     	stp	d11, d10, [sp, #-0x80]!
   19764: 6d0123e9     	stp	d9, d8, [sp, #0x10]
   19768: a9026ffc     	stp	x28, x27, [sp, #0x20]
   1976c: a90367fa     	stp	x26, x25, [sp, #0x30]
   19770: a9045ff8     	stp	x24, x23, [sp, #0x40]
   19774: a90557f6     	stp	x22, x21, [sp, #0x50]
   19778: a9064ff4     	stp	x20, x19, [sp, #0x60]
   1977c: a9077bfd     	stp	x29, x30, [sp, #0x70]
   19780: 9101c3fd     	add	x29, sp, #0x70
   19784: d10683ff     	sub	sp, sp, #0x1a0
   19788: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   1978c: f9400108     	ldr	x8, [x8]
   19790: f9400108     	ldr	x8, [x8]
   19794: f81803a8     	stur	x8, [x29, #-0x80]
   19798: 34003a60     	cbz	w0, 0x19ee4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x784>
   1979c: aa0603f4     	mov	x20, x6
   197a0: aa0403f5     	mov	x21, x4
   197a4: aa0303f3     	mov	x19, x3
   197a8: aa0103f7     	mov	x23, x1
   197ac: aa0003f9     	mov	x25, x0
   197b0: a90997e2     	stp	x2, x5, [sp, #0x98]
   197b4: 52800108     	mov	w8, #0x8                ; =8
   197b8: 7100201f     	cmp	w0, #0x8
   197bc: 1a883008     	csel	w8, w0, w8, lo
   197c0: f9005be8     	str	x8, [sp, #0xb0]
   197c4: f94000c0     	ldr	x0, [x6]
   197c8: f90000df     	str	xzr, [x6]
   197cc: 94000000     	bl	0x197cc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6c>
   197d0: f9400680     	ldr	x0, [x20, #0x8]
   197d4: f900069f     	str	xzr, [x20, #0x8]
   197d8: 94000000     	bl	0x197d8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x78>
   197dc: f9400a80     	ldr	x0, [x20, #0x10]
   197e0: f9000a9f     	str	xzr, [x20, #0x10]
   197e4: 94000000     	bl	0x197e4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x84>
   197e8: a901fe9f     	stp	xzr, xzr, [x20, #0x18]
   197ec: f900169f     	str	xzr, [x20, #0x28]
   197f0: 7100073f     	cmp	w25, #0x1
   197f4: 54000b40     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   197f8: f9401a80     	ldr	x0, [x20, #0x30]
   197fc: f9001a9f     	str	xzr, [x20, #0x30]
   19800: 94000000     	bl	0x19800 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0xa0>
   19804: f9401e80     	ldr	x0, [x20, #0x38]
   19808: f9001e9f     	str	xzr, [x20, #0x38]
   1980c: 94000000     	bl	0x1980c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0xac>
   19810: f9402280     	ldr	x0, [x20, #0x40]
   19814: f900229f     	str	xzr, [x20, #0x40]
   19818: 94000000     	bl	0x19818 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0xb8>
   1981c: a904fe9f     	stp	xzr, xzr, [x20, #0x48]
   19820: f9002e9f     	str	xzr, [x20, #0x58]
   19824: 71000b3f     	cmp	w25, #0x2
   19828: 540009a0     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   1982c: f9403280     	ldr	x0, [x20, #0x60]
   19830: f900329f     	str	xzr, [x20, #0x60]
   19834: 94000000     	bl	0x19834 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0xd4>
   19838: f9403680     	ldr	x0, [x20, #0x68]
   1983c: f900369f     	str	xzr, [x20, #0x68]
   19840: 94000000     	bl	0x19840 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0xe0>
   19844: f9403a80     	ldr	x0, [x20, #0x70]
   19848: f9003a9f     	str	xzr, [x20, #0x70]
   1984c: 94000000     	bl	0x1984c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0xec>
   19850: a907fe9f     	stp	xzr, xzr, [x20, #0x78]
   19854: f900469f     	str	xzr, [x20, #0x88]
   19858: 71000f3f     	cmp	w25, #0x3
   1985c: 54000800     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   19860: f9404a80     	ldr	x0, [x20, #0x90]
   19864: f9004a9f     	str	xzr, [x20, #0x90]
   19868: 94000000     	bl	0x19868 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x108>
   1986c: f9404e80     	ldr	x0, [x20, #0x98]
   19870: f9004e9f     	str	xzr, [x20, #0x98]
   19874: 94000000     	bl	0x19874 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x114>
   19878: f9405280     	ldr	x0, [x20, #0xa0]
   1987c: f900529f     	str	xzr, [x20, #0xa0]
   19880: 94000000     	bl	0x19880 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x120>
   19884: a90afe9f     	stp	xzr, xzr, [x20, #0xa8]
   19888: f9005e9f     	str	xzr, [x20, #0xb8]
   1988c: 7100133f     	cmp	w25, #0x4
   19890: 54000660     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   19894: f9406280     	ldr	x0, [x20, #0xc0]
   19898: f900629f     	str	xzr, [x20, #0xc0]
   1989c: 94000000     	bl	0x1989c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x13c>
   198a0: f9406680     	ldr	x0, [x20, #0xc8]
   198a4: f900669f     	str	xzr, [x20, #0xc8]
   198a8: 94000000     	bl	0x198a8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x148>
   198ac: f9406a80     	ldr	x0, [x20, #0xd0]
   198b0: f9006a9f     	str	xzr, [x20, #0xd0]
   198b4: 94000000     	bl	0x198b4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x154>
   198b8: a90dfe9f     	stp	xzr, xzr, [x20, #0xd8]
   198bc: f900769f     	str	xzr, [x20, #0xe8]
   198c0: 7100173f     	cmp	w25, #0x5
   198c4: 540004c0     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   198c8: f9407a80     	ldr	x0, [x20, #0xf0]
   198cc: f9007a9f     	str	xzr, [x20, #0xf0]
   198d0: 94000000     	bl	0x198d0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x170>
   198d4: f9407e80     	ldr	x0, [x20, #0xf8]
   198d8: f9007e9f     	str	xzr, [x20, #0xf8]
   198dc: 94000000     	bl	0x198dc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x17c>
   198e0: f9408280     	ldr	x0, [x20, #0x100]
   198e4: f900829f     	str	xzr, [x20, #0x100]
   198e8: 94000000     	bl	0x198e8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x188>
   198ec: a910fe9f     	stp	xzr, xzr, [x20, #0x108]
   198f0: f9008e9f     	str	xzr, [x20, #0x118]
   198f4: 71001b3f     	cmp	w25, #0x6
   198f8: 54000320     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   198fc: f9409280     	ldr	x0, [x20, #0x120]
   19900: f900929f     	str	xzr, [x20, #0x120]
   19904: 94000000     	bl	0x19904 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1a4>
   19908: f9409680     	ldr	x0, [x20, #0x128]
   1990c: f900969f     	str	xzr, [x20, #0x128]
   19910: 94000000     	bl	0x19910 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1b0>
   19914: f9409a80     	ldr	x0, [x20, #0x130]
   19918: f9009a9f     	str	xzr, [x20, #0x130]
   1991c: 94000000     	bl	0x1991c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1bc>
   19920: a913fe9f     	stp	xzr, xzr, [x20, #0x138]
   19924: f900a69f     	str	xzr, [x20, #0x148]
   19928: 71001f3f     	cmp	w25, #0x7
   1992c: 54000180     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   19930: f940aa80     	ldr	x0, [x20, #0x150]
   19934: f900aa9f     	str	xzr, [x20, #0x150]
   19938: 94000000     	bl	0x19938 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1d8>
   1993c: f940ae80     	ldr	x0, [x20, #0x158]
   19940: f900ae9f     	str	xzr, [x20, #0x158]
   19944: 94000000     	bl	0x19944 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1e4>
   19948: f940b280     	ldr	x0, [x20, #0x160]
   1994c: f900b29f     	str	xzr, [x20, #0x160]
   19950: 94000000     	bl	0x19950 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1f0>
   19954: a916fe9f     	stp	xzr, xzr, [x20, #0x168]
   19958: f900be9f     	str	xzr, [x20, #0x178]
   1995c: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19960: 91000108     	add	x8, x8, #0x0
   19964: b9402509     	ldr	w9, [x8, #0x24]
   19968: 2940290b     	ldp	w11, w10, [x8]
   1996c: 7100015f     	cmp	w10, #0x0
   19970: 1a8903e9     	csel	w9, wzr, w9, eq
   19974: b9408908     	ldr	w8, [x8, #0x88]
   19978: 6b08013f     	cmp	w9, w8
   1997c: 1a88312a     	csel	w10, w9, w8, lo
   19980: 7100011f     	cmp	w8, #0x0
   19984: 7a401924     	ccmp	w9, #0x0, #0x4, ne
   19988: 1a891148     	csel	w8, w10, w9, ne
   1998c: 51000508     	sub	w8, w8, #0x1
   19990: 6b0b011f     	cmp	w8, w11
   19994: 54002a82     	b.hs	0x19ee4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x784>
   19998: f90007f4     	str	x20, [sp, #0x8]
   1999c: 52800018     	mov	w24, #0x0               ; =0
   199a0: 2a1303e9     	mov	w9, w19
   199a4: 2a1703e8     	mov	w8, w23
   199a8: a908a7e8     	stp	x8, x9, [sp, #0x88]
   199ac: f9405bea     	ldr	x10, [sp, #0xb0]
   199b0: 51000548     	sub	w8, w10, #0x1
   199b4: 11000549     	add	w9, w10, #0x1
   199b8: 927f0d2b     	and	x11, x9, #0x1e
   199bc: d37e7d09     	ubfiz	x9, x8, #2, #32
   199c0: a9072fe9     	stp	x9, x11, [sp, #0x70]
   199c4: d1000549     	sub	x9, x10, #0x1
   199c8: d37d7d08     	ubfiz	x8, x8, #3, #32
   199cc: f90037e8     	str	x8, [sp, #0x68]
   199d0: 4e080d20     	dup.2d	v0, x9
   199d4: 3d8017e0     	str	q0, [sp, #0x50]
   199d8: d10403b3     	sub	x19, x29, #0x100
   199dc: 91002268     	add	x8, x19, #0x8
   199e0: f90027e8     	str	x8, [sp, #0x48]
   199e4: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   199e8: 3dc00101     	ldr	q1, [x8]
   199ec: 52800048     	mov	w8, #0x2                ; =2
   199f0: 4e080d00     	dup.2d	v0, x8
   199f4: ad0107e0     	stp	q0, q1, [sp, #0x20]
   199f8: 5280121a     	mov	w26, #0x90              ; =144
   199fc: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a00: 3dc00100     	ldr	q0, [x8]
   19a04: 3d8007e0     	str	q0, [sp, #0x10]
   19a08: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a0c: 39400108     	ldrb	w8, [x8]
   19a10: 7100051f     	cmp	w8, #0x1
   19a14: 540001a1     	b.ne	0x19a48 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2e8>
   19a18: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a1c: b9400114     	ldr	w20, [x8]
   19a20: 340003b4     	cbz	w20, 0x19a94 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x334>
   19a24: 910303e1     	add	x1, sp, #0xc0
   19a28: 528000c0     	mov	w0, #0x6                ; =6
   19a2c: 94000000     	bl	0x19a2c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2cc>
   19a30: 3dc033e0     	ldr	q0, [sp, #0xc0]
   19a34: 4e61d800     	scvtf.2d	v0, v0
   19a38: 3dc007e1     	ldr	q1, [sp, #0x10]
   19a3c: 6e61dc00     	fmul.2d	v0, v0, v1
   19a40: 7e70d808     	faddp.2d	d8, v0
   19a44: 14000015     	b	0x19a98 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x338>
   19a48: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a4c: 91000000     	add	x0, x0, #0x0
   19a50: 94000000     	bl	0x19a50 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2f0>
   19a54: b50000a0     	cbnz	x0, 0x19a68 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x308>
   19a58: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a5c: 91000000     	add	x0, x0, #0x0
   19a60: 94000000     	bl	0x19a60 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x300>
   19a64: b40022c0     	cbz	x0, 0x19ebc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x75c>
   19a68: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a6c: 91000000     	add	x0, x0, #0x0
   19a70: 94000000     	bl	0x19a70 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x310>
   19a74: f100001f     	cmp	x0, #0x0
   19a78: 1a9f17f4     	cset	w20, eq
   19a7c: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a80: 91000108     	add	x8, x8, #0x0
   19a84: b900e514     	str	w20, [x8, #0xe4]
   19a88: 52800029     	mov	w9, #0x1                ; =1
   19a8c: 39000109     	strb	w9, [x8]
   19a90: 35fffcb4     	cbnz	w20, 0x19a24 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2c4>
   19a94: 6f00e408     	movi.2d	v8, #0000000000000000
   19a98: b90087f4     	str	w20, [sp, #0x84]
   19a9c: 90000009     	adrp	x9, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19aa0: 91000129     	add	x9, x9, #0x0
   19aa4: f944ed28     	ldr	x8, [x9, #0x9d8]
   19aa8: f100011f     	cmp	x8, #0x0
   19aac: 1a9f07e8     	cset	w8, ne
   19ab0: f9400129     	ldr	x9, [x9]
   19ab4: f100013f     	cmp	x9, #0x0
   19ab8: 1a9f07e9     	cset	w9, ne
   19abc: 6a090108     	ands	w8, w8, w9
   19ac0: b900bfe8     	str	w8, [sp, #0xbc]
   19ac4: 12800008     	mov	w8, #-0x1               ; =-1
   19ac8: da9f0114     	csinv	x20, x8, xzr, eq
   19acc: d10283a0     	sub	x0, x29, #0xa0
   19ad0: f9403bf6     	ldr	x22, [sp, #0x70]
   19ad4: 910012c2     	add	x2, x22, #0x4
   19ad8: 52801fe1     	mov	w1, #0xff               ; =255
   19adc: 94000000     	bl	0x19adc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x37c>
   19ae0: d10303a0     	sub	x0, x29, #0xc0
   19ae4: 910012c2     	add	x2, x22, #0x4
   19ae8: 52801fe1     	mov	w1, #0xff               ; =255
   19aec: 94000000     	bl	0x19aec <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x38c>
   19af0: 910343e0     	add	x0, sp, #0xd0
   19af4: f94037e8     	ldr	x8, [sp, #0x68]
   19af8: 91002102     	add	x2, x8, #0x8
   19afc: 52801fe1     	mov	w1, #0xff               ; =255
   19b00: 94000000     	bl	0x19b00 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3a0>
   19b04: f9403fe8     	ldr	x8, [sp, #0x78]
   19b08: f94027e9     	ldr	x9, [sp, #0x48]
   19b0c: ad4103e3     	ldp	q3, q0, [sp, #0x20]
   19b10: a949bbef     	ldp	x15, x14, [sp, #0x98]
   19b14: a948c3f1     	ldp	x17, x16, [sp, #0x88]
   19b18: 3dc017e2     	ldr	q2, [sp, #0x50]
   19b1c: 910343e0     	add	x0, sp, #0xd0
   19b20: 90000001     	adrp	x1, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19b24: 91000021     	add	x1, x1, #0x0
   19b28: 529b0002     	mov	w2, #0xd800             ; =55296
   19b2c: 5280c004     	mov	w4, #0x600              ; =1536
   19b30: 90000005     	adrp	x5, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19b34: 910000a5     	add	x5, x5, #0x0
   19b38: 14000005     	b	0x19b4c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3ec>
   19b3c: 4ee38400     	add.2d	v0, v0, v3
   19b40: 91004129     	add	x9, x9, #0x10
   19b44: f1000908     	subs	x8, x8, #0x2
   19b48: 540001a0     	b.eq	0x19b7c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x41c>
   19b4c: 6ee03c41     	cmhs.2d	v1, v2, v0
   19b50: 0ea12821     	xtn.2s	v1, v1
   19b54: 1e26002a     	fmov	w10, s1
   19b58: 360000aa     	tbz	w10, #0x0, 0x19b6c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x40c>
   19b5c: f81f8134     	stur	x20, [x9, #-0x8]
   19b60: 0e0c3c2a     	mov.s	w10, v1[1]
   19b64: 3607feca     	tbz	w10, #0x0, 0x19b3c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3dc>
   19b68: 14000003     	b	0x19b74 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x414>
   19b6c: 0e0c3c2a     	mov.s	w10, v1[1]
   19b70: 3607fe6a     	tbz	w10, #0x0, 0x19b3c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3dc>
   19b74: f9000134     	str	x20, [x9]
   19b78: 17fffff1     	b	0x19b3c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3dc>
   19b7c: b90083f8     	str	w24, [sp, #0x80]
   19b80: d280001b     	mov	x27, #0x0               ; =0
   19b84: 52800019     	mov	w25, #0x0               ; =0
   19b88: 52800006     	mov	w6, #0x0                ; =0
   19b8c: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19b90: f9400107     	ldr	x7, [x8]
   19b94: 14000004     	b	0x19ba4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x444>
   19b98: 9100077b     	add	x27, x27, #0x1
   19b9c: f101437f     	cmp	x27, #0x50
   19ba0: 54000e80     	b.eq	0x19d70 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x610>
   19ba4: d280001c     	mov	x28, #0x0               ; =0
   19ba8: f10001ff     	cmp	x15, #0x0
   19bac: fa511360     	ccmp	x27, x17, #0x0, ne
   19bb0: 1a9f07f4     	cset	w20, ne
   19bb4: 9b020778     	madd	x24, x27, x2, x1
   19bb8: 9b041776     	madd	x22, x27, x4, x5
   19bbc: 14000005     	b	0x19bd0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x470>
   19bc0: 52800026     	mov	w6, #0x1                ; =1
   19bc4: 9100079c     	add	x28, x28, #0x1
   19bc8: f106039f     	cmp	x28, #0x180
   19bcc: 54fffe60     	b.eq	0x19b98 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x438>
   19bd0: 9b1a6397     	madd	x23, x28, x26, x24
   19bd4: 394232e8     	ldrb	w8, [x23, #0x8c]
   19bd8: 34ffff68     	cbz	w8, 0x19bc4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x464>
   19bdc: f94002e8     	ldr	x8, [x23]
   19be0: b4ffff28     	cbz	x8, 0x19bc4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x464>
   19be4: f94006e8     	ldr	x8, [x23, #0x8]
   19be8: b4fffee8     	cbz	x8, 0x19bc4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x464>
   19bec: f9400ae8     	ldr	x8, [x23, #0x10]
   19bf0: b4fffea8     	cbz	x8, 0x19bc4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x464>
   19bf4: f94022e8     	ldr	x8, [x23, #0x40]
   19bf8: eb15011f     	cmp	x8, x21
   19bfc: 54fffe41     	b.ne	0x19bc4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x464>
   19c00: f94026e8     	ldr	x8, [x23, #0x48]
   19c04: eb0e011f     	cmp	x8, x14
   19c08: 54fffde1     	b.ne	0x19bc4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x464>
   19c0c: f94042e8     	ldr	x8, [x23, #0x80]
   19c10: eb07011f     	cmp	x8, x7
   19c14: 54fffd68     	b.hi	0x19bc0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x460>
   19c18: aa0f03e8     	mov	x8, x15
   19c1c: aa1003e9     	mov	x9, x16
   19c20: 370000d4     	tbnz	w20, #0x0, 0x19c38 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x4d8>
   19c24: b840450a     	ldr	w10, [x8], #0x4
   19c28: eb0a039f     	cmp	x28, x10
   19c2c: 54fffcc0     	b.eq	0x19bc4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x464>
   19c30: f1000529     	subs	x9, x9, #0x1
   19c34: 54ffff81     	b.ne	0x19c24 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x4c4>
   19c38: b87c7ac3     	ldr	w3, [x22, x28, lsl #2]
   19c3c: b940bfe8     	ldr	w8, [sp, #0xbc]
   19c40: 36000388     	tbz	w8, #0x0, 0x19cb0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x550>
   19c44: f9402ee2     	ldr	x2, [x23, #0x58]
   19c48: aa1b03e0     	mov	x0, x27
   19c4c: aa1c03e1     	mov	x1, x28
   19c50: b900aff4     	str	w20, [sp, #0xac]
   19c54: aa1503f4     	mov	x20, x21
   19c58: aa1903f5     	mov	x21, x25
   19c5c: aa0603f9     	mov	x25, x6
   19c60: aa1603f3     	mov	x19, x22
   19c64: aa0703f6     	mov	x22, x7
   19c68: 94000000     	bl	0x19c68 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x508>
   19c6c: aa1603e7     	mov	x7, x22
   19c70: aa1303f6     	mov	x22, x19
   19c74: aa1903e6     	mov	x6, x25
   19c78: aa1503f9     	mov	x25, x21
   19c7c: aa1403f5     	mov	x21, x20
   19c80: b940aff4     	ldr	w20, [sp, #0xac]
   19c84: d10403b3     	sub	x19, x29, #0x100
   19c88: 90000005     	adrp	x5, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19c8c: 910000a5     	add	x5, x5, #0x0
   19c90: 5280c004     	mov	w4, #0x600              ; =1536
   19c94: 529b0002     	mov	w2, #0xd800             ; =55296
   19c98: 90000001     	adrp	x1, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19c9c: 91000021     	add	x1, x1, #0x0
   19ca0: a948c3f1     	ldp	x17, x16, [sp, #0x88]
   19ca4: a949bbef     	ldp	x15, x14, [sp, #0x98]
   19ca8: aa0003e3     	mov	x3, x0
   19cac: 910343e0     	add	x0, sp, #0xd0
   19cb0: f9402ee8     	ldr	x8, [x23, #0x58]
   19cb4: f9405be9     	ldr	x9, [sp, #0xb0]
   19cb8: 6b09033f     	cmp	w25, w9
   19cbc: 54000122     	b.hs	0x19ce0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x580>
   19cc0: d10283a9     	sub	x9, x29, #0xa0
   19cc4: b839593b     	str	w27, [x9, w25, uxtw #2]
   19cc8: d10303a9     	sub	x9, x29, #0xc0
   19ccc: b839593c     	str	w28, [x9, w25, uxtw #2]
   19cd0: f8395a63     	str	x3, [x19, w25, uxtw #3]
   19cd4: f8395808     	str	x8, [x0, w25, uxtw #3]
   19cd8: 11000739     	add	w25, w25, #0x1
   19cdc: 17ffffba     	b	0x19bc4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x464>
   19ce0: 71000b3f     	cmp	w25, #0x2
   19ce4: 54000263     	b.lo	0x19d30 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x5d0>
   19ce8: d2800009     	mov	x9, #0x0                ; =0
   19cec: 2a1903ea     	mov	w10, w25
   19cf0: 5280002b     	mov	w11, #0x1               ; =1
   19cf4: 14000005     	b	0x19d08 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x5a8>
   19cf8: 2a0b03e9     	mov	w9, w11
   19cfc: 9100056b     	add	x11, x11, #0x1
   19d00: eb0b015f     	cmp	x10, x11
   19d04: 54000180     	b.eq	0x19d34 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x5d4>
   19d08: f86b7a6c     	ldr	x12, [x19, x11, lsl #3]
   19d0c: f8697a6d     	ldr	x13, [x19, x9, lsl #3]
   19d10: eb0d019f     	cmp	x12, x13
   19d14: 54ffff28     	b.hi	0x19cf8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x598>
   19d18: 54ffff21     	b.ne	0x19cfc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x59c>
   19d1c: f86b780c     	ldr	x12, [x0, x11, lsl #3]
   19d20: f869780d     	ldr	x13, [x0, x9, lsl #3]
   19d24: eb0d019f     	cmp	x12, x13
   19d28: 54fffe88     	b.hi	0x19cf8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x598>
   19d2c: 17fffff4     	b	0x19cfc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x59c>
   19d30: d2800009     	mov	x9, #0x0                ; =0
   19d34: f8697a6a     	ldr	x10, [x19, x9, lsl #3]
   19d38: eb0a007f     	cmp	x3, x10
   19d3c: 54000102     	b.hs	0x19d5c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x5fc>
   19d40: d10283aa     	sub	x10, x29, #0xa0
   19d44: b829795b     	str	w27, [x10, x9, lsl #2]
   19d48: d10303aa     	sub	x10, x29, #0xc0
   19d4c: b829795c     	str	w28, [x10, x9, lsl #2]
   19d50: f8297a63     	str	x3, [x19, x9, lsl #3]
   19d54: f8297808     	str	x8, [x0, x9, lsl #3]
   19d58: 17ffff9b     	b	0x19bc4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x464>
   19d5c: 54fff341     	b.ne	0x19bc4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x464>
   19d60: f869780a     	ldr	x10, [x0, x9, lsl #3]
   19d64: eb0a011f     	cmp	x8, x10
   19d68: 54fff2e2     	b.hs	0x19bc4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x464>
   19d6c: 17fffff5     	b	0x19d40 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x5e0>
   19d70: b94087fb     	ldr	w27, [sp, #0x84]
   19d74: 340001fb     	cbz	w27, 0x19db0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x650>
   19d78: aa0603f4     	mov	x20, x6
   19d7c: 910303e1     	add	x1, sp, #0xc0
   19d80: 528000c0     	mov	w0, #0x6                ; =6
   19d84: 94000000     	bl	0x19d84 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x624>
   19d88: 6d4c27ea     	ldp	d10, d9, [sp, #0xc0]
   19d8c: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19d90: 39400108     	ldrb	w8, [x8]
   19d94: 7100051f     	cmp	w8, #0x1
   19d98: 54000101     	b.ne	0x19db8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x658>
   19d9c: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19da0: b9400108     	ldr	w8, [x8]
   19da4: aa1403e6     	mov	x6, x20
   19da8: 35000308     	cbnz	w8, 0x19e08 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6a8>
   19dac: 1400002f     	b	0x19e68 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x708>
   19db0: 340005f9     	cbz	w25, 0x19e6c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x70c>
   19db4: 14000067     	b	0x19f50 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x7f0>
   19db8: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19dbc: 91000000     	add	x0, x0, #0x0
   19dc0: 94000000     	bl	0x19dc0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x660>
   19dc4: b50000a0     	cbnz	x0, 0x19dd8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x678>
   19dc8: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19dcc: 91000000     	add	x0, x0, #0x0
   19dd0: 94000000     	bl	0x19dd0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x670>
   19dd4: b4000840     	cbz	x0, 0x19edc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x77c>
   19dd8: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19ddc: 91000000     	add	x0, x0, #0x0
   19de0: 94000000     	bl	0x19de0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x680>
   19de4: f100001f     	cmp	x0, #0x0
   19de8: 1a9f17e8     	cset	w8, eq
   19dec: aa1403e6     	mov	x6, x20
   19df0: 90000009     	adrp	x9, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19df4: 91000129     	add	x9, x9, #0x0
   19df8: b900e528     	str	w8, [x9, #0xe4]
   19dfc: 5280002a     	mov	w10, #0x1               ; =1
   19e00: 3900012a     	strb	w10, [x9]
   19e04: 34000328     	cbz	w8, 0x19e68 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x708>
   19e08: 5e61d940     	scvtf	d0, d10
   19e0c: d2c80008     	mov	x8, #0x400000000000     ; =70368744177664
   19e10: f2e811e8     	movk	x8, #0x408f, lsl #48
   19e14: 9e670101     	fmov	d1, x8
   19e18: 1e610800     	fmul	d0, d0, d1
   19e1c: 5e61d921     	scvtf	d1, d9
   19e20: d29db1a8     	mov	x8, #0xed8d             ; =60813
   19e24: f2b416a8     	movk	x8, #0xa0b5, lsl #16
   19e28: f2d8dee8     	movk	x8, #0xc6f7, lsl #32
   19e2c: f2e7d608     	movk	x8, #0x3eb0, lsl #48
   19e30: 9e670102     	fmov	d2, x8
   19e34: 1e620821     	fmul	d1, d1, d2
   19e38: 9000000a     	adrp	x10, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19e3c: 9100014a     	add	x10, x10, #0x0
   19e40: a9402548     	ldp	x8, x9, [x10]
   19e44: 91000508     	add	x8, x8, #0x1
   19e48: 528f000b     	mov	w11, #0x7800            ; =30720
   19e4c: 8b0b0129     	add	x9, x9, x11
   19e50: a9002548     	stp	x8, x9, [x10]
   19e54: fd400942     	ldr	d2, [x10, #0x10]
   19e58: 1e683800     	fsub	d0, d0, d8
   19e5c: 1e622821     	fadd	d1, d1, d2
   19e60: 1e612800     	fadd	d0, d0, d1
   19e64: fd000940     	str	d0, [x10, #0x10]
   19e68: 35000639     	cbnz	w25, 0x19f2c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x7cc>
   19e6c: 710000df     	cmp	w6, #0x0
   19e70: 1a9f17e8     	cset	w8, eq
   19e74: b94083e9     	ldr	w9, [sp, #0x80]
   19e78: 2a080128     	orr	w8, w9, w8
   19e7c: 37000348     	tbnz	w8, #0x0, 0x19ee4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x784>
   19e80: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19e84: f9400108     	ldr	x8, [x8]
   19e88: b4000068     	cbz	x8, 0x19e94 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x734>
   19e8c: 94000000     	bl	0x19e8c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x72c>
   19e90: 340002a0     	cbz	w0, 0x19ee4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x784>
   19e94: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19e98: f9400100     	ldr	x0, [x8]
   19e9c: 94000000     	bl	0x19e9c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x73c>
   19ea0: 52800038     	mov	w24, #0x1               ; =1
   19ea4: b4ffdb20     	cbz	x0, 0x19a08 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2a8>
   19ea8: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19eac: 91000000     	add	x0, x0, #0x0
   19eb0: 94000000     	bl	0x19eb0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x750>
   19eb4: 35ffdaa0     	cbnz	w0, 0x19a08 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2a8>
   19eb8: 1400000b     	b	0x19ee4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x784>
   19ebc: 52800014     	mov	w20, #0x0               ; =0
   19ec0: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19ec4: 91000108     	add	x8, x8, #0x0
   19ec8: b900e514     	str	w20, [x8, #0xe4]
   19ecc: 52800029     	mov	w9, #0x1                ; =1
   19ed0: 39000109     	strb	w9, [x8]
   19ed4: 35ffda94     	cbnz	w20, 0x19a24 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2c4>
   19ed8: 17fffeef     	b	0x19a94 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x334>
   19edc: 52800008     	mov	w8, #0x0                ; =0
   19ee0: 17ffffc3     	b	0x19dec <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x68c>
   19ee4: 52800014     	mov	w20, #0x0               ; =0
   19ee8: f85803a8     	ldur	x8, [x29, #-0x80]
   19eec: 90000009     	adrp	x9, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19ef0: f9400129     	ldr	x9, [x9]
   19ef4: f9400129     	ldr	x9, [x9]
   19ef8: eb08013f     	cmp	x9, x8
   19efc: 54001021     	b.ne	0x1a100 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x9a0>
   19f00: aa1403e0     	mov	x0, x20
   19f04: 910683ff     	add	sp, sp, #0x1a0
   19f08: a9477bfd     	ldp	x29, x30, [sp, #0x70]
   19f0c: a9464ff4     	ldp	x20, x19, [sp, #0x60]
   19f10: a94557f6     	ldp	x22, x21, [sp, #0x50]
   19f14: a9445ff8     	ldp	x24, x23, [sp, #0x40]
   19f18: a94367fa     	ldp	x26, x25, [sp, #0x30]
   19f1c: a9426ffc     	ldp	x28, x27, [sp, #0x20]
   19f20: 6d4123e9     	ldp	d9, d8, [sp, #0x10]
   19f24: 6cc82beb     	ldp	d11, d10, [sp], #0x80
   19f28: d65f03c0     	ret
   19f2c: 910303e1     	add	x1, sp, #0xc0
   19f30: 528000c0     	mov	w0, #0x6                ; =6
   19f34: 94000000     	bl	0x19f34 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x7d4>
   19f38: 3dc033e0     	ldr	q0, [sp, #0xc0]
   19f3c: 4e61d800     	scvtf.2d	v0, v0
   19f40: 3dc007e1     	ldr	q1, [sp, #0x10]
   19f44: 6e61dc00     	fmul.2d	v0, v0, v1
   19f48: 7e70d808     	faddp.2d	d8, v0
   19f4c: 14000002     	b	0x19f54 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x7f4>
   19f50: 6f00e408     	movi.2d	v8, #0000000000000000
   19f54: 52800014     	mov	w20, #0x0               ; =0
   19f58: d10283b3     	sub	x19, x29, #0xa0
   19f5c: d10303b6     	sub	x22, x29, #0xc0
   19f60: 52800617     	mov	w23, #0x30              ; =48
   19f64: 52800058     	mov	w24, #0x2               ; =2
   19f68: 9000001a     	adrp	x26, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19f6c: 2a1903f9     	mov	w25, w25
   19f70: 1400000e     	b	0x19fa8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x848>
   19f74: f90002bf     	str	xzr, [x21]
   19f78: 94000000     	bl	0x19f78 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x818>
   19f7c: f94006a0     	ldr	x0, [x21, #0x8]
   19f80: f90006bf     	str	xzr, [x21, #0x8]
   19f84: 94000000     	bl	0x19f84 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x824>
   19f88: f9400aa0     	ldr	x0, [x21, #0x10]
   19f8c: f9000abf     	str	xzr, [x21, #0x10]
   19f90: 94000000     	bl	0x19f90 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x830>
   19f94: a901febf     	stp	xzr, xzr, [x21, #0x18]
   19f98: f90016bf     	str	xzr, [x21, #0x28]
   19f9c: 910012d6     	add	x22, x22, #0x4
   19fa0: f1000739     	subs	x25, x25, #0x1
   19fa4: 54000460     	b.eq	0x1a030 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8d0>
   19fa8: b8404660     	ldr	w0, [x19], #0x4
   19fac: 3100041f     	cmn	w0, #0x1
   19fb0: 54ffff60     	b.eq	0x19f9c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x83c>
   19fb4: b94002c1     	ldr	w1, [x22]
   19fb8: 3100043f     	cmn	w1, #0x1
   19fbc: 54ffff00     	b.eq	0x19f9c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x83c>
   19fc0: f94007e8     	ldr	x8, [sp, #0x8]
   19fc4: 9bb72295     	umaddl	x21, w20, w23, x8
   19fc8: 52800022     	mov	w2, #0x1                ; =1
   19fcc: 52800023     	mov	w3, #0x1                ; =1
   19fd0: aa1503e4     	mov	x4, x21
   19fd4: 94000000     	bl	0x19fd4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x874>
   19fd8: f94002a0     	ldr	x0, [x21]
   19fdc: b4fffcc0     	cbz	x0, 0x19f74 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x814>
   19fe0: f94006a8     	ldr	x8, [x21, #0x8]
   19fe4: b4fffc88     	cbz	x8, 0x19f74 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x814>
   19fe8: f9400aa9     	ldr	x9, [x21, #0x10]
   19fec: b4fffc49     	cbz	x9, 0x19f74 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x814>
   19ff0: eb09001f     	cmp	x0, x9
   19ff4: 1a9f17ea     	cset	w10, eq
   19ff8: eb09011f     	cmp	x8, x9
   19ffc: 1a9f1549     	csinc	w9, w10, wzr, ne
   1a000: eb08001f     	cmp	x0, x8
   1a004: 1a9f17e8     	cset	w8, eq
   1a008: 1a9f1529     	csinc	w9, w9, wzr, ne
   1a00c: 7100013f     	cmp	w9, #0x0
   1a010: 9a981709     	cinc	x9, x24, eq
   1a014: 6a0a011f     	tst	w8, w10
   1a018: 9a9f0528     	csinc	x8, x9, xzr, eq
   1a01c: f9400349     	ldr	x9, [x26]
   1a020: 8b080128     	add	x8, x9, x8
   1a024: f9000348     	str	x8, [x26]
   1a028: 11000694     	add	w20, w20, #0x1
   1a02c: 17ffffdc     	b	0x19f9c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x83c>
   1a030: 34fff5db     	cbz	w27, 0x19ee8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x788>
   1a034: 910303e1     	add	x1, sp, #0xc0
   1a038: 528000c0     	mov	w0, #0x6                ; =6
   1a03c: 94000000     	bl	0x1a03c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8dc>
   1a040: 6d4c27ea     	ldp	d10, d9, [sp, #0xc0]
   1a044: 90000008     	adrp	x8, 0x1a000 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8a0>
   1a048: 39400108     	ldrb	w8, [x8]
   1a04c: 7100051f     	cmp	w8, #0x1
   1a050: 54000081     	b.ne	0x1a060 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x900>
   1a054: 90000008     	adrp	x8, 0x1a000 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8a0>
   1a058: b9400108     	ldr	w8, [x8]
   1a05c: 14000013     	b	0x1a0a8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x948>
   1a060: 90000000     	adrp	x0, 0x1a000 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8a0>
   1a064: 91000000     	add	x0, x0, #0x0
   1a068: 94000000     	bl	0x1a068 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x908>
   1a06c: b50000a0     	cbnz	x0, 0x1a080 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x920>
   1a070: 90000000     	adrp	x0, 0x1a000 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8a0>
   1a074: 91000000     	add	x0, x0, #0x0
   1a078: 94000000     	bl	0x1a078 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x918>
   1a07c: b40003e0     	cbz	x0, 0x1a0f8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x998>
   1a080: 90000000     	adrp	x0, 0x1a000 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8a0>
   1a084: 91000000     	add	x0, x0, #0x0
   1a088: 94000000     	bl	0x1a088 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x928>
   1a08c: f100001f     	cmp	x0, #0x0
   1a090: 1a9f17e8     	cset	w8, eq
   1a094: 9000000a     	adrp	x10, 0x1a000 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8a0>
   1a098: 9100014a     	add	x10, x10, #0x0
   1a09c: b900e548     	str	w8, [x10, #0xe4]
   1a0a0: 52800029     	mov	w9, #0x1                ; =1
   1a0a4: 39000149     	strb	w9, [x10]
   1a0a8: 34fff208     	cbz	w8, 0x19ee8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x788>
   1a0ac: d2c80008     	mov	x8, #0x400000000000     ; =70368744177664
   1a0b0: f2e811e8     	movk	x8, #0x408f, lsl #48
   1a0b4: d29db1a9     	mov	x9, #0xed8d             ; =60813
   1a0b8: f2b416a9     	movk	x9, #0xa0b5, lsl #16
   1a0bc: f2d8dee9     	movk	x9, #0xc6f7, lsl #32
   1a0c0: f2e7d609     	movk	x9, #0x3eb0, lsl #48
   1a0c4: 5e61d940     	scvtf	d0, d10
   1a0c8: 9e670101     	fmov	d1, x8
   1a0cc: 5e61d922     	scvtf	d2, d9
   1a0d0: 9e670123     	fmov	d3, x9
   1a0d4: 1e610800     	fmul	d0, d0, d1
   1a0d8: 1e630841     	fmul	d1, d2, d3
   1a0dc: 90000008     	adrp	x8, 0x1a000 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8a0>
   1a0e0: fd400102     	ldr	d2, [x8]
   1a0e4: 1e683800     	fsub	d0, d0, d8
   1a0e8: 1e622821     	fadd	d1, d1, d2
   1a0ec: 1e612800     	fadd	d0, d0, d1
   1a0f0: fd000100     	str	d0, [x8]
   1a0f4: 17ffff7d     	b	0x19ee8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x788>
   1a0f8: 52800008     	mov	w8, #0x0                ; =0
   1a0fc: 17ffffe6     	b	0x1a094 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x934>
   1a100: 94000000     	bl	0x1a100 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x9a0>
