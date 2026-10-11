
ds4_metal.o:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000000019760 <_ds4_gpu_stream_expert_cache_take_reusable_batch>:
   19760: d10743ff     	sub	sp, sp, #0x1d0
   19764: 6d152beb     	stp	d11, d10, [sp, #0x150]
   19768: 6d1623e9     	stp	d9, d8, [sp, #0x160]
   1976c: a9176ffc     	stp	x28, x27, [sp, #0x170]
   19770: a91867fa     	stp	x26, x25, [sp, #0x180]
   19774: a9195ff8     	stp	x24, x23, [sp, #0x190]
   19778: a91a57f6     	stp	x22, x21, [sp, #0x1a0]
   1977c: a91b4ff4     	stp	x20, x19, [sp, #0x1b0]
   19780: a91c7bfd     	stp	x29, x30, [sp, #0x1c0]
   19784: 910703fd     	add	x29, sp, #0x1c0
   19788: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   1978c: f9400108     	ldr	x8, [x8]
   19790: f9400108     	ldr	x8, [x8]
   19794: f81803a8     	stur	x8, [x29, #-0x80]
   19798: 34003220     	cbz	w0, 0x19ddc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x67c>
   1979c: aa0603f3     	mov	x19, x6
   197a0: aa0403f5     	mov	x21, x4
   197a4: aa0303f8     	mov	x24, x3
   197a8: aa0103f7     	mov	x23, x1
   197ac: aa0003f9     	mov	x25, x0
   197b0: a90497e2     	stp	x2, x5, [sp, #0x48]
   197b4: 52800108     	mov	w8, #0x8                ; =8
   197b8: 7100201f     	cmp	w0, #0x8
   197bc: 1a883008     	csel	w8, w0, w8, lo
   197c0: f90023e8     	str	x8, [sp, #0x40]
   197c4: f94000c0     	ldr	x0, [x6]
   197c8: f90000df     	str	xzr, [x6]
   197cc: 94000000     	bl	0x197cc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6c>
   197d0: f9400660     	ldr	x0, [x19, #0x8]
   197d4: f900067f     	str	xzr, [x19, #0x8]
   197d8: 94000000     	bl	0x197d8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x78>
   197dc: f9400a60     	ldr	x0, [x19, #0x10]
   197e0: f9000a7f     	str	xzr, [x19, #0x10]
   197e4: 94000000     	bl	0x197e4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x84>
   197e8: a901fe7f     	stp	xzr, xzr, [x19, #0x18]
   197ec: f900167f     	str	xzr, [x19, #0x28]
   197f0: 7100073f     	cmp	w25, #0x1
   197f4: 54000b40     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   197f8: f9401a60     	ldr	x0, [x19, #0x30]
   197fc: f9001a7f     	str	xzr, [x19, #0x30]
   19800: 94000000     	bl	0x19800 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0xa0>
   19804: f9401e60     	ldr	x0, [x19, #0x38]
   19808: f9001e7f     	str	xzr, [x19, #0x38]
   1980c: 94000000     	bl	0x1980c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0xac>
   19810: f9402260     	ldr	x0, [x19, #0x40]
   19814: f900227f     	str	xzr, [x19, #0x40]
   19818: 94000000     	bl	0x19818 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0xb8>
   1981c: a904fe7f     	stp	xzr, xzr, [x19, #0x48]
   19820: f9002e7f     	str	xzr, [x19, #0x58]
   19824: 71000b3f     	cmp	w25, #0x2
   19828: 540009a0     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   1982c: f9403260     	ldr	x0, [x19, #0x60]
   19830: f900327f     	str	xzr, [x19, #0x60]
   19834: 94000000     	bl	0x19834 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0xd4>
   19838: f9403660     	ldr	x0, [x19, #0x68]
   1983c: f900367f     	str	xzr, [x19, #0x68]
   19840: 94000000     	bl	0x19840 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0xe0>
   19844: f9403a60     	ldr	x0, [x19, #0x70]
   19848: f9003a7f     	str	xzr, [x19, #0x70]
   1984c: 94000000     	bl	0x1984c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0xec>
   19850: a907fe7f     	stp	xzr, xzr, [x19, #0x78]
   19854: f900467f     	str	xzr, [x19, #0x88]
   19858: 71000f3f     	cmp	w25, #0x3
   1985c: 54000800     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   19860: f9404a60     	ldr	x0, [x19, #0x90]
   19864: f9004a7f     	str	xzr, [x19, #0x90]
   19868: 94000000     	bl	0x19868 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x108>
   1986c: f9404e60     	ldr	x0, [x19, #0x98]
   19870: f9004e7f     	str	xzr, [x19, #0x98]
   19874: 94000000     	bl	0x19874 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x114>
   19878: f9405260     	ldr	x0, [x19, #0xa0]
   1987c: f900527f     	str	xzr, [x19, #0xa0]
   19880: 94000000     	bl	0x19880 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x120>
   19884: a90afe7f     	stp	xzr, xzr, [x19, #0xa8]
   19888: f9005e7f     	str	xzr, [x19, #0xb8]
   1988c: 7100133f     	cmp	w25, #0x4
   19890: 54000660     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   19894: f9406260     	ldr	x0, [x19, #0xc0]
   19898: f900627f     	str	xzr, [x19, #0xc0]
   1989c: 94000000     	bl	0x1989c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x13c>
   198a0: f9406660     	ldr	x0, [x19, #0xc8]
   198a4: f900667f     	str	xzr, [x19, #0xc8]
   198a8: 94000000     	bl	0x198a8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x148>
   198ac: f9406a60     	ldr	x0, [x19, #0xd0]
   198b0: f9006a7f     	str	xzr, [x19, #0xd0]
   198b4: 94000000     	bl	0x198b4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x154>
   198b8: a90dfe7f     	stp	xzr, xzr, [x19, #0xd8]
   198bc: f900767f     	str	xzr, [x19, #0xe8]
   198c0: 7100173f     	cmp	w25, #0x5
   198c4: 540004c0     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   198c8: f9407a60     	ldr	x0, [x19, #0xf0]
   198cc: f9007a7f     	str	xzr, [x19, #0xf0]
   198d0: 94000000     	bl	0x198d0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x170>
   198d4: f9407e60     	ldr	x0, [x19, #0xf8]
   198d8: f9007e7f     	str	xzr, [x19, #0xf8]
   198dc: 94000000     	bl	0x198dc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x17c>
   198e0: f9408260     	ldr	x0, [x19, #0x100]
   198e4: f900827f     	str	xzr, [x19, #0x100]
   198e8: 94000000     	bl	0x198e8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x188>
   198ec: a910fe7f     	stp	xzr, xzr, [x19, #0x108]
   198f0: f9008e7f     	str	xzr, [x19, #0x118]
   198f4: 71001b3f     	cmp	w25, #0x6
   198f8: 54000320     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   198fc: f9409260     	ldr	x0, [x19, #0x120]
   19900: f900927f     	str	xzr, [x19, #0x120]
   19904: 94000000     	bl	0x19904 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1a4>
   19908: f9409660     	ldr	x0, [x19, #0x128]
   1990c: f900967f     	str	xzr, [x19, #0x128]
   19910: 94000000     	bl	0x19910 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1b0>
   19914: f9409a60     	ldr	x0, [x19, #0x130]
   19918: f9009a7f     	str	xzr, [x19, #0x130]
   1991c: 94000000     	bl	0x1991c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1bc>
   19920: a913fe7f     	stp	xzr, xzr, [x19, #0x138]
   19924: f900a67f     	str	xzr, [x19, #0x148]
   19928: 71001f3f     	cmp	w25, #0x7
   1992c: 54000180     	b.eq	0x1995c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1fc>
   19930: f940aa60     	ldr	x0, [x19, #0x150]
   19934: f900aa7f     	str	xzr, [x19, #0x150]
   19938: 94000000     	bl	0x19938 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1d8>
   1993c: f940ae60     	ldr	x0, [x19, #0x158]
   19940: f900ae7f     	str	xzr, [x19, #0x158]
   19944: 94000000     	bl	0x19944 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1e4>
   19948: f940b260     	ldr	x0, [x19, #0x160]
   1994c: f900b27f     	str	xzr, [x19, #0x160]
   19950: 94000000     	bl	0x19950 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x1f0>
   19954: a916fe7f     	stp	xzr, xzr, [x19, #0x168]
   19958: f900be7f     	str	xzr, [x19, #0x178]
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
   19994: 54002242     	b.hs	0x19ddc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x67c>
   19998: f90007f3     	str	x19, [sp, #0x8]
   1999c: 52800013     	mov	w19, #0x0               ; =0
   199a0: 2a1803e8     	mov	w8, w24
   199a4: f9002fe8     	str	x8, [sp, #0x58]
   199a8: 2a1703e8     	mov	w8, w23
   199ac: f90037e8     	str	x8, [sp, #0x68]
   199b0: f94023e9     	ldr	x9, [sp, #0x40]
   199b4: 51000528     	sub	w8, w9, #0x1
   199b8: d37e7d0a     	ubfiz	x10, x8, #2, #32
   199bc: d37d7d08     	ubfiz	x8, x8, #3, #32
   199c0: a902abe8     	stp	x8, x10, [sp, #0x28]
   199c4: d37df128     	lsl	x8, x9, #3
   199c8: f90013e8     	str	x8, [sp, #0x20]
   199cc: 910303fa     	add	x26, sp, #0xc0
   199d0: 5280121c     	mov	w28, #0x90              ; =144
   199d4: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   199d8: 3dc00100     	ldr	q0, [x8]
   199dc: 3d8007e0     	str	q0, [sp, #0x10]
   199e0: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   199e4: 39400108     	ldrb	w8, [x8]
   199e8: 7100051f     	cmp	w8, #0x1
   199ec: 540001c1     	b.ne	0x19a24 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2c4>
   199f0: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   199f4: b9400109     	ldr	w9, [x8]
   199f8: 290727f3     	stp	w19, w9, [sp, #0x38]
   199fc: 340003c9     	cbz	w9, 0x19a74 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x314>
   19a00: 9101c3e1     	add	x1, sp, #0x70
   19a04: 528000c0     	mov	w0, #0x6                ; =6
   19a08: 94000000     	bl	0x19a08 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2a8>
   19a0c: 3dc01fe0     	ldr	q0, [sp, #0x70]
   19a10: 4e61d800     	scvtf.2d	v0, v0
   19a14: 3dc007e1     	ldr	q1, [sp, #0x10]
   19a18: 6e61dc00     	fmul.2d	v0, v0, v1
   19a1c: 7e70d808     	faddp.2d	d8, v0
   19a20: 14000016     	b	0x19a78 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x318>
   19a24: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a28: 91000000     	add	x0, x0, #0x0
   19a2c: 94000000     	bl	0x19a2c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2cc>
   19a30: b50000a0     	cbnz	x0, 0x19a44 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2e4>
   19a34: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a38: 91000000     	add	x0, x0, #0x0
   19a3c: 94000000     	bl	0x19a3c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2dc>
   19a40: b4001ba0     	cbz	x0, 0x19db4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x654>
   19a44: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a48: 91000000     	add	x0, x0, #0x0
   19a4c: 94000000     	bl	0x19a4c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2ec>
   19a50: f100001f     	cmp	x0, #0x0
   19a54: 1a9f17e9     	cset	w9, eq
   19a58: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a5c: 91000108     	add	x8, x8, #0x0
   19a60: b900e509     	str	w9, [x8, #0xe4]
   19a64: 5280002a     	mov	w10, #0x1               ; =1
   19a68: 3900010a     	strb	w10, [x8]
   19a6c: 290727f3     	stp	w19, w9, [sp, #0x38]
   19a70: 35fffc89     	cbnz	w9, 0x19a00 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2a0>
   19a74: 6f00e408     	movi.2d	v8, #0000000000000000
   19a78: d10283a0     	sub	x0, x29, #0xa0
   19a7c: f9401bf3     	ldr	x19, [sp, #0x30]
   19a80: 91001262     	add	x2, x19, #0x4
   19a84: 52801fe1     	mov	w1, #0xff               ; =255
   19a88: 94000000     	bl	0x19a88 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x328>
   19a8c: d10303a0     	sub	x0, x29, #0xc0
   19a90: 91001262     	add	x2, x19, #0x4
   19a94: 52801fe1     	mov	w1, #0xff               ; =255
   19a98: 94000000     	bl	0x19a98 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x338>
   19a9c: 910203e0     	add	x0, sp, #0x80
   19aa0: f94017e8     	ldr	x8, [sp, #0x28]
   19aa4: 91002102     	add	x2, x8, #0x8
   19aa8: 52801fe1     	mov	w1, #0xff               ; =255
   19aac: 94000000     	bl	0x19aac <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x34c>
   19ab0: 910303e0     	add	x0, sp, #0xc0
   19ab4: 90000001     	adrp	x1, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19ab8: 91000021     	add	x1, x1, #0x0
   19abc: f94013e2     	ldr	x2, [sp, #0x20]
   19ac0: 94000000     	bl	0x19ac0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x360>
   19ac4: d280001b     	mov	x27, #0x0               ; =0
   19ac8: 52800018     	mov	w24, #0x0               ; =0
   19acc: b90067ff     	str	wzr, [sp, #0x64]
   19ad0: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19ad4: f9400117     	ldr	x23, [x8]
   19ad8: a944bbef     	ldp	x15, x14, [sp, #0x48]
   19adc: 14000004     	b	0x19aec <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x38c>
   19ae0: 9100077b     	add	x27, x27, #0x1
   19ae4: f101437f     	cmp	x27, #0x50
   19ae8: 54000c40     	b.eq	0x19c70 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x510>
   19aec: d2800019     	mov	x25, #0x0               ; =0
   19af0: f10001ff     	cmp	x15, #0x0
   19af4: f94037e8     	ldr	x8, [sp, #0x68]
   19af8: fa481360     	ccmp	x27, x8, #0x0, ne
   19afc: 1a9f07f6     	cset	w22, ne
   19b00: 529b0008     	mov	w8, #0xd800             ; =55296
   19b04: 90000009     	adrp	x9, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19b08: 91000129     	add	x9, x9, #0x0
   19b0c: 9b082773     	madd	x19, x27, x8, x9
   19b10: 14000006     	b	0x19b28 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3c8>
   19b14: 52800028     	mov	w8, #0x1                ; =1
   19b18: b90067e8     	str	w8, [sp, #0x64]
   19b1c: 91000739     	add	x25, x25, #0x1
   19b20: f106033f     	cmp	x25, #0x180
   19b24: 54fffde0     	b.eq	0x19ae0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x380>
   19b28: 9b1c4f34     	madd	x20, x25, x28, x19
   19b2c: 39423288     	ldrb	w8, [x20, #0x8c]
   19b30: 34ffff68     	cbz	w8, 0x19b1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3bc>
   19b34: f9400288     	ldr	x8, [x20]
   19b38: b4ffff28     	cbz	x8, 0x19b1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3bc>
   19b3c: f9400688     	ldr	x8, [x20, #0x8]
   19b40: b4fffee8     	cbz	x8, 0x19b1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3bc>
   19b44: f9400a88     	ldr	x8, [x20, #0x10]
   19b48: b4fffea8     	cbz	x8, 0x19b1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3bc>
   19b4c: f9402288     	ldr	x8, [x20, #0x40]
   19b50: eb15011f     	cmp	x8, x21
   19b54: 54fffe41     	b.ne	0x19b1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3bc>
   19b58: f9402688     	ldr	x8, [x20, #0x48]
   19b5c: eb0e011f     	cmp	x8, x14
   19b60: 54fffde1     	b.ne	0x19b1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3bc>
   19b64: f9404288     	ldr	x8, [x20, #0x80]
   19b68: eb17011f     	cmp	x8, x23
   19b6c: 54fffd48     	b.hi	0x19b14 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3b4>
   19b70: aa0f03e8     	mov	x8, x15
   19b74: f9402fe9     	ldr	x9, [sp, #0x58]
   19b78: 370000d6     	tbnz	w22, #0x0, 0x19b90 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x430>
   19b7c: b840450a     	ldr	w10, [x8], #0x4
   19b80: eb0a033f     	cmp	x25, x10
   19b84: 54fffcc0     	b.eq	0x19b1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3bc>
   19b88: f1000529     	subs	x9, x9, #0x1
   19b8c: 54ffff81     	b.ne	0x19b7c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x41c>
   19b90: f9402e82     	ldr	x2, [x20, #0x58]
   19b94: aa1b03e0     	mov	x0, x27
   19b98: aa1903e1     	mov	x1, x25
   19b9c: 94000000     	bl	0x19b9c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x43c>
   19ba0: f9402e88     	ldr	x8, [x20, #0x58]
   19ba4: f94023e9     	ldr	x9, [sp, #0x40]
   19ba8: 6b09031f     	cmp	w24, w9
   19bac: 54000162     	b.hs	0x19bd8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x478>
   19bb0: d10283a9     	sub	x9, x29, #0xa0
   19bb4: b838593b     	str	w27, [x9, w24, uxtw #2]
   19bb8: d10303a9     	sub	x9, x29, #0xc0
   19bbc: b8385939     	str	w25, [x9, w24, uxtw #2]
   19bc0: fc385b40     	str	d0, [x26, w24, uxtw #3]
   19bc4: 910203e9     	add	x9, sp, #0x80
   19bc8: f8385928     	str	x8, [x9, w24, uxtw #3]
   19bcc: 11000718     	add	w24, w24, #0x1
   19bd0: a944bbef     	ldp	x15, x14, [sp, #0x48]
   19bd4: 17ffffd2     	b	0x19b1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3bc>
   19bd8: 71000b1f     	cmp	w24, #0x2
   19bdc: f9402bee     	ldr	x14, [sp, #0x50]
   19be0: 910203f0     	add	x16, sp, #0x80
   19be4: d2800009     	mov	x9, #0x0                ; =0
   19be8: 54000263     	b.lo	0x19c34 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x4d4>
   19bec: 2a1803ea     	mov	w10, w24
   19bf0: 5280002b     	mov	w11, #0x1               ; =1
   19bf4: f94027ef     	ldr	x15, [sp, #0x48]
   19bf8: 14000005     	b	0x19c0c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x4ac>
   19bfc: 2a0b03e9     	mov	w9, w11
   19c00: 9100056b     	add	x11, x11, #0x1
   19c04: eb0b015f     	cmp	x10, x11
   19c08: 54000180     	b.eq	0x19c38 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x4d8>
   19c0c: fc6b7b41     	ldr	d1, [x26, x11, lsl #3]
   19c10: fc697b42     	ldr	d2, [x26, x9, lsl #3]
   19c14: 1e622020     	fcmp	d1, d2
   19c18: 54ffff2c     	b.gt	0x19bfc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x49c>
   19c1c: 54ffff21     	b.ne	0x19c00 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x4a0>
   19c20: f86b7a0c     	ldr	x12, [x16, x11, lsl #3]
   19c24: f8697a0d     	ldr	x13, [x16, x9, lsl #3]
   19c28: eb0d019f     	cmp	x12, x13
   19c2c: 54fffe88     	b.hi	0x19bfc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x49c>
   19c30: 17fffff4     	b	0x19c00 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x4a0>
   19c34: f94027ef     	ldr	x15, [sp, #0x48]
   19c38: fc697b41     	ldr	d1, [x26, x9, lsl #3]
   19c3c: 1e612000     	fcmp	d0, d1
   19c40: 540000ab     	b.lt	0x19c54 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x4f4>
   19c44: 54fff6c1     	b.ne	0x19b1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3bc>
   19c48: f8697a0a     	ldr	x10, [x16, x9, lsl #3]
   19c4c: eb0a011f     	cmp	x8, x10
   19c50: 54fff662     	b.hs	0x19b1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3bc>
   19c54: d10283aa     	sub	x10, x29, #0xa0
   19c58: b829795b     	str	w27, [x10, x9, lsl #2]
   19c5c: d10303aa     	sub	x10, x29, #0xc0
   19c60: b8297959     	str	w25, [x10, x9, lsl #2]
   19c64: fc297b40     	str	d0, [x26, x9, lsl #3]
   19c68: f8297a08     	str	x8, [x16, x9, lsl #3]
   19c6c: 17ffffac     	b	0x19b1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3bc>
   19c70: b9403ff3     	ldr	w19, [sp, #0x3c]
   19c74: 340001b3     	cbz	w19, 0x19ca8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x548>
   19c78: 9101c3e1     	add	x1, sp, #0x70
   19c7c: 528000c0     	mov	w0, #0x6                ; =6
   19c80: 94000000     	bl	0x19c80 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x520>
   19c84: 6d4727ea     	ldp	d10, d9, [sp, #0x70]
   19c88: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19c8c: 39400108     	ldrb	w8, [x8]
   19c90: 7100051f     	cmp	w8, #0x1
   19c94: 540000e1     	b.ne	0x19cb0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x550>
   19c98: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19c9c: b9400108     	ldr	w8, [x8]
   19ca0: 350002e8     	cbnz	w8, 0x19cfc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x59c>
   19ca4: 1400002e     	b	0x19d5c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x5fc>
   19ca8: 340005d8     	cbz	w24, 0x19d60 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x600>
   19cac: 14000068     	b	0x19e4c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6ec>
   19cb0: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19cb4: 91000000     	add	x0, x0, #0x0
   19cb8: 94000000     	bl	0x19cb8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x558>
   19cbc: b50000a0     	cbnz	x0, 0x19cd0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x570>
   19cc0: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19cc4: 91000000     	add	x0, x0, #0x0
   19cc8: 94000000     	bl	0x19cc8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x568>
   19ccc: b4000780     	cbz	x0, 0x19dbc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x65c>
   19cd0: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19cd4: 91000000     	add	x0, x0, #0x0
   19cd8: 94000000     	bl	0x19cd8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x578>
   19cdc: f100001f     	cmp	x0, #0x0
   19ce0: 1a9f17e8     	cset	w8, eq
   19ce4: 90000009     	adrp	x9, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19ce8: 91000129     	add	x9, x9, #0x0
   19cec: b900e528     	str	w8, [x9, #0xe4]
   19cf0: 5280002a     	mov	w10, #0x1               ; =1
   19cf4: 3900012a     	strb	w10, [x9]
   19cf8: 34000328     	cbz	w8, 0x19d5c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x5fc>
   19cfc: 5e61d940     	scvtf	d0, d10
   19d00: d2c80008     	mov	x8, #0x400000000000     ; =70368744177664
   19d04: f2e811e8     	movk	x8, #0x408f, lsl #48
   19d08: 9e670101     	fmov	d1, x8
   19d0c: 1e610800     	fmul	d0, d0, d1
   19d10: 5e61d921     	scvtf	d1, d9
   19d14: d29db1a8     	mov	x8, #0xed8d             ; =60813
   19d18: f2b416a8     	movk	x8, #0xa0b5, lsl #16
   19d1c: f2d8dee8     	movk	x8, #0xc6f7, lsl #32
   19d20: f2e7d608     	movk	x8, #0x3eb0, lsl #48
   19d24: 9e670102     	fmov	d2, x8
   19d28: 1e620821     	fmul	d1, d1, d2
   19d2c: 9000000a     	adrp	x10, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19d30: 9100014a     	add	x10, x10, #0x0
   19d34: a9402548     	ldp	x8, x9, [x10]
   19d38: 91000508     	add	x8, x8, #0x1
   19d3c: 528f000b     	mov	w11, #0x7800            ; =30720
   19d40: 8b0b0129     	add	x9, x9, x11
   19d44: a9002548     	stp	x8, x9, [x10]
   19d48: fd400942     	ldr	d2, [x10, #0x10]
   19d4c: 1e683800     	fsub	d0, d0, d8
   19d50: 1e622821     	fadd	d1, d1, d2
   19d54: 1e612800     	fadd	d0, d0, d1
   19d58: fd000940     	str	d0, [x10, #0x10]
   19d5c: 35000658     	cbnz	w24, 0x19e24 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6c4>
   19d60: b94067e8     	ldr	w8, [sp, #0x64]
   19d64: 7100011f     	cmp	w8, #0x0
   19d68: 1a9f17e8     	cset	w8, eq
   19d6c: b9403be9     	ldr	w9, [sp, #0x38]
   19d70: 2a080128     	orr	w8, w9, w8
   19d74: 37000348     	tbnz	w8, #0x0, 0x19ddc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x67c>
   19d78: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19d7c: f9400108     	ldr	x8, [x8]
   19d80: b4000068     	cbz	x8, 0x19d8c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x62c>
   19d84: 94000000     	bl	0x19d84 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x624>
   19d88: 340002a0     	cbz	w0, 0x19ddc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x67c>
   19d8c: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19d90: f9400100     	ldr	x0, [x8]
   19d94: 94000000     	bl	0x19d94 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x634>
   19d98: 52800033     	mov	w19, #0x1               ; =1
   19d9c: b4ffe220     	cbz	x0, 0x199e0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x280>
   19da0: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19da4: 91000000     	add	x0, x0, #0x0
   19da8: 94000000     	bl	0x19da8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x648>
   19dac: 35ffe1a0     	cbnz	w0, 0x199e0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x280>
   19db0: 1400000b     	b	0x19ddc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x67c>
   19db4: 52800009     	mov	w9, #0x0                ; =0
   19db8: 17ffff28     	b	0x19a58 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2f8>
   19dbc: 52800008     	mov	w8, #0x0                ; =0
   19dc0: 90000009     	adrp	x9, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19dc4: 91000129     	add	x9, x9, #0x0
   19dc8: b900e528     	str	w8, [x9, #0xe4]
   19dcc: 5280002a     	mov	w10, #0x1               ; =1
   19dd0: 3900012a     	strb	w10, [x9]
   19dd4: 35fff948     	cbnz	w8, 0x19cfc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x59c>
   19dd8: 17ffffe1     	b	0x19d5c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x5fc>
   19ddc: 52800014     	mov	w20, #0x0               ; =0
   19de0: f85803a8     	ldur	x8, [x29, #-0x80]
   19de4: 90000009     	adrp	x9, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19de8: f9400129     	ldr	x9, [x9]
   19dec: f9400129     	ldr	x9, [x9]
   19df0: eb08013f     	cmp	x9, x8
   19df4: 54001061     	b.ne	0x1a000 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8a0>
   19df8: aa1403e0     	mov	x0, x20
   19dfc: a95c7bfd     	ldp	x29, x30, [sp, #0x1c0]
   19e00: a95b4ff4     	ldp	x20, x19, [sp, #0x1b0]
   19e04: a95a57f6     	ldp	x22, x21, [sp, #0x1a0]
   19e08: a9595ff8     	ldp	x24, x23, [sp, #0x190]
   19e0c: a95867fa     	ldp	x26, x25, [sp, #0x180]
   19e10: a9576ffc     	ldp	x28, x27, [sp, #0x170]
   19e14: 6d5623e9     	ldp	d9, d8, [sp, #0x160]
   19e18: 6d552beb     	ldp	d11, d10, [sp, #0x150]
   19e1c: 910743ff     	add	sp, sp, #0x1d0
   19e20: d65f03c0     	ret
   19e24: aa1303fc     	mov	x28, x19
   19e28: 9101c3e1     	add	x1, sp, #0x70
   19e2c: 528000c0     	mov	w0, #0x6                ; =6
   19e30: 94000000     	bl	0x19e30 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6d0>
   19e34: 3dc01fe0     	ldr	q0, [sp, #0x70]
   19e38: 4e61d800     	scvtf.2d	v0, v0
   19e3c: 3dc007e1     	ldr	q1, [sp, #0x10]
   19e40: 6e61dc00     	fmul.2d	v0, v0, v1
   19e44: 7e70d808     	faddp.2d	d8, v0
   19e48: 14000003     	b	0x19e54 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6f4>
   19e4c: aa1303fc     	mov	x28, x19
   19e50: 6f00e408     	movi.2d	v8, #0000000000000000
   19e54: 52800014     	mov	w20, #0x0               ; =0
   19e58: d10283b3     	sub	x19, x29, #0xa0
   19e5c: d10303b6     	sub	x22, x29, #0xc0
   19e60: 52800617     	mov	w23, #0x30              ; =48
   19e64: 52800059     	mov	w25, #0x2               ; =2
   19e68: 9000001a     	adrp	x26, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19e6c: 2a1803f8     	mov	w24, w24
   19e70: f94007fb     	ldr	x27, [sp, #0x8]
   19e74: 1400000e     	b	0x19eac <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x74c>
   19e78: f90002bf     	str	xzr, [x21]
   19e7c: 94000000     	bl	0x19e7c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x71c>
   19e80: f94006a0     	ldr	x0, [x21, #0x8]
   19e84: f90006bf     	str	xzr, [x21, #0x8]
   19e88: 94000000     	bl	0x19e88 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x728>
   19e8c: f9400aa0     	ldr	x0, [x21, #0x10]
   19e90: f9000abf     	str	xzr, [x21, #0x10]
   19e94: 94000000     	bl	0x19e94 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x734>
   19e98: a901febf     	stp	xzr, xzr, [x21, #0x18]
   19e9c: f90016bf     	str	xzr, [x21, #0x28]
   19ea0: 910012d6     	add	x22, x22, #0x4
   19ea4: f1000718     	subs	x24, x24, #0x1
   19ea8: 54000440     	b.eq	0x19f30 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x7d0>
   19eac: b8404660     	ldr	w0, [x19], #0x4
   19eb0: 3100041f     	cmn	w0, #0x1
   19eb4: 54ffff60     	b.eq	0x19ea0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x740>
   19eb8: b94002c1     	ldr	w1, [x22]
   19ebc: 3100043f     	cmn	w1, #0x1
   19ec0: 54ffff00     	b.eq	0x19ea0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x740>
   19ec4: 9bb76e95     	umaddl	x21, w20, w23, x27
   19ec8: 52800022     	mov	w2, #0x1                ; =1
   19ecc: 52800023     	mov	w3, #0x1                ; =1
   19ed0: aa1503e4     	mov	x4, x21
   19ed4: 94000000     	bl	0x19ed4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x774>
   19ed8: f94002a0     	ldr	x0, [x21]
   19edc: b4fffce0     	cbz	x0, 0x19e78 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x718>
   19ee0: f94006a8     	ldr	x8, [x21, #0x8]
   19ee4: b4fffca8     	cbz	x8, 0x19e78 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x718>
   19ee8: f9400aa9     	ldr	x9, [x21, #0x10]
   19eec: b4fffc69     	cbz	x9, 0x19e78 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x718>
   19ef0: eb09001f     	cmp	x0, x9
   19ef4: 1a9f17ea     	cset	w10, eq
   19ef8: eb09011f     	cmp	x8, x9
   19efc: 1a9f1549     	csinc	w9, w10, wzr, ne
   19f00: eb08001f     	cmp	x0, x8
   19f04: 1a9f17e8     	cset	w8, eq
   19f08: 1a9f1529     	csinc	w9, w9, wzr, ne
   19f0c: 7100013f     	cmp	w9, #0x0
   19f10: 9a991729     	cinc	x9, x25, eq
   19f14: 6a0a011f     	tst	w8, w10
   19f18: 9a9f0528     	csinc	x8, x9, xzr, eq
   19f1c: f9400349     	ldr	x9, [x26]
   19f20: 8b080128     	add	x8, x9, x8
   19f24: f9000348     	str	x8, [x26]
   19f28: 11000694     	add	w20, w20, #0x1
   19f2c: 17ffffdd     	b	0x19ea0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x740>
   19f30: 34fff59c     	cbz	w28, 0x19de0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x680>
   19f34: 9101c3e1     	add	x1, sp, #0x70
   19f38: 528000c0     	mov	w0, #0x6                ; =6
   19f3c: 94000000     	bl	0x19f3c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x7dc>
   19f40: 6d4727ea     	ldp	d10, d9, [sp, #0x70]
   19f44: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19f48: 39400108     	ldrb	w8, [x8]
   19f4c: 7100051f     	cmp	w8, #0x1
   19f50: 54000081     	b.ne	0x19f60 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x800>
   19f54: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19f58: b9400108     	ldr	w8, [x8]
   19f5c: 14000013     	b	0x19fa8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x848>
   19f60: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19f64: 91000000     	add	x0, x0, #0x0
   19f68: 94000000     	bl	0x19f68 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x808>
   19f6c: b50000a0     	cbnz	x0, 0x19f80 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x820>
   19f70: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19f74: 91000000     	add	x0, x0, #0x0
   19f78: 94000000     	bl	0x19f78 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x818>
   19f7c: b40003e0     	cbz	x0, 0x19ff8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x898>
   19f80: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19f84: 91000000     	add	x0, x0, #0x0
   19f88: 94000000     	bl	0x19f88 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x828>
   19f8c: f100001f     	cmp	x0, #0x0
   19f90: 1a9f17e8     	cset	w8, eq
   19f94: 9000000a     	adrp	x10, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19f98: 9100014a     	add	x10, x10, #0x0
   19f9c: b900e548     	str	w8, [x10, #0xe4]
   19fa0: 52800029     	mov	w9, #0x1                ; =1
   19fa4: 39000149     	strb	w9, [x10]
   19fa8: 34fff1c8     	cbz	w8, 0x19de0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x680>
   19fac: d2c80008     	mov	x8, #0x400000000000     ; =70368744177664
   19fb0: f2e811e8     	movk	x8, #0x408f, lsl #48
   19fb4: d29db1a9     	mov	x9, #0xed8d             ; =60813
   19fb8: f2b416a9     	movk	x9, #0xa0b5, lsl #16
   19fbc: f2d8dee9     	movk	x9, #0xc6f7, lsl #32
   19fc0: f2e7d609     	movk	x9, #0x3eb0, lsl #48
   19fc4: 5e61d940     	scvtf	d0, d10
   19fc8: 9e670101     	fmov	d1, x8
   19fcc: 5e61d922     	scvtf	d2, d9
   19fd0: 9e670123     	fmov	d3, x9
   19fd4: 1e610800     	fmul	d0, d0, d1
   19fd8: 1e630841     	fmul	d1, d2, d3
   19fdc: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19fe0: fd400102     	ldr	d2, [x8]
   19fe4: 1e683800     	fsub	d0, d0, d8
   19fe8: 1e622821     	fadd	d1, d1, d2
   19fec: 1e612800     	fadd	d0, d0, d1
   19ff0: fd000100     	str	d0, [x8]
   19ff4: 17ffff7b     	b	0x19de0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x680>
   19ff8: 52800008     	mov	w8, #0x0                ; =0
   19ffc: 17ffffe6     	b	0x19f94 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x834>
   1a000: 94000000     	bl	0x1a000 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8a0>
