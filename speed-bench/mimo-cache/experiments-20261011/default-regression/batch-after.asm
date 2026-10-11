
ds4_metal.o:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000000019760 <_ds4_gpu_stream_expert_cache_take_reusable_batch>:
   19760: d10783ff     	sub	sp, sp, #0x1e0
   19764: 6d162beb     	stp	d11, d10, [sp, #0x160]
   19768: 6d1723e9     	stp	d9, d8, [sp, #0x170]
   1976c: a9186ffc     	stp	x28, x27, [sp, #0x180]
   19770: a91967fa     	stp	x26, x25, [sp, #0x190]
   19774: a91a5ff8     	stp	x24, x23, [sp, #0x1a0]
   19778: a91b57f6     	stp	x22, x21, [sp, #0x1b0]
   1977c: a91c4ff4     	stp	x20, x19, [sp, #0x1c0]
   19780: a91d7bfd     	stp	x29, x30, [sp, #0x1d0]
   19784: 910743fd     	add	x29, sp, #0x1d0
   19788: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   1978c: f9400108     	ldr	x8, [x8]
   19790: f9400108     	ldr	x8, [x8]
   19794: f81803a8     	stur	x8, [x29, #-0x80]
   19798: 34003600     	cbz	w0, 0x19e58 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6f8>
   1979c: aa0603f3     	mov	x19, x6
   197a0: aa0403f5     	mov	x21, x4
   197a4: aa0303f8     	mov	x24, x3
   197a8: aa0103f7     	mov	x23, x1
   197ac: aa0003f9     	mov	x25, x0
   197b0: a90617e2     	stp	x2, x5, [sp, #0x60]
   197b4: 52800108     	mov	w8, #0x8                ; =8
   197b8: 7100201f     	cmp	w0, #0x8
   197bc: 1a883008     	csel	w8, w0, w8, lo
   197c0: f9003fe8     	str	x8, [sp, #0x78]
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
   19994: 54002622     	b.hs	0x19e58 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6f8>
   19998: f90007f3     	str	x19, [sp, #0x8]
   1999c: 52800013     	mov	w19, #0x0               ; =0
   199a0: 2a1803e9     	mov	w9, w24
   199a4: 2a1703e8     	mov	w8, w23
   199a8: a90527e8     	stp	x8, x9, [sp, #0x50]
   199ac: f9403fe9     	ldr	x9, [sp, #0x78]
   199b0: 51000528     	sub	w8, w9, #0x1
   199b4: d37e7d0a     	ubfiz	x10, x8, #2, #32
   199b8: d37d7d08     	ubfiz	x8, x8, #3, #32
   199bc: a9032be8     	stp	x8, x10, [sp, #0x30]
   199c0: d37df128     	lsl	x8, x9, #3
   199c4: f90017e8     	str	x8, [sp, #0x28]
   199c8: 910343f6     	add	x22, sp, #0xd0
   199cc: 5280121c     	mov	w28, #0x90              ; =144
   199d0: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   199d4: 3dc00100     	ldr	q0, [x8]
   199d8: 3d8007e0     	str	q0, [sp, #0x10]
   199dc: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   199e0: 39400108     	ldrb	w8, [x8]
   199e4: 7100051f     	cmp	w8, #0x1
   199e8: 540001c1     	b.ne	0x19a20 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2c0>
   199ec: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   199f0: b9400109     	ldr	w9, [x8]
   199f4: 2908a7f3     	stp	w19, w9, [sp, #0x44]
   199f8: 340003c9     	cbz	w9, 0x19a70 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x310>
   199fc: 910203e1     	add	x1, sp, #0x80
   19a00: 528000c0     	mov	w0, #0x6                ; =6
   19a04: 94000000     	bl	0x19a04 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2a4>
   19a08: 3dc023e0     	ldr	q0, [sp, #0x80]
   19a0c: 4e61d800     	scvtf.2d	v0, v0
   19a10: 3dc007e1     	ldr	q1, [sp, #0x10]
   19a14: 6e61dc00     	fmul.2d	v0, v0, v1
   19a18: 7e70d808     	faddp.2d	d8, v0
   19a1c: 14000016     	b	0x19a74 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x314>
   19a20: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a24: 91000000     	add	x0, x0, #0x0
   19a28: 94000000     	bl	0x19a28 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2c8>
   19a2c: b50000a0     	cbnz	x0, 0x19a40 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2e0>
   19a30: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a34: 91000000     	add	x0, x0, #0x0
   19a38: 94000000     	bl	0x19a38 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2d8>
   19a3c: b4002060     	cbz	x0, 0x19e48 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6e8>
   19a40: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a44: 91000000     	add	x0, x0, #0x0
   19a48: 94000000     	bl	0x19a48 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2e8>
   19a4c: f100001f     	cmp	x0, #0x0
   19a50: 1a9f17e9     	cset	w9, eq
   19a54: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19a58: 91000108     	add	x8, x8, #0x0
   19a5c: b900e509     	str	w9, [x8, #0xe4]
   19a60: 5280002a     	mov	w10, #0x1               ; =1
   19a64: 3900010a     	strb	w10, [x8]
   19a68: 2908a7f3     	stp	w19, w9, [sp, #0x44]
   19a6c: 35fffc89     	cbnz	w9, 0x199fc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x29c>
   19a70: 6f00e408     	movi.2d	v8, #0000000000000000
   19a74: d10283a0     	sub	x0, x29, #0xa0
   19a78: f9401ff3     	ldr	x19, [sp, #0x38]
   19a7c: 91001262     	add	x2, x19, #0x4
   19a80: 52801fe1     	mov	w1, #0xff               ; =255
   19a84: 94000000     	bl	0x19a84 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x324>
   19a88: d10303a0     	sub	x0, x29, #0xc0
   19a8c: 91001262     	add	x2, x19, #0x4
   19a90: 52801fe1     	mov	w1, #0xff               ; =255
   19a94: 94000000     	bl	0x19a94 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x334>
   19a98: 910243e0     	add	x0, sp, #0x90
   19a9c: f9401be8     	ldr	x8, [sp, #0x30]
   19aa0: 91002102     	add	x2, x8, #0x8
   19aa4: 52801fe1     	mov	w1, #0xff               ; =255
   19aa8: 94000000     	bl	0x19aa8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x348>
   19aac: 910343e0     	add	x0, sp, #0xd0
   19ab0: 90000001     	adrp	x1, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19ab4: 91000021     	add	x1, x1, #0x0
   19ab8: f94017e2     	ldr	x2, [sp, #0x28]
   19abc: 94000000     	bl	0x19abc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x35c>
   19ac0: d280001b     	mov	x27, #0x0               ; =0
   19ac4: 52800018     	mov	w24, #0x0               ; =0
   19ac8: 52800006     	mov	w6, #0x0                ; =0
   19acc: 90000009     	adrp	x9, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19ad0: 91000129     	add	x9, x9, #0x0
   19ad4: f9443d27     	ldr	x7, [x9, #0x878]
   19ad8: f944ed28     	ldr	x8, [x9, #0x9d8]
   19adc: f100011f     	cmp	x8, #0x0
   19ae0: f9400128     	ldr	x8, [x9]
   19ae4: fa401904     	ccmp	x8, #0x0, #0x4, ne
   19ae8: 1a9f07f4     	cset	w20, ne
   19aec: a9463bef     	ldp	x15, x14, [sp, #0x60]
   19af0: a94543f1     	ldp	x17, x16, [sp, #0x50]
   19af4: 910243e0     	add	x0, sp, #0x90
   19af8: 90000001     	adrp	x1, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19afc: 91000021     	add	x1, x1, #0x0
   19b00: 529b0002     	mov	w2, #0xd800             ; =55296
   19b04: 5280c004     	mov	w4, #0x600              ; =1536
   19b08: 90000005     	adrp	x5, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19b0c: 910000a5     	add	x5, x5, #0x0
   19b10: b9004ff4     	str	w20, [sp, #0x4c]
   19b14: 14000004     	b	0x19b24 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3c4>
   19b18: 9100077b     	add	x27, x27, #0x1
   19b1c: f101437f     	cmp	x27, #0x50
   19b20: 54000ee0     	b.eq	0x19cfc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x59c>
   19b24: d2800019     	mov	x25, #0x0               ; =0
   19b28: f10001ff     	cmp	x15, #0x0
   19b2c: fa511360     	ccmp	x27, x17, #0x0, ne
   19b30: 1a9f07fa     	cset	w26, ne
   19b34: 9b020773     	madd	x19, x27, x2, x1
   19b38: 9b04177e     	madd	x30, x27, x4, x5
   19b3c: 14000005     	b	0x19b50 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3f0>
   19b40: 52800026     	mov	w6, #0x1                ; =1
   19b44: 91000739     	add	x25, x25, #0x1
   19b48: f106033f     	cmp	x25, #0x180
   19b4c: 54fffe60     	b.eq	0x19b18 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3b8>
   19b50: 9b1c4f37     	madd	x23, x25, x28, x19
   19b54: 394232e8     	ldrb	w8, [x23, #0x8c]
   19b58: 34ffff68     	cbz	w8, 0x19b44 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3e4>
   19b5c: f94002e8     	ldr	x8, [x23]
   19b60: b4ffff28     	cbz	x8, 0x19b44 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3e4>
   19b64: f94006e8     	ldr	x8, [x23, #0x8]
   19b68: b4fffee8     	cbz	x8, 0x19b44 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3e4>
   19b6c: f9400ae8     	ldr	x8, [x23, #0x10]
   19b70: b4fffea8     	cbz	x8, 0x19b44 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3e4>
   19b74: f94022e8     	ldr	x8, [x23, #0x40]
   19b78: eb15011f     	cmp	x8, x21
   19b7c: 54fffe41     	b.ne	0x19b44 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3e4>
   19b80: f94026e8     	ldr	x8, [x23, #0x48]
   19b84: eb0e011f     	cmp	x8, x14
   19b88: 54fffde1     	b.ne	0x19b44 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3e4>
   19b8c: f94042e8     	ldr	x8, [x23, #0x80]
   19b90: eb07011f     	cmp	x8, x7
   19b94: 54fffd68     	b.hi	0x19b40 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3e0>
   19b98: aa0f03e8     	mov	x8, x15
   19b9c: aa1003e9     	mov	x9, x16
   19ba0: 370000da     	tbnz	w26, #0x0, 0x19bb8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x458>
   19ba4: b840450a     	ldr	w10, [x8], #0x4
   19ba8: eb0a033f     	cmp	x25, x10
   19bac: 54fffcc0     	b.eq	0x19b44 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3e4>
   19bb0: f1000529     	subs	x9, x9, #0x1
   19bb4: 54ffff81     	b.ne	0x19ba4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x444>
   19bb8: b8797bc3     	ldr	w3, [x30, x25, lsl #2]
   19bbc: 36000694     	tbz	w20, #0x0, 0x19c8c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x52c>
   19bc0: f9402ee2     	ldr	x2, [x23, #0x58]
   19bc4: aa1b03e0     	mov	x0, x27
   19bc8: aa1903e1     	mov	x1, x25
   19bcc: b90077fa     	str	w26, [sp, #0x74]
   19bd0: aa1803f4     	mov	x20, x24
   19bd4: aa0603f8     	mov	x24, x6
   19bd8: aa1503f6     	mov	x22, x21
   19bdc: aa0703f5     	mov	x21, x7
   19be0: aa1e03fa     	mov	x26, x30
   19be4: 94000000     	bl	0x19be4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x484>
   19be8: aa1a03fe     	mov	x30, x26
   19bec: aa1503e7     	mov	x7, x21
   19bf0: aa1603f5     	mov	x21, x22
   19bf4: 910343f6     	add	x22, sp, #0xd0
   19bf8: aa1803e6     	mov	x6, x24
   19bfc: aa1403f8     	mov	x24, x20
   19c00: b9404ff4     	ldr	w20, [sp, #0x4c]
   19c04: b94077fa     	ldr	w26, [sp, #0x74]
   19c08: 90000005     	adrp	x5, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19c0c: 910000a5     	add	x5, x5, #0x0
   19c10: 5280c004     	mov	w4, #0x600              ; =1536
   19c14: 529b0002     	mov	w2, #0xd800             ; =55296
   19c18: 90000001     	adrp	x1, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19c1c: 91000021     	add	x1, x1, #0x0
   19c20: 910243e0     	add	x0, sp, #0x90
   19c24: a94543f1     	ldp	x17, x16, [sp, #0x50]
   19c28: a9463bef     	ldp	x15, x14, [sp, #0x60]
   19c2c: f9402ee8     	ldr	x8, [x23, #0x58]
   19c30: f9403fe9     	ldr	x9, [sp, #0x78]
   19c34: 6b09031f     	cmp	w24, w9
   19c38: 54000343     	b.lo	0x19ca0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x540>
   19c3c: 71000b1f     	cmp	w24, #0x2
   19c40: 54000403     	b.lo	0x19cc0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x560>
   19c44: d2800009     	mov	x9, #0x0                ; =0
   19c48: 2a1803ea     	mov	w10, w24
   19c4c: 5280002b     	mov	w11, #0x1               ; =1
   19c50: 14000005     	b	0x19c64 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x504>
   19c54: 2a0b03e9     	mov	w9, w11
   19c58: 9100056b     	add	x11, x11, #0x1
   19c5c: eb0b015f     	cmp	x10, x11
   19c60: 54000320     	b.eq	0x19cc4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x564>
   19c64: fc6b7ac1     	ldr	d1, [x22, x11, lsl #3]
   19c68: fc697ac2     	ldr	d2, [x22, x9, lsl #3]
   19c6c: 1e622020     	fcmp	d1, d2
   19c70: 54ffff2c     	b.gt	0x19c54 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x4f4>
   19c74: 54ffff21     	b.ne	0x19c58 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x4f8>
   19c78: f86b780c     	ldr	x12, [x0, x11, lsl #3]
   19c7c: f869780d     	ldr	x13, [x0, x9, lsl #3]
   19c80: eb0d019f     	cmp	x12, x13
   19c84: 54fffe88     	b.hi	0x19c54 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x4f4>
   19c88: 17fffff4     	b	0x19c58 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x4f8>
   19c8c: 1e630060     	ucvtf	d0, w3
   19c90: f9402ee8     	ldr	x8, [x23, #0x58]
   19c94: f9403fe9     	ldr	x9, [sp, #0x78]
   19c98: 6b09031f     	cmp	w24, w9
   19c9c: 54fffd02     	b.hs	0x19c3c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x4dc>
   19ca0: d10283a9     	sub	x9, x29, #0xa0
   19ca4: b838593b     	str	w27, [x9, w24, uxtw #2]
   19ca8: d10303a9     	sub	x9, x29, #0xc0
   19cac: b8385939     	str	w25, [x9, w24, uxtw #2]
   19cb0: fc385ac0     	str	d0, [x22, w24, uxtw #3]
   19cb4: f8385808     	str	x8, [x0, w24, uxtw #3]
   19cb8: 11000718     	add	w24, w24, #0x1
   19cbc: 17ffffa2     	b	0x19b44 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3e4>
   19cc0: d2800009     	mov	x9, #0x0                ; =0
   19cc4: fc697ac1     	ldr	d1, [x22, x9, lsl #3]
   19cc8: 1e612000     	fcmp	d0, d1
   19ccc: 540000ab     	b.lt	0x19ce0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x580>
   19cd0: 54fff3a1     	b.ne	0x19b44 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3e4>
   19cd4: f869780a     	ldr	x10, [x0, x9, lsl #3]
   19cd8: eb0a011f     	cmp	x8, x10
   19cdc: 54fff342     	b.hs	0x19b44 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3e4>
   19ce0: d10283aa     	sub	x10, x29, #0xa0
   19ce4: b829795b     	str	w27, [x10, x9, lsl #2]
   19ce8: d10303aa     	sub	x10, x29, #0xc0
   19cec: b8297959     	str	w25, [x10, x9, lsl #2]
   19cf0: fc297ac0     	str	d0, [x22, x9, lsl #3]
   19cf4: f8297808     	str	x8, [x0, x9, lsl #3]
   19cf8: 17ffff93     	b	0x19b44 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x3e4>
   19cfc: b9404bf3     	ldr	w19, [sp, #0x48]
   19d00: 340001f3     	cbz	w19, 0x19d3c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x5dc>
   19d04: aa0603f4     	mov	x20, x6
   19d08: 910203e1     	add	x1, sp, #0x80
   19d0c: 528000c0     	mov	w0, #0x6                ; =6
   19d10: 94000000     	bl	0x19d10 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x5b0>
   19d14: 6d4827ea     	ldp	d10, d9, [sp, #0x80]
   19d18: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19d1c: 39400108     	ldrb	w8, [x8]
   19d20: 7100051f     	cmp	w8, #0x1
   19d24: 54000101     	b.ne	0x19d44 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x5e4>
   19d28: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19d2c: b9400108     	ldr	w8, [x8]
   19d30: aa1403e6     	mov	x6, x20
   19d34: 35000308     	cbnz	w8, 0x19d94 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x634>
   19d38: 1400002f     	b	0x19df4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x694>
   19d3c: 340005f8     	cbz	w24, 0x19df8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x698>
   19d40: 14000062     	b	0x19ec8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x768>
   19d44: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19d48: 91000000     	add	x0, x0, #0x0
   19d4c: 94000000     	bl	0x19d4c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x5ec>
   19d50: b50000a0     	cbnz	x0, 0x19d64 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x604>
   19d54: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19d58: 91000000     	add	x0, x0, #0x0
   19d5c: 94000000     	bl	0x19d5c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x5fc>
   19d60: b4000780     	cbz	x0, 0x19e50 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6f0>
   19d64: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19d68: 91000000     	add	x0, x0, #0x0
   19d6c: 94000000     	bl	0x19d6c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x60c>
   19d70: f100001f     	cmp	x0, #0x0
   19d74: 1a9f17e8     	cset	w8, eq
   19d78: aa1403e6     	mov	x6, x20
   19d7c: 90000009     	adrp	x9, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19d80: 91000129     	add	x9, x9, #0x0
   19d84: b900e528     	str	w8, [x9, #0xe4]
   19d88: 5280002a     	mov	w10, #0x1               ; =1
   19d8c: 3900012a     	strb	w10, [x9]
   19d90: 34000328     	cbz	w8, 0x19df4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x694>
   19d94: 5e61d940     	scvtf	d0, d10
   19d98: d2c80008     	mov	x8, #0x400000000000     ; =70368744177664
   19d9c: f2e811e8     	movk	x8, #0x408f, lsl #48
   19da0: 9e670101     	fmov	d1, x8
   19da4: 1e610800     	fmul	d0, d0, d1
   19da8: 5e61d921     	scvtf	d1, d9
   19dac: d29db1a8     	mov	x8, #0xed8d             ; =60813
   19db0: f2b416a8     	movk	x8, #0xa0b5, lsl #16
   19db4: f2d8dee8     	movk	x8, #0xc6f7, lsl #32
   19db8: f2e7d608     	movk	x8, #0x3eb0, lsl #48
   19dbc: 9e670102     	fmov	d2, x8
   19dc0: 1e620821     	fmul	d1, d1, d2
   19dc4: 9000000a     	adrp	x10, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19dc8: 9100014a     	add	x10, x10, #0x0
   19dcc: a9402548     	ldp	x8, x9, [x10]
   19dd0: 91000508     	add	x8, x8, #0x1
   19dd4: 528f000b     	mov	w11, #0x7800            ; =30720
   19dd8: 8b0b0129     	add	x9, x9, x11
   19ddc: a9002548     	stp	x8, x9, [x10]
   19de0: fd400942     	ldr	d2, [x10, #0x10]
   19de4: 1e683800     	fsub	d0, d0, d8
   19de8: 1e622821     	fadd	d1, d1, d2
   19dec: 1e612800     	fadd	d0, d0, d1
   19df0: fd000940     	str	d0, [x10, #0x10]
   19df4: 35000578     	cbnz	w24, 0x19ea0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x740>
   19df8: 710000df     	cmp	w6, #0x0
   19dfc: 1a9f17e8     	cset	w8, eq
   19e00: b94047e9     	ldr	w9, [sp, #0x44]
   19e04: 2a080128     	orr	w8, w9, w8
   19e08: 37000288     	tbnz	w8, #0x0, 0x19e58 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6f8>
   19e0c: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19e10: f9400108     	ldr	x8, [x8]
   19e14: b4000068     	cbz	x8, 0x19e20 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6c0>
   19e18: 94000000     	bl	0x19e18 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6b8>
   19e1c: 340001e0     	cbz	w0, 0x19e58 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6f8>
   19e20: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19e24: f9400100     	ldr	x0, [x8]
   19e28: 94000000     	bl	0x19e28 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6c8>
   19e2c: 52800033     	mov	w19, #0x1               ; =1
   19e30: b4ffdd60     	cbz	x0, 0x199dc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x27c>
   19e34: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19e38: 91000000     	add	x0, x0, #0x0
   19e3c: 94000000     	bl	0x19e3c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6dc>
   19e40: 35ffdce0     	cbnz	w0, 0x199dc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x27c>
   19e44: 14000005     	b	0x19e58 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6f8>
   19e48: 52800009     	mov	w9, #0x0                ; =0
   19e4c: 17ffff02     	b	0x19a54 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x2f4>
   19e50: 52800008     	mov	w8, #0x0                ; =0
   19e54: 17ffffc9     	b	0x19d78 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x618>
   19e58: 52800014     	mov	w20, #0x0               ; =0
   19e5c: f85803a8     	ldur	x8, [x29, #-0x80]
   19e60: 90000009     	adrp	x9, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19e64: f9400129     	ldr	x9, [x9]
   19e68: f9400129     	ldr	x9, [x9]
   19e6c: eb08013f     	cmp	x9, x8
   19e70: 54001061     	b.ne	0x1a07c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x91c>
   19e74: aa1403e0     	mov	x0, x20
   19e78: a95d7bfd     	ldp	x29, x30, [sp, #0x1d0]
   19e7c: a95c4ff4     	ldp	x20, x19, [sp, #0x1c0]
   19e80: a95b57f6     	ldp	x22, x21, [sp, #0x1b0]
   19e84: a95a5ff8     	ldp	x24, x23, [sp, #0x1a0]
   19e88: a95967fa     	ldp	x26, x25, [sp, #0x190]
   19e8c: a9586ffc     	ldp	x28, x27, [sp, #0x180]
   19e90: 6d5723e9     	ldp	d9, d8, [sp, #0x170]
   19e94: 6d562beb     	ldp	d11, d10, [sp, #0x160]
   19e98: 910783ff     	add	sp, sp, #0x1e0
   19e9c: d65f03c0     	ret
   19ea0: aa1303fc     	mov	x28, x19
   19ea4: 910203e1     	add	x1, sp, #0x80
   19ea8: 528000c0     	mov	w0, #0x6                ; =6
   19eac: 94000000     	bl	0x19eac <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x74c>
   19eb0: 3dc023e0     	ldr	q0, [sp, #0x80]
   19eb4: 4e61d800     	scvtf.2d	v0, v0
   19eb8: 3dc007e1     	ldr	q1, [sp, #0x10]
   19ebc: 6e61dc00     	fmul.2d	v0, v0, v1
   19ec0: 7e70d808     	faddp.2d	d8, v0
   19ec4: 14000003     	b	0x19ed0 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x770>
   19ec8: aa1303fc     	mov	x28, x19
   19ecc: 6f00e408     	movi.2d	v8, #0000000000000000
   19ed0: 52800014     	mov	w20, #0x0               ; =0
   19ed4: d10283b3     	sub	x19, x29, #0xa0
   19ed8: d10303b6     	sub	x22, x29, #0xc0
   19edc: 52800617     	mov	w23, #0x30              ; =48
   19ee0: 52800059     	mov	w25, #0x2               ; =2
   19ee4: 9000001a     	adrp	x26, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19ee8: 2a1803f8     	mov	w24, w24
   19eec: f94007fb     	ldr	x27, [sp, #0x8]
   19ef0: 1400000e     	b	0x19f28 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x7c8>
   19ef4: f90002bf     	str	xzr, [x21]
   19ef8: 94000000     	bl	0x19ef8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x798>
   19efc: f94006a0     	ldr	x0, [x21, #0x8]
   19f00: f90006bf     	str	xzr, [x21, #0x8]
   19f04: 94000000     	bl	0x19f04 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x7a4>
   19f08: f9400aa0     	ldr	x0, [x21, #0x10]
   19f0c: f9000abf     	str	xzr, [x21, #0x10]
   19f10: 94000000     	bl	0x19f10 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x7b0>
   19f14: a901febf     	stp	xzr, xzr, [x21, #0x18]
   19f18: f90016bf     	str	xzr, [x21, #0x28]
   19f1c: 910012d6     	add	x22, x22, #0x4
   19f20: f1000718     	subs	x24, x24, #0x1
   19f24: 54000440     	b.eq	0x19fac <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x84c>
   19f28: b8404660     	ldr	w0, [x19], #0x4
   19f2c: 3100041f     	cmn	w0, #0x1
   19f30: 54ffff60     	b.eq	0x19f1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x7bc>
   19f34: b94002c1     	ldr	w1, [x22]
   19f38: 3100043f     	cmn	w1, #0x1
   19f3c: 54ffff00     	b.eq	0x19f1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x7bc>
   19f40: 9bb76e95     	umaddl	x21, w20, w23, x27
   19f44: 52800022     	mov	w2, #0x1                ; =1
   19f48: 52800023     	mov	w3, #0x1                ; =1
   19f4c: aa1503e4     	mov	x4, x21
   19f50: 94000000     	bl	0x19f50 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x7f0>
   19f54: f94002a0     	ldr	x0, [x21]
   19f58: b4fffce0     	cbz	x0, 0x19ef4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x794>
   19f5c: f94006a8     	ldr	x8, [x21, #0x8]
   19f60: b4fffca8     	cbz	x8, 0x19ef4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x794>
   19f64: f9400aa9     	ldr	x9, [x21, #0x10]
   19f68: b4fffc69     	cbz	x9, 0x19ef4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x794>
   19f6c: eb09001f     	cmp	x0, x9
   19f70: 1a9f17ea     	cset	w10, eq
   19f74: eb09011f     	cmp	x8, x9
   19f78: 1a9f1549     	csinc	w9, w10, wzr, ne
   19f7c: eb08001f     	cmp	x0, x8
   19f80: 1a9f17e8     	cset	w8, eq
   19f84: 1a9f1529     	csinc	w9, w9, wzr, ne
   19f88: 7100013f     	cmp	w9, #0x0
   19f8c: 9a991729     	cinc	x9, x25, eq
   19f90: 6a0a011f     	tst	w8, w10
   19f94: 9a9f0528     	csinc	x8, x9, xzr, eq
   19f98: f9400349     	ldr	x9, [x26]
   19f9c: 8b080128     	add	x8, x9, x8
   19fa0: f9000348     	str	x8, [x26]
   19fa4: 11000694     	add	w20, w20, #0x1
   19fa8: 17ffffdd     	b	0x19f1c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x7bc>
   19fac: 34fff59c     	cbz	w28, 0x19e5c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6fc>
   19fb0: 910203e1     	add	x1, sp, #0x80
   19fb4: 528000c0     	mov	w0, #0x6                ; =6
   19fb8: 94000000     	bl	0x19fb8 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x858>
   19fbc: 6d4827ea     	ldp	d10, d9, [sp, #0x80]
   19fc0: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19fc4: 39400108     	ldrb	w8, [x8]
   19fc8: 7100051f     	cmp	w8, #0x1
   19fcc: 54000081     	b.ne	0x19fdc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x87c>
   19fd0: 90000008     	adrp	x8, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19fd4: b9400108     	ldr	w8, [x8]
   19fd8: 14000013     	b	0x1a024 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8c4>
   19fdc: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19fe0: 91000000     	add	x0, x0, #0x0
   19fe4: 94000000     	bl	0x19fe4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x884>
   19fe8: b50000a0     	cbnz	x0, 0x19ffc <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x89c>
   19fec: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   19ff0: 91000000     	add	x0, x0, #0x0
   19ff4: 94000000     	bl	0x19ff4 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x894>
   19ff8: b40003e0     	cbz	x0, 0x1a074 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x914>
   19ffc: 90000000     	adrp	x0, 0x19000 <_ds4_gpu_stream_expert_cache_begin_selected_load+0x8a4>
   1a000: 91000000     	add	x0, x0, #0x0
   1a004: 94000000     	bl	0x1a004 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8a4>
   1a008: f100001f     	cmp	x0, #0x0
   1a00c: 1a9f17e8     	cset	w8, eq
   1a010: 9000000a     	adrp	x10, 0x1a000 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8a0>
   1a014: 9100014a     	add	x10, x10, #0x0
   1a018: b900e548     	str	w8, [x10, #0xe4]
   1a01c: 52800029     	mov	w9, #0x1                ; =1
   1a020: 39000149     	strb	w9, [x10]
   1a024: 34fff1c8     	cbz	w8, 0x19e5c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6fc>
   1a028: d2c80008     	mov	x8, #0x400000000000     ; =70368744177664
   1a02c: f2e811e8     	movk	x8, #0x408f, lsl #48
   1a030: d29db1a9     	mov	x9, #0xed8d             ; =60813
   1a034: f2b416a9     	movk	x9, #0xa0b5, lsl #16
   1a038: f2d8dee9     	movk	x9, #0xc6f7, lsl #32
   1a03c: f2e7d609     	movk	x9, #0x3eb0, lsl #48
   1a040: 5e61d940     	scvtf	d0, d10
   1a044: 9e670101     	fmov	d1, x8
   1a048: 5e61d922     	scvtf	d2, d9
   1a04c: 9e670123     	fmov	d3, x9
   1a050: 1e610800     	fmul	d0, d0, d1
   1a054: 1e630841     	fmul	d1, d2, d3
   1a058: 90000008     	adrp	x8, 0x1a000 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8a0>
   1a05c: fd400102     	ldr	d2, [x8]
   1a060: 1e683800     	fsub	d0, d0, d8
   1a064: 1e622821     	fadd	d1, d1, d2
   1a068: 1e612800     	fadd	d0, d0, d1
   1a06c: fd000100     	str	d0, [x8]
   1a070: 17ffff7b     	b	0x19e5c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x6fc>
   1a074: 52800008     	mov	w8, #0x0                ; =0
   1a078: 17ffffe6     	b	0x1a010 <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x8b0>
   1a07c: 94000000     	bl	0x1a07c <_ds4_gpu_stream_expert_cache_take_reusable_batch+0x91c>
